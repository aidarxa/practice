#include "core/execution.h"
#include "core/optimizer_rules.h"
#include "core/translator.h"
#include "core/visitor.h"
#include "core/pcuda_plan_runtime.h"
#include "core/result_layout.h"
#include "core/text_plan.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace db {

namespace {
using Clock = std::chrono::steady_clock;
static double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
}


static std::string tableNameFromColumn(const std::string& col) {
    if (col.rfind("lo_", 0) == 0) return "LINEORDER";
    if (col.rfind("s_", 0) == 0) return "SUPPLIER";
    if (col.rfind("c_", 0) == 0) return "CUSTOMER";
    if (col.rfind("p_", 0) == 0) return "PART";
    if (col.rfind("d_", 0) == 0) return "DDATE";
    return "";
}

/** @brief Loaded modules are immutable; the adapter binds only this execution's values/resources. */
void IExecutor::execute(const KernelLookup& ready,const ExecutionPlan& plan,const ParameterValues& parameters,ExecutionContext* ctx) {
    LoadedKernels kernels;
    for (size_t i=0;i<plan.kernels.size();++i) kernels.push_back(ready(i));
    execute(kernels,plan,parameters,ctx);
}
/** @brief One timer definition covers eager and staged host-plan execution. */
template<class Modules> static void executeNative(const Modules& kernels,const ExecutionPlan& plan,
                                                 const ParameterValues& parameters,ExecutionContext* ctx) {
    if (!ctx || !ctx->runtime_) throw std::runtime_error("missing PCUDA execution context");
    const auto start=Clock::now();
    executePcudaplan(plan,parameters,kernels,*ctx);
    ctx->timing_.generated_execute_ms+=elapsedMs(start,Clock::now());
    ctx->timing_.gpu_execute_ms=ctx->timing_.generated_execute_ms;
    ctx->timing_.jit_execute_ms=ctx->timing_.generated_execute_ms;
}
void DynamicLibraryExecutor::execute(const LoadedKernels& kernels, const ExecutionPlan& plan,
                                      const ParameterValues& parameters, ExecutionContext* ctx) {
    executeNative(kernels,plan,parameters,ctx);
}
void DynamicLibraryExecutor::execute(const KernelLookup& ready,const ExecutionPlan& plan,const ParameterValues& parameters,ExecutionContext* ctx) {
    executeNative(ready,plan,parameters,ctx);
}

// --- QueryEngine ---

QueryEngine::QueryEngine(std::shared_ptr<Catalog> catalog,
                         std::unique_ptr<ProcessModuleCache> cache,
                         std::unique_ptr<ICompiler> compiler,
                         std::unique_ptr<IExecutor> executor)
    : catalog_(std::move(catalog)),
      cache_(std::move(cache)),
      compiler_(std::move(compiler)),
      executor_(std::move(executor)) {}

// ============================================================================
// Helpers
// ============================================================================

/** @brief Find the first requested operator using the same depth-first traversal for every result shape. */
template<class Node,OperatorType Type> static const Node* findPlanNode(const OperatorNode* node) {
    if (!node) return nullptr;
    if (node->getType()==Type) return static_cast<const Node*>(node);
    for (const auto& child : node->getChildren()) {
        const auto* found=findPlanNode<Node,Type>(child.get());
        if (found) return found;
    }
    return nullptr;
}

static const AggregateNode* findAggregateNode(const OperatorNode* node) {
    return findPlanNode<AggregateNode,OperatorType::AGGREGATE>(node);
}
static const ProjectionNode* findProjectionNode(const OperatorNode* node) {
    return findPlanNode<ProjectionNode,OperatorType::PROJECTION>(node);
}



static void collectTableScansForResultLayout(const OperatorNode* node,
                                             std::vector<const TableScanNode*>& out) {
    if (!node) return;
    if (node->getType() == OperatorType::TABLE_SCAN) {
        out.push_back(static_cast<const TableScanNode*>(node));
    }
    for (const auto& child : node->getChildren()) {
        collectTableScansForResultLayout(child.get(), out);
    }
}


static bool expressionMayBeNullable(const ExprNode* expr, const Catalog& catalog) {
    if (!expr) return true;
    switch (expr->getType()) {
        case ExprType::LITERAL_NULL:
            return true;
        case ExprType::PARAMETER_REF:
            return true;
        case ExprType::STAR:
        case ExprType::LITERAL_INT:
        case ExprType::LITERAL_FLOAT:
            return false;
        case ExprType::COLUMN_REF: {
            const auto* c = static_cast<const ColumnRefExpr*>(expr);
            const std::string table = tableNameFromColumn(c->column_name);
            try {
                return catalog.getTableMetadata(table).isColumnNullable(c->column_name);
            } catch (...) {
                return true;
            }
        }
        case ExprType::OP_ADD:
        case ExprType::OP_SUB:
        case ExprType::OP_MUL:
        case ExprType::OP_DIV:
        case ExprType::OP_EQ:
        case ExprType::OP_NEQ:
        case ExprType::OP_LT:
        case ExprType::OP_LTE:
        case ExprType::OP_GT:
        case ExprType::OP_GTE:
        case ExprType::OP_AND:
        case ExprType::OP_OR: {
            const auto* b = static_cast<const BinaryExpr*>(expr);
            return expressionMayBeNullable(b->left.get(), catalog) ||
                   expressionMayBeNullable(b->right.get(), catalog);
        }
        case ExprType::OP_IS_NULL:
        case ExprType::OP_IS_NOT_NULL:
            return false;
        case ExprType::CASE_WHEN: {
            const auto* c = static_cast<const CaseWhenExpr*>(expr);
            return expressionMayBeNullable(c->then_expr.get(), catalog) ||
                   expressionMayBeNullable(c->else_expr.get(), catalog);
        }
        default:
            return true;
    }
}

static std::vector<ResultColumnDesc> inferResultColumns(const OperatorNode* node,
                                                        const Catalog& catalog, const ParameterSchema& schema) {
    std::vector<ResultColumnDesc> descs;
    if (!node) return descs;

    if (const auto* agg = findAggregateNode(node)) {
        for (const auto& gexpr : agg->group_by_exprs) {
            const auto type = expressionResultType(gexpr.get(), schema);
            descs.push_back({type, 0, expressionMayBeNullable(gexpr.get(), catalog)});
        }
        for (const auto& agg_def : agg->aggregates)
            descs.push_back({aggregateResultType(agg_def, schema), 0,
                            !agg_def.isCount() && (agg->group_by_exprs.empty() || expressionMayBeNullable(agg_def.agg_expr.get(), catalog))});
        return descs;
    }

    if (const auto* proj = findProjectionNode(node)) {
        std::vector<const TableScanNode*> scans;
        if (!proj->getChildren().empty()) {
            collectTableScansForResultLayout(proj->getChildren()[0].get(), scans);
        }
        for (const auto& expr : proj->select_exprs) {
            if (!expr) continue;
            if (expr->getType() == ExprType::STAR) {
                for (const auto* scan : scans) {
                    const auto& meta = catalog.getTableMetadata(scan->table_name);
                    for (const auto& col_name : meta.getColumnNames()) {
                        descs.push_back({meta.getColumnStorage(col_name).isNumeric() ? LogicalType::Int64 : LogicalType::Text, 0, meta.isColumnNullable(col_name)});
                    }
                }
            } else {
                descs.push_back({expressionResultType(expr.get(), schema), 0, expressionMayBeNullable(expr.get(), catalog)});
            }
        }
    }
    return descs;
}


static std::string defaultExpressionColumnName(const ExprNode* expr) {
    if (!expr) return "expr";
    switch (expr->getType()) {
        case ExprType::COLUMN_REF:
            return static_cast<const ColumnRefExpr*>(expr)->column_name;
        case ExprType::STAR:
            return "*";
        case ExprType::CASE_WHEN:
            return "case_when";
        case ExprType::LITERAL_INT:
        case ExprType::LITERAL_FLOAT:
            return "literal";
        case ExprType::LITERAL_NULL:
            return "NULL";
        default:
            return "expr";
    }
}

static std::vector<std::string> inferResultColumnNames(const OperatorNode* node,
                                                       const Catalog& catalog) {
    std::vector<std::string> names;
    if (!node) return names;

    if (const auto* agg = findAggregateNode(node)) {
        for (std::size_t i = 0; i < agg->group_by_exprs.size(); ++i) {
            std::string name;
            if (i < agg->output_aliases.size()) name = agg->output_aliases[i];
            if (name.empty()) name = defaultExpressionColumnName(agg->group_by_exprs[i].get());
            names.push_back(name);
        }
        for (std::size_t i = 0; i < agg->aggregates.size(); ++i) {
            const std::size_t slot = agg->group_by_exprs.size() + i;
            std::string name;
            if (slot < agg->output_aliases.size()) name = agg->output_aliases[slot];
            if (name.empty()) name = agg->aggregates[i].func_name;
            names.push_back(name);
        }
        return names;
    }

    if (const auto* proj = findProjectionNode(node)) {
        std::vector<const TableScanNode*> scans;
        if (!proj->getChildren().empty()) {
            collectTableScansForResultLayout(proj->getChildren()[0].get(), scans);
        }
        std::size_t visible_idx = 0;
        for (std::size_t expr_idx = 0; expr_idx < proj->select_exprs.size(); ++expr_idx) {
            const auto& expr = proj->select_exprs[expr_idx];
            if (!expr) continue;
            if (expr->getType() == ExprType::STAR) {
                for (const auto* scan : scans) {
                    const auto& meta = catalog.getTableMetadata(scan->table_name);
                    for (const auto& col_name : meta.getColumnNames()) {
                        names.push_back(col_name);
                        ++visible_idx;
                    }
                }
                continue;
            }
            std::string name;
            if (expr_idx < proj->output_aliases.size()) name = proj->output_aliases[expr_idx];
            if (name.empty()) name = defaultExpressionColumnName(expr.get());
            names.push_back(name);
            ++visible_idx;
        }
        return names;
    }
    return names;
}



// Парсинг SQL → SelectStatement. Бросает runtime_error при ошибке.
static hsql::SelectStatement* parseSql(const std::string& sql,
                                        hsql::SQLParserResult& result) {
    hsql::SQLParser::parse(sql, &result);
    if (!result.isValid() || result.size() == 0) {
        throw std::runtime_error("SQL syntax error: " + std::string(result.errorMsg()?result.errorMsg():"empty statement"));
    }
    if (!result.getStatement(0)->isType(hsql::kStmtSelect)) throw std::runtime_error("Only SELECT statements are supported");
    return const_cast<hsql::SelectStatement*>(
        static_cast<const hsql::SelectStatement*>(result.getStatement(0)));
}

/** @brief Extract runtime values before optimization. No literal-bound copy enters GPU lowering. */
static PreparedQuery prepareQuery(
        const hsql::SelectStatement* ast, const Catalog& catalog) {
    return QueryTranslator{}.prepare(ast,catalog);
}

/** @brief Physical reuse includes catalog layout/statistics, but never base pointers or current values. */
static std::string catalogPlanIdentity(const Catalog& catalog) {
    std::ostringstream out;
    for (const auto& table : catalog.getTablesMetadata()) {
        out << table.getName().size() << ':' << table.getName() << ':' << table.getSize() << ':' << table.isFactTable() << ';';
        for (const auto& column : table.getColumnNames()) {
            out << column.size() << ':' << column << ':' << table.hasColumnStats(column) << ';';
            const auto storage = table.getColumnStorage(column);
            out << static_cast<unsigned>(storage.encoding) << ':' << storage.width << ';';
            if (!table.hasColumnStats(column)) continue;
            const auto& stats=table.getColumnStats(column);
            out << stats.min_value_ << ',' << stats.max_value_ << ',' << stats.cardinality_ << ','
                << stats.is_unique_ << ',' << stats.is_primary_key_ << ',' << stats.is_nullable_ << ';';
        }
    }
    return out.str();
}

/** @brief SQL text has no explicit typed-NULL syntax: retain the ready template's slot type before cache lookup. */
static void retainKnownNullTypes(PreparedQuery& prepared,const ParameterSchema& known) {
    if (prepared.parameters.size()!=known.size()) throw std::runtime_error("Known SQL parameter topology differs from its template");
    for (size_t i=0;i<known.size();++i) {
        if (!prepared.parameters[i].isNull() || known[i]==ParameterType::NullOnly) continue;
        prepared.query.parameter_schema[i]=known[i];
        prepared.parameters[i]=ParameterValue::null(known[i]);
    }
}

/** @brief Cached physical template and loaded code; never a current value vector or resource pointer. */
struct QueryEngine::CachedQuery {
    PlannedQuery physical;
    LoadedKernels kernels;
    size_t result_elements = 1;
    std::vector<ResultColumnDesc> columns;
    std::vector<std::string> names;
    bool dense = false;
};

// ============================================================================
// generateQueryCode — новый конвейер (Translator → Optimizer → JITVisitor)
// ============================================================================
std::string QueryEngine::generateQueryCode(const std::string& sql) {
    hsql::SQLParserResult parse_result;
    auto* ast = parseSql(sql, parse_result);

    auto prepared = prepareQuery(ast,*catalog_);
    const auto known=known_parameter_types_.find(prepared.query.structureKey()+catalogPlanIdentity(*catalog_));
    if (known!=known_parameter_types_.end()) retainKnownNullTypes(prepared,known->second);
    prepared.query.plan = Optimizer{}.optimize(std::move(prepared.query.plan));

    if (usesTextPlan(*prepared.query.plan, *catalog_, prepared.query.parameter_schema))
        return describePcudamodules(lowerTextPlan(*prepared.query.plan, *catalog_, prepared.query.parameter_schema));
    JITContext jit_ctx;
    jit_ctx.parameter_schema = prepared.query.parameter_schema;
    JITOperatorVisitor visitor(jit_ctx, *catalog_);
    prepared.query.plan->accept(visitor);

    return describePcudamodules(visitor.buildPhysicalPlan());
}

// ============================================================================
// executeQuery — новый конвейер
// ============================================================================
void QueryEngine::executeQuery(const std::string& sql, ExecutionContext* ctx) {
    const auto engine_start = Clock::now();
    if (sql.empty()) throw std::runtime_error("Empty query");
    if (!ctx || !ctx->runtime_) throw std::runtime_error("missing PCUDA execution context");
    ctx->resetForQuery();
    const bool overlap=ctx->config_.jit_overlap_experiment;
    if (overlap) ctx->timing_.model="pcuda-modules-overlap-exp-v1";
    const auto before = counters_;
    const auto frontend_start = Clock::now();
    const auto sources_ready = [&] { ctx->timing_.sql_to_pcuda_ms = elapsedMs(frontend_start, Clock::now()); };
    hsql::SQLParserResult parse_result;
    auto* ast = parseSql(sql, parse_result);
    auto prepared = prepareQuery(ast,*catalog_);
    const auto shape=prepared.query.structureKey()+catalogPlanIdentity(*catalog_);
    const auto known=known_parameter_types_.find(shape);
    if (known!=known_parameter_types_.end()) retainKnownNullTypes(prepared,known->second);
    const auto key = shape+prepared.query.schemaKey();
    auto found = plans_.find(key);
    ctx->timing_.prepare_ms = elapsedMs(engine_start,Clock::now());
    bool executed=false;
    const auto prepare_result=[&](const CachedQuery& query) {
        ctx->expected_result_size_=query.result_elements;
        ctx->result_columns_=query.columns;
        ctx->result_column_names_=query.names;
        ctx->result_is_dense_=query.dense;
        const auto start=Clock::now();
        preparePcudaplanResult(*ctx,query.result_elements,key);
        ctx->timing_.prepare_ms+=elapsedMs(start,Clock::now());
    };
    if (found == plans_.end()) {
        const auto codegen_start = Clock::now();
        auto query = std::make_shared<CachedQuery>();
        prepared.query.plan = Optimizer{}.optimize(std::move(prepared.query.plan));
        const auto* tree = prepared.query.plan.get();
        const bool text = usesTextPlan(*tree, *catalog_, prepared.query.parameter_schema);
        if (!text) if (const auto* aggregate = findAggregateNode(tree)) query->result_elements = aggregate->calculateResultSize(*catalog_);
        query->columns = inferResultColumns(tree,*catalog_,prepared.query.parameter_schema);
        query->names = inferResultColumnNames(tree,*catalog_);
        query->dense = text || findProjectionNode(tree) != nullptr;
        if (text) {
            if (const auto* aggregate = findAggregateNode(tree)) {
                if (!aggregate->select_output_slots.empty()) {
                    const auto columns = query->columns;
                    const auto names = query->names;
                    query->columns.clear(); query->names.clear();
                    for (auto slot : aggregate->select_output_slots) {
                        query->columns.push_back(columns.at(slot));
                        query->names.push_back(names.at(slot));
                    }
                }
            }
            query->physical = lowerTextPlan(*tree, *catalog_, prepared.query.parameter_schema);
        } else {
        JITContext jit;
        jit.parameter_schema = prepared.query.parameter_schema;
        JITOperatorVisitor visitor(jit,*catalog_);
        tree->accept(visitor);
        query->physical = visitor.buildPhysicalPlan();
        }
        recordModuleEvent(counters_.plan_codegen);
        ctx->timing_.codegen_ms = elapsedMs(codegen_start,Clock::now());
        if (overlap) {
            query->kernels=cache_->prepareOverlapped(query->physical,*compiler_,counters_,ctx->timing_.module_build,ctx->config_.jit_jobs,
                [&](const KernelLookup& ready) {prepare_result(*query);executor_->execute(ready,query->physical.plan,prepared.parameters,ctx);},sources_ready);
            executed=true;
        } else query->kernels = cache_->prepare(query->physical,*compiler_,counters_,ctx->timing_.module_build,sources_ready);
        found = plans_.emplace(key,std::move(query)).first;
    } else {
        sources_ready();
        recordModuleEvent(counters_.hits,found->second->kernels.size());
    }
    const auto& query = *found->second;
    known_parameter_types_[shape]=prepared.query.parameter_schema;
    if (!executed) prepare_result(query);
    ctx->timing_.compile_ms = ctx->timing_.module_build.build_elapsed_ms;
    ctx->timing_.library_load_ms = ctx->timing_.module_build.load_ms;
    auto& events = ctx->timing_.module_events;
    events.plan_codegen=counters_.plan_codegen-before.plan_codegen;
    events.module_codegen=counters_.module_codegen-before.module_codegen;
    events.compiler=counters_.compiler-before.compiler;
    events.linker=counters_.linker-before.linker;
    events.hits=counters_.hits-before.hits;
    events.misses=counters_.misses-before.misses;
    events.loads=counters_.loads-before.loads;
    events.symbols=counters_.symbols-before.symbols;
    if (!executed) executor_->execute(query.kernels,query.physical.plan,prepared.parameters,ctx);
    ctx->timing_.engine_ms = elapsedMs(engine_start,Clock::now());
    ctx->timing_.total_engine_ms = ctx->timing_.engine_ms;
}

} // namespace db
