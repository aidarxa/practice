#include "core/visitor.h"
#include "core/optimizer_rules.h" // extractTableNames, collectTableNames

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <stdexcept>

// ============================================================================
// crystal/utils.h helpers (getTableName, lookup, etc.) are used for
// mapping column names to tables and generating device-pointer names.
// We include it here for table prefix / size macro logic.
// ============================================================================
#include "crystal/utils.h"

namespace db {

// ============================================================================
// Static helpers
// ============================================================================

static std::string tablePrefix(const std::string& table_name) {
    if (table_name == "LINEORDER") return "lo";
    if (table_name == "SUPPLIER")  return "s";
    if (table_name == "CUSTOMER")  return "c";
    if (table_name == "PART")      return "p";
    if (table_name == "DDATE")     return "d";
    return "";
}

static std::string sizeMacroFor(const std::string& table_name) {
    if (table_name == "LINEORDER") return "LO_LEN";
    if (table_name == "SUPPLIER")  return "S_LEN";
    if (table_name == "CUSTOMER")  return "C_LEN";
    if (table_name == "PART")      return "P_LEN";
    if (table_name == "DDATE")     return "D_LEN";
    return "0";
}

// ============================================================================
// JITExprVisitor — constructor
// ============================================================================

JITExprVisitor::JITExprVisitor(JITContext& ctx,
                               std::stringstream& stream,
                               const std::string& target_mask,
                               bool is_or_context,
                               bool* first_pred)
    : ctx_(ctx), stream_(stream), target_mask_(target_mask),
      is_or_context_(is_or_context), first_pred_(first_pred),
      own_first_pred_(false)
{
    if (!first_pred_) {
        own_first_pred_ = true;
        first_pred_ = &local_first_pred_storage_;
    }
}

// ============================================================================
// JITExprVisitor — leaf visitors (ColumnRef, LiteralInt, LiteralFloat)
// These are only called directly by BinaryExpr::accept when the node is
// a leaf — they don't emit code on their own.
// ============================================================================

void JITExprVisitor::visit(const ColumnRefExpr& /*node*/) {
    // No standalone code generation — handled inside visitComparison
}

void JITExprVisitor::visit(const LiteralIntExpr& /*node*/) {
    // No standalone code generation — value used in visitComparison
}

void JITExprVisitor::visit(const LiteralFloatExpr& /*node*/) {
    // No standalone code generation
}

// ============================================================================
// JITExprVisitor — predSuffix
// ============================================================================

const char* JITExprVisitor::predSuffix(ExprType t) {
    switch (t) {
        case ExprType::OP_EQ:  return "Eq";
        case ExprType::OP_NEQ: return "NEq";
        case ExprType::OP_LT:  return "LT";
        case ExprType::OP_LTE: return "LTE";
        case ExprType::OP_GT:  return "GT";
        case ExprType::OP_GTE: return "GTE";
        default: return "Eq";
    }
}

// ============================================================================
// JITExprVisitor — ensureLoaded
// ============================================================================
/*
void JITExprVisitor::ensureLoaded(const std::string& col, const std::string& reg) {
    // Check if column already loaded in probe context
    if (ctx_.loaded_in_probe.count(col)) return;
    ctx_.loaded_in_probe.insert(col);

    // Register external column for ctx buffer extraction
    ctx_.external_columns.insert("d_" + col);

    stream_ << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
            << "d_" << col << " + tile_offset, tid, tile_offset, "
            << reg << ", num_tile_items);\n";
}
*/
// ============================================================================
// JITExprVisitor — visitComparison
// ============================================================================

void JITExprVisitor::visitComparison(const BinaryExpr& node) {
    const ExprNode* col_node = nullptr;
    const ExprNode* lit_node = nullptr;

    if (node.left && node.left->getType() == ExprType::COLUMN_REF && node.right && (node.right->getType() == ExprType::LITERAL_INT || node.right->getType() == ExprType::LITERAL_FLOAT)) {
        col_node = node.left.get(); lit_node = node.right.get();
    } else if (node.right && node.right->getType() == ExprType::COLUMN_REF && node.left && (node.left->getType() == ExprType::LITERAL_INT || node.left->getType() == ExprType::LITERAL_FLOAT)) {
        col_node = node.right.get(); lit_node = node.left.get();
    } else { return; }

    const auto* col = static_cast<const ColumnRefExpr*>(col_node);
    std::string col_name = col->column_name;

    std::string lit_value;
    if (lit_node->getType() == ExprType::LITERAL_INT) {
        lit_value = std::to_string(static_cast<const LiteralIntExpr*>(lit_node)->value);
    } else { lit_value = std::to_string(static_cast<const LiteralFloatExpr*>(lit_node)->value); }

    bool is_build = (&stream_ != &ctx_.probe_kernel);
    std::string reg_name = is_build ? "items" : col_name;

    if (is_build) {
        // ОПТИМИЗАЦИЯ: Избегаем повторных загрузок в items
        if (ctx_.current_items_col != col_name) {
            ctx_.external_columns.insert("d_" + col_name);
            stream_ << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                    << "d_" << col_name << " + tile_offset, tid, tile_offset, "
                    << reg_name << ", num_tile_items);\n";
            ctx_.current_items_col = col_name; // Запоминаем!
        }
    } else {
        if (!ctx_.loaded_in_probe.count(col_name)) {
            ctx_.loaded_in_probe.insert(col_name);
            ctx_.col_to_reg[col_name] = col_name;
            ctx_.external_columns.insert("d_" + col_name);
            stream_ << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                    << "d_" << col_name << " + tile_offset, tid, tile_offset, "
                    << reg_name << ", num_tile_items);\n";
        }
    }

    std::string prefix;
    if (is_or_context_) {
        prefix = *first_pred_ ? "BlockPred" : "BlockPredOr";
        if (*first_pred_) *first_pred_ = false;
    } else {
        prefix = *first_pred_ ? "BlockPred" : "BlockPredA";
        if (*first_pred_) *first_pred_ = false;
    }

    stream_ << "            " << prefix << predSuffix(node.op_type)
            << "<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, " << reg_name << ", "
            << target_mask_ << ", " << lit_value << ", num_tile_items);\n";
}
// ============================================================================
// JITExprVisitor — visit(BinaryExpr)
// ============================================================================

void JITExprVisitor::visit(const BinaryExpr& node) {
    if (node.op_type == ExprType::OP_AND) {
        // AND: both children accumulate into the same target_mask_
        if (node.left)  node.left->accept(*this);
        if (node.right) node.right->accept(*this);

    } else if (node.op_type == ExprType::OP_OR) {
        // OR: create a new temporary mask, visit children into it,
        // then AND the temp mask into target_mask_.
        std::string new_mask = ctx_.getNewMask();
        stream_ << "            int " << new_mask << "[ITEMS_PER_THREAD];\n";
        stream_ << "            InitFlagsZero<BLOCK_THREADS, ITEMS_PER_THREAD>("
                << new_mask << ");\n";

        // Children write into new_mask using OR-accumulation
        JITExprVisitor or_visitor(ctx_, stream_, new_mask,
                                  /*is_or_context=*/true,
                                  /*first_pred=*/nullptr);
        if (node.left)  node.left->accept(or_visitor);
        if (node.right) node.right->accept(or_visitor);

        // Apply the OR-mask to our parent mask
        stream_ << "            BlockApplyMaskAnd<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, "
                << target_mask_ << ", " << new_mask << ");\n";

    } else {
        // Comparison operator (EQ, LT, GT, GTE, LTE, NEQ)
        visitComparison(node);
    }
}

// ============================================================================
// JITOperatorVisitor — static helpers
// ============================================================================

std::string JITOperatorVisitor::tablePrefix(const std::string& table_name) {
    return db::tablePrefix(table_name);
}

std::string JITOperatorVisitor::sizeMacro(const std::string& table_name) {
    return sizeMacroFor(table_name);
}

// ============================================================================
// JITOperatorVisitor — constructor
// ============================================================================

JITOperatorVisitor::JITOperatorVisitor(JITContext& ctx, const Catalog& catalog)
    : ctx_(ctx), catalog_(catalog), active_stream_(&ctx.probe_kernel) {
    // select_kernel is always the last kernel
    ctx_.kernel_class_names.push_back("select_kernel");
}

// ============================================================================
// JITOperatorVisitor — visit(TableScanNode)
// ============================================================================

void JITOperatorVisitor::visit(const TableScanNode& /*node*/) {
    // TableScan is a leaf — it records which table we're scanning.
    // Actual BlockLoad calls are emitted lazily by Filter/HashJoin visitors
    // when they need specific columns.
}

// ============================================================================
// JITOperatorVisitor — visit(FilterNode)
// ============================================================================

void JITOperatorVisitor::visit(const FilterNode& node) {
    // 1. Generate code for children first (TableScan, sub-joins, etc.)
    for (const auto& child : node.getChildren()) {
        child->accept(*this);
    }

    // 2. Generate predicate filtering code using JITExprVisitor
    if (node.predicate) {
        bool* first_flag = nullptr;
        if (!in_build_kernel_) {
            first_flag = &probe_first_pred_;
        }
        JITExprVisitor expr_visitor(ctx_, *active_stream_, "flags",
                                    /*is_or_context=*/false,
                                    first_flag);
        node.predicate->accept(expr_visitor);
    }
}

// ============================================================================
// JITOperatorVisitor — computeBuildInfo
//
// Figures out the hash-table name, size, key_mins, variant (1 or 2)
// by inspecting the join condition and catalog metadata.
// ============================================================================

JITOperatorVisitor::BuildInfo JITOperatorVisitor::computeBuildInfo(
        const std::string& dim_table, const FilterNode* /*filter*/) const {
    BuildInfo bi;
    bi.dim_prefix = tablePrefix(dim_table);
    bi.size_macro = sizeMacro(dim_table);
    bi.ht_name = "d_" + bi.dim_prefix + "_hash_table";
    bi.variant = 1; // default: keys only
    bi.key_mins = "0";

    // Try to find pk column stats for key_mins and sizing
    try {
        const auto& meta = catalog_.getTableMetadata(dim_table);
        // Look for the primary key — conventionally the first column or the _key column
        for (const auto& col_name : meta.getColumnNames()) {
            if (col_name.find("key") != std::string::npos || col_name.find("Key") != std::string::npos) {
                if (meta.hasColumnStats(col_name)) {
                    const auto& stats = meta.getColumnStats(col_name);
                    int64_t range = stats.max_value_ - stats.min_value_ + 1;
                    bi.key_mins = std::to_string(stats.min_value_);
                    bi.ht_size_expr = std::to_string(range);
                    break;
                }
            }
        }
    } catch (...) {}

    if (bi.ht_size_expr.empty()) {
        bi.ht_size_expr = bi.size_macro;
    }

    return bi;
}

// ============================================================================
// JITOperatorVisitor — emitBuildKernel
// ============================================================================

void JITOperatorVisitor::emitBuildKernel(const std::string& dim_table,
                                          const FilterNode* filter,
                                          const TableScanNode& scan) {
    BuildInfo bi = computeBuildInfo(dim_table, filter);

    // 1. СНАЧАЛА определяем pk_col и variant ДО генерации заголовка!
    std::string pk_col;
    try {
        const auto& meta = catalog_.getTableMetadata(dim_table);
        for (const auto& col_name : meta.getColumnNames()) {
            if (col_name.find("key") != std::string::npos) {
                pk_col = col_name; break;
            }
        }
    } catch (...) {}

    // Автоматическое включение Variant 2
    for (const auto& [col, reg] : ctx_.col_to_reg) {
        if (getTableName(col) == dim_table && col != pk_col) {
            bi.variant = 2;
            bi.val_col = col;
            break;
        }
    }

    std::string kernel_name = ctx_.getNewBuildKernelName(bi.dim_prefix);

    // 2. Теперь пишем заголовок, ТОЧНО зная bi.variant
    ctx_.build_kernels << "    q.submit([&](sycl::handler& h) {\n";
    ctx_.build_kernels << "        int num_tiles = (" << bi.size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    ctx_.build_kernels << "        size_t local = BLOCK_THREADS;\n";
    ctx_.build_kernels << "        size_t global = num_tiles * BLOCK_THREADS;\n";
    ctx_.build_kernels << "        h.parallel_for<class " << kernel_name
                       << ">(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {\n";
    ctx_.build_kernels << "            int items[ITEMS_PER_THREAD];\n";
    ctx_.build_kernels << "            int flags[ITEMS_PER_THREAD];\n";
    if (bi.variant == 2) {
        ctx_.build_kernels << "            int items2[ITEMS_PER_THREAD];\n";
    }
    ctx_.build_kernels << "\n            int tid = it.get_local_linear_id();\n";
    ctx_.build_kernels << "            int tile_offset = it.get_group_linear_id() * TILE_SIZE;\n";
    ctx_.build_kernels << "            int num_tiles_local = (" << bi.size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    ctx_.build_kernels << "            int num_tile_items = TILE_SIZE;\n";
    ctx_.build_kernels << "            if (it.get_group_linear_id() == num_tiles_local - 1) {\n";
    ctx_.build_kernels << "                num_tile_items = " << bi.size_macro << " - tile_offset;\n";
    ctx_.build_kernels << "            }\n\n";
    ctx_.build_kernels << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);\n\n";

    ctx_.current_items_col = "";
    // --- filter predicates (if any) ---
    if (filter && filter->predicate) {
        bool build_first_pred = true;
        JITExprVisitor expr_vis(ctx_, ctx_.build_kernels, "flags", false, &build_first_pred);
        filter->predicate->accept(expr_vis);
    }

    if (!pk_col.empty()) {
        if (ctx_.current_items_col != pk_col) {
            ctx_.external_columns.insert("d_" + pk_col);
            ctx_.build_kernels << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                               << "d_" << pk_col << " + tile_offset, tid, tile_offset, items, num_tile_items);\n";
            ctx_.current_items_col = pk_col;
        }
    }

    if (bi.variant == 2 && !bi.val_col.empty()) {
        ctx_.external_columns.insert("d_" + bi.val_col);
        ctx_.build_kernels << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                           << "d_" << bi.val_col << " + tile_offset, tid, tile_offset, items2, num_tile_items);\n";
        ctx_.build_kernels << "            BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                           << "tid, items, items2, flags, " << bi.ht_name << ", " << bi.ht_size_expr
                           << ", " << bi.key_mins << ", num_tile_items);\n";
    } else {
        ctx_.build_kernels << "            BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                           << "tid, items, flags, " << bi.ht_name << ", " << bi.ht_size_expr
                           << ", " << bi.key_mins << ", num_tile_items);\n";
    }

    ctx_.build_kernels << "        });\n    });\n\n";
    std::string ht_size = (bi.variant == 2) ? "2*" + bi.ht_size_expr : bi.ht_size_expr;
    ctx_.hash_tables.push_back({bi.ht_name, "int", ht_size});
    bi.fk_col = "";
    build_infos_.push_back(bi);
}

// ============================================================================
// JITOperatorVisitor — visit(HashJoinNode)
// ============================================================================

void JITOperatorVisitor::visit(const HashJoinNode& node) {
    const auto& children = node.getChildren();
    if (children.size() < 2) return;

    const OperatorNode* left_child  = children[0].get();
    const OperatorNode* right_child = children[1].get();

    auto left_tables = collectTableNames(left_child);
    std::string dim_table;
    if (!left_tables.empty()) dim_table = *left_tables.begin();

    std::string pk_col, fk_col;
    if (node.join_condition) {
        auto tables = extractTableNames(node.join_condition.get());
        std::function<void(const ExprNode*)> findCols = [&](const ExprNode* e) {
            if (!e) return;
            if (e->getType() == ExprType::OP_EQ) {
                const auto* bin = static_cast<const BinaryExpr*>(e);
                if (bin->left && bin->left->getType() == ExprType::COLUMN_REF &&
                    bin->right && bin->right->getType() == ExprType::COLUMN_REF) {
                    const auto* lc = static_cast<const ColumnRefExpr*>(bin->left.get());
                    const auto* rc = static_cast<const ColumnRefExpr*>(bin->right.get());
                    std::string lt = getTableName(lc->column_name);
                    std::string rt = getTableName(rc->column_name);
                    if (lt == dim_table) { pk_col = lc->column_name; fk_col = rc->column_name; }
                    else if (rt == dim_table) { pk_col = rc->column_name; fk_col = lc->column_name; }
                }
            } else if (e->getType() == ExprType::OP_AND) {
                const auto* bin = static_cast<const BinaryExpr*>(e);
                findCols(bin->left.get());
                findCols(bin->right.get());
            }
        };
        findCols(node.join_condition.get());
    }

    const FilterNode* filter = nullptr;
    const TableScanNode* scan = nullptr;
    if (left_child->getType() == OperatorType::FILTER) {
        filter = static_cast<const FilterNode*>(left_child);
        if (!filter->getChildren().empty() && filter->getChildren()[0]->getType() == OperatorType::TABLE_SCAN) {
            scan = static_cast<const TableScanNode*>(filter->getChildren()[0].get());
        }
    } else if (left_child->getType() == OperatorType::TABLE_SCAN) {
        scan = static_cast<const TableScanNode*>(left_child);
    }

    if (scan) {
        emitBuildKernel(dim_table, filter, *scan);
        if (!build_infos_.empty() && !fk_col.empty()) {
            build_infos_.back().fk_col = fk_col;
        }
    } else {
        std::stringstream temp_build;
        auto* saved_stream = active_stream_;
        active_stream_ = &temp_build;
        bool saved_in_build = in_build_kernel_;
        in_build_kernel_ = true;
        left_child->accept(*this);
        in_build_kernel_ = saved_in_build;
        active_stream_ = saved_stream;
        ctx_.build_kernels << temp_build.str();
    }

    // ИСПРАВЛЕНИЕ: Запоминаем индекс хеш-таблицы ПЕРЕД обходом правой ветви (Right-Deep)
    size_t build_idx = build_infos_.size() - 1;

    active_stream_ = &ctx_.probe_kernel;
    right_child->accept(*this);

    // ИСПРАВЛЕНИЕ: Используем сохраненный индекс для Probe
    if (build_idx < build_infos_.size()) {
        const auto& bi = build_infos_[build_idx];
        if (!bi.fk_col.empty()) {
            ctx_.external_columns.insert("d_" + bi.fk_col);
            ctx_.probe_kernel << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                              << "d_" << bi.fk_col << " + tile_offset, tid, tile_offset, items, num_tile_items);\n";

            if (bi.variant == 2 && !bi.val_col.empty()) {
                // Отмечаем колонку как загруженную, чтобы AggregateNode не грузил её повторно!
                ctx_.loaded_in_probe.insert(bi.val_col);
                ctx_.col_to_reg[bi.val_col] = bi.val_col;

                ctx_.probe_kernel << "            BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                                  << "tid, items, " << bi.val_col << ", flags, "
                                  << bi.ht_name << ", " << bi.ht_size_expr
                                  << ", " << bi.key_mins << ", num_tile_items);\n";
            } else {
                ctx_.probe_kernel << "            BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                                  << "tid, items, flags, "
                                  << bi.ht_name << ", " << bi.ht_size_expr
                                  << ", " << bi.key_mins << ", num_tile_items);\n";
            }
        }
    }
}

// ============================================================================
// Helper: translate ExprNode math into C++ string (for aggregation expressions)
// ============================================================================
static std::string translateMathExpr(
        const ExprNode* expr,
        const std::unordered_map<std::string, std::string>& col_to_reg,
        bool cast_to_ull = false) {
    if (!expr) return "0";

    switch (expr->getType()) {
        case ExprType::COLUMN_REF: {
            const auto* col = static_cast<const ColumnRefExpr*>(expr);
            auto it = col_to_reg.find(col->column_name);
            std::string res = (it != col_to_reg.end())
                ? it->second + "[i]"
                : col->column_name + "[i]";                     // не учитывает префиксы, вместо d_lo_extendedprice и d_lo_discount пишет просто 
            if (cast_to_ull) return "(unsigned long long)" + res;
            return res;
        }
        case ExprType::LITERAL_INT: {
            const auto* lit = static_cast<const LiteralIntExpr*>(expr);
            return std::to_string(lit->value);
        }
        case ExprType::LITERAL_FLOAT: {
            const auto* lit = static_cast<const LiteralFloatExpr*>(expr);
            return std::to_string(lit->value);
        }
        case ExprType::OP_ADD:
        case ExprType::OP_SUB:
        case ExprType::OP_MUL:
        case ExprType::OP_DIV: {
            const auto* bin = static_cast<const BinaryExpr*>(expr);
            std::string left = translateMathExpr(bin->left.get(), col_to_reg, cast_to_ull);
            std::string right = translateMathExpr(bin->right.get(), col_to_reg, false);
            const char* op = " ? ";
            switch (expr->getType()) {
                case ExprType::OP_ADD: op = " + "; break;
                case ExprType::OP_SUB: op = " - "; break;
                case ExprType::OP_MUL: op = " * "; break;
                case ExprType::OP_DIV: op = " / "; break;
                default: break;
            }
            return "(" + left + op + right + ")";
        }
        default:
            return "0";
    }
}

// ============================================================================
// Helper: generate perfect-hash expression from GROUP BY columns
// ============================================================================
static std::pair<std::string, uint64_t> generatePerfectHash(
        const std::vector<std::unique_ptr<ExprNode>>& group_by,
        const Catalog& catalog) {
    std::string hash_expr;
    uint64_t total_size = 1;

    for (std::size_t i = 0; i < group_by.size(); ++i) {
        if (group_by[i]->getType() != ExprType::COLUMN_REF) continue;
        const auto* col = static_cast<const ColumnRefExpr*>(group_by[i].get());
        std::string col_name = col->column_name;
        std::string table_name = getTableName(col_name);

        uint64_t min_val = 0;
        uint64_t card = 1;
        try {
            const auto& meta = catalog.getTableMetadata(table_name);
            if (meta.hasColumnStats(col_name)) {
                const auto& stats = meta.getColumnStats(col_name);
                min_val = stats.min_value_;
                card = stats.cardinality_;
            } else if (meta.hasColumnStats(col_name)) {
                const auto& stats = meta.getColumnStats(col_name);
                min_val = stats.min_value_;
                card = stats.cardinality_;
            }
        } catch (...) {}

        std::string term = "(" + col_name + "[i] - " + std::to_string(min_val) + ")";
        if (i == 0) {
            hash_expr = term;
        } else {
            hash_expr = "(" + hash_expr + " * " + std::to_string(card) + " + " + term + ")";
        }
        total_size *= card;
    }

    if (!hash_expr.empty()) {
        hash_expr = "(" + hash_expr + ") % " + std::to_string(total_size);
    } else {
        hash_expr = "0";
    }
    return {hash_expr, total_size};
}

// ============================================================================
// JITOperatorVisitor — visit(AggregateNode)
// ============================================================================

void JITOperatorVisitor::visit(const AggregateNode& node) {
    // -------------------------------------------------------------------------
    // ШАГ 1: Предварительная регистрация колонок в col_to_reg.
    // Мы должны сделать это ДО вызова accept для детей, чтобы дочерние 
    // HashJoinNode знали, какие колонки измерений потребуются (для Variant 2).
    // -------------------------------------------------------------------------

    // Регистрируем колонки, участвующие в GROUP BY
    for (const auto& g : node.group_by_exprs) {
        if (g->getType() == ExprType::COLUMN_REF) {
            const auto* col = static_cast<const ColumnRefExpr*>(g.get());
            ctx_.col_to_reg[col->column_name] = col->column_name;
        }
    }

    // Рекурсивно регистрируем все колонки из математических выражений агрегации
    for (const auto& agg : node.aggregates) {
        if (agg.agg_expr) {
            std::function<void(const ExprNode*)> walk = [&](const ExprNode* e) {
                if (!e) return;
                if (e->getType() == ExprType::COLUMN_REF) {
                    const auto* col = static_cast<const ColumnRefExpr*>(e);
                    ctx_.col_to_reg[col->column_name] = col->column_name;
                }
                // Проверяем, является ли узел бинарным оператором (мат. логика или предикат)
                // Используем проверку типов из expressions.h
                if (e->getType() >= ExprType::OP_AND) { 
                    const auto* bin = static_cast<const BinaryExpr*>(e);
                    walk(bin->left.get());
                    walk(bin->right.get());
                }
            };
            walk(agg.agg_expr.get());
        }
    }

    // -------------------------------------------------------------------------
    // ШАГ 2: Обход дочерних узлов.
    // Теперь, когда col_to_reg заполнен, HashJoinNode правильно сгенерирует
    // BlockProbeAndPHT_2 для нужных измерений.
    // -------------------------------------------------------------------------
    for (const auto& child : node.getChildren()) {
        child->accept(*this);
    }

    // Определяем наличие группировки
    bool has_group_by = !node.group_by_exprs.empty();
    if (has_group_by) {
        // -------------------------------------------------------------------------
        // ШАГ 3: Загрузка колонок фактовой таблицы для агрегации
        // -------------------------------------------------------------------------
        for (const auto& [col, reg] : ctx_.col_to_reg) {
            // Если колонка еще не была загружена в текущем ядре (Probe Phase)
            // Это гарантирует, что мы не загружаем колонки измерений из Variant 2 дважды
            if (!ctx_.loaded_in_probe.count(col)) {
                ctx_.external_columns.insert("d_" + col);
                ctx_.loaded_in_probe.insert(col);
                *active_stream_ << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                                << "d_" << col << " + tile_offset, tid, tile_offset, "
                                << reg << ", num_tile_items);\n";
            }
        }

        // -------------------------------------------------------------------------
        // ШАГ 4: Вычисление формулы Perfect Hash
        // -------------------------------------------------------------------------
        // generatePerfectHash использует схему Горнера и оборачивает итог в % total_size
        auto [hash_expr, total_size] = generatePerfectHash(node.group_by_exprs, catalog_);

        uint8_t tuple_size = static_cast<uint8_t>(node.group_by_exprs.size() + node.aggregates.size());
        ctx_.tuple_size = tuple_size;
        uint64_t res_array_size = total_size * tuple_size;
        ctx_.result_size_expr = std::to_string(res_array_size);

        // -------------------------------------------------------------------------
        // ШАГ 5: Генерация цикла групповой агрегации
        // -------------------------------------------------------------------------
        *active_stream_ << "\n            #pragma unroll\n";
        *active_stream_ << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n";
        *active_stream_ << "                if (flags[i] && (tid + BLOCK_THREADS * i < num_tile_items)) {\n";
        *active_stream_ << "                    int hash = " << hash_expr << ";\n";

        // Запись ключей группировки (значений измерений и фактов)
        for (std::size_t g = 0; g < node.group_by_exprs.size(); ++g) {
            if (node.group_by_exprs[g]->getType() == ExprType::COLUMN_REF) {
                const auto* col = static_cast<const ColumnRefExpr*>(node.group_by_exprs[g].get());
                *active_stream_ << "                    d_result[hash*"
                                << std::to_string(tuple_size) << "+"
                                << std::to_string(g) << "] = "
                                << col->column_name << "[i];\n";
            }
        }

        // Атомарное добавление значений агрегации
        uint8_t agg_off = static_cast<uint8_t>(node.group_by_exprs.size());
        for (std::size_t a = 0; a < node.aggregates.size(); ++a) {
            uint8_t slot = agg_off + static_cast<uint8_t>(a);
            
            // Транслируем математику агрегации с флагом cast_to_ull = true
            // Это защищает нас от 32-битного Integer Overflow перед умножением
            std::string agg_val = translateMathExpr(
                node.aggregates[a].agg_expr.get(), ctx_.col_to_reg, true);

            *active_stream_ << "                    sycl::atomic_ref<unsigned long long, "
                            << "sycl::memory_order::relaxed, sycl::memory_scope::device, "
                            << "sycl::access::address_space::global_space> atomic_agg_"
                            << std::to_string(a) << "(d_result[hash*"
                            << std::to_string(tuple_size) << "+"
                            << std::to_string(slot) << "]);\n";
            *active_stream_ << "                    atomic_agg_" << std::to_string(a)
                            << ".fetch_add(" << agg_val << ");\n";
        }

        *active_stream_ << "                }\n            }\n";

    } else {
        // -------------------------------------------------------------------------
        // АЛЬТЕРНАТИВА: Скалярная агрегация (запросы типа Q1.x, без GROUP BY)
        // -------------------------------------------------------------------------
        ctx_.tuple_size = 1;
        ctx_.result_size_expr = "1";

        // Загрузка нужных колонок фактовой таблицы
        for (const auto& [col, reg] : ctx_.col_to_reg) {
            if (!ctx_.loaded_in_probe.count(col)) {
                ctx_.external_columns.insert("d_" + col);
                ctx_.loaded_in_probe.insert(col);
                *active_stream_ << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                                << "d_" << col << " + tile_offset, tid, tile_offset, "
                                << reg << ", num_tile_items);\n";
            }
        }

        std::string agg_expr = "0";
        if (!node.aggregates.empty() && node.aggregates[0].agg_expr) {
            agg_expr = translateMathExpr(node.aggregates[0].agg_expr.get(), ctx_.col_to_reg, true);
        }

        *active_stream_ << "\n            unsigned long long sum = 0;\n";
        *active_stream_ << "            #pragma unroll\n";
        *active_stream_ << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n";
        *active_stream_ << "                if (tid + BLOCK_THREADS * i < num_tile_items) {\n";
        *active_stream_ << "                    if (flags[i]) {\n";
        *active_stream_ << "                        sum += " << agg_expr << ";\n";
        *active_stream_ << "                    }\n                }\n            }\n";
        
        // Быстрая редукция суммы на уровне группы потоков (Group Reduce)
        *active_stream_ << "            unsigned long long aggregate = "
                        << "sycl::reduce_over_group(it.get_group(), sum, "
                        << "sycl::plus<unsigned long long>());\n";
        
        // Только 0-й поток из группы пишет результат в глобальную память
        *active_stream_ << "            if (tid == 0) {\n";
        *active_stream_ << "                sycl::atomic_ref<unsigned long long, "
                        << "sycl::memory_order::relaxed, sycl::memory_scope::device, "
                        << "sycl::access::address_space::global_space> "
                        << "atomic_result(d_result[0]);\n";
        *active_stream_ << "                atomic_result.fetch_add(aggregate);\n";
        *active_stream_ << "            }\n";
    }
}

// ============================================================================
// JITOperatorVisitor — generateCode
//
// Assembles the complete C++ source file for extern "C" void execute_query().
// ============================================================================

std::string JITOperatorVisitor::generateCode() const {
    std::stringstream code;

    // --- includes ---
    code << "#include \"core/execution.h\"\n";
    code << "#include <sycl/sycl.hpp>\n";
    code << "#include \"crystal/load.h\"\n";
    code << "#include \"crystal/pred.h\"\n";
    code << "#include \"crystal/join.h\"\n";
    code << "#include \"crystal/utils.h\"\n\n";
    code << "using namespace sycl;\n\n";
    code << "constexpr int BLOCK_THREADS = 128;\n";
    code << "constexpr int ITEMS_PER_THREAD = 4;\n";
    code << "constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;\n\n";

    // --- forward declarations ---
    for (const auto& name : ctx_.kernel_class_names) {
        code << "class " << name << ";\n";
    }

    // --- function signature ---
    code << "\nextern \"C\" void execute_query(db::ExecutionContext* ctx) {\n";
    code << "    sycl::queue& q = *(ctx->q_);\n";

    // --- external column buffers ---
    for (const auto& col : ctx_.external_columns) {
        code << "    int* " << col << " = ctx->getBuffer<int>(\"" << col << "\");\n";
    }

    // --- result buffer ---
    code << "    unsigned long long* d_result = ctx->getResultPointer();\n";

    // --- hash table allocations ---
    for (const auto& ht : ctx_.hash_tables) {
        code << "    " << ht.type << "* " << ht.name
             << " = sycl::malloc_device<" << ht.type << ">("
             << ht.size_expr << ", q);\n";
        code << "    q.memset(" << ht.name << ", 0, "
             << ht.size_expr << " * sizeof(" << ht.type << "));\n";
    }

    // --- zero result buffer ---
    code << "    q.memset(d_result, 0, " << ctx_.result_size_expr
         << " * sizeof(unsigned long long));\n\n";

    // --- build kernels ---
    code << ctx_.build_kernels.str();

    // --- probe/select kernel ---
    code << "    q.submit([&](sycl::handler& h) {\n";
    code << "        int num_tiles = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;\n";
    code << "        size_t local = BLOCK_THREADS;\n";
    code << "        size_t global = num_tiles * BLOCK_THREADS;\n";
    code << "        h.parallel_for<class select_kernel>"
         << "(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {\n";

    // Registers
    code << "            int items[ITEMS_PER_THREAD];\n";
    code << "            int flags[ITEMS_PER_THREAD];\n";

    // Additional named registers for group-by / agg columns
    std::set<std::string> emitted_regs;
    for (const auto& [col, reg] : ctx_.col_to_reg) {
        if (reg != "items" && reg != "flags" && !emitted_regs.count(reg)) {
            code << "            int " << reg << "[ITEMS_PER_THREAD];\n";
            emitted_regs.insert(reg);
        }
    }

    code << "\n            int tid = it.get_local_linear_id();\n";
    code << "            int tile_offset = it.get_group_linear_id() * TILE_SIZE;\n";
    code << "            int num_tiles_local = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;\n";
    code << "            int num_tile_items = TILE_SIZE;\n";
    code << "            if (it.get_group_linear_id() == num_tiles_local - 1) {\n";
    code << "                num_tile_items = LO_LEN - tile_offset;\n";
    code << "            }\n\n";
    code << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);\n\n";

    // probe kernel body
    code << ctx_.probe_kernel.str();

    code << "        });\n    });\n\n";

    // --- wait and free ---
    code << "    q.wait();\n\n";
    for (const auto& ht : ctx_.hash_tables) {
        code << "    sycl::free(" << ht.name << ", q);\n";
    }

    code << "    ctx->tuple_size_ = " << ctx_.tuple_size << ";\n";
    code << "}\n";

    return code.str();
}

} // namespace db
