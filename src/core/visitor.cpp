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
// JITExprVisitor — translateInlineExpr
// ============================================================================

std::string JITExprVisitor::translateInlineExpr(const ExprNode* expr, bool is_probe) {
    if (!expr) return "";

    switch (expr->getType()) {
        case ExprType::COLUMN_REF: {
            const auto* col = static_cast<const ColumnRefExpr*>(expr);
            std::string col_name = col->column_name;
            
            if (is_probe && !ctx_.loaded_in_probe.count(col_name)) {
                ctx_.loaded_in_probe.insert(col_name);
                ctx_.col_to_reg[col_name] = col_name;
                ctx_.external_columns.insert("d_" + col_name);
                stream_ << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                        << "d_" << col_name << " + tile_offset, tid, tile_offset, "
                        << col_name << ", num_tile_items);\n";
            }
            return col_name + "[i]";
        }
        case ExprType::LITERAL_INT:
            return std::to_string(static_cast<const LiteralIntExpr*>(expr)->value);
        case ExprType::LITERAL_FLOAT:
            return std::to_string(static_cast<const LiteralFloatExpr*>(expr)->value);
            
        case ExprType::OP_ADD:
            return "db::safe_add(" + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->left.get(), is_probe) + ", " 
                                   + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->right.get(), is_probe) + ")";
        case ExprType::OP_SUB:
            return "db::safe_sub(" + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->left.get(), is_probe) + ", " 
                                   + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->right.get(), is_probe) + ")";
        case ExprType::OP_MUL:
            return "db::safe_mul(" + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->left.get(), is_probe) + ", " 
                                   + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->right.get(), is_probe) + ")";
        case ExprType::OP_DIV:
            return "db::safe_div(" + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->left.get(), is_probe) + ", " 
                                   + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->right.get(), is_probe) + ")";
        default: 
            return "";
    }
}

// ============================================================================
// JITExprVisitor — visitComparison
// ============================================================================

void JITExprVisitor::visitComparison(const BinaryExpr& node) {
    // ===== ВЕТКА: Theta Join (COLUMN OP COLUMN) =====
    if (node.left && node.left->getType() == ExprType::COLUMN_REF &&
        node.right && node.right->getType() == ExprType::COLUMN_REF) {

        const auto* left_col  = static_cast<const ColumnRefExpr*>(node.left.get());
        const auto* right_col = static_cast<const ColumnRefExpr*>(node.right.get());
        const std::string col1_name = left_col->column_name;
        const std::string col2_name = right_col->column_name;

        bool is_build = false; // push-model: JITExprVisitor always in probe context

        auto load_col_if_needed = [&](const std::string& cname, const std::string& reg) {
            if (!ctx_.loaded_in_probe.count(cname)) {
                ctx_.loaded_in_probe.insert(cname);
                ctx_.col_to_reg[cname] = reg;
                ctx_.external_columns.insert("d_" + cname);
                stream_ << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                        << "d_" << cname << " + tile_offset, tid, tile_offset, "
                        << reg << ", num_tile_items);\n";
            }
        };

        // Загрузка обеих колонок
        load_col_if_needed(col1_name, col1_name);
        load_col_if_needed(col2_name, col2_name);

        // Выбор префикса BlockPred / BlockPredA / BlockPredOr
        std::string prefix;
        if (is_or_context_) {
            prefix = *first_pred_ ? "BlockPred" : "BlockPredOr";
        } else {
            prefix = *first_pred_ ? "BlockPred" : "BlockPredA";
        }
        if (*first_pred_) *first_pred_ = false;

        // Генерация вызова
        stream_ << "            " << prefix << predSuffix(node.op_type) << "_Cols"
                << "<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, " 
                << col1_name << ", " << col2_name << ", "
                << target_mask_ << ", num_tile_items);\n";
        
        return;
    }

    // ===== ВЕТКА: COLUMN OP LITERAL =====
    bool is_build = false; // push-model: always probe context

    // ВЕТКА: Probe Phase Inline Math
    if (!is_build) {
        std::string left_expr = translateInlineExpr(node.left.get(), true);
        std::string right_expr = translateInlineExpr(node.right.get(), true);
        
        std::string func;
        switch(node.op_type) {
            case ExprType::OP_LT:  func = "db::safe_lt"; break;
            case ExprType::OP_GT:  func = "db::safe_gt"; break;
            case ExprType::OP_LTE: func = "db::safe_lte"; break;
            case ExprType::OP_GTE: func = "db::safe_gte"; break;
            case ExprType::OP_EQ:  func = "db::safe_eq"; break;
            case ExprType::OP_NEQ: func = "db::safe_neq"; break;
            default: return;
        }

        std::string condition = func + "(" + left_expr + ", " + right_expr + ")";

        stream_ << "            #pragma unroll\n"
                << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n"
                << "                if (tid + BLOCK_THREADS * i < num_tile_items) {\n";
        
        if (is_or_context_) {
            if (*first_pred_) {
                stream_ << "                    " << target_mask_ << "[i] = " << condition << ";\n";
                *first_pred_ = false;
            } else {
                stream_ << "                    " << target_mask_ << "[i] = " << target_mask_ << "[i] || " << condition << ";\n";
            }
        } else {
            if (*first_pred_) {
                stream_ << "                    " << target_mask_ << "[i] = " << condition << ";\n";
                *first_pred_ = false;
            } else {
                stream_ << "                    " << target_mask_ << "[i] = " << target_mask_ << "[i] && " << condition << ";\n";
            }
        }
        
        stream_ << "                }\n            }\n";
        return;
    }

    // ВЕТКА: Build Phase (COLUMN OP LITERAL ONLY)
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

    std::string reg_name = "items";

    // ОПТИМИЗАЦИЯ: Избегаем повторных загрузок в items
    if (ctx_.current_items_col != col_name) {
        ctx_.external_columns.insert("d_" + col_name);
        stream_ << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                << "d_" << col_name << " + tile_offset, tid, tile_offset, "
                << reg_name << ", num_tile_items);\n";
        ctx_.current_items_col = col_name; // Запоминаем!
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
        if (!is_or_context_) {
            node.left->accept(*this);
            node.right->accept(*this);
        } else {
            // AND внутри OR: создаем новую маску и применяем ее
            std::string new_mask = ctx_.getNewMask();
            stream_ << "            int " << new_mask << "[ITEMS_PER_THREAD];\n";
            stream_ << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(" << new_mask << ");\n";
            
            bool first_pred = true;
            JITExprVisitor and_visitor(ctx_, stream_, new_mask, false, &first_pred);
            node.left->accept(and_visitor);
            node.right->accept(and_visitor);
            
            stream_ << "            BlockApplyMaskOr<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, " 
                    << target_mask_ << ", " << new_mask << ");\n";
            if (first_pred_ && *first_pred_) *first_pred_ = false;
        }
    } else if (node.op_type == ExprType::OP_OR) {
        if (is_or_context_) {
            // OR внутри OR: Сплющиваем! Не создаем новую маску, пишем напрямую в target_mask_
            node.left->accept(*this);
            node.right->accept(*this);
        } else {
            // OR внутри AND: выделяем один накопитель для всей цепочки OR
            std::string new_mask = ctx_.getNewMask();
            stream_ << "            int " << new_mask << "[ITEMS_PER_THREAD];\n";
            stream_ << "            InitFlagsZero<BLOCK_THREADS, ITEMS_PER_THREAD>(" << new_mask << ");\n";
            
            bool first_pred = true;
            JITExprVisitor or_visitor(ctx_, stream_, new_mask, true, &first_pred);
            node.left->accept(or_visitor);
            node.right->accept(or_visitor);
            
            stream_ << "            BlockApplyMaskAnd<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, " 
                    << target_mask_ << ", " << new_mask << ");\n";
            if (first_pred_ && *first_pred_) *first_pred_ = false;
        }
    } else {
        visitComparison(node);
    }
}

// ============================================================================
// ============================================================================
// JITOperatorVisitor — static helpers
// ============================================================================

std::string JITOperatorVisitor::tablePrefix(const std::string& table_name) {
    return db::tablePrefix(table_name);
}

std::string JITOperatorVisitor::sizeMacro(const std::string& table_name) {
    return sizeMacroFor(table_name);
}

// Helper to extract columns for a table from an expression tree
static void extractColumnsForTable(const ExprNode* e, const std::string& table_name, std::vector<std::string>& cols) {
    if (!e) return;
    if (e->getType() == ExprType::COLUMN_REF) {
        const auto* col = static_cast<const ColumnRefExpr*>(e);
        if (getTableName(col->column_name) == table_name) {
            if (std::find(cols.begin(), cols.end(), col->column_name) == cols.end()) {
                cols.push_back(col->column_name);
            }
        }
    } 
    // Безопасный каст: только если это действительно бинарный оператор (тип >= OP_AND)
    else if (e->getType() >= ExprType::OP_AND) {
        const auto* bin = static_cast<const BinaryExpr*>(e);
        if (bin->left) extractColumnsForTable(bin->left.get(), table_name, cols);
        if (bin->right) extractColumnsForTable(bin->right.get(), table_name, cols);
    }
}

// Recursive helper to find all columns needed by the whole tree for a specific table
static void findAllColumnsForTable(const OperatorNode* node, const std::string& table_name, std::vector<std::string>& cols) {
    if (!node) return;
    
    if (node->getType() == OperatorType::FILTER) {
        const auto* filter = static_cast<const FilterNode*>(node);
        extractColumnsForTable(filter->predicate.get(), table_name, cols);
    } else if (node->getType() == OperatorType::HASH_JOIN) {
        const auto* join = static_cast<const HashJoinNode*>(node);
        extractColumnsForTable(join->join_condition.get(), table_name, cols);
    } else if (node->getType() == OperatorType::AGGREGATE) {
        const auto* agg = static_cast<const AggregateNode*>(node);
        for (const auto& g : agg->group_by_exprs) {
            extractColumnsForTable(g.get(), table_name, cols);
        }
        for (const auto& a : agg->aggregates) {
            extractColumnsForTable(a.agg_expr.get(), table_name, cols);
        }
    }
    
    // We don't recurse into HashJoin's left child (build side) if we are scanning fact table (probe side),
    // but the columns collected here are globally scoped by table_name anyway.
    for (const auto& child : node->getChildren()) {
        findAllColumnsForTable(child.get(), table_name, cols);
    }
}

// ============================================================================
// JITOperatorVisitor — constructor
// ============================================================================

JITOperatorVisitor::JITOperatorVisitor(JITContext& ctx, const Catalog& catalog)
    : ctx_(ctx), catalog_(catalog) {
}

// ============================================================================
// visit() — Legacy OperatorVisitor interface.
// The root is always AggregateNode; execution.cpp calls accept(visitor) on it.
// visit(AggregateNode) is the single entry point that kicks off produce().
// The other three stubs exist only to satisfy the pure-virtual contract.
// ============================================================================

void JITOperatorVisitor::visit(const AggregateNode& node) {
    // 1. ПОЧИНКА УКАЗАТЕЛЕЙ: Восстанавливаем parent_ для всего дерева
    // Это необходимо, так как Оптимизатор мог разрушить связи при перестройке плана.
    std::function<void(const OperatorNode*, OperatorNode*)> repairParents = [&](const OperatorNode* current, OperatorNode* parent) {
        if (!current) return;
        const_cast<OperatorNode*>(current)->parent_ = parent;
        for (const auto& child : current->getChildren()) {
            repairParents(child.get(), const_cast<OperatorNode*>(current));
        }
    };
    repairParents(&node, nullptr);

    // 2. СБОР ВСЕХ КОЛОНОК: Обходим все дерево один раз для надежного обнаружения требований
    agg_cols_.clear();
    filter_cols_.clear();
    collectAllColumnsFromTree(&node);

    // 3. Старт генерации конвейера
    produce(&node, ctx_);
}

void JITOperatorVisitor::visit(const TableScanNode& /*node*/) {}
void JITOperatorVisitor::visit(const FilterNode&    /*node*/) {}
void JITOperatorVisitor::visit(const HashJoinNode&  /*node*/) {}

// ============================================================================
// Main Push-Model Dispatchers
// ============================================================================

void JITOperatorVisitor::produce(const OperatorNode* node, JITContext& ctx) {
    // Дерево уже починено в visit(AggregateNode), поэтому здесь просто диспетчеризация
    if (node->getType() == OperatorType::TABLE_SCAN) produceTableScan(static_cast<const TableScanNode*>(node), ctx);
    else if (node->getType() == OperatorType::FILTER) produceFilter(static_cast<const FilterNode*>(node), ctx);
    else if (node->getType() == OperatorType::HASH_JOIN) produceHashJoin(static_cast<const HashJoinNode*>(node), ctx);
    else if (node->getType() == OperatorType::AGGREGATE) produceAggregate(static_cast<const AggregateNode*>(node), ctx);
}

void JITOperatorVisitor::consume(const OperatorNode* node, JITContext& ctx, const OperatorNode* sender, const std::vector<std::string>& active_vars) {
    if (!node) return;
    switch (node->getType()) {
        case OperatorType::FILTER: consumeFilter(static_cast<const FilterNode*>(node), ctx, sender, active_vars); break;
        case OperatorType::HASH_JOIN: consumeHashJoin(static_cast<const HashJoinNode*>(node), ctx, sender, active_vars); break;
        case OperatorType::AGGREGATE: consumeAggregate(static_cast<const AggregateNode*>(node), ctx, sender, active_vars); break;
        default: break;
    }
}

// ============================================================================
// Handlers
// ============================================================================

void JITOperatorVisitor::produceTableScan(const TableScanNode* node, JITContext& ctx) {
    ctx.startNewPipeline("Scan_" + node->table_name);
    const std::string size_macro = sizeMacro(node->table_name);

    // Collect all columns needed from this table (traversing upward through parents)
    const OperatorNode* root = node;
    while (root->parent_) root = root->parent_;

    std::vector<std::string> active_vars;
    findAllColumnsForTable(root, node->table_name, active_vars);

    // 1. Пишем заголовок (подготовка параметров запуска)
    std::stringstream header;
    header << "    q.submit([&](sycl::handler& h) {\n";
    header << "        int num_tiles = (" << size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    header << "        h.parallel_for<class " << ctx.current_pipeline->kernel_name << ">"
           << "(sycl::nd_range<1>(num_tiles * BLOCK_THREADS, BLOCK_THREADS), [=](sycl::nd_item<1> it) {\n";
    header << "            int tid = it.get_local_linear_id();\n";
    header << "            int tile_offset = it.get_group_linear_id() * TILE_SIZE;\n";
    header << "            int num_tile_items = (it.get_group_linear_id() == it.get_group_range(0) - 1) ? "
           << size_macro << " - tile_offset : TILE_SIZE;\n";

    // 2. Временно перехватываем вывод ядра
    // Мы очищаем текущий поток, чтобы захватить только логику, генерируемую consumeVector.
    std::string existing_body = ctx.current_pipeline->kernel_body.str();
    ctx.current_pipeline->kernel_body.str(""); 

    // 3. ЗАПУСК КОНВЕЙЕРА (Эта фаза заполнит ctx.col_to_reg)
    if (node->parent_) {
        consumeVector(node->parent_, ctx, node, active_vars);
    }
    std::string pipeline_logic = ctx.current_pipeline->kernel_body.str();

    // 4. Теперь мы знаем все нужные регистры (items, items2, items3...). Генерируем декларации!
    std::stringstream declarations;
    declarations << "            int items[ITEMS_PER_THREAD];\n";
    declarations << "            int flags[ITEMS_PER_THREAD];\n";
    declarations << "            int items2[ITEMS_PER_THREAD];\n"; // ВСЕГДА объявляем items2
    
    std::set<std::string> emitted;
    for (const auto& pair : ctx.col_to_reg) {
        const std::string& reg = pair.second;
        // Пропускаем уже объявленные базовые регистры
        if (reg != "items" && reg != "flags" && reg != "items2" && !emitted.count(reg)) {
            declarations << "            int " << reg << "[ITEMS_PER_THREAD];\n";
            emitted.insert(reg);
        }
    }
    declarations << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);\n\n";

    // 5. Склеиваем всё обратно: Префикс + Заголовок + Декларации + Логика + Футер
    ctx.current_pipeline->kernel_body.str("");
    ctx.current_pipeline->kernel_body << existing_body 
                                      << header.str() 
                                      << declarations.str() 
                                      << pipeline_logic 
                                      << "        });\n";
    ctx.current_pipeline->kernel_body << "    });\n\n";
}

void JITOperatorVisitor::produceFilter(const FilterNode* node, JITContext& ctx) {
    if (!node->getChildren().empty()) {
        produce(node->getChildren()[0].get(), ctx);
    }
}

void JITOperatorVisitor::consumeFilter(const FilterNode* node, JITContext& ctx, const OperatorNode* sender, const std::vector<std::string>& active_vars) {
    auto& code = ctx.current_pipeline->kernel_body;

    if (node->predicate) {
        // We are already inside the per-item loop.
        // Emit a scalar if() guard using inline expression translation.
        JITExprVisitor expr_visitor(ctx, code, "", false, nullptr);
        std::string cond = expr_visitor.translateInlineExpr(node->predicate.get(), true);
        code << "                    if (" << cond << ") {\n";
        if (node->parent_) consume(node->parent_, ctx, node, active_vars);
        code << "                    }\n";
    } else {
        if (node->parent_) consume(node->parent_, ctx, node, active_vars);
    }
}

// ============================================================================
// loadIntoReg — helper for Vectorized Push Model to reuse registers
// ============================================================================
void JITOperatorVisitor::ensureLoaded(const std::string& col_name, JITContext& ctx) const {} // DEPRECATED

static void loadIntoReg(const std::string& col_name, const std::string& reg_name, JITContext& ctx) {
    // ОГРОМНЫЙ ФИКС: Если колонка УЖЕ является самостоятельным регистром (например, d_year, 
    // извлеченный из HashJoin), мы НЕ ДОЛЖНЫ пытаться загрузить ее из глобальной памяти фактов!
    if (ctx.col_to_reg.count(col_name) && ctx.col_to_reg[col_name] == col_name) {
        return; // Колонка уже доступна локально!
    }

    if (ctx.col_to_reg[col_name] != reg_name) {
        ctx.external_columns.insert("d_" + col_name);
        ctx.col_to_reg[col_name] = reg_name;
        auto& code = ctx.current_pipeline->kernel_body;
        code << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
             << "d_" << col_name << " + tile_offset, tid, tile_offset, " 
             << reg_name << ", num_tile_items);\n";
    }
}

// ============================================================================
// consumeVector / consumeItem — main dispatchers
// ============================================================================

void JITOperatorVisitor::consumeVector(const OperatorNode* node, JITContext& ctx,
                                       const OperatorNode* sender,
                                       const std::vector<std::string>& active_vars) {
    if (!node) return;
    switch (node->getType()) {
        case OperatorType::FILTER:
            consumeFilterVector(static_cast<const FilterNode*>(node), ctx, sender, active_vars); break;
        case OperatorType::HASH_JOIN:
            consumeHashJoinVector(static_cast<const HashJoinNode*>(node), ctx, sender, active_vars); break;
        case OperatorType::AGGREGATE:
            consumeAggregateVector(static_cast<const AggregateNode*>(node), ctx, sender, active_vars); break;
        default: break;
    }
}

void JITOperatorVisitor::consumeItem(const OperatorNode* node, JITContext& ctx,
                                     const OperatorNode* sender,
                                     const std::vector<std::string>& active_vars) {
    if (!node) return;
    switch (node->getType()) {
        case OperatorType::FILTER:
            consumeFilterItem(static_cast<const FilterNode*>(node), ctx, sender, active_vars); break;
        case OperatorType::HASH_JOIN:
            consumeHashJoinItem(static_cast<const HashJoinNode*>(node), ctx, sender, active_vars); break;
        case OperatorType::AGGREGATE:
            consumeAggregateItem(static_cast<const AggregateNode*>(node), ctx, sender, active_vars); break;
        default: break;
    }
}

// ============================================================================
// consumeFilterVector — BlockPred* via JITExprVisitor, then consumeVector
// ============================================================================
void JITOperatorVisitor::consumeFilterVector(const FilterNode* node, JITContext& ctx,
                                              const OperatorNode* /*sender*/,
                                              const std::vector<std::string>& active_vars) {
    auto& code = ctx.current_pipeline->kernel_body;
    if (node->predicate) {
        // Recursive lambda to handle AND chains and simple binary comparisons
        std::function<bool(const ExprNode*, bool&)> process_expr = [&](const ExprNode* expr, bool& first) -> bool {
            if (!expr) return true;
            if (expr->getType() == ExprType::OP_AND) {
                const auto* bin = static_cast<const BinaryExpr*>(expr);
                bool ok1 = process_expr(bin->left.get(), first);
                bool ok2 = process_expr(bin->right.get(), first);
                return ok1 && ok2;
            } else if (expr->getType() >= ExprType::OP_EQ && expr->getType() <= ExprType::OP_GTE) {
                const auto* bin = static_cast<const BinaryExpr*>(expr);
                if (bin->left->getType() == ExprType::COLUMN_REF && bin->right->getType() == ExprType::LITERAL_INT) {
                    const auto* col = static_cast<const ColumnRefExpr*>(bin->left.get());
                    const auto* lit = static_cast<const LiteralIntExpr*>(bin->right.get());
                    
                    loadIntoReg(col->column_name, "items", ctx);
                    
                    std::string pred_macro;
                    switch(expr->getType()) {
                        case ExprType::OP_EQ:  pred_macro = first ? "BlockPredEQ"  : "BlockPredAEQ"; break;
                        case ExprType::OP_NEQ: pred_macro = first ? "BlockPredNEQ" : "BlockPredANEQ"; break;
                        case ExprType::OP_LT:  pred_macro = first ? "BlockPredLT"  : "BlockPredALT"; break;
                        case ExprType::OP_LTE: pred_macro = first ? "BlockPredLTE" : "BlockPredALTE"; break;
                        case ExprType::OP_GT:  pred_macro = first ? "BlockPredGT"  : "BlockPredAGT"; break;
                        case ExprType::OP_GTE: pred_macro = first ? "BlockPredGTE" : "BlockPredAGTE"; break;
                        default: break;
                    }
                    
                    code << "            " << pred_macro << "<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                         << "tid, items, flags, " << lit->value << ", num_tile_items);\n";
                    first = false;
                    return true;
                }
            }
            return false; // Complex expression, cannot optimize
        };
        
        bool first_flag = true;
        bool fully_optimized = process_expr(node->predicate.get(), first_flag);
        
        if (!fully_optimized) {
            // Fallback for complex expressions
            code << "            // Fallback for complex expression! Type: " << (int)node->predicate->getType() << "\n";
            if (node->predicate->getType() >= ExprType::OP_EQ && node->predicate->getType() <= ExprType::OP_NEQ) {
                const auto* bin = static_cast<const BinaryExpr*>(node->predicate.get());
                code << "            // Left type: " << (int)bin->left->getType() << ", Right type: " << (int)bin->right->getType() << "\n";
            }
            bool fp = true;
            JITExprVisitor expr_vis(ctx, code, "flags", false, &fp);
            node->predicate->accept(expr_vis);
        }
    }
    if (node->parent_) consumeVector(node->parent_, ctx, node, active_vars);
}

// ============================================================================
// consumeFilterItem — scalar inline condition inside expansion loop
// ============================================================================
void JITOperatorVisitor::consumeFilterItem(const FilterNode* node, JITContext& ctx,
                                            const OperatorNode* /*sender*/,
                                            const std::vector<std::string>& active_vars) {
    auto& code = ctx.current_pipeline->kernel_body;
    if (node->predicate) {
        JITExprVisitor expr_vis(ctx, code, "", false, nullptr);
        std::string cond = expr_vis.translateInlineExpr(node->predicate.get(), true);
        code << "                        if (" << cond << ") {\n";
        if (node->parent_) consumeItem(node->parent_, ctx, node, active_vars);
        code << "                        }\n";
    } else {
        if (node->parent_) consumeItem(node->parent_, ctx, node, active_vars);
    }
}

void JITOperatorVisitor::produceAggregate(const AggregateNode* node, JITContext& ctx) {
    if (!node->getChildren().empty()) {
        produce(node->getChildren()[0].get(), ctx);
    }
}

static std::string translateMathExprPush(
        const ExprNode* expr,
        JITContext& ctx,
        bool cast_to_ull = false) {
    if (!expr) return "0";

    switch (expr->getType()) {
        case ExprType::COLUMN_REF: {
            const auto* col = static_cast<const ColumnRefExpr*>(expr);
            std::string reg = ctx.col_to_reg.count(col->column_name) ? ctx.col_to_reg[col->column_name] : col->column_name;
            std::string res = reg + "[i]";
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
            std::string left = translateMathExprPush(bin->left.get(), ctx, cast_to_ull);
            std::string right = translateMathExprPush(bin->right.get(), ctx, false);
            std::string op;
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
// generatePerfectHash — builds a minimal perfect hash expression string and
// total table size for a GROUP BY column list.
// Uses Horner's scheme:  hash = (...((col0 - min0) * card1 + (col1 - min1)) * card2 ...) % total
// ============================================================================
static std::pair<std::string, uint64_t> generatePerfectHash(
        const std::vector<std::unique_ptr<ExprNode>>& group_by,
        const Catalog& catalog) {
    std::string hash_expr;
    uint64_t total_size = 1;

    for (std::size_t i = 0; i < group_by.size(); ++i) {
        if (group_by[i]->getType() != ExprType::COLUMN_REF) continue;
        const auto* col = static_cast<const ColumnRefExpr*>(group_by[i].get());
        const std::string& col_name = col->column_name;
        const std::string  table_name = getTableName(col_name);

        uint64_t min_val = 0;
        uint64_t card    = 1;
        try {
            const auto& meta = catalog.getTableMetadata(table_name);
            if (meta.hasColumnStats(col_name)) {
                const auto& stats = meta.getColumnStats(col_name);
                min_val = (uint64_t)stats.min_value_;
                card    = stats.cardinality_;
            }
        } catch (...) {}

        std::string term = "(" + col_name + "[i] - " + std::to_string(min_val) + ")";
        if (hash_expr.empty()) {
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
        total_size = 1;
    }
    return {hash_expr, total_size};
}

// ============================================================================
// consumeAggregateVector — opens scalar reduction loop, reduce_over_group + atomic
// ============================================================================
static void extractAllColumns(const ExprNode* e, std::vector<std::string>& cols) {
    if (!e) return;
    if (e->getType() == ExprType::COLUMN_REF) {
        const auto* col = static_cast<const ColumnRefExpr*>(e);
        if (std::find(cols.begin(), cols.end(), col->column_name) == cols.end()) {
            cols.push_back(col->column_name);
        }
    } 
    else if (e->getType() >= ExprType::OP_AND) {
        const auto* bin = static_cast<const BinaryExpr*>(e);
        if (bin->left) extractAllColumns(bin->left.get(), cols);
        if (bin->right) extractAllColumns(bin->right.get(), cols);
    }
}

static void extractAllColumns(const ExprNode* e, std::set<std::string>& cols) {
    if (!e) return;
    if (e->getType() == ExprType::COLUMN_REF) {
        cols.insert(static_cast<const ColumnRefExpr*>(e)->column_name);
    } 
    else if (e->getType() >= ExprType::OP_AND) {
        const auto* bin = static_cast<const BinaryExpr*>(e);
        if (bin->left) extractAllColumns(bin->left.get(), cols);
        if (bin->right) extractAllColumns(bin->right.get(), cols);
    }
}

void JITOperatorVisitor::consumeAggregateVector(const AggregateNode* node, JITContext& ctx,
                                                 const OperatorNode* /*sender*/,
                                                 const std::vector<std::string>& /*active_vars*/) {
    auto& code = ctx.current_pipeline->kernel_body;
    const bool has_group_by = !node->group_by_exprs.empty();

    // 1. Extract and ensure all aggregation/groupby columns are loaded into reused buffers
    std::vector<std::string> agg_cols;
    for (const auto& agg : node->aggregates) {
        extractAllColumns(agg.agg_expr.get(), agg_cols);
    }
    for (const auto& g : node->group_by_exprs) {
        extractAllColumns(g.get(), agg_cols);
    }
    
    int reg_idx = 0;
    for (const auto& col : agg_cols) {
        // Пропускаем загрузку, если это payload измерения, уже лежащий в своем регистре
        if (ctx.col_to_reg.count(col) && ctx.col_to_reg[col] == col) {
            continue; 
        }
        std::string reg_name = (reg_idx == 0) ? "items" : ("items" + std::to_string(reg_idx + 1));
        loadIntoReg(col, reg_name, ctx);
        reg_idx++;
    }

    if (!has_group_by) {
        // ---- Simple aggregation: local sum + reduce_over_group ----
        ctx.tuple_size = (int)node->aggregates.size();
        ctx.result_size_expr = std::to_string(node->aggregates.size());

        for (std::size_t a = 0; a < node->aggregates.size(); ++a)
            code << "            unsigned long long sum_" << a << " = 0;\n";

        code << "            #pragma unroll\n";
        code << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n";
        code << "                if (flags[i] && (tid + BLOCK_THREADS * i < num_tile_items)) {\n";
        for (std::size_t a = 0; a < node->aggregates.size(); ++a) {
            std::string val = "1";
            if (node->aggregates[a].func_name == "SUM" && node->aggregates[a].agg_expr)
                val = translateMathExprPush(node->aggregates[a].agg_expr.get(), ctx, true);
            code << "                    sum_" << a << " += " << val << ";\n";
        }
        code << "                }\n";
        code << "            }\n";

        for (std::size_t a = 0; a < node->aggregates.size(); ++a) {
            code << "            unsigned long long agg_" << a
                 << " = sycl::reduce_over_group(it.get_group(), sum_" << a
                 << ", sycl::plus<unsigned long long>{});\n";
            code << "            if (tid == 0) {\n";
            code << "                sycl::atomic_ref<unsigned long long,"
                 << " sycl::memory_order::relaxed,"
                 << " sycl::memory_scope::device,"
                 << " sycl::access::address_space::global_space>"
                 << " at_r(d_result[" << a << "]);\n";
            code << "                at_r.fetch_add(agg_" << a << ");\n";
            code << "            }\n";
        }
    } else {
        // ---- GROUP BY aggregation: per-slot atomic ----
        auto [hash_expr, total_size] = generatePerfectHash(node->group_by_exprs, catalog_);
        int ts = (int)(node->group_by_exprs.size() + node->aggregates.size());
        ctx.tuple_size = ts;
        ctx.result_size_expr = std::to_string((uint64_t)total_size * ts);

        code << "            #pragma unroll\n";
        code << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n";
        code << "                if (flags[i] && (tid + BLOCK_THREADS * i < num_tile_items)) {\n";
        code << "                    int hash = " << hash_expr << ";\n";

        int slot = 0;
        for (const auto& g : node->group_by_exprs) {
            if (g->getType() == ExprType::COLUMN_REF) {
                const auto* col = static_cast<const ColumnRefExpr*>(g.get());
                code << "                    d_result[(unsigned long long)hash*" << ts
                     << "+" << slot << "] = (unsigned long long)" << col->column_name << "[i];\n";
            }
            ++slot;
        }
        for (const auto& agg : node->aggregates) {
            std::string val = "1";
            if (agg.func_name == "SUM" && agg.agg_expr)
                val = translateMathExprPush(agg.agg_expr.get(), ctx, true);
            code << "                    sycl::atomic_ref<unsigned long long,"
                 << " sycl::memory_order::relaxed,"
                 << " sycl::memory_scope::device,"
                 << " sycl::access::address_space::global_space>"
                 << " at_a(d_result[(unsigned long long)hash*" << ts << "+" << slot << "]);\n";
            code << "                    at_a.fetch_add(" << val << ");\n";
            ++slot;
        }
        code << "                }\n";
        code << "            }\n";
    }
}

// ============================================================================
// consumeAggregateItem — direct atomic inside MHT expansion loop
// ============================================================================
void JITOperatorVisitor::consumeAggregateItem(const AggregateNode* node, JITContext& ctx,
                                               const OperatorNode* /*sender*/,
                                               const std::vector<std::string>& /*active_vars*/) {
    auto& code = ctx.current_pipeline->kernel_body;
    const bool has_group_by = !node->group_by_exprs.empty();

    if (!has_group_by) {
        ctx.tuple_size = (int)node->aggregates.size();
        ctx.result_size_expr = std::to_string(node->aggregates.size());
        for (std::size_t a = 0; a < node->aggregates.size(); ++a) {
            std::string val = "1";
            if (node->aggregates[a].func_name == "SUM" && node->aggregates[a].agg_expr)
                val = translateMathExprPush(node->aggregates[a].agg_expr.get(), ctx, true);
            code << "                        sycl::atomic_ref<unsigned long long,"
                 << " sycl::memory_order::relaxed,"
                 << " sycl::memory_scope::device,"
                 << " sycl::access::address_space::global_space>"
                 << " at_item(d_result[" << a << "]);\n";
            code << "                        at_item.fetch_add(" << val << ");\n";
        }
    } else {
        auto [hash_expr, total_size] = generatePerfectHash(node->group_by_exprs, catalog_);
        int ts = (int)(node->group_by_exprs.size() + node->aggregates.size());
        ctx.tuple_size = ts;
        ctx.result_size_expr = std::to_string((uint64_t)total_size * ts);
        code << "                        int hash = " << hash_expr << ";\n";
        int slot = 0;
        for (const auto& g : node->group_by_exprs) {
            if (g->getType() == ExprType::COLUMN_REF) {
                const auto* col = static_cast<const ColumnRefExpr*>(g.get());
                code << "                        d_result[(unsigned long long)hash*" << ts
                     << "+" << slot << "] = (unsigned long long)" << col->column_name << "[i];\n";
            }
            ++slot;
        }
        for (const auto& agg : node->aggregates) {
            std::string val = "1";
            if (agg.func_name == "SUM" && agg.agg_expr)
                val = translateMathExprPush(agg.agg_expr.get(), ctx, true);
            code << "                        sycl::atomic_ref<unsigned long long,"
                 << " sycl::memory_order::relaxed,"
                 << " sycl::memory_scope::device,"
                 << " sycl::access::address_space::global_space>"
                 << " at_a(d_result[(unsigned long long)hash*" << ts << "+" << slot << "]);\n";
            code << "                        at_a.fetch_add(" << val << ");\n";
            ++slot;
        }
    }
}

// Legacy consume() — routes to consumeVector for backward compat
void JITOperatorVisitor::consumeAggregate(const AggregateNode* node, JITContext& ctx,
                                           const OperatorNode* sender,
                                           const std::vector<std::string>& active_vars) {
    consumeAggregateVector(node, ctx, sender, active_vars);
}

JITOperatorVisitor::BuildInfo JITOperatorVisitor::computeBuildInfo(
        const std::string& dim_table,
        const FilterNode* /*filter*/,
        const HashJoinNode* join_node) const {
    BuildInfo bi;
    bi.dim_prefix  = tablePrefix(dim_table);
    bi.size_macro  = sizeMacro(dim_table);
    bi.ht_name     = "d_" + bi.dim_prefix + "_hash_table";
    bi.variant     = 1;
    bi.use_mht     = false;
    bi.key_mins    = "0";

    // Determine HT size from catalog key stats
    try {
        const auto& meta = catalog_.getTableMetadata(dim_table);
        for (const auto& col_name : meta.getColumnNames()) {
            if (col_name.find("key") != std::string::npos ||
                col_name.find("Key") != std::string::npos) {
                if (meta.hasColumnStats(col_name)) {
                    const auto& stats = meta.getColumnStats(col_name);
                    int64_t range = stats.max_value_ - stats.min_value_ + 1;
                    bi.key_mins    = std::to_string(stats.min_value_);
                    bi.ht_size_expr = std::to_string(range);
                    break;
                }
            }
        }
    } catch (...) {}

    if (bi.ht_size_expr.empty()) bi.ht_size_expr = bi.size_macro;

    // Extract pk_col / fk_col from join condition
    if (join_node && join_node->join_condition) {
        std::function<void(const ExprNode*)> extractKeys = [&](const ExprNode* e) {
            if (!e) return;
            if (e->getType() == ExprType::OP_EQ) {
                const auto* eq = static_cast<const BinaryExpr*>(e);
                if (eq->left && eq->right && 
                    eq->left->getType() == ExprType::COLUMN_REF && 
                    eq->right->getType() == ExprType::COLUMN_REF) {
                    const auto* lc = static_cast<const ColumnRefExpr*>(eq->left.get());
                    const auto* rc = static_cast<const ColumnRefExpr*>(eq->right.get());
                    if (getTableName(lc->column_name) == dim_table) {
                        bi.pk_col = lc->column_name;
                        bi.fk_cols.push_back(rc->column_name);
                    } else {
                        bi.pk_col = rc->column_name;
                        bi.fk_cols.push_back(lc->column_name);
                    }
                }
            } else if (e->getType() == ExprType::OP_AND || e->getType() == ExprType::OP_OR) {
                const auto* bin = static_cast<const BinaryExpr*>(e);
                extractKeys(bin->left.get());
                extractKeys(bin->right.get());
            }
        };
        extractKeys(join_node->join_condition.get());
    }

    // 1. Собираем колонки, которые требуются узлам ВЫШЕ джойна
    // Разделяем на те, что нужны для агрегации (приоритет) и те, что для фильтрации.
    std::vector<std::string> agg_required;
    std::vector<std::string> filter_required;
    const OperatorNode* curr = join_node->parent_;
    while (curr) {
        if (curr->getType() == OperatorType::AGGREGATE) {
            const auto* agg = static_cast<const AggregateNode*>(curr);
            // Извлекаем колонки из всех агрегатных выражений
            for (const auto& a : agg->aggregates) {
                extractAllColumns(a.agg_expr.get(), agg_required);
            }
            // Извлекаем колонки из всех выражений группировки
            for (const auto& gb : agg->group_by_exprs) {
                extractAllColumns(gb.get(), agg_required);
            }
        } else if (curr->getType() == OperatorType::FILTER) {
            const auto* flt = static_cast<const FilterNode*>(curr);
            extractAllColumns(flt->predicate.get(), filter_required);
        }
        curr = curr->parent_;
    }

    // 2. Ищем val_col: сначала среди агрегатных, потом среди фильтровых
    bi.variant = 1;
    bi.val_col = "";

    auto findMatch = [&](const std::set<std::string>& cols) -> bool {
        for (const auto& col : cols) {
            if (getTableName(col) == dim_table && col != bi.pk_col) {
                bi.val_col = col;
                bi.variant = 2;
                return true;
            }
        }
        return false;
    };

    if (!findMatch(agg_cols_)) {
        findMatch(filter_cols_);
    }

    return bi;
}

void JITOperatorVisitor::collectAllColumnsFromTree(const OperatorNode* node) {
    if (!node) return;
    if (node->getType() == OperatorType::AGGREGATE) {
        const auto* agg = static_cast<const AggregateNode*>(node);
        for (const auto& a : agg->aggregates) {
            extractAllColumns(a.agg_expr.get(), agg_cols_);
        }
        for (const auto& gb : agg->group_by_exprs) {
            extractAllColumns(gb.get(), agg_cols_);
        }
    } else if (node->getType() == OperatorType::FILTER) {
        const auto* flt = static_cast<const FilterNode*>(node);
        extractAllColumns(flt->predicate.get(), filter_cols_);
    }
    
    for (const auto& child : node->getChildren()) {
        collectAllColumnsFromTree(child.get());
    }
}

// Emits block-level filter predicates for a build-side dimension kernel.
// `flags` starts fully initialised (all 1).  For each comparison in the
// filter predicate we load the column (if not already loaded) and emit the
// appropriate BlockPred* call.  Returns the column currently in `items`.
std::string JITOperatorVisitor::emitBuildFilter(
        const FilterNode* filter,
        const std::string& /*dim_table*/,
        std::stringstream& code,
        JITContext& ctx) const {
    if (!filter || !filter->predicate) return "";

    // Flatten AND into leaves
    std::vector<const ExprNode*> leaves;
    std::function<void(const ExprNode*)> flatten = [&](const ExprNode* e) {
        if (!e) return;
        if (e->getType() == ExprType::OP_AND) {
            const auto* b = static_cast<const BinaryExpr*>(e);
            flatten(b->left.get());
            flatten(b->right.get());
        } else {
            leaves.push_back(e);
        }
    };
    flatten(filter->predicate.get());

    std::string current_items_col;
    bool first_pred = true;

    for (const ExprNode* leaf : leaves) {
        if (!leaf || leaf->getType() < ExprType::OP_EQ) continue;
        const auto* cmp = static_cast<const BinaryExpr*>(leaf);
        if (!cmp->left || !cmp->right) continue;

        const ExprNode* col_node = nullptr;
        const ExprNode* lit_node = nullptr;
        if (cmp->left->getType()  == ExprType::COLUMN_REF &&
            (cmp->right->getType() == ExprType::LITERAL_INT ||
             cmp->right->getType() == ExprType::LITERAL_FLOAT)) {
            col_node = cmp->left.get(); lit_node = cmp->right.get();
        } else if (cmp->right->getType() == ExprType::COLUMN_REF &&
                   (cmp->left->getType()  == ExprType::LITERAL_INT ||
                    cmp->left->getType()  == ExprType::LITERAL_FLOAT)) {
            col_node = cmp->right.get(); lit_node = cmp->left.get();
        } else { continue; }

        const auto* col = static_cast<const ColumnRefExpr*>(col_node);
        std::string col_name = col->column_name;
        std::string lit_val;
        if (lit_node->getType() == ExprType::LITERAL_INT)
            lit_val = std::to_string(static_cast<const LiteralIntExpr*>(lit_node)->value);
        else
            lit_val = std::to_string(static_cast<const LiteralFloatExpr*>(lit_node)->value);

        // Load column into `items` if it changed
        if (current_items_col != col_name) {
            ctx.external_columns.insert("d_" + col_name);
            code << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                 << "d_" << col_name << " + tile_offset, tid, tile_offset, items, num_tile_items);\n";
            current_items_col = col_name;
        }

        // BlockPred / BlockPredA suffix
        static const char* suffixes[] = {"Eq","NEq","LT","LTE","GT","GTE"};
        static const ExprType ops[] = {
            ExprType::OP_EQ, ExprType::OP_NEQ,
            ExprType::OP_LT, ExprType::OP_LTE,
            ExprType::OP_GT, ExprType::OP_GTE
        };
        std::string suf = "Eq";
        for (int k = 0; k < 6; ++k) {
            if (cmp->op_type == ops[k]) { suf = suffixes[k]; break; }
        }
        std::string prefix = first_pred ? "BlockPred" : "BlockPredA";
        first_pred = false;

        code << "            " << prefix << suf
             << "<int, BLOCK_THREADS, ITEMS_PER_THREAD>"
             << "(tid, items, flags, " << lit_val << ", num_tile_items);\n";
    }
    return current_items_col;
}

// ============================================================================
// emitPHTBuildKernel — single-pass PHT build (mirrors q21.cpp reference pattern)
// ============================================================================
void JITOperatorVisitor::emitPHTBuildKernel(
        const BuildInfo& bi,
        const FilterNode* build_filter,
        JITContext& ctx) {

    std::string kname = ctx.getNewBuildKernelName(bi.dim_prefix);
    ctx.startNewPipeline("Build_" + bi.dim_prefix);
    auto& code = ctx.current_pipeline->kernel_body;

    // HT size: 2× for open-addressing headroom
    std::string ht_size = "2*" + bi.ht_size_expr;
    ctx.hash_tables.push_back({bi.ht_name, "int",
                               bi.variant == 1 ? ht_size : "2*2*" + bi.ht_size_expr});

    // Register PK column as external
    if (!bi.pk_col.empty()) ctx.external_columns.insert("d_" + bi.pk_col);
    if (!bi.val_col.empty()) ctx.external_columns.insert("d_" + bi.val_col);

    code << "    q.submit([&](sycl::handler& h) {\n";
    code << "        int num_tiles = (" << bi.size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    code << "        size_t local  = BLOCK_THREADS;\n";
    code << "        size_t global = num_tiles * BLOCK_THREADS;\n";
    code << "        h.parallel_for<class " << kname << ">"
         << "(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {\n";
    code << "            int items[ITEMS_PER_THREAD];\n";
    if (bi.variant == 2) code << "            int items2[ITEMS_PER_THREAD];\n";
    code << "            int flags[ITEMS_PER_THREAD];\n";
    code << "            int tid = it.get_local_linear_id();\n";
    code << "            int tile_offset = it.get_group_linear_id() * TILE_SIZE;\n";
    code << "            int num_tiles_local = (" << bi.size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    code << "            int num_tile_items = TILE_SIZE;\n";
    code << "            if (it.get_group_linear_id() == num_tiles_local - 1) {\n";
    code << "                num_tile_items = " << bi.size_macro << " - tile_offset;\n";
    code << "            }\n";
    code << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);\n";

    // Emit dimension filter predicates (Option A: block-level BlockPred*)
    emitBuildFilter(build_filter, bi.dim_prefix, code, ctx);

    // Load PK key column into items
    if (!bi.pk_col.empty()) {
        code << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
             << "d_" << bi.pk_col << " + tile_offset, tid, tile_offset, items, num_tile_items);\n";
    }

    if (bi.variant == 1) {
        // PHT_1: key-only
        code << "            BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>"
             << "(tid, items, flags, " << bi.ht_name << ", "
             << bi.ht_size_expr << ", " << bi.key_mins << ", num_tile_items);\n";
    } else {
        // PHT_2: key+value — load payload into items2 then build
        if (!bi.val_col.empty()) {
            code << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                 << "d_" << bi.val_col << " + tile_offset, tid, tile_offset, items2, num_tile_items);\n";
        }
        code << "            BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>"
             << "(tid, items, items2, flags, " << bi.ht_name << ", "
             << bi.ht_size_expr << ", " << bi.key_mins << ", num_tile_items);\n";
    }

    code << "        });\n    });\n\n";
}

// ============================================================================
// emitMHTBuildKernels — two-pass MHT build for 1-to-N joins (future use)
// ============================================================================
void JITOperatorVisitor::emitMHTBuildKernels(
        const BuildInfo& bi,
        const FilterNode* build_filter,
        JITContext& ctx) {

    std::string counts_name    = "d_" + bi.dim_prefix + "_counts";
    std::string offsets_name   = "d_" + bi.dim_prefix + "_offsets";
    std::string block_sums_name = "d_" + bi.dim_prefix + "_block_sums";

    if (!bi.pk_col.empty()) ctx.external_columns.insert("d_" + bi.pk_col);
    if (!bi.val_col.empty()) ctx.external_columns.insert("d_" + bi.val_col);

    ctx.hash_tables.push_back({bi.ht_name,    "int", "3*" + bi.ht_size_expr});
    ctx.hash_tables.push_back({counts_name,   "int", bi.ht_size_expr});
    ctx.hash_tables.push_back({offsets_name,  "int", bi.ht_size_expr});

    // ---- Pass 1: Count ----
    std::string kname1 = ctx.getNewBuildKernelName(bi.dim_prefix + "_cnt");
    ctx.startNewPipeline("MHT_Count_" + bi.dim_prefix);
    auto& c1 = ctx.current_pipeline->kernel_body;

    c1 << "    q.submit([&](sycl::handler& h) {\n";
    c1 << "        int num_tiles = (" << bi.size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    c1 << "        h.parallel_for<class " << kname1 << ">"
       << "(sycl::nd_range<1>(num_tiles*BLOCK_THREADS, BLOCK_THREADS), [=](sycl::nd_item<1> it) {\n";
    c1 << "            int items[ITEMS_PER_THREAD]; int flags[ITEMS_PER_THREAD];\n";
    c1 << "            int tid = it.get_local_linear_id();\n";
    c1 << "            int tile_offset = it.get_group_linear_id() * TILE_SIZE;\n";
    c1 << "            int ntiles = (" << bi.size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    c1 << "            int num_tile_items = (it.get_group_linear_id() == ntiles-1) ? "
       << bi.size_macro << " - tile_offset : TILE_SIZE;\n";
    c1 << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);\n";
    emitBuildFilter(build_filter, bi.dim_prefix, c1, ctx);
    if (!bi.pk_col.empty())
        c1 << "            BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>"
           << "(d_" << bi.pk_col << "+tile_offset,tid,tile_offset,items,num_tile_items);\n";
    c1 << "            BlockBuildMHT_Count<int,BLOCK_THREADS,ITEMS_PER_THREAD>"
       << "(tid,items,flags," << counts_name << "," << bi.ht_size_expr
       << "," << bi.key_mins << ",num_tile_items);\n";
    c1 << "        });\n    });\n\n";

    // ---- 3-Step Global Prefix Sum (using exclusive_scan_over_group) ----
    ctx.startNewPipeline("MHT_PfxSum_" + bi.dim_prefix);
    auto& cs = ctx.current_pipeline->kernel_body;

    cs << "    {\n";
    cs << "        int num_ps_blocks = (" << bi.ht_size_expr << " + 255) / 256;\n";
    cs << "        int* " << block_sums_name << " = sycl::malloc_device<int>(num_ps_blocks, q);\n";
    // Step 1: block-wise exclusive scan, save block sums
    cs << "        q.submit([&](sycl::handler& h) {\n";
    cs << "            h.parallel_for<class MHTBlockScan_" << bi.dim_prefix << ">"
       << "(sycl::nd_range<1>(num_ps_blocks*256,256),[=](sycl::nd_item<1> it){\n";
    cs << "                int gid = it.get_global_id(0);\n";
    cs << "                int val = (gid < " << bi.ht_size_expr << ") ? " << counts_name << "[gid] : 0;\n";
    cs << "                int scanned = sycl::exclusive_scan_over_group(it.get_group(), val, sycl::plus<int>{});\n";
    cs << "                if (gid < " << bi.ht_size_expr << ") " << offsets_name << "[gid] = scanned;\n";
    cs << "                if (it.get_local_id(0)==255) "
       << block_sums_name << "[it.get_group(0)] = scanned + val;\n";
    cs << "            });\n        });\n";
    // Step 2: scan block sums
    cs << "        q.submit([&](sycl::handler& h) {\n";
    cs << "            h.parallel_for<class MHTSumScan_" << bi.dim_prefix << ">"
       << "(sycl::nd_range<1>(256,256),[=](sycl::nd_item<1> it){\n";
    cs << "                int gid = it.get_global_id(0);\n";
    cs << "                int val = (gid < num_ps_blocks) ? " << block_sums_name << "[gid] : 0;\n";
    cs << "                int sc = sycl::exclusive_scan_over_group(it.get_group(), val, sycl::plus<int>{});\n";
    cs << "                if (gid < num_ps_blocks) " << block_sums_name << "[gid] = sc;\n";
    cs << "            });\n        });\n";
    // Step 3: add block sums back
    cs << "        q.submit([&](sycl::handler& h) {\n";
    cs << "            h.parallel_for<class MHTAddSums_" << bi.dim_prefix << ">"
       << "(sycl::nd_range<1>(num_ps_blocks*256,256),[=](sycl::nd_item<1> it){\n";
    cs << "                int gid = it.get_global_id(0);\n";
    cs << "                if (gid < " << bi.ht_size_expr << ") "
       << offsets_name << "[gid] += " << block_sums_name << "[it.get_group(0)];\n";
    cs << "            });\n        });\n";
    // Determine payload size, allocate on device
    cs << "        int mht_last_off=0, mht_last_cnt=0;\n";
    cs << "        q.memcpy(&mht_last_off," << offsets_name
       << "+" << bi.ht_size_expr << "-1,sizeof(int)).wait();\n";
    cs << "        q.memcpy(&mht_last_cnt," << counts_name
       << "+" << bi.ht_size_expr << "-1,sizeof(int)).wait();\n";
    cs << "        int payload_sz_" << bi.dim_prefix << " = mht_last_off + mht_last_cnt;\n";
    cs << "        if (payload_sz_" << bi.dim_prefix << " == 0) payload_sz_" << bi.dim_prefix << " = 1;\n";
    cs << "        int* payload_" << bi.dim_prefix << " = sycl::malloc_device<int>(payload_sz_"
       << bi.dim_prefix << ", q);\n";
    cs << "        q.memset(" << counts_name << ", 0, " << bi.ht_size_expr << "*sizeof(int));\n";
    cs << "        sycl::free(" << block_sums_name << ", q);\n";
    cs << "    }\n\n";

    // ---- Pass 2: Write ----
    std::string kname2 = ctx.getNewBuildKernelName(bi.dim_prefix + "_wrt");
    ctx.startNewPipeline("MHT_Write_" + bi.dim_prefix);
    auto& c2 = ctx.current_pipeline->kernel_body;

    c2 << "    q.submit([&](sycl::handler& h) {\n";
    c2 << "        int num_tiles = (" << bi.size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    c2 << "        h.parallel_for<class " << kname2 << ">"
       << "(sycl::nd_range<1>(num_tiles*BLOCK_THREADS,BLOCK_THREADS),[=](sycl::nd_item<1> it){\n";
    c2 << "            int items[ITEMS_PER_THREAD]; int items2[ITEMS_PER_THREAD];\n";
    c2 << "            int flags[ITEMS_PER_THREAD];\n";
    c2 << "            int tid = it.get_local_linear_id();\n";
    c2 << "            int tile_offset = it.get_group_linear_id() * TILE_SIZE;\n";
    c2 << "            int ntiles = (" << bi.size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    c2 << "            int num_tile_items = (it.get_group_linear_id() == ntiles-1) ? "
       << bi.size_macro << " - tile_offset : TILE_SIZE;\n";
    c2 << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);\n";
    emitBuildFilter(build_filter, bi.dim_prefix, c2, ctx);
    if (!bi.pk_col.empty())
        c2 << "            BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>"
           << "(d_" << bi.pk_col << "+tile_offset,tid,tile_offset,items,num_tile_items);\n";
    if (!bi.val_col.empty())
        c2 << "            BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>"
           << "(d_" << bi.val_col << "+tile_offset,tid,tile_offset,items2,num_tile_items);\n";
    c2 << "            BlockBuildMHT_Write<int,int,BLOCK_THREADS,ITEMS_PER_THREAD>"
       << "(tid,items,items2,flags," << bi.ht_name << "," << offsets_name << ","
       << counts_name << "," << counts_name << ",payload_" << bi.dim_prefix
       << "," << bi.ht_size_expr << "," << bi.key_mins << ",num_tile_items);\n";
    c2 << "        });\n    });\n\n";
}

void JITOperatorVisitor::produceHashJoin(const HashJoinNode* node, JITContext& ctx) {
    const auto& children = node->getChildren();
    if (children.size() < 2) return;

    auto is_dim_op = [&](const OperatorNode* n) {
        const TableScanNode* scan = nullptr;
        if (n->getType() == OperatorType::TABLE_SCAN) {
            scan = static_cast<const TableScanNode*>(n);
        } else if (n->getType() == OperatorType::FILTER && !n->getChildren().empty() && 
                   n->getChildren()[0]->getType() == OperatorType::TABLE_SCAN) {
            scan = static_cast<const TableScanNode*>(n->getChildren()[0].get());
        }
        if (scan && !scan->table_name.empty()) {
            try {
                return !catalog_.getTableMetadata(scan->table_name).isFactTable();
            } catch (...) { return false; }
        }
        return false;
    };

    const OperatorNode* build_side = nullptr;
    const OperatorNode* probe_side = nullptr;

    if (is_dim_op(children[0].get())) {
        build_side = children[0].get();
        probe_side = children[1].get();
    } else if (is_dim_op(children[1].get())) {
        build_side = children[1].get();
        probe_side = children[0].get();
    } else {
        return; // Both complex
    }

    const FilterNode*   build_filter = nullptr;
    const TableScanNode* build_scan  = nullptr;

    if (build_side->getType() == OperatorType::FILTER) {
        build_filter = static_cast<const FilterNode*>(build_side);
        build_scan = static_cast<const TableScanNode*>(build_filter->getChildren()[0].get());
    } else {
        build_scan = static_cast<const TableScanNode*>(build_side);
    }
    if (!build_scan) return;

    const std::string& dim_table = build_scan->table_name;

    // --- Compute BuildInfo ---
    BuildInfo bi = computeBuildInfo(dim_table, build_filter, node);

    // Find payload (val) column: any dim column used in GROUP BY / agg that isn't the PK
    {
        std::vector<std::string> dim_cols;
        findAllColumnsForTable(node, dim_table, dim_cols);
        for (auto& v : dim_cols) {
            if (v != bi.pk_col) { bi.val_col = v; break; }
        }
    }

    // --- Route to PHT or MHT ---
    if (!bi.use_mht) {
        bi.variant = bi.val_col.empty() ? 1 : 2;
        emitPHTBuildKernel(bi, build_filter, ctx);
    } else {
        bi.variant = 2;
        emitMHTBuildKernels(bi, build_filter, ctx);
    }

    build_infos_[node] = bi;

    // --- Trigger probe pipeline ---
    produce(probe_side, ctx);
}

// ============================================================================
// consumeHashJoinVector — vector mode probe (outside scalar loop)
// PHT (1-to-1): BlockProbeAndPHT → stay in vector mode → consumeVector
// MHT (1-to-N): open scalar expansion loop → consumeItem
// ============================================================================
void JITOperatorVisitor::consumeHashJoinVector(const HashJoinNode* node, JITContext& ctx,
                                                const OperatorNode* /*sender*/,
                                                const std::vector<std::string>& active_vars) {
    auto& code = ctx.current_pipeline->kernel_body;

    if (!build_infos_.count(node)) return;
    const BuildInfo& bi = build_infos_[node];
    if (bi.fk_cols.empty()) return;
    const std::string& fk = bi.fk_cols.front();

    if (!bi.use_mht) {
        // ---- PHT probe: stays in vector mode ----
        // Load FK column if not already loaded
        if (!ctx.loaded_in_probe.count(fk)) {
            ctx.external_columns.insert("d_" + fk);
            ctx.loaded_in_probe.insert(fk);
            code << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                 << "d_" << fk << " + tile_offset, tid, tile_offset, items, num_tile_items);\n";
        } else {
            // fk was loaded into its own register; copy to items[] for probe
            code << "            #pragma unroll\n";
            code << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) items[i] = " << fk << "[i];\n";
        }

        if (bi.variant == 1) {
            // PHT_1: key-only lookup
            code << "            BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>"
                 << "(tid, items, flags, " << bi.ht_name << ", "
                 << bi.ht_size_expr << ", " << bi.key_mins << ", num_tile_items);\n";
        } else {
            // PHT_2: key+value — результаты land in items2
            // ВНИМАНИЕ: Убрано локальное объявление int items2, так как оно теперь в заголовке ядра
            code << "            BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>"
                 << "(tid, items, items2, flags, " << bi.ht_name << ", "
                 << bi.ht_size_expr << ", " << bi.key_mins << ", num_tile_items);\n";
            // Scatter payload into the val_col register for use by aggregation
            if (!bi.val_col.empty()) {
                // РЕГИСТРАЦИЯ: Говорим TableScan, что нужно сгенерировать `int val_col[ITEMS]`;
                ctx.col_to_reg[bi.val_col] = bi.val_col; 
                
                code << "            #pragma unroll\n";
                code << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) "
                     << bi.val_col << "[i] = items2[i];\n";
            }
        }

        // Still in vector mode — pass to parent
        if (node->parent_) consumeVector(node->parent_, ctx, node, active_vars);

    } else {
        // ---- MHT probe: must expand rows → open scalar loop ----
        code << "            #pragma unroll\n";
        code << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n";
        code << "                if (flags[i] && (tid + BLOCK_THREADS * i < num_tile_items)) {\n";
        code << "                    int offset = 0, count = 0;\n";
        code << "                    ProbeMultiHT(" << fk << "[i], offset, count, flags[i], "
             << bi.ht_name << ", " << bi.ht_size_expr << ", " << bi.key_mins << ");\n";
        code << "                    for (int j = 0; j < count; ++j) {\n";

        std::vector<std::string> new_vars = active_vars;
        if (!bi.val_col.empty()) {
            // РЕГИСТРАЦИЯ: Говорим TableScan, что нужно сгенерировать `int val_col[ITEMS]`;
            ctx.col_to_reg[bi.val_col] = bi.val_col;
            code << "                        " << bi.val_col << "[i] = payload_"
                 << bi.dim_prefix << "[offset + j];\n";
            new_vars.push_back(bi.val_col);
        }

        // Drop to item mode for all parents
        if (node->parent_) consumeItem(node->parent_, ctx, node, new_vars);

        code << "                    }\n";
        code << "                }\n";
        code << "            }\n";
    }
}

// ============================================================================
// consumeHashJoinItem — PHT probe inside a scalar expansion loop (rare)
// ============================================================================
void JITOperatorVisitor::consumeHashJoinItem(const HashJoinNode* node, JITContext& ctx,
                                              const OperatorNode* /*sender*/,
                                              const std::vector<std::string>& active_vars) {
    auto& code = ctx.current_pipeline->kernel_body;
    if (!build_infos_.count(node)) return;
    const BuildInfo& bi = build_infos_[node];
    if (bi.fk_cols.empty()) return;
    const std::string& fk = bi.fk_cols.front();

    // In item mode, we already have a scalar 'i' in scope.
    if (bi.variant == 1) {
        code << "                        if (ProbePHT_1(" << fk << "[i], "
             << bi.ht_name << ", " << bi.ht_size_expr << ", " << bi.key_mins << ")) {\n";
    } else {
        code << "                        int ht_val = 0;\n";
        code << "                        if (ProbePHT_2(" << fk << "[i], ht_val, "
             << bi.ht_name << ", " << bi.ht_size_expr << ", " << bi.key_mins << ")) {\n";
        if (!bi.val_col.empty())
            code << "                            " << bi.val_col << "[i] = ht_val;\n";
    }
    std::vector<std::string> new_vars = active_vars;
    if (!bi.val_col.empty()) new_vars.push_back(bi.val_col);
    if (node->parent_) consumeItem(node->parent_, ctx, node, new_vars);
    code << "                        }\n";
}

// Legacy consume() — routes to consumeHashJoinVector
void JITOperatorVisitor::consumeHashJoin(const HashJoinNode* node, JITContext& ctx,
                                          const OperatorNode* sender,
                                          const std::vector<std::string>& active_vars) {
    consumeHashJoinVector(node, ctx, sender, active_vars);
}


// ============================================================================
// generateCode
// ============================================================================

std::string JITOperatorVisitor::generateCode() const {
    std::stringstream code;

    code << "#include \"core/execution.h\"\n";
    code << "#include <sycl/sycl.hpp>\n";
    code << "#include \"crystal/load.h\"\n";
    code << "#include \"crystal/pred.h\"\n";
    code << "#include \"crystal/join.h\"\n";
    code << "#include \"crystal/utils.h\"\n";
    code << "#include \"core/inline_math.h\"\n\n";
    code << "using namespace sycl;\n\n";
    code << "constexpr int BLOCK_THREADS = 128;\n";
    code << "constexpr int ITEMS_PER_THREAD = 4;\n";
    code << "constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;\n\n";

    // Forward-declare all kernel class names
    for (const auto& p : ctx_.pipelines) {
        code << "class " << p.kernel_name << ";\n";
    }
    for (const auto& kname : ctx_.kernel_class_names) {
        code << "class " << kname << ";\n";
    }

    code << "\nextern \"C\" void execute_query(db::ExecutionContext* ctx) {\n";
    code << "    sycl::queue& q = *(ctx->q_);\n";

    for (const auto& col : ctx_.external_columns) {
        code << "    int* " << col << " = ctx->getBuffer<int>(\"" << col << "\");\n";
    }

    code << "    unsigned long long* d_result = ctx->getResultPointer();\n";

    for (const auto& ht : ctx_.hash_tables) {
        code << "    " << ht.type << "* " << ht.name
             << " = sycl::malloc_device<" << ht.type << ">("
             << ht.size_expr << ", q);\n";
        code << "    q.memset(" << ht.name << ", 0, ("
             << ht.size_expr << ") * sizeof(" << ht.type << "));\n";
    }

    code << "    q.memset(d_result, 0, " << ctx_.result_size_expr
         << " * sizeof(unsigned long long));\n\n";

    for (const auto& p : ctx_.pipelines) {
        code << p.includes_and_globals.str();
        code << p.kernel_body.str();
    }

    code << "    q.wait();\n\n";
    for (const auto& ht : ctx_.hash_tables) {
        code << "    sycl::free(" << ht.name << ", q);\n";
    }

    // Free MHT-only allocations (payload chunks)
    for (const auto& pair : build_infos_) {
        const auto& bi = pair.second;
        if (bi.use_mht) {
            code << "    sycl::free(payload_" << bi.dim_prefix << ", q);\n";
        }
    }

    code << "    ctx->tuple_size_ = " << ctx_.tuple_size << ";\n";
    code << "}\n";

    return code.str();
}

} // namespace db
