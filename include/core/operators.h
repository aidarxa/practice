#pragma once

#include "expressions.h"
#include "memory.h"
#include "../crystal/utils.h"

#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <functional>

namespace db {


inline void collectCaseIntegerOutputs(const ExprNode* expr,
                                      std::vector<int64_t>& values,
                                      bool& may_be_null) {
    if (!expr) {
        may_be_null = true;
        return;
    }
    if (expr->getType() == ExprType::LITERAL_INT) {
        values.push_back(static_cast<const LiteralIntExpr*>(expr)->value);
        return;
    }
    if (expr->getType() == ExprType::LITERAL_NULL) {
        may_be_null = true;
        return;
    }
    if (expr->getType() == ExprType::CASE_WHEN) {
        const auto* c = static_cast<const CaseWhenExpr*>(expr);
        collectCaseIntegerOutputs(c->then_expr.get(), values, may_be_null);
        collectCaseIntegerOutputs(c->else_expr.get(), values, may_be_null);
        return;
    }
    throw std::runtime_error("GROUP BY CASE currently requires integer THEN/ELSE result expressions");
}

inline uint64_t integerCaseCardinality(const ExprNode* expr, bool include_null_bucket) {
    std::vector<int64_t> values;
    bool may_be_null = false;
    collectCaseIntegerOutputs(expr, values, may_be_null);
    if (values.empty()) return may_be_null ? 1ULL : 0ULL;
    const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    const uint64_t range = static_cast<uint64_t>(*max_it - *min_it + 1);
    return range + ((include_null_bucket && may_be_null) ? 1ULL : 0ULL);
}

// ============================================================================
// 1. OperatorType
// ============================================================================
enum class OperatorType {
    TABLE_SCAN,
    FILTER,
    HASH_JOIN,
    AGGREGATE,
    PROJECTION,
    SORT_LIMIT,
};

// ============================================================================
// 2. Forward declarations
// ============================================================================
class OperatorVisitor;
class TableScanNode;
class FilterNode;
class HashJoinNode;
class AggregateNode;
class ProjectionNode;
class SortLimitNode;

// ============================================================================
// 3. Абстрактный базовый класс узла оператора (реляционная алгебра)
//    Дерево — это DAG; владение дочерними узлами через unique_ptr.
// ============================================================================
class OperatorNode {
protected:
    std::vector<std::unique_ptr<OperatorNode>> children_;

public:
    // Non-owning back-pointer to parent in the operator tree.
    // Set automatically by addChild(). Used by push-model consume() to climb.
    OperatorNode* parent_ = nullptr;

    virtual ~OperatorNode() = default;

    virtual OperatorType getType() const = 0;
    virtual void accept(OperatorVisitor& visitor) const = 0;

    void addChild(std::unique_ptr<OperatorNode> child) {
        child->parent_ = this;   // wire the back-pointer before moving ownership
        children_.push_back(std::move(child));
    }

    const std::vector<std::unique_ptr<OperatorNode>>& getChildren() const {
        return children_;
    }

    std::vector<std::unique_ptr<OperatorNode>>& getChildrenMutable() {
        return children_;
    }
};

// ============================================================================
// 4. Конкретные узлы операторного дерева
// ============================================================================

// Сканирование таблицы — листовой узел
class TableScanNode final : public OperatorNode {
public:
    std::string table_name;
    std::string table_alias;

    explicit TableScanNode(std::string name, std::string alias = "")
        : table_name(std::move(name)), table_alias(std::move(alias)) {}

    OperatorType getType() const override { return OperatorType::TABLE_SCAN; }
    void accept(OperatorVisitor& visitor) const override;
};

// Фильтрация строк по предикату
// Ровно 1 дочерний узел (вход)
class FilterNode final : public OperatorNode {
public:
    std::unique_ptr<ExprNode> predicate; // дерево выражения условия WHERE

    explicit FilterNode(std::unique_ptr<ExprNode> pred)
        : predicate(std::move(pred)) {}

    OperatorType getType() const override { return OperatorType::FILTER; }
    void accept(OperatorVisitor& visitor) const override;
};

// Хеш-джойн двух потоков
// children_[0] — build-side (таблица измерений)
// children_[1] — probe-side (таблица фактов или следующий джойн)
// join_condition может быть nullptr в наивной фазе (заполняется Оптимизатором)
class HashJoinNode final : public OperatorNode {
public:
    std::unique_ptr<ExprNode> join_condition; // nullptr в наивной фазе

    explicit HashJoinNode(std::unique_ptr<ExprNode> cond = nullptr)
        : join_condition(std::move(cond)) {}

    OperatorType getType() const override { return OperatorType::HASH_JOIN; }
    void accept(OperatorVisitor& visitor) const override;
};


// Projection for non-aggregate SELECT lists.
// SELECT * is represented as a StarExpr inside select_exprs.
class ProjectionNode final : public OperatorNode {
public:
    std::vector<std::unique_ptr<ExprNode>> select_exprs;
    std::vector<std::string> output_aliases;

    ProjectionNode() = default;

    OperatorType getType() const override { return OperatorType::PROJECTION; }
    void accept(OperatorVisitor& visitor) const override;

    uint64_t calculateResultSize(const Catalog& catalog) const {
        uint64_t row_count = 1;
        if (!children_.empty()) {
            std::vector<const TableScanNode*> scans;
            std::function<void(const OperatorNode*)> collect = [&](const OperatorNode* n) {
                if (!n) return;
                if (n->getType() == OperatorType::TABLE_SCAN) {
                    scans.push_back(static_cast<const TableScanNode*>(n));
                }
                for (const auto& child : n->getChildren()) collect(child.get());
            };
            collect(children_[0].get());
            for (const auto* scan : scans) {
                const auto& meta = catalog.getTableMetadata(scan->table_name);
                if (row_count == 1 || meta.isFactTable()) row_count = meta.getSize();
            }
        }

        uint64_t tuple_size = 0;
        for (const auto& expr : select_exprs) {
            if (expr && expr->getType() == ExprType::STAR) {
                if (children_.empty()) continue;
                std::vector<const TableScanNode*> scans;
                std::function<void(const OperatorNode*)> collect = [&](const OperatorNode* n) {
                    if (!n) return;
                    if (n->getType() == OperatorType::TABLE_SCAN) {
                        scans.push_back(static_cast<const TableScanNode*>(n));
                    }
                    for (const auto& child : n->getChildren()) collect(child.get());
                };
                collect(children_[0].get());
                for (const auto* scan : scans) {
                    tuple_size += catalog.getTableMetadata(scan->table_name).getColumnCount();
                }
            } else {
                ++tuple_size;
            }
        }
        if (tuple_size == 0) tuple_size = 1;
        return row_count * tuple_size;
    }
};

struct SortKeyDef {
    std::size_t column_index = 0;
    bool descending = false;
};

// ORDER BY / LIMIT over the final visible result.
// The child must produce a dense, columnar result. The JIT visitor enforces this
// for aggregate children by disabling the sparse fast-output path when needed.
class SortLimitNode final : public OperatorNode {
public:
    std::vector<SortKeyDef> sort_keys;
    bool has_limit = false;
    std::size_t limit = 0;

    SortLimitNode() = default;

    OperatorType getType() const override { return OperatorType::SORT_LIMIT; }
    void accept(OperatorVisitor& visitor) const override;
};

// ============================================================================
// 5. AggregateDef — описание одной агрегатной функции
// ============================================================================
struct AggregateDef {
    std::string func_name;              // "COUNT", "SUM", "MIN", "MAX", "AVG"
    std::unique_ptr<ExprNode> agg_expr; // StarExpr only for COUNT(*); nullptr is invalid after translation

    AggregateDef(std::string name, std::unique_ptr<ExprNode> expr)
        : func_name(std::move(name)), agg_expr(std::move(expr)) {
        std::transform(func_name.begin(), func_name.end(), func_name.begin(), ::toupper);
        if (!isSupportedFunction()) {
            throw std::runtime_error("Unsupported aggregate function: " + func_name);
        }
        if (!agg_expr) {
            throw std::runtime_error("Aggregate function " + func_name + " requires an argument; use COUNT(*) for row count");
        }
        if (hasStarArgument() && !isCount()) {
            throw std::runtime_error("Aggregate function " + func_name + " does not accept '*' according to SQL semantics");
        }
    }

    bool isSupportedFunction() const {
        return func_name == "COUNT" || func_name == "SUM" || func_name == "MIN" ||
               func_name == "MAX" || func_name == "AVG";
    }

    bool isCount() const { return func_name == "COUNT"; }
    bool isSum()   const { return func_name == "SUM"; }
    bool isMin()   const { return func_name == "MIN"; }
    bool isMax()   const { return func_name == "MAX"; }
    bool isAvg()   const { return func_name == "AVG"; }

    bool hasNoArgument() const { return agg_expr == nullptr; }
    bool hasStarArgument() const {
        return agg_expr && agg_expr->getType() == ExprType::STAR;
    }
    bool argumentIsRowCountLike() const {
        return isCount() && hasStarArgument();
    }

    // Logical output: every aggregate contributes one visible value.
    uint64_t visibleSlotCount() const { return 1; }

    // Physical storage: every aggregate contributes one visible state slot.
    // Aggregates that may return NULL on an all-NULL input get a separate
    // hidden non-null counter at AggregateNode level. COUNT never needs one.
    uint64_t storageSlotCount() const { return 1; }

    // Only AVG needs a persistent hidden count state to compute the final
    // denominator. SUM/MIN/MAX validity is represented by the result validity
    // bitmap, which is set when at least one valid input updates the state.
    bool needsNonNullCount() const { return isAvg(); }

    // Explicit move-only: нет копирования из-за unique_ptr
    AggregateDef(const AggregateDef&) = delete;
    AggregateDef& operator=(const AggregateDef&) = delete;
    AggregateDef(AggregateDef&&) = default;
    AggregateDef& operator=(AggregateDef&&) = default;
};

// Группировка и агрегация
// Ровно 1 дочерний узел (вход)
class AggregateNode final : public OperatorNode {
public:
    std::vector<std::unique_ptr<ExprNode>> group_by_exprs; // выражения GROUP BY
    std::vector<AggregateDef>              aggregates;     // список агрегаций
    std::unique_ptr<ExprNode>              having_predicate; // normalized post-aggregate predicate over visible slots
    std::vector<std::string>               output_aliases;   // visible output aliases, group cols first, aggregates after

    AggregateNode() = default;

    OperatorType getType() const override { return OperatorType::AGGREGATE; }
    void accept(OperatorVisitor& visitor) const override;

    uint64_t visibleTupleSize() const {
        return static_cast<uint64_t>(group_by_exprs.size() + aggregates.size());
    }

    uint64_t hiddenCountSlotCount() const {
        uint64_t count = 0;
        for (const auto& agg : aggregates) {
            if (agg.needsNonNullCount()) ++count;
        }
        return count;
    }

    bool needsHiddenCountSlot() const {
        return hiddenCountSlotCount() != 0;
    }

    uint64_t storageTupleSize() const {
        uint64_t slots = static_cast<uint64_t>(group_by_exprs.size());
        for (const auto& agg : aggregates) slots += agg.storageSlotCount();
        slots += hiddenCountSlotCount();
        return slots == 0 ? 1 : slots;
    }

    // Вычисляет точный размер буфера результатов (в элементах unsigned long long).
    // Для групповой агрегации: cardinality(group1) * cardinality(group2) * ... * tuple_size
    // Для скалярной агрегации: 1
    // Фолбэк (нет статистики): table.getSize() для данной колонки
    uint64_t calculateResultSize(const Catalog& catalog) const {
        if (group_by_exprs.empty()) {
            return storageTupleSize();
        }

        uint64_t total_groups = 1;
        for (const auto& g : group_by_exprs) {
            uint64_t cardinality = 1;
            if (g->getType() == ExprType::COLUMN_REF) {
                const auto* col = static_cast<const ColumnRefExpr*>(g.get());
                const std::string& col_name = col->column_name;
                // Определяем таблицу по первому символу имени колонки (как в crystal/utils.h)
                std::string table_name = getTableName(col_name);

                try {
                    const auto& meta = catalog.getTableMetadata(table_name);
                    if (meta.hasColumnStats(col_name)) {
                        cardinality = meta.getColumnStats(col_name).cardinality_;
                    } else {
                        // Фолбэк: используем размер таблицы как верхнюю оценку
                        cardinality = meta.getSize();
                    }
                    if (meta.isColumnNullable(col_name)) {
                        // One extra GROUP BY bucket for SQL NULL.
                        ++cardinality;
                    }
                } catch (...) {
                    // Таблица не найдена — минимальный фолбэк
                    cardinality = 1;
                }
            } else if (g->getType() == ExprType::CASE_WHEN) {
                cardinality = integerCaseCardinality(g.get(), true);
                if (cardinality == 0) cardinality = 1;
            }
            total_groups *= cardinality;
        }

        return total_groups * storageTupleSize();
    }
};

// ============================================================================
// 6. Visitor — абстрактный обходчик операторного дерева
// ============================================================================
class OperatorVisitor {
public:
    virtual ~OperatorVisitor() = default;
    virtual void visit(const TableScanNode& node)  = 0;
    virtual void visit(const FilterNode& node)     = 0;
    virtual void visit(const HashJoinNode& node)   = 0;
    virtual void visit(const AggregateNode& node)  = 0;
    virtual void visit(const ProjectionNode& node) = 0;
    virtual void visit(const SortLimitNode& node)  = 0;
};

// ============================================================================
// 7. OperatorPrinter — отладочный printer (реализация OperatorVisitor)
//    Выводит дерево в std::ostream с отступами.
//    Рекурсивно обходит детей.
// ============================================================================
class OperatorPrinter final : public OperatorVisitor {
public:
    explicit OperatorPrinter(std::ostream& out, int indent = 0)
        : out_(out), indent_(indent) {}

    void visit(const TableScanNode& node)  override;
    void visit(const FilterNode& node)     override;
    void visit(const HashJoinNode& node)   override;
    void visit(const AggregateNode& node)  override;
    void visit(const ProjectionNode& node) override;
    void visit(const SortLimitNode& node)  override;

private:
    std::ostream& out_;
    int indent_;

    // Напечатать отступ и стрелку для дерева
    void printIndent() const;
    // Рекурсивно вызвать visitor для дочерних узлов с увеличенным indent
    void visitChildren(const OperatorNode& node);
};

} // namespace db
