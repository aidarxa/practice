#pragma once

#include "expressions.h"
#include "memory.h"
#include "../crystal/utils.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace db {

// ============================================================================
// 1. OperatorType
// ============================================================================
enum class OperatorType {
    TABLE_SCAN,
    FILTER,
    HASH_JOIN,
    AGGREGATE,
};

// ============================================================================
// 2. Forward declarations
// ============================================================================
class OperatorVisitor;
class TableScanNode;
class FilterNode;
class HashJoinNode;
class AggregateNode;

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

    explicit TableScanNode(std::string name) : table_name(std::move(name)) {}

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

// ============================================================================
// 5. AggregateDef — описание одной агрегатной функции
// ============================================================================
struct AggregateDef {
    std::string func_name;              // "SUM", "COUNT", "AVG"
    std::unique_ptr<ExprNode> agg_expr; // выражение аргумента (например, lo_revenue - lo_supplycost)

    AggregateDef(std::string name, std::unique_ptr<ExprNode> expr)
        : func_name(std::move(name)), agg_expr(std::move(expr)) {}

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

    AggregateNode() = default;

    OperatorType getType() const override { return OperatorType::AGGREGATE; }
    void accept(OperatorVisitor& visitor) const override;

    // Вычисляет размер буфера результатов в элементах unsigned long long.
    // Для GROUP BY: product(cardinality(group_key_i)) * tuple_size.
    // Для скалярной агрегации: количество агрегатов, но не меньше 1.
    uint64_t calculateResultSize(const Catalog& catalog) const {
        const uint64_t tuple_size = static_cast<uint64_t>(group_by_exprs.size() + aggregates.size());
        if (group_by_exprs.empty()) {
            return tuple_size == 0 ? 1 : tuple_size;
        }

        uint64_t total_groups = 1;
        for (const auto& g : group_by_exprs) {
            uint64_t cardinality = 1;

            if (g && g->getType() == ExprType::COLUMN_REF) {
                const auto* col = static_cast<const ColumnRefExpr*>(g.get());
                const std::string& col_name = col->column_name;
                const std::string table_name = col->table_name.empty()
                    ? getTableName(col_name)
                    : col->table_name;

                try {
                    const auto& meta = catalog.getTableMetadata(table_name);
                    if (meta.hasColumnStats(col_name)) {
                        cardinality = meta.getColumnStats(col_name).cardinality_;
                    }
                    if (cardinality == 0) {
                        cardinality = meta.getSize();
                    }
                } catch (...) {
                    cardinality = 1;
                }
            }

            if (cardinality != 0 &&
                total_groups > std::numeric_limits<uint64_t>::max() / cardinality) {
                throw std::overflow_error("Aggregate result cardinality overflows uint64_t");
            }
            total_groups *= cardinality;
        }

        if (tuple_size != 0 &&
            total_groups > std::numeric_limits<uint64_t>::max() / tuple_size) {
            throw std::overflow_error("Aggregate result buffer size overflows uint64_t");
        }
        return total_groups * tuple_size;
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

private:
    std::ostream& out_;
    int indent_;

    // Напечатать отступ и стрелку для дерева
    void printIndent() const;
    // Рекурсивно вызвать visitor для дочерних узлов с увеличенным indent
    void visitChildren(const OperatorNode& node);
};

} // namespace db
