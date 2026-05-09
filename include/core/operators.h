#pragma once

#include "expressions.h"
#include "memory.h"
#include "../crystal/utils.h"

#include <memory>
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
    virtual ~OperatorNode() = default;

    virtual OperatorType getType() const = 0;
    virtual void accept(OperatorVisitor& visitor) const = 0;

    void addChild(std::unique_ptr<OperatorNode> child) {
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

    // Вычисляет точный размер буфера результатов (в элементах unsigned long long).
    // Для групповой агрегации: cardinality(group1) * cardinality(group2) * ... * tuple_size
    // Для скалярной агрегации: 1
    // Фолбэк (нет статистики): table.getSize() для данной колонки
    uint64_t calculateResultSize(const Catalog& catalog) const {
        if (group_by_exprs.empty()) {
            // Скалярная агрегация: один элемент ULL
            return 1;
        }

        uint64_t total_groups = 1;
        for (const auto& g : group_by_exprs) {
            if (g->getType() != ExprType::COLUMN_REF) continue;
            const auto* col = static_cast<const ColumnRefExpr*>(g.get());
            const std::string& col_name = col->column_name;

            // Определяем таблицу по первому символу имени колонки (как в crystal/utils.h)
            std::string table_name = getTableName(col_name);

            uint64_t cardinality = 1;
            try {
                const auto& meta = catalog.getTableMetadata(table_name);
                if (meta.hasColumnStats(col_name)) {
                    cardinality = meta.getColumnStats(col_name).cardinality_;
                } else {
                    // Фолбэк: используем размер таблицы как верхнюю оценку
                    cardinality = meta.getSize();
                }
            } catch (...) {
                // Таблица не найдена — минимальный фолбэк
                cardinality = 1;
            }
            total_groups *= cardinality;
        }

        uint64_t tuple_size = group_by_exprs.size() + aggregates.size();
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
