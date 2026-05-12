#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace db {

// ============================================================================
// 1. ExprType — полный набор типов узлов выражения
// ============================================================================
enum class ExprType {
    COLUMN_REF,    // ссылка на колонку (table.column)
    LITERAL_INT,   // целочисленный литерал
    LITERAL_FLOAT, // вещественный литерал
    OP_AND,        // логическое И
    OP_OR,         // логическое ИЛИ
    OP_EQ,         // =
    OP_NEQ,        // <>
    OP_LT,         // <
    OP_LTE,        // <=
    OP_GT,         // >
    OP_GTE,        // >=
    OP_ADD,        // +
    OP_SUB,        // -
    OP_MUL,        // *
    OP_DIV,        // /
    OP_IS_NULL,    // IS NULL
    OP_IS_NOT_NULL,// IS NOT NULL
    STAR,          // * in SELECT * or aggregate arguments
};

// ============================================================================
// 2. Forward declarations
// ============================================================================
class ExprVisitor;
class ColumnRefExpr;
class LiteralIntExpr;
class LiteralFloatExpr;
class BinaryExpr;
class StarExpr;

// ============================================================================
// 3. Абстрактный базовый класс узла выражения
// ============================================================================
class ExprNode {
public:
    virtual ~ExprNode() = default;
    virtual ExprType getType() const = 0;
    virtual void accept(ExprVisitor& visitor) const = 0;
    // Клонирование необходимо для Оптимизатора (копирование поддеревьев)
    virtual std::unique_ptr<ExprNode> clone() const = 0;
};

// ============================================================================
// 4. Конкретные узлы
// ============================================================================

// Ссылка на колонку: table_name.column_name
// table_name может быть пустым, если таблица не квалифицирована
class ColumnRefExpr final : public ExprNode {
public:
    std::string table_name;   // может быть пустым
    std::string column_name;  // всегда заполнено

    explicit ColumnRefExpr(std::string column, std::string table = "")
        : table_name(std::move(table)), column_name(std::move(column)) {}

    ExprType getType() const override { return ExprType::COLUMN_REF; }

    void accept(ExprVisitor& visitor) const override;

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<ColumnRefExpr>(column_name, table_name);
    }
};

// Целочисленный литерал
class LiteralIntExpr final : public ExprNode {
public:
    int64_t value;

    explicit LiteralIntExpr(int64_t v) : value(v) {}

    ExprType getType() const override { return ExprType::LITERAL_INT; }

    void accept(ExprVisitor& visitor) const override;

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<LiteralIntExpr>(value);
    }
};

// Вещественный литерал (пока не генерируется Translator-ом, но нужен для полноты)
class LiteralFloatExpr final : public ExprNode {
public:
    double value;

    explicit LiteralFloatExpr(double v) : value(v) {}

    ExprType getType() const override { return ExprType::LITERAL_FLOAT; }

    void accept(ExprVisitor& visitor) const override;

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<LiteralFloatExpr>(value);
    }
};

// Бинарное выражение: left op right
// Покрывает арифметику, сравнения, логические операторы
class BinaryExpr final : public ExprNode {
public:
    ExprType op_type;
    std::unique_ptr<ExprNode> left;
    std::unique_ptr<ExprNode> right;

    BinaryExpr(ExprType op,
               std::unique_ptr<ExprNode> l,
               std::unique_ptr<ExprNode> r)
        : op_type(op), left(std::move(l)), right(std::move(r)) {}

    ExprType getType() const override { return op_type; }

    void accept(ExprVisitor& visitor) const override;

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<BinaryExpr>(
            op_type,
            left  ? left->clone()  : nullptr,
            right ? right->clone() : nullptr
        );
    }
};


// SQL star: SELECT * and aggregate arguments such as COUNT(*)
class StarExpr final : public ExprNode {
public:
    StarExpr() = default;

    ExprType getType() const override { return ExprType::STAR; }

    void accept(ExprVisitor& visitor) const override;

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<StarExpr>();
    }
};

// ============================================================================
// 5. Visitor — абстрактный обходчик дерева выражений
// ============================================================================
class ExprVisitor {
public:
    virtual ~ExprVisitor() = default;
    virtual void visit(const ColumnRefExpr& node)   = 0;
    virtual void visit(const LiteralIntExpr& node)  = 0;
    virtual void visit(const LiteralFloatExpr& node)= 0;
    virtual void visit(const BinaryExpr& node)      = 0;
    virtual void visit(const StarExpr& node)        = 0;
};

// ============================================================================
// 6. ExprPrinter — отладочный printer (реализация Visitor)
//    Выводит дерево в человекочитаемом виде в std::ostream
// ============================================================================
class ExprPrinter final : public ExprVisitor {
public:
    explicit ExprPrinter(std::ostream& out, int indent = 0)
        : out_(out), indent_(indent) {}

    void visit(const ColumnRefExpr& node) override;
    void visit(const LiteralIntExpr& node) override;
    void visit(const LiteralFloatExpr& node) override;
    void visit(const BinaryExpr& node) override;
    void visit(const StarExpr& node) override;

private:
    std::ostream& out_;
    int indent_;

    void printIndent() const;
    static const char* exprTypeName(ExprType t);
};

} // namespace db
