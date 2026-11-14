#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include "core/parameter_value.h"

namespace db {

// ============================================================================
// 1. ExprType — полный набор типов узлов выражения
// ============================================================================
enum class ExprType {
    COLUMN_REF,    // ссылка на колонку (table.column)
    LITERAL_INT,   // целочисленный литерал
    LITERAL_FLOAT, // вещественный литерал
    LITERAL_NULL,  // SQL NULL literal
    OP_AND,        // логическое И
    OP_OR,         // логическое ИЛИ
    OP_NOT,        // логическое НЕ
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
    CASE_WHEN,     // searched CASE WHEN condition THEN value ELSE value END
    STAR,          // * in SELECT * or aggregate arguments
    PARAMETER_REF, ///< Request-local slot; its type and value live outside the expression.
    LITERAL_TEXT, ///< Validated during extraction; bytes never specialize a module.
};

// ============================================================================
// 2. Forward declarations
// ============================================================================
class ExprVisitor;
class ColumnRefExpr;
class LiteralIntExpr;
class LiteralFloatExpr;
class LiteralNullExpr;
class LiteralTextExpr;
class ParameterRefExpr;
class BinaryExpr;
class StarExpr;
class CaseWhenExpr;

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
    ParameterType value_type = ParameterType::Int64; ///< Resolved catalog type, independent of current data.

    explicit ColumnRefExpr(std::string column, std::string table = "")
        : table_name(std::move(table)), column_name(std::move(column)) {}

    ExprType getType() const override { return ExprType::COLUMN_REF; }

    void accept(ExprVisitor& visitor) const override;

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<ColumnRefExpr>(*this);
    }
};

/**
 * @brief Literal provenance before parameter extraction.
 * @details An origin identifies one AST occurrence, never a value. Copies made
 * while normalizing simple CASE retain it; unrelated equal literals do not share
 * it. Origins are local to a translation and are absent from template keys.
 */
class LiteralExprNode : public ExprNode {
public:
    std::optional<std::size_t> origin;

protected:
    explicit LiteralExprNode(std::optional<std::size_t> source = {}) : origin(source) {}
};

// Целочисленный литерал
class LiteralIntExpr final : public LiteralExprNode {
public:
    int64_t value;

    explicit LiteralIntExpr(int64_t v, std::optional<std::size_t> source = {})
        : LiteralExprNode(source), value(v) {}

    ExprType getType() const override { return ExprType::LITERAL_INT; }

    void accept(ExprVisitor& visitor) const override;

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<LiteralIntExpr>(*this);
    }
};

// Вещественный литерал (пока не генерируется Translator-ом, но нужен для полноты)
class LiteralFloatExpr final : public LiteralExprNode {
public:
    double value;

    explicit LiteralFloatExpr(double v, std::optional<std::size_t> source = {})
        : LiteralExprNode(source), value(v) {}

    ExprType getType() const override { return ExprType::LITERAL_FLOAT; }

    void accept(ExprVisitor& visitor) const override;

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<LiteralFloatExpr>(*this);
    }
};


// SQL NULL literal. Value codegen emits 0, validity codegen emits false.
class LiteralTextExpr final : public LiteralExprNode {
public:
    std::string value;
    explicit LiteralTextExpr(std::string text, std::optional<std::size_t> source = {})
        : LiteralExprNode(source), value(std::move(text)) {}
    ExprType getType() const override { return ExprType::LITERAL_TEXT; }
    void accept(ExprVisitor& visitor) const override;
    std::unique_ptr<ExprNode> clone() const override { return std::make_unique<LiteralTextExpr>(*this); }
};

class LiteralNullExpr final : public LiteralExprNode {
public:
    explicit LiteralNullExpr(std::optional<std::size_t> source = {}) : LiteralExprNode(source) {}

    ExprType getType() const override { return ExprType::LITERAL_NULL; }

    void accept(ExprVisitor& visitor) const override;

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<LiteralNullExpr>(*this);
    }
};

/** @brief A stable parameter identity preserved by expression cloning and pushdown. */
class ParameterRefExpr final : public ExprNode {
public:
    const std::size_t slot;

    explicit ParameterRefExpr(std::size_t index) : slot(index) {}
    ExprType getType() const override { return ExprType::PARAMETER_REF; }
    void accept(ExprVisitor& visitor) const override;
    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<ParameterRefExpr>(slot);
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



// Searched CASE expression with one WHEN arm.
// Multiple SQL WHEN arms are represented as nested CaseWhenExpr nodes.
// Simple CASE is normalized by the translator into searched comparisons.
class CaseWhenExpr final : public ExprNode {
public:
    std::unique_ptr<ExprNode> condition;
    std::unique_ptr<ExprNode> then_expr;
    std::unique_ptr<ExprNode> else_expr;

    CaseWhenExpr(std::unique_ptr<ExprNode> cond,
                 std::unique_ptr<ExprNode> then_value,
                 std::unique_ptr<ExprNode> else_value)
        : condition(std::move(cond)),
          then_expr(std::move(then_value)),
          else_expr(std::move(else_value)) {
        if (!condition || !then_expr || !else_expr) {
            throw std::runtime_error("CASE WHEN requires condition, THEN expression, and ELSE expression");
        }
    }

    ExprType getType() const override { return ExprType::CASE_WHEN; }

    void accept(ExprVisitor& visitor) const override;

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<CaseWhenExpr>(
            condition ? condition->clone() : nullptr,
            then_expr ? then_expr->clone() : nullptr,
            else_expr ? else_expr->clone() : nullptr);
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
    virtual void visit(const LiteralNullExpr& node) = 0;
    virtual void visit(const LiteralTextExpr&) { throw std::runtime_error("Unextracted Text literal"); }
    virtual void visit(const ParameterRefExpr& node) = 0;
    virtual void visit(const BinaryExpr& node)      = 0;
    virtual void visit(const CaseWhenExpr& node)    = 0;
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
    void visit(const LiteralNullExpr& node) override;
    void visit(const LiteralTextExpr& node) override;
    void visit(const ParameterRefExpr& node) override;
    void visit(const BinaryExpr& node) override;
    void visit(const CaseWhenExpr& node) override;
    void visit(const StarExpr& node) override;

private:
    std::ostream& out_;
    int indent_;

    void printIndent() const;
    static const char* exprTypeName(ExprType t);
};

} // namespace db
