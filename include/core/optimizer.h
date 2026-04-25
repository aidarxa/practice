#pragma once
#include <memory>
#include <stdexcept>
#include <vector>
#include "../deps/include/SQLParser.h"
#include "core/memory.h"
#include "crystal/utils.h"
#include "sql/Expr.h"
#include "sql/Table.h"
#include "utils.h"

namespace db{
//TODO
class IAgregation;
class IPhysicalOperator {
  virtual const std::string &code() = 0;
  virtual ~IPhysicalOperator() = default;
protected:
  hsql::TableRef* table_;
};

enum class ExpressionType {
  NONE,
  JOIN,
  FILTER
};

struct Expression {
  ExpressionType type{ExpressionType::NONE};
  hsql::Expr *expr;
};
class LogicalPlan {
public:
  const std::vector<Expression>& getConditions()const{return conditions_;}

private:
  std::vector<hsql::Expr *> select_list_; // Select
  std::vector<hsql::TableRef*> tables_;         //from
  std::vector<Expression> conditions_;        //where
  std::vector<hsql::Expr *> group_by_;          // GROUP BY
  std::vector<IAgregation *> aggregation_list_; // Aggregation
};

class IOptimizationRule {
  public:
    virtual void apply(LogicalPlan& plan, const Catalog& catalog) = 0;
    virtual ~IOptimizationRule() = default;
};


class PredicatePushdownRule : public IOptimizationRule {
public:
  void apply(LogicalPlan &plan, const Catalog &catalog) override {
    std::vector<Expression> conditions = plan.getConditions();
    for (auto &expr : conditions) {
      if(expr.expr->isType(hsql::kExprOperator) && expr.expr->opType == hsql::kOpEquals){
        std::string first_table = expr.expr->expr->getName();
        std::string second_table = expr.expr->expr2->getName();
        if (getTableName(first_table) != getTableName(second_table)) {
          // Скрытый JOIN
          expr.type = ExpressionType::JOIN;
        } else {
          // Фильтр
          expr.type = ExpressionType::FILTER;
        }
      }
    }
    
  }
};

class DeadColumnEliminationRule : public IOptimizationRule {
public:
  void apply(LogicalPlan& plan, const Catalog& catalog) override {
    
  }
};

class StarSchemaJoinRule : public IOptimizationRule {
public:
  void apply(LogicalPlan& plan, const Catalog& catalog) override {
    
  }
};
} // namespace db