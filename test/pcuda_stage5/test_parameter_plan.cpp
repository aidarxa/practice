/** @file @brief Value-free physical lowering, exercised through the real SQL frontend. */
#include "core/translator.h"
#include "core/optimizer_rules.h"
#include "core/visitor.h"
#include <cassert>
#include <iostream>

using namespace db;

static void expressionContracts() {
    const ColumnExpression column=[](const ColumnRefExpr& ref){return ExpressionCode{ref.column_name,"column_valid"};};
    const ParameterSchema schema={ParameterType::Int64,ParameterType::Float64};
    const ParameterRefExpr parameter(0);
    assert(emitSqlExpression(&parameter,column).value=="db_parameter_0");
    assert(emitSqlExpression(&parameter,column).valid=="db_parameter_0_valid");
    assert(emitSqlExpression(nullptr,column).valid=="0");
    LiteralIntExpr minimum(INT64_MIN);LiteralFloatExpr fraction(1.25);StarExpr star;
    assert(emitSqlExpression(&minimum,column).value=="(-9223372036854775807LL - 1LL)");
    assert(emitSqlExpression(&fraction,column).value=="1.25000000000000000e+00");
    assert(emitSqlExpression(&star,column).value=="1");
    assert(sqlExpressionType(&fraction,schema)==ParameterType::Float64);
    ParameterRefExpr floating(1);assert(sqlExpressionType(&floating,schema)==ParameterType::Float64);
    for(const auto op:{ExprType::OP_ADD,ExprType::OP_SUB,ExprType::OP_MUL,ExprType::OP_DIV,ExprType::OP_EQ,ExprType::OP_NEQ,
                      ExprType::OP_LT,ExprType::OP_LTE,ExprType::OP_GT,ExprType::OP_GTE,ExprType::OP_AND,ExprType::OP_OR}) {
        BinaryExpr expression(op,std::make_unique<ColumnRefExpr>("input"),std::make_unique<ParameterRefExpr>(0));
        const auto result=emitSqlExpression(&expression,column);
        assert(result.value.find("input")!=std::string::npos && result.valid.find("column_valid")!=std::string::npos);
        (void)sqlExpressionType(&expression,schema);
    }
    for(const auto op:{ExprType::OP_NOT,ExprType::OP_IS_NULL,ExprType::OP_IS_NOT_NULL}) {
        BinaryExpr expression(op,std::make_unique<LiteralNullExpr>(),nullptr);
        assert(!emitSqlExpression(&expression,column).value.empty());
    }
    CaseWhenExpr branch(std::make_unique<LiteralIntExpr>(0),std::make_unique<ParameterRefExpr>(0),std::make_unique<LiteralNullExpr>());
    assert(caseLeafCount(&branch)==2 && emitCaseGroupCode(&branch,column,schema).find("db::safe_eq")!=std::string::npos);
    assert(sqlExpressionType(&branch,schema)==ParameterType::Int64);
    bool rejected=false;
    try{emitCaseGroupCode(&branch,column,{ParameterType::Float64});}catch(const std::runtime_error&){rejected=true;}
    assert(rejected);
}

static PlannedQuery lower(const std::string& sql, const Catalog& catalog) {
    hsql::SQLParserResult parsed;
    hsql::SQLParser::parse(sql, &parsed);
    assert(parsed.isValid());
    auto prepared = QueryTranslator{}.prepare(
        static_cast<const hsql::SelectStatement*>(parsed.getStatement(0)), catalog);
    prepared.query.plan = Optimizer{}.optimize(std::move(prepared.query.plan));
    JITContext context;
    context.parameter_schema = prepared.query.parameter_schema;
    JITOperatorVisitor visitor(context, catalog);
    prepared.query.plan->accept(visitor);
    return visitor.buildPhysicalPlan();
}

int main() try {
    expressionContracts();
    Catalog catalog;
    TableMetadata table("LINEORDER", {"lo_quantity", "lo_revenue"}, 19, true);
    table.setColumnStats("lo_quantity", {0, 9, 10});
    catalog.pushTableMetadata(table);
    const auto first = lower("SELECT SUM(lo_revenue) FROM lineorder WHERE lo_quantity > 3", catalog);
    const auto second = lower("SELECT SUM(lo_revenue) FROM lineorder WHERE lo_quantity > 8", catalog);
    assert(first.bodies.size() == second.bodies.size());
    bool parameter = false;
    for (size_t i = 0; i < first.bodies.size(); ++i) {
        assert(first.bodies[i].code == second.bodies[i].code);
        for (const auto& binding : first.plan.kernels[i].arguments)
            parameter |= binding.schema.kind == DB_ARG_PARAMETER;
    }
    assert(parameter);
    const auto a=lower("SELECT CASE WHEN lo_quantity < 4 THEN 1 ELSE 1000000000 END,COUNT(*) FROM lineorder GROUP BY CASE WHEN lo_quantity < 4 THEN 1 ELSE 1000000000 END ORDER BY 1 LIMIT 0",catalog);
    const auto b=lower("SELECT CASE WHEN lo_quantity < 7 THEN 99 ELSE 99 END,COUNT(*) FROM lineorder GROUP BY CASE WHEN lo_quantity < 7 THEN 99 ELSE 99 END ORDER BY 1 LIMIT 9999",catalog);
    assert(a.bodies.size()==b.bodies.size());
    for(size_t i=0;i<a.bodies.size();++i) assert(a.bodies[i].code==b.bodies[i].code);
    std::cout << "Value-free physical parameter bindings passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
