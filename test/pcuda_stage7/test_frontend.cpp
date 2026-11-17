/** @file @brief Real storage types resolve without raw Int32 or dictionary substitution. */
#include "core/translator.h"
#include <SQLParser.h>
#include <cassert>
#include <iostream>

static db::PreparedQuery prepare(const std::string& sql, const Catalog& catalog) {
    hsql::SQLParserResult ast;
    hsql::SQLParser::parse(sql, &ast);
    assert(ast.isValid() && ast.size() == 1);
    return db::QueryTranslator{}.prepare(static_cast<const hsql::SelectStatement*>(ast.getStatement(0)), catalog);
}
#ifdef SSB_COMBINED_TEST
int frontendChecks() {
#else
int main() {
#endif
    Catalog catalog;
    TableMetadata table("SUPPLIER", db::ssbColumnNames("SUPPLIER"), 3, false);
    for (const auto& column : db::ssbColumns)
        if (std::string(column.table) == "SUPPLIER") table.setColumnStorage(column.name, column.storage);
    catalog.pushTableMetadata(table);
    for (const auto* sql : {"SELECT s_name FROM supplier", "SELECT x.s_name AS v FROM supplier x ORDER BY v",
            "SELECT COUNT(s_name) FROM supplier", "SELECT SUM(s_suppkey) FROM supplier WHERE s_name='Alice'",
            "SELECT s_name,COUNT(*) FROM supplier GROUP BY s_name", "SELECT * FROM supplier"}) {
        assert(prepare(sql, catalog).query.plan);
    }
    for (const auto* sql : {"SELECT s_name FROM supplier WHERE s_name=1", "SELECT SUM(s_name) FROM supplier",
                            "SELECT AVG(s_name) FROM supplier"}) {
        bool rejected = false;
        try { (void)prepare(sql, catalog); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) std::cerr << "Incompatible Text operation was not rejected: " << sql << '\n';
        assert(rejected);
    }
    assert(prepare("SELECT COUNT(*) FROM supplier", catalog).query.plan);
    assert(prepare("SELECT s_suppkey FROM supplier WHERE s_region='AMERICA'", catalog).query.plan);
    std::cout << "Typed storage frontend checks passed\n";
    return 0;
}
