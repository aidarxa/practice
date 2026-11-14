/** @file @brief Behavioral frontend contract for real Text, without a GPU dependency. */
#include "core/translator.h"
#include "core/query_template.h"
#include <iostream>
#include <stdexcept>

namespace {
void prepare(const char* sql, const Catalog& catalog) {
    hsql::SQLParserResult parsed;
    hsql::SQLParser::parse(sql, &parsed);
    if (!parsed.isValid()) throw std::runtime_error("fixture SQL must parse");
    auto query = db::QueryTranslator{}.prepare(
        static_cast<const hsql::SelectStatement*>(parsed.getStatement(0)), catalog);
    if (!query.query.plan) throw std::runtime_error("missing Text plan");
}
/** @brief Unsupported implicit Text-to-Boolean/number coercion must fail before codegen. */
void rejects(const char* sql, const Catalog& catalog) {
    try { prepare(sql, catalog); }
    catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("Text") != std::string::npos) return;
        throw;
    }
    throw std::runtime_error(std::string("Expected Text type error: ") + sql);
}
}

#ifdef TEXT_COMBINED_TEST
int textFrontendChecks() {
#else
int main() {
#endif
    try {
        Catalog catalog;
        TableMetadata table("CUSTOMER", {"c_custkey", "c_name"}, 4, false);
        table.setColumnStorage("c_name", {db::PhysicalEncoding::FixedText, 25});
        catalog.pushTableMetadata(table);
        prepare("SELECT c_name FROM customer", catalog);
        prepare("SELECT c_name FROM customer WHERE c_name = 'Alice'", catalog);
        prepare("SELECT CASE WHEN c_custkey=1 THEN c_name ELSE 'other' END FROM customer", catalog);
        prepare("SELECT c_name,COUNT(*) FROM customer GROUP BY c_name ORDER BY 1", catalog);
        prepare("SELECT * FROM customer LIMIT 2", catalog);
        prepare("SELECT * FROM customer ORDER BY c_name LIMIT 2", catalog);
        prepare("SELECT *,c_name AS copied FROM customer ORDER BY copied,2 LIMIT 2", catalog);
        for (const auto* sql : {"SELECT c_name FROM customer WHERE c_name", "SELECT NOT c_name FROM customer",
                               "SELECT CASE WHEN c_name THEN 1 ELSE 2 END FROM customer",
                               "SELECT c_name+1 FROM customer", "SELECT c_name FROM customer WHERE c_name=1",
                               "SELECT CASE WHEN c_custkey=1 THEN c_name ELSE 2 END FROM customer"})
            rejects(sql, catalog);
        std::cout << "Text frontend behavior passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
