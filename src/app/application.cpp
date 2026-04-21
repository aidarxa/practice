#include "app/aplication.h"
#include <iostream>
#include <string>
#include <hsql/SQLParser.h>
#include <hsql/util/sqlhelper.h>

void Application::run(){
    while(true){
        std::cout << "Write input query:";
        std::string query;
        std::cin >> query;
        std::cout << "\n";
        if(!query.empty()){
            //parsing
            hsql::SQLParserResult result;
            hsql::SQLParser::parse(query, &result);
            if(result.isValid() && result.size() > 0){
                const hsql::SQLStatement* statement = result.getStatement(0);
                hsql::printStatementInfo(statement);
                if(statement->isType(hsql::kStmtSelect)){
                    const hsql::SelectStatement* select = static_cast<const hsql::SelectStatement*>(statement);
                    
                }
            }
        }
    }
}