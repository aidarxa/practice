#include <cassert>
#include <iostream>

enum class SqlBool { False, True, Unknown };

static SqlBool sql_not(SqlBool v) {
    if (v == SqlBool::True) return SqlBool::False;
    if (v == SqlBool::False) return SqlBool::True;
    return SqlBool::Unknown;
}

static SqlBool sql_and(SqlBool a, SqlBool b) {
    if (a == SqlBool::False || b == SqlBool::False) return SqlBool::False;
    if (a == SqlBool::True && b == SqlBool::True) return SqlBool::True;
    return SqlBool::Unknown;
}

static SqlBool sql_or(SqlBool a, SqlBool b) {
    if (a == SqlBool::True || b == SqlBool::True) return SqlBool::True;
    if (a == SqlBool::False && b == SqlBool::False) return SqlBool::False;
    return SqlBool::Unknown;
}

int main() {
    assert(sql_and(SqlBool::True, SqlBool::Unknown) == SqlBool::Unknown);
    assert(sql_and(SqlBool::False, SqlBool::Unknown) == SqlBool::False);
    assert(sql_and(SqlBool::Unknown, SqlBool::Unknown) == SqlBool::Unknown);
    assert(sql_or(SqlBool::True, SqlBool::Unknown) == SqlBool::True);
    assert(sql_or(SqlBool::False, SqlBool::Unknown) == SqlBool::Unknown);
    assert(sql_or(SqlBool::Unknown, SqlBool::Unknown) == SqlBool::Unknown);
    assert(sql_not(SqlBool::Unknown) == SqlBool::Unknown);
    std::cout << "SQL three-valued logic model tests passed\n";
    return 0;
}
