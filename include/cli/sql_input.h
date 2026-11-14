#pragma once
/** @file @brief SQL input framing without changing whitespace inside UTF-8 literals. */
#include <string_view>

namespace db { namespace cli {
/** @brief Local lexer state for framing, not SQL parsing or literal decoding. */
class SqlInputState {
    enum class State { Sql, Single, Double, LineComment, BlockComment };
    State state_=State::Sql;
    bool ordinary(char c,char next,size_t& i) {
        if(c==';') return true;
        if(c=='\'') state_=State::Single;
        else if(c=='"') state_=State::Double;
        else if(c=='-' && next=='-') { state_=State::LineComment; ++i; }
        else if(c=='/' && next=='*') { state_=State::BlockComment; ++i; }
        return false;
    }
    void quoted(char c,char next,size_t& i) {
        const char quote=state_==State::Single?'\'':'"';
        if(c==quote) { if(next==quote) ++i; else state_=State::Sql; }
    }
public:
    /** @brief Advance one lexical state; true denotes an unquoted semicolon. */
    bool consume(char c,char next,size_t& i) {
        switch(state_) {
            case State::Sql: return ordinary(c,next,i);
            case State::Single:
            case State::Double: quoted(c,next,i); break;
            case State::LineComment: if(c=='\n') state_=State::Sql; break;
            case State::BlockComment: if(c=='*' && next=='/') { state_=State::Sql; ++i; } break;
        }
        return false;
    }
};
/** @brief Recognize a statement terminator outside SQL quotes/comments; doubled quotes stay literal. */
inline bool hasSqlTerminator(std::string_view sql) {
    SqlInputState state;
    for (size_t i=0;i<sql.size();++i) {
        const auto next=i+1<sql.size()?sql[i+1]:'\0';
        if(state.consume(sql[i],next,i)) return true;
    }
    return false;
}
}} // namespace db::cli
