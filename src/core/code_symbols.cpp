#include "core/code_symbols.h"
#include <cctype>

namespace db {
namespace {
bool start(char c) { return std::isalpha(static_cast<unsigned char>(c)) || c=='_'; }
bool part(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c=='_'; }
size_t quotedEnd(const std::string& text,size_t i) {
    const auto quote=text[i++];
    while(i<text.size()) {
        const auto c=text[i++];
        if(c=='\\' && i<text.size()) ++i;
        else if(c==quote) break;
    }
    return i;
}
size_t nonCodeEnd(const std::string& text,size_t i) {
    if(text[i]=='"' || text[i]=='\'') return quotedEnd(text,i);
    if(text.compare(i,2,"//")==0) {
        const auto end=text.find('\n',i);
        return end==std::string::npos?text.size():end;
    }
    if(text.compare(i,2,"/*")==0) {
        const auto end=text.find("*/",i+2);
        return end==std::string::npos?text.size():end+2;
    }
    return i;
}
template<class Visitor> void symbols(const std::string& text,Visitor visit) {
    size_t i=0;
    while(i<text.size()) {
        const auto skipped=nonCodeEnd(text,i);
        if(skipped!=i) i=skipped;
        else if(start(text[i])) {
            const auto begin=i++;
            while(i<text.size() && part(text[i])) ++i;
            visit(begin,i);
        } else ++i;
    }
}
} // namespace
std::set<std::string> codeIdentifiers(const std::string& code) {
    std::set<std::string> result;
    symbols(code,[&](size_t begin,size_t end) {result.insert(code.substr(begin,end-begin));});
    return result;
}
std::string replaceCodeIdentifiers(const std::string& code,const std::map<std::string,std::string>& replacements) {
    std::string result;
    size_t copied=0;
    symbols(code,[&](size_t begin,size_t end) {
        const auto found=replacements.find(code.substr(begin,end-begin));
        if(found==replacements.end()) return;
        result.append(code,copied,begin-copied);
        result+=found->second;
        copied=end;
    });
    result.append(code,copied,std::string::npos);
    return result;
}
} // namespace db
