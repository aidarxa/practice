#pragma once
/** @file @brief Version-bound ELF passthrough lookup, for diagnostics only. */
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

namespace db { namespace s2probe {
/** @brief Consult an already-loaded local DSO without loading or promoting its scope. */
inline void* localSymbol(const char* library,const char* name) {
    if (!library) return nullptr;
    void* handle=dlopen(library,RTLD_NOW|RTLD_NOLOAD);
    if (!handle) return nullptr;
    void* symbol=dlsym(handle,name);
    dlclose(handle);
    return symbol;
}
/** @brief Forward to the existing ELF definition; a missing original is a diagnostic failure. */
template<class Function> Function original(const char* name,const char* local_library=nullptr) {
    dlerror();
    void* symbol=dlsym(RTLD_NEXT,name);
    if (!symbol) symbol=localSymbol(local_library,name);
    if (!symbol) {std::fprintf(stderr,"S2 missing original symbol: %s\n",name);std::abort();}
    return reinterpret_cast<Function>(symbol);
}
}} // namespace db::s2probe
