/** @file @brief An RTLD_LOCAL fixture must remain local while its original entry is callable. */
#include "probe_symbols.h"
#include <iostream>
#include <stdexcept>
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc,char** argv) {
    try {
        if (argc!=2) throw std::runtime_error("fixture path required");
        if (db::s2probe::localSymbol(nullptr,"nothing")) throw std::runtime_error("missing owner must not resolve");
        if (db::s2probe::localSymbol(argv[1],"probe_fixture_value")) throw std::runtime_error("NOLOAD must not load the fixture");
        if (db::s2probe::original<pid_t(*)()>("getpid")()!=getpid()) throw std::runtime_error("global original not preserved");
        auto* handle=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
        if (!handle) throw std::runtime_error(dlerror());
        if (dlsym(RTLD_DEFAULT,"probe_fixture_value")) throw std::runtime_error("fixture must not become global");
        auto function=db::s2probe::original<int(*)()>("probe_fixture_value",argv[1]);
        if (function()!=73) throw std::runtime_error("original entry was not called");
        if (dlsym(RTLD_DEFAULT,"probe_fixture_value")) throw std::runtime_error("lookup promoted the fixture");
        const auto child=fork();
        if (child<0) throw std::runtime_error("fork failed");
        if (!child) {
            const rlimit zero{0,0};setrlimit(RLIMIT_CORE,&zero);
            db::s2probe::original<int(*)()>("missing_fixture_entry",argv[1]);
            _exit(90);
        }
        int status=0;
        if (waitpid(child,&status,0)!=child || !WIFSIGNALED(status) || WTERMSIG(status)!=SIGABRT)
            throw std::runtime_error("missing original must fail closed");
        dlclose(handle);
        std::cout << "local original preserved\n";
    } catch (const std::exception& error) {std::cerr << error.what() << '\n';return 1;}
}
