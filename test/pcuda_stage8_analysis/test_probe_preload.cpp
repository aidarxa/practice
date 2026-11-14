/** @file @brief Prove ld.so --preload does not propagate observer mappings across a child exec. */
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc,char** argv) {
    if (argc!=3 || std::getenv("LD_PRELOAD")) return 1;
    void* handle=dlopen(argv[1],RTLD_NOW|RTLD_NOLOAD);
    const bool child=std::strcmp(argv[2],"child")==0;
    if (handle) dlclose(handle);
    if (child) return handle?2:0;
    if (!handle) return 3;
    const auto pid=fork();
    if (pid<0) return 4;
    if (!pid) {execl(argv[2],argv[2],argv[1],"child",nullptr);_exit(5);}
    int status=0;
    if (waitpid(pid,&status,0)!=pid || !WIFEXITED(status) || WEXITSTATUS(status)) return 6;
    std::cout << "observer mapped only in parent; child environment and mappings clean\n";
}
