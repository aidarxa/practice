/** @file @brief Transparent, version-bound S2 host timers for AdaptiveCpp 5989e1cf/LLVM17.
 * @details Calls the original ELF functions. No optimizer levels, backend route or passes change.
 * This diagnostic DSO is not linked into or installed with the engine.
 */
#include "probe_log.h"
#include "probe_symbols.h"
#include <hipSYCL/compiler/llvm-to-backend/Utils.hpp>
#include <cstdlib>
#include <limits>
#include <mutex>

namespace {
using Translator=hipsycl::compiler::LLVMToBackendTranslator;
using db::s2probe::original;
using db::s2probe::Span;

template<class Function> bool observedBool(const char* kind, Function function, std::uint64_t bytes=0, std::string name={}) {
    Span interval(kind,bytes,std::move(name));
    const bool result=function();
    interval.finish(result?0:1);
    return result;
}
std::uint64_t nextTool() {
    static std::mutex mutex;
    static std::uint64_t next=0;
    std::lock_guard<std::mutex> lock(mutex);
    if (next==std::numeric_limits<std::uint64_t>::max()) std::abort();
    return ++next;
}
}

namespace hipsycl { namespace compiler {
bool LLVMToBackendTranslator::fullTransformation(const std::string& input,std::string& output) {
    using Function=bool(*)(Translator*,const std::string&,std::string&);
    static auto function=original<Function>("_ZN7hipsycl8compiler23LLVMToBackendTranslator18fullTransformationERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERS7_");
    db::s2probe::CompilationScope scope;
    return observedBool("full_transformation",[&] {return function(this,input,output);},input.size(),getCompilationIdentifier());
}
bool LLVMToBackendTranslator::optimizeFlavoredIR(llvm::Module& module,PassHandler& passes) {
    using Function=bool(*)(Translator*,llvm::Module&,PassHandler&);
    static auto function=original<Function>("_ZN7hipsycl8compiler23LLVMToBackendTranslator18optimizeFlavoredIRERN4llvm6ModuleERNS0_11PassHandlerE");
    return observedBool("optimizer",[&] {return function(this,module,passes);});
}
int executeAndWait(llvm::StringRef program,llvm::ArrayRef<llvm::StringRef> args,
                   std::optional<llvm::ArrayRef<llvm::StringRef>> environment,
                   llvm::ArrayRef<std::optional<llvm::StringRef>> redirects) {
    using Function=int(*)(llvm::StringRef,llvm::ArrayRef<llvm::StringRef>,std::optional<llvm::ArrayRef<llvm::StringRef>>,llvm::ArrayRef<std::optional<llvm::StringRef>>);
    static auto function=original<Function>("_ZN7hipsycl8compiler14executeAndWaitEN4llvm9StringRefENS1_8ArrayRefIS2_EESt8optionalIS4_ENS3_IS5_IS2_EEE");
    std::vector<std::string> actual;
    for (auto argument:args) actual.push_back(argument.str());
    auto command=actual;
    std::string executable=program.str(),prefix;
    const char* timer=std::getenv("CRYSTAL_S2_NATIVE_TIMER");
    const char* directory=std::getenv("CRYSTAL_S2_DIRECTORY");
    if (timer && directory) {
        prefix=std::string(directory)+"/s2-tool-"+std::to_string(nextTool());
        command.insert(command.begin(),{"bash",timer,prefix});
        executable="/bin/bash";
    }
    std::vector<llvm::StringRef> references;
    for (const auto& argument:command) references.emplace_back(argument);
    Span interval("external_tool",0,prefix,std::move(actual));
    const int result=function(executable,references,environment,redirects);
    interval.finish(result);
    return result;
}
}} // namespace hipsycl::compiler
