#include "core/module_cache.h"
#include "core/content_hash.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;

namespace db {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
std::string readText(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot read JIT artifact: " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
void writeText(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path);
    output << text;
    output.close();
    if (!output) throw std::runtime_error("Cannot write JIT artifact: " + path.string());
}
std::string shellQuote(const std::string& text) {
    std::string result = "'";
    for (const auto ch : text) result += ch == '\'' ? "'\\''" : std::string(1,ch);
    return result + "'";
}
std::string command(const std::vector<std::string>& argv) {
    std::string result;
    for (const auto& arg : argv) result += shellQuote(arg) + ' ';
    return result;
}
void runScript(const std::filesystem::path& path) {
    std::string shell = "bash", script = path.string();
    char* argv[] = {shell.data(), script.data(), nullptr};
    pid_t child;
    const auto error = posix_spawnp(&child, shell.c_str(), nullptr, nullptr, argv, environ);
    if (error) throw std::runtime_error("Cannot start native JIT build, error " + std::to_string(error));
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) throw std::runtime_error("Cannot wait for native JIT build");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status))
        throw std::runtime_error("PCUDA module build failed; inspect " + path.parent_path().string());
}
struct Toolchain {
    std::string compiler, pch, header, fingerprint;
    std::vector<std::string> compile, link;
    std::unordered_set<std::string> checked;
    std::vector<std::string> environment;
};
std::string verifyFile(const std::string& text) {
    const auto separator = text.find('\t');
    if (separator != 64) throw std::runtime_error("Malformed PCUDA PCH fingerprint record");
    const auto path=text.substr(separator+1);
    if (!std::filesystem::path(path).is_absolute() || fileContentHash(path) != text.substr(0, separator))
        throw std::runtime_error("PCUDA PCH/toolchain input changed: " + text.substr(separator + 1) + "; rebuild target pcuda_jit_pch");
    return path;
}
bool compilerValue(Toolchain& toolchain,const std::string& key,const std::string& value) {
    const std::pair<const char*,std::string*> fields[]={{"compiler",&toolchain.compiler},{"pch",&toolchain.pch},{"header",&toolchain.header}};
    for (const auto& field:fields) {
        if (key==field.first) {*field.second=value;return true;}
    }
    return false;
}
bool compilerEnvironment(Toolchain& toolchain,const std::string& key,const std::string& value) {
    if (key=="compiler_env_unset") toolchain.environment.push_back("unset "+shellQuote(value)+"\n");
    else if (key=="compiler_env_set") {
        const auto separator=value.find('\t');
        if (separator==std::string::npos) throw std::runtime_error("Malformed PCUDA compiler environment record");
        toolchain.environment.push_back("export "+shellQuote(value.substr(0,separator)+"="+value.substr(separator+1))+"\n");
    } else return false;
    return true;
}
void parseRecord(Toolchain& toolchain,const std::string& key,const std::string& value) {
    if (compilerValue(toolchain,key,value) || compilerEnvironment(toolchain,key,value)) return;
    if (key=="compile_arg") toolchain.compile.push_back(value);
    else if (key=="link_arg") toolchain.link.push_back(value);
    else if (key=="file") {
        if (!toolchain.checked.insert(verifyFile(value)).second) throw std::runtime_error("Duplicate PCUDA PCH fingerprint record");
    } else throw std::runtime_error("Unknown PCUDA PCH manifest record: "+key);
}
void requireFlag(const std::vector<std::string>& flags, const std::string& flag) {
    if (std::find(flags.begin(), flags.end(), flag) == flags.end())
        throw std::runtime_error("PCUDA PCH manifest requires " + flag);
}
void validateCompileFlag(const std::string& flag) {
    const bool wrong_optimization=flag.rfind("-O",0)==0 && flag!="-O0";
    const bool wrong_standard=flag.rfind("-std=",0)==0 && flag!="-std=c++17";
    if (wrong_optimization || wrong_standard || flag=="-ffast-math" || flag=="-fno-validate-pch")
        throw std::runtime_error("Incompatible PCUDA SQL compiler flag: "+flag+"; rebuild target pcuda_jit_pch with C++17/O0 and normal PCH validation");
}
std::string manifestPath(std::string path) {
    if (const auto* value = std::getenv("CRYSTAL_PCUDA_MANIFEST")) path = value;
    if (!path.empty()) return path;
    std::error_code error;
    const auto executable=std::filesystem::read_symlink("/proc/self/exe",error);
    const auto installed=executable.parent_path().parent_path()/"lib/crystal/pcuda-jit/toolchain.manifest";
    if (!error && std::filesystem::is_regular_file(installed)) return installed.string();
    if (path.empty()) {
#ifdef CRYSTAL_PCUDA_MANIFEST_DEFAULT
        path = CRYSTAL_PCUDA_MANIFEST_DEFAULT;
#endif
    }
    if (path.empty()) throw std::runtime_error("Set pcuda_manifest or build target pcuda_jit_pch");
    return path;
}
void validateToolchain(const Toolchain& toolchain) {
    for (const auto& flag:toolchain.compile) validateCompileFlag(flag);
    for (const auto* flag : {"-O0","-fPIC","-std=c++17","--acpp-targets=generic","--acpp-pcuda"}) requireFlag(toolchain.compile,flag);
    for (const auto& path:{toolchain.compiler,toolchain.pch,toolchain.header}) {
        if (!toolchain.checked.count(path)) throw std::runtime_error("Missing compiler/PCH/header fingerprints in PCUDA manifest; rebuild target pcuda_jit_pch");
    }
}
Toolchain readToolchain(std::string path) {
    path=manifestPath(std::move(path));
    const auto contents = readText(path);
    std::istringstream input(contents);
    std::string line;
    std::getline(input,line);
    if (line != "DB-PCUDA-PCH-1") throw std::runtime_error("Incompatible PCUDA PCH manifest; rebuild target pcuda_jit_pch");
    Toolchain result;
    while (std::getline(input,line)) {
        const auto separator = line.find('\t');
        if (separator == std::string::npos) throw std::runtime_error("Malformed PCUDA PCH manifest");
        parseRecord(result,line.substr(0,separator),line.substr(separator+1));
    }
    validateToolchain(result);
    result.fingerprint = contentHash(contents);
    return result;
}
void appendTiming(const std::filesystem::path& path, double& wall, double& user, double& system) {
    std::istringstream input(readText(path));
    double a,b,c;
    if (!(input >> a >> b >> c) || a < 0 || b < 0 || c < 0) throw std::runtime_error("Invalid native JIT timer: " + path.string());
    wall += a*1000; user += b*1000; system += c*1000;
}
void timedCommand(std::ostream& script, const std::vector<std::string>& argv, const std::filesystem::path& prefix) {
    script << ": >" << shellQuote(prefix.string()+".started") << "\nif { time " << command(argv) << " >" << shellQuote(prefix.string()+".stdout")
           << " 2>" << shellQuote(prefix.string()+".stderr") << "; } 2>" << shellQuote(prefix.string()+".time")
           << "; then echo 0 >" << shellQuote(prefix.string()+".status")
           << "; else status=$?; echo \"$status\" >" << shellQuote(prefix.string()+".status") << "; exit \"$status\"; fi\n";
    writeText(prefix.string()+".command",command(argv)+"\n");
}
} // namespace

ModuleArtifacts::ModuleArtifacts(const std::filesystem::path& parent, bool keep) : keep_(keep) {
    const auto directory = parent.empty() ? std::filesystem::temp_directory_path() : parent;
    if (!std::filesystem::is_directory(directory)) throw std::runtime_error("JIT artifact parent is not a directory: " + directory.string());
    auto pattern = (directory/"db-pcuda-XXXXXX").string();
    if (!mkdtemp(pattern.data())) throw std::runtime_error("Cannot create private PCUDA artifact directory");
    path_ = pattern;
}
ModuleArtifacts::~ModuleArtifacts() noexcept {
    if (keep_) return;
    std::error_code error;
    std::filesystem::remove_all(path_,error); // only the exact private mkdtemp directory this object created
}
struct AdaptiveCppCompiler::Impl {
    Toolchain toolchain;
    CrystalConfig config;
    std::shared_ptr<ModuleArtifacts> artifacts;
    size_t batch = 0;
    explicit Impl(const CrystalConfig& input) : toolchain(readToolchain(input.pcuda_manifest)), config(input),
        artifacts(std::make_shared<ModuleArtifacts>(input.jit_artifact_dir,input.jit_keep_artifacts)) {
        if (!config.jit_jobs || config.jit_jobs > 4) throw std::runtime_error("jit_jobs must be between 1 and 4");
    }
    void chain(const KernelModuleSource& module, const std::filesystem::path& base, std::ostream& script) const {
        writeText(base.string()+".cpp",module.source);
        auto compile = toolchain.compile;
        compile.insert(compile.begin(),toolchain.compiler);
        compile.insert(compile.end(),{"-include-pch",toolchain.pch,"-c",base.string()+".cpp","-o",base.string()+".o"});
        auto link = toolchain.link;
        link.insert(link.begin(),toolchain.compiler);
        link.insert(link.end(),{"-shared",base.string()+".o","-o",base.string()+".so"});
        if (config.jit_traces) {
            compile.push_back("-ftime-trace="+base.string()+".clang.json");
            link.push_back("-Wl,--time-trace="+base.string()+".lld.json");
        }
        script << "(\nset -e\n";
        timedCommand(script,compile,base.string()+".compile");
        timedCommand(script,link,base.string()+".link");
        script << ") &\npids+=(\"$!\")\n";
    }
    std::vector<CompiledModule> script(const std::filesystem::path& directory,const std::vector<KernelModuleSource>& modules) const {
        std::ostringstream output;
        output << "set -u\nexport LC_ALL=C\nTIMEFORMAT='%6R %6U %6S'\npids=()\nfailed=0\n";
        for (const auto& environment:toolchain.environment) output << environment;
        std::vector<CompiledModule> result;
        for (size_t i=0;i<modules.size();++i) {
            const auto base=directory/modules[i].identity;
            chain(modules[i],base,output);
            result.push_back({base.string()+".so",artifacts});
            if ((i+1)%config.jit_jobs==0 || i+1==modules.size())
                output << "for pid in \"${pids[@]}\"; do wait \"$pid\" || failed=1; done\npids=()\nif ((failed)); then exit 1; fi\n";
        }
        writeText(directory/"build.sh",output.str());
        return result;
    }
    void collect(const std::filesystem::path& base,ModuleCounters& counters,ModuleBuildTiming& timing) const {
        if (std::filesystem::exists(base.string()+".compile.started")) {
            recordModuleEvent(counters.compiler);
            appendTiming(base.string()+".compile.time",timing.compile_wall_ms,timing.compile_user_ms,timing.compile_system_ms);
        }
        if (std::filesystem::exists(base.string()+".link.started")) {
            recordModuleEvent(counters.linker);
            appendTiming(base.string()+".link.time",timing.link_wall_ms,timing.link_user_ms,timing.link_system_ms);
        }
    }
};
AdaptiveCppCompiler::AdaptiveCppCompiler() : AdaptiveCppCompiler(loadCrystalConfig()) {}
AdaptiveCppCompiler::AdaptiveCppCompiler(const CrystalConfig& config) : impl_(std::make_unique<Impl>(config)) {}
AdaptiveCppCompiler::~AdaptiveCppCompiler() = default;
const std::string& AdaptiveCppCompiler::fingerprint() const { return impl_->toolchain.fingerprint; }
const std::filesystem::path& AdaptiveCppCompiler::artifactPath() const { return impl_->artifacts->path(); }
std::vector<CompiledModule> AdaptiveCppCompiler::compile(const std::vector<KernelModuleSource>& modules,
                                                       ModuleCounters& counters, ModuleBuildTiming& timing) {
    if (modules.empty()) return {};
    const auto directory = impl_->artifacts->path()/std::to_string(impl_->batch);
    impl_->batch=checkedSizeAdd(impl_->batch,1);
    std::filesystem::create_directory(directory);
    const auto result=impl_->script(directory,modules);
    const auto path = directory/"build.sh";
    const auto start = Clock::now();
    std::exception_ptr failure;
    try { runScript(path); }
    catch (...) { impl_->artifacts->preserve(); failure=std::current_exception(); }
    timing.build_elapsed_ms += milliseconds(start);
    for (const auto& module:modules) impl_->collect(directory/module.identity,counters,timing);
    if (failure) std::rethrow_exception(failure);
    return result;
}
} // namespace db
