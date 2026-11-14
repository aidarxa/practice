#!/usr/bin/env python3
"""Read-only environment/source inventory; run outside performance series."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
from run import ROOT, HERE, sha

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acpp-source", type=Path, required=True)
    parser.add_argument("--acpp-prefix", type=Path, required=True)
    parser.add_argument("--llvm-prefix", type=Path, required=True)
    parser.add_argument("--control-build", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("build/pcuda-jit-tiering"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="inventory-", dir=args.output_root.resolve()))
    sources = out / "sources"
    sources.mkdir()
    report = {"recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "commands": [],
        "environment": {k: v for k, v in os.environ.items() if k.startswith(("ACPP_", "HIP_", "ROCM_", "LLVM_", "OMP_"))
                        or k in ("PATH", "LD_LIBRARY_PATH", "LC_ALL")}, "files": {}, "web": []}
    commands = {
        "acpp-version": [str(args.acpp_prefix / "bin/acpp"), "--version"],
        "clang-version": [str(args.llvm_prefix / "bin/clang++"), "--version"],
        "lld-version": [str(args.llvm_prefix / "bin/ld.lld"), "--version"],
        "llvm-config": [str(args.llvm_prefix / "bin/llvm-config"), "--version", "--cxxflags", "--libdir"],
        "source-revision": ["git", "-C", str(args.acpp_source), "rev-parse", "HEAD"],
        "source-status": ["git", "-C", str(args.acpp_source), "status", "--short"],
        "project-status": ["git", "status", "--short"],
        "cpu": ["lscpu"], "kernel": ["uname", "-a"], "gpu": [str(args.llvm_prefix / "bin/rocminfo")],
        "hip": [str(args.llvm_prefix / "bin/hipconfig"), "--full"],
        "memory": ["free", "-b"],
    }
    for name, command in commands.items():
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, "LC_ALL": "C"}, capture_output=True, timeout=60)
        (out / (name + ".stdout")).write_bytes(result.stdout)
        (out / (name + ".stderr")).write_bytes(result.stderr)
        report["commands"].append({"name": name, "argv": command, "returncode": result.returncode})
    relative = ["src/compiler/llvm-to-backend/LLVMToBackend.cpp", "src/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.cpp",
        "src/compiler/llvm-to-backend/Utils.cpp", "src/compiler/llvm-to-backend/CMakeLists.txt",
        "src/runtime/hip/hip_queue.cpp", "src/runtime/hip/hip_code_object.cpp", "src/runtime/kernel_cache.cpp",
        "src/runtime/adaptivity_engine.cpp", "include/hipSYCL/runtime/kernel_cache.hpp",
        "include/hipSYCL/runtime/kernel_configuration.hpp", "include/hipSYCL/glue/llvm-sscp/jit.hpp",
        "doc/compilation.md", "doc/pcuda.md", "doc/env_variables.md", "doc/generic-sscp.md", "LICENSE"]
    for name in relative:
        path = args.acpp_source / name
        target = sources / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        report["files"][str(path)] = {"sha256": sha(path), "snapshot": str(target)}
    binaries = [*args.acpp_prefix.glob("lib/**/*.so"), *args.acpp_prefix.glob("lib/**/*.bc"),
                args.llvm_prefix / "bin/clang++", args.llvm_prefix / "bin/opt", args.llvm_prefix / "bin/ld.lld",
                args.llvm_prefix / "lib/libLLVM-17git.so", args.llvm_prefix / "lib/libhiprtc.so",
                args.llvm_prefix / "lib/libamd_comgr.so", args.control_build / "libjit_control.so"]
    for library in ("libamd_comgr.so*", "libamdhip64.so*", "libhsa-runtime64.so*"):
        binaries.extend(args.llvm_prefix.glob("lib*/" + library))
    binaries.extend((args.acpp_prefix / "include/AdaptiveCpp/hipSYCL/pcuda").glob("**/*.hpp"))
    binaries.extend((args.acpp_prefix / "include/AdaptiveCpp/hipSYCL/compiler/llvm-to-backend").glob("**/*.hpp"))
    binaries.append(args.acpp_prefix / "include/AdaptiveCpp/pcuda.hpp")
    for path in binaries:
        if path.is_file(): report["files"][str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    for name in ("CMakeCache.txt", "CMakeFiles/jit_control.dir/flags.make", "CMakeFiles/jit_control.dir/link.txt"):
        path = args.control_build / name
        if path.is_file():
            target = out / ("control-" + path.name)
            shutil.copy2(path, target)
            report["files"][str(path)] = {"sha256": sha(path), "snapshot": str(target)}
    docs = out / "web"
    docs.mkdir()
    def fetch(name, url):
        entry = {"name": name, "url": url}
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "pcuda-jit-research"})
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
            path = docs / name
            path.write_bytes(content)
            entry.update(path=str(path), sha256=sha(path), bytes=len(content))
        except Exception as error:
            entry["error"] = str(error)
            content = b""
        report["web"].append(entry)
        return content
    latest = fetch("adaptivecpp-head.json", "https://api.github.com/repos/AdaptiveCpp/AdaptiveCpp/commits/develop")
    if latest:
        revision = json.loads(latest)["sha"]
        report["upstream_head"] = revision
        for path in ("doc/env_variables.md", "src/compiler/llvm-to-backend/LLVMToBackend.cpp",
                     "src/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.cpp", "include/hipSYCL/runtime/kernel_cache.hpp"):
            fetch("upstream-" + Path(path).name, "https://raw.githubusercontent.com/AdaptiveCpp/AdaptiveCpp/" + revision + "/" + path)
    for name, url in {
        "llvm17-orc.rst": "https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-17.0.0/llvm/docs/ORCv2.rst",
        "llvm17-speculative.cpp": "https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-17.0.0/llvm/examples/SpeculativeJIT/SpeculativeJIT.cpp",
        "clang17-manual.html": "https://releases.llvm.org/17.0.1/tools/clang/docs/UsersManual.html",
        "android-art.html": "https://source.android.com/docs/core/runtime/jit-compiler",
        "rocm-hiprtc.cpp": "https://raw.githubusercontent.com/ROCm/clr/rocm-6.1.2/hipamd/src/hiprtc/hiprtcInternal.cpp",
    }.items(): fetch(name, url)
    (out / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(out)

if __name__ == "__main__": main()
