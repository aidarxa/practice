#!/usr/bin/env python3
"""Capture installed toolchain help, effective commands, sources and documentation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import urllib.request


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acpp", default="acpp")
    parser.add_argument("--extra-flag", action="append", default=[])
    parser.add_argument("--acpp-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("build/pcuda-compile-perf"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="inventory-", dir=args.output_root.resolve()))
    root = Path(__file__).resolve().parents[2]
    acpp = shutil.which(args.acpp)
    if not acpp:
        parser.error("acpp not found")
    records = []

    def capture(name, argv, cwd=root):
        result = subprocess.run([str(x) for x in argv], cwd=cwd, capture_output=True,
                                env={**os.environ, "LC_ALL": "C"}, text=True, timeout=120)
        (out / (name + ".stdout")).write_text(result.stdout)
        (out / (name + ".stderr")).write_text(result.stderr)
        records.append({"name": name, "argv": [str(x) for x in argv],
                        "cwd": str(cwd), "returncode": result.returncode})
        return result.stdout

    base = [acpp, "--acpp-targets=generic", "--acpp-pcuda", *args.extra_flag,
            "-std=c++17", "-fPIC", "-O0"]
    effective = capture("acpp-compile-dryrun", [*base, "--acpp-dryrun", "-c",
                        root / "test/pcuda_stage0/q33_probe.cpp", "-o", out / "probe.o"])
    compiler = shlex.split(effective)[0]
    compile_prefix = shlex.split(effective)[:-4]
    # Keep the full effective command too; the reusable prefix is not a second compiler.
    for name, argv in (
        ("acpp-help", [acpp, "--help"]),
        ("acpp-version", [acpp, "--acpp-version"]),
        ("clang-version", [compiler, "--version"]),
        ("clang-help", [compiler, "--help"]),
        ("clang-hidden", [compiler, "--help-hidden"]),
        ("clang-cc1-help", [compiler, "-cc1", "--help"]),
        ("llvm-hidden", [*compile_prefix, "-mllvm", "--help-hidden", "-x", "c++", "/dev/null", "-o", out / "empty.o"]),
        ("lld-version", [str(Path(compiler).with_name("ld.lld")), "--version"]),
        ("lld-help", [str(Path(compiler).with_name("ld.lld")), "--help"]),
        ("bfd-version", ["ld.bfd", "--version"]),
        ("bfd-help", ["ld.bfd", "--help"]),
        ("gold-version", ["ld.gold", "--version"]),
        ("gold-help", ["ld.gold", "--help"]),
        ("bash-version", ["bash", "--version"]),
        ("cpu", ["lscpu"]), ("memory", ["free", "-b"]),
        ("kernel", ["uname", "-a"]), ("load", ["uptime"]),
        ("acpp-git", ["git", "rev-parse", "HEAD"]),
    ):
        capture(name, argv, args.acpp_source if name == "acpp-git" else root)
    capture("clang-cc1-dryrun", [*base, "-###", "-c", root / "test/pcuda_stage0/q33_probe.cpp", "-o", out / "probe.o"])
    capture("acpp-link-dryrun", [*base, "--acpp-dryrun", "-shared", "-fuse-ld=lld", "-Wl,-O0", out / "probe.o", "-o", out / "probe.so"])
    capture("link-cc1-dryrun", [*base, "-###", "-shared", "-fuse-ld=lld", "-Wl,-O0", out / "probe.o", "-o", out / "probe.so"])
    snapshot = {}
    files = [Path(acpp), Path(compiler).resolve(), Path(compiler).with_name("ld.lld").resolve()]
    prefix = Path(acpp).resolve().parents[1]
    files += list((prefix / "etc/AdaptiveCpp").glob("*"))
    files += list((prefix / "lib").rglob("*.so"))
    files += list((prefix / "lib").rglob("*.bc"))
    files += [p for p in (prefix / "include/AdaptiveCpp").rglob("*") if p.suffix in (".hpp", ".h")]
    for name in ("librt-backend-hip.so", "llvm-to-backend/libllvm-to-amdgpu.so"):
        library = prefix / "lib/hipSYCL" / name
        if library.exists():
            text = capture("dependencies-" + library.stem, ["ldd", library])
            for line in text.splitlines():
                if " => /" in line:
                    files.append(Path(line.split(" => ", 1)[1].split(" (", 1)[0]))
    for path in files:
        if path.is_file():
            snapshot[str(path)] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    shutil.copy2(acpp, out / "acpp-driver.py")
    docs = out / "docs"
    docs.mkdir()
    for path in ("doc/pcuda.md", "doc/performance.md", "doc/compilation.md", "doc/env_variables.md",
                 "src/compiler/AdaptiveCppLlvmPasses.cpp", "src/compiler/sscp/TargetSeparationPass.cpp",
                 "src/compiler/llvm-to-backend/LLVMToBackend.cpp", "src/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.cpp",
                 "src/compiler/llvm-to-backend/CMakeLists.txt", "src/compiler/llvm-to-backend/Utils.cpp"):
        src = args.acpp_source / path
        dst = docs / path.replace("/", "__")
        shutil.copy2(src, dst)
        snapshot[str(src)] = {"sha256": sha256(src), "saved": str(dst)}
    urls = {
        "clang-users-manual.html": "https://releases.llvm.org/17.0.1/tools/clang/docs/UsersManual.html",
        "clang-reference.html": "https://releases.llvm.org/17.0.1/tools/clang/docs/ClangCommandLineReference.html",
        "clang-pch.html": "https://releases.llvm.org/17.0.1/tools/clang/docs/PCHInternals.html",
        "lld-options.txt": "https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-17.0.0/lld/ELF/Options.td",
    }
    downloads = []
    for name, url in urls.items():
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                (docs / name).write_bytes(response.read())
            downloads.append({"url": url, "path": str(docs / name), "sha256": sha256(docs / name)})
        except Exception as error:
            downloads.append({"url": url, "error": str(error)})
    env = {k: v for k, v in os.environ.items() if k.startswith(("ACPP_", "HIPSYCL_", "ROCM_", "HIP_", "OMP_"))
           or k in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "LC_ALL", "LANG", "CXXFLAGS", "LDFLAGS")}
    report = {"commands": records, "hashes": snapshot, "environment": env, "documentation": downloads,
              "effective_compile_argv": shlex.split(effective), "tools_available":
              {name: shutil.which(name) for name in ("mold", "ld.gold", "ld.bfd", "strace", "cmake")}}
    (out / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
