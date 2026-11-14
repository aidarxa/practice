#!/usr/bin/env python3
"""Real CMake rebuild/no-op checks in private fixtures, with the old rule as a negative control."""
import argparse
import json
from pathlib import Path
import shutil
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def run(argv, directory, name):
    with (directory / (name + ".stdout")).open("w") as out, (directory / (name + ".stderr")).open("w") as err:
        result = subprocess.run(argv, stdout=out, stderr=err)
    (directory / (name + ".command.json")).write_text(json.dumps({"argv": argv, "returncode": result.returncode}, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"CMake fixture failed: {directory}/{name}")


def check(directory, compiler, cxx, flags, old_rule):
    directory.mkdir(parents=True)
    (directory / "cmake").mkdir()
    (directory / "tools").mkdir()
    shutil.copyfile(ROOT / "cmake/pcuda_jit.cmake", directory / "cmake/pcuda_jit.cmake")
    shutil.copyfile(ROOT / "cmake/install_pcuda_jit.cmake.in", directory / "cmake/install_pcuda_jit.cmake.in")
    writer = (ROOT / "tools/pcuda_pch_manifest.py").read_text()
    if old_rule:
        writer = writer.replace("    write_depfile(args.depfile, pch, files)\n", "")
    (directory / "tools/pcuda_pch_manifest.py").write_text(writer)
    (directory / "dummy.cpp").write_text("int fixture;\n")
    (directory / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.20)\nproject(PchDependencyFixture LANGUAGES CXX)\n'
        'add_library(pcuda_modules STATIC dummy.cpp)\n'
        f'set(CRYSTAL_INCLUDE_DIR "{ROOT / "include"}")\ninclude(cmake/pcuda_jit.cmake)\n')
    sdk = directory / "sdk"
    (sdk / "bin").mkdir(parents=True)
    (sdk / "etc/AdaptiveCpp").mkdir(parents=True)
    proxy = sdk / "bin/acpp"
    proxy.write_text("#!/bin/bash\nexec " + shlex.quote(compiler) + ' "$@"\n')
    proxy.chmod(0o700)
    observed = sdk / "etc/AdaptiveCpp/fixture.json"
    observed.write_text("old driver identity\n")
    build = directory / "build"
    run(["cmake", "-S", str(directory), "-B", str(build), f"-DCMAKE_CXX_COMPILER={cxx}",
         f"-DCMAKE_CXX_FLAGS={flags}", f"-DCRYSTAL_PCUDA_FLAGS={flags}", f"-DCRYSTAL_PCUDA_COMPILER={proxy}"], directory, "configure")
    argv = ["cmake", "--build", str(build), "--target", "pcuda_jit_pch"]
    run(argv, directory, "initial-build")
    manifest = build / "pcuda-jit/toolchain.manifest"
    assert str(observed) in manifest.read_text() # actually discovered, never manually inserted
    run(argv, directory, "discover-input")
    pch = build / "pcuda-jit/jit_headers.pch"
    before = pch.stat().st_mtime_ns
    observed.write_text("updated driver identity\n")
    run(argv, directory, "changed-input")
    after = pch.stat().st_mtime_ns
    rebuilt = after != before
    run(argv, directory, "no-op-build")
    unchanged = pch.stat().st_mtime_ns == after
    return {"old_rule": old_rule, "rebuilt_after_changed_input": rebuilt, "no_op_kept_pch": unchanged,
            "success": rebuilt != old_rule and unchanged}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--flags", default="")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = [check((args.output / name).resolve(), args.compiler, args.cxx, args.flags, old)
               for name, old in (("old-negative-control", True), ("manifest-dependencies", False))]
    result = {"success": all(record["success"] for record in records), "records": records,
              "method": "mtime detects rebuild/no-op only, not a process or performance timer; no source/SDK input is changed"}
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
