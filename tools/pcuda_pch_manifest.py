#!/usr/bin/env python3
"""Record build-time PCH dependencies and effective acpp commands; never used on the SQL path."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess


def write_depfile(path, pch, files):
    """One dependency authority for headers and effective compiler/plugin/runtime inputs."""
    def escape(value):
        return str(value).replace("\\", "\\\\").replace("$", "$$").replace("#", "\\#").replace(" ", "\\ ").replace(":", "\\:")
    dependencies = " ".join(escape(file) for file in sorted(files) if file != pch)
    Path(path).write_text(escape(pch) + ": " + dependencies + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compiler", "pch", "depfile", "header", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--compile-arg", action="append", default=[])
    parser.add_argument("--link-arg", action="append", default=[])
    args = parser.parse_args()
    compiler = Path(shutil.which(args.compiler) or args.compiler).resolve()
    pch = Path(args.pch).resolve()
    dependency_text = Path(args.depfile).read_text().replace("\\\n", " ")
    dependencies = shlex.split(dependency_text.partition(":")[2])
    files = {Path(name).resolve() for name in dependencies}
    files.update((compiler, pch, Path(args.header).resolve()))
    dryrun = [str(compiler), *args.compile_arg, "--acpp-dryrun", "-c", args.header, "-o", str(pch) + ".probe.o"]
    effective = subprocess.run(dryrun, text=True, capture_output=True, check=True)
    Path(args.output + ".dryrun.json").write_text(json.dumps({"argv": dryrun, "stdout": effective.stdout, "stderr": effective.stderr}, indent=2))
    # acpp names the real compiler and registration plugins in its effective argv.
    # Also record configuration JSON and runtime libraries beside the selected SDK.
    effective_text = effective.stdout + "\n" + effective.stderr
    for item in shlex.split(effective_text):
        value = item.partition("=")[2] if "=" in item else item
        candidate = Path(value)
        if candidate.is_absolute() and candidate.is_file():
            files.add(candidate.resolve())
    sdk = compiler.parent.parent
    files.update(path.resolve() for path in sdk.glob("etc/AdaptiveCpp/*.json"))
    files.update(path.resolve() for path in sdk.glob("lib/libacpp*.so*"))
    files.update(path.resolve() for path in sdk.glob("lib/libhipSYCL*.so*"))
    records = ["DB-PCUDA-PCH-1", "compiler\t" + str(compiler), "pch\t" + str(pch), "header\t" + str(Path(args.header).resolve())]
    records += ["compile_arg\t" + flag for flag in args.compile_arg]
    records += ["link_arg\t" + flag for flag in args.link_arg]
    # Freeze only compiler-affecting overrides in child compile/link processes.
    # Runtime/device-JIT settings of db_cli remain untouched and are reported separately.
    driver_text = compiler.read_text()
    variables = set(re.findall(r'"(ACPP_[A-Z0-9_]+)"', driver_text))
    variables.update(name.replace("ACPP_", "HIPSYCL_", 1) for name in tuple(variables))
    variables.update(("PATH", "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX", "SDKROOT", "LD_LIBRARY_PATH", "SOURCE_DATE_EPOCH"))
    for name in sorted(variables):
        if name in os.environ:
            value = os.environ[name]
            if "\t" in value or "\n" in value:
                raise ValueError("Compiler environment values cannot contain tab/newline")
            records.append("compiler_env_set\t" + name + "\t" + value)
        else:
            records.append("compiler_env_unset\t" + name)
    for path in sorted(files):
        if "\n" in str(path) or "\t" in str(path):
            raise ValueError("Manifest paths cannot contain tab/newline")
        records.append("file\t" + hashlib.sha256(path.read_bytes()).hexdigest() + "\t" + str(path))
    Path(args.output).write_text("\n".join(records) + "\n")
    write_depfile(args.depfile, pch, files)


if __name__ == "__main__":
    main()
