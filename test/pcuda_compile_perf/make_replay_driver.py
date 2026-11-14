#!/usr/bin/env python3
"""Prepare a version-bound Bash replay driver for correctness-only stage-0 tests."""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from run import STAGE, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acpp", default="acpp")
    parser.add_argument("--extra-flag", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acpp = shutil.which(args.acpp)
    if not acpp:
        parser.error("acpp not found")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = [acpp, "--acpp-targets=generic", "--acpp-pcuda", *args.extra_flag]
    compile_command = [*base, "--acpp-dryrun", "-c", str(STAGE / "q33_probe.cpp"), "-o", str(args.output.parent / "unused.o")]
    link_command = [*base, "--acpp-dryrun", "-shared", str(args.output.parent / "unused.o"), "-o", str(args.output.parent / "unused.so")]
    compile_text = subprocess.check_output(compile_command, text=True)
    link_text = subprocess.check_output(link_command, text=True)
    compile_argv = shlex.split(compile_text)[:-4]
    link_argv = shlex.split(link_text)
    # Remove exactly the probe's forwarded arguments, retain all runtime search/link flags.
    compiler = link_argv[0]
    for arg in ("-shared", str(args.output.parent / "unused.o"), "-o", str(args.output.parent / "unused.so")):
        link_argv.remove(arg)
    script = ["#!/usr/bin/env bash", "set -euo pipefail", "real=" + shlex.quote(acpp),
              "compile=(" + shlex.join(compile_argv) + ")", "link=(" + shlex.join(link_argv) + ")",
              "args=()", "source=0", 'for arg in "$@"; do',
              '  case "$arg" in',
              '    --acpp-version|--help|--version) exec "$real" "$@" ;;',
              '    --acpp-targets=generic|--acpp-pcuda) continue ;;',
              '    *.cpp|*.hpp|*.h|*.cxx|*.cc) source=1 ;;',
              '  esac', '  args+=("$arg")', 'done',
              'if (( source )); then exec "${compile[@]}" "${args[@]}"; fi',
              'exec "${link[@]}" "${args[@]}"']
    args.output.write_text("\n".join(script) + "\n")
    args.output.chmod(0o700)
    args.output.with_suffix(".json").write_text(json.dumps({"acpp_sha256": sha256(acpp), "clang_sha256": sha256(compiler),
        "compile_dryrun_argv": compile_command, "link_dryrun_argv": link_command,
        "compile_effective": compile_text, "link_effective": link_text, "driver_sha256": sha256(args.output),
        "scope": "correctness driver; direct Clang compile/link, NOT acpp native wall"}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
