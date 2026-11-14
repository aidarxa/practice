#!/usr/bin/env python3
"""Offline ELF/HCF diagnostics for the installed AdaptiveCpp format, not an engine API."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def command(argv, prefix):
    result = subprocess.run(argv, text=True, capture_output=True)
    prefix.with_suffix(".stdout").write_text(result.stdout)
    prefix.with_suffix(".stderr").write_text(result.stderr)
    prefix.with_suffix(".command.json").write_text(json.dumps({"argv": argv, "returncode": result.returncode}, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"Diagnostic command failed: {argv}; inspect {prefix}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effective", action="store_true", help="Record representative compile/link dry-runs (no actual rebuild)")
    parser.add_argument("--abi",type=int,choices=(1,2),default=1,help="Explicit native ABI; retain v1 inspection for historical artifacts")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for index, directory in enumerate(args.directories):
        output = args.output / str(index)
        output.mkdir()
        ids, names = {}, {}
        records = []
        for number, library in enumerate(sorted(directory.glob("db-pcuda-*/*/*.so"))):
            content = library.read_bytes()
            hcf = re.findall(rb"object-id\s*=\s*([0-9]+)\n", content)
            kernels = re.findall(rb"\{\.(_Z20__pcuda_parallel_for[^\n]+)\n", content)
            if len(hcf) != 1 or len(kernels) != 1:
                raise RuntimeError(f"Expected exactly one HCF and one native kernel in {library}")
            identity = library.stem
            if not re.fullmatch(r"[0-9a-f]{64}", identity):
                raise RuntimeError(f"Invalid structural module identity: {library}")
            hcf_id, kernel = hcf[0].decode(), kernels[0].decode()
            if identity not in kernel:
                raise RuntimeError(f"Kernel namespace does not contain its identity: {library}")
            for key, seen in ((hcf_id, ids), (kernel, names)):
                if key in seen and seen[key] != identity:
                    raise RuntimeError(f"Registration conflict between {seen[key]} and {identity}")
                seen[key] = identity
            symbols = command(["nm", "-D", "--defined-only", str(library.resolve())], output / f"module-{number}-symbols")
            if not all(re.search(r"\b" + symbol + r"$", symbols, re.MULTILINE) for symbol in
                       (f"db_kernel_identity_v{args.abi}", f"db_kernel_descriptor_v{args.abi}", f"db_kernel_launch_v{args.abi}")):
                raise RuntimeError(f"Missing fixed C entry in {library}")
            records.append({"path": str(library.resolve()), "identity": identity, "hcf_id": hcf_id,
                            "kernel": kernel, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
            if args.effective and index == 0:
                base = library.with_suffix("")
                compile_argv = shlex.split(Path(str(base) + ".compile.command").read_text())
                link_argv = shlex.split(Path(str(base) + ".link.command").read_text())
                command([*compile_argv, "--acpp-dryrun"], output / f"module-{number}-compile-effective")
                command([*link_argv, "--acpp-dryrun"], output / f"module-{number}-link-driver")
                command([*link_argv, "-###"], output / f"module-{number}-lld-effective")
        if not records:
            raise RuntimeError(f"No SQL module artifacts in {directory}")
        reports.append({"directory": str(directory.resolve()), "modules": records,
                        "unique_hcf_ids": len(ids), "unique_kernel_names": len(names)})
    (args.output / "report.json").write_text(json.dumps({"success": True, "scopes": reports,
                                                       "format": "installed AdaptiveCpp HCF text in ELF; identical identities may repeat, different identities must never collide"}, indent=2) + "\n")
    print(f"Verified {sum(len(report['modules']) for report in reports)} modules across {len(reports)} process scopes")


if __name__ == "__main__":
    main()
