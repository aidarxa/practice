#!/usr/bin/env python3
"""Offline hashes, dependencies and effective commands for completed series."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from run import ROOT, STAGE, sha

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("build/pcuda-jit-tiering"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="audit-", dir=args.output_root.resolve()))
    checks = {"reports": {}, "data_sha256": {}, "effective_commands": []}
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD" and not k.startswith("JIT_POC_")}
    for index, path in enumerate(args.reports):
        report = json.loads(path.read_text())
        base = path.parent.resolve()
        snapshots = base / "sources"
        source_checks = {}
        for source, expected in report.get("source_sha256", {}).items():
            original = Path(source)
            snapshot = snapshots / (original.parent.name + "-" + original.name)
            source_checks[source] = snapshot.is_file() and sha(snapshot) == expected
        artifact_hashes = {str(p): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in base.iterdir() if p.is_file() and p.suffix in (".pch", ".o", ".so")}
        for name in ("q33", "tiered", "first_result"):
            p = base / name
            if p.is_file(): artifact_hashes[str(p)] = {"sha256": sha(p), "bytes": p.stat().st_size}
        sessions = []
        representatives = set()
        for session in report.get("sessions", []):
            directory = Path(session["directory"])
            modules = list(directory.glob("*.so"))
            ids = [value.decode() for p in modules for value in re.findall(rb"object-id=([0-9]+)\n", p.read_bytes())]
            dependencies = list(directory.glob("*.d"))
            item = {"directory": str(directory), "returncode": session["returncode"],
                "hcf_ids": ids, "hcf_unique": len(ids) == len(set(ids)) == len(modules),
                "no_full_sycl": all("sycl/sycl.hpp" not in p.read_text() for p in dependencies),
                "modules": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in modules}}
            if session.get("verified") and "jit" in session:
                expected = (1 if session.get("mode") == "failure" else 2) if "mode" in session else 4
                item["jit_count_matches"] = session["jit"]["compilations"] == expected
            if session.get("verified") and "mode" not in session:
                e = session["execution"]
                item["summary_gap_ms"] = e["summary_ms"] - e["preparation_ms"] - e["build_elapsed_ms"] - e["load_ms"] - e["runs"][0]["total_ms"]
            sessions.append(item)
            # Stage-1 settings are the same for every runtime profile. Recover
            # one full set per diagnostic mode without rerunning compilation.
            key = session["traced"]
            if key in representatives or not session.get("verified"): continue
            representatives.add(key)
            command_files = list(directory.glob("*commands.json"))
            for command_file in command_files:
                for command in json.loads(command_file.read_text()):
                    label = f"r{index}-{int(key)}-{command.get('module', command.get('tier'))}-{command['phase']}"
                    result = subprocess.run([*command["argv"], "--acpp-dryrun"], cwd=ROOT,
                        env=env, capture_output=True, text=True, timeout=30)
                    (out / (label + ".txt")).write_text(result.stdout + result.stderr)
                    checks["effective_commands"].append({"argv": [*command["argv"], "--acpp-dryrun"],
                        "returncode": result.returncode, "output": label + ".txt"})
        checks["reports"][str(path)] = {"report_sha256": sha(path), "source_snapshots": source_checks,
            "sources_unchanged_during_measurements": report.get("sources_unchanged"),
            "control_snapshot_matches": sha(base / "libjit_control.so") == report.get("control_sha256"),
            "artifacts": artifact_hashes, "sessions": sessions}
    for name in ("DDATE0", "DDATE4", "CUSTOMER0", "CUSTOMER3", "SUPPLIER0", "SUPPLIER3", "LINEORDER5", "LINEORDER2", "LINEORDER4", "LINEORDER12"):
        p = args.data / name
        checks["data_sha256"][str(p)] = sha(p)
    checks["oracle_sha256"] = sha(ROOT / "test/queries/result/scale10/33.txt")
    checks["success"] = all(all(r["source_snapshots"].values()) and r["control_snapshot_matches"]
        and r["sources_unchanged_during_measurements"] and all(s["hcf_unique"] and s["no_full_sycl"]
            and s.get("jit_count_matches", True) and s.get("summary_gap_ms", 0) >= -0.01 for s in r["sessions"])
        for r in checks["reports"].values()) and all(c["returncode"] == 0 for c in checks["effective_commands"])
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"Audit {checks['success']}: {out}")

if __name__ == "__main__": main()
