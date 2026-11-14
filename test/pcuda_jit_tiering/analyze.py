#!/usr/bin/env python3
"""Offline export; never changes raw reports, never runs during measurements."""
import argparse
import json
from pathlib import Path
import tempfile
from run import sha, stats

def load_native(prefix):
    path = Path(str(prefix) + ".time.json")
    return {k.replace("_s", "_ms"): v * 1000 for k, v in json.loads(path.read_text()).items()} if path.is_file() else None

def trace(path):
    names = ("ExecuteCompiler", "Total Frontend", "Total Backend", "Total Optimizer", "ExecuteLinker")
    events = json.loads(path.read_text())["traceEvents"]
    # These are nested intervals, exported separately, never added to each other.
    return {name: max(e.get("dur", 0) for e in events if e.get("name") == name) / 1000
            for name in names if any(e.get("name") == name for e in events)}

def summarize(report):
    sessions = report.get("sessions", [])
    keys = sorted({(s["name"], s.get("mode", "q33"), s["traced"]) for s in sessions})
    groups = []
    for name, mode, traced in keys:
        all_group = [s for s in sessions if (s["name"], s.get("mode", "q33"), s["traced"]) == (name, mode, traced)]
        group = [s for s in all_group if s.get("verified")]
        metrics = {}
        for s in group:
            directory = Path(s["directory"])
            execution = s["execution"]
            sample = {k: v for k, v in execution.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            sample["process_start_to_first_result_ms"] = s["first_result"]["process_start_to_first_result_ms"]
            sample["process_native_wall_ms"] = load_native(directory / "process")["wall_ms"]
            if mode == "q33":
                for key in ("execute_ms", "fetch_ms", "total_ms"):
                    sample["first_" + key] = execution["runs"][0][key]
                sample["summary_coverage_gap_ms"] = execution["summary_ms"] - execution["preparation_ms"] - execution["build_elapsed_ms"] - execution["load_ms"] - execution["runs"][0]["total_ms"]
            for phase in ("compile", "link"):
                for part in ("wall_ms", "user_ms", "system_ms"):
                    prefixes = list(directory.glob("*-" + phase + ".time.json"))
                    for tier in ("all", "low", "high") if mode != "q33" else ("all",):
                        selected = [p for p in prefixes if tier == "all" or p.name.startswith(tier + "-")]
                        if selected:
                            sample[tier + "_" + phase + "_sum_" + part] = sum(load_native(str(p)[:-10])[part] for p in selected)
            jit = s.get("jit", {})
            for kind in ("optimizer", "external_tool", "full_transformation", "hiprtc_complete"):
                events = [e for e in jit.get("events", []) if e["kind"] == kind]
                if events: sample[kind + "_sum_ms"] = sum(e["end_ms"] - e["start_ms"] for e in events)
            for part in ("wall_ms", "user_ms", "system_ms"):
                native_tools = [load_native(e["prefix"]) for e in jit.get("events", []) if e["kind"] == "external_tool" and e.get("prefix")]
                if native_tools: sample["s2_native_tools_sum_" + part] = sum(t[part] for t in native_tools if t)
            images = [e["output_bytes"] for e in jit.get("events", []) if e["kind"] == "full_transformation" and e.get("success")]
            if images: sample["device_image_bytes_sum"] = sum(images)
            sample["module_bytes_sum"] = sum(p.stat().st_size for p in directory.glob("*.so"))
            for key, value in sample.items():
                metrics.setdefault(key, []).append(value)
        entry = {"name": name, "mode": mode, "traced": traced, "attempted": len(all_group), "verified": len(group),
                 "metrics": {key: stats(values) for key, values in metrics.items()}, "runs": {}}
        for tier in sorted({r.get("tier", "warm") for s in group for r in s["execution"]["runs"]}):
            selected = [r for s in group for r in (s["execution"]["runs"][1:] if mode == "q33" else s["execution"]["runs"]) if r.get("tier", "warm") == tier]
            entry["runs"][tier] = {key: stats([r[key] for r in selected]) for key in ("execute_ms", "fetch_ms", "total_ms")}
        entry["traces"] = {str(p): trace(p) for s in group for p in Path(s["directory"]).glob("*.json")
                           if p.name.endswith(("-compile.json", "-link.json"))}
        groups.append(entry)
    return {"raw_success": report.get("success"), "sessions": len(sessions), "verified": sum(s.get("verified", False) for s in sessions),
            "verified_results": sum(len(s["execution"]["runs"]) for s in sessions if s.get("verified")),
            "oracle_elements": sum(s.get("oracle", {}).get("checked_elements", 0) for s in sessions), "groups": groups}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("build/pcuda-jit-tiering"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="analysis-", dir=args.output_root.resolve()))
    result = {}
    for path in args.reports:
        item = summarize(json.loads(path.read_text()))
        item["input_sha256"] = sha(path)
        result[str(path)] = item
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = []
    for name, report in result.items():
        lines.append(name)
        for group in report["groups"]:
            text = f"{group['name']} {group['mode']} {'trace' if group['traced'] else 'plain'} n={group['verified']}/{group['attempted']}"
            for key, metric in group["metrics"].items():
                text += f"\n  {key}: {metric['median']:.6f} [{metric['min']:.6f}, {metric['max']:.6f}]"
            for tier, values in group["runs"].items():
                for key, metric in values.items():
                    if metric: text += f"\n  {tier} {key}: {metric['median']:.6f} [{metric['min']:.6f}, {metric['max']:.6f}] n={metric['n']}"
            lines.append(text)
    (out / "tables.txt").write_text("\n".join(lines) + "\n")
    print(out)

if __name__ == "__main__": main()
