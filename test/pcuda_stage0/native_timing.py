"""Native Bash time measurements and offline Clang/lld trace analysis."""

import json
from pathlib import Path


def timed_command(argv, timing_file, stderr_file):
    script = ("export LC_ALL=C\nTIMEFORMAT='{\"wall_s\":%6R,\"user_s\":%6U,\"system_s\":%6S}'\n"
              'timing=$1\nerrors=$2\nshift 2\n{ time "$@" 2>"$errors"; } 2>"$timing"\n')
    return ["bash", "-c", script, "stage0-native-time", str(timing_file), str(stderr_file),
            *(str(arg) for arg in argv)]


def read_time(path):
    result = json.loads(Path(path).read_text())
    return {key.replace("_s", "_ms"): value * 1000 for key, value in result.items()}


def trace_summary(path):
    events = json.loads(Path(path).read_text())["traceEvents"]
    # Summary fields are reported independently; never add them to their parents.
    names = ("ExecuteCompiler", "Total ExecuteCompiler", "Total Frontend", "Total Backend",
             "Total Optimizer", "Total CodeGenPasses", "Total ExecuteLinker", "ExecuteLinker")
    return {name: max(e.get("dur", 0) for e in events if e.get("name") == name) / 1000
            for name in names if any(e.get("name") == name for e in events)}


def collect_events(directory):
    from phase_trace import parse_trace
    result = []
    for path in sorted(Path(directory).glob("*.meta.json")):
        stem = path.name.removesuffix(".meta.json")
        prefix = path.parent / stem
        event = json.loads(path.read_text())
        event.update(read_time(str(prefix) + ".time.json"))
        for suffix, name in ((".argv", "argv"), (".original.argv", "original_argv")):
            event[name] = [arg.decode() for arg in Path(str(prefix) + suffix).read_bytes().split(b"\0") if arg]
        for suffix, name in ((".compile.json", "compiler_trace_ms"), (".link.json", "linker_trace_ms")):
            trace = Path(str(prefix) + suffix)
            if trace.exists():
                event[name] = trace_summary(trace)
        trace = Path(str(prefix) + ".trace")
        if trace.exists() and event["returncode"] == 0:
            event["phase_trace"] = parse_trace(trace.read_text())
        event["measurement"] = "Bash time; no Python measurement wrapper; acpp driver overhead retained"
        result.append(event)
    return result
