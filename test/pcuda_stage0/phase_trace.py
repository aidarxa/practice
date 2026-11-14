"""Observe compiler/linker process intervals without rewriting a compiler command."""

import ast
from pathlib import Path
import re


def union_ms(intervals):
    end = None
    total = 0.0
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("negative process interval")
        if end is None or start >= end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = max(stop, end) if end is not None else stop
    return total * 1000


def parse_trace(text):
    running = {}
    pending_exec = {}
    processes = []
    for line in text.splitlines():
        match = re.match(r"(\d+)\s+(\d+\.\d+)\s+(.*)", line)
        if not match:
            continue
        pid, timestamp, message = match.groups()
        timestamp = float(timestamp)
        if message.startswith("execve(") and message.endswith("<unfinished ...>"):
            pending_exec[pid] = (timestamp, message.removesuffix("<unfinished ...>"))
            continue
        if message.startswith("<... execve resumed>"):
            if pid not in pending_exec:
                raise ValueError("resumed execve without its start")
            timestamp, prefix = pending_exec.pop(pid)
            message = prefix + message.removeprefix("<... execve resumed>")
        execute = re.match(r'execve\(("(?:[^"\\]|\\.)*").*\)\s+=\s+0$', message)
        if execute:
            path = ast.literal_eval(execute[1])
            if pid in running:
                old = running.pop(pid)
                processes.append({**old, "end": timestamp})
            running[pid] = {"pid": int(pid), "path": path, "start": timestamp}
        elif message.startswith("+++ exited with") or message.startswith("+++ killed by"):
            if pid in running:
                processes.append({**running.pop(pid), "end": timestamp})
    if running or pending_exec:
        raise ValueError("trace contains unfinished executable processes")
    compiler = [(p["start"], p["end"]) for p in processes
                if re.fullmatch(r"clang(?:\+\+)?(?:-\d+)?", Path(p["path"]).name)]
    linker = [(p["start"], p["end"]) for p in processes
              if Path(p["path"]).name in ("ld", "ld.lld", "ld.bfd", "ld.gold", "lld")]
    if not compiler:
        raise ValueError("no completed Clang process found")
    # Clang's driver includes the linker child. Subtract only their intersection.
    overlap = [(max(a, c), min(b, d)) for a, b in compiler for c, d in linker
               if max(a, c) < min(b, d)]
    return {"compiler_process_ms": union_ms(compiler) - union_ms(overlap),
            "linker_process_ms": union_ms(linker), "processes": processes,
            "method": "strace process intervals; nested Clang/linker intervals counted once",
            "tracing_overhead_included": True}
