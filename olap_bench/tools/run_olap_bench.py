#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys
from statistics import mean

METRIC_PATTERNS = {
    'rows_returned': re.compile(r'Rows returned:\s*(\d+)'),
    'rows_shown': re.compile(r'Rows shown:\s*(\d+)'),
    'gpu_execution_ms': re.compile(r'GPU execution time:\s*([0-9.]+)\s*ms'),
    'result_materialization_fetch_ms': re.compile(r'Result materialization/fetch time:\s*([0-9.]+)\s*ms'),
    'host_result_fetch_ms_legacy': re.compile(r'Host result fetch time:\s*([0-9.]+)\s*ms'),
    'code_generation_ms': re.compile(r'Code generation time:\s*([0-9.]+)\s*ms'),
    'acpp_compilation_ms': re.compile(r'ACPP compilation time:\s*([0-9.]+)\s*ms'),
    'library_load_ms': re.compile(r'Library load time:\s*([0-9.]+)\s*ms'),
    'library_load_old_ms': re.compile(r'Library load \+ execution start time:\s*([0-9.]+)\s*ms'),
    'engine_processing_ms': re.compile(r'Engine processing time:\s*([0-9.]+)\s*ms'),
}


def parse_metrics(text: str) -> dict:
    out = {}
    for key, pattern in METRIC_PATTERNS.items():
        m = pattern.search(text)
        if not m:
            continue
        value = m.group(1)
        out[key] = int(value) if value.isdigit() else float(value)
    if 'result_materialization_fetch_ms' not in out and 'host_result_fetch_ms_legacy' in out:
        out['result_materialization_fetch_ms'] = out['host_result_fetch_ms_legacy']
    if 'library_load_ms' not in out and 'library_load_old_ms' in out:
        out['library_load_ms'] = out['library_load_old_ms']
    return out


def iter_queries(manifest: dict, include_future: bool, include_heavy: bool, classes: set[str] | None):
    allowed = {'current'}
    if include_heavy:
        allowed.add('current_heavy')
    if include_future:
        allowed.add('future')
    for cls in manifest['classes']:
        if classes and cls['id'] not in classes and cls['name'] not in classes:
            continue
        for q in cls['queries']:
            if q['status'] in allowed:
                yield cls, q


def run_one(db_cli: str, sql_text: str, limit: str, timing: bool, timeout: int) -> subprocess.CompletedProcess:
    commands = []
    commands.append('\\timing on' if timing else '\\timing off')
    if limit:
        commands.append(f'\\limit {limit}')
    commands.append(sql_text.strip())
    stdin = '\n'.join(commands) + '\n'
    return subprocess.run(
        [db_cli],
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description='Run Crystal-SYCL OLAP benchmark SQL files through db_cli.')
    ap.add_argument('--db-cli', default='./build/db_cli')
    ap.add_argument('--manifest', default='bench/olap_bench/manifest.json')
    ap.add_argument('--out', default=None)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--warmups', type=int, default=1)
    ap.add_argument('--limit', default='1000')
    ap.add_argument('--timing', action='store_true', default=True)
    ap.add_argument('--include-future', action='store_true')
    ap.add_argument('--include-heavy', action='store_true')
    ap.add_argument('--class', dest='classes', action='append', default=[])
    ap.add_argument('--fail-on-error', action='store_true')
    args = ap.parse_args()

    manifest_path = pathlib.Path(args.manifest)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    out_dir = pathlib.Path(args.out) if args.out else root / 'results' / dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_dir = out_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    class_filter = set(args.classes) if args.classes else None

    rows = []
    failures = 0
    selected = list(iter_queries(manifest, args.include_future, args.include_heavy, class_filter))
    for cls, q in selected:
        sql_path = root / q['path']
        sql_text = sql_path.read_text()
        total_runs = args.warmups + args.runs
        for run_idx in range(total_runs):
            measured = run_idx >= args.warmups
            label = 'run' if measured else 'warmup'
            timeout = int(q.get('timeout_sec', 120))
            try:
                proc = run_one(args.db_cli, sql_text, args.limit, args.timing, timeout)
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                proc = None
                timed_out = True
                stdout = exc.stdout or ''
                stderr = exc.stderr or ''
            if proc is not None:
                stdout = proc.stdout
                stderr = proc.stderr
                returncode = proc.returncode
            else:
                returncode = 124
            log_base = f"{q['id']}__{label}_{run_idx}"
            (logs_dir / f'{log_base}.out').write_text(stdout)
            (logs_dir / f'{log_base}.err').write_text(stderr)
            metrics = parse_metrics(stdout + '\n' + stderr)
            if returncode != 0 or timed_out:
                failures += 1
            record = {
                'class_id': cls['id'],
                'query_id': q['id'],
                'path': q['path'],
                'status': q['status'],
                'phase': label,
                'run_index': run_idx,
                'measured': measured,
                'returncode': returncode,
                'timed_out': timed_out,
                **metrics,
            }
            if measured:
                rows.append(record)
            print(f"[{label}] {q['id']}: rc={returncode} timeout={timed_out} rows={metrics.get('rows_returned')} gpu_ms={metrics.get('gpu_execution_ms')}")
            if args.fail_on_error and (returncode != 0 or timed_out):
                break

    fields = [
        'class_id','query_id','path','status','phase','run_index','measured','returncode','timed_out',
        'rows_returned','rows_shown','gpu_execution_ms','result_materialization_fetch_ms',
        'code_generation_ms','acpp_compilation_ms','library_load_ms','engine_processing_ms'
    ]
    with (out_dir / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out_dir / 'summary.json').write_text(json.dumps({'rows': rows}, indent=2))

    aggregates = {}
    for r in rows:
        key = r['query_id']
        aggregates.setdefault(key, []).append(r)
    summary = []
    for key, vals in sorted(aggregates.items()):
        gpu_vals = [v['gpu_execution_ms'] for v in vals if isinstance(v.get('gpu_execution_ms'), (int, float))]
        fetch_vals = [v['result_materialization_fetch_ms'] for v in vals if isinstance(v.get('result_materialization_fetch_ms'), (int, float))]
        summary.append({
            'query_id': key,
            'runs': len(vals),
            'gpu_execution_ms_mean': mean(gpu_vals) if gpu_vals else None,
            'result_materialization_fetch_ms_mean': mean(fetch_vals) if fetch_vals else None,
            'rows_returned_last': vals[-1].get('rows_returned'),
        })
    (out_dir / 'aggregate_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Wrote results to {out_dir}')
    return 1 if failures and args.fail_on_error else 0


if __name__ == '__main__':
    raise SystemExit(main())
