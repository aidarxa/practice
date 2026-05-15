#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import pathlib
import re
import subprocess
from statistics import mean
from typing import Optional

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

MARKER_PREFIX = '__CRYSTAL_BENCH_BEGIN__'
MARKER_RE = re.compile(rf'{re.escape(MARKER_PREFIX)}\|(\d+)\|([^|]+)\|([^|]+)\|(\d+)')


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


def has_sql_error(text: str) -> bool:
    return '[ERROR]' in text or 'Fatal error:' in text


def has_process_crash_marker(text: str) -> bool:
    crash_markers = (
        'HSA_STATUS',
        'aborting with error',
        'AdaptiveCpp error report',
        'hipMalloc() failed',
        'terminate called',
    )
    return any(marker in text for marker in crash_markers)


def iter_queries(manifest: dict, include_future: bool, include_heavy: bool, classes: Optional[set]):
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


def sql_command(sql_text: str) -> str:
    sql = sql_text.strip()
    if not sql.endswith(';'):
        sql += ';'
    return sql


def run_one(db_cli: str, sql_text: str, limit: str, timing: bool, timeout: int, is_dump: bool = False) -> subprocess.CompletedProcess:
    commands = []
    commands.append('\\timing on' if timing else '\\timing off')
    if limit:
        commands.append(f'\\limit {limit}')
    if is_dump:
        commands.append('\\dump on')
    commands.append(sql_command(sql_text))
    if is_dump:
        commands.append('\\dump off')
    commands.append('\\q')
    stdin = '\n'.join(commands) + '\n'
    return subprocess.run(
        [db_cli],
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def build_run_items(selected, root: pathlib.Path, warmups: int, runs: int, dump_code: bool):
    items = []
    for cls, q in selected:
        sql_path = root / q['path']
        sql_text = sql_path.read_text()
        
        if dump_code:
            items.append({
                'class_id': cls['id'],
                'query_id': q['id'],
                'path': q['path'],
                'status': q['status'],
                'phase': 'dump',
                'run_index': 0,
                'measured': False,
                'timeout': int(q.get('timeout_sec', 120)),
                'sql': sql_text,
            })

        total_runs = warmups + runs
        for run_idx in range(total_runs):
            measured = run_idx >= warmups
            items.append({
                'class_id': cls['id'],
                'query_id': q['id'],
                'path': q['path'],
                'status': q['status'],
                'phase': 'run' if measured else 'warmup',
                'run_index': run_idx,
                'measured': measured,
                'timeout': int(q.get('timeout_sec', 120)),
                'sql': sql_text,
            })
    return items


def split_session_output(text: str) -> dict:
    matches = list(MARKER_RE.finditer(text))
    segments = {}
    for pos, match in enumerate(matches):
        idx = int(match.group(1))
        start = match.end()
        end = matches[pos + 1].start() if pos + 1 < len(matches) else len(text)
        segments[idx] = text[start:end]
    return segments


def run_single_session(db_cli: str, run_items: list, limit: str, timing: bool, timeout_pad: int) -> tuple[list, int, bool, str]:
    commands = ['\\timing on' if timing else '\\timing off']
    if limit:
        commands.append(f'\\limit {limit}')
    for idx, item in enumerate(run_items):
        commands.append(f"\\echo {MARKER_PREFIX}|{idx}|{item['query_id']}|{item['phase']}|{item['run_index']}")
        if item['phase'] == 'dump':
            commands.append('\\dump on')
            commands.append(sql_command(item['sql']))
            commands.append('\\dump off')
        else:
            commands.append(sql_command(item['sql']))
    commands.append('\\q')
    stdin = '\n'.join(commands) + '\n'
    timeout = sum(item['timeout'] for item in run_items) + timeout_pad
    try:
        proc = subprocess.run(
            [db_cli],
            input=stdin,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        combined = proc.stdout or ''
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        combined = exc.stdout or ''
        if isinstance(combined, bytes):
            combined = combined.decode(errors='replace')
        returncode = 124
        timed_out = True

    segments = split_session_output(combined)
    records = []
    last_segment_index = max(segments.keys(), default=-1)
    for idx, item in enumerate(run_items):
        segment = segments.get(idx, '')
        metrics = parse_metrics(segment)
        segment_error = has_sql_error(segment)
        segment_crash = has_process_crash_marker(segment)
        segment_missing = idx not in segments
        item_returncode = 0
        item_timed_out = False
        if timed_out and (segment_missing or idx == last_segment_index):
            item_returncode = 124
            item_timed_out = True
        elif returncode != 0 and (segment_crash or segment_missing or idx == last_segment_index):
            item_returncode = returncode
        elif segment_error:
            item_returncode = 1
        record = dict(item)
        del record['sql']
        record.update({
            '_session_index': idx,
            'returncode': item_returncode,
            'timed_out': item_timed_out,
            '_raw_output': segment,
            **metrics,
        })
        records.append(record)
    return records, returncode, timed_out, combined


def run_isolated(db_cli: str, run_items: list, limit: str, timing: bool, logs_dir: pathlib.Path,
                 fail_on_error: bool, codes_dir: Optional[pathlib.Path] = None) -> tuple[list, int]:
    records = []
    failures = 0
    for item in run_items:
        try:
            proc = run_one(db_cli, item['sql'], limit, timing, item['timeout'], is_dump=item['phase'] == 'dump')
            timed_out = False
            stdout = proc.stdout or ''
            stderr = proc.stderr or ''
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ''
            stderr = exc.stderr or ''
            returncode = 124
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors='replace')
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors='replace')
        log_base = f"{item['query_id']}__{item['phase']}_{item['run_index']}"
        (logs_dir / f'{log_base}.out').write_text(stdout)
        (logs_dir / f'{log_base}.err').write_text(stderr)
        
        if item['phase'] == 'dump' and codes_dir:
            code_match = re.search(r'---\s*Generated Code\s*---\n(.*?)---\s*End\s*---', stdout, re.DOTALL)
            if code_match:
                (codes_dir / f"{item['query_id']}.cpp").write_text(code_match.group(1).strip() + '\n')
                
        combined_output = stdout + '\n' + stderr
        metrics = parse_metrics(combined_output)
        record_returncode = returncode
        if record_returncode == 0 and has_sql_error(combined_output):
            record_returncode = 1
        if record_returncode != 0 or timed_out or has_sql_error(combined_output):
            failures += 1
        record = dict(item)
        del record['sql']
        record.update({
            'returncode': record_returncode,
            'timed_out': timed_out,
            '_raw_output': stdout,
            **metrics,
        })
        records.append(record)
        print(f"[{item['phase']}] {item['query_id']}: rc={record_returncode} timeout={timed_out} rows={metrics.get('rows_returned')} gpu_ms={metrics.get('gpu_execution_ms')}")
        if fail_on_error and (record_returncode != 0 or timed_out):
            break
    return records, failures


def resolve_manifest_path(path_text: str) -> pathlib.Path:
    path = pathlib.Path(path_text)
    if path.exists():
        return path
    fallback = pathlib.Path('olap_bench/manifest.json')
    if path_text == 'bench/olap_bench/manifest.json' and fallback.exists():
        return fallback
    return path


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
    ap.add_argument('--mode', choices=['single-session', 'isolated'], default='single-session')
    ap.add_argument('--session-timeout-pad', type=int, default=300)
    ap.add_argument('--dump-code', action='store_true', default=True, help='Save generated code for each query (default: True)')
    ap.add_argument('--no-dump-code', dest='dump_code', action='store_false', help='Do not save generated code')
    args = ap.parse_args()

    manifest_path = resolve_manifest_path(args.manifest)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    out_dir = pathlib.Path(args.out) if args.out else root / 'results' / dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_dir = out_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    codes_dir = out_dir / 'codes'
    if args.dump_code:
        codes_dir.mkdir(parents=True, exist_ok=True)
    class_filter = set(args.classes) if args.classes else None

    selected = list(iter_queries(manifest, args.include_future, args.include_heavy, class_filter))
    run_items = build_run_items(selected, root, args.warmups, args.runs, args.dump_code)

    if args.mode == 'single-session':
        all_records, proc_rc, timed_out, combined = run_single_session(
            args.db_cli, run_items, args.limit, args.timing, args.session_timeout_pad)
        (logs_dir / 'single_session.out').write_text(combined)
        failures = 0
        segments = split_session_output(combined)
        for record in all_records:
            idx = record.get('_session_index', -1)
            segment = segments.get(idx, '')
            log_base = f"{record['query_id']}__{record['phase']}_{record['run_index']}"
            (logs_dir / f'{log_base}.out').write_text(segment)
            (logs_dir / f'{log_base}.err').write_text('')
            
            if record['phase'] == 'dump' and args.dump_code:
                code_match = re.search(r'---\s*Generated Code\s*---\n(.*?)---\s*End\s*---', segment, re.DOTALL)
                if code_match:
                    (codes_dir / f"{record['query_id']}.cpp").write_text(code_match.group(1).strip() + '\n')

            if record['returncode'] != 0 or record['timed_out']:
                failures += 1
            print(f"[{record['phase']}] {record['query_id']}: rc={record['returncode']} timeout={record['timed_out']} rows={record.get('rows_returned')} gpu_ms={record.get('gpu_execution_ms')}")
        if proc_rc != 0 and failures == 0:
            failures = 1
        if timed_out and failures == 0:
            failures = 1
    else:
        all_records, failures = run_isolated(args.db_cli, run_items, args.limit, args.timing, logs_dir, args.fail_on_error, codes_dir if args.dump_code else None)

    warmup_metrics = {}
    for r in all_records:
        if r.get('phase') == 'warmup':
            key = r['query_id']
            if key not in warmup_metrics:
                warmup_metrics[key] = {
                    'code_generation_ms': r.get('code_generation_ms', 0),
                    'acpp_compilation_ms': r.get('acpp_compilation_ms', 0)
                }

    rows = []
    for r in all_records:
        if r.get('measured'):
            key = r['query_id']
            if key in warmup_metrics:
                if not r.get('code_generation_ms'):
                    r['code_generation_ms'] = warmup_metrics[key].get('code_generation_ms', 0)
                if not r.get('acpp_compilation_ms'):
                    r['acpp_compilation_ms'] = warmup_metrics[key].get('acpp_compilation_ms', 0)
            rows.append(r)

    aggregates = {}
    for r in rows:
        key = r['query_id']
        aggregates.setdefault(key, []).append(r)

    outputs_dir = out_dir / 'outputs'
    outputs_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for key, vals in sorted(aggregates.items()):
        last_run = vals[-1]
        if '_raw_output' in last_run:
            (outputs_dir / f"{key}.out").write_text(last_run['_raw_output'])

        successful_vals = [v for v in vals if v.get('returncode') == 0 and not v.get('timed_out')]
        last_success = successful_vals[-1] if successful_vals else None
        gpu_vals = [v['gpu_execution_ms'] for v in successful_vals if isinstance(v.get('gpu_execution_ms'), (int, float))]
        fetch_vals = [v['result_materialization_fetch_ms'] for v in successful_vals if isinstance(v.get('result_materialization_fetch_ms'), (int, float))]
        summary.append({
            'query_id': key,
            'runs': len(vals),
            'successful_runs': len(successful_vals),
            'failed_runs': len(vals) - len(successful_vals),
            'last_returncode': last_run.get('returncode'),
            'last_timed_out': last_run.get('timed_out'),
            'gpu_execution_ms_mean': mean(gpu_vals) if gpu_vals else None,
            'result_materialization_fetch_ms_mean': mean(fetch_vals) if fetch_vals else None,
            'rows_returned_last': last_success.get('rows_returned') if last_success else None,
        })

    for r in rows:
        r.pop('_raw_output', None)

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
    (out_dir / 'summary.json').write_text(json.dumps({'mode': args.mode, 'rows': rows}, indent=2))


    (out_dir / 'aggregate_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Wrote results to {out_dir}')
    return 1 if failures and args.fail_on_error else 0


if __name__ == '__main__':
    raise SystemExit(main())
