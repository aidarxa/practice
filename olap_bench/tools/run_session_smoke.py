#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess


def iter_queries(manifest, include_future, include_heavy):
    allowed = {'current'}
    if include_future:
        allowed.add('future')
    if include_heavy:
        allowed.add('current_heavy')
    for cls in manifest['classes']:
        for q in cls['queries']:
            if q['status'] in allowed:
                yield q


def main():
    ap = argparse.ArgumentParser(description='Run many SQL files in one db_cli session to detect state leakage.')
    ap.add_argument('--db-cli', default='./build/db_cli')
    ap.add_argument('--manifest', default='bench/olap_bench/manifest.json')
    ap.add_argument('--out', default='bench/olap_bench/results/session_smoke.txt')
    ap.add_argument('--limit', default='20')
    ap.add_argument('--timeout', type=int, default=600)
    ap.add_argument('--include-future', action='store_true')
    ap.add_argument('--include-heavy', action='store_true')
    args = ap.parse_args()

    manifest_path = pathlib.Path(args.manifest)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    commands = ['\\timing on', f'\\limit {args.limit}']
    for q in iter_queries(manifest, args.include_future, args.include_heavy):
        sql = (root / q['path']).read_text().strip()
        commands.append(sql)
    stdin = '\n'.join(commands) + '\n'
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run([args.db_cli], input=stdin, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=args.timeout)
        out_path.write_text(proc.stdout + '\n--- STDERR ---\n' + proc.stderr)
        print(f'rc={proc.returncode}; transcript={out_path}')
        return proc.returncode
    except subprocess.TimeoutExpired as exc:
        out_path.write_text((exc.stdout or '') + '\n--- STDERR ---\n' + (exc.stderr or '') + '\n--- TIMEOUT ---\n')
        print(f'timeout after {args.timeout}s; transcript={out_path}')
        return 124


if __name__ == '__main__':
    raise SystemExit(main())
