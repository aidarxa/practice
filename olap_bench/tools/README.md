# Benchmark tools

- `run_olap_bench.py`: benchmark runner. Default mode is `single-session`: one `db_cli` process for the whole selected workload. Use `--mode isolated` only when intentionally measuring cold process startup/JIT cache behavior.
- `run_session_smoke.py`: one-process sequential smoke runner. Use for state-leak and scratch-buffer lifetime testing.

Both tools use only Python standard library.
