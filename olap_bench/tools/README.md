# Benchmark tools

- `run_olap_bench.py`: isolated per-query benchmark runner. Use for stable performance measurement.
- `run_session_smoke.py`: one-process sequential runner. Use for state-leak and scratch-buffer lifetime testing.

Both tools use only Python standard library.
