# Benchmark tools

- `run_olap_bench.py`: benchmark runner. Default mode is `single-session`: one `db_cli` process for the whole selected workload. Use `--mode isolated` only when intentionally measuring cold process startup/JIT cache behavior.
- `run_session_smoke.py`: one-process sequential smoke runner. Use for state-leak and scratch-buffer lifetime testing.

Both tools use only Python standard library.

The native PCUDA engine (stage 5) requires `--timing-model pcuda-modules-v1`.
The runner preserves native compile/link wall/user/system sums, direct parallel
build elapsed and per-query codegen/compiler/link/load/symbol counters. Compatible
value changes reuse ready plans and modules; zero build counters, not elapsed time,
are the evidence. `--dump-code` writes separate translation units and `modules.json`
per query; they are inspection candidates, not one whole-query source to relink.
See [timing.md](../../timing.md#этап-5-pcuda-modules-v1--2026-11-07).

Historical stage-2 engine intervals are declared with `--timing-model explicit-plan-v1` in
`run_olap_bench.py`; the chosen model is recorded in CSV. Warm `codegen_ms` may
include repeated physical lowering, while `compile_ms` remains compile + link
and is zero on a compiled-query hit. Historical raw fields/gates are not weakened.
See [timing.md](../../timing.md#этап-2-explicit-plan-v1--2026-11-05).

Stage-0 migration measurements are separate from these SQL timing fields:
[native Bash time, Clang/lld traces, and baseline methodology](../../test/pcuda_stage0/REPORT.md#stage0-completion).
Python orchestrates and analyzes those experiments; no additional Python measurement wrapper runs inside compiler commands. The acpp driver's own interpreter overhead remains included.

[PERF-COLD](../../test/pcuda_compile_perf/REPORT.md) adds a separate PCUDA prototype
runner with direct cold `summary_ms`, build elapsed, process-to-first-result and
paired trace/plain series. It does not change this SQL runner's JSON schema or
implement PCUDA SQL execution. Replayed Clang/LLD command times are not acpp wall.

## Stage-8 additive timing observations

With `--timing-model pcuda-modules-v1`, raw CSV also exports `sql_to_pcuda_ms`,
`startup_ms`, `data_load_ms` and `main_to_first_result_ms`. Missing historical
fields remain unknown. These are overlapping observations, **not** new component
sum terms; startup fields occur only on the first successful extended-timing query.
The existing mixed mean/first aggregation is unchanged and is not a cold-median
acceptance table. For independent-process SSB plain/traced medians use
`test/pcuda_stage8/measure.py` and the boundaries in `timing.md`.
