# Crystal — PCUDA SQL engine

## Quick start

Run from the repository root on Linux with a GPU driver, PCUDA-capable AdaptiveCpp
(`acpp`, `clang++` and `ld.lld` on `PATH`), CMake 3.20+, Make and Python 3.10+.
On a fresh checkout without `deps/`, run `bash install-deps.sh` once.

Configure and build (**one command each**). This example uses the tested ROCm
SDK's Clang 17 with GCC 14.3.1 headers; see
[other toolchains](docs/BUILD_AND_RUN.md#toolchain) if yours differs.

```bash
cmake -S . -B build/pcuda-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -DACPP_TARGETS=generic \
  -DCMAKE_CXX_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1 \
  -DCRYSTAL_PCUDA_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1
cmake --build build/pcuda-release -j4
```

If `data/ssb/sf10/dataset.manifest` is missing, run `bash bench/prepare_ssb.sh`
once. Then run a query:

```bash
./build/pcuda-release/db_cli < test/queries/sql/q33.sql
```

No project environment variables or config file are needed. Run the same binary
without input redirection for the SQL console. See the
**[Russian guide](docs/BUILD_AND_RUN.md#quick-start)** for
[SSB validation](docs/BUILD_AND_RUN.md#ssb) or [a smaller dataset](docs/BUILD_AND_RUN.md#data).

## Build and SQL JIT

The engine uses C++17 and AdaptiveCpp generic PCUDA. Each specialized GPU kernel
is compiled and linked separately with its launch wrapper. Loaded modules and
entry points are cached **only in the current process**; ordinary parameter values
are passed through [ABI v2](test/pcuda_stage7/TEXT_ABI.md). The thirteen manual SSB examples use independent
prebuilt PCUDA algorithms, not the SQL generator.

Numeric `SUM`/`MIN`/`MAX` preserve signed Int64 or Float64 input; `COUNT` returns
UInt64 and `AVG` returns Float64. Known typed NULL retains its slot type. Changing
compatible values (including negative numeric literals), CASE ranges or LIMIT
reuses ready modules. A new type/output layout may prepare a missing variant;
each stable kernel/layout family keeps up to eight typed modules and one tagged
fallback. Numeric `GROUP BY CASE` accepts integer/NULL leaves; Text CASE keys
are supported separately. See the
[stage-6 report](test/pcuda_stage6/REPORT.md) for the tested boundaries.
Mixed COUNT/signed arithmetic and comparisons preserve negative values without
routing integers through floating point.

GPU PCH dependency tracking requires CMake ≥3.20 or the Ninja generator.

A minimal PCH and universal GPU services are prepared at build/install time, never
silently rebuilt by a SQL query. Host code remains Release; dynamic kernels use
`-O0`. The manifest fingerprints PCH dependencies, compiler,
plugins/runtime and compiler-affecting environment. An incompatible input causes
a diagnostic requesting a rebuild of `pcuda_jit_pch`, not unchecked PCH reuse.

See the guide for [developer builds and tests](docs/BUILD_AND_RUN.md#build),
[installation](docs/BUILD_AND_RUN.md#install), and
[optional runtime settings](docs/BUILD_AND_RUN.md#config).

## SSB storage and manual examples

`bash bench/prepare_ssb.sh` creates a ready-to-use native dataset without a GPU
or engine build. It uses the pinned `ssb-dbgen` and a host-only C++17 importer
to generate, validate and publish the columns and `dataset.manifest`. Existing
output is never overwritten. See [other scales and import options](docs/BUILD_AND_RUN.md#data).

`include/storage/ssb_schema.h` describes the native unencoded files, including all
17 DDATE columns. Numeric Int32 files must have the exact expected length;
ASCII byte flags are decoded from `'0'`/`'1'` into Int32 values 0/1. No prefix
truncation or manufactured zero tail is used. City/nation/region and manufacturer/
category/brand retain their **original Text**, not numeric codes. FixedText is decoded
to validated UTF-8 **Text**, never an
integer/dictionary code: the first bounded NUL ends a value, otherwise the full
width is used. The physical tail is ignored; trailing spaces are significant.
NULL is separate from the empty string. Comparisons use unsigned UTF-8 bytes,
case-sensitively, without normalization or locale collation.

Text supports projection/star, parameterized literals, WHERE/3VL and all six
comparisons, CASE, mixed GROUP BY keys, existing inner equijoins/OR joins, HAVING,
ORDER BY/LIMIT and COUNT/MIN/MAX. SUM/AVG(Text), implicit Text/number/Boolean
conversion, logical NUL and malformed UTF-8 are errors. LIKE/regex, string
functions, new join kinds and Float64 GROUP BY are not added.

The GPU owns compact bytes/offsets/validity; intermediate values are borrowed
views. CPU results own real strings. CLI Text cells are quoted JSON strings,
so `""`, `"NULL"`, `"a "` and SQL `NULL` are unambiguous. Compatible values reuse
ready modules and the active plan's high-water workspace; changing the plan
retires its old workspace, not the process-local module cache.

The [7-TEXT report](test/pcuda_stage7/REPORT.md) and
[string research](test/pcuda_string_research/REPORT.md) describe implementation,
evidence and limitations. `custom2.sql` now uses the approved
`'BUILDING'`/`'FURNITURE'` literals. Historical failures are retained; this does not
close stage 8 or establish full-SQL performance targets.

Data defaults to `data/ssb/sf10` relative to the working directory. Set runtime
`data_dir` or `CRYSTAL_DATA_DIR` to select a dataset; actual row counts and schema
come from its mandatory manifest, before GPU allocations. The old `data/10/`,
encoded loader and numeric reference files are historical, not native-format inputs.
Manual examples also accept the directory directly:

```bash
./build/pcuda-release/ssb_q33 data/ssb/sf10
```

Targets `ssb_q11` through `ssb_q43` cover all thirteen queries. Shared manual
support owns buffers/stream and supplies handwritten PHT build/fused Text probe kernels;
it does not link the SQL parser, generator or executor. A presence buffer retains
zero/negative sums; GPU count/scan/write fetches a compact result with checked
capacity. Manual inputs must be nonnullable and have unique dimension keys;
the examples explicitly reject null sidecars and duplicate keys. Engine fixtures
separately test nullable data and nonunique joins.
Native groups hash/compare original values; representative row references are not
a dictionary. Typed manual result rows are JSON arrays. The independent streaming
CPU oracle can additionally verify all 58 columns against `.tbl` using `--tables`.
See the [native dataset integration report](test/ssb_native_data/REPORT.md).

Manual Q1.1 and `scale10/11.txt` use 1993 only. The original SQL file also includes
1994-01-01; that known difference is checked independently, not silently corrected
to make a comparison pass. See the [stage-7 report](test/pcuda_stage7/REPORT.md).

## Acceptance and measurement (stage 8)

`ctest` runs host checks with active assertions even in Release. Hardware clients
remain separate so they do not compete for the GPU. The acceptance runner reuses
the existing nullable/MHT/OR, ABI/resource, Text, manual and full current/heavy
manifest clients; `future` is not promoted to supported SQL.

See [the regression commands](docs/BUILD_AND_RUN.md#regression).

The older numeric-SF10 measurement scripts/results are historical and are not
native-Text performance oracles. No performance/SLA conclusion follows from the
format migration. Do not run builds, other GPU clients or oracle analysis alongside
a performance series. Historical methodology and measurements are in
[the stage-8 report](test/pcuda_stage8/REPORT.md) for reproduction and actual status.
Cold-path/JIT analysis and the deferred warm backlog are maintained separately in
[8-ANALYSIS](test/pcuda_stage8_analysis/REPORT.md); proposed optimizations are not
implemented merely by documenting them.
The measured S1/execution-overlap experiment is **off by default**. Explicit
`jit_overlap_experiment=on` (or `CRYSTAL_JIT_OVERLAP_EXPERIMENT=on`) selects
`pcuda-modules-overlap-exp-v1`; build/load/execution overlap, so component sums
are unavailable. The ordinary `pcuda-modules-v1` path remains the default;
SQL acceptance targets still require agreement.

Extended timing adds a **direct** `SQL to PCUDA time`, stopping when missing module
sources are ready, before compile/link/load or query GPU allocations. It overlaps
old prepare/codegen intervals and is not added to component sums. Startup/data-load
and main-to-first-CPU-result are reported separately for the first successful query.
The native first-result launcher measures spawn-to-result before formatting/printing;
the wall time of the whole multiquery process is a different metric. See
[timing.md](timing.md). Prototype limits are not automatically full-SQL SLAs.
