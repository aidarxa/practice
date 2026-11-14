# Crystal — PCUDA SQL engine

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
`-O0`. Configure a compatible compiler/SDK and, if needed, its additional flags:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DAdaptiveCpp_DIR=/path/to/AdaptiveCpp/lib/cmake/AdaptiveCpp \
  -DCRYSTAL_PCUDA_COMPILER=/path/to/AdaptiveCpp/bin/acpp \
  -DCRYSTAL_INCLUDE_DIR="$PWD/include" \
  -DCRYSTAL_DEPS_INCLUDE_DIR="$PWD/deps/include"
cmake --build build -j4
```

`CRYSTAL_PCUDA_FLAGS` and `CRYSTAL_PCUDA_LINK_FLAGS` are CMake lists for the
PCH/modules, not flags for the whole host engine. The default linker flags are
`-fuse-ld=lld;-Wl,-O0`. The manifest fingerprints PCH dependencies, compiler,
plugins/runtime and compiler-affecting environment. An incompatible input causes
a diagnostic requesting a rebuild of `pcuda_jit_pch`, not unchecked PCH reuse.

CPU tests can be built without finding AdaptiveCpp (the SQL parser dependency is
still required):

```bash
cmake -S . -B build-cpu -DCRYSTAL_BUILD_GPU=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j4
ctest --test-dir build-cpu --output-on-failure
```

### Installation and PCH selection

```bash
cmake --install build --prefix /final/deployment/prefix
```

Installation prepares a new PCH against the installed headers at the **final**
prefix. Clang PCH contains absolute input paths: relocation/`DESTDIR` is not
supported by this installation recipe; prepare it again after deployment.
The compiler/SDK must remain available. Manifest selection, in priority order:

1. `CRYSTAL_PCUDA_MANIFEST` environment variable;
2. `pcuda_manifest` in the runtime config;
3. `../lib/crystal/pcuda-jit/toolchain.manifest` relative to the executable;
4. the build-tree manifest compiled into the engine.

`jit_jobs` allows 1–4 independent compile/link chains. Each session gets a private
temporary directory. `jit_keep_artifacts=on` retains commands, sources, modules
and native timing logs for inspection, **not** future-process cache reuse.
`jit_traces=on` also retains Clang/LLD traces.

## Runtime configuration

At startup `db_cli` reads runtime settings from the first existing file in this order:

1. path from `CRYSTAL_CONFIG`
2. `./crystal.conf`
3. `./crystal-sycl.conf`
4. `$HOME/.config/crystal-sycl/config.conf`

`CRYSTAL_INCLUDE_DIR` and `CRYSTAL_DEPS_INCLUDE_DIR` remain accepted compatibility
settings, but changing them at runtime does not reconfigure an already prepared
PCH. Change its CMake inputs and rebuild instead.

See `crystal.conf.example`, [timing.md](timing.md), and the
[stage-5 report](test/pcuda_stage5/REPORT.md) for scope and reproducible checks.

## SSB storage and manual examples

`include/storage/ssb_schema.h` describes the converted SF10 files, including all
17 DDATE columns. Numeric Int32 files must have the exact expected length;
ASCII byte flags are decoded from `'0'`/`'1'` into Int32 values 0/1. No prefix
truncation or manufactured zero tail is used. Existing numeric city/nation/brand
codes are unchanged. FixedText is decoded to validated UTF-8 **Text**, never an
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

Data defaults to `data/10/` relative to the working directory. A deployment can
override the existing `BASE_PATH` compile definition; manual examples also accept
the directory directly:

```bash
cmake --build build --target ssb_q33 test_manual_ssb_gpu -j4
build/ssb_q33 "$PWD/data/10" 4
build/test_manual_ssb_gpu
python3 test/pcuda_stage7/validate.py manual --build build \
  --output-root build/stage7-check --timing-model pcuda-modules-v1
```

Targets `ssb_q11` through `ssb_q43` cover all thirteen queries. Shared manual
support owns buffers/stream and supplies handwritten PHT build/fused probe kernels;
it does not link the SQL parser, generator or executor. A presence buffer retains
zero/negative sums; GPU count/scan/write fetches a compact result with checked
capacity. Manual inputs must be nonnullable and have unique dimension keys;
the examples explicitly reject null sidecars and duplicate keys. Engine fixtures
separately test nullable data and nonunique joins.

Manual Q1.1 and `scale10/11.txt` use 1993 only. The original SQL file also includes
1994-01-01; that known difference is checked independently, not silently corrected
to make a comparison pass. See the [stage-7 report](test/pcuda_stage7/REPORT.md).

## Acceptance and measurement (stage 8)

`ctest` runs host checks with active assertions even in Release. Hardware clients
remain separate so they do not compete for the GPU. The acceptance runner reuses
the existing nullable/MHT/OR, ABI/resource, Text, manual and full current/heavy
manifest clients; `future` is not promoted to supported SQL.

```bash
ctest --test-dir build --output-on-failure
python3 test/pcuda_stage8/validate.py --build build \
  --output-root build/stage8-functional --timing-model pcuda-modules-v1
python3 test/pcuda_stage8/measure.py measure --build build \
  --output-root build/stage8-performance --sessions 7 --runs 9 \
  --timing-model pcuda-modules-v1
```

Do not run builds, other GPU clients or oracle analysis alongside the performance
series. Each query gets its own process and initially empty device cache, followed
by eight warm executions; Q3.3 also changes parameters eight times. Analysis is a
separate command after all measurements; see
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
