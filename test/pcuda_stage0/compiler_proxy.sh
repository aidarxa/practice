#!/usr/bin/env bash
# Measurement-only Bash wrapper; acpp's own interpreter/driver overhead is retained.
set -u
export LC_ALL=C
real=${CRYSTAL_STAGE0_REAL_ACPP:?}
directory=${CRYSTAL_STAGE0_EVENT_DIR:?}
id="${EPOCHREALTIME//./}-${BASHPID}"
prefix="$directory/$id"
phase=compile_and_link
for arg in "$@"; do
    if [[ "$arg" == c++-header ]]; then phase=pch_prepare; fi
done
printf '%s\0' "$real" "$@" > "$prefix.original.argv"
args=()
skip=0
for arg in "$@"; do
    if (( skip )); then skip=0; continue; fi
    if [[ "$arg" == -O[0-3] && -n "${CRYSTAL_STAGE0_OPTIMIZATION:-}" ]]; then
        arg="-${CRYSTAL_STAGE0_OPTIMIZATION}"
    fi
    if [[ "$arg" == -include-pch && "$phase" != pch_prepare && "${CRYSTAL_STAGE0_USE_PCH:-on}" == off ]]; then
        skip=1
        continue
    fi
    if [[ "$arg" == -ftime-trace* ]]; then continue; fi
    if [[ "$arg" == -fuse-ld=* ]]; then continue; fi
    args+=("$arg")
done
args+=("-ftime-trace=$prefix.compile.json")
if [[ "$phase" != pch_prepare ]]; then
    args+=(-fuse-ld=lld "-Wl,--time-trace=$prefix.link.json")
    if [[ -n "${CRYSTAL_STAGE0_LINK_OPT:-}" ]]; then args+=("-Wl,-O${CRYSTAL_STAGE0_LINK_OPT}"); fi
fi
command=("$real" "${args[@]}")
printf '%s\0' "${command[@]}" > "$prefix.argv"
if [[ "${CRYSTAL_STAGE0_TRACE_COMPILER:-0}" == 1 ]]; then
    command=(strace -f -ttt -s 4096 -e trace=process -o "$prefix.trace" "${command[@]}")
fi
TIMEFORMAT='{"wall_s":%6R,"user_s":%6U,"system_s":%6S}'
{ time "${command[@]}" 2>"$prefix.stderr"; } 2>"$prefix.time.json"
status=$?
printf '{"phase":"%s","returncode":%d}\n' "$phase" "$status" > "$prefix.meta.json"
cat "$prefix.stderr" >&2
exit "$status"
