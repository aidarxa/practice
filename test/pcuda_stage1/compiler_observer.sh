#!/usr/bin/env bash
## @file
## @brief Correctness-run compiler observer; does not change optimization or replay commands.
set -u
export LC_ALL=C
real=${CRYSTAL_STAGE0_REAL_ACPP:?}
directory=${CRYSTAL_STAGE0_EVENT_DIR:?}
prefix="$directory/${EPOCHREALTIME//./}-${BASHPID}"
phase=compile_and_link
args=()
printf '%s\0' "$real" "$@" > "$prefix.original.argv"
for arg in "$@"; do
    if [[ "$arg" == c++-header ]]; then phase=pch_prepare; fi
    if [[ "$arg" == -ftime-trace* ]]; then continue; fi
    args+=("$arg")
done
if [[ "${CRYSTAL_STAGE1_TRACES:-0}" == 1 ]]; then
    args+=("-ftime-trace=$prefix.compile.json")
    if [[ "$phase" != pch_prepare ]]; then
        args+=(-fuse-ld=lld "-Wl,--time-trace=$prefix.link.json")
    fi
fi
printf '%s\0' "$real" "${args[@]}" > "$prefix.argv"
TIMEFORMAT='{"wall_s":%6R,"user_s":%6U,"system_s":%6S}'
{ time "$real" "${args[@]}" 2>"$prefix.stderr"; } 2>"$prefix.time.json"
status=$?
printf '{"phase":"%s","returncode":%d}\n' "$phase" "$status" > "$prefix.meta.json"
cat "$prefix.stderr" >&2
exit "$status"
