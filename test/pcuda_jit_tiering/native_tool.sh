#!/usr/bin/env bash
set -u
export LC_ALL=C
prefix=$1
shift
# Never preload the experimental LLVM library into a compiler executable:
# ROCm opt has its own LLVM globals and would register options twice.
unset LD_PRELOAD JIT_POC_DIRECTORY JIT_POC_NATIVE_TIMER
TIMEFORMAT='{"wall_s":%6R,"user_s":%6U,"system_s":%6S}'
{ time "$@" 2>"$prefix.stderr"; } 2>"$prefix.time.json"
exit $?
