#!/usr/bin/env bash
set -u
export LC_ALL=C
prefix=$1
shift
TIMEFORMAT='{"wall_s":%6R,"user_s":%6U,"system_s":%6S}'
{ time "$@" 2>"$prefix.stderr"; } 2>"$prefix.time.json"
exit $?
