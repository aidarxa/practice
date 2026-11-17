#!/usr/bin/env bash
# Build the pinned generator and host-only importer; never overwrite a dataset.
set -euo pipefail
export LC_ALL=C
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SCALE=10
OUTPUT=
TABLES=
KEEP_TABLES=off
JOBS=4
usage() {
    cat <<'EOF'
Usage: bash bench/prepare_ssb.sh [--scale SF] [--output DIRECTORY] [--jobs N]
                                 [--keep-tbl] [--from-tbl DIRECTORY]
Default: SF10 -> data/ssb/sf10 (native Text, no category encoding).
Requires CMake, a C/C++17 toolchain and Git; no GPU or SQL dependencies.
Missing ssb-dbgen sources are initialized at the repository's pinned revision.
An existing output is never replaced. --from-tbl imports existing dbgen files.
--keep-tbl retains the generated source tables in the printed build directory.
EOF
}
while (($#)); do
    case "$1" in
        --scale|--output|--jobs|--from-tbl)
            [[ $# -ge 2 && -n $2 ]] || { usage >&2; exit 2; }
            case "$1" in
                --scale) SCALE=$2;; --output) OUTPUT=$2;; --jobs) JOBS=$2;; --from-tbl) TABLES=$2;;
            esac
            shift 2;;
        --keep-tbl) KEEP_TABLES=on; shift;;
        --help|-h) usage; exit 0;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2;;
    esac
done
[[ $SCALE =~ ^[0-9]+([.][0-9]+)?$ && $JOBS =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid scale or jobs' >&2; exit 2; }
# Upstream truncates nonintegral SF >=1. Refuse to label such data with a false SF.
awk -v sf="$SCALE" 'BEGIN {exit !(sf > 0 && sf <= 100000 && (sf < 1 || sf == int(sf)))}' || {
    echo 'dbgen requires 0 < SF < 1 or a positive integer SF <= 100000' >&2; exit 2;
}
OUTPUT=${OUTPUT:-"$ROOT/data/ssb/sf$SCALE"}
[[ ! -e $OUTPUT && ! -L $OUTPUT ]] || { printf 'Output already exists: %s\n' "$OUTPUT" >&2; exit 1; }
mkdir -p -- "$(dirname -- "$OUTPUT")" "$ROOT/build/ssb-tools"
OUTPUT=$(realpath -m -- "$OUTPUT")
if [[ -n $TABLES ]]; then TABLES=$(realpath -e -- "$TABLES"); fi
RUN=$(mktemp -d "$ROOT/build/ssb-tools/prepare-XXXXXXXX")
printf 'Preparation artifacts: %s\n' "$RUN"
cleanup() {
    # Only this invocation's generated .tbl directory is disposable; never input/output.
    if [[ $KEEP_TABLES == off && -d $RUN/tables ]]; then rm -r -- "$RUN/tables"; fi
}
trap cleanup EXIT
cmake -S "$ROOT/bench/loader" -B "$RUN/loader" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$RUN/loader" --target ssb_convert test_ssb_dataset -j "$JOBS"
ctest --test-dir "$RUN/loader" --output-on-failure
CONVERTER="$RUN/loader/ssb_convert"
"$CONVERTER" scale "$SCALE"
GENERATOR=external-dbgen-tables
if [[ -z $TABLES ]]; then
    PIN=$(git -C "$ROOT" ls-files --stage -- bench/ssb-dbgen | awk '{print $2}')
    [[ -n $PIN ]] || { echo 'Missing ssb-dbgen gitlink' >&2; exit 1; }
    if [[ ! -f $ROOT/bench/ssb-dbgen/CMakeLists.txt ]]; then
        git -C "$ROOT" submodule update --init -- bench/ssb-dbgen
    fi
    ACTUAL=$(git -C "$ROOT/bench/ssb-dbgen" rev-parse HEAD)
    [[ $ACTUAL == "$PIN" && -z $(git -C "$ROOT/bench/ssb-dbgen" status --porcelain) ]] || {
        echo 'ssb-dbgen differs from the pinned clean revision; preserve local work and resolve it explicitly.' >&2; exit 1;
    }
    cmake -S "$ROOT/bench/ssb-dbgen" -B "$RUN/dbgen" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DWORKLOAD=SSB -DCSV_OUTPUT_FORMAT=OFF -DEOL_HANDLING=OFF -DYMD_DASH_DATE=OFF
    cmake --build "$RUN/dbgen" --target dbgen -j "$JOBS"
    mkdir -- "$RUN/tables"
    TABLES="$RUN/tables"
    (cd -- "$TABLES" && DSS_PATH="$TABLES" "$RUN/dbgen/dbgen" -b "$ROOT/bench/ssb-dbgen/dists.dss" -s "$SCALE" -v)
    GENERATOR="ssb-dbgen:$PIN"
fi
printf '%s\n' "$GENERATOR" > "$RUN/generator.txt"
"$CONVERTER" import "$TABLES" "$OUTPUT" "$SCALE" "$GENERATOR"
printf '\nReady. After building db_cli, for example:\n  CRYSTAL_DATA_DIR=%q ./build/db_cli\n' "$OUTPUT"
if [[ $KEEP_TABLES == on ]]; then printf 'Source tables: %s\n' "$TABLES"; fi
