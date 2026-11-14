#!/usr/bin/env bash
# Build native Text behavioral clients in an owned source/build copy, with explicit external toolchain flags.
set -euo pipefail
output=$1
compiler=$2
shift 2
printf -v host_flags '%q ' "$@"
cmake -S . -B "$output/engine" -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_CXX_COMPILER=$compiler" "-DCMAKE_CXX_FLAGS=$host_flags"
cmake --build "$output/engine" --target test_pcuda_text_gpu test_pcuda_text_resources -j4
