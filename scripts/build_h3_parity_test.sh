#!/usr/bin/env bash
# Builds scripts/h3_parity_test.cpp against a CPU build in ${1:-build} (configure with -DSD_BUILD_EXAMPLES=ON first).
set -euo pipefail
B="${1:-build}"
g++ -std=c++17 -O2 -fopenmp -DGGML_MAX_NAME=160 -DGGML_SCHED_MAX_COPIES=4 -DSD_GGML_FUSED_ROPE_ATTN -DSD_GGML_H3_FUSED_OPS -DSD_GGML_H3_FUSED_QK_NORM \
    -I src -I src/core -I include -I ggml/include -I thirdparty scripts/h3_parity_test.cpp \
    "$B/libstable-diffusion.a" "$B/ggml/src/libggml.a" "$B/ggml/src/libggml-cpu.a" "$B/ggml/src/libggml-base.a" \
    -lpthread -o "$B/h3_parity_test"
