// Standalone check of the CUDA quantized-matmul route table (ggml patch 0063, src/ggml-cuda/quant-route.h).
// After the ggml patches are applied:
//   g++ -std=c++17 -I ggml/include -I ggml/src scripts/quant_route_table_test.cpp && ./a.out
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#include "ggml-cuda/quant-route.h"

static int failures = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                 \
        }                                                               \
    } while (0)

static int64_t dflt(int cc, ggml_type t) {
    return ggml_cuda_quant_route_default_min_rows(cc, t);
}

// What the env resolution returns, with the names ggml_type_name() gives these types.
static const char * name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q4_K: return "q4_K";
        case GGML_TYPE_Q5_K: return "q5_K";
        case GGML_TYPE_Q6_K: return "q6_K";
        default:             return "other";
    }
}

static int64_t resolve(int cc, ggml_type t, int64_t env, const char * types) {
    return ggml_cuda_quant_route_min_rows(cc, t, name(t), env, types);
}

int main() {
    const ggml_type kq[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K };
    const ggml_type unmeasured[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q2_K,
                                     GGML_TYPE_Q3_K, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ2_XXS };

    // Turing (no BF16 tensor cores: the caller refuses the route anyway), consumer Ampere, Orin: MMQ for all.
    for (int cc : { 610, 700, 750, 860, 870 }) {
        CHECK(dflt(cc, GGML_TYPE_Q8_0) == 0);
        for (ggml_type t : kq) {
            CHECK(dflt(cc, t) == 0);
        }
    }
    // Ada: Q6_K from 4096 rows, everything else MMQ.
    CHECK(dflt(890, GGML_TYPE_Q6_K) == 4096);
    CHECK(dflt(890, GGML_TYPE_Q4_K) == 0);
    CHECK(dflt(890, GGML_TYPE_Q5_K) == 0);
    CHECK(dflt(890, GGML_TYPE_Q8_0) == 0);
    // A100: Q8_0 and the measured K-quants from 1024 rows.
    CHECK(dflt(800, GGML_TYPE_Q8_0) == 1024);
    for (ggml_type t : kq) {
        CHECK(dflt(800, t) == 1024);
    }
    // Hopper / datacenter Blackwell: Q8_0 only (patch 0016), K-quants unmeasured -> MMQ.
    for (int cc : { 900, 1000, 1030 }) {
        CHECK(dflt(cc, GGML_TYPE_Q8_0) == 1024);
        for (ggml_type t : kq) {
            CHECK(dflt(cc, t) == 0);
        }
    }
    // sm120: Q8_0 and Q4_K / Q5_K from 4096 rows, Q6_K from 1024. sm121 (GB10): Q8_0 only, as patch 0016.
    CHECK(dflt(1200, GGML_TYPE_Q8_0) == 4096);
    CHECK(dflt(1200, GGML_TYPE_Q4_K) == 4096);
    CHECK(dflt(1200, GGML_TYPE_Q5_K) == 4096);
    CHECK(dflt(1200, GGML_TYPE_Q6_K) == 1024);
    CHECK(dflt(1210, GGML_TYPE_Q8_0) == 4096);
    for (ggml_type t : kq) {
        CHECK(dflt(1210, t) == 0);
    }
    // Rubin and later: nothing until measured.
    CHECK(dflt(1300, GGML_TYPE_Q8_0) == 0);
    // Types never measured keep MMQ everywhere.
    for (int cc : { 800, 900, 1000, 1200 }) {
        for (ggml_type t : unmeasured) {
            CHECK(dflt(cc, t) == 0);
        }
    }

    // Table hygiene: ranges are non-empty, one rule per (cc, type), and a routed row count is past the MMVQ sizes.
    const size_t n = sizeof(ggml_cuda_quant_route_table) / sizeof(ggml_cuda_quant_route_table[0]);
    for (size_t i = 0; i < n; i++) {
        const ggml_cuda_quant_route_rule & a = ggml_cuda_quant_route_table[i];
        CHECK(a.cc_min < a.cc_max);
        CHECK(a.min_rows == 0 || a.min_rows > 8);
        for (size_t j = i + 1; j < n; j++) {
            const ggml_cuda_quant_route_rule & b = ggml_cuda_quant_route_table[j];
            CHECK(a.type != b.type || a.cc_max <= b.cc_min || b.cc_max <= a.cc_min);
        }
    }

    // Env: unset -> table; a value -> every quantized type; 0 -> MMQ everywhere.
    CHECK(resolve(800, GGML_TYPE_Q4_K, -1, nullptr) == 1024);
    CHECK(resolve(890, GGML_TYPE_Q4_K, 512, nullptr) == 512);
    CHECK(resolve(800, GGML_TYPE_Q4_0, 2048, nullptr) == 2048);
    CHECK(resolve(800, GGML_TYPE_Q8_0, 0, nullptr) == 0);
    CHECK(resolve(1200, GGML_TYPE_Q6_K, 0, nullptr) == 0);
    // GGML_CUDA_QUANT_CUBLAS_TYPES limits the override; unlisted types keep the table default.
    CHECK(resolve(1200, GGML_TYPE_Q4_K, 1024, "q4_K,q6_K") == 1024);
    CHECK(resolve(1200, GGML_TYPE_Q6_K, 1024, "q4_K,q6_K") == 1024);
    CHECK(resolve(1200, GGML_TYPE_Q5_K, 1024, "q4_K,q6_K") == 4096);
    CHECK(resolve(1200, GGML_TYPE_Q6_K, -1, "q4_K") == 1024);
    CHECK(resolve(1200, GGML_TYPE_Q8_0, 0, "q4_K") == 4096);
    CHECK(resolve(890, GGML_TYPE_Q4_K, 0, "q4_K") == 0);
    // Exact names only: no prefix, no case folding; empty entries and a trailing comma are harmless.
    CHECK(resolve(890, GGML_TYPE_Q4_K, 1024, "q4") == 0);
    CHECK(resolve(890, GGML_TYPE_Q4_K, 1024, "q4_K_M") == 0);
    CHECK(resolve(890, GGML_TYPE_Q4_K, 1024, "Q4_K") == 0);
    CHECK(resolve(890, GGML_TYPE_Q4_K, 1024, ",,q4_K,") == 1024);
    CHECK(resolve(890, GGML_TYPE_Q4_K, 1024, "") == 0);

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("quant route table: all checks passed\n");
    return 0;
}
