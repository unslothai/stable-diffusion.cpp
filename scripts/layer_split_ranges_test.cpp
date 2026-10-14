// Standalone check of the layer-split range planner: g++ -std=c++17 -I src scripts/layer_split_ranges_test.cpp && ./a.out
#include <cstdio>
#include <cstdlib>

#include "core/layer_split_ranges.h"

static int failures = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                 \
        }                                                               \
    } while (0)

static bool contiguous(const sd::LayerSplitRangesResult& r) {
    for (size_t i = 1; i < r.device_by_segment.size(); i++) {
        if (r.device_by_segment[i] < r.device_by_segment[i - 1]) {
            return false;
        }
    }
    return true;
}

int main() {
    const int64_t MB = 1024ll * 1024;
    // 54 equal segments, 10.98 GB total, two cards with 12.6 GB weight capacity each (16 GB card budget less headroom).
    std::vector<int64_t> segs(54, (int64_t)(10.98 * 1024 * MB / 54));
    std::vector<int64_t> caps = {12599 * MB, 12599 * MB};
    auto fill                 = sd::plan_layer_split_ranges_fill_first(segs, caps);
    CHECK(fill.ok);
    CHECK(fill.bytes_by_device[1] == 0);  // the old plan: everything on card 0
    auto prop = sd::plan_layer_split_ranges(segs, caps);
    CHECK(prop.ok);
    CHECK(!prop.used_fill_first);
    CHECK(contiguous(prop));
    CHECK(prop.device_by_segment.front() == 0 && prop.device_by_segment.back() == 1);
    CHECK(std::llabs(prop.bytes_by_device[0] - prop.bytes_by_device[1]) <= segs[0]);

    // Unequal capacities: shares follow the capacities.
    caps = {4000 * MB, 12000 * MB};
    prop = sd::plan_layer_split_ranges(segs, caps);
    CHECK(prop.ok && contiguous(prop));
    CHECK(prop.bytes_by_device[0] <= caps[0] && prop.bytes_by_device[1] <= caps[1]);
    CHECK(prop.bytes_by_device[0] < prop.bytes_by_device[1]);

    // Exactly full: proportional must still fit (falls back to fill-first if rounding misses).
    std::vector<int64_t> four(4, 100);
    caps = {200, 200};
    prop = sd::plan_layer_split_ranges(four, caps);
    CHECK(prop.ok && prop.bytes_by_device[0] == 200 && prop.bytes_by_device[1] == 200);

    // Does not fit at all: failure names the last device.
    caps = {150, 150};
    prop = sd::plan_layer_split_ranges(four, caps);
    CHECK(!prop.ok);
    CHECK(prop.failed_device == 1);

    // Uneven segment sizes where the proportional plan overflows: fill-first rescues it.
    std::vector<int64_t> uneven = {10, 10, 10, 70};
    caps                        = {30, 70};
    prop                        = sd::plan_layer_split_ranges(uneven, caps);
    CHECK(prop.ok);
    CHECK(prop.bytes_by_device[0] == 30 && prop.bytes_by_device[1] == 70);

    // Unknown capacity sentinel (no free-memory report): still splits evenly.
    caps = {INT64_MAX / 4, INT64_MAX / 4};
    prop = sd::plan_layer_split_ranges(segs, caps);
    CHECK(prop.ok && prop.bytes_by_device[1] > 0);

    // Zero-byte segments inherit the current device and never trigger a move.
    std::vector<int64_t> zeros = {0, 50, 0, 50, 0};
    caps                       = {60, 60};
    prop                       = sd::plan_layer_split_ranges(zeros, caps);
    CHECK(prop.ok && contiguous(prop) && prop.bytes_by_device[0] == 50 && prop.bytes_by_device[1] == 50);

    // Single device: same as fill-first.
    caps = {INT64_MAX / 4};
    prop = sd::plan_layer_split_ranges(segs, caps);
    CHECK(prop.ok && prop.bytes_by_device.size() == 1);

    if (failures == 0) {
        std::printf("layer_split_ranges_test: all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
