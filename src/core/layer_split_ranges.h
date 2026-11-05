#ifndef __SD_CORE_LAYER_SPLIT_RANGES_H__
#define __SD_CORE_LAYER_SPLIT_RANGES_H__

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sd {

    struct LayerSplitRangesResult {
        bool ok = false;
        // Device index per segment (contiguous, non-decreasing).
        std::vector<size_t> device_by_segment;
        // Weight bytes placed on each device.
        std::vector<int64_t> bytes_by_device;
        // On failure: the segment that did not fit, the device it was tried on and the bytes that device would need.
        size_t failed_segment = 0;
        size_t failed_device  = 0;
        int64_t failed_need   = 0;
        // True when the proportional plan did not fit and the fill-first plan was used instead.
        bool used_fill_first = false;
    };

    // Fill-first: each device takes segments until the next one would exceed its capacity.
    inline LayerSplitRangesResult plan_layer_split_ranges_fill_first(const std::vector<int64_t>& segment_bytes,
                                                                     const std::vector<int64_t>& capacities) {
        LayerSplitRangesResult r;
        r.device_by_segment.assign(segment_bytes.size(), 0);
        r.bytes_by_device.assign(capacities.size(), 0);
        if (capacities.empty()) {
            return r;
        }
        size_t dev   = 0;
        int64_t used = 0;
        for (size_t s = 0; s < segment_bytes.size(); s++) {
            const int64_t bytes = segment_bytes[s];
            while (dev + 1 < capacities.size() && bytes > 0 && used + bytes > capacities[dev]) {
                dev++;
                used = 0;
            }
            if (bytes > 0 && used + bytes > capacities[dev]) {
                r.failed_segment = s;
                r.failed_device  = dev;
                r.failed_need    = used + bytes;
                return r;
            }
            used += bytes;
            r.device_by_segment[s] = dev;
            r.bytes_by_device[dev] += bytes;
        }
        r.ok = true;
        return r;
    }

    // Contiguous ranges sized in proportion to each device's capacity, so every device keeps the same fraction of its
    // budget free for the compute buffers of the segments it runs. A fill-first plan packs the main device up to its
    // weight capacity and leaves its compute buffers no room; it is only used when the proportional plan does not fit.
    inline LayerSplitRangesResult plan_layer_split_ranges(const std::vector<int64_t>& segment_bytes,
                                                          const std::vector<int64_t>& capacities) {
        LayerSplitRangesResult r;
        r.device_by_segment.assign(segment_bytes.size(), 0);
        r.bytes_by_device.assign(capacities.size(), 0);
        if (capacities.empty()) {
            return r;
        }
        double total = 0.0;
        for (int64_t b : segment_bytes) {
            total += (double)(b > 0 ? b : 0);
        }
        double capacity_sum = 0.0;
        std::vector<double> caps(capacities.size(), 0.0);
        for (size_t i = 0; i < capacities.size(); i++) {
            // An unknown capacity (very large sentinel) counts as able to hold everything.
            caps[i] = capacities[i] > 0 ? std::min((double)capacities[i], total) : 0.0;
            capacity_sum += caps[i];
        }
        if (capacities.size() == 1 || total <= 0.0 || capacity_sum <= 0.0) {
            return plan_layer_split_ranges_fill_first(segment_bytes, capacities);
        }

        // Cumulative byte boundary at which each device's range ends.
        std::vector<double> boundary(capacities.size(), total);
        double acc = 0.0;
        for (size_t i = 0; i + 1 < capacities.size(); i++) {
            acc += total * caps[i] / capacity_sum;
            boundary[i] = acc;
        }

        size_t dev      = 0;
        int64_t used    = 0;
        double consumed = 0.0;
        bool fits       = true;
        for (size_t s = 0; s < segment_bytes.size(); s++) {
            const int64_t bytes = segment_bytes[s];
            if (bytes > 0) {
                // Move on once this segment's midpoint lies past the current device's share, or it would not fit.
                while (dev + 1 < capacities.size() &&
                       (consumed + bytes * 0.5 > boundary[dev] || used + bytes > capacities[dev])) {
                    dev++;
                    used = 0;
                }
                if (used + bytes > capacities[dev]) {
                    fits = false;
                    break;
                }
            }
            used += bytes;
            consumed += (double)(bytes > 0 ? bytes : 0);
            r.device_by_segment[s] = dev;
            r.bytes_by_device[dev] += bytes;
        }
        if (fits) {
            r.ok = true;
            return r;
        }
        LayerSplitRangesResult fallback = plan_layer_split_ranges_fill_first(segment_bytes, capacities);
        fallback.used_fill_first        = true;
        return fallback;
    }

}  // namespace sd

#endif  // __SD_CORE_LAYER_SPLIT_RANGES_H__
