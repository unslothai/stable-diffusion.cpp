#include "runtime/tiling.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "core/util.h"
#include "ggml.h"

struct TileSpan {
    int offset;
    int overlap_before;
    int overlap_after;
};

static std::vector<TileSpan> sd_tiling_plan_axis(int dimension,
                                                 int tile_size,
                                                 float target_overlap,
                                                 bool circular) {
    if (tile_size >= dimension) {
        return {{0, 0, 0}};
    }

    const int extent = circular ? dimension : dimension - tile_size;
    int intervals    = extent;
    if (tile_size > 1) {
        const float target = extent / ((1.0f - target_overlap) * tile_size);
        // Adjacent tiles must overlap, but three tiles must not cover the same point.
        const int min_intervals = 1 + (extent - 1) / (tile_size - 1);
        const int max_intervals = std::max(1, static_cast<int>(2LL * extent / tile_size));
        const int lower         = std::clamp(static_cast<int>(std::floor(target)), min_intervals, max_intervals);
        const int upper         = std::clamp(static_cast<int>(std::ceil(target)), min_intervals, max_intervals);
        auto error              = [&](int count) {
            return std::abs(1.0f / count - 1.0f / target);
        };
        intervals = error(upper) < error(lower) ? upper : lower;
    }

    const int num_tiles = circular ? intervals : intervals + 1;
    auto position       = [&](int index) -> int {
        return static_cast<int>(int64_t(index) * extent / intervals);
    };
    std::vector<TileSpan> tiles;
    tiles.reserve(num_tiles);
    for (int i = 0; i < num_tiles; ++i) {
        const int offset = position(i);
        // Keep wrapped neighbors in the same unwrapped coordinate space.
        const int previous = i > 0 ? position(i - 1) : position(num_tiles - 1) - dimension;
        const int next     = i + 1 < num_tiles ? position(i + 1) : dimension;
        const int before   = i > 0 || circular ? previous + tile_size - offset : 0;
        const int after    = i + 1 < num_tiles || circular ? offset + tile_size - next : 0;
        tiles.push_back({offset, before, after});
    }
    return tiles;
}

int sd_tiling_seam_safe_tile_size(int small_dim, int tile_size, float tile_overlap_factor) {
    // A tile a little over half the axis (32 of 62 latents) gets two tiles overlapping by 1-2 latents, which shows
    // as a seam. Pick the nearest size (smaller first, larger only up to 2x) whose overlaps are >= half the target
    // and >= 2 latents.
    const int min_tile_size = 4;
    if (tile_overlap_factor <= 0.f || tile_size >= small_dim) {
        return tile_size;
    }
    auto overlaps_enough = [&](int size) {
        const int min_overlap = std::max(2, static_cast<int>(size * tile_overlap_factor * 0.5f));
        const auto tiles      = sd_tiling_plan_axis(small_dim, size, tile_overlap_factor, false);
        for (size_t i = 0; i + 1 < tiles.size(); ++i) {
            if (tiles[i].overlap_after < min_overlap) {
                return false;
            }
        }
        return true;
    };
    for (int size = tile_size; size >= min_tile_size; --size) {
        if (overlaps_enough(size)) {
            return size;
        }
    }
    for (int size = tile_size + 1; size < small_dim && size <= 2 * tile_size; ++size) {
        if (overlaps_enough(size)) {
            return size;
        }
    }
    return tile_size;
}

static int64_t sd_tensor_plane_size(const sd::Tensor<float>& tensor) {
    GGML_ASSERT(tensor.dim() >= 2);
    return tensor.shape()[0] * tensor.shape()[1];
}

static sd::Tensor<float> sd_tensor_split_2d(const sd::Tensor<float>& input, int width, int height, int x, int y) {
    GGML_ASSERT(input.dim() >= 4);
    std::vector<int64_t> output_shape = input.shape();
    output_shape[0]                   = width;
    output_shape[1]                   = height;
    sd::Tensor<float> output(std::move(output_shape));
    int64_t input_width  = input.shape()[0];
    int64_t input_height = input.shape()[1];
    int64_t input_plane  = sd_tensor_plane_size(input);
    int64_t output_plane = sd_tensor_plane_size(output);
    int64_t plane_count  = input.numel() / input_plane;
    // Plane-outer: planes are contiguous, so the inner loop streams instead of striding a plane per element.
    std::vector<int64_t> src_x(static_cast<size_t>(width));
    for (int ix = 0; ix < width; ix++) {
        src_x[ix] = (ix + x) % input_width;
    }
    for (int64_t plane = 0; plane < plane_count; ++plane) {
        const float* src = input.data() + plane * input_plane;
        float* dst       = output.data() + plane * output_plane;
        for (int iy = 0; iy < height; iy++) {
            const float* src_row = src + input_width * ((iy + y) % input_height);
            float* dst_row       = dst + static_cast<int64_t>(width) * iy;
            for (int ix = 0; ix < width; ix++) {
                dst_row[ix] = src_row[src_x[ix]];
            }
        }
    }
    return output;
}

template <typename Fn>
static void sd_parallel_for_planes(int64_t plane_count, int64_t plane_elements, Fn&& fn) {
    // Planes are independent and each element is still produced by one thread with the same
    // arithmetic, so splitting by plane does not change the result.
    const int64_t min_elements = 1 << 18;
    int64_t threads            = std::min<int64_t>(plane_count, std::max<int64_t>(1, plane_count * plane_elements / min_elements));
    threads                    = std::min<int64_t>(threads, std::max(1u, std::min(16u, std::thread::hardware_concurrency())));
    if (threads <= 1) {
        fn(0, plane_count);
        return;
    }
    std::vector<std::thread> workers;
    const int64_t per_thread = (plane_count + threads - 1) / threads;
    for (int64_t begin = per_thread; begin < plane_count; begin += per_thread) {
        workers.emplace_back([&fn, begin, end = std::min(plane_count, begin + per_thread)]() { fn(begin, end); });
    }
    fn(0, std::min(plane_count, per_thread));
    for (auto& worker : workers) {
        worker.join();
    }
}

static void sd_tensor_merge_2d(const float* input_data,
                               int64_t in_width,
                               int64_t in_height,
                               int64_t plane_count,
                               sd::Tensor<float>* output,
                               int x,
                               int y,
                               int overlap_left,
                               int overlap_right,
                               int overlap_top,
                               int overlap_bottom) {
    GGML_ASSERT(output != nullptr);

    int64_t out_width  = output->shape()[0];
    int64_t out_height = output->shape()[1];
    int64_t in_size    = in_width * in_height;
    int64_t out_size   = sd_tensor_plane_size(*output);

    GGML_ASSERT(output->numel() == plane_count * out_size);
    GGML_ASSERT(x >= 0 && y >= 0);
    GGML_ASSERT(x < out_width && in_width <= out_width);
    GGML_ASSERT(y < out_height && in_height <= out_height);
    GGML_ASSERT(overlap_left >= 0 && overlap_right >= 0);
    GGML_ASSERT(overlap_top >= 0 && overlap_bottom >= 0);

    auto smootherstep_f32 = [](const float x) -> float {
        return x * x * x * (x * (6.0f * x - 15.0f) + 10.0f);
    };
    // Weights depend only on the column or row; the per-element arithmetic order is unchanged.
    std::vector<float> x_weights(static_cast<size_t>(in_width));
    std::vector<float> y_weights(static_cast<size_t>(in_height));
    std::vector<int64_t> dst_x(static_cast<size_t>(in_width));
    for (int ix = 0; ix < in_width; ++ix) {
        float x_f = 1.0f;
        if (ix < overlap_left) {
            x_f = static_cast<float>(ix) / overlap_left;
        }
        if (ix >= in_width - overlap_right) {
            x_f = static_cast<float>(in_width - ix) / overlap_right;
        }
        x_weights[ix] = smootherstep_f32(std::clamp(x_f, 0.0f, 1.0f));
        dst_x[ix]     = (x + ix) % out_width;
    }
    for (int iy = 0; iy < in_height; ++iy) {
        float y_f = 1.0f;
        if (iy < overlap_top) {
            y_f = static_cast<float>(iy) / overlap_top;
        }
        if (iy >= in_height - overlap_bottom) {
            y_f = static_cast<float>(in_height - iy) / overlap_bottom;
        }
        y_weights[iy] = smootherstep_f32(std::clamp(y_f, 0.0f, 1.0f));
    }
    float* output_data = output->data();
    sd_parallel_for_planes(plane_count, in_size, [&](int64_t first_plane, int64_t last_plane) {
        for (int64_t plane = first_plane; plane < last_plane; ++plane) {
            const float* src = input_data + plane * in_size;
            float* dst       = output_data + plane * out_size;
            for (int iy = 0; iy < in_height; ++iy) {
                const float y_weight = y_weights[iy];
                const float* src_row = src + in_width * iy;
                float* dst_row       = dst + out_width * ((y + iy) % out_height);
                for (int ix = 0; ix < in_width; ++ix) {
                    dst_row[dst_x[ix]] += x_weights[ix] * y_weight * src_row[ix];
                }
            }
        }
    });
}

sd::Tensor<float> process_tiles_2d(const sd::Tensor<float>& input,
                                   int output_width,
                                   int output_height,
                                   int scale,
                                   int p_tile_size_w,
                                   int p_tile_size_h,
                                   float tile_overlap_factor,
                                   bool circular_x,
                                   bool circular_y,
                                   const TileProcessCallback& on_processing,
                                   bool silent) {
    return process_tiles_2d_batched(
        input,
        output_width,
        output_height,
        scale,
        p_tile_size_w,
        p_tile_size_h,
        tile_overlap_factor,
        circular_x,
        circular_y,
        [](int) { return 1; },
        [&](const std::vector<sd::Tensor<float>>& tiles) {
            TileBatchOutput output;
            output.data = on_processing(tiles[0]);
            if (!output.data.empty()) {
                output.stack_dim = static_cast<size_t>(output.data.dim() - 1);
            }
            return output;
        },
        silent);
}

sd::Tensor<float> process_tiles_2d_batched(const sd::Tensor<float>& input,
                                           int output_width,
                                           int output_height,
                                           int scale,
                                           int p_tile_size_w,
                                           int p_tile_size_h,
                                           float tile_overlap_factor,
                                           bool circular_x,
                                           bool circular_y,
                                           const TileBatchSizeCallback& batch_size,
                                           const TileBatchProcessCallback& on_processing,
                                           bool silent) {
    sd::Tensor<float> output;
    int input_width  = static_cast<int>(input.shape()[0]);
    int input_height = static_cast<int>(input.shape()[1]);

    GGML_ASSERT(((input_width / output_width) == scale) ||
                ((output_width / input_width) == scale));

    bool decode      = output_width > input_width;  // scale up
    int small_width  = decode ? input_width : output_width;
    int small_height = decode ? input_height : output_height;
    int scale_in     = decode ? 1 : scale;
    int scale_out    = decode ? scale : 1;

    const auto tiles_x = sd_tiling_plan_axis(small_width, p_tile_size_w, tile_overlap_factor, circular_x);
    const auto tiles_y = sd_tiling_plan_axis(small_height, p_tile_size_h, tile_overlap_factor, circular_y);

    int tile_width         = std::min(p_tile_size_w, small_width);
    int tile_height        = std::min(p_tile_size_h, small_height);
    int input_tile_width   = tile_width * scale_in;
    int input_tile_height  = tile_height * scale_in;
    int output_tile_width  = tile_width * scale_out;
    int output_tile_height = tile_height * scale_out;

    struct TilePlacement {
        const TileSpan* x;
        const TileSpan* y;
    };
    std::vector<TilePlacement> placements;
    for (const auto& y : tiles_y) {
        for (const auto& x : tiles_x) {
            placements.push_back({&x, &y});
        }
    }

    const int num_tiles = static_cast<int>(placements.size());
    int tile_count      = 0;
    if (!silent) {
        LOG_VERBOSE("num tiles : %d, %d ", static_cast<int>(tiles_x.size()), static_cast<int>(tiles_y.size()));
        LOG_VERBOSE("processing %i tiles", num_tiles);
        pretty_progress(0, num_tiles, 0.0f);
    }
    // Blend on a worker while the next batch computes, still in tile order; SD_TILE_ASYNC_MERGE=0 blends inline.
    static const bool async_merge = []() {
        const char* value = getenv("SD_TILE_ASYNC_MERGE");
        return value == nullptr || value[0] == '\0' || atoi(value) != 0;
    }();
    std::thread merger;
    bool merge_failed = false;
    auto join_merger  = [&]() {
        if (merger.joinable()) {
            merger.join();
        }
    };
    struct MergerGuard {
        std::thread& thread;
        ~MergerGuard() {
            if (thread.joinable()) {
                thread.join();
            }
        }
    } merger_guard{merger};
    std::vector<sd::Tensor<float>> input_tiles;
    for (size_t first = 0; first < placements.size();) {
        const int remaining = static_cast<int>(placements.size() - first);
        const int count     = std::max(1, std::min(remaining, batch_size(remaining)));
        const size_t last   = first + static_cast<size_t>(count);
        int64_t t1          = ggml_time_ms();
        input_tiles.clear();
        for (size_t i = first; i < last; ++i) {
            input_tiles.push_back(sd_tensor_split_2d(input,
                                                     input_tile_width,
                                                     input_tile_height,
                                                     placements[i].x->offset * scale_in,
                                                     placements[i].y->offset * scale_in));
        }
        auto batch = on_processing(input_tiles);
        if (batch.data.empty()) {
            return {};
        }
        const int64_t batch_tiles = static_cast<int64_t>(last - first);
        GGML_ASSERT(batch.stack_dim < static_cast<size_t>(batch.data.dim()) && batch.data.shape()[batch.stack_dim] % batch_tiles == 0);
        for (size_t d = batch.stack_dim + 1; d < static_cast<size_t>(batch.data.dim()); ++d) {
            GGML_ASSERT(batch.data.shape()[d] == 1);
        }
        std::vector<int64_t> tile_shape = batch.data.shape();
        tile_shape[batch.stack_dim] /= batch_tiles;
        GGML_ASSERT(tile_shape[0] == output_tile_width && tile_shape[1] == output_tile_height);
        auto merge_batch = [&, first, last, tile_shape](const sd::Tensor<float>& data) {
            const int64_t tile_numel  = data.numel() / static_cast<int64_t>(last - first);
            const int64_t plane_count = tile_numel / (tile_shape[0] * tile_shape[1]);
            if (output.empty()) {
                std::vector<int64_t> output_shape = tile_shape;
                output_shape[0]                   = output_width;
                output_shape[1]                   = output_height;
                output                            = sd::Tensor<float>::zeros(std::move(output_shape));
            }
            for (size_t i = first; i < last; ++i) {
                const auto& x = *placements[i].x;
                const auto& y = *placements[i].y;
                sd_tensor_merge_2d(data.data() + static_cast<int64_t>(i - first) * tile_numel,
                                   tile_shape[0],
                                   tile_shape[1],
                                   plane_count,
                                   &output,
                                   x.offset * scale_out,
                                   y.offset * scale_out,
                                   x.overlap_before * scale_out,
                                   x.overlap_after * scale_out,
                                   y.overlap_before * scale_out,
                                   y.overlap_after * scale_out);
            }
        };
        join_merger();
        if (merge_failed) {
            return {};
        }
        bool merged_async = false;
        if (async_merge && last < placements.size()) {
            auto data = std::make_shared<sd::Tensor<float>>(std::move(batch.data));
            try {
                merger = std::thread([&merge_failed, merge_batch, data]() {
                    try {
                        merge_batch(*data);
                    } catch (const std::exception& error) {
                        LOG_ERROR("tile merge failed: %s", error.what());
                        merge_failed = true;
                    }
                });
                merged_async = true;
            } catch (const std::system_error&) {
                merge_batch(*data);  // no thread available: merge inline
                merged_async = true;
            }
        }
        if (!merged_async) {
            merge_batch(batch.data);
        }

        const float tile_time = (ggml_time_ms() - t1) / 1000.0f / static_cast<float>(last - first);
        for (size_t i = first; i < last; ++i) {
            ++tile_count;
            if (!silent) {
                pretty_progress(tile_count, num_tiles, tile_time);
            }
        }
        first = last;
    }
    join_merger();
    if (merge_failed) {
        return {};
    }
    return output;
}
