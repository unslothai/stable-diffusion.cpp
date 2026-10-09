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

static void sd_tiling_calc_tiles(int& num_tiles_dim,
                                 float& tile_overlap_factor_dim,
                                 int small_dim,
                                 int tile_size,
                                 const float tile_overlap_factor,
                                 bool circular) {
    int tile_overlap     = static_cast<int>(tile_size * tile_overlap_factor);
    int non_tile_overlap = tile_size - tile_overlap;

    if (circular) {
        // circular means the last and first tile are overlapping (wraping around)
        num_tiles_dim = small_dim / non_tile_overlap;

        if (num_tiles_dim < 1) {
            num_tiles_dim = 1;
        }

        tile_overlap_factor_dim = (tile_size - small_dim / num_tiles_dim) / (float)tile_size;

        // if single tile and tile_overlap_factor is not 0, add one to ensure we have at least two overlapping tiles
        if (num_tiles_dim == 1 && tile_overlap_factor_dim > 0) {
            num_tiles_dim++;
            tile_overlap_factor_dim = 0.5;
        }

        return;
    }
    // else, non-circular means the last and first tile are not overlapping

    num_tiles_dim     = (small_dim - tile_overlap) / non_tile_overlap;
    int overshoot_dim = ((num_tiles_dim + 1) * non_tile_overlap + tile_overlap) % small_dim;

    if ((overshoot_dim != non_tile_overlap) && (overshoot_dim <= num_tiles_dim * (tile_size / 2 - tile_overlap))) {
        // if tiles don't fit perfectly using the desired overlap
        // and there is enough room to squeeze an extra tile without overlap becoming >0.5
        num_tiles_dim++;
    }

    tile_overlap_factor_dim = (float)(tile_size * num_tiles_dim - small_dim) / (float)(tile_size * (num_tiles_dim - 1));
    if (num_tiles_dim <= 2) {
        if (small_dim <= tile_size) {
            num_tiles_dim           = 1;
            tile_overlap_factor_dim = 0;
        } else {
            num_tiles_dim           = 2;
            tile_overlap_factor_dim = (2 * tile_size - small_dim) / (float)tile_size;
        }
    }
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
                               int64_t width,
                               int64_t height,
                               int64_t plane_count,
                               sd::Tensor<float>* output,
                               int x,
                               int y,
                               int overlap_x,
                               int overlap_y,
                               bool circular_x,
                               bool circular_y,
                               int x_skip,
                               int y_skip) {
    GGML_ASSERT(output != nullptr);
    int64_t img_width    = output->shape()[0];
    int64_t img_height   = output->shape()[1];
    int64_t input_plane  = width * height;
    int64_t output_plane = sd_tensor_plane_size(*output);
    GGML_ASSERT(output->numel() / output_plane == plane_count);

    // unclamped -> expects x in the range [0-1]
    auto smootherstep_f32 = [](const float x) -> float {
        GGML_ASSERT(x >= 0.f && x <= 1.f);
        return x * x * x * (x * (6.0f * x - 15.0f) + 10.0f);
    };

    // Same per-element arithmetic as the per-pixel form, so the result stays bit-identical.
    const bool blend = overlap_x > 0 || overlap_y > 0;
    std::vector<float> wx, wy;
    std::vector<int64_t> dst_x(static_cast<size_t>(width));
    for (int64_t ix = x_skip; ix < width; ix++) {
        dst_x[ix] = (x + ix) % img_width;
    }
    if (blend) {
        wx.resize(static_cast<size_t>(width));
        wy.resize(static_cast<size_t>(height));
        for (int64_t ix = x_skip; ix < width; ix++) {
            const float x_f_0 = (circular_x || (overlap_x > 0 && x > 0)) ? (ix - x_skip) / float(overlap_x) : 1.f;
            const float x_f_1 = (circular_x || (overlap_x > 0 && x < (img_width - width))) ? (width - ix) / float(overlap_x) : 1.f;
            wx[ix]            = smootherstep_f32(std::min(std::min(x_f_0, x_f_1), 1.f));
        }
        for (int64_t iy = y_skip; iy < height; iy++) {
            const float y_f_0 = (circular_y || (overlap_y > 0 && y > 0)) ? (iy - y_skip) / float(overlap_y) : 1.f;
            const float y_f_1 = (circular_y || (overlap_y > 0 && y < (img_height - height))) ? (height - iy) / float(overlap_y) : 1.f;
            wy[iy]            = smootherstep_f32(std::min(std::min(y_f_0, y_f_1), 1.f));
        }
    }
    float* output_data = output->data();
    sd_parallel_for_planes(plane_count, input_plane, [&](int64_t first_plane, int64_t last_plane) {
        for (int64_t plane = first_plane; plane < last_plane; ++plane) {
            const float* src = input_data + plane * input_plane;
            float* dst       = output_data + plane * output_plane;
            for (int64_t iy = y_skip; iy < height; iy++) {
                const float* src_row = src + width * iy;
                float* dst_row       = dst + img_width * ((y + iy) % img_height);
                if (blend) {
                    const float sy = wy[iy];
                    for (int64_t ix = x_skip; ix < width; ix++) {
                        float& out = dst_row[dst_x[ix]];
                        out        = out + src_row[ix] * sy * wx[ix];
                    }
                } else {
                    for (int64_t ix = x_skip; ix < width; ix++) {
                        dst_row[dst_x[ix]] = src_row[ix];
                    }
                }
            }
        }
    });
}

sd::Tensor<float> process_tiles_2d(const sd::Tensor<float>& input,
                                   int output_width,
                                   int output_height,
                                   int scale,
                                   int p_tile_size_x,
                                   int p_tile_size_y,
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
        p_tile_size_x,
        p_tile_size_y,
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
                                           int p_tile_size_x,
                                           int p_tile_size_y,
                                           float tile_overlap_factor,
                                           bool circular_x,
                                           bool circular_y,
                                           const TileBatchSizeCallback& batch_size,
                                           const TileBatchProcessCallback& on_processing,
                                           bool silent) {
    sd::Tensor<float> output;
    int input_width  = static_cast<int>(input.shape()[0]);
    int input_height = static_cast<int>(input.shape()[1]);

    GGML_ASSERT(((input_width / output_width) == (input_height / output_height)) &&
                ((output_width / input_width) == (output_height / input_height)));
    GGML_ASSERT(((input_width / output_width) == scale) ||
                ((output_width / input_width) == scale));

    int small_width  = output_width;
    int small_height = output_height;
    bool decode      = output_width > input_width;
    if (decode) {
        small_width  = input_width;
        small_height = input_height;
    }

    int num_tiles_x;
    float tile_overlap_factor_x;
    sd_tiling_calc_tiles(num_tiles_x, tile_overlap_factor_x, small_width, p_tile_size_x, tile_overlap_factor, circular_x);

    int num_tiles_y;
    float tile_overlap_factor_y;
    sd_tiling_calc_tiles(num_tiles_y, tile_overlap_factor_y, small_height, p_tile_size_y, tile_overlap_factor, circular_y);

    int tile_overlap_x     = static_cast<int32_t>(p_tile_size_x * tile_overlap_factor_x);
    int non_tile_overlap_x = p_tile_size_x - tile_overlap_x;
    int tile_overlap_y     = static_cast<int32_t>(p_tile_size_y * tile_overlap_factor_y);
    int non_tile_overlap_y = p_tile_size_y - tile_overlap_y;
    int tile_size_x        = p_tile_size_x < small_width ? p_tile_size_x : small_width;
    int tile_size_y        = p_tile_size_y < small_height ? p_tile_size_y : small_height;
    int input_tile_size_x  = tile_size_x;
    int input_tile_size_y  = tile_size_y;
    int output_tile_size_x = tile_size_x;
    int output_tile_size_y = tile_size_y;
    if (decode) {
        output_tile_size_x *= scale;
        output_tile_size_y *= scale;
    } else {
        input_tile_size_x *= scale;
        input_tile_size_y *= scale;
    }

    struct TilePlacement {
        int x_in;
        int y_in;
        int x_out;
        int y_out;
        int dx;
        int dy;
    };
    std::vector<TilePlacement> placements;
    bool last_y = false;
    bool last_x = false;
    for (int y = 0; y < small_height && !last_y; y += non_tile_overlap_y) {
        int dy = 0;
        if (!circular_y && y + tile_size_y >= small_height) {
            int original_y = y;
            y              = small_height - tile_size_y;
            dy             = original_y - y;
            if (decode) {
                dy *= scale;
            }
            last_y = true;
        }
        for (int x = 0; x < small_width && !last_x; x += non_tile_overlap_x) {
            int dx = 0;
            if (!circular_x && x + tile_size_x >= small_width) {
                int original_x = x;
                x              = small_width - tile_size_x;
                dx             = original_x - x;
                if (decode) {
                    dx *= scale;
                }
                last_x = true;
            }
            placements.push_back({decode ? x : scale * x,
                                  decode ? y : scale * y,
                                  decode ? x * scale : x,
                                  decode ? y * scale : y,
                                  dx,
                                  dy});
        }
        last_x = false;
    }

    int overlap_x_out = decode ? tile_overlap_x * scale : tile_overlap_x;
    int overlap_y_out = decode ? tile_overlap_y * scale : tile_overlap_y;
    int num_tiles     = num_tiles_x * num_tiles_y;
    int tile_count    = 1;
    float last_time   = 0.0f;
    if (!silent) {
        LOG_VERBOSE("num tiles : %d, %d ", num_tiles_x, num_tiles_y);
        LOG_VERBOSE("optimal overlap : %f, %f (targeting %f)", tile_overlap_factor_x, tile_overlap_factor_y, tile_overlap_factor);
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
            input_tiles.push_back(sd_tensor_split_2d(input, input_tile_size_x, input_tile_size_y, placements[i].x_in, placements[i].y_in));
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
        GGML_ASSERT(tile_shape[0] == output_tile_size_x && tile_shape[1] == output_tile_size_y);
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
                const auto& placement = placements[i];
                sd_tensor_merge_2d(data.data() + static_cast<int64_t>(i - first) * tile_numel,
                                   tile_shape[0],
                                   tile_shape[1],
                                   plane_count,
                                   &output,
                                   placement.x_out,
                                   placement.y_out,
                                   overlap_x_out,
                                   overlap_y_out,
                                   circular_x,
                                   circular_y,
                                   placement.dx,
                                   placement.dy);
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

        if (!silent) {
            int64_t t2 = ggml_time_ms();
            last_time  = (t2 - t1) / 1000.0f / static_cast<float>(last - first);
            for (size_t i = first; i < last; ++i) {
                pretty_progress(tile_count, num_tiles, last_time);
                tile_count++;
            }
        } else {
            tile_count += static_cast<int>(last - first);
        }
        first = last;
    }
    join_merger();
    if (merge_failed) {
        return {};
    }
    if (!silent && tile_count < num_tiles) {
        pretty_progress(num_tiles, num_tiles, last_time);
    }
    if (output.empty()) {
        return {};
    }
    return output;
}
