#ifndef __SD_RUNTIME_TILING_H__
#define __SD_RUNTIME_TILING_H__

#include <functional>
#include <vector>

#include "core/tensor.hpp"

using TileProcessCallback   = std::function<sd::Tensor<float>(const sd::Tensor<float>&)>;
using TileBatchSizeCallback = std::function<int(int remaining_tiles)>;
// Output tiles of one batch stacked along stack_dim, which must be the highest axis with an extent
// above 1, so tile i is the i-th contiguous block of data. Each tile's shape is data.shape() with
// shape[stack_dim] divided by the number of tiles.
struct TileBatchOutput {
    sd::Tensor<float> data;
    size_t stack_dim = 0;
};
using TileBatchProcessCallback = std::function<TileBatchOutput(const std::vector<sd::Tensor<float>>&)>;

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
                                   bool silent = false);

// Same tile plan and merge order as process_tiles_2d, but hands up to batch_size(remaining)
// consecutive tiles to the callback at once.
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
                                           bool silent = false);

// Returns tile_size, adjusted if needed so a non-circular axis of small_dim is never split into tiles that
// barely overlap (see the definition for why that shows as a seam).
int sd_tiling_seam_safe_tile_size(int small_dim, int tile_size, float tile_overlap_factor);

#endif  // __SD_RUNTIME_TILING_H__
