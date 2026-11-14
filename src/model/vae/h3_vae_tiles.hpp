#ifndef __SD_MODEL_VAE_H3_VAE_TILES_HPP__
#define __SD_MODEL_VAE_H3_VAE_TILES_HPP__

// Spatial tile layout and overlap blending of the MiniMax-H3 VAEs, as the reference implementation tiles them.
// Dependency-free (standard library only) so scripts/h3_vae_tiles_test.cpp can build it without ggml.
//
// Layout: the fewest `tile`-wide tiles whose overlaps are all >= `min_overlap`; the slack is spread round-robin over
// the overlaps in `step` increments (the VAE's spatial ratio), so tile boundaries stay latent aligned. Both encode and
// decode split in pixel units; encode blends in latent units (pixel overlaps / ratio), decode in pixel units.
//
// Blending: linear across each overlap, weight i / extent for the newer tile, rows first, then columns.
//   - encode blends every tile against its raw (unblended) upper and left neighbours;
//   - decode blends every tile against the already blended canvas above it and the already blended right edge of
//     its left neighbour, so seams stay continuous where overlaps cross.

#include <algorithm>
#include <cstdint>
#include <vector>

namespace H3VAETiles {

    struct Axis {
        std::vector<int64_t> starts;
        int64_t size = 0;               // length of every tile (the input length when one tile covers it)
        std::vector<int64_t> overlaps;  // overlaps[i] is between tile i and tile i + 1
    };

    inline Axis split(int64_t length, int64_t tile, int64_t min_overlap, int64_t step) {
        Axis axis;
        if (tile >= length) {
            axis.starts = {0};
            axis.size   = length;
            return axis;
        }
        int64_t count = (length + tile - 1) / tile;
        while (tile * count - min_overlap * (count - 1) - length < 0) {
            ++count;
        }
        axis.overlaps.assign(static_cast<size_t>(count - 1), min_overlap);
        const int64_t remaining = tile * count - min_overlap * (count - 1) - length;
        for (int64_t i = 0; i < remaining / step; ++i) {
            axis.overlaps[static_cast<size_t>(i % (count - 1))] += step;
        }
        axis.starts = {0};
        for (int64_t i = 0; i + 1 < count; ++i) {
            axis.starts.push_back(axis.starts.back() + tile - axis.overlaps[static_cast<size_t>(i)]);
        }
        axis.size = tile;
        return axis;
    }

    // The same layout in units `ratio` times coarser (pixel layout -> latent layout).
    inline Axis scale_down(Axis axis, int64_t ratio) {
        for (auto& v : axis.starts) {
            v /= ratio;
        }
        for (auto& v : axis.overlaps) {
            v /= ratio;
        }
        axis.size /= ratio;
        return axis;
    }

    // Tiles are [W, H, planes] float buffers (x fastest); the output is [out_w, out_h, planes].
    struct TileView {
        const float* data = nullptr;
        int64_t w         = 0;
        int64_t h         = 0;
    };

    // Encode blend: every tile against its raw upper and left neighbours. tiles[i][j] is row i, column j.
    inline void stitch_raw(const std::vector<std::vector<TileView>>& tiles,
                           const Axis& rows,
                           const Axis& cols,
                           int64_t planes,
                           float* out,
                           int64_t out_w,
                           int64_t out_h) {
        int64_t y0 = 0;
        for (size_t i = 0; i < tiles.size(); ++i) {
            int64_t x0     = 0;
            int64_t keep_h = 0;
            for (size_t j = 0; j < tiles[i].size(); ++j) {
                const TileView& t    = tiles[i][j];
                const TileView* up   = i > 0 ? &tiles[i - 1][j] : nullptr;
                const TileView* left = j > 0 ? &tiles[i][j - 1] : nullptr;
                const int64_t ext_h  = up ? std::min({rows.overlaps[i - 1], up->h, t.h}) : 0;
                const int64_t ext_w  = left ? std::min({cols.overlaps[j - 1], left->w, t.w}) : 0;
                const int64_t crop_h = i + 1 < tiles.size() ? t.h - rows.overlaps[i] : t.h;
                const int64_t crop_w = j + 1 < tiles[i].size() ? t.w - cols.overlaps[j] : t.w;
                keep_h               = crop_h;
                for (int64_t p = 0; p < planes; ++p) {
                    const float* cur = t.data + p * t.w * t.h;
                    for (int64_t y = 0; y < crop_h && y0 + y < out_h; ++y) {
                        float* dst = out + p * out_w * out_h + (y0 + y) * out_w + x0;
                        for (int64_t x = 0; x < crop_w && x0 + x < out_w; ++x) {
                            float value = cur[y * t.w + x];
                            if (y < ext_h) {
                                const float wy = static_cast<float>(y) / ext_h;
                                const float a  = up->data[p * up->w * up->h + (up->h - ext_h + y) * up->w + x];
                                value          = a * (1.f - wy) + value * wy;
                            }
                            if (x < ext_w) {
                                const float wx = static_cast<float>(x) / ext_w;
                                const float a  = left->data[p * left->w * left->h + y * left->w + (left->w - ext_w + x)];
                                value          = a * (1.f - wx) + value * wx;
                            }
                            dst[x] = value;
                        }
                    }
                }
                x0 += crop_w;
            }
            y0 += keep_h;
        }
    }

    // Decode blend, fed one tile at a time in row-major order. The canvas holds every blended value written so far:
    // the previous row's bottom overlap is still there when the next row reads it, because a tile only writes the
    // columns left of its right neighbour's start. The right overlap of the last tile goes to a small side buffer
    // (the left tail) that the next tile blends against.
    class CanvasStitcher {
    public:
        CanvasStitcher(const Axis& rows, const Axis& cols, int64_t planes, float* out, int64_t out_w, int64_t out_h)
            : rows_(rows), cols_(cols), planes_(planes), out_(out), out_w_(out_w), out_h_(out_h) {}

        void add(size_t i, size_t j, const TileView& t) {
            const bool last_col  = j + 1 >= cols_.starts.size();
            const int64_t y0     = rows_.starts[i];
            const int64_t x0     = cols_.starts[j];
            const int64_t ext_h  = i > 0 ? std::min(rows_.overlaps[i - 1], t.h) : 0;
            const int64_t ext_w  = j > 0 ? std::min({cols_.overlaps[j - 1], tail_w_, t.w}) : 0;
            const int64_t crop_w = last_col ? t.w : t.w - cols_.overlaps[j];
            const int64_t new_tw = last_col ? 0 : t.w - crop_w;
            next_tail_.resize(static_cast<size_t>(planes_ * t.h * new_tw));
            for (int64_t p = 0; p < planes_; ++p) {
                const float* cur  = t.data + p * t.w * t.h;
                float* canvas     = out_ + p * out_w_ * out_h_;
                const float* tail = tail_.data() + p * tail_h_ * tail_w_;
                float* next_tail  = next_tail_.data() + p * t.h * new_tw;
                for (int64_t y = 0; y < t.h && y0 + y < out_h_; ++y) {
                    float* row = canvas + (y0 + y) * out_w_ + x0;
                    for (int64_t x = 0; x < t.w && x0 + x < out_w_; ++x) {
                        float value = cur[y * t.w + x];
                        if (y < ext_h) {
                            const float wy = static_cast<float>(y) / ext_h;
                            value          = row[x] * (1.f - wy) + value * wy;
                        }
                        if (x < ext_w) {
                            const float wx = static_cast<float>(x) / ext_w;
                            value          = tail[y * tail_w_ + (tail_w_ - ext_w + x)] * (1.f - wx) + value * wx;
                        }
                        if (x < crop_w) {
                            row[x] = value;
                        } else {
                            next_tail[y * new_tw + (x - crop_w)] = value;
                        }
                    }
                }
            }
            tail_.swap(next_tail_);
            tail_w_ = new_tw;
            tail_h_ = t.h;
        }

    private:
        Axis rows_;
        Axis cols_;
        int64_t planes_;
        float* out_;
        int64_t out_w_;
        int64_t out_h_;
        std::vector<float> tail_;
        std::vector<float> next_tail_;
        int64_t tail_w_ = 0;
        int64_t tail_h_ = 0;
    };

}  // namespace H3VAETiles

#endif  // __SD_MODEL_VAE_H3_VAE_TILES_HPP__
