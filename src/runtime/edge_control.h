#ifndef __SD_RUNTIME_EDGE_CONTROL_H__
#define __SD_RUNTIME_EDGE_CONTROL_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sd::edge {

    struct GradientEdgeParams {
        // Magnitudes at or above this percentile map to 255; <= 0 normalizes by the maximum.
        float clip_percentile = 99.f;
        // Values above this level (0..1 after normalization) become 255 and the rest 0; <= 0 keeps the soft map.
        float binary_threshold = 0.f;
        // Keep only local maxima across the edge (Canny-style non-maximum suppression), giving 1 px lines.
        bool thin = false;
    };

    // Grayscale, 5-tap binomial blur, Sobel x/y, gradient magnitude, normalized to 0..255.
    // rgb is interleaved 8-bit RGB with `channels` (3 or 4) bytes per pixel; out is width*height bytes.
    inline void gradient_magnitude(const uint8_t* rgb,
                                   int width,
                                   int height,
                                   int channels,
                                   uint8_t* out,
                                   const GradientEdgeParams& params = {}) {
        const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
        std::vector<float> gray(n);
        std::vector<float> tmp(n);
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* p = rgb + i * static_cast<size_t>(channels);
            gray[i]          = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
        }

        auto clamp_x = [width](int x) { return std::min(std::max(x, 0), width - 1); };
        auto clamp_y = [height](int y) { return std::min(std::max(y, 0), height - 1); };

        static const float kBlur[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
        for (int y = 0; y < height; ++y) {
            const float* row = gray.data() + static_cast<size_t>(y) * width;
            float* dst       = tmp.data() + static_cast<size_t>(y) * width;
            for (int x = 0; x < width; ++x) {
                float acc = 0.f;
                for (int k = -2; k <= 2; ++k) {
                    acc += kBlur[k + 2] * row[clamp_x(x + k)];
                }
                dst[x] = acc;
            }
        }
        for (int y = 0; y < height; ++y) {
            float* dst = gray.data() + static_cast<size_t>(y) * width;
            for (int x = 0; x < width; ++x) {
                float acc = 0.f;
                for (int k = -2; k <= 2; ++k) {
                    acc += kBlur[k + 2] * tmp[static_cast<size_t>(clamp_y(y + k)) * width + x];
                }
                dst[x] = acc;
            }
        }

        std::vector<uint8_t> dir;
        if (params.thin) {
            dir.resize(n);
        }
        float max_mag = 0.f;
        for (int y = 0; y < height; ++y) {
            const float* up   = gray.data() + static_cast<size_t>(clamp_y(y - 1)) * width;
            const float* mid  = gray.data() + static_cast<size_t>(y) * width;
            const float* down = gray.data() + static_cast<size_t>(clamp_y(y + 1)) * width;
            float* dst        = tmp.data() + static_cast<size_t>(y) * width;
            for (int x = 0; x < width; ++x) {
                const int xl   = clamp_x(x - 1);
                const int xr   = clamp_x(x + 1);
                const float gx = (up[xr] + 2.f * mid[xr] + down[xr]) - (up[xl] + 2.f * mid[xl] + down[xl]);
                const float gy = (down[xl] + 2.f * down[x] + down[xr]) - (up[xl] + 2.f * up[x] + up[xr]);
                dst[x]         = std::sqrt(gx * gx + gy * gy);
                max_mag        = std::max(max_mag, dst[x]);
                if (params.thin) {
                    // 0: horizontal gradient, 1: 45 deg, 2: vertical, 3: 135 deg (tan 22.5 = 0.4142).
                    const float ax = std::fabs(gx), ay = std::fabs(gy);
                    uint8_t d = 0;
                    if (ay > 2.4142f * ax) {
                        d = 2;
                    } else if (ay > 0.4142f * ax) {
                        d = (gx > 0.f) == (gy > 0.f) ? 1 : 3;
                    }
                    dir[static_cast<size_t>(y) * width + x] = d;
                }
            }
        }
        // Percentile over the pixels that have any gradient, before thinning: flat
        // areas and suppressed pixels would otherwise pull it to zero on sparse frames.
        float scale_ref = max_mag;
        if (params.clip_percentile > 0.f && params.clip_percentile < 100.f && max_mag > 0.f) {
            constexpr int kBins = 2048;
            std::vector<uint32_t> hist(kBins, 0);
            const float to_bin = (kBins - 1) / max_mag;
            size_t nonzero     = 0;
            for (size_t i = 0; i < n; ++i) {
                if (tmp[i] > 0.f) {
                    hist[static_cast<size_t>(tmp[i] * to_bin)]++;
                    nonzero++;
                }
            }
            const double target = nonzero * (params.clip_percentile / 100.0);
            double seen         = 0.0;
            for (int b = 0; b < kBins; ++b) {
                seen += hist[b];
                if (seen >= target) {
                    scale_ref = (b + 1) / to_bin;
                    break;
                }
            }
        }
        if (params.thin) {
            static const int kOffsets[4][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}};
            std::vector<float> thinned(n, 0.f);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const size_t i = static_cast<size_t>(y) * width + x;
                    const int* o   = kOffsets[dir[i]];
                    const float a  = tmp[static_cast<size_t>(clamp_y(y + o[1])) * width + clamp_x(x + o[0])];
                    const float b  = tmp[static_cast<size_t>(clamp_y(y - o[1])) * width + clamp_x(x - o[0])];
                    if (tmp[i] >= a && tmp[i] > b) {
                        thinned[i] = tmp[i];
                    }
                }
            }
            tmp.swap(thinned);
        }

        const float inv = scale_ref > 0.f ? 1.f / scale_ref : 0.f;
        for (size_t i = 0; i < n; ++i) {
            float v = std::min(tmp[i] * inv, 1.f);
            if (params.binary_threshold > 0.f) {
                v = v > params.binary_threshold ? 1.f : 0.f;
            }
            out[i] = static_cast<uint8_t>(std::lround(v * 255.f));
        }
    }

}  // namespace sd::edge

#endif  // __SD_RUNTIME_EDGE_CONTROL_H__
