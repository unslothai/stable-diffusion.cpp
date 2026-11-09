// Ported from NVIDIA cosmos-framework (data/generator/sequence_packing/mrope.py,
// autoregressive.py, model/generator/omni_mot_causal_model.py).
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the OpenMDW License 1.1 (https://openmdw.ai/license/1-1/);
// see licenses/OpenMDW-1.1.txt and NOTICE.

#ifndef __SD_PIPELINE_COSMOS3_TRANSFER_UTILS_H__
#define __SD_PIPELINE_COSMOS3_TRANSFER_UTILS_H__

#include <cstdint>
#include <vector>

namespace sd::cosmos3 {

    constexpr int kTemporalModalityMargin = 15000;
    constexpr float kBaseFps              = 24.f;
    constexpr uint64_t kFramePrime        = 1000003ULL;
    constexpr uint64_t kStepPrime         = 9176ULL;

    // Fixed few-step schedule of the distilled SDE sampler, ending at zero.
    inline std::vector<float> distilled_sigmas() {
        return {1.0f, 0.9375f, static_cast<float>(0.8333333333333334), 0.625f, 0.0f};
    }

    // Noise seed for the reinjection after step `step` of latent frame `frame`.
    inline uint64_t sde_noise_seed(int64_t seed, int64_t frame, int step) {
        return static_cast<uint64_t>(seed) + static_cast<uint64_t>(frame) * kFramePrime + static_cast<uint64_t>(step + 1) * kStepPrime;
    }

    // One distilled step on flat float buffers: x0 = x - sigma * v, then either the
    // final x0 or the stochastic re-noise to the next sigma.
    inline void distilled_step(std::vector<float>& x,
                               const std::vector<float>& velocity,
                               float sigma,
                               float sigma_next,
                               const std::vector<float>* noise) {
        for (size_t i = 0; i < x.size(); ++i) {
            const float x0 = x[i] - sigma * velocity[i];
            x[i]           = sigma_next == 0.f ? x0 : (1.f - sigma_next) * x0 + sigma_next * (*noise)[i];
        }
    }

    struct Grid {
        int64_t latent_w = 0;
        int64_t latent_h = 0;
        int64_t channels = 0;
        int64_t patch    = 2;

        int64_t patch_w() const { return (latent_w + patch - 1) / patch; }
        int64_t patch_h() const { return (latent_h + patch - 1) / patch; }
        int64_t tokens() const { return patch_w() * patch_h(); }
        int64_t patch_dim() const { return channels * patch * patch; }
    };

    // frame: one latent frame laid out [C][H][W] (W fastest). Returns tokens-major
    // patches [token][(p * patch + q) * C + c], token = hp * patch_w + wp, zero-padded.
    inline std::vector<float> patchify(const float* frame, const Grid& g) {
        std::vector<float> out(static_cast<size_t>(g.tokens() * g.patch_dim()), 0.f);
        for (int64_t c = 0; c < g.channels; ++c) {
            for (int64_t h = 0; h < g.latent_h; ++h) {
                for (int64_t w = 0; w < g.latent_w; ++w) {
                    const int64_t token   = (h / g.patch) * g.patch_w() + (w / g.patch);
                    const int64_t feature = ((h % g.patch) * g.patch + (w % g.patch)) * g.channels + c;
                    out[static_cast<size_t>(token * g.patch_dim() + feature)] =
                        frame[(c * g.latent_h + h) * g.latent_w + w];
                }
            }
        }
        return out;
    }

    inline std::vector<float> unpatchify(const float* patches, const Grid& g) {
        std::vector<float> out(static_cast<size_t>(g.channels * g.latent_h * g.latent_w));
        for (int64_t c = 0; c < g.channels; ++c) {
            for (int64_t h = 0; h < g.latent_h; ++h) {
                for (int64_t w = 0; w < g.latent_w; ++w) {
                    const int64_t token   = (h / g.patch) * g.patch_w() + (w / g.patch);
                    const int64_t feature = ((h % g.patch) * g.patch + (w % g.patch)) * g.channels + c;
                    out[static_cast<size_t>((c * g.latent_h + h) * g.latent_w + w)] =
                        patches[token * g.patch_dim() + feature];
                }
            }
        }
        return out;
    }

    // mRoPE ids as [axis][token]: text uses the same running index on all axes.
    inline std::vector<float> text_positions(int64_t length) {
        std::vector<float> ids(static_cast<size_t>(3 * length));
        for (int axis = 0; axis < 3; ++axis) {
            for (int64_t i = 0; i < length; ++i) {
                ids[static_cast<size_t>(axis * length + i)] = static_cast<float>(i);
            }
        }
        return ids;
    }

    // Control frame t and target frame t share these ids; spatial ids restart at 0.
    inline std::vector<float> frame_positions(const Grid& g, int64_t text_length, int64_t frame, float fps) {
        const double stride = static_cast<double>(kBaseFps) / static_cast<double>(fps);
        const float t       = static_cast<float>(static_cast<double>(text_length + kTemporalModalityMargin) + frame * stride);
        const int64_t n     = g.tokens();
        std::vector<float> ids(static_cast<size_t>(3 * n));
        for (int64_t hp = 0; hp < g.patch_h(); ++hp) {
            for (int64_t wp = 0; wp < g.patch_w(); ++wp) {
                const int64_t i                     = hp * g.patch_w() + wp;
                ids[static_cast<size_t>(i)]         = t;
                ids[static_cast<size_t>(n + i)]     = static_cast<float>(hp);
                ids[static_cast<size_t>(2 * n + i)] = static_cast<float>(wp);
            }
        }
        return ids;
    }

    // Each latent frame adds a clean control and a clean RGB entry; a finite window pins the first
    // `sink` pairs and keeps only what the next forward reads (the reference KV ring).
    struct HistoryPolicy {
        int64_t window_frames = 0;  // 0 keeps the full history
        int64_t sink_frames   = 0;

        int64_t recent_pairs() const {
            return window_frames - sink_frames - 1;
        }

        // Entries kept after caching control frame t (the target denoise reads them).
        int64_t keep_after_control() const {
            return window_frames > 0 ? 2 * recent_pairs() + 1 : -1;
        }

        // Entries kept after caching the clean RGB frame (the next control reads them).
        int64_t keep_after_refresh() const {
            return window_frames > 0 ? 2 * recent_pairs() : -1;
        }

        int64_t sink_entries() const {
            return window_frames > 0 ? 2 * sink_frames : 0;
        }

        bool valid() const {
            // A 1-frame window would carry no history between latent frames.
            return window_frames == 0 || (window_frames >= 2 && sink_frames >= 0 && recent_pairs() >= 0);
        }
    };

}  // namespace sd::cosmos3

#endif  // __SD_PIPELINE_COSMOS3_TRANSFER_UTILS_H__
