// Cosmos3-Nano-Sim-Transfer: control video + prompt -> photoreal video, generated
// one latent frame at a time with a KV cache (causal streaming transfer).
//
// Ported from NVIDIA cosmos-framework (model/generator/omni_mot_causal_model.py,
// iter_samples_from_batch_autoregressive, video_transfer mode).
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the OpenMDW License 1.1 (https://openmdw.ai/license/1-1/);
// see licenses/OpenMDW-1.1.txt and NOTICE.

#include "cosmos3_transfer.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/util.h"
#include "diffusion_engine.h"
#include "model/diffusion/cosmos3.hpp"
#include "pipeline/cosmos3_prompt.h"
#include "pipeline/cosmos3_transfer_utils.h"
#include "runtime/edge_control.h"
#include "tokenizers/qwen2_tokenizer.h"

namespace sd::pipeline {

    // Test hooks for numerical comparison against the reference implementation.
    static const char* debug_dir() {
        const char* dir = std::getenv("SD_COSMOS3_DEBUG_DIR");
        return dir != nullptr && dir[0] != '\0' ? dir : nullptr;
    }

    template <typename T>
    static bool read_raw(const std::string& path, std::vector<T>* out, size_t expected) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            return false;
        }
        std::fseek(f, 0, SEEK_END);
        const long bytes = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (bytes < 0 || (expected > 0 && static_cast<size_t>(bytes) != expected * sizeof(T))) {
            std::fclose(f);
            LOG_WARN("cosmos3 debug: %s has %ld bytes, expected %zu values", path.c_str(), bytes, expected);
            return false;
        }
        out->resize(static_cast<size_t>(bytes) / sizeof(T));
        const size_t read = std::fread(out->data(), sizeof(T), out->size(), f);
        std::fclose(f);
        return read == out->size();
    }

    static void write_raw(const std::string& path, const float* data, size_t count) {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) {
            return;
        }
        std::fwrite(data, sizeof(float), count, f);
        std::fclose(f);
    }

    static std::string debug_path(const char* dir, const std::string& name) {
        return std::string(dir) + "/" + name;
    }

    static float round_bf16(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        bits &= 0xFFFF0000u;
        std::memcpy(&value, &bits, sizeof(bits));
        return value;
    }

    static std::vector<float> randn(const std::shared_ptr<RNG>& rng, uint64_t seed, size_t count, bool bf16) {
        rng->manual_seed(seed);
        std::vector<float> values = rng->randn(static_cast<uint32_t>(count));
        if (bf16) {
            for (float& v : values) {
                v = round_bf16(v);
            }
        }
        return values;
    }

    static bool tokenize_prompt(const std::string& caption, std::vector<int32_t>* ids) {
        static const char* kSystemPrompt =
            "You are a helpful assistant that generates images or videos following the user's instructions "
            "and control signals (edge maps, blur, depth, or segmentation).";
        const std::string text = std::string("<|im_start|>system\n") + kSystemPrompt + "<|im_end|>\n" +
                                 "<|im_start|>user\n" + caption + "<|im_end|>\n" +
                                 "<|im_start|>assistant\n";
        Qwen2Tokenizer tokenizer;
        std::vector<int> tokens;
        std::string error;
        if (!tokenizer.encode(text, tokens, nullptr, &error)) {
            LOG_ERROR("cosmos3: prompt tokenization failed: %s", error.c_str());
            return false;
        }
        ids->assign(tokens.begin(), tokens.end());
        return true;
    }

    // [W, H, 3, 1] in [0, 1]: gray frames are replicated, alpha is dropped.
    static sd::Tensor<float> rgb_frame(const sd_image_t& image, int width, int height) {
        auto frame             = sd_image_to_tensor(image, width, height);
        const int64_t channels = frame.shape()[2];
        if (channels == 3) {
            return frame;
        }
        sd::Tensor<float> rgb = sd::zeros<float>({width, height, 3, 1});
        const size_t plane    = static_cast<size_t>(width) * height;
        for (int64_t c = 0; c < 3; ++c) {
            const float* src = frame.data() + static_cast<size_t>(channels >= 3 ? c : 0) * plane;
            std::copy(src, src + plane, rgb.data() + c * plane);
        }
        return rgb;
    }

    // Control video in [0, 1], [W, H, T, 3, 1]. Precomputed control frames win;
    // otherwise an edge map is derived from the reference video's RGB frames.
    static sd::Tensor<float> build_control_video(const sd_vid_gen_params_t* params, int width, int height, int frames, float edge_threshold) {
        sd::Tensor<float> video = sd::zeros<float>({width, height, frames, 3, 1});
        if (params->control_frames_size > 0) {
            if (params->control_frames_size < frames) {
                LOG_ERROR("cosmos3: %d control frames given, %d needed", params->control_frames_size, frames);
                return {};
            }
            for (int i = 0; i < frames; ++i) {
                if (params->control_frames[i].channel == 0) {
                    LOG_ERROR("cosmos3: control frame %d has no channels", i);
                    return {};
                }
                auto frame = rgb_frame(params->control_frames[i], width, height);
                sd::ops::slice_assign(&video, 2, i, i + 1, frame.unsqueeze(2));
            }
            return video;
        }
        if (params->ref_videos_count > 0 && params->ref_videos[0].frame_count >= frames) {
            const sd_ref_video_t& source = params->ref_videos[0];
            std::vector<uint8_t> edges(static_cast<size_t>(width) * height);
            sd::edge::GradientEdgeParams edge_params;
            edge_params.thin             = edge_threshold > 0.f;
            edge_params.binary_threshold = edge_threshold;
            const int64_t edge_start     = ggml_time_ms();
            const size_t plane           = static_cast<size_t>(width) * height;
            std::vector<uint8_t> pixels;
            for (int i = 0; i < frames; ++i) {
                const sd_image_t& frame = source.frames[i];
                const uint8_t* rgb      = frame.data;
                if (frame.channel == 0) {
                    LOG_ERROR("cosmos3: reference video frame %d has no channels", i);
                    return {};
                }
                if (static_cast<int>(frame.width) != width || static_cast<int>(frame.height) != height || frame.channel != 3) {
                    auto resized = rgb_frame(frame, width, height);
                    pixels.resize(plane * 3);
                    for (size_t p = 0; p < plane; ++p) {
                        for (int c = 0; c < 3; ++c) {
                            pixels[p * 3 + c] = static_cast<uint8_t>(std::lround(std::min(std::max(resized.data()[c * plane + p], 0.f), 1.f) * 255.f));
                        }
                    }
                    rgb = pixels.data();
                }
                sd::edge::gradient_magnitude(rgb, width, height, 3, edges.data(), edge_params);
                for (int c = 0; c < 3; ++c) {
                    float* dst = video.data() + (static_cast<size_t>(c) * frames + i) * plane;
                    for (size_t p = 0; p < plane; ++p) {
                        dst[p] = edges[p] / 255.f;
                    }
                }
            }
            LOG_INFO("cosmos3: edge control for %d frames in %" PRId64 " ms", frames, ggml_time_ms() - edge_start);
            return video;
        }
        LOG_ERROR("cosmos3: Sim-Transfer needs a control video (control frames) or a reference video to derive edges from");
        return {};
    }

    bool generate_cosmos3_transfer_video(StableDiffusionGGML* sd,
                                         const sd_vid_gen_params_t* params,
                                         const GenerationRequest& request,
                                         sd_image_t** frames_out,
                                         int* num_frames_out) {
        auto runner = std::dynamic_pointer_cast<Cosmos3::Cosmos3Runner>(sd->diffusion_model);
        if (runner == nullptr) {
            LOG_ERROR("cosmos3: diffusion model is not a Cosmos3 transformer");
            return false;
        }
        sd::cosmos3::HistoryPolicy history;
        history.window_frames = runner->window_frames;
        history.sink_frames   = runner->sink_frames;
        if (!history.valid()) {
            LOG_ERROR("cosmos3: invalid KV window (window %" PRId64 ", sink %" PRId64 "): use 0, or 2 or more with sink < window", history.window_frames, history.sink_frames);
            return false;
        }
        const char* dbg = debug_dir();
        if (const char* stop = std::getenv("SD_COSMOS3_DEBUG_STOP_LAYER"); dbg != nullptr && stop != nullptr) {
            runner->debug_stop_layer = std::strtoll(stop, nullptr, 10);
        }
        const int width  = request.width;
        const int height = request.height;
        const int frames = request.frames;
        const int fps    = request.fps > 0 ? request.fps : 30;
        const int64_t t0 = ggml_time_ms();

        sd::Tensor<float> control = build_control_video(params, width, height, frames, runner->edge_threshold);
        if (control.empty()) {
            return false;
        }
        if (dbg != nullptr) {
            // Channel 0 of the control video as [T, H, W] uint8.
            const size_t count = static_cast<size_t>(width) * height * frames;
            std::vector<uint8_t> bytes(count);
            for (size_t i = 0; i < count; ++i) {
                bytes[i] = static_cast<uint8_t>(std::lround(std::min(std::max(control.data()[i], 0.f), 1.f) * 255.f));
            }
            FILE* f = std::fopen(debug_path(dbg, "port_control_u8.bin").c_str(), "wb");
            if (f != nullptr) {
                std::fwrite(bytes.data(), 1, bytes.size(), f);
                std::fclose(f);
            }
        }

        sd::Tensor<float> control_latents;
        std::vector<float> dbg_latents;
        const int64_t latent_w = width / request.vae_scale_factor;
        const int64_t latent_h = height / request.vae_scale_factor;
        const int64_t latent_t = (frames - 1) / 4 + 1;
        const int64_t channels = runner->config.latent_channels;
        if (dbg != nullptr && read_raw(debug_path(dbg, "control_latents.bin"), &dbg_latents, static_cast<size_t>(latent_w * latent_h * latent_t * channels))) {
            LOG_INFO("cosmos3 debug: using control latents from %s", dbg);
            control_latents = sd::Tensor<float>::from_vector(dbg_latents);
            control_latents.reshape_({latent_w, latent_h, latent_t, channels, 1});
        } else {
            control_latents = sd->encode_first_stage(control);
            if (control_latents.empty()) {
                LOG_ERROR("cosmos3: control video encode failed");
                return false;
            }
            if (dbg != nullptr) {
                write_raw(debug_path(dbg, "port_control_latents.bin"), control_latents.data(), static_cast<size_t>(control_latents.numel()));
            }
        }
        const int64_t t_encoded = ggml_time_ms();
        LOG_INFO("cosmos3: control video encoded to %" PRId64 "x%" PRId64 "x%" PRId64 " latents in %.2fs",
                 control_latents.shape()[0], control_latents.shape()[1], control_latents.shape()[2],
                 (t_encoded - t0) / 1000.f);

        std::vector<int32_t> text_ids;
        std::vector<int32_t> dbg_ids;
        if (dbg != nullptr && read_raw(debug_path(dbg, "text_ids.bin"), &dbg_ids, 0)) {
            LOG_INFO("cosmos3 debug: using %zu text token ids from %s", dbg_ids.size(), dbg);
            text_ids = dbg_ids;
        } else if (!tokenize_prompt(sd::cosmos3::format_prompt(request.prompt, frames, static_cast<float>(fps), height, width), &text_ids)) {
            return false;
        }
        // Split layout: caption tokens, EOS, start-of-generation.
        if (dbg_ids.empty()) {
            text_ids.push_back(151645);
            text_ids.push_back(151652);
            if (dbg != nullptr) {
                FILE* f = std::fopen(debug_path(dbg, "port_text_ids.bin").c_str(), "wb");
                if (f != nullptr) {
                    std::fwrite(text_ids.data(), sizeof(int32_t), text_ids.size(), f);
                    std::fclose(f);
                }
            }
        }
        const int64_t text_len = static_cast<int64_t>(text_ids.size());
        sd::Tensor<int32_t> text_tensor({text_len, 1});
        std::copy(text_ids.begin(), text_ids.end(), text_tensor.data());

        sd::cosmos3::Grid grid;
        grid.latent_w            = control_latents.shape()[0];
        grid.latent_h            = control_latents.shape()[1];
        grid.channels            = channels;
        grid.patch               = runner->config.patch_size;
        const int64_t n_latent   = control_latents.shape()[2];
        const int64_t tokens     = grid.tokens();
        const size_t frame_numel = static_cast<size_t>(grid.channels * grid.latent_h * grid.latent_w);

        Cosmos3::HistoryUpdate after_control;
        after_control.write                  = true;
        after_control.sink_tokens            = history.sink_entries() * tokens;
        after_control.keep_recent_tokens     = history.keep_after_control() < 0 ? -1 : history.keep_after_control() * tokens;
        Cosmos3::HistoryUpdate after_refresh = after_control;
        after_refresh.keep_recent_tokens     = history.keep_after_refresh() < 0 ? -1 : history.keep_after_refresh() * tokens;

        const std::vector<float> sigmas = sd::cosmos3::distilled_sigmas();
        const sd::Tensor<int32_t> no_text;
        const std::vector<float> text_positions = sd::cosmos3::text_positions(text_len);
        sd::Tensor<float> output                = sd::zeros<float>({grid.latent_w, grid.latent_h, n_latent, grid.channels, 1});

        runner->reset_stream();
        int64_t denoise_ms = 0;
        for (int64_t t = 0; t < n_latent; ++t) {
            if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
                LOG_ERROR("cosmos3: cancelled at latent frame %" PRId64, t);
                runner->reset_stream();
                return false;
            }
            const int64_t chunk_start = ggml_time_ms();
            const auto positions      = sd::cosmos3::frame_positions(grid, text_len, t, static_cast<float>(fps));

            auto control_frame = sd::ops::slice(control_latents, 2, t, t + 1);
            auto control_patch = sd::Tensor<float>::from_vector(sd::cosmos3::patchify(control_frame.data(), grid));
            control_patch.reshape_({grid.patch_dim(), tokens});
            if (runner->forward(sd->n_threads, control_patch, positions, -1.f,
                                t == 0 ? text_tensor : no_text, text_positions, after_control)
                    .empty()) {
                LOG_ERROR("cosmos3: caching control frame %" PRId64 " failed", t);
                runner->reset_stream();
                return false;
            }

            std::vector<float> x;
            const std::string noise0 = "noise_c" + std::to_string(t) + "_s0.bin";
            if (dbg == nullptr || !read_raw(debug_path(dbg, noise0), &x, frame_numel)) {
                x = randn(sd->rng, static_cast<uint64_t>(request.seed + t), frame_numel, true);
            }
            for (size_t step = 0; step + 1 < sigmas.size(); ++step) {
                auto patches = sd::Tensor<float>::from_vector(sd::cosmos3::patchify(x.data(), grid));
                patches.reshape_({grid.patch_dim(), tokens});
                auto pred = runner->forward(sd->n_threads, patches, positions, sigmas[step], no_text, text_positions, {});
                if (pred.empty()) {
                    LOG_ERROR("cosmos3: denoise failed at frame %" PRId64 " step %zu", t, step);
                    runner->reset_stream();
                    return false;
                }
                if (runner->debug_stop_layer >= 0) {
                    write_raw(debug_path(dbg, "port_hidden_l" + std::to_string(runner->debug_stop_layer) + ".bin"),
                              pred.data(), static_cast<size_t>(pred.numel()));
                    runner->debug_stop_layer = -1;
                    runner->reset_stream();
                    return false;
                }
                std::vector<float> velocity = sd::cosmos3::unpatchify(pred.data(), grid);
                if (dbg != nullptr && t < 2) {
                    write_raw(debug_path(dbg, "port_velocity_c" + std::to_string(t) + "_s" + std::to_string(step) + ".bin"),
                              velocity.data(), velocity.size());
                }
                std::vector<float> noise;
                if (sigmas[step + 1] != 0.f) {
                    const std::string name = "noise_c" + std::to_string(t) + "_s" + std::to_string(step + 1) + ".bin";
                    if (dbg == nullptr || !read_raw(debug_path(dbg, name), &noise, frame_numel)) {
                        noise = randn(sd->rng, sd::cosmos3::sde_noise_seed(request.seed, t, static_cast<int>(step)), frame_numel, false);
                    }
                }
                sd::cosmos3::distilled_step(x, velocity, sigmas[step], sigmas[step + 1], noise.empty() ? nullptr : &noise);
            }

            auto clean = sd::Tensor<float>::from_vector(x);
            clean.reshape_({grid.latent_w, grid.latent_h, 1, grid.channels, 1});
            sd::ops::slice_assign(&output, 2, t, t + 1, clean);

            if (t + 1 < n_latent) {
                std::vector<float> clean_bf16 = x;
                for (float& v : clean_bf16) {
                    v = round_bf16(v);
                }
                auto clean_patch = sd::Tensor<float>::from_vector(sd::cosmos3::patchify(clean_bf16.data(), grid));
                clean_patch.reshape_({grid.patch_dim(), tokens});
                if (runner->forward(sd->n_threads, clean_patch, positions, -1.f, no_text, text_positions, after_refresh).empty()) {
                    LOG_ERROR("cosmos3: caching generated frame %" PRId64 " failed", t);
                    runner->reset_stream();
                    return false;
                }
            }
            const int64_t chunk_ms = ggml_time_ms() - chunk_start;
            denoise_ms += chunk_ms;
            LOG_INFO("cosmos3: latent frame %" PRId64 "/%" PRId64 " in %" PRId64 " ms", t + 1, n_latent, chunk_ms);
            if (auto cb = sd_get_progress_callback()) {
                cb(static_cast<int>(t + 1), static_cast<int>(n_latent), chunk_ms / 1000.f, sd_get_progress_callback_data());
            }
        }
        runner->reset_stream();
        runner->runner_end();

        if (dbg != nullptr) {
            write_raw(debug_path(dbg, "port_output_latents.bin"), output.data(), static_cast<size_t>(output.numel()));
            std::vector<float> ref_latents;
            if (std::getenv("SD_COSMOS3_DEBUG_DECODE_REF") != nullptr &&
                read_raw(debug_path(dbg, "ref_output_latents.bin"), &ref_latents, static_cast<size_t>(output.numel()))) {
                LOG_INFO("cosmos3 debug: decoding the reference latents instead");
                std::copy(ref_latents.begin(), ref_latents.end(), output.data());
            }
        }
        const float pixel_frames = static_cast<float>((n_latent - 1) * 4 + 1);
        LOG_INFO("cosmos3: %" PRId64 " latent frames in %.2fs (%.1f ms per latent frame, %.2f output fps excluding VAE)",
                 n_latent, denoise_ms / 1000.f, static_cast<float>(denoise_ms) / n_latent, pixel_frames * 1000.f / denoise_ms);

        if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
            LOG_ERROR("cosmos3: cancelled before video decode");
            return false;
        }
        const int64_t decode_start = ggml_time_ms();
        sd::Tensor<float> video    = sd->decode_first_stage(output, true);
        if (video.empty()) {
            LOG_ERROR("cosmos3: decode failed");
            return false;
        }
        LOG_INFO("cosmos3: decoded in %.2fs, total %.2fs", (ggml_time_ms() - decode_start) / 1000.f, (ggml_time_ms() - t0) / 1000.f);
        if (video.shape()[2] > frames) {
            video = sd::ops::slice(video, 2, 0, frames);
        }
        if (dbg != nullptr) {
            // [T, H, W, 3] uint8, the layout of the reference frame dump.
            const int64_t W = video.shape()[0], H = video.shape()[1], T = video.shape()[2];
            std::vector<uint8_t> bytes(static_cast<size_t>(T * H * W * 3));
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t y = 0; y < H; ++y) {
                    for (int64_t x = 0; x < W; ++x) {
                        for (int64_t c = 0; c < 3; ++c) {
                            const float v                                             = std::min(std::max(video.index(x, y, t, c, 0), 0.f), 1.f);
                            bytes[static_cast<size_t>(((t * H + y) * W + x) * 3 + c)] = static_cast<uint8_t>(v * 255.f);
                        }
                    }
                }
            }
            FILE* f = std::fopen(debug_path(dbg, "port_frames_u8.bin").c_str(), "wb");
            if (f != nullptr) {
                std::fwrite(bytes.data(), 1, bytes.size(), f);
                std::fclose(f);
            }
        }
        const int count    = static_cast<int>(video.shape()[2]);
        sd_image_t* images = static_cast<sd_image_t*>(calloc(static_cast<size_t>(count), sizeof(sd_image_t)));
        if (images == nullptr) {
            return false;
        }
        for (int i = 0; i < count; ++i) {
            images[i] = tensor_to_sd_image(video, i);
        }
        if (num_frames_out != nullptr) {
            *num_frames_out = count;
        }
        if (frames_out == nullptr) {
            for (int i = 0; i < count; ++i) {
                free(images[i].data);
            }
            free(images);
            return true;
        }
        *frames_out = images;
        return true;
    }

}  // namespace sd::pipeline
