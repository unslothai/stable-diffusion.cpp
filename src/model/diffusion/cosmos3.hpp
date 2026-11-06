// Cosmos3 omni transformer (Mixture-of-Transformers) with the causal streaming
// mode used by Cosmos3-Nano-Sim-Transfer.
//
// Ported from NVIDIA cosmos-framework (cosmos_framework/model/generator/
// omni_mot_causal_model.py, mot/unified_mot.py, data/generator/sequence_packing/)
// and the diffusers Cosmos3OmniTransformer layout of the published checkpoint.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the OpenMDW License 1.1 (https://openmdw.ai/license/1-1/);
// see licenses/OpenMDW-1.1.txt and NOTICE.

#ifndef __SD_MODEL_DIFFUSION_COSMOS3_HPP__
#define __SD_MODEL_DIFFUSION_COSMOS3_HPP__

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "core/ggml_extend.h"
#include "core/util.h"
#include "model/common/rope.hpp"
#include "model/diffusion/model.hpp"
#include "model/te/llm.hpp"
#include "model_loader.h"

namespace Cosmos3 {
    constexpr int COSMOS3_GRAPH_SIZE = 327680;

    struct Cosmos3Config {
        int64_t hidden_size       = 4096;
        int64_t intermediate_size = 12288;
        int64_t num_layers        = 36;
        int64_t num_heads         = 32;
        int64_t num_kv_heads      = 8;
        int64_t head_dim          = 128;
        int64_t vocab_size        = 151936;
        int64_t latent_channels   = 48;
        int64_t patch_size        = 2;
        float rms_norm_eps        = 1e-6f;
        float rope_theta          = 5000000.f;
        float timestep_scale      = 0.001f;
        std::vector<int> mrope_section{24, 20, 20};

        int64_t patch_dim() const {
            return latent_channels * patch_size * patch_size;
        }

        static Cosmos3Config detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                 const std::string& prefix) {
            Cosmos3Config config;
            config.num_layers      = 0;
            const std::string root = prefix.empty() ? "" : prefix + ".";
            for (const auto& [name, storage] : tensor_storage_map) {
                if (!starts_with(name, root)) {
                    continue;
                }
                const std::string key = name.substr(root.size());
                if (key == "embed_tokens.weight" && storage.n_dims == 2) {
                    config.hidden_size = storage.ne[0];
                    config.vocab_size  = storage.ne[1];
                } else if (key == "layers.0.mlp.gate_proj.weight" && storage.n_dims == 2) {
                    config.intermediate_size = storage.ne[1];
                } else if (key == "layers.0.self_attn.norm_q.weight") {
                    config.head_dim = storage.ne[0];
                } else if (key == "proj_in.weight" && storage.n_dims == 2) {
                    config.latent_channels = storage.ne[0] / (config.patch_size * config.patch_size);
                }
                if (starts_with(key, "layers.")) {
                    config.num_layers = std::max<int64_t>(config.num_layers, std::strtoll(key.c_str() + 7, nullptr, 10) + 1);
                }
            }
            for (const auto& [name, storage] : tensor_storage_map) {
                if (name == root + "layers.0.self_attn.to_q.weight") {
                    config.num_heads = storage.ne[1] / config.head_dim;
                } else if (name == root + "layers.0.self_attn.to_k.weight") {
                    config.num_kv_heads = storage.ne[1] / config.head_dim;
                }
            }
            if (config.num_layers == 0) {
                config.num_layers = 36;
            }
            LOG_DEBUG("cosmos3: layers=%" PRId64 ", hidden=%" PRId64 ", heads=%" PRId64 ", kv_heads=%" PRId64 ", latent=%" PRId64,
                      config.num_layers, config.hidden_size, config.num_heads, config.num_kv_heads, config.latent_channels);
            return config;
        }
    };

    // Holds the K/V the next gen forward needs: text keys of the understanding
    // path and the cached clean control/RGB frames, both as [d_head, tokens, kv_heads].
    struct LayerKV {
        ggml_tensor* und_k  = nullptr;
        ggml_tensor* und_v  = nullptr;
        ggml_tensor* hist_k = nullptr;
        ggml_tensor* hist_v = nullptr;
    };

    struct HistoryUpdate {
        bool write                 = false;
        int64_t sink_tokens        = 0;
        int64_t keep_recent_tokens = -1;  // < 0 keeps every cached token
    };

    struct Attention : public GGMLBlock {
        Cosmos3Config config;

        explicit Attention(const Cosmos3Config& config)
            : config(config) {
            const int64_t q_dim    = config.num_heads * config.head_dim;
            const int64_t kv_dim   = config.num_kv_heads * config.head_dim;
            blocks["to_q"]         = std::make_shared<Linear>(config.hidden_size, q_dim, false);
            blocks["to_k"]         = std::make_shared<Linear>(config.hidden_size, kv_dim, false);
            blocks["to_v"]         = std::make_shared<Linear>(config.hidden_size, kv_dim, false);
            blocks["to_out"]       = std::make_shared<Linear>(q_dim, config.hidden_size, false);
            blocks["add_q_proj"]   = std::make_shared<Linear>(config.hidden_size, q_dim, false);
            blocks["add_k_proj"]   = std::make_shared<Linear>(config.hidden_size, kv_dim, false);
            blocks["add_v_proj"]   = std::make_shared<Linear>(config.hidden_size, kv_dim, false);
            blocks["to_add_out"]   = std::make_shared<Linear>(q_dim, config.hidden_size, false);
            blocks["norm_q"]       = std::make_shared<LLM::LLMRMSNorm>(config.head_dim, config.rms_norm_eps);
            blocks["norm_k"]       = std::make_shared<LLM::LLMRMSNorm>(config.head_dim, config.rms_norm_eps);
            blocks["norm_added_q"] = std::make_shared<LLM::LLMRMSNorm>(config.head_dim, config.rms_norm_eps);
            blocks["norm_added_k"] = std::make_shared<LLM::LLMRMSNorm>(config.head_dim, config.rms_norm_eps);
        }

        std::shared_ptr<Linear> linear(const std::string& name) {
            return std::dynamic_pointer_cast<Linear>(blocks[name]);
        }

        std::shared_ptr<LLM::LLMRMSNorm> norm(const std::string& name) {
            return std::dynamic_pointer_cast<LLM::LLMRMSNorm>(blocks[name]);
        }

        // x: [hidden, L]. Returns q roped as [d, L, heads] and k, v as F16 [d, L, kv_heads]: the layout
        // flash attention reads, so the cached history is concatenated and attended without a copy.
        void project(GGMLRunnerContext* ctx,
                     ggml_tensor* x,
                     ggml_tensor* pe,
                     bool generation,
                     ggml_tensor** q_out,
                     ggml_tensor** k_store,
                     ggml_tensor** v_store) {
            auto gctx            = ctx->ggml_ctx;
            const int64_t length = x->ne[1];
            auto q               = linear(generation ? "add_q_proj" : "to_q")->forward(ctx, x);
            auto k               = linear(generation ? "add_k_proj" : "to_k")->forward(ctx, x);
            auto v               = linear(generation ? "add_v_proj" : "to_v")->forward(ctx, x);
            q                    = ggml_reshape_4d(gctx, q, config.head_dim, config.num_heads, length, 1);
            k                    = ggml_reshape_4d(gctx, k, config.head_dim, config.num_kv_heads, length, 1);
            v                    = ggml_reshape_3d(gctx, v, config.head_dim, config.num_kv_heads, length);
            q                    = norm(generation ? "norm_added_q" : "norm_q")->forward(ctx, q);
            k                    = norm(generation ? "norm_added_k" : "norm_k")->forward(ctx, k);
            q                    = Rope::apply_rope(ctx, q, pe, false);  // [d, L, heads]
            k                    = Rope::apply_rope(ctx, k, pe, false);  // [d, L, kv_heads]
            *q_out               = q;
            *k_store             = ggml_cast(gctx, k, GGML_TYPE_F16);
            *v_store             = ggml_cast(gctx, ggml_permute(gctx, v, 0, 2, 1, 3), GGML_TYPE_F16);
        }

        ggml_tensor* attend(GGMLRunnerContext* ctx,
                            ggml_tensor* q,
                            ggml_tensor* k_store,
                            ggml_tensor* v_store,
                            ggml_tensor* mask,
                            bool generation) {
            auto gctx = ctx->ggml_ctx;
            // attention_ext takes v as [d, kv_heads, Lk] and permutes it back: a view, no copy.
            auto v    = ggml_permute(gctx, v_store, 0, 2, 1, 3);
            auto out  = ggml_ext_attention_ext(gctx,
                                               ctx->backend,
                                               q,
                                               k_store,
                                               v,
                                               config.num_heads,
                                               mask,
                                               true,
                                               ctx->flash_attn_enabled);  // [heads * d, Lq, 1]
            return linear(generation ? "to_add_out" : "to_out")->forward(ctx, out);
        }
    };

    struct DecoderLayer : public GGMLBlock {
        Cosmos3Config config;

        explicit DecoderLayer(const Cosmos3Config& config)
            : config(config) {
            blocks["self_attn"]                        = std::make_shared<Attention>(config);
            blocks["mlp"]                              = std::make_shared<LLM::MLP>(config.hidden_size, config.intermediate_size, false);
            blocks["mlp_moe_gen"]                      = std::make_shared<LLM::MLP>(config.hidden_size, config.intermediate_size, false);
            blocks["input_layernorm"]                  = std::make_shared<LLM::LLMRMSNorm>(config.hidden_size, config.rms_norm_eps);
            blocks["input_layernorm_moe_gen"]          = std::make_shared<LLM::LLMRMSNorm>(config.hidden_size, config.rms_norm_eps);
            blocks["post_attention_layernorm"]         = std::make_shared<LLM::LLMRMSNorm>(config.hidden_size, config.rms_norm_eps);
            blocks["post_attention_layernorm_moe_gen"] = std::make_shared<LLM::LLMRMSNorm>(config.hidden_size, config.rms_norm_eps);
        }

        std::shared_ptr<Attention> attention() {
            return std::dynamic_pointer_cast<Attention>(blocks["self_attn"]);
        }

        ggml_tensor* ffn(GGMLRunnerContext* ctx, ggml_tensor* x, bool generation) {
            auto post_norm = std::dynamic_pointer_cast<LLM::LLMRMSNorm>(
                blocks[generation ? "post_attention_layernorm_moe_gen" : "post_attention_layernorm"]);
            auto mlp = std::dynamic_pointer_cast<LLM::MLP>(blocks[generation ? "mlp_moe_gen" : "mlp"]);
            return ggml_add(ctx->ggml_ctx, x, mlp->forward(ctx, post_norm->forward(ctx, x)));
        }

        ggml_tensor* input_norm(GGMLRunnerContext* ctx, ggml_tensor* x, bool generation) {
            return std::dynamic_pointer_cast<LLM::LLMRMSNorm>(
                       blocks[generation ? "input_layernorm_moe_gen" : "input_layernorm"])
                ->forward(ctx, x);
        }
    };

    struct TimestepEmbedder : public GGMLBlock {
        int64_t frequency_embedding_size = 256;

        explicit TimestepEmbedder(int64_t hidden_size) {
            blocks["linear_1"] = std::make_shared<Linear>(frequency_embedding_size, hidden_size, true);
            blocks["linear_2"] = std::make_shared<Linear>(hidden_size, hidden_size, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* timesteps) {
            auto x = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, static_cast<int>(frequency_embedding_size), 10000, 1.f);
            x      = std::dynamic_pointer_cast<Linear>(blocks["linear_1"])->forward(ctx, x);
            x      = ggml_silu_inplace(ctx->ggml_ctx, x);
            return std::dynamic_pointer_cast<Linear>(blocks["linear_2"])->forward(ctx, x);
        }
    };

    struct Cosmos3Model : public GGMLBlock {
        Cosmos3Config config;

        explicit Cosmos3Model(const Cosmos3Config& config)
            : config(config) {
            blocks["embed_tokens"] = std::make_shared<Embedding>(config.vocab_size, config.hidden_size);
            for (int64_t i = 0; i < config.num_layers; ++i) {
                blocks["layers." + std::to_string(i)] = std::make_shared<DecoderLayer>(config);
            }
            blocks["norm_moe_gen"]  = std::make_shared<LLM::LLMRMSNorm>(config.hidden_size, config.rms_norm_eps);
            blocks["proj_in"]       = std::make_shared<Linear>(config.patch_dim(), config.hidden_size, true);
            blocks["proj_out"]      = std::make_shared<Linear>(config.hidden_size, config.patch_dim(), true);
            blocks["time_embedder"] = std::make_shared<TimestepEmbedder>(config.hidden_size);
        }

        std::shared_ptr<DecoderLayer> layer(int64_t i) {
            return std::dynamic_pointer_cast<DecoderLayer>(blocks["layers." + std::to_string(i)]);
        }

        template <typename T>
        std::shared_ptr<T> get(const std::string& name) {
            return std::dynamic_pointer_cast<T>(blocks[name]);
        }
    };

    // Runs the transfer model one causal step at a time. The persistent state is
    // the text K/V (written once with the first control frame) and the history
    // of clean control/RGB frames, both kept in the runner cache between graphs.
    struct Cosmos3Runner : public DiffusionModelRunner {
        Cosmos3Config config;
        Cosmos3Model model;

        int64_t und_tokens  = 0;
        int64_t hist_tokens = 0;

        // KV window in latent frames (0 = full history) and pinned sink frames,
        // the runtime equivalent of kv_cache_inference_size / num_sink_frames.
        int64_t window_frames = 0;
        int64_t sink_frames   = 1;

        // Edge control derived from RGB input: thinned lines above this fraction of the
        // 99th-percentile gradient become 255 (Canny-like); <= 0 keeps the soft gradient map.
        float edge_threshold = 0.2f;

        // Numerical-comparison hook: return the generation hidden states after this
        // layer (num_layers = after the final norm) instead of the velocity.
        int64_t debug_stop_layer = -1;

        std::vector<float> und_pe_vec;
        std::vector<float> gen_pe_vec;
        std::vector<float> causal_mask_vec;
        std::vector<float> timestep_vec;

        Cosmos3Runner(ggml_backend_t backend,
                      const String2TensorStorage& tensor_storage_map      = {},
                      const std::string& prefix                           = "model.diffusion_model",
                      std::shared_ptr<RunnerWeightManager> weight_manager = nullptr,
                      const char* model_args                              = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(Cosmos3Config::detect_from_weights(tensor_storage_map, prefix)),
              model(config) {
            model.init(params_ctx, tensor_storage_map, prefix);
            for (const auto& [key, value] : parse_key_value_args(model_args, "model arg")) {
                int parsed_int     = 0;
                float parsed_float = 0.f;
                bool ok            = true;
                if (key == "cosmos3_window") {
                    ok            = parse_strict_int(value, parsed_int) && parsed_int >= 0;
                    window_frames = ok ? parsed_int : window_frames;
                } else if (key == "cosmos3_sink") {
                    ok          = parse_strict_int(value, parsed_int) && parsed_int >= 0;
                    sink_frames = ok ? parsed_int : sink_frames;
                } else if (key == "cosmos3_edge_threshold") {
                    ok             = parse_strict_float(value, parsed_float) && parsed_float < 1.f;
                    edge_threshold = ok ? parsed_float : edge_threshold;
                }
                if (!ok) {
                    LOG_WARN("ignoring invalid Cosmos3 model arg '%s=%s'", key.c_str(), value.c_str());
                }
            }
        }

        std::string get_desc() override {
            return "Cosmos3";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors,
                               const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> compute(int n_threads, const DiffusionParams& diffusion_params) override {
            SD_UNUSED(n_threads);
            SD_UNUSED(diffusion_params);
            LOG_ERROR("Cosmos3 runs through the transfer pipeline, not the generic sampler");
            return {};
        }

        void reset_stream() {
            free_cache_ctx_and_buffer();
            und_tokens  = 0;
            hist_tokens = 0;
        }

        static std::string cache_name(const char* kind, int64_t layer, const char* tensor) {
            return std::string("cosmos3.") + kind + "." + std::to_string(layer) + "." + tensor;
        }

        ggml_tensor* make_pe(const std::vector<float>& positions, std::vector<float>& storage, const char* name) {
            const int64_t length = static_cast<int64_t>(positions.size() / 3);
            std::vector<std::vector<float>> ids(static_cast<size_t>(length), std::vector<float>(3));
            for (int64_t i = 0; i < length; ++i) {
                for (int axis = 0; axis < 3; ++axis) {
                    ids[i][axis] = positions[static_cast<size_t>(axis * length + i)];
                }
            }
            storage = Rope::embed_interleaved_mrope(ids, 1, config.rope_theta, static_cast<int>(config.head_dim), config.mrope_section);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.head_dim / 2, length);
            ggml_set_name(pe, name);
            set_backend_tensor_data(pe, storage.data());
            return pe;
        }

        ggml_tensor* trim_history(ggml_context* ctx, ggml_tensor* x, const HistoryUpdate& update) {
            // x: [d, tokens, kv_heads].
            const int64_t total = x->ne[1];
            if (update.keep_recent_tokens < 0 || total - update.sink_tokens <= update.keep_recent_tokens) {
                return x;
            }
            const int64_t sink   = std::min(update.sink_tokens, total);
            const int64_t recent = update.keep_recent_tokens;
            auto tail            = ggml_view_3d(ctx, x, x->ne[0], recent, x->ne[2], x->nb[1], x->nb[2], (total - recent) * x->nb[1]);
            if (sink == 0) {
                return ggml_cont(ctx, tail);
            }
            auto head = ggml_view_3d(ctx, x, x->ne[0], sink, x->ne[2], x->nb[1], x->nb[2], 0);
            return ggml_concat(ctx, ggml_cont(ctx, head), ggml_cont(ctx, tail), 1);
        }

        int64_t trimmed_length(int64_t total, const HistoryUpdate& update) const {
            if (update.keep_recent_tokens < 0 || total - update.sink_tokens <= update.keep_recent_tokens) {
                return total;
            }
            return std::min(update.sink_tokens, total) + update.keep_recent_tokens;
        }

        // patches: [patch_dim, L] clean or noisy latent patches of one latent frame.
        // gen_positions: 3*L floats (t, h, w). timestep < 0 marks a clean frame.
        // text_ids/text_positions: only for the first forward of a stream.
        ggml_cgraph* build_graph(const sd::Tensor<float>& patches_tensor,
                                 const std::vector<float>& gen_positions,
                                 float timestep,
                                 const sd::Tensor<int32_t>& text_ids,
                                 const std::vector<float>& text_positions,
                                 const HistoryUpdate& update) {
            ggml_cgraph* graph = new_graph_custom(COSMOS3_GRAPH_SIZE);
            auto runner_ctx    = get_context(graph);
            auto ctx           = &runner_ctx;
            auto gctx          = compute_ctx;

            ggml_tensor* patches = make_input(patches_tensor);
            ggml_tensor* gen_pe  = make_pe(gen_positions, gen_pe_vec, "cosmos3.gen_pe");

            auto gen = model.get<Linear>("proj_in")->forward(ctx, patches);
            if (timestep >= 0.f) {
                timestep_vec = {timestep * 1000.f * config.timestep_scale};
                auto t       = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 1);
                ggml_set_name(t, "cosmos3.timestep");
                set_backend_tensor_data(t, timestep_vec.data());
                auto t_emb = model.get<TimestepEmbedder>("time_embedder")->forward(ctx, t);
                gen        = ggml_add(gctx, gen, t_emb);
            }

            const bool with_text = !text_ids.empty();
            ggml_tensor* und     = nullptr;
            ggml_tensor* und_pe  = nullptr;
            ggml_tensor* mask    = nullptr;
            if (with_text) {
                auto ids        = make_input(text_ids);
                und             = model.get<Embedding>("embed_tokens")->forward(ctx, ids);
                und             = ggml_reshape_2d(gctx, und, und->ne[0], und->ne[1] * und->ne[2]);
                und_pe          = make_pe(text_positions, und_pe_vec, "cosmos3.und_pe");
                const int64_t n = und->ne[1];
                causal_mask_vec.assign(static_cast<size_t>(n * n), 0.f);
                for (int64_t q = 0; q < n; ++q) {
                    for (int64_t k = q + 1; k < n; ++k) {
                        causal_mask_vec[static_cast<size_t>(q * n + k)] = -INFINITY;
                    }
                }
                mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n, n);
                ggml_set_name(mask, "cosmos3.causal_mask");
                set_backend_tensor_data(mask, causal_mask_vec.data());
            }

            for (int64_t i = 0; i < config.num_layers; ++i) {
                auto layer = model.layer(i);
                auto attn  = layer->attention();

                ggml_tensor* und_k = nullptr;
                ggml_tensor* und_v = nullptr;
                if (with_text) {
                    ggml_tensor* q_u = nullptr;
                    attn->project(ctx, layer->input_norm(ctx, und, false), und_pe, false, &q_u, &und_k, &und_v);
                    auto und_out = attn->attend(ctx, q_u, und_k, und_v, mask, false);
                    und          = layer->ffn(ctx, ggml_add(gctx, und, und_out), false);
                    ctx->persist_cache_tensor(cache_name("und", i, "k"), und_k);
                    ctx->persist_cache_tensor(cache_name("und", i, "v"), und_v);
                } else {
                    und_k = ctx->load_cache_tensor(cache_name("und", i, "k"));
                    und_v = ctx->load_cache_tensor(cache_name("und", i, "v"));
                    GGML_ASSERT(und_k != nullptr && und_v != nullptr);
                }

                ggml_tensor* q_g   = nullptr;
                ggml_tensor* gen_k = nullptr;
                ggml_tensor* gen_v = nullptr;
                attn->project(ctx, layer->input_norm(ctx, gen, true), gen_pe, true, &q_g, &gen_k, &gen_v);

                ggml_tensor* hist_k = hist_tokens > 0 ? ctx->load_cache_tensor(cache_name("hist", i, "k")) : nullptr;
                ggml_tensor* hist_v = hist_tokens > 0 ? ctx->load_cache_tensor(cache_name("hist", i, "v")) : nullptr;
                GGML_ASSERT(hist_tokens == 0 || (hist_k != nullptr && hist_v != nullptr));
                ggml_tensor* past_k = hist_k != nullptr ? ggml_concat(gctx, hist_k, gen_k, 1) : gen_k;
                ggml_tensor* past_v = hist_v != nullptr ? ggml_concat(gctx, hist_v, gen_v, 1) : gen_v;

                auto all_k   = ggml_concat(gctx, und_k, past_k, 1);
                auto all_v   = ggml_concat(gctx, und_v, past_v, 1);
                auto gen_out = attn->attend(ctx, q_g, all_k, all_v, nullptr, true);
                if (update.write) {
                    ctx->persist_cache_tensor(cache_name("hist", i, "k"), trim_history(gctx, past_k, update));
                    ctx->persist_cache_tensor(cache_name("hist", i, "v"), trim_history(gctx, past_v, update));
                }
                gen = layer->ffn(ctx, ggml_add(gctx, gen, gen_out), true);
                if (i == debug_stop_layer) {
                    ggml_build_forward_expand(graph, gen);
                    return graph;
                }
            }

            gen = model.get<LLM::LLMRMSNorm>("norm_moe_gen")->forward(ctx, gen);
            if (debug_stop_layer == config.num_layers) {
                ggml_build_forward_expand(graph, gen);
                return graph;
            }
            gen = model.get<Linear>("proj_out")->forward(ctx, gen);
            ggml_build_forward_expand(graph, gen);
            return graph;
        }

        sd::Tensor<float> forward(int n_threads,
                                  const sd::Tensor<float>& patches,
                                  const std::vector<float>& gen_positions,
                                  float timestep,
                                  const sd::Tensor<int32_t>& text_ids,
                                  const std::vector<float>& text_positions,
                                  const HistoryUpdate& update) {
            auto get_graph = [&]() {
                return build_graph(patches, gen_positions, timestep, text_ids, text_positions, update);
            };
            auto result = GGMLRunner::compute(get_graph, n_threads, false);
            if (!result.has_value()) {
                return {};
            }
            if (!text_ids.empty()) {
                und_tokens = text_ids.numel();
            }
            if (update.write) {
                hist_tokens = trimmed_length(hist_tokens + patches.shape()[1], update);
            }
            return restore_trailing_singleton_dims(std::move(result), 2);
        }
    };
}  // namespace Cosmos3

#endif  // __SD_MODEL_DIFFUSION_COSMOS3_HPP__
