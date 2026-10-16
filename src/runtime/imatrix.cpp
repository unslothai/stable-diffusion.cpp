#include "runtime/imatrix.h"

/* Adapted from llama.cpp (credits: Kawrakow). */

#include "core/util.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "stable-diffusion.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

static IMatrixCollector imatrix_collector;

IMatrixCollector& get_imatrix_collector() {
    return imatrix_collector;
}

// remove any prefix and suffixes from the name
// CUDA0#blk.0.attn_k.weight#0 => blk.0.attn_k.weight
static std::string filter_tensor_name(const char* name) {
    std::string wname;
    const char* p = strchr(name, '#');
    if (p != NULL) {
        p             = p + 1;
        const char* q = strchr(p, '#');
        if (q != NULL) {
            wname = std::string(p, q - p);
        } else {
            wname = p;
        }
    } else {
        wname = name;
    }
    return wname;
}

bool IMatrixCollector::collect_imatrix(struct ggml_tensor* t, bool ask, void* user_data) {
    GGML_UNUSED(user_data);
    if (t == nullptr) {
        return false;
    }
    if (t->op != GGML_OP_MUL_MAT && t->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }

    const struct ggml_tensor* src0 = t->src[0];
    const struct ggml_tensor* src1 = t->src[1];
    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    std::string wname = filter_tensor_name(src0->name);

    // when ask is true, the scheduler wants to know if we are interested in data from this tensor
    // if we return true, a follow-up call will be made with ask=false in which we can do the actual collection
    if (ask) {
        if (t->op == GGML_OP_MUL_MAT_ID) {
            return true;  // collect all indirect matrix multiplications
        }
        // why are small batches ignored (<16 tokens)?
        // if (src1->ne[1] < 16 || src1->type != GGML_TYPE_F32) return false;
        if (!(wname.substr(0, 6) == "model." || wname.substr(0, 17) == "cond_stage_model." || wname.substr(0, 14) == "text_encoders.")) {
            return false;
        }
        return true;
    }
    std::lock_guard<std::mutex> lock(mutex_);

    ++layout_.calls;
    // Rows are read through nb[1..3], so any layout whose rows are packed elements works; a strided
    // nb[0] or an element type with no float conversion here is refused, counted, and reported at save time.
    const bool type_ok = src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16;
    if (!type_ok || src1->nb[0] != ggml_type_size(src1->type)) {
        ++layout_.rejected;
        LOG_ERROR("imatrix: cannot collect '%s': activation type %s with element stride %zu is not supported; this tensor gets no data from this call\n",
                  wname.c_str(), ggml_type_name(src1->type), (size_t)src1->nb[0]);
        return true;
    }
    if (src1->type == GGML_TYPE_F16) {
        ++layout_.f16;
    } else if (src1->type == GGML_TYPE_BF16) {
        ++layout_.bf16;
    }
    if (!ggml_is_contiguous(src1)) {
        ++layout_.non_contiguous;
    }
    if (src1->ne[2] * src1->ne[3] > 1) {
        ++layout_.batched;
    }
    if (src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && src1->ne[2] * src1->ne[3] == 1) {
        ++layout_.contiguous_f32_2d;
    }

    // copy the data from the GPU memory if needed; ggml_nbytes spans the whole (possibly strided) view
    const bool is_host = src1->buffer == NULL || ggml_backend_buffer_is_host(src1->buffer);

    if (!is_host) {
        src1_bytes_.resize(ggml_nbytes(src1));
        ggml_backend_tensor_get(src1, src1_bytes_.data(), 0, ggml_nbytes(src1));
    }

    const char* data  = is_host ? (const char*)src1->data : src1_bytes_.data();
    const int64_t ne0 = src1->ne[0];
    row_f32_.resize(ne0);

    // One activation row as f32, addressed through the byte strides.
    auto row_at = [&](int64_t i1, int64_t i2, int64_t i3) -> const float* {
        const char* p = data + i1 * src1->nb[1] + i2 * src1->nb[2] + i3 * src1->nb[3];
        if (src1->type == GGML_TYPE_F32) {
            return (const float*)p;
        }
        if (src1->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t*)p, row_f32_.data(), ne0);
        } else {
            ggml_bf16_to_fp32_row((const ggml_bf16_t*)p, row_f32_.data(), ne0);
        }
        return row_f32_.data();
    };

    // this has been adapted to the new format of storing merged experts in a single 3d tensor
    // ref: https://github.com/ggml-org/llama.cpp/pull/6387
    if (t->op == GGML_OP_MUL_MAT_ID) {
        //   ids  -> [n_experts_used, n_tokens]
        //   src1 -> [cols, n_expert_used, n_tokens]
        const ggml_tensor* ids = t->src[2];
        const int n_as         = static_cast<int>(src0->ne[2]);
        const int n_ids        = static_cast<int>(ids->ne[0]);

        // the top-k selected expert ids are stored in the ids tensor
        // for simplicity, always copy ids to host, because it is small
        // take into account that ids is not contiguous!

        GGML_ASSERT(ids->ne[1] == src1->ne[2]);

        ids_.resize(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, ids_.data(), 0, ggml_nbytes(ids));

        auto& e = stats_[wname];

        ++e.ncall;

        if (e.values.empty()) {
            e.values.resize(src1->ne[0] * n_as, 0);
            e.counts.resize(src1->ne[0] * n_as, 0);
        } else if (e.values.size() != (size_t)src1->ne[0] * n_as) {
            LOG_ERROR("inconsistent size for %s (%d vs %d)\n", wname.c_str(), (int)e.values.size(), (int)src1->ne[0] * n_as);
            exit(1);  // GGML_ABORT("fatal error");
        }
        // loop over all possible experts, regardless if they are used or not in the batch
        for (int ex = 0; ex < n_as; ++ex) {
            size_t e_start = ex * src1->ne[0];

            for (int idx = 0; idx < n_ids; ++idx) {
                for (int row = 0; row < (int)src1->ne[2]; ++row) {
                    const int excur = *(const int32_t*)(ids_.data() + row * ids->nb[1] + idx * ids->nb[0]);

                    GGML_ASSERT(excur >= 0 && excur < n_as);  // sanity check

                    if (excur != ex)
                        continue;

                    const int64_t i11 = idx % src1->ne[1];
                    const int64_t i12 = row;
                    const float* x    = row_at(i11, i12, 0);

                    for (int j = 0; j < (int)src1->ne[0]; ++j) {
                        e.values[e_start + j] += x[j] * x[j];
                        e.counts[e_start + j]++;
                        if (!std::isfinite(e.values[e_start + j])) {
                            LOG_ERROR("%f detected in %s\n", e.values[e_start + j], wname.c_str());
                            exit(1);
                        }
                    }
                }
            }
        }
    } else {
        auto& e = stats_[wname];
        if (e.values.empty()) {
            e.values.resize(src1->ne[0], 0);
            e.counts.resize(src1->ne[0], 0);
        } else if (e.values.size() != (size_t)src1->ne[0]) {
            LOG_WARN("inconsistent size for %s (%d vs %d)\n", wname.c_str(), (int)e.values.size(), (int)src1->ne[0]);
            exit(1);  // GGML_ABORT("fatal error");
        }

        ++e.ncall;
        // Every row of every plane: src0 broadcasts over src1's ne[2]/ne[3], and each plane's rows hit the
        // same ne[0] input columns of the weight, so all of them accumulate into the one per-column vector.
        for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
                for (int64_t row = 0; row < src1->ne[1]; ++row) {
                    const float* x = row_at(row, i2, i3);
                    for (int64_t j = 0; j < ne0; ++j) {
                        if (std::isfinite(x[j])) {
                            e.values[j] += x[j] * x[j];
                            e.counts[j]++;
                            if (!std::isfinite(e.values[j])) {
                                LOG_WARN("%f detected in %s\n", e.values[j], wname.c_str());
                                exit(1);
                            }
                        } else {
                            // Likely something from an attention mask?
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool load_imatrix(const char* imatrix_path) {
    return imatrix_collector.load_imatrix(imatrix_path);
}

void save_imatrix(const char* imatrix_path) {
    imatrix_collector.save_imatrix(imatrix_path);
}

static bool collect_imatrix(struct ggml_tensor* t, bool ask, void* user_data) {
    return imatrix_collector.collect_imatrix(t, ask, user_data);
}

void enable_imatrix_collection() {
    sd_set_backend_eval_callback(collect_imatrix, nullptr);
}

void disable_imatrix_collection() {
    sd_set_backend_eval_callback(nullptr, nullptr);
}

void IMatrixCollector::save_imatrix(std::string fname, int ncall) const {
    if (ncall > 0) {
        fname += ".at_";
        fname += std::to_string(ncall);
    }
    // avoid writing imatrix entries that do not have full data
    // this can happen with MoE models where some of the experts end up not being exercised by the provided training data

    int n_entries = 0;
    std::vector<std::string> to_store;

    for (const auto& kv : stats_) {
        const int n_all = static_cast<int>(kv.second.counts.size());

        if (n_all == 0) {
            continue;
        }

        int n_zeros = 0;
        for (const int c : kv.second.counts) {
            if (c == 0) {
                n_zeros++;
            }
        }

        if (n_zeros == n_all) {
            LOG_WARN("entry '%40s' has no data - skipping\n", kv.first.c_str());
            continue;
        }

        if (n_zeros > 0) {
            LOG_WARN("entry '%40s' has partial data (%.2f%%) - skipping\n", kv.first.c_str(), 100.0f * (n_all - n_zeros) / n_all);
            continue;
        }

        n_entries++;
        to_store.push_back(kv.first);
    }

    LOG_INFO("imatrix: %lld matmul calls collected (contiguous f32 2-D %lld, non-contiguous %lld, batched ne[2]/ne[3] %lld, f16 %lld, bf16 %lld, rejected %lld)\n",
             layout_.calls, layout_.contiguous_f32_2d, layout_.non_contiguous, layout_.batched, layout_.f16, layout_.bf16, layout_.rejected);
    if (layout_.rejected > 0) {
        LOG_ERROR("imatrix: %lld matmul calls were not collected (unsupported activation layout), see the errors above\n", layout_.rejected);
    }

    if (to_store.size() < stats_.size()) {
        LOG_WARN("storing only %zu out of %zu entries\n", to_store.size(), stats_.size());
    }

    std::ofstream out(fname, std::ios::binary);
    out.write((const char*)&n_entries, sizeof(n_entries));
    for (const auto& name : to_store) {
        const auto& stat = stats_.at(name);
        int len          = static_cast<int>(name.size());
        out.write((const char*)&len, sizeof(len));
        out.write(name.c_str(), len);
        out.write((const char*)&stat.ncall, sizeof(stat.ncall));
        int nval = static_cast<int>(stat.values.size());
        out.write((const char*)&nval, sizeof(nval));
        if (nval > 0) {
            std::vector<float> tmp(nval);
            for (int i = 0; i < nval; i++) {
                tmp[i] = (stat.values[i] / static_cast<float>(stat.counts[i])) * static_cast<float>(stat.ncall);
            }
            out.write((const char*)tmp.data(), nval * sizeof(float));
        }
    }

    // Write the number of call the matrix was computed with
    out.write((const char*)&last_call_, sizeof(last_call_));
}

bool IMatrixCollector::load_imatrix(const char* fname) {
    std::ifstream in(fname, std::ios::binary);
    if (!in) {
        LOG_ERROR("failed to open %s\n", fname);
        return false;
    }
    int n_entries;
    in.read((char*)&n_entries, sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        LOG_ERROR("no data in file %s\n", fname);
        return false;
    }
    for (int i = 0; i < n_entries; ++i) {
        int len;
        in.read((char*)&len, sizeof(len));
        std::vector<char> name_as_vec(len + 1);
        in.read((char*)name_as_vec.data(), len);
        if (in.fail()) {
            LOG_ERROR("failed reading name for entry %d from %s\n", i + 1, fname);
            return false;
        }
        name_as_vec[len] = 0;
        std::string name{name_as_vec.data()};
        auto& e = stats_[std::move(name)];
        int ncall;
        in.read((char*)&ncall, sizeof(ncall));
        int nval;
        in.read((char*)&nval, sizeof(nval));
        if (in.fail() || nval < 1) {
            LOG_ERROR("failed reading number of values for entry %d\n", i);
            stats_ = {};
            return false;
        }

        if (e.values.empty()) {
            e.values.resize(nval, 0);
            e.counts.resize(nval, 0);
        } else if (e.values.size() != (size_t)nval) {
            LOG_ERROR("inconsistent size for a repeated entry (%d vs %d)\n", (int)e.values.size(), nval);
            stats_ = {};
            return false;
        }

        std::vector<float> tmp(nval);
        in.read((char*)tmp.data(), nval * sizeof(float));
        if (in.fail()) {
            LOG_ERROR("failed reading data for entry %d\n", i);
            stats_ = {};
            return false;
        }

        // Recreate the state as expected by save_imatrix(), and correct for weighted sum.
        for (int i = 0; i < nval; i++) {
            e.values[i] += tmp[i];
            e.counts[i] += ncall;
        }
        e.ncall += ncall;
    }
    return true;
}
