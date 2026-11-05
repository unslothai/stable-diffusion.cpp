#ifndef __SD_RUNTIME_IMATRIX_H__
#define __SD_RUNTIME_IMATRIX_H__

#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

/* Adapted from llama.cpp (credits: Kawrakow). */

struct ggml_tensor;

struct IMatrixStats {
    std::vector<float> values{};
    std::vector<int> counts{};
    int ncall = 0;
};

// How many collected matmuls saw each src1 layout; reported when the imatrix is saved.
struct IMatrixLayoutCounts {
    long long calls             = 0;
    long long contiguous_f32_2d = 0;
    long long non_contiguous    = 0;
    long long batched           = 0;
    long long f16               = 0;
    long long bf16              = 0;
    long long rejected          = 0;
};

class IMatrixCollector {
private:
    std::unordered_map<std::string, IMatrixStats> stats_ = {};
    std::mutex mutex_;
    int last_call_ = 0;
    std::vector<char> src1_bytes_;
    std::vector<float> row_f32_;
    IMatrixLayoutCounts layout_;
    std::vector<char> ids_;  // the expert ids from ggml_mul_mat_id

public:
    IMatrixCollector() = default;
    bool collect_imatrix(struct ggml_tensor* t, bool ask, void* user_data);
    void save_imatrix(std::string fname, int ncall = -1) const;
    bool load_imatrix(const char* fname);
    const IMatrixLayoutCounts& layout_counts() const { return layout_; }
    std::vector<float> get_values(const std::string& key) const {
        auto it = stats_.find(key);
        if (it != stats_.end()) {
            return it->second.values;
        } else {
            return {};
        }
    }
};

IMatrixCollector& get_imatrix_collector();

#endif  // __SD_RUNTIME_IMATRIX_H__
