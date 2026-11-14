// MiniMax-H3 reference-parity checks: prompt segmentation, RoPE positions, the simple sigma grid and res_multistep.
// Build after a CPU build in ./build: scripts/build_h3_parity_test.sh && ./build/h3_parity_test
// `--dump` prints the computed values as JSON for an external comparison.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "conditioning/text_segments.h"
#include "model/diffusion/minimax_h3.hpp"
#include "runtime/denoiser.hpp"
#include "tokenizers/qwen2_tokenizer.h"

static int failures = 0;
#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                         \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------- tokenization
struct PromptBuilder {
    std::string prompt;
    std::vector<size_t> starts;
    void text(const std::string& piece) {
        starts.push_back(prompt.size());
        prompt += piece;
    }
    void vision() {
        starts.push_back(prompt.size());
        prompt += "<|vision_start|><|image_pad|><|vision_end|>";
    }
};

static std::vector<int> encode_segmented(Qwen2Tokenizer& tok, const PromptBuilder& b, bool joined) {
    std::vector<int> ids;
    std::vector<std::string> pieces = joined ? std::vector<std::string>{b.prompt} : split_text_segments(b.prompt, b.starts);
    for (const auto& piece : pieces) {
        std::vector<int> part;
        tok.encode(piece, part, nullptr);
        ids.insert(ids.end(), part.begin(), part.end());
    }
    for (int& id : ids) {
        if (id == 151655) {  // <|image_pad|>
            id = -1;
        }
    }
    return ids;
}

// Label sequences laid out the way the MiniMax-H3 conditioner builds them; expected ids come from the
// reference tokenizer, which tokenizes every label, timestamp and the prompt separately.
static void test_tokenization(bool dump) {
    Qwen2Tokenizer tok;
    struct Case {
        const char* name;
        PromptBuilder b;
        std::vector<int> expected;
    };
    std::vector<Case> cases;
    {
        Case c{"t2va", {}, {32, 883, 2727, 23811, 366, 14615, 29, 320, 52178, 398, 568}};
        c.b.text("A man says hello <world> (loudly).");
        cases.push_back(c);
    }
    {
        Case c{"image_audio_video_audio", {}, {21604, 3826, 220, 16, 26818, 220, 151652, -1, 151653, 65406, 220, 16, 26818, 220, 27, 10724, 220, 16, 26818, 220, 27, 15, 13, 17, 6486, 29, 151652, -1, 151653, 27, 16, 13, 15, 6486, 29, 151652, -1, 151653, 65406, 220, 17, 26818, 220, 21604, 3826, 220, 16, 29, 22479, 26, 1221, 366, 14755, 220, 17, 29, 11088, 13}};
        c.b.text("<Picture 1>: ");
        c.b.vision();
        c.b.text("<Audio 1>: ");
        c.b.text("<Video 1>: ");
        c.b.text("<0.2 seconds>");
        c.b.vision();
        c.b.text("<1.0 seconds>");
        c.b.vision();
        c.b.text("<Audio 2>: ");
        c.b.text("<Picture 1> walks; then <Audio 2> plays.");
        cases.push_back(c);
    }
    {
        Case c{"video_only", {}, {27, 10724, 220, 16, 26818, 220, 27, 15, 13, 17, 6486, 29, 151652, -1, 151653, 27, 16, 13, 15, 6486, 29, 151652, -1, 151653, 58043, 3550, 9934, 13}};
        c.b.text("<Video 1>: ");
        c.b.text("<0.2 seconds>");
        c.b.vision();
        c.b.text("<1.0 seconds>");
        c.b.vision();
        c.b.text(" Leading space prompt.");
        cases.push_back(c);
    }
    {
        Case c{"audio_then_prompt", {}, {65406, 220, 16, 26818, 220, 27, 15, 13, 20, 6486, 29, 8471, 1588}};
        c.b.text("<Audio 1>: ");
        c.b.text("<0.5 seconds> starts here");
        cases.push_back(c);
    }
    bool legacy_differs = false;
    for (const auto& c : cases) {
        auto ids = encode_segmented(tok, c.b, false);
        CHECK(ids == c.expected);
        if (ids != c.expected) {
            std::fprintf(stderr, "  tokenization case %s differs\n", c.name);
        }
        legacy_differs |= encode_segmented(tok, c.b, true) != c.expected;
        if (dump) {
            std::printf("{\"tokens\": \"%s\", \"ids\": [", c.name);
            for (size_t i = 0; i < ids.size(); ++i) {
                std::printf("%s%d", i ? ", " : "", ids[i]);
            }
            std::printf("]}\n");
        }
    }
    // The joined-string encoding merges ": " with the next label's "<" and must not match.
    CHECK(legacy_differs);
    CHECK((split_text_segments("ab", {1, 1, 5}) == std::vector<std::string>{"a", "b"}));
}

// ---------------------------------------------------------------- RoPE positions
// Reference rule: float64 grid (text rows, then target audio rows, then target video rows), cast once.
static void test_positions(bool dump) {
    const int64_t text_len = 20000, latent_t = 37, latent_h = 48, latent_w = 84, audio_t = 207;
    auto layout = MiniMaxH3::build_layout(text_len, latent_t, latent_h, latent_w, audio_t,
                                          {}, {}, sd::Tensor<int32_t>(), {}, sd::Tensor<int32_t>(),
                                          1.f, 1.f, 1.f, 1.f);
    std::vector<double> expected;
    auto axis = [](int64_t dim, double sqrt_area) {
        std::vector<double> r;
        double ratio = dim / sqrt_area;
        int64_t n    = dim / 2;
        for (int64_t i = 0; i < n; ++i) {
            r.push_back((i * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0);
        }
        return r;
    };
    double area = std::sqrt(static_cast<double>(latent_h * latent_w));
    auto hs     = axis(latent_h, area);
    auto ws     = axis(latent_w, area);
    for (int64_t i = 0; i < text_len; ++i) {
        expected.insert(expected.end(), {static_cast<double>(i), 0.0, 0.0});
    }
    double cursor = static_cast<double>(text_len);
    for (int ch = 0; ch < 2; ++ch) {
        for (int64_t t = 0; t < audio_t; ++t) {
            expected.insert(expected.end(), {cursor + t, 0.0, ch == 0 ? ws.front() : ws.back()});
        }
    }
    // Exclusive cumulative sum of the per-latent-frame spans, summed in float64.
    const double spans[5] = {1, 4, 4, 4, 4};
    std::vector<double> t_grid(latent_t, cursor);
    double acc = 0.0;
    for (int64_t t = 0; t < latent_t; ++t) {
        t_grid[t] = cursor + acc;
        acc += 5.0 / 3.0 * spans[t % 5];
    }
    for (int64_t t = 0; t < latent_t; ++t) {
        for (double h : hs) {
            for (double w : ws) {
                expected.insert(expected.end(), {t_grid[t], h, w});
            }
        }
    }
    CHECK(layout.positions.size() == expected.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < std::min(layout.positions.size(), expected.size()); ++i) {
        mismatches += layout.positions[i] != static_cast<float>(expected[i]);
    }
    if (mismatches) {
        std::fprintf(stderr, "  %zu of %zu positions differ from the float64 grid cast once\n", mismatches, expected.size());
    }
    CHECK(mismatches == 0);
    if (dump) {
        // Raw float32 [rows, 3] for an external comparison.
        if (FILE* f = std::fopen("h3_positions_t2va.f32", "wb")) {
            std::fwrite(layout.positions.data(), sizeof(float), layout.positions.size(), f);
            std::fclose(f);
        }
        std::printf("{\"positions_rows\": %zu}\n", layout.positions.size() / 3);
    }
}

// ---------------------------------------------------------------- sigma grid
static double shift_t(double alpha, double t) {
    return alpha * t / (1.0 + (alpha - 1.0) * t);
}

static void test_simple_sigmas(bool dump) {
    for (int n : {4, 7, 8, 20, 25, 30, 33, 40, 50}) {
        H3AVFlowDenoiser denoiser(12.f, 3.f, 24);
        auto sigmas = denoiser.get_sigmas(n, 0, SIMPLE_SCHEDULER, VERSION_MINIMAX_H3);
        CHECK(sigmas.size() == static_cast<size_t>(n + 1));
        // Reference: sigmas table index -(1 + int(i * 1000 / n)) of shift((k + 1) / 1000), then 0.
        double ss = 1000.0 / n;
        for (int i = 0; i < n && i < static_cast<int>(sigmas.size()); ++i) {
            int k        = 999 - static_cast<int>(i * ss);
            double ref   = shift_t(12.0, (k + 1) / 1000.0);
            bool matches = std::fabs(sigmas[i] - ref) <= 2e-7;
            CHECK(matches);
            if (!matches) {
                std::fprintf(stderr, "  n=%d sigma[%d]=%.9g, reference %.9g\n", n, i, sigmas[i], ref);
            }
        }
        CHECK(sigmas.back() == 0.f);
        if (dump) {
            std::printf("{\"sigmas_n\": %d, \"sigmas\": [", n);
            for (size_t i = 0; i < sigmas.size(); ++i) {
                std::printf("%s%.9g", i ? ", " : "", sigmas[i]);
            }
            std::printf("]}\n");
        }
    }
}

// ---------------------------------------------------------------- res_multistep
// A linear toy model (denoised = a * x + b * sigma) run through sample_res_multistep and through the
// reference update rule evaluated in double, over closely spaced sigmas where exp(h) - 1 cancels.
static void test_res_multistep(bool dump) {
    for (const auto& sigmas : std::vector<std::vector<float>>{{1.f, 0.9999f, 0.9998f, 0.9997f, 0.f},
                                                              {0.997f, 0.9f, 0.6f, 0.2f, 0.f}}) {
        sd::Tensor<float> x({4});
        for (int i = 0; i < 4; ++i) {
            x[i] = 0.25f * (i + 1) - 0.4f;
        }
        auto model = [](const sd::Tensor<float>& in, float sigma, int) {
            sd::guidance::GuiderOutput out;
            out.pred = in * 0.8f + sigma * 0.1f;
            return out;
        };
        auto rng = std::make_shared<STDDefaultRNG>();
        IIDGaussianNoiseSampler noise(x, rng);
        auto result = sample_res_multistep(model, x, sigmas, noise, true, 0.f);

        std::vector<double> xr(4), old_den(4);
        for (int i = 0; i < 4; ++i) {
            xr[i] = x[i];
        }
        double old_sigma_down = 0.0;
        for (size_t i = 0; i + 1 < sigmas.size(); ++i) {
            std::vector<double> den(4);
            for (int j = 0; j < 4; ++j) {
                den[j] = 0.8 * xr[j] + sigmas[i] * 0.1;
            }
            double s = sigmas[i], s_down = sigmas[i + 1];
            if (s_down == 0.0 || i == 0) {
                for (int j = 0; j < 4; ++j) {
                    xr[j] += (xr[j] - den[j]) / s * (s_down - s);
                }
            } else {
                double t = -std::log(s), t_old = -std::log(old_sigma_down), t_next = -std::log(s_down);
                double t_prev = -std::log(static_cast<double>(sigmas[i - 1]));
                double h = t_next - t, c2 = (t_prev - t_old) / h;
                double phi1 = std::expm1(-h) / -h, phi2 = (phi1 - 1.0) / -h;
                double b1 = phi1 - phi2 / c2, b2 = phi2 / c2;
                for (int j = 0; j < 4; ++j) {
                    xr[j] = std::exp(-h) * xr[j] + h * (b1 * den[j] + b2 * old_den[j]);
                }
            }
            old_den        = den;
            old_sigma_down = s_down;
        }
        double max_rel = 0.0;
        for (int j = 0; j < 4; ++j) {
            max_rel = std::max(max_rel, std::fabs(result[j] - xr[j]) / std::max(1e-3, std::fabs(xr[j])));
        }
        if (max_rel > 1e-4) {
            std::fprintf(stderr, "  res_multistep relative error %.3g\n", max_rel);
        }
        CHECK(max_rel <= 1e-4);
        if (dump) {
            std::printf("{\"res_multistep_max_rel\": %.3g, \"sigmas\": [", max_rel);
            for (size_t i = 0; i < sigmas.size(); ++i) {
                std::printf("%s%.9g", i ? ", " : "", sigmas[i]);
            }
            std::printf("], \"x\": [");
            for (int j = 0; j < 4; ++j) {
                std::printf("%s%.9g", j ? ", " : "", result[j]);
            }
            std::printf("]}\n");
        }
    }
}

// ---------------------------------------------------------------- vision resize
// The reference resizes Qwen vision inputs with bilinear, align_corners=False, no antialias: source coordinate
// (dst + 0.5) * in / out - 0.5, clamped at 0, the upper neighbour clamped to the last pixel.
static void test_vision_resize(bool dump) {
    const int64_t iw = 7, ih = 5, ow = 4, oh = 9;
    sd::Tensor<float> image({iw, ih, 3, 1});
    for (int64_t i = 0; i < image.numel(); ++i) {
        image[i] = static_cast<float>((i * 37) % 101) / 100.f;
    }
    auto resized = sd::ops::interpolate(image, std::vector<int64_t>{ow, oh, 3, 1}, sd::ops::InterpolateMode::Bilinear);
    auto coord   = [](int64_t dst, int64_t in, int64_t out, int64_t& lo, int64_t& hi, double& frac) {
        double src = std::max(0.0, (dst + 0.5) * static_cast<double>(in) / out - 0.5);
        lo         = static_cast<int64_t>(src);
        hi         = std::min(lo + 1, in - 1);
        frac       = src - lo;
    };
    double max_err = 0.0;
    for (int64_t c = 0; c < 3; ++c) {
        for (int64_t y = 0; y < oh; ++y) {
            for (int64_t x = 0; x < ow; ++x) {
                int64_t y0, y1, x0, x1;
                double fy, fx;
                coord(y, ih, oh, y0, y1, fy);
                coord(x, iw, ow, x0, x1, fx);
                auto at    = [&](int64_t xx, int64_t yy) { return static_cast<double>(image[xx + iw * (yy + ih * c)]); };
                double ref = (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x1, y0)) + fy * ((1 - fx) * at(x0, y1) + fx * at(x1, y1));
                max_err    = std::max(max_err, std::fabs(resized[x + ow * (y + oh * c)] - ref));
            }
        }
    }
    CHECK(max_err < 1e-6);
    if (dump) {
        std::printf("{\"bilinear_max_err\": %.3g, \"bilinear\": [", max_err);
        for (int64_t i = 0; i < resized.numel(); ++i) {
            std::printf("%s%.8g", i ? ", " : "", resized[i]);
        }
        std::printf("]}\n");
    }
}

int main(int argc, char** argv) {
    bool dump = argc > 1 && std::strcmp(argv[1], "--dump") == 0;
    test_tokenization(dump);
    test_positions(dump);
    test_simple_sigmas(dump);
    test_res_multistep(dump);
    test_vision_resize(dump);
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("h3_parity_test: all checks passed\n");
    return 0;
}
