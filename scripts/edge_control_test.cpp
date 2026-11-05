// Standalone check and timing of the gradient-magnitude edge control:
// g++ -std=c++17 -O2 -I src scripts/edge_control_test.cpp && ./a.out
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "runtime/edge_control.h"

static int failures = 0;
#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                         \
        }                                                                       \
    } while (0)

static std::vector<uint8_t> make_image(int w, int h, uint8_t (*fn)(int, int)) {
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t v = fn(x, y);
            for (int c = 0; c < 3; ++c) {
                rgb[(static_cast<size_t>(y) * w + x) * 3 + c] = v;
            }
        }
    }
    return rgb;
}

int main() {
    const int w = 64, h = 48;
    std::vector<uint8_t> out(static_cast<size_t>(w) * h);

    auto flat = make_image(w, h, [](int, int) -> uint8_t { return 128; });
    sd::edge::gradient_magnitude(flat.data(), w, h, 3, out.data());
    bool all_zero = true;
    for (uint8_t v : out) {
        all_zero = all_zero && v == 0;
    }
    CHECK(all_zero);

    auto step = make_image(w, h, [](int x, int) -> uint8_t { return x < 32 ? 0 : 255; });
    sd::edge::GradientEdgeParams no_clip;
    no_clip.clip_percentile = 0.f;
    sd::edge::gradient_magnitude(step.data(), w, h, 3, out.data(), no_clip);
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = out.data() + static_cast<size_t>(y) * w;
        CHECK(row[31] == 255 || row[32] == 255);
        CHECK(row[0] == 0 && row[10] == 0 && row[50] == 0 && row[w - 1] == 0);
        CHECK(row[31] == row[32]);  // symmetric across the step
    }

    auto horizontal = make_image(w, h, [](int, int y) -> uint8_t { return y < 24 ? 255 : 0; });
    sd::edge::gradient_magnitude(horizontal.data(), w, h, 3, out.data(), no_clip);
    CHECK(out[static_cast<size_t>(23) * w + 5] == 255 || out[static_cast<size_t>(24) * w + 5] == 255);
    CHECK(out[static_cast<size_t>(5) * w + 5] == 0);

    sd::edge::GradientEdgeParams binary;
    binary.clip_percentile  = 0.f;
    binary.binary_threshold = 0.5f;
    sd::edge::gradient_magnitude(step.data(), w, h, 3, out.data(), binary);
    for (uint8_t v : out) {
        CHECK(v == 0 || v == 255);
    }

    sd::edge::GradientEdgeParams thin;
    thin.clip_percentile  = 0.f;
    thin.binary_threshold = 0.2f;
    thin.thin             = true;
    sd::edge::gradient_magnitude(step.data(), w, h, 3, out.data(), thin);
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = out.data() + static_cast<size_t>(y) * w;
        int on             = 0;
        for (int x = 0; x < w; ++x) {
            on += row[x] == 255;
            CHECK(row[x] == 0 || row[x] == 255);
        }
        CHECK(on == 1);  // one line where the soft map is several pixels wide
        CHECK(row[31] == 255 || row[32] == 255);
    }

    // Thinned edges under 1% of the image: the threshold must still drop the weak edge.
    {
        const int sw = 200, sh = 200;
        std::vector<uint8_t> sparse(static_cast<size_t>(sw) * sh * 3);
        for (int y = 0; y < sh; ++y) {
            for (int x = 0; x < sw; ++x) {
                const uint8_t v = x >= 100 ? 255 : (y >= 150 ? 20 : 0);
                for (int c = 0; c < 3; ++c) {
                    sparse[(static_cast<size_t>(y) * sw + x) * 3 + c] = v;
                }
            }
        }
        std::vector<uint8_t> edges(static_cast<size_t>(sw) * sh);
        sd::edge::GradientEdgeParams lines;
        lines.thin             = true;
        lines.binary_threshold = 0.2f;
        sd::edge::gradient_magnitude(sparse.data(), sw, sh, 3, edges.data(), lines);
        int weak = 0, strong = 0;
        for (int y = 145; y < 155; ++y) {
            for (int x = 10; x < 80; ++x) {
                weak += edges[static_cast<size_t>(y) * sw + x] != 0;
            }
        }
        for (int y = 0; y < sh; ++y) {
            strong += edges[static_cast<size_t>(y) * sw + 99] == 255 || edges[static_cast<size_t>(y) * sw + 100] == 255;
        }
        CHECK(weak == 0);
        CHECK(strong == sh);
    }

    // Gradients under 1% of the image even before thinning: two small squares on black.
    {
        const int sw = 200, sh = 200;
        std::vector<uint8_t> squares(static_cast<size_t>(sw) * sh * 3, 0);
        for (int y = 0; y < 6; ++y) {
            for (int x = 0; x < 6; ++x) {
                for (int c = 0; c < 3; ++c) {
                    squares[(static_cast<size_t>(50 + y) * sw + 50 + x) * 3 + c]   = 255;
                    squares[(static_cast<size_t>(150 + y) * sw + 150 + x) * 3 + c] = 20;
                }
            }
        }
        std::vector<uint8_t> edges(static_cast<size_t>(sw) * sh);
        sd::edge::GradientEdgeParams lines;
        lines.thin             = true;
        lines.binary_threshold = 0.2f;
        sd::edge::gradient_magnitude(squares.data(), sw, sh, 3, edges.data(), lines);
        int weak = 0, strong = 0;
        for (int y = 140; y < 166; ++y) {
            for (int x = 140; x < 166; ++x) {
                weak += edges[static_cast<size_t>(y) * sw + x] != 0;
            }
        }
        for (int y = 40; y < 66; ++y) {
            for (int x = 40; x < 66; ++x) {
                strong += edges[static_cast<size_t>(y) * sw + x] == 255;
            }
        }
        CHECK(weak == 0);
        CHECK(strong > 0);
    }

    // Timing on a busy image (deterministic noise plus gradients).
    const int sizes[2][2] = {{832, 480}, {1280, 720}};
    for (const auto& size : sizes) {
        const int tw = size[0], th = size[1];
        std::vector<uint8_t> rgb(static_cast<size_t>(tw) * th * 3);
        uint32_t state = 12345;
        for (size_t i = 0; i < rgb.size(); ++i) {
            state  = state * 1664525u + 1013904223u;
            rgb[i] = static_cast<uint8_t>((state >> 24) / 4 + (i / 3 % tw) * 128 / tw);
        }
        std::vector<uint8_t> edges(static_cast<size_t>(tw) * th);
        sd::edge::gradient_magnitude(rgb.data(), tw, th, 3, edges.data());
        const int runs = 50;
        auto t0        = std::chrono::steady_clock::now();
        for (int r = 0; r < runs; ++r) {
            sd::edge::gradient_magnitude(rgb.data(), tw, th, 3, edges.data());
        }
        auto t1   = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
        std::printf("gradient_magnitude %dx%d: %.2f ms/frame (single thread, mean of %d)\n", tw, th, ms, runs);

        sd::edge::GradientEdgeParams lines;
        lines.thin             = true;
        lines.binary_threshold = 0.2f;
        t0                     = std::chrono::steady_clock::now();
        for (int r = 0; r < runs; ++r) {
            sd::edge::gradient_magnitude(rgb.data(), tw, th, 3, edges.data(), lines);
        }
        t1 = std::chrono::steady_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
        std::printf("gradient_magnitude thin %dx%d: %.2f ms/frame (single thread, mean of %d)\n", tw, th, ms, runs);
    }

    if (failures == 0) {
        std::printf("edge_control_test: all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
