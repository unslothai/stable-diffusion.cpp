// Standalone check of the MiniMax-H3 VAE tile layout and blends: g++ -std=c++17 -I src scripts/h3_vae_tiles_test.cpp && ./a.out
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "model/vae/h3_vae_tiles.hpp"

static int failures = 0;
#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                         \
        }                                                                       \
    } while (0)

struct Expected {
    int64_t length;  // pixels
    std::vector<int64_t> starts;
    std::vector<int64_t> overlaps;
};

static bool same_layout(const H3VAETiles::Axis& axis, const std::vector<int64_t>& starts, int64_t size, const std::vector<int64_t>& overlaps) {
    return axis.starts == starts && axis.size == size && axis.overlaps == overlaps;
}

int main() {
    // The reference layout (256 px tiles, >= 64 px overlaps, slack in 16 px steps) for every axis length of
    // 1344x768, 768x1344, 1024x1024, 1376x736 and 1920x1088.
    const std::vector<Expected> expected = {
        {736, {0, 160, 320, 480}, {96, 96, 96}},
        {768, {0, 160, 336, 512}, {96, 80, 80}},
        {1024, {0, 192, 384, 576, 768}, {64, 64, 64, 64}},
        {1088, {0, 160, 320, 480, 656, 832}, {96, 96, 96, 80, 80}},
        {1344, {0, 176, 352, 528, 704, 896, 1088}, {80, 80, 80, 80, 64, 64}},
        {1376, {0, 176, 352, 544, 736, 928, 1120}, {80, 80, 64, 64, 64, 64}},
        {1920, {0, 176, 352, 528, 704, 896, 1088, 1280, 1472, 1664}, {80, 80, 80, 80, 64, 64, 64, 64, 64}},
    };
    for (const auto& e : expected) {
        // Pixel units (decode layout and the encode split).
        auto pixels = H3VAETiles::split(e.length, 256, 64, 16);
        CHECK(same_layout(pixels, e.starts, 256, e.overlaps));
        if (!same_layout(pixels, e.starts, 256, e.overlaps)) {
            std::fprintf(stderr, "  pixel layout mismatch for length %lld\n", (long long)e.length);
        }
        // Latent units (encode blend, decode latent slices): everything / 16.
        std::vector<int64_t> lat_starts, lat_overlaps;
        for (auto v : e.starts) {
            lat_starts.push_back(v / 16);
        }
        for (auto v : e.overlaps) {
            lat_overlaps.push_back(v / 16);
        }
        auto latents = H3VAETiles::scale_down(pixels, 16);
        CHECK(same_layout(latents, lat_starts, 16, lat_overlaps));
        // Tiles cover the axis exactly.
        CHECK(pixels.starts.back() + pixels.size == e.length);
    }
    // One tile when it fits.
    CHECK(same_layout(H3VAETiles::split(256, 256, 64, 16), {0}, 256, {}));
    CHECK(same_layout(H3VAETiles::split(160, 256, 64, 16), {0}, 160, {}));
    // Slack not a multiple of the step stays in the last tile's tail (the reference does the same).
    CHECK(same_layout(H3VAETiles::split(1000, 256, 64, 16), {0, 176, 368, 560, 752}, 256, {80, 64, 64, 64}));

    // 2 x 2 tiles of 4 x 4 over a 6 x 6 output (overlap 2 both ways), constant tiles 0, 4 / 8, 16.
    auto axis = H3VAETiles::split(6, 4, 2, 1);
    CHECK(same_layout(axis, {0, 2}, 4, {2}));
    const float values[2][2] = {{0.f, 4.f}, {8.f, 16.f}};
    std::vector<float> buffers[2][2];
    std::vector<std::vector<H3VAETiles::TileView>> views(2, std::vector<H3VAETiles::TileView>(2));
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            buffers[i][j].assign(16, values[i][j]);
            views[i][j] = {buffers[i][j].data(), 4, 4};
        }
    }
    std::vector<float> raw(36, -1.f), canvas(36, -1.f);
    H3VAETiles::stitch_raw(views, axis, axis, 1, raw.data(), 6, 6);
    H3VAETiles::CanvasStitcher stitcher(axis, axis, 1, canvas.data(), 6, 6);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            stitcher.add(i, j, views[i][j]);
        }
    }
    // Row 0 only: both blends agree, linear over the overlap with weight x / extent for the newer tile.
    const float row0[6] = {0.f, 0.f, 0.f, 2.f, 4.f, 4.f};
    for (int x = 0; x < 6; ++x) {
        CHECK(std::fabs(raw[x] - row0[x]) < 1e-6f);
        CHECK(std::fabs(canvas[x] - row0[x]) < 1e-6f);
    }
    // The overlap corner (3, 3): raw neighbours give 9, the already blended canvas gives 7.
    CHECK(std::fabs(raw[3 * 6 + 3] - 9.f) < 1e-6f);
    CHECK(std::fabs(canvas[3 * 6 + 3] - 7.f) < 1e-6f);
    // Nothing left unwritten.
    for (int k = 0; k < 36; ++k) {
        CHECK(raw[k] >= 0.f && canvas[k] >= 0.f);
    }

    if (failures == 0) {
        std::printf("h3_vae_tiles_test: OK\n");
    }
    return failures == 0 ? 0 : 1;
}
