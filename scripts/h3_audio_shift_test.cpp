// Standalone check of the MiniMax-H3 audio shift carry: g++ -O2 -std=c++17 -fopenmp -I src -I include -I ggml/include scripts/h3_audio_shift_test.cpp build/libstable-diffusion.a build/ggml/src/libggml.a build/ggml/src/libggml-cpu.a build/ggml/src/libggml-base.a -lpthread && ./a.out
// Needs a configured build/ tree. The denoiser's audio shift must be settable (default 3) and the
// carried audio scale must follow it, matching sigma_v / sigma_a at the clean end of the schedule.
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "runtime/denoiser.hpp"

static int failures = 0;
#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                         \
        }                                                                       \
    } while (0)

// Same mapping as the DiT's time_shift_sigma: video sigma at shift `from` to audio sigma at shift `to`.
static double shift_sigma(double sigma, double from, double to) {
    double base = sigma / (from + sigma * (1.0 - from));
    return to * base / (1.0 + (to - 1.0) * base);
}

static double audio_scale_after_in(H3AVFlowDenoiser& d, int64_t video_channels) {
    sd::Tensor<float> latent({2, 2, 1, video_channels + 2, 1});
    latent.fill_(1.f);
    auto carried = d.process_latent_in(latent);
    if (carried.empty()) {
        return 1.0;  // scale 1 leaves the latent untouched
    }
    return carried.index(0, 0, 0, video_channels, 0);
}

int main() {
    const int64_t video_channels = 4;
    H3AVFlowDenoiser d(12.f, 3.f, video_channels);
    CHECK(std::fabs(audio_scale_after_in(d, video_channels) - 4.0) < 1e-6);  // default 12 / 3

    d.set_audio_shift(5.f);
    const double scale = audio_scale_after_in(d, video_channels);
    CHECK(std::fabs(scale - 12.0 / 5.0) < 1e-6);
    // The carry matches sigma_v / sigma_a as sigma -> 0 with the same audio shift the DiT receives.
    const double s = 1e-6;
    CHECK(std::fabs(s / shift_sigma(s, 12.0, 5.0) - scale) < 1e-4);

    // Out undoes in.
    sd::Tensor<float> latent({2, 2, 1, video_channels + 2, 1});
    latent.fill_(0.5f);
    auto round_trip = d.process_latent_out(d.process_latent_in(latent));
    CHECK(std::fabs(round_trip.index(1, 1, 0, video_channels + 1, 0) - 0.5f) < 1e-6);
    CHECK(std::fabs(round_trip.index(1, 1, 0, 0, 0) - 0.5f) < 1e-6);

    // Equal shifts: no carry at all.
    d.set_shift(5.f);
    CHECK(audio_scale_after_in(d, video_channels) == 1.0);

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
