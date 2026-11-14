// Standalone check of the MiniMax-H3 reference audio resampler and crop: g++ -O2 -std=c++17 -I src scripts/h3_audio_resample_test.cpp && ./a.out
// Add -DH3_AUDIO_LEGACY to run the same checks against the previous linear interpolation with zero padding (expected to fail).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "runtime/sinc_resample.h"

static int failures = 0;
#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                         \
        }                                                                       \
    } while (0)

#ifdef H3_AUDIO_LEGACY
// The previous implementation: linear interpolation, then right zero padding to 800 samples.
static std::vector<float> prepare(const float* data, int64_t n, int ch, int64_t sr, int64_t tsr, int64_t* out_samples) {
    uint64_t output_samples = std::max<uint64_t>(1, (uint64_t)std::llround((long double)n * tsr / sr));
    uint64_t padded         = (output_samples + 799) / 800 * 800;
    std::vector<float> planar(2 * padded, 0.f);
    for (uint64_t i = 0; i < output_samples; ++i) {
        long double pos = (long double)i * sr / tsr;
        uint64_t s0     = std::min<uint64_t>((uint64_t)pos, n - 1);
        uint64_t s1     = std::min<uint64_t>(s0 + 1, n - 1);
        float f         = (float)(pos - s0);
        for (int c = 0; c < 2; ++c) {
            int sc                 = ch == 1 ? 0 : std::min(c, ch - 1);
            float a                = data[s0 * ch + sc];
            float b                = data[s1 * ch + sc];
            planar[c * padded + i] = std::clamp(a + (b - a) * f, -1.f, 1.f);
        }
    }
    *out_samples = (int64_t)padded;
    return planar;
}
static std::vector<float> resample_mono(const std::vector<float>& x, int64_t sr, int64_t tsr) {
    int64_t n  = 0;
    auto p     = prepare(x.data(), (int64_t)x.size(), 1, sr, tsr, &n);
    uint64_t L = std::max<uint64_t>(1, (uint64_t)std::llround((long double)x.size() * tsr / sr));
    return std::vector<float>(p.begin(), p.begin() + L);
}
#else
static std::vector<float> prepare(const float* data, int64_t n, int ch, int64_t sr, int64_t tsr, int64_t* out_samples) {
    return sd::sinc_resample::prepare_h3_reference_waveform(data, n, ch, sr, tsr, 800, out_samples);
}
static std::vector<float> resample_mono(const std::vector<float>& x, int64_t sr, int64_t tsr) {
    return sd::sinc_resample::resample(x.data(), (int64_t)x.size(), 1, sr, tsr);
}
#endif

static bool near(double a, double b, double tol) {
    return std::fabs(a - b) <= tol;
}

int main() {
    const double pi = 3.14159265358979323846;

    // Golden values from the reference resampler (float64) on x[i] = 0.6 sin(0.05 i) + 0.3 sin(1.3 i), 2000 samples.
    std::vector<float> x(2000);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = (float)(0.6 * std::sin(0.05 * i) + 0.3 * std::sin(1.3 * i));
    }
    struct Golden {
        int64_t sr;
        size_t len;
        size_t idx[5];
        double val[5];
    };
    const Golden goldens[] = {
        {44100, 1452, {0, 1, 100, 777, 1451}, {0.0395881442015768, 0.3229993032050239, 0.3169402644144828, -0.17368200237609455, -0.2234446858477057}},
        {48000, 1334, {0, 1, 100, 777, 1333}, {0.0513550577298757, 0.30993691259582273, 0.6291532535062806, 0.8295677661836642, -0.27957241697563684}},
        {22050, 2903, {0, 1, 100, 777, 2902}, {0.0013031131204535998, 0.22731650922449828, 0.12015761748340194, 0.3030123683604793, -0.18469472202031625}},
    };
    for (const auto& g : goldens) {
        auto y = resample_mono(x, g.sr, 32000);
        CHECK(y.size() == g.len);
        if (y.size() != g.len) {
            continue;
        }
        for (int k = 0; k < 5; ++k) {
            CHECK(near(y[g.idx[k]], g.val[k], 1e-5));
        }
    }

    // Output length rule: ceil(new * L / orig) after gcd reduction.
    CHECK(sd::sinc_resample::output_length(1001, 48000, 32000) == 668);
    CHECK(sd::sinc_resample::output_length(12345, 44100, 32000) == 8958);
    CHECK(sd::sinc_resample::output_length(7357, 22050, 32000) == 10677);
    CHECK(sd::sinc_resample::output_length(5, 32000, 32000) == 5);

    // Anti-aliasing: a 20 kHz tone at 48 kHz is above the 16 kHz output Nyquist and must be removed,
    // not folded down to 12 kHz (the reference leaves an RMS of about 0.003).
    {
        std::vector<float> tone(48000);
        for (size_t i = 0; i < tone.size(); ++i) {
            tone[i] = (float)(0.8 * std::sin(2 * pi * 20000.0 * i / 48000.0));
        }
        auto y      = resample_mono(tone, 48000, 32000);
        double sum2 = 0.0;
        size_t n    = 0;
        for (size_t i = 100; i + 100 < y.size(); ++i, ++n) {
            sum2 += (double)y[i] * y[i];
        }
        CHECK(n > 0 && std::sqrt(sum2 / n) < 0.01);
    }

    // Centre crop to whole 800-sample frames: 48 kHz, 1300 samples -> 867 -> 800 kept, offset 33.
    {
        std::vector<float> ramp(1300 * 2);
        for (int i = 0; i < 1300; ++i) {
            ramp[2 * i]     = 0.25f;
            ramp[2 * i + 1] = -0.5f;
        }
        int64_t n   = 0;
        auto planar = prepare(ramp.data(), 1300, 2, 48000, 32000, &n);
        CHECK(n == 800);
        if (n == 800) {
            // Interior of a constant stays constant; no zero padding at the tail.
            CHECK(near(planar[400], 0.25, 2e-3));
            CHECK(near(planar[n - 1], 0.25, 3e-2));
            CHECK(near(planar[n + 400], -0.5, 4e-3));
        }
    }

    // Already at 32 kHz: 2001 samples are centre-cropped to 1600, offset 200; mono goes to both channels.
    {
        std::vector<float> mono(2001);
        for (int i = 0; i < 2001; ++i) {
            mono[i] = (float)i / 4000.f;
        }
        int64_t n   = 0;
        auto planar = prepare(mono.data(), 2001, 1, 32000, 32000, &n);
        CHECK(n == 1600);
        if (n == 1600) {
            CHECK(planar[0] == mono[200]);
            CHECK(planar[1599] == mono[1799]);
            CHECK(planar[1600] == mono[200]);
        }
    }

    // First two of three channels kept, values clamped to [-1, 1].
    {
        std::vector<float> three(800 * 3);
        for (int i = 0; i < 800; ++i) {
            three[3 * i]     = 3.f;
            three[3 * i + 1] = -0.125f;
            three[3 * i + 2] = 0.9f;
        }
        int64_t n   = 0;
        auto planar = prepare(three.data(), 800, 3, 32000, 32000, &n);
        CHECK(n == 800);
        if (n == 800) {
            CHECK(planar[10] == 1.f);
            CHECK(planar[800 + 10] == -0.125f);
        }
    }

    // Shorter than one frame after resampling (48 kHz, 1001 samples -> 668): nothing to encode.
    {
        std::vector<float> s(1001, 0.1f);
        int64_t n   = -1;
        auto planar = prepare(s.data(), 1001, 1, 48000, 32000, &n);
        CHECK(planar.empty() && n == 0);
    }

    // Speed: 15 s of 48 kHz and 44.1 kHz stereo.
    for (int64_t sr : {48000, 44100}) {
        std::vector<float> clip((size_t)(sr * 15 * 2));
        for (size_t i = 0; i < clip.size(); ++i) {
            clip[i] = (float)std::sin(0.001 * i);
        }
        int64_t n   = 0;
        auto start  = std::chrono::steady_clock::now();
        auto planar = prepare(clip.data(), sr * 15, 2, sr, 32000, &n);
        double ms   = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::printf("15 s %lld Hz stereo: %.1f ms\n", (long long)sr, ms);
        CHECK(n == 480000);
        CHECK(ms < 500.0);
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
