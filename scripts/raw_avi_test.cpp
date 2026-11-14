// Uncompressed AVI ("avi_raw") round trip: frames and float audio come back bit-exact from the movi chunks.
// Build after a CPU build in ./build (objects of the sd-cli target):
//   O=build/examples/cli/CMakeFiles/sd-cli.dir/__/common; g++ -std=c++17 -O2 -fopenmp -DGGML_MAX_NAME=160 \
//     -I examples -I src -I include -I ggml/include -I thirdparty scripts/raw_avi_test.cpp $O/*.o \
//     build/libstable-diffusion.a build/ggml/src/libggml.a build/ggml/src/libggml-cpu.a \
//     build/ggml/src/libggml-base.a -lpthread -o build/raw_avi_test && ./build/raw_avi_test [out.avi]
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common/media_io.h"

static int failures = 0;
#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                         \
        }                                                                       \
    } while (0)

static uint32_t u32(const uint8_t* p) {
    return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

int main(int argc, char** argv) {
    const uint32_t w = 66, h = 34, n = 5;  // width * 3 not a multiple of 4: rows are padded
    std::vector<std::vector<uint8_t>> pixels(n, std::vector<uint8_t>(w * h * 3));
    std::vector<sd_image_t> frames(n);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < w * h * 3; ++j) {
            pixels[i][j] = static_cast<uint8_t>((j * 7 + i * 31) % 256);
        }
        frames[i] = {w, h, 3, pixels[i].data()};
    }
    std::vector<float> samples(2 * 1000);
    for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = static_cast<float>((static_cast<int>(i * 37) % 2001) - 1000) / 999.f + 1e-6f * i;
    }
    sd_audio_t audio{};
    audio.data         = samples.data();
    audio.sample_count = 1000;
    audio.channels     = 2;
    audio.sample_rate  = 32000;

    auto avi = create_video_from_sd_images_to_vector("avi_raw", frames.data(), n, 24, 90, &audio, "");
    CHECK(!avi.empty());
    if (argc > 1) {
        FILE* f = std::fopen(argv[1], "wb");
        std::fwrite(avi.data(), 1, avi.size(), f);
        std::fclose(f);
    }
    const uint8_t* movi = nullptr;
    for (size_t i = 0; i + 4 <= avi.size(); ++i) {
        if (std::memcmp(avi.data() + i, "movi", 4) == 0) {
            movi = avi.data() + i + 4;
            break;
        }
    }
    CHECK(movi != nullptr);
    const size_t row = (w * 3 + 3) & ~3u;
    uint32_t seen    = 0;
    bool audio_exact = false;
    for (const uint8_t* p = movi; movi && p + 8 <= avi.data() + avi.size();) {
        uint32_t size = u32(p + 4);
        if (std::memcmp(p, "00db", 4) == 0) {
            CHECK(size == row * h);
            bool exact = true;
            for (uint32_t y = 0; y < h; ++y) {
                for (uint32_t x = 0; x < w; ++x) {
                    const uint8_t* d = p + 8 + y * row + 3 * x;
                    const uint8_t* s = pixels[seen].data() + ((h - 1 - y) * w + x) * 3;
                    exact &= d[0] == s[2] && d[1] == s[1] && d[2] == s[0];
                }
            }
            CHECK(exact);
            seen++;
        } else if (std::memcmp(p, "01wb", 4) == 0) {
            audio_exact = size == samples.size() * 4 && std::memcmp(p + 8, samples.data(), size) == 0;
        } else {
            break;
        }
        p += 8 + size + (size & 1);
    }
    CHECK(seen == n);
    CHECK(audio_exact);
    // The MJPG path still exists and differs.
    auto mjpg = create_video_from_sd_images_to_vector("avi", frames.data(), n, 24, 90, &audio, "");
    CHECK(!mjpg.empty() && mjpg.size() < avi.size());
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("raw_avi_test: all checks passed (%zu bytes)\n", avi.size());
    return 0;
}
