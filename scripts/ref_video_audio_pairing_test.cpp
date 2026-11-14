// Standalone check of reference-video soundtrack pairing: g++ -std=c++17 -I examples/common scripts/ref_video_audio_pairing_test.cpp && ./a.out
// Add -DREF_VIDEO_AUDIO_LEGACY to run the same checks against the previous position-only pairing (expected to fail).
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef REF_VIDEO_AUDIO_LEGACY
// Previous behaviour: soundtrack i always belongs to video i; no explicit indices.
static std::string validate(const std::vector<int>&, size_t audio_count, size_t video_count) {
    return audio_count > video_count ? "each --ref-video-audio needs a corresponding --ref-video" : "";
}
static int audio_for_video(const std::vector<int>&, size_t audio_count, size_t video_index) {
    return video_index < audio_count ? static_cast<int>(video_index) : -1;
}
#else
#include "ref_video_audio_pairing.h"
using sd_ref_video_audio::audio_for_video;
using sd_ref_video_audio::validate;
#endif

static int failures = 0;
#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                         \
        }                                                                       \
    } while (0)

int main() {
    // Positional pairing is unchanged.
    CHECK(validate({}, 2, 3).empty());
    CHECK(!validate({}, 3, 2).empty());
    CHECK(audio_for_video({}, 2, 0) == 0);
    CHECK(audio_for_video({}, 2, 1) == 1);
    CHECK(audio_for_video({}, 2, 2) == -1);

    // A silent video followed by one with a soundtrack.
    CHECK(validate({1}, 1, 2).empty());
    CHECK(audio_for_video({1}, 1, 0) == -1);
    CHECK(audio_for_video({1}, 1, 1) == 0);

    // Out of order: soundtrack 0 -> video 2, soundtrack 1 -> video 0, video 1 silent.
    CHECK(validate({2, 0}, 2, 3).empty());
    CHECK(audio_for_video({2, 0}, 2, 0) == 1);
    CHECK(audio_for_video({2, 0}, 2, 1) == -1);
    CHECK(audio_for_video({2, 0}, 2, 2) == 0);

    // Errors: count mismatch, out of range, duplicate.
    CHECK(!validate({0}, 2, 3).empty());
    CHECK(!validate({0, 1}, 1, 3).empty());
    CHECK(!validate({3}, 1, 3).empty());
    CHECK(!validate({-1}, 1, 3).empty());
    CHECK(!validate({1, 1}, 2, 3).empty());
    CHECK(!validate({0}, 1, 0).empty());

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
