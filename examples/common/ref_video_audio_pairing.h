#ifndef __SD_EXAMPLES_REF_VIDEO_AUDIO_PAIRING_H__
#define __SD_EXAMPLES_REF_VIDEO_AUDIO_PAIRING_H__

// Pairing of reference-video soundtracks with reference videos. With no
// indices, soundtrack i belongs to video i. With indices, soundtrack i belongs
// to video indices[i]; there must be one index per soundtrack, each in range
// and used at most once.

#include <cstddef>
#include <string>
#include <vector>

namespace sd_ref_video_audio {

    // Returns an empty string when valid, otherwise an error message.
    inline std::string validate(const std::vector<int>& indices, size_t audio_count, size_t video_count) {
        if (indices.empty()) {
            if (audio_count > video_count) {
                return "each --ref-video-audio needs a corresponding --ref-video";
            }
            return {};
        }
        if (indices.size() != audio_count) {
            return "got " + std::to_string(indices.size()) + " --ref-video-audio-index / ref_video_audio_indices value(s) for " +
                   std::to_string(audio_count) + " soundtrack(s); give one per soundtrack";
        }
        std::vector<bool> used(video_count, false);
        for (int video_index : indices) {
            if (video_index < 0 || static_cast<size_t>(video_index) >= video_count) {
                return "soundtrack video index " + std::to_string(video_index) + " is out of range for " +
                       std::to_string(video_count) + " reference video(s)";
            }
            if (used[static_cast<size_t>(video_index)]) {
                return "reference video " + std::to_string(video_index) + " has more than one soundtrack";
            }
            used[static_cast<size_t>(video_index)] = true;
        }
        return {};
    }

    // Soundtrack index for `video_index`, or -1 when that video is silent.
    inline int audio_for_video(const std::vector<int>& indices, size_t audio_count, size_t video_index) {
        if (indices.empty()) {
            return video_index < audio_count ? static_cast<int>(video_index) : -1;
        }
        for (size_t j = 0; j < indices.size() && j < audio_count; ++j) {
            if (indices[j] >= 0 && static_cast<size_t>(indices[j]) == video_index) {
                return static_cast<int>(j);
            }
        }
        return -1;
    }

}  // namespace sd_ref_video_audio

#endif  // __SD_EXAMPLES_REF_VIDEO_AUDIO_PAIRING_H__
