#ifndef __SD_CONDITIONING_TEXT_SEGMENTS_H__
#define __SD_CONDITIONING_TEXT_SEGMENTS_H__

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

// Splits `text` at the byte offsets in `starts` (any order, duplicates and out-of-range offsets ignored).
// Prompts assembled from labels, timestamps and free text are tokenized one piece at a time, so a BPE
// merge never spans two pieces (": " followed by "<0.5 seconds>" must not become ":" + " <" + ...).
inline std::vector<std::string> split_text_segments(const std::string& text, std::vector<size_t> starts) {
    starts.push_back(0);
    starts.push_back(text.size());
    std::sort(starts.begin(), starts.end());
    starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
    std::vector<std::string> pieces;
    for (size_t i = 0; i + 1 < starts.size() && starts[i + 1] <= text.size(); ++i) {
        if (starts[i + 1] > starts[i]) {
            pieces.push_back(text.substr(starts[i], starts[i + 1] - starts[i]));
        }
    }
    return pieces;
}

// Offsets of `starts` that fall before `end`, for encoding a prefix of the same text with the same splits.
inline std::vector<size_t> text_segment_starts_before(const std::vector<size_t>& starts, size_t end) {
    std::vector<size_t> result;
    for (size_t start : starts) {
        if (start < end) {
            result.push_back(start);
        }
    }
    return result;
}

#endif  // __SD_CONDITIONING_TEXT_SEGMENTS_H__
