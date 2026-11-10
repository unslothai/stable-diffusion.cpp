#ifndef __SD_CORE_DEVICE_RELEASE_CREDIT_H__
#define __SD_CORE_DEVICE_RELEASE_CREDIT_H__

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace sd {
    // Device memory this process released that a free-memory reading may not show yet.
    //
    // Some drivers report a release late (Windows WDDM destroys allocations asynchronously), so a reading
    // taken right after a large free can count that memory as used. Memory used outside what the process
    // tracks is (total - free - tracked). That figure growing right after the process released tracked
    // memory is credited back, up to the bytes released since the last consistent reading. A driver that
    // reports releases at once shows no growth, so the credit is 0 and the reading is used as is.
    // window_ms bounds how long a release can be credited, so growth that really is another process taking
    // memory stops being credited; 0 disables the credit. Credits below MIN_CREDIT_BYTES are ignored: allocator
    // pools growing while the process releases a little look the same, and the capacity margin covers them.
    // A caller that relies on a credit confirms it with a real allocation of what it needs (set_verified) or drops it
    // (reject); a later, larger request needs its own confirmation.
    class DeviceReleaseCredit {
    public:
        static constexpr size_t MIN_CREDIT_BYTES = 256ULL * 1024ULL * 1024ULL;

        explicit DeviceReleaseCredit(int64_t window_ms = 0)
            : window_ms_(window_ms) {}

        void note_release(size_t bytes, int64_t now_ms) {
            if (bytes == 0) {
                return;
            }
            released_ = bytes > UINT64_MAX - released_ ? UINT64_MAX : released_ + bytes;
            if (anchored_ && first_release_ms_ < 0) {
                first_release_ms_ = now_ms;
            }
        }

        // Bytes to add to reported_free. fresh: the reading was just taken (only fresh readings re-anchor).
        size_t credit(size_t reported_free, size_t total, size_t tracked_bytes, int64_t now_ms, bool fresh) {
            if (total == 0) {
                return 0;
            }
            const size_t used      = total - std::min(reported_free, total);
            const size_t untracked = used > tracked_bytes ? used - tracked_bytes : 0;
            const uint64_t pending = released_ - released_at_anchor_;
            size_t result          = 0;
            if (anchored_ && pending > 0 && window_ms_ > 0 && first_release_ms_ >= 0 &&
                now_ms - first_release_ms_ <= window_ms_ && untracked > anchor_untracked_) {
                result = static_cast<size_t>(std::min<uint64_t>(untracked - anchor_untracked_, pending));
                if (result < MIN_CREDIT_BYTES) {
                    result = 0;
                }
            }
            if (fresh && result == 0) {
                anchored_           = true;
                anchor_untracked_   = untracked;
                released_at_anchor_ = released_;
                first_release_ms_   = -1;
                verified_bytes_     = 0;
            }
            return result;
        }

        bool verified(size_t bytes) const { return bytes <= verified_bytes_; }
        void set_verified(size_t bytes) { verified_bytes_ = std::max(verified_bytes_, bytes); }
        // The next fresh reading re-anchors, so nothing released so far is credited again.
        void reject() {
            anchored_       = false;
            verified_bytes_ = 0;
        }

    private:
        int64_t window_ms_          = 0;
        bool anchored_              = false;
        size_t anchor_untracked_    = 0;
        uint64_t released_          = 0;
        uint64_t released_at_anchor_ = 0;
        int64_t first_release_ms_   = -1;
        size_t verified_bytes_      = 0;
    };
}

#endif  // __SD_CORE_DEVICE_RELEASE_CREDIT_H__
