#ifndef __SD_RUNTIME_SINC_RESAMPLE_H__
#define __SD_RUNTIME_SINC_RESAMPLE_H__

// Dependency-free bandlimited resampler (windowed sinc with a Hann window).
//
// Defaults follow torchaudio.functional.resample: rates are reduced by their
// gcd, the cutoff is min(orig, new) * rolloff with rolloff 0.99, the kernel
// spans lowpass_filter_width zero crossings on each side, and the output
// length is ceil(new * L / orig) evaluated in float32. One kernel is built per
// output phase (new rate after reduction) and reused for every input frame.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace sd {
    namespace sinc_resample {

        struct Kernel {
            int64_t orig_rate = 1;  // reduced input rate (input samples per output block)
            int64_t new_rate  = 1;  // reduced output rate (phases per block)
            int64_t width     = 0;  // left padding in input samples
            // Per phase: offset of the first non-zero tap relative to block start - width,
            // and the non-zero taps themselves.
            std::vector<int64_t> first_tap;
            std::vector<std::vector<double>> taps;
        };

        // Output length for an input of `length` samples, matching torchaudio's rule.
        inline int64_t output_length(int64_t length, int64_t orig_rate, int64_t new_rate) {
            if (length <= 0 || orig_rate <= 0 || new_rate <= 0) {
                return 0;
            }
            if (orig_rate == new_rate) {
                return length;
            }
            const int64_t g   = std::gcd(orig_rate, new_rate);
            const int64_t o   = orig_rate / g;
            const int64_t n   = new_rate / g;
            const float ratio = static_cast<float>(static_cast<double>(n) * static_cast<double>(length) / static_cast<double>(o));
            return static_cast<int64_t>(std::ceil(ratio));
        }

        inline Kernel make_kernel(int64_t orig_rate,
                                  int64_t new_rate,
                                  int lowpass_filter_width = 6,
                                  double rolloff           = 0.99) {
            Kernel k;
            if (orig_rate <= 0 || new_rate <= 0 || lowpass_filter_width <= 0) {
                return k;
            }
            const int64_t g      = std::gcd(orig_rate, new_rate);
            const int64_t o      = orig_rate / g;
            const int64_t n      = new_rate / g;
            const double base    = static_cast<double>(std::min(o, n)) * rolloff;
            const double lpw     = static_cast<double>(lowpass_filter_width);
            const double pi      = 3.14159265358979323846;
            const int64_t width  = static_cast<int64_t>(std::ceil(lpw * static_cast<double>(o) / base));
            const int64_t n_taps = 2 * width + o;
            const double gain    = base / static_cast<double>(o);
            k.orig_rate          = o;
            k.new_rate           = n;
            k.width              = width;
            k.first_tap.assign(static_cast<size_t>(n), 0);
            k.taps.assign(static_cast<size_t>(n), {});
            std::vector<double> full(static_cast<size_t>(n_taps));
            for (int64_t phase = 0; phase < n; ++phase) {
                for (int64_t tap = 0; tap < n_taps; ++tap) {
                    // Time of input tap relative to output sample `phase`, in input periods, scaled to the cutoff.
                    double t     = (static_cast<double>(tap - width) / static_cast<double>(o) -
                                    static_cast<double>(phase) / static_cast<double>(n)) *
                                   base;
                    double value = 0.0;
                    if (t > -lpw && t < lpw) {
                        const double c      = std::cos(t * pi / lpw / 2.0);
                        const double window = c * c;
                        const double x      = t * pi;
                        value               = (x == 0.0 ? 1.0 : std::sin(x) / x) * window * gain;
                    }
                    full[static_cast<size_t>(tap)] = value;
                }
                int64_t lo = 0;
                int64_t hi = n_taps;
                while (lo < hi && full[static_cast<size_t>(lo)] == 0.0) {
                    ++lo;
                }
                while (hi > lo && full[static_cast<size_t>(hi - 1)] == 0.0) {
                    --hi;
                }
                k.first_tap[static_cast<size_t>(phase)] = lo;
                k.taps[static_cast<size_t>(phase)].assign(full.begin() + lo, full.begin() + hi);
            }
            return k;
        }

        // Resamples one channel. `input` is read with `stride` floats between samples
        // (so interleaved audio can be read in place). Writes output_length() samples.
        inline void resample_channel(const Kernel& k,
                                     const float* input,
                                     int64_t length,
                                     int64_t stride,
                                     float* output,
                                     int64_t out_length) {
            if (length <= 0 || out_length <= 0) {
                return;
            }
            const int64_t o = k.orig_rate;
            const int64_t n = k.new_rate;
            for (int64_t m = 0; m < out_length; ++m) {
                const int64_t block             = m / n;
                const int64_t phase             = m % n;
                const std::vector<double>& taps = k.taps[static_cast<size_t>(phase)];
                // Index into the zero-padded input of the first tap, shifted back to the unpadded input.
                const int64_t start = block * o + k.first_tap[static_cast<size_t>(phase)] - k.width;
                const int64_t count = static_cast<int64_t>(taps.size());
                const int64_t lo    = std::max<int64_t>(0, -start);
                const int64_t hi    = std::min<int64_t>(count, length - start);
                double acc          = 0.0;
                for (int64_t j = lo; j < hi; ++j) {
                    acc += taps[static_cast<size_t>(j)] * static_cast<double>(input[(start + j) * stride]);
                }
                output[m] = static_cast<float>(acc);
            }
        }

        // Convenience wrapper for one channel; equal rates copy the input unchanged.
        inline std::vector<float> resample(const float* input,
                                           int64_t length,
                                           int64_t stride,
                                           int64_t orig_rate,
                                           int64_t new_rate,
                                           int lowpass_filter_width = 6,
                                           double rolloff           = 0.99) {
            std::vector<float> out(static_cast<size_t>(output_length(length, orig_rate, new_rate)));
            if (out.empty()) {
                return out;
            }
            if (orig_rate == new_rate) {
                for (int64_t i = 0; i < length; ++i) {
                    out[static_cast<size_t>(i)] = input[i * stride];
                }
                return out;
            }
            Kernel k = make_kernel(orig_rate, new_rate, lowpass_filter_width, rolloff);
            resample_channel(k, input, length, stride, out.data(), static_cast<int64_t>(out.size()));
            return out;
        }

        // MiniMax-H3 reference soundtrack preparation, as planar stereo at the VAE rate:
        // keep the first two channels (mono is copied to both), resample each, then
        // centre-crop to a whole number of `frame_samples` samples and clamp to [-1, 1].
        // Returns an empty vector when fewer than `frame_samples` samples remain.
        // Layout: [channel 0 samples..., channel 1 samples...]; *out_samples receives
        // the per-channel length.
        inline std::vector<float> prepare_h3_reference_waveform(const float* interleaved,
                                                                int64_t sample_count,
                                                                int channels,
                                                                int64_t sample_rate,
                                                                int64_t target_rate,
                                                                int64_t frame_samples,
                                                                int64_t* out_samples) {
            if (out_samples != nullptr) {
                *out_samples = 0;
            }
            if (interleaved == nullptr || sample_count <= 0 || channels <= 0 || sample_rate <= 0 ||
                target_rate <= 0 || frame_samples <= 0) {
                return {};
            }
            const int kept = std::min(channels, 2);
            std::vector<std::vector<float>> resampled(static_cast<size_t>(kept));
            if (sample_rate == target_rate) {
                for (int c = 0; c < kept; ++c) {
                    resampled[static_cast<size_t>(c)] = resample(interleaved + c, sample_count, channels, sample_rate, target_rate);
                }
            } else {
                Kernel k        = make_kernel(sample_rate, target_rate);
                const int64_t L = output_length(sample_count, sample_rate, target_rate);
                for (int c = 0; c < kept; ++c) {
                    auto& out = resampled[static_cast<size_t>(c)];
                    out.resize(static_cast<size_t>(L));
                    resample_channel(k, interleaved + c, sample_count, channels, out.data(), L);
                }
            }
            const int64_t length = static_cast<int64_t>(resampled[0].size());
            const int64_t kept_n = length / frame_samples * frame_samples;
            if (kept_n <= 0) {
                return {};
            }
            const int64_t offset = (length % frame_samples) / 2;
            std::vector<float> planar(static_cast<size_t>(2 * kept_n));
            for (int c = 0; c < 2; ++c) {
                const auto& src = resampled[static_cast<size_t>(std::min(c, kept - 1))];
                float* dst      = planar.data() + static_cast<size_t>(c) * static_cast<size_t>(kept_n);
                for (int64_t i = 0; i < kept_n; ++i) {
                    dst[i] = std::clamp(src[static_cast<size_t>(offset + i)], -1.f, 1.f);
                }
            }
            if (out_samples != nullptr) {
                *out_samples = kept_n;
            }
            return planar;
        }

    }  // namespace sinc_resample
}  // namespace sd

#endif  // __SD_RUNTIME_SINC_RESAMPLE_H__
