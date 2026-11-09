// Prompt decoration for Cosmos3 transfer, matching the training-time augmentors.
//
// Ported from NVIDIA cosmos-framework (cosmos_framework/inference/inference.py,
// _format_sample_prompt and helpers).
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the OpenMDW License 1.1 (https://openmdw.ai/license/1-1/);
// see licenses/OpenMDW-1.1.txt and NOTICE.

#ifndef __SD_PIPELINE_COSMOS3_PROMPT_H__
#define __SD_PIPELINE_COSMOS3_PROMPT_H__

#include <charconv>
#include <cmath>
#include <cstdio>
#include <string>

#include "json.hpp"

namespace sd::cosmos3 {

    namespace detail {
        inline void append_python_string(std::string& out, const std::string& s) {
            out += '"';
            for (size_t i = 0; i < s.size();) {
                const unsigned char c = static_cast<unsigned char>(s[i]);
                uint32_t cp           = c;
                size_t len            = 1;
                if (c >= 0xF0 && i + 3 < s.size()) {
                    cp  = ((c & 0x07u) << 18) | ((s[i + 1] & 0x3Fu) << 12) | ((s[i + 2] & 0x3Fu) << 6) | (s[i + 3] & 0x3Fu);
                    len = 4;
                } else if (c >= 0xE0 && i + 2 < s.size()) {
                    cp  = ((c & 0x0Fu) << 12) | ((s[i + 1] & 0x3Fu) << 6) | (s[i + 2] & 0x3Fu);
                    len = 3;
                } else if (c >= 0xC0 && i + 1 < s.size()) {
                    cp  = ((c & 0x1Fu) << 6) | (s[i + 1] & 0x3Fu);
                    len = 2;
                }
                i += len;
                char buf[16];
                switch (cp) {
                    case '"':
                        out += "\\\"";
                        continue;
                    case '\\':
                        out += "\\\\";
                        continue;
                    case '\n':
                        out += "\\n";
                        continue;
                    case '\r':
                        out += "\\r";
                        continue;
                    case '\t':
                        out += "\\t";
                        continue;
                    case '\b':
                        out += "\\b";
                        continue;
                    case '\f':
                        out += "\\f";
                        continue;
                    default:
                        break;
                }
                if (cp >= 0x20 && cp < 0x7F) {
                    out += static_cast<char>(cp);
                } else if (cp < 0x10000) {
                    std::snprintf(buf, sizeof(buf), "\\u%04x", cp);
                    out += buf;
                } else {
                    const uint32_t v = cp - 0x10000;
                    std::snprintf(buf, sizeof(buf), "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF));
                    out += buf;
                }
            }
            out += '"';
        }

        inline void append_python_float(std::string& out, double v) {
            if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e16) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.1f", v);
                out += buf;
                return;
            }
            char buf[64];
            auto res = std::to_chars(buf, buf + sizeof(buf), v);
            out.append(buf, res.ptr);
        }

        // json.dumps() with Python's defaults: ", " / ": " separators, ensure_ascii.
        inline void dump_python(std::string& out, const nlohmann::ordered_json& j) {
            if (j.is_object()) {
                out += '{';
                bool first = true;
                for (auto it = j.begin(); it != j.end(); ++it) {
                    if (!first) {
                        out += ", ";
                    }
                    first = false;
                    append_python_string(out, it.key());
                    out += ": ";
                    dump_python(out, it.value());
                }
                out += '}';
            } else if (j.is_array()) {
                out += '[';
                for (size_t i = 0; i < j.size(); ++i) {
                    if (i > 0) {
                        out += ", ";
                    }
                    dump_python(out, j[i]);
                }
                out += ']';
            } else if (j.is_string()) {
                append_python_string(out, j.get<std::string>());
            } else if (j.is_number_float()) {
                append_python_float(out, j.get<double>());
            } else if (j.is_boolean()) {
                out += j.get<bool>() ? "true" : "false";
            } else if (j.is_null()) {
                out += "null";
            } else {
                out += j.dump();
            }
        }

        inline std::string strip(const std::string& s) {
            const size_t b = s.find_first_not_of(" \t\n\r\f\v");
            if (b == std::string::npos) {
                return "";
            }
            const size_t e = s.find_last_not_of(" \t\n\r\f\v");
            return s.substr(b, e - b + 1);
        }

        inline std::string rstrip_dots(std::string s) {
            while (!s.empty() && s.back() == '.') {
                s.pop_back();
            }
            return s;
        }
    }  // namespace detail

    // JSON-object prompts get resolution/duration/fps fields; plain prompts get
    // the duration and resolution sentences appended.
    inline std::string format_prompt(const std::string& prompt, int frames, float fps, int height, int width) {
        nlohmann::ordered_json obj = nlohmann::ordered_json::parse(prompt, nullptr, false);
        if (!obj.is_discarded() && obj.is_object()) {
            if (frames > 1) {
                const int seconds = fps > 0.f ? static_cast<int>(frames / fps) : 0;
                obj["duration"]   = std::to_string(seconds) + "s";
                obj["fps"]        = static_cast<double>(fps);
            } else {
                obj.erase("duration");
                obj.erase("fps");
            }
            obj["resolution"] = nlohmann::ordered_json{{"H", height}, {"W", width}};
            std::string out;
            detail::dump_python(out, obj);
            return out;
        }
        char buf[160];
        std::string text = detail::strip(prompt);
        if (frames > 1) {
            std::snprintf(buf, sizeof(buf), "The video is %.1f seconds long and is of %.0f FPS.", frames / static_cast<double>(fps), static_cast<double>(fps));
            text = detail::rstrip_dots(text) + ". " + buf;
        }
        text = detail::strip(text);
        std::snprintf(buf, sizeof(buf), "This video is of %dx%d resolution.", height, width);
        return detail::rstrip_dots(text) + ". " + buf;
    }

}  // namespace sd::cosmos3

#endif  // __SD_PIPELINE_COSMOS3_PROMPT_H__
