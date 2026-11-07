#include "tnc/tnc.hpp"

#include <cctype>
#include <sstream>
#include <string>

namespace data2g::tnc {

std::optional<bool> refs_command(std::span<const std::uint8_t> payload) {
    std::string t(payload.begin(), payload.end());  // whitespace splits, so only NUL/CR/LF need trimming
    for (auto& ch : t) ch = (ch == '\0' || ch == '\r' || ch == '\n') ? ' ' : static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    std::istringstream in(t);
    std::string a, b, rest;
    if (!(in >> a >> b) || (in >> rest) || a != "REFS") return std::nullopt;
    if (b == "ON") return true;
    if (b == "OFF") return false;
    return std::nullopt;
}

Bytes kiss_encode(std::span<const std::uint8_t> data, int port, int cmd) {
    Bytes out{FEND};
    for (int i = -1; i < static_cast<int>(data.size()); ++i) {
        const std::uint8_t b = i < 0 ? static_cast<std::uint8_t>(port << 4 | cmd) : data[static_cast<std::size_t>(i)];
        if (b == FESC) out.insert(out.end(), {FESC, TFESC});
        else if (b == FEND) out.insert(out.end(), {FESC, TFEND});
        else out.push_back(b);
    }
    out.push_back(FEND);
    return out;
}

std::vector<std::pair<int, Bytes>> KissDecoder::feed(std::span<const std::uint8_t> data) {
    std::vector<std::pair<int, Bytes>> out;
    for (const std::uint8_t b : data) {
        if (b == FEND) {
            if (!buf_.empty() && !skip_) out.emplace_back(buf_[0], Bytes(buf_.begin() + 1, buf_.end()));
            buf_.clear();
            esc_ = skip_ = false;
        } else if (skip_) {
        } else if (buf_.size() >= KISS_MAX_FRAME) {  // no FEND in sight
            buf_.clear();
            skip_ = true;
        } else if (esc_) {
            buf_.push_back(b == TFEND ? FEND : b == TFESC ? FESC : b);
            esc_ = false;
        } else if (b == FESC) {
            esc_ = true;
        } else {
            buf_.push_back(b);
        }
    }
    return out;
}

}  // namespace data2g::tnc
