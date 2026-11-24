// FM ping / level check (data2g/arq/ping.py, normative; FM plan M7): an identity burst with the sounding
// probe (arq/probe) behind it in one over; the reply carries a probe report and its own probe. FM only.
#pragma once
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include "arq/frames.hpp"
namespace data2g::arq::ping {
inline const std::string PING_MODE = "f55-qpsk-r21";
inline constexpr double REPLY_HOLDOFF_S = 10.0;
inline constexpr double CLIP_DBFS = -0.1;
struct Level {
    double rms_dbfs = -64, peak_dbfs = -64;
    int clip_permille = 0;
};
double tx_level(std::string_view tx_path);  // the quietest top mode's gain on mic, 1 on flat
Bytes ping_body(const std::string& frm, const std::string& to, const std::optional<Bytes>& report);
// -> (from, to, report if a reply); throws std::invalid_argument if malformed
std::tuple<std::string, std::string, std::optional<Bytes>> unpack_ping(ByteView body);
Level level(std::span<const double> x);
}  // namespace data2g::arq::ping
