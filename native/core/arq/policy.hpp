// data2g/arq/policy.py: the gear shifter (docs/arq.md §8), one per station.
// The study toggles DATA2G_DROP_MODES and DATA2G_BIAS_FIX are Python only:
// no dropped modes, BIAS_MAX 6.
#pragma once

#include <array>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "arq/modes.hpp"
#include "arq/predictor.hpp"

namespace data2g::arq {

inline constexpr std::array<double, 4> SIZE_S = {1.0, 3.0, 6.0, 12.0};
inline constexpr double TURN_S = 1.3;
inline constexpr double TIMEOUT_S = 1.0 + 1.0 + 1.5;
inline constexpr double PREV_MAX_S = 30.0;
inline constexpr double BIAS_STEP = 1.0, BIAS_MAX = 6.0;
inline constexpr double DUP_BELOW = 0.9;
inline constexpr int CHAT_BYTES = 200;  // frames.CHAT_LINE_BYTES
inline constexpr double CPM_SIZE_SCALE = 4.0;
inline constexpr int CTL_BYTES = 12;
inline constexpr int CPM_CODE = 3;
inline constexpr std::string_view ROBUST_CONNECT = "n4-qpsk-r1/3";

int cap_hz(int cap);  // CAP_HZ; throws for an unknown cap
std::string_view fallback(int cap);
std::string_view connect_mode(int cap, int tries = 0);
double width_hz(const Mode& m);
std::vector<const Mode*> allowed(int cap);  // MODES' order

int encode(std::string_view submode);  // throws for an unknown mode
const Mode* decode(int rec);           // nullptr: no such mode

int ctl_slots(const Mode& m);
int slots_for(const Mode& m, double seconds, bool data = true, bool dup = false);

// What the shifter reads from its link.Station.
struct StationView {
    int cap = 2;
    bool pending = true;  // tx.pending()
    std::optional<int> peer_recommend, peer_reply_recommend;
    int peer_size_hint = 1;
    bool peer_wants_dup = false;
    bool chat = false;     // station.chat or station.peer_chat
    long peer_queued = 0;
    long held = 0;         // codewords held beyond the cumulative ACK: len(rx.buf)
};

struct Recommendation {
    int data, hint, reply;
};

class GearShifter {
public:
    struct Heard {
        Measured m;
        std::string band;
        double at = 0.0;
    };
    struct LogEntry {
        std::string data;
        int hint;
        std::string reply;
    };
    using Map = std::map<std::string, double, std::less<>>;

    double gap_s = 2.5;
    bool use_cpm = true;
    double min_success = 0.0;
    std::optional<Measured> measured;  // the peer's last burst
    std::string measured_band = "w";
    double measured_at = 0.0;
    std::optional<Heard> prev;  // the peer burst before the last
    Map bias, bias_burst;
    bool want_dup = false;
    std::map<std::string, std::pair<double, double>, std::less<>> predicted;
    bool peer_had_data = true;
    std::vector<LogEntry> log;

    std::pair<std::string_view, int> choose(const StationView& st, int escalation) const;
    int next_capacity(const StationView& st) const;
    void observe(const Measured& m, std::string_view submode, double now);
    // usable: nullopt (KISS: no control) = any codeword decoded.
    void outcome(std::string_view submode, int decoded, int sent, std::optional<bool> usable = std::nullopt);
    Recommendation recommend(const StationView& st);
};

}  // namespace data2g::arq
