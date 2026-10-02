// data2g/kisslink.py: the KISS TNC's link layer, mode shifting without a
// session. Every burst carries reports (for each station heard lately, the
// mode it should use to reach us, as the ARQ shifter recommends it); a
// connected-mode AX.25 frame goes in its next hop's reported mode, anything
// else in the cap's robust broadcast mode. The burst layout is in the
// Python module's docstring.
#pragma once

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "arq/phy.hpp"
#include "arq/policy.hpp"

namespace data2g::kisslink {

using arq::Bytes;
using arq::ByteView;

inline constexpr int KISS_KEY = 0x4B53;
inline constexpr int VERSION = 1;
inline constexpr double REPORT_MAX_S = 180.0;  // a report older than this is not followed
inline constexpr double HEARD_MAX_S = 600.0;   // stations reported on: heard this recently
inline constexpr double SLOT_S = 1.0;          // the shortest p-persistence slot
inline constexpr double MIN_SUCCESS = 0.9;     // recommended modes' first-transmission success
inline constexpr double BROADCAST_S = arq::SIZE_S.back();
std::string_view broadcast_mode(int cap);  // BROADCAST; throws std::out_of_range

struct Ax25 {
    std::string dst, src;
    std::string next_hop;  // who hears it next on RF
    std::string sender;    // who puts it on RF
    bool connected;        // I, S, or a U frame other than UI
};
std::optional<Ax25> parse_ax25(ByteView frame);  // nullopt: not AX.25
int station_hash(std::string_view call);         // nonzero 16-bit FNV-1a

struct Peer {
    arq::GearShifter shifter;
    double heard = 0.0;                             // when we last heard it
    std::optional<std::pair<int, double>> report;  // (rec byte, time): how it wants us to send to it
};

class KissLink {
public:
    // broadcast empty: broadcast_mode(cap). Throws std::invalid_argument for a
    // broadcast mode outside the cap.
    explicit KissLink(int cap = 2, std::string broadcast = {}, arq::Clock clock = arq::monotonic);

    int cap;
    std::vector<Bytes> queue;                    // frames waiting
    std::vector<std::pair<int, Peer>> peers;    // station hash -> Peer, first heard first
    std::set<int> me;                            // hashes this TNC has sent as
    arq::Clock clock;
    std::int64_t n_sent = 0;
    std::string broadcast;
    // channel access (the engine applies them)
    int persist = 63;  // after BUSY, a slot is taken with probability (P + 1) / 256
    double slot_s = SLOT_S;
    double busy_limit_s = 60.0;  // BUSY held a burst this long: send anyway

    void command(int cmd, ByteView payload);  // KISS P and SLOTTIME; the rest ignored
    void enqueue(Bytes frame) { queue.push_back(std::move(frame)); }
    // The first queued frame's mode, with every queued frame that goes in
    // it, as many as fit. nullptr: nothing queued.
    arq::TxBurstPtr next_burst();
    // A received burst -> its frames, nullopt if it isn't a KISS burst.
    // Updates what we know of its sender. soft: arq::soft_bits(r), if made.
    std::optional<std::vector<Bytes>> on_burst(const arq::Heard& r, std::shared_ptr<const arq::SlotSoft> soft = nullptr);

private:
    std::pair<std::string, int> route(ByteView frame) const;
    std::vector<Bytes> control(const arq::Mode& m, int sender);
    Peer* peer(int h);
};

}  // namespace data2g::kisslink
