// A promiscuous receiver: every burst heard as a packet dump, whoever it is
// for (data2g-monitor, the GUI's Monitor window). Nothing here transmits.
//
// Everything on air is plain (docs/arq.md §2): the scrambler is unkeyed and
// a mask only decides whose CRC check passes. So a burst is read under every
// mask it might carry, in order: a broadcast group's (the control names it,
// docs/broadcast.md §2), mask 0 (CONNECT, CONNECT_ACK, CQ and ID frames),
// then each session key learned from a CONNECT or an ID frame, both
// directions. A connected-mode burst under a known key is followed as its
// receiver would: resends mapped from the other side's last ACK, each
// direction's codewords reassembled in seq order and inflated, so the dump
// shows the host bytes. A burst no mask fits is dumped raw, unverified.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "arq/engine.hpp"

namespace data2g::monitor {

using arq::Bytes;
using arq::ByteView;

enum class Format { TEXT, HEX };

// One burst: its description, then its payloads.
struct Dump {
    std::string head;  // lines, each ending in '\n'
    std::vector<std::pair<std::string, Bytes>> payloads;  // (label, bytes)
};

// Non-printables as <AB>; a line break follows each <0A>.
std::string text_dump(ByteView b);
// As xxd: 16 bytes a line, offset, hex in pairs, then the bytes with dots for non-printables.
std::string hex_dump(ByteView b);
// The dump, each payload line indented.
std::string render(const Dump& d, Format f);

class Monitor {
public:
    // A burst the engine heard (BurstHeard::heard set unless lost). `when`
    // starts the first line: the caller's clock.
    Dump burst(const arq::BurstHeard& b, const std::string& when);

    // Its own receiver, for the CLI: audio at FS -> the bursts it completes,
    // as (engine time s, dump). when(t) stamps each.
    std::vector<Dump> feed(std::span<const double> x, const std::function<std::string(double t)>& when);

private:
    struct Pair {
        std::string caller, callee;  // direction 0's and 1's station; empty: not known
        std::string id;              // heard in an ID frame for this key, side unknown
        std::uint64_t seen = 0;      // when last heard (seen_)
    };
    // One direction of a session: what its station has said it holds, and
    // the stream it sends, as reassembled here.
    struct Side {
        std::map<int, std::pair<int, std::set<int>>> snaps;  // its burst seq -> (cum, received above it), 7-bit
        arq::RxSide rx;
        bool started = false;
        int epoch = 0;
        bool epoch_known = false;
        std::map<std::int64_t, arq::SoftEntry> soft;  // resends' soft bits, combined (IR) until one decodes
        std::int64_t gap_plain = -1;                  // rx.plain at the last gap (-1: none)
    };

    std::string who(int key, int direction) const;
    void learn(int key, const std::string& caller = {}, const std::string& callee = {}, const std::string& id = {});
    bool broadcast(arq::ModemRx& rx, Dump& d);
    bool keyed(arq::ModemRx& rx, Dump& d);
    void session_frame(const Bytes& body, int key, int direction, Dump& d);
    void arq_burst(arq::ModemRx& rx, const arq::Control& ctl, int key, int direction, int dup, Dump& d);
    void deliver(Side& s, int key, int direction, std::int64_t seq, Bytes p, bool comp, Dump& d);
    void skip(Side& s, int key, int direction, std::int64_t to, Dump& d);  // jump the stream to `to` over what wasn't heard
    void flush(Side& s, int key, int direction, Dump& d);                  // its delivered bytes -> a payload

    std::map<int, Pair> keys_;                            // session key -> its stations
    std::map<int, std::pair<std::string, std::string>> connects_;  // CONNECT nonce -> (caller, callee)
    std::map<std::pair<int, int>, Side> sides_;           // (key, direction)
    std::uint64_t seen_ = 0;
    std::unique_ptr<tnc::Receiver> receiver_;             // feed()'s, made on first use
};

}  // namespace data2g::monitor
