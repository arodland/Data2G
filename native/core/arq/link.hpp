// data2g/arq/link.py: one station's data transfer state for both
// directions, independent of the PHY (docs/arq.md §3-§6, §9a, §10).
//
// A line-for-line port: the same frames for the same inputs, the same calls
// on the policy and the burst in the same order. Seqs are unbounded
// (int64) here, 7 bits on the wire.
#pragma once

#include <compare>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "arq/frames.hpp"
#include "arq/predictor.hpp"

namespace data2g::arq {

inline constexpr int NO_PROGRESS_TURNS = 8;
inline constexpr int RESYNCS_BEFORE_FAIL = 3;
inline constexpr int REPEATS_BEFORE_SHRINK = 1;
inline constexpr int MAX_ESCALATION = 4;
inline constexpr int FLOOR_DECAY_TURNS = 4;  // clean turns (no escalation) that lower the escalation floor by one
inline constexpr int LINK_LOST_MISSES = 12;

// Python's logging, by logger name ("data2g.link", "data2g.session") and
// level (20 info, 30 warning). Unset: silent. Set once, before any use.
struct LogSink {
    std::function<bool(const char* logger, int level)> enabled;
    std::function<void(const char* logger, int level, const std::string& msg)> write;
};
void set_log_sink(LogSink sink);
bool log_enabled(const char* logger, int level);
void log_write(const char* logger, int level, const std::string& msg);
std::string format(const char* fmt, ...);

// (session key, direction, 7-bit seq or SEQ_MOD + control index)
struct MaskId {
    int key = 0, direction = 0, seq = 0;
    bool operator==(const MaskId&) const = default;
};

inline MaskId ctl_mask(int direction, int i, int key = 0) { return {key, direction, SEQ_MOD + i}; }
inline const MaskId COMPACT_CONNECT = ctl_mask(0, 4);  // a compact CONNECT's mask (frames pack_connect); ctl_mask's i is 0-3
inline constexpr int EPOCH_MOD = 64;  // abandon epochs in a data codeword's identity
// comp: deflated (T_COMP); epoch: the sender's abandon epoch (its slicing).
// Both folded into the direction byte, so a codeword decoded under the
// wrong compression or slicing assumption fails its CRC.
inline MaskId data_mask(int direction, std::int64_t seq, int key = 0, bool comp = false, int epoch = 0) {
    return {key, direction | (comp ? 2 : 0) | static_cast<int>(pmod(epoch, EPOCH_MOD)) << 2,
            static_cast<int>(pmod(seq, SEQ_MOD))};
}
// The absolute seq = s7 (mod SEQ_MOD) nearest anchor.
inline std::int64_t unwrap(std::int64_t s7, std::int64_t anchor) {
    std::int64_t d = pmod(s7 - anchor, SEQ_MOD);
    if (d >= SEQ_MOD / 2) d -= SEQ_MOD;
    return anchor + d;
}

struct Slot {
    MaskId mask_id;
    int rv = 0;
    Bytes payload;
};

struct TxBurst {
    std::string submode;
    std::vector<Slot> slots;
    std::int64_t burst_seq = 0;  // absolute count of this station's bursts
};
using TxBurstPtr = std::shared_ptr<const TxBurst>;

bool dup_ctl(const TxBurst& b);  // its control sent twice (ARQ_DUP)

// A soft-bit store key: data seq `seq` of direction `peer` (Python's
// (peer, seq)), or control codeword `index` of this burst, combined as a
// pair (Python's ("ctl", id(rx), index)).
struct SoftKey {
    bool ctl = false;
    int peer = 0;
    std::int64_t seq = 0;
    int index = 0;
    auto operator<=>(const SoftKey&) const = default;
};

class RxBurst {
public:
    virtual ~RxBurst() = default;
    virtual const std::string& submode() const = 0;
    virtual int n_cw() const = 0;
    // Payload if slot decodes with this mask at this RV (combined with soft
    // bits under key, which a failed decode adds to).
    virtual std::optional<Bytes> decode(int slot, const MaskId& mask, int rv, const SoftKey* key) = 0;
    virtual void forget(const SoftKey& key) = 0;
};

class Station;

struct Recommendation {
    int rec = 0, hint = 1;
    std::optional<int> reply;
};

// link.Policy plus what Session needs. The optional Python methods have
// defaults here; has_*() say whether the policy has them at all.
class Policy {
public:
    virtual ~Policy() = default;
    virtual std::pair<std::string, int> choose(Station& st, int escalation) = 0;
    virtual int payload_bytes(const std::string& submode) = 0;
    virtual int ctl_payload_bytes(const std::string& submode) { return payload_bytes(submode); }
    virtual int max_ctl(const std::string&) { return 4; }
    virtual std::optional<Recommendation> recommend(Station&) { return std::nullopt; }  // nullopt: none (0, 1, -)
    virtual bool want_dup() { return false; }
    virtual int rv_cycle(const std::string& submode) = 0;
    virtual bool has_outcome() { return false; }
    virtual void outcome(const std::string&, int /*decoded*/, int /*sent*/, bool /*usable*/) {}
    // for the log
    virtual std::string mode_name(int rec) { return std::to_string(rec); }
    virtual bool has_airtime() { return false; }
    virtual std::optional<double> snr_est() { return std::nullopt; }
    // session
    virtual double airtime(const std::string&, int /*n_cw*/, bool /*dup*/) { return 0.0; }
    virtual std::string connect_mode(int /*cap*/, int /*tries*/) { return {}; }
    // seconds past t_turn to wait for a reply to `burst` (nullopt: none, REPLY_START_S alone)
    virtual std::optional<double> reply_hold(Station&, const TxBurst&) { return std::nullopt; }
    // engine: the receiver's measurements of a peer burst
    virtual void observe(const Measured&, const std::string& /*submode*/, double /*now*/) {}
};

struct Codeword {
    std::int64_t seq = 0, start = 0, length = 0;
    std::string submode;
    Bytes payload;
    int heard = 0;
    bool comp = false;
    std::int64_t cstart = 0;
    std::int64_t first_bn = -1;
    bool comp_known = false;
};

struct ProtocolError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class TxSide {
public:
    Bytes buf;  // stream bytes from buf_off on
    std::int64_t buf_off = 0, stream_end = 0;
    std::map<std::int64_t, Codeword> cws;  // outstanding
    std::int64_t base = 0, next = 0;
    std::optional<std::pair<std::int64_t, std::set<std::int64_t>>> ack;
    std::int64_t acked = 0, acked_wire = 0, acked_plain = 0;
    Bytes hist;
    std::int64_t hist_off = 0;

    void write(ByteView data);
    bool pending() const;
    bool on_ack(std::int64_t cum, const std::set<std::int64_t>& received);
    Codeword& new_codeword(const std::string& submode, int pb, bool compress, std::int64_t bn);
    std::vector<std::int64_t> missing() const;
    std::int64_t abandon();
};

class RxSide {
public:
    std::int64_t cum = 0;
    std::map<std::int64_t, std::pair<Bytes, bool>> buf;  // held above cum: (payload, compressed)
    RecordReader reader;
    Bytes out;
    Bytes hist;
    std::int64_t wire = 0, plain = 0;
    std::set<std::int64_t> comp_seqs;

    bool accept(std::int64_t seq, Bytes payload, bool comp = false);
    std::vector<std::int64_t> abandon(std::int64_t a);
};

enum class LinkState { ACTIVE, FAILED };

class Station {
public:
    Station(int direction, std::shared_ptr<Policy> policy, bool master = false, int key = 0,
            std::optional<int> max_misses = LINK_LOST_MISSES, int cap = 2, bool chat = false);
    virtual ~Station() = default;

    int direction;
    std::shared_ptr<Policy> policy;
    bool master;
    int key;
    std::optional<int> max_misses;
    int cap;
    bool last_rx_data = false;
    std::optional<int> peer_recommend;
    int peer_size_hint = 1;
    bool chat;
    bool peer_chat = false;
    int peer_queued = 0;
    bool peer_wants_dup = false;
    std::optional<int> peer_reply_recommend;
    TxSide tx;
    std::map<std::string, std::int64_t> stats;
    RxSide rx;
    LinkState state = LinkState::ACTIVE;
    std::string fail_reason;
    std::int64_t bursts_sent = 0;
    TxBurstPtr last_sent;
    std::optional<int> peer_burst;
    bool reply_lost = false;
    int misses = 0;
    int reply_escalation = 0;
    // the escalation the last recovery took: the next drop starts there
    int esc_floor = 0;
    int clean_ = 0;     // clean turns since the floor last moved
    int sent_esc_ = 0;  // the escalation my last built burst went at
    int no_progress = 0;
    int resyncs = 0;
    bool resync_due = false;
    // link.COMPRESS (DATA2G_COMPRESS=0 turns it off at construction)
    bool compress;

    std::map<std::int64_t, std::pair<std::int64_t, std::set<std::int64_t>>> snapshots;
    std::map<std::int64_t, std::vector<std::int64_t>> sent_seqs;
    int acted_on_ = BURST_MOD - 1;
    std::int64_t latest = -1, confirmed = -1;
    std::optional<Bytes> abandon_tlv;
    std::set<std::int64_t> abandon_bursts;
    int abandon_epoch = 0, peer_epoch = 0;
    bool answered_ = false;
    bool stale_ = false;  // the last burst handled repeated one already answered: its ACK may be stale

    int peer() const { return 1 - direction; }
    void write(ByteView data) { tx.write(data); }
    Bytes read();
    // Virtual so a binding can see a test's override of the Python method.
    virtual TxBurstPtr build(bool fresh = true);
    TxBurstPtr on_timeout(bool allow_repeat = true);  // nullptr: link lost
    bool handle(RxBurst& rx);
    void answered() { answered_ = true; }
    std::optional<Bytes> ctl_pair(RxBurst& rx, int slot, int i);
    std::int64_t new_available() const { return tx.buf_off + static_cast<std::int64_t>(tx.buf.size()) - tx.stream_end; }

private:
    bool handle_inner(RxBurst& rx);
    // What is wrong with a CRC-valid control, checked before any state changes.
    static std::optional<std::string> check(const Core& core, const Ext& ext, int n_data);
    bool malformed(const std::string& why);  // log, -> false: dropped like a failed control
    bool comp_bit(std::int64_t seq) const;
    std::vector<std::pair<std::optional<std::int64_t>, int>> map_slots(const Core& core, const Ext& ext, int n_cw,
                                                                      std::int64_t acted);
    void forget_all(RxBurst& rx);
    void fail(const std::string& why);
    void watchdog(bool progress);
    std::string mode(std::optional<int> rec);
    std::string burst_desc(const std::string& submode, int n_cw, bool dup = false);
    std::string snr();
};

}  // namespace data2g::arq
