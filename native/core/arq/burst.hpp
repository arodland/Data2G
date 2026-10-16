// data2g/arq/burst.py: what a burst is to the PHY (docs/arq.md §2, §3): its
// slots, each a payload at an RV under a codeword identity (MaskId), and
// the received side's interface. Shared by the ARQ (link, session, engine)
// and KISS, which has no Station; phy.hpp turns a MaskId into the CRC mask
// value (mask_value) and a TxBurst into audio.
#pragma once

#include <compare>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "arq/frames.hpp"

namespace data2g::arq {

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

struct Slot {
    MaskId mask_id;
    int rv = 0;
    Bytes payload;
};

struct TxBurst {
    std::string submode;
    std::vector<Slot> slots;
    std::int64_t burst_seq = 0;  // absolute count of this station's bursts
    int cap = 0;                 // bandwidth cap code the TX filter is for (cpm::tx_filter; modes cap_hz)
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

// One decode() the caller is about to ask for.
struct DecodeRequest {
    int slot;
    MaskId mask;
    int rv;
    SoftKey key;
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
    // The decode()s the caller is about to make, in this order, offered up front so an implementation can run
    // their independent parts at once (a slow machine's idle cores). Optional, and the answers are the same.
    virtual void prepare(std::span<const DecodeRequest>) {}
};

}  // namespace data2g::arq
