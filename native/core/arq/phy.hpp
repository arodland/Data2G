// data2g/arq/phy.py: the ARQ layer on the real modem. A TxBurst -> audio;
// a received burst -> an RxBurst (masked CRCs, soft-bit combining across
// resends under a SoftKey), decision-directed re-estimation (DD), and the
// gear shifter's measurements.
//
// Threads: a ModemRx is one burst's decoder state, used by one thread at a
// time; separate ModemRx share nothing mutable (the burst and its soft bits
// are const and shared, codes:: caches are thread-safe), so a whole burst
// can decode on a worker while the engine keeps searching. Within a burst,
// the split points are marked "split:" in phy.cpp.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "arq/link.hpp"
#include "arq/modes.hpp"
#include "arq/predictor.hpp"
#include "codes/codes.hpp"
#include "cpm/cpm.hpp"
#include "modem/modem.hpp"

namespace data2g::arq {

// DATA2G_DD (default 1; 0 turns it off), read once: when a slot fails, its
// decoder's a-posteriori LLRs and every decoded codeword of the burst
// become soft pilots (equalizer::refine) and it decodes again, up to
// DD_ITERS times.
bool dd_default();
inline constexpr int DD_ITERS = 2;
// A live receiver's DD starts no refines this long after the burst's decode began.
inline constexpr double DD_BUDGET_S = 1.0;

// A slot's CRC mask (zlib.crc32 of key, direction, seq; key 0: mask 0).
std::uint32_t mask_value(const MaskId& m);

// A TxBurst -> unit-RMS audio. Throws std::out_of_range for an unknown mode.
std::vector<double> tx_audio(const TxBurst& burst);

// A received burst: modem::receive's (OFDM) or cpm.receive's.
struct Heard {
    std::shared_ptr<const modem::Received> ofdm;
    std::shared_ptr<const cpm::Received> cpm;  // reads spec, dup, soft, E
    const Mode& mode() const;
    int n_cw() const;
};

// Per-slot soft bits in mapping order (phy.soft_bits): computed once per
// burst, shared by everyone who decodes it (KISS, then ARQ).
using SlotSoft = std::vector<std::vector<double>>;
std::shared_ptr<const SlotSoft> soft_bits(const Heard& r);

// The gear shifter's inputs (phy.measure).
Measured measure(const Heard& r);

// A codeword's stored soft bits: Python's (buffer, highest RV, submode, where).
struct SoftEntry {
    std::vector<double> buf;  // buffer_len, code order
    int top = 0;
    std::string submode;
    int slot = 0, rv = 0;  // where it was last stored from
    MaskId mask;
};

using SoftStore = std::map<SoftKey, SoftEntry>;
using Clock = std::function<double()>;  // seconds, monotonic
double monotonic();

class ModemRx : public RxBurst {
public:
    // store: this station's soft bits across bursts (nullptr: decode() with
    // a key keeps nothing). dd_budget: seconds from here after which DD
    // starts no more refines (nullopt: no limit). soft: soft_bits(r), if
    // already made.
    ModemRx(Heard r, SoftStore* store = nullptr, std::optional<double> dd_budget = std::nullopt,
            std::shared_ptr<const SlotSoft> soft = nullptr, bool dd = dd_default(), Clock clock = monotonic);

    const std::string& submode() const override { return submode_; }
    int n_cw() const override { return n_cw_; }
    std::optional<Bytes> decode(int slot, const MaskId& mask, int rv, const SoftKey* key) override;
    void forget(const SoftKey& key) override;

    // How the decodes so far went, for the log: channel re-estimates made (DD passes) and whether the DD
    // budget ran out.
    int dd_refines() const { return refines_; }
    bool dd_spent() const { return late(); }

    // decode() without a store: `stored` is what the key holds (nullopt:
    // nothing); a failed decode replaces it, a miss before decoding (slot
    // out of range, CPM control/data mismatch) leaves it. An entry stored under
    // another submode is dropped (logged) and the decode starts fresh.
    std::optional<Bytes> decode_stored(int slot, const MaskId& mask, int rv, std::optional<SoftEntry>& stored);
    std::optional<Bytes> decode_plain(int slot, const MaskId& mask);  // no key
    // The slot's mask-free decode as payload bytes, CRC unchecked: polar's
    // list best first, LDPC's one candidate if it converged. A broadcast
    // control reads its group from these (docs/broadcast.md §2).
    std::vector<Bytes> raw(int slot);

    const std::shared_ptr<const SlotSoft>& soft() const { return soft_; }

private:
    struct Raw {
        std::vector<std::uint8_t> cands, usable;
    };
    using Est = std::shared_ptr<const modem::DataEstimate>;  // nullptr: the burst's own
    using SoftCache = std::shared_ptr<std::map<int, std::vector<double>>>;
    struct DdState {
        Est est;
        SoftCache soft;
    };

    const codes::Spec& spec(int slot) const;
    const std::vector<double>& slot_soft(int slot);
    const Raw& decoded(int slot, const codes::Spec& s);
    bool dd(const codes::Spec& s) const;
    bool late() const;
    void learn(int slot, const codes::Spec& s, const Bytes& payload, int rv, std::uint32_t m);
    void refine(int slot, const codes::Spec& s, std::vector<double> post);
    void undo(int slot, const DdState& saved);
    Est dd_estimate() const;

    Heard r_;
    SoftStore* store_;
    std::shared_ptr<const SlotSoft> soft_;
    bool dd_;
    Clock clock_;
    std::optional<double> dd_until_;
    std::string submode_;
    int n_cw_, n_ctl_slots_;
    const codes::Spec* data_spec_;
    const codes::Spec* ctl_spec_;
    std::map<std::pair<int, std::uint32_t>, std::optional<Bytes>> memo_;
    std::map<int, Raw> raw_;
    std::map<int, std::vector<double>> post_;  // DD: slot -> LLRs of its coded bits
    bool blind_ = false;
    int refines_ = 0;
    // dd_estimate()'s per-symbol results, keyed on the symbol's input LLRs: a DD iteration changes
    // one codeword's posterior, and the rest of the burst's symbols come out the same again.
    struct SymCache {
        std::vector<double> L;               // m LLRs per symbol, as last computed
        std::vector<std::complex<double>> z;  // the soft pilot (0: none)
        std::vector<double> w;                // its weight
        std::vector<std::uint8_t> valid;
    };
    mutable SymCache sym_cache_;
    DdState cur_{nullptr, std::make_shared<std::map<int, std::vector<double>>>()};
};

}  // namespace data2g::arq
