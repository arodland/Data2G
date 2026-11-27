// data2g/arq/frames.py: the 32-bit core control word, extension TLVs,
// bitmaps, callsigns, stream records and compression (docs/arq.md §4, §5,
// §7, §9a).
//
// Malformed input fails as Python does: what raises ValueError there throws
// std::invalid_argument here, an IndexError std::out_of_range.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace data2g::arq {

using Bytes = std::vector<std::uint8_t>;
using ByteView = std::span<const std::uint8_t>;

inline constexpr int SEQ_BITS = 7;
inline constexpr int SEQ_MOD = 1 << SEQ_BITS;
inline constexpr int BITMAP_BITS = 64;
inline constexpr int WINDOW = SEQ_MOD / 2 - 1;
inline constexpr int BURST_MOD = 8;
inline constexpr int MAX_CTL = 4;  // control codewords a burst may have: the core's 2-bit n_ctl

// frame types (core word)
inline constexpr int ARQ = 0, SESSION = 1, PROBE = 2, ARQ_DUP = 3;
// extension types (slots 5, 19, 20 retired at wire 1.0: T_RESYNC, T_NODS, T_KEEP).
// Forward-compatibility split: 1..127 ignorable if unknown; 128..255 must-understand.
inline constexpr int T_PAD = 0, T_NEW = 1, T_ABANDON = 2, T_RV = 3, T_BITMAP = 4, T_REPORT = 6,
                     T_SURVEY = 7, T_SOUND = 8, T_BUFFER = 9,
                     T_SESS = 10,  // a session frame's body (arq/session.hpp)
                     T_REPLY = 11, T_CHAT = 12, T_CQ = 13, T_DUPCTL = 14,
                     T_COMP = 15,
                     T_ID = 16,  // packed callsign + session key: an ID frame (docs/arq.md §7a), mask 0
                     T_DRAINED = 17,  // empty: this burst's new codewords end the sender's queued data
                     // a data codeword in the control's padding (docs/arq.md §5a): seq | comp << 7, its
                     // payload with trailing zeros cut, then 2-byte inline_check
                     T_INLINE = 18,
                     T_PING = 21,     // FM only (arq/ping): a ping or its reply, the TX path and level check
                     T_REC_EXT = 22,  // extended mode recommendation (1-2 B) when Core.recommend == REC_EXT (63)
                     T_CAPS = 23;     // extended capability/constraint TLV reserved for CAP_EXT
inline constexpr int TLV_CRITICAL = 128;  // 1..127 ignorable if unknown; 128..255 must-understand

// Capability / constraint bitmasks exchanged in CONNECT and CONNECT_ACK (wire 1.0).
inline constexpr int CAP_NODS = 1 << 0;  // this station listens for nods (docs/arq.md §3b; cleared by --no-nods)
inline constexpr int CAP_EXT = 1 << 15;  // escape hatch: extended capability/constraint TLV (T_CAPS) follows
inline constexpr int DEFAULT_CAPS = CAP_NODS;
inline constexpr int SUPPORTED_REQ = 0;  // mandatory peer-constraint bits this version knows how to honor
// An extension type's short name ("ext N" for one not listed): monitor/'s
// dumps. A new T_* gets its name here too. (Native only.)
std::string ext_name(int t);
inline constexpr int CHAT_LINE_BYTES = 200;
inline constexpr int HIST = 4096;
inline constexpr int MAX_INFLATE = 1 << 16;
// session control subtypes
inline constexpr int CONNECT = 1, CONNECT_ACK = 2, CONNECT_NAK = 3, DISC = 4, DISC_ACK = 5;

// Python's % and //: the result takes the divisor's sign.
inline std::int64_t pmod(std::int64_t a, std::int64_t m) {
    const std::int64_t r = a % m;
    return r < 0 ? r + m : r;
}
// -(-a // b) for a >= 0, b > 0
inline std::int64_t ceil_div(std::int64_t a, std::int64_t b) { return (a + b - 1) / b; }

// b[i], or std::out_of_range as Python's IndexError.
std::uint8_t at(ByteView b, std::size_t i);

struct Core {
    int ftype = ARQ;     // 2 bits
    int n_ctl = 1;       // 1-MAX_CTL control codewords
    int burst_seq = 0;   // 3 bits
    int acted_on = 0;    // 3 bits
    int cum = 0;         // 7 bits
    bool reply_lost = false;
    int k = 0;           // 6 bits
    int recommend = 0;   // 6 bits
    int size_hint = 1;   // 2 bits

    Bytes pack() const;
    static Core unpack(ByteView b);  // the first 4 bytes (fewer: as many as there are)
    bool operator==(const Core&) const = default;
};

using Ext = std::map<int, Bytes>;  // type -> value, packed in type order

struct Control {
    Core core;
    Ext ext;

    // -> control codeword payloads; sets core.n_ctl to fit.
    std::vector<Bytes> pack(int payload_bytes);
    static Control unpack(const std::vector<Bytes>& payloads);
};

Bytes pack_bitmap(const std::set<int>& received, int cum);
std::set<int> unpack_bitmap(ByteView b, int cum);
Bytes pack_rv(const std::vector<int>& rvs);
std::vector<int> unpack_rv(ByteView b, int k);
Bytes pack_flags(const std::vector<bool>& flags);
std::vector<bool> unpack_flags(ByteView b, int n);

// Raw deflate (level 9, window 15, memLevel 9) primed with ZDICT + hist.
Bytes deflate(ByteView hist, ByteView data);
// The longest prefix of data past pb bytes whose deflate fits pb bytes.
std::optional<std::pair<std::int64_t, Bytes>> deflate_fit(ByteView hist, ByteView data, int pb);
// A compressed codeword (zero padded) -> its stream bytes.
Bytes inflate(ByteView hist, ByteView payload);

inline constexpr int CALL_CHARS = 10;
inline constexpr std::uint64_t CALL_BASE = 40;
inline constexpr std::uint64_t CALL_B40_MAX = 10485760000000000ULL;  // 40^10 < 2^54
Bytes pack_call(const std::string& call);
std::string unpack_call(ByteView b);
std::uint64_t pack_call_b40(const std::string& call);
std::string unpack_call_b40(std::uint64_t v);

// --- session frame bodies (docs/arq.md §7, §7a) ---
// CONNECT uses a single unified 20-byte (160-bit) bit-packed payload on all
// modes under burst COMPACT_CONNECT: no Core or T_SESS wrapper.
// CONNECT_ACK, CONNECT_NAK, DISC and DISC_ACK are T_SESS bodies led by their
// subtype byte; DISC and DISC_ACK are the subtype alone. T_CQ and T_ID bodies
// have no subtype.

inline constexpr int COMPACT_BYTES = 20;

// Unified 20-byte (160-bit) CONNECT payload:
// version (4 b) | caller (54 b base-40) | callee (54 b base-40) | nonce (16 b) |
// cap (2 b) | t_turn10 (6 b) | caps (16 b) | req (8 b)
struct Connect {
    static constexpr std::size_t BYTES = COMPACT_BYTES;
    static constexpr unsigned F_VERSION = 1u << 0, F_CALLER = 1u << 1, F_CALLEE = 1u << 2, F_NONCE = 1u << 3,
                              F_CAP = 1u << 4, F_TTURN = 1u << 5, F_CAPS = 1u << 6, F_REQ = 1u << 7;
    int version = 0;
    std::string caller, callee;
    std::int64_t nonce = 0;
    int cap = 0, t_turn10 = 0;
    int caps = 0, req = 0;

    Bytes pack() const;
    static Connect unpack(ByteView p);  // BYTES (20 B) at least
    static std::pair<std::string, std::string> calls(ByteView p);
    static Bytes mask(unsigned field_bits);  // 20-byte bitmask for primed reception
};

// subtype, CONNECT's nonce (2 B), agreed cap code, callee's t_turn in tenths, caps (2 B), req (1 B)
struct ConnectAck {
    static constexpr std::size_t BYTES = 8;
    std::int64_t nonce = 0;
    int cap = 0, t_turn10 = 0;
    int caps = 0, req = 0;

    Bytes pack() const;
    static ConnectAck unpack(ByteView body);  // BYTES at least
};

// subtype, CONNECT's nonce (2 B), reason (NAK_*), detail (1 B)
struct ConnectNak {
    static constexpr std::size_t BYTES = 5;
    std::int64_t nonce = 0;
    int reason = 0;
    int detail = 0;

    Bytes pack() const;
    static ConnectNak unpack(ByteView body);  // BYTES at least
};

Bytes pack_cq(const std::string& call, int cap);  // a T_CQ body: the call and its bandwidth cap code
std::pair<std::string, int> unpack_cq(ByteView body);  // invalid_argument: "CQ of n B"
Bytes pack_id(const std::string& call, int key);  // a T_ID body: the call and the session key it identifies for
std::pair<std::string, int> unpack_id(ByteView body);  // invalid_argument: "ID of n B"

Bytes to_records(ByteView data);

class RecordReader {
public:
    Bytes buf;
    std::int64_t delivered = 0;  // host bytes out
    Bytes feed(ByteView b);
};

}  // namespace data2g::arq
