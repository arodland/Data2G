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

// frame types (core word)
inline constexpr int ARQ = 0, SESSION = 1, PROBE = 2, ARQ_DUP = 3;
// extension types
inline constexpr int T_PAD = 0, T_NEW = 1, T_ABANDON = 2, T_RV = 3, T_BITMAP = 4, T_RESYNC = 5, T_REPORT = 6,
                     T_SURVEY = 7, T_SOUND = 8, T_BUFFER = 9, T_REPLY = 11, T_CHAT = 12, T_CQ = 13, T_DUPCTL = 14,
                     T_COMP = 15;
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
    int n_ctl = 1;       // 1-4 control codewords
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
Bytes pack_call(const std::string& call);
std::string unpack_call(ByteView b);

Bytes to_records(ByteView data);

class RecordReader {
public:
    Bytes buf;
    std::int64_t delivered = 0;  // host bytes out
    Bytes feed(ByteView b);
};

}  // namespace data2g::arq
