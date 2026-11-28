#include "arq/frames.hpp"

#include <cstddef>
#include <zlib.h>

#include <algorithm>
#include <iterator>
#include <stdexcept>

#include "tables/tables.hpp"

namespace data2g::arq {

std::uint8_t at(ByteView b, std::size_t i) {
    if (i >= b.size()) throw std::out_of_range("index out of range");
    return b[i];
}

// --- core word ----------------------------------------------------------------

Bytes Core::pack() const {
    if (!(1 <= n_ctl && n_ctl <= MAX_CTL && 0 <= k && k < 64) || ftype < 0 || ftype > 3)
        throw std::logic_error("Core.pack: field out of range");
    std::uint32_t v = static_cast<std::uint32_t>(ftype);
    const std::pair<int, int> fields[] = {{n_ctl - 1, 2}, {burst_seq, 3}, {acted_on, 3}, {cum, 7},
                                          {reply_lost ? 1 : 0, 1}, {k, 6}, {recommend, 6}, {size_hint, 2}};
    for (auto [val, bits] : fields) {
        if (val < 0 || val >= (1 << bits)) throw std::logic_error("Core.pack: field out of range");
        v = (v << bits) | static_cast<std::uint32_t>(val);
    }
    return {static_cast<std::uint8_t>(v >> 24), static_cast<std::uint8_t>(v >> 16), static_cast<std::uint8_t>(v >> 8),
            static_cast<std::uint8_t>(v)};
}

Core Core::unpack(ByteView b) {
    std::uint32_t v = 0;
    for (std::size_t i = 0; i < std::min<std::size_t>(4, b.size()); ++i) v = v << 8 | b[i];
    int f[8];
    const int widths[] = {2, 6, 6, 1, 7, 3, 3, 2};  // from the least significant end
    for (int i = 0; i < 8; ++i) {
        f[i] = static_cast<int>(v & ((1u << widths[i]) - 1));
        v >>= widths[i];
    }
    Core c;
    c.ftype = static_cast<int>(v);
    c.n_ctl = f[7] + 1;
    c.burst_seq = f[6];
    c.acted_on = f[5];
    c.cum = f[4];
    c.reply_lost = f[3] != 0;
    c.k = f[2];
    c.recommend = f[1];
    c.size_hint = f[0];
    return c;
}

std::vector<Bytes> Control::pack(int payload_bytes) {
    Bytes body;
    for (const auto& [t, v] : ext) {
        if (t < 0 || t > 255 || v.size() > 255) throw std::invalid_argument("bytes must be in range(0, 256)");
        body.push_back(static_cast<std::uint8_t>(t));
        body.push_back(static_cast<std::uint8_t>(v.size()));
        body.insert(body.end(), v.begin(), v.end());
    }
    const auto total = static_cast<std::int64_t>(4 + body.size());
    const auto n = std::max<std::int64_t>(1, ceil_div(total, payload_bytes));
    if (n > MAX_CTL)
        throw std::invalid_argument("control needs " + std::to_string(n) + " codewords of " +
                                    std::to_string(payload_bytes) + " B (max " + std::to_string(MAX_CTL) + ")");
    core.n_ctl = static_cast<int>(n);
    Bytes stream = core.pack();
    stream.insert(stream.end(), body.begin(), body.end());
    stream.resize(static_cast<std::size_t>(n * payload_bytes), 0);
    std::vector<Bytes> out;
    for (std::int64_t i = 0; i < n; ++i)
        out.emplace_back(stream.begin() + i * payload_bytes, stream.begin() + (i + 1) * payload_bytes);
    return out;
}

Control Control::unpack(const std::vector<Bytes>& payloads) {
    Bytes stream;
    for (const auto& p : payloads) stream.insert(stream.end(), p.begin(), p.end());
    Control c{Core::unpack(stream), {}};
    std::size_t i = 4;
    while (i + 2 <= stream.size()) {
        const int t = stream[i], n = stream[i + 1];
        if (t == T_PAD) break;
        if (i + 2 + n > stream.size()) throw std::invalid_argument("truncated extension");
        if (t >= TLV_CRITICAL) throw std::invalid_argument("unknown critical extension " + std::to_string(t));
        c.ext[t] = Bytes(stream.begin() + i + 2, stream.begin() + i + 2 + n);
        i += 2 + n;
    }
    return c;
}

// --- extension payloads -----------------------------------------------------------

Bytes pack_bitmap(const std::set<int>& received, int cum) {
    std::uint64_t v = 0;
    for (int i = 0; i < BITMAP_BITS; ++i)
        if (received.count(static_cast<int>(pmod(cum + 1 + i, SEQ_MOD)))) v |= std::uint64_t{1} << (BITMAP_BITS - 1 - i);
    Bytes out;
    for (int i = 0; i < BITMAP_BITS / 8; ++i) out.push_back(static_cast<std::uint8_t>(v >> (56 - 8 * i)));
    while (!out.empty() && out.back() == 0) out.pop_back();
    return out;
}

std::set<int> unpack_bitmap(ByteView b, int cum) {
    // ljust to 8 bytes; a longer value keeps its low 64 bits' meaning as
    // Python's int would: bit i counted from the top of the whole value
    const std::size_t n = std::max<std::size_t>(b.size(), BITMAP_BITS / 8);
    std::set<int> out;
    for (int i = 0; i < BITMAP_BITS; ++i) {
        // Python: v >> (63 - i) & 1 over the whole (ljust) value
        const std::size_t bit = (n * 8 - 1) - static_cast<std::size_t>(BITMAP_BITS - 1 - i);  // from the top
        const std::size_t byte = bit / 8;
        const int x = byte < b.size() ? b[byte] : 0;
        if (x >> (7 - bit % 8) & 1) out.insert(static_cast<int>(pmod(cum + 1 + i, SEQ_MOD)));
    }
    return out;
}

Bytes pack_rv(const std::vector<int>& rvs) {
    if (rvs.empty()) return {};
    Bytes out(static_cast<std::size_t>(ceil_div(2 * static_cast<std::int64_t>(rvs.size()), 8)), 0);
    for (std::size_t i = 0; i < rvs.size(); ++i) out[i / 4] |= static_cast<std::uint8_t>((rvs[i] & 3) << (6 - 2 * (i % 4)));
    return out;
}

std::vector<int> unpack_rv(ByteView b, int k) {
    // Python shifts the whole value right by 8 len(b) - 2 (i + 1): negative
    // (ValueError) once the bits run out
    std::vector<int> out;
    for (int i = 0; i < k; ++i) {
        if (2 * static_cast<std::size_t>(i + 1) > 8 * b.size()) throw std::invalid_argument("negative shift count");
        out.push_back(b[i / 4] >> (6 - 2 * (i % 4)) & 3);
    }
    return out;
}

Bytes pack_flags(const std::vector<bool>& flags) {
    Bytes out(static_cast<std::size_t>(ceil_div(static_cast<std::int64_t>(flags.size()), 8)), 0);
    for (std::size_t i = 0; i < flags.size(); ++i)
        if (flags[i]) out[i / 8] |= static_cast<std::uint8_t>(0x80 >> i % 8);
    while (!out.empty() && out.back() == 0) out.pop_back();
    return out;
}

std::vector<bool> unpack_flags(ByteView b, int n) {
    std::vector<bool> out;
    for (int i = 0; i < n; ++i) out.push_back(static_cast<std::size_t>(i) < 8 * b.size() && (b[i / 8] >> (7 - i % 8) & 1));
    return out;
}

// --- compression ------------------------------------------------------------------

namespace {

Bytes primed(ByteView hist) {
    Bytes d(tables::ZDICT.begin(), tables::ZDICT.end());
    d.insert(d.end(), hist.begin(), hist.end());
    return d;
}

std::string zerror(int err, const char* msg, const char* what) {
    std::string out = "Error " + std::to_string(err) + " " + what;
    if (msg) out += std::string(": ") + msg;
    return out;
}

}  // namespace

Bytes deflate(ByteView hist, ByteView data) {
    z_stream s{};
    if (deflateInit2(&s, 9, Z_DEFLATED, -15, 9, Z_DEFAULT_STRATEGY) != Z_OK) throw std::runtime_error("deflateInit2");
    const Bytes dict = primed(hist);
    deflateSetDictionary(&s, dict.data(), static_cast<uInt>(dict.size()));
    Bytes out(deflateBound(&s, static_cast<uLong>(data.size())) + 64);
    s.next_in = const_cast<Bytef*>(data.data());
    s.avail_in = static_cast<uInt>(data.size());
    int r;
    do {
        if (s.total_out == out.size()) out.resize(out.size() * 2);
        s.next_out = out.data() + s.total_out;
        s.avail_out = static_cast<uInt>(out.size() - s.total_out);
        r = ::deflate(&s, Z_FINISH);
    } while (r == Z_OK || r == Z_BUF_ERROR);
    out.resize(s.total_out);
    deflateEnd(&s);
    if (r != Z_STREAM_END) throw std::runtime_error("deflate failed");
    return out;
}

std::optional<std::pair<std::int64_t, Bytes>> deflate_fit(ByteView hist, ByteView data, int pb) {
    std::int64_t lo = pb + 1, hi = std::min<std::int64_t>(static_cast<std::int64_t>(data.size()), 16 * pb);
    if (lo > hi) return std::nullopt;
    Bytes z = deflate(hist, data.first(static_cast<std::size_t>(lo)));
    if (static_cast<std::int64_t>(z.size()) > pb) return std::nullopt;  // incompressible: one trial
    std::pair<std::int64_t, Bytes> best{lo, std::move(z)};
    lo += 1;
    while (lo <= hi) {
        const std::int64_t m = (lo + hi) / 2;
        Bytes zm = deflate(hist, data.first(static_cast<std::size_t>(m)));
        if (static_cast<std::int64_t>(zm.size()) <= pb) {
            best = {m, std::move(zm)};
            lo = m + 1;
        } else {
            hi = m - 1;
        }
    }
    return best;
}

Bytes inflate(ByteView hist, ByteView payload) {
    z_stream s{};
    if (inflateInit2(&s, -15) != Z_OK) throw std::runtime_error("inflateInit2");
    const Bytes dict = primed(hist);
    inflateSetDictionary(&s, dict.data(), static_cast<uInt>(dict.size()));
    Bytes out(MAX_INFLATE);
    s.next_in = const_cast<Bytef*>(payload.data());
    s.avail_in = static_cast<uInt>(payload.size());
    s.next_out = out.data();
    s.avail_out = MAX_INFLATE;
    const int r = ::inflate(&s, Z_SYNC_FLUSH);  // as Python's decompress(payload, MAX_INFLATE)
    std::string err;
    if (r != Z_OK && r != Z_BUF_ERROR && r != Z_STREAM_END) err = zerror(r, s.msg, "while decompressing data");
    out.resize(s.total_out);
    inflateEnd(&s);
    if (!err.empty()) throw std::invalid_argument("inflate: " + err);
    if (r != Z_STREAM_END) throw std::invalid_argument("inflate: truncated or over MAX_INFLATE");
    return out;
}

// --- callsigns ----------------------------------------------------------------------

namespace {
constexpr std::string_view CALL_ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-";
}

Bytes pack_call(const std::string& call_in) {
    std::string call = call_in;
    for (auto& c : call)
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    bool ok = call.size() <= CALL_CHARS;
    // space pads at the end, so a name may hold one ("VARA KISS"); trailing ones don't survive the round trip
    for (char c : call) ok = ok && CALL_ALPHABET.find(c) != std::string_view::npos;
    if (!ok) throw std::invalid_argument("callsign '" + call + "': up to 10 of A-Z 0-9 / - space");
    std::uint64_t v = 0;
    for (int i = 0; i < CALL_CHARS; ++i)
        v = v << 6 | (static_cast<std::size_t>(i) < call.size() ? CALL_ALPHABET.find(call[i]) : 0);
    Bytes out(CALL_CHARS * 6 / 8 + 1);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<std::uint8_t>(v >> (8 * (out.size() - 1 - i)));
    return out;
}

std::string unpack_call(ByteView b) {
    std::uint64_t v = 0;  // the low 64 bits: all the characters use
    for (auto x : b) v = v << 8 | x;
    std::string out;
    std::uint64_t worst = 0;
    for (int i = 0; i < CALL_CHARS; ++i) {
        const auto c = (v >> (6 * (CALL_CHARS - 1 - i))) & 63;
        worst = std::max(worst, c);
        if (c < CALL_ALPHABET.size()) out += CALL_ALPHABET[c];
    }
    if (worst >= CALL_ALPHABET.size()) throw std::invalid_argument("callsign code " + std::to_string(worst));
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

std::uint64_t pack_call_b40(const std::string& call_in) {
    std::string call = call_in;
    for (auto& c : call)
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    bool ok = call.size() <= CALL_CHARS;
    for (char c : call) ok = ok && CALL_ALPHABET.find(c) != std::string_view::npos;
    if (!ok) throw std::invalid_argument("callsign '" + call + "': up to 10 of A-Z 0-9 / - space");
    std::uint64_t v = 0;
    for (int i = 0; i < CALL_CHARS; ++i)
        v = v * CALL_BASE + (static_cast<std::size_t>(i) < call.size() ? CALL_ALPHABET.find(call[i]) : 0);
    return v;
}

std::string unpack_call_b40(std::uint64_t v) {
    if (v >= CALL_B40_MAX) throw std::invalid_argument("callsign code " + std::to_string(v));
    std::string out(CALL_CHARS, ' ');
    for (int i = CALL_CHARS - 1; i >= 0; --i) {
        const auto d = v % CALL_BASE;
        v /= CALL_BASE;
        if (d >= CALL_ALPHABET.size()) throw std::invalid_argument("callsign code " + std::to_string(d));
        out[static_cast<std::size_t>(i)] = CALL_ALPHABET[d];
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

// --- session frame bodies -------------------------------------------------------------

namespace {

std::uint8_t u8(std::int64_t v) {  // bytes([v])
    if (v < 0 || v > 255) throw std::invalid_argument("bytes must be in range(0, 256)");
    return static_cast<std::uint8_t>(v);
}

void put_u16(Bytes& out, std::int64_t v) {  // v.to_bytes(2, "big")
    if (v < 0 || v > 65535) throw std::invalid_argument("u16 out of range");
    out.push_back(static_cast<std::uint8_t>(v >> 8));
    out.push_back(static_cast<std::uint8_t>(v));
}

std::int64_t u16_at(ByteView b, std::size_t i) { return b[i] << 8 | b[i + 1]; }

void write_bits(Bytes& out, int& pos, std::uint64_t val, int bits) {
    if (bits < 64 && (val >> bits))
        throw std::invalid_argument("CONNECT field " + std::to_string(val) + " over " + std::to_string(bits) + " bits");
    for (int b = bits - 1; b >= 0; --b, ++pos)
        if ((val >> b) & 1) out[static_cast<std::size_t>(pos / 8)] |= static_cast<std::uint8_t>(0x80 >> (pos % 8));
}

std::uint64_t read_bits(ByteView p, int& pos, int bits) {
    std::uint64_t x = 0;
    for (int b = 0; b < bits; ++b, ++pos)
        x = (x << 1) | ((p[static_cast<std::size_t>(pos / 8)] >> (7 - pos % 8)) & 1);
    return x;
}

constexpr std::pair<unsigned, int> CONNECT_BIT_LAYOUT[] = {
    {Connect::F_VERSION, 4}, {Connect::F_CALLER, 54}, {Connect::F_CALLEE, 54}, {Connect::F_NONCE, 16},
    {Connect::F_CAP, 2},     {Connect::F_TTURN, 6},   {Connect::F_CAPS, 16},   {Connect::F_REQ, 8},
};

}  // namespace

Bytes Connect::pack() const {
    if (version < 0 || nonce < 0 || cap < 0 || t_turn10 < 0 || caps < 0 || req < 0)
        throw std::invalid_argument("CONNECT negative field");
    Bytes out(COMPACT_BYTES, 0);
    int pos = 0;
    write_bits(out, pos, static_cast<std::uint64_t>(version), 4);
    write_bits(out, pos, pack_call_b40(caller), 54);
    write_bits(out, pos, pack_call_b40(callee), 54);
    write_bits(out, pos, static_cast<std::uint64_t>(nonce), 16);
    write_bits(out, pos, static_cast<std::uint64_t>(cap), 2);
    write_bits(out, pos, static_cast<std::uint64_t>(t_turn10), 6);
    write_bits(out, pos, static_cast<std::uint64_t>(caps), 16);
    write_bits(out, pos, static_cast<std::uint64_t>(req), 8);
    return out;
}

std::string ext_name(int t) {
    switch (t) {
        case T_PAD: return "pad";
        case T_NEW: return "new";
        case T_ABANDON: return "abandon";
        case T_RV: return "rv";
        case T_BITMAP: return "bitmap";
        case T_REPORT: return "report";
        case T_SURVEY: return "survey";
        case T_SOUND: return "sound";
        case T_BUFFER: return "buffer";
        case T_SESS: return "sess";
        case T_REPLY: return "reply";
        case T_CHAT: return "chat";
        case T_CQ: return "cq";
        case T_DUPCTL: return "dupctl";
        case T_COMP: return "comp";
        case T_ID: return "id";
        case T_DRAINED: return "drained";
        case T_INLINE: return "inline";
        case T_PING: return "ping";
        case T_REC_EXT: return "rec_ext";
        case T_CAPS: return "caps";
        default: return "ext " + std::to_string(t);
    }
}

Connect Connect::unpack(ByteView p) {
    if (p.size() < BYTES) throw std::invalid_argument("CONNECT of " + std::to_string(p.size()) + " B");
    int pos = 0;
    const int ver = static_cast<int>(read_bits(p, pos, 4));
    const std::uint64_t caller_i = read_bits(p, pos, 54);
    const std::uint64_t callee_i = read_bits(p, pos, 54);
    const std::int64_t nonce_v = static_cast<std::int64_t>(read_bits(p, pos, 16));
    const int cap_v = static_cast<int>(read_bits(p, pos, 2));
    const int t10_v = static_cast<int>(read_bits(p, pos, 6));
    const int caps_v = static_cast<int>(read_bits(p, pos, 16));
    const int req_v = static_cast<int>(read_bits(p, pos, 8));
    return {ver, unpack_call_b40(caller_i), unpack_call_b40(callee_i), nonce_v, cap_v, t10_v, caps_v, req_v};
}

std::pair<std::string, std::string> Connect::calls(ByteView p) {
    const Connect c = unpack(p);
    return {c.caller, c.callee};
}

Bytes Connect::mask(unsigned field_bits) {
    Bytes out(COMPACT_BYTES, 0);
    int pos = 0;
    for (const auto& [flag, bits] : CONNECT_BIT_LAYOUT) {
        const std::uint64_t v = (field_bits & flag) ? ((std::uint64_t{1} << bits) - 1) : 0;
        write_bits(out, pos, v, bits);
    }
    return out;
}

Bytes ConnectAck::pack() const {
    Bytes out{CONNECT_ACK};
    put_u16(out, nonce);
    out.push_back(u8(cap));
    out.push_back(u8(t_turn10));
    put_u16(out, caps);
    out.push_back(u8(req));
    return out;
}

ConnectAck ConnectAck::unpack(ByteView body) {
    if (body.size() < BYTES) throw std::invalid_argument("CONNECT_ACK of " + std::to_string(body.size()) + " B");
    return {u16_at(body, 1), body[3], body[4], static_cast<int>(u16_at(body, 5)), body[7]};
}

Bytes ConnectNak::pack() const {
    Bytes out{CONNECT_NAK};
    put_u16(out, nonce);
    out.push_back(u8(reason));
    out.push_back(u8(detail));
    return out;
}

ConnectNak ConnectNak::unpack(ByteView body) {
    if (body.size() < BYTES) throw std::invalid_argument("CONNECT_NAK of " + std::to_string(body.size()) + " B");
    return {u16_at(body, 1), body[3], body[4]};
}

Bytes pack_cq(const std::string& call, int cap) {
    Bytes out = pack_call(call);
    out.push_back(static_cast<std::uint8_t>(cap));
    return out;
}

std::pair<std::string, int> unpack_cq(ByteView body) {
    if (body.size() < 9) throw std::invalid_argument("CQ of " + std::to_string(body.size()) + " B");
    return {unpack_call(body.first(8)), body[8]};
}

Bytes pack_id(const std::string& call, int key) {
    Bytes out = pack_call(call);
    out.push_back(static_cast<std::uint8_t>(key >> 8));
    out.push_back(static_cast<std::uint8_t>(key & 255));
    return out;
}

std::pair<std::string, int> unpack_id(ByteView body) {
    if (body.size() < 10) throw std::invalid_argument("ID of " + std::to_string(body.size()) + " B");
    return {unpack_call(body.first(8)), body[8] << 8 | body[9]};
}

// --- stream records -----------------------------------------------------------------

Bytes to_records(ByteView data) {
    Bytes out;
    for (std::size_t i = 0; i < data.size(); i += 255) {
        const auto n = std::min<std::size_t>(255, data.size() - i);
        out.push_back(static_cast<std::uint8_t>(n));
        out.insert(out.end(), data.begin() + i, data.begin() + i + n);
    }
    return out;
}

Bytes RecordReader::feed(ByteView b) {
    buf.insert(buf.end(), b.begin(), b.end());
    Bytes out;
    std::size_t i = 0;
    while (i < buf.size()) {
        const std::size_t n = buf[i];
        if (n == 0) {
            ++i;
            continue;
        }
        if (i + 1 + n > buf.size()) break;
        delivered += static_cast<std::int64_t>(n);
        out.insert(out.end(), buf.begin() + i + 1, buf.begin() + i + 1 + n);
        i += 1 + n;
    }
    buf.erase(buf.begin(), buf.begin() + i);
    return out;
}

}  // namespace data2g::arq
