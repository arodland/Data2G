#include "codes/codes.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "tables/tables.hpp"

namespace data2g::codes {

namespace {

// MSB-first CRC table, as codes._crc24_table builds it.
template <std::uint32_t Poly, int Bits>
constexpr std::array<std::uint32_t, 256> msb_table() {
    constexpr std::uint32_t top = 1u << (Bits - 1), mask = (1u << Bits) - 1;
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
        std::uint32_t c = i << (Bits - 8);
        for (int j = 0; j < 8; ++j) c = (c & top) ? ((c << 1) ^ Poly) & mask : (c << 1) & mask;
        t[i] = c;
    }
    return t;
}

constexpr auto CRC16 = msb_table<0x1021, 16>();
constexpr auto CRC24 = msb_table<0xB2B117, 24>();

constexpr std::array<std::uint32_t, 256> crc32_table() {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
        std::uint32_t c = i;
        for (int j = 0; j < 8; ++j) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        t[i] = c;
    }
    return t;
}
constexpr auto CRC32 = crc32_table();

std::size_t row(const config::Submode& spec) {
    return static_cast<std::size_t>(&spec - config::SUBMODES.data());
}

}  // namespace

std::uint16_t crc16(std::span<const std::uint8_t> data) {
    std::uint32_t crc = 0xFFFF;
    for (auto b : data) crc = ((crc << 8) & 0xFFFF) ^ CRC16[((crc >> 8) ^ b) & 0xFF];
    return static_cast<std::uint16_t>(crc);
}

std::uint32_t crc24(std::span<const std::uint8_t> data, std::uint32_t crc) {
    for (auto b : data) crc = ((crc << 8) & 0xFFFFFF) ^ CRC24[((crc >> 16) ^ b) & 0xFF];
    return crc;
}

std::uint32_t crc32(std::span<const std::uint8_t> data) {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (auto b : data) crc = CRC32[(crc ^ b) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

std::vector<std::uint8_t> with_crc(std::span<const std::uint8_t> payload, int n_crc, std::uint32_t mask) {
    std::uint32_t c;
    if (n_crc == 16) c = (crc16(payload) ^ mask) & 0xFFFF;
    else if (n_crc == 24) c = (crc24(payload) ^ mask) & 0xFFFFFF;
    else if (n_crc == 32) c = crc32(payload) ^ mask;
    else throw std::invalid_argument("n_crc must be 16, 24 or 32");
    std::vector<std::uint8_t> out(payload.begin(), payload.end());
    for (int shift = n_crc - 8; shift >= 0; shift -= 8) out.push_back(static_cast<std::uint8_t>(c >> shift));
    return out;
}

int scramble_seed(int index) {
    if (index == PLAIN) return 0;
    if (index < 0) throw std::invalid_argument("burst position must be >= 0 or PLAIN");
    return 1 + static_cast<int>((static_cast<std::uint64_t>(index) * 0x9E3779B1u) % 511);
}

std::vector<std::uint8_t> scrambler(int k, int seed) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(k));
    unsigned reg = static_cast<unsigned>(seed);
    for (auto& o : out) {
        const unsigned bit = ((reg >> 8) ^ (reg >> 4)) & 1;
        o = static_cast<std::uint8_t>(bit);
        reg = ((reg << 1) | bit) & 0x1FF;
    }
    return out;
}

const config::Submode* submode(std::string_view name) {
    for (const auto& s : config::SUBMODES)
        if (s.name == name) return &s;
    return nullptr;
}

std::span<const std::uint16_t> interleaver(const config::Submode& spec) {
    return tables::FORMATS.at(row(spec)).perm;
}

std::span<const std::uint16_t> info_pos(const config::Submode& spec) {
    return tables::FORMATS.at(row(spec)).info_pos;
}

// --- the codec ---------------------------------------------------------------

namespace {

std::vector<Spec> build_specs() {
    std::vector<Spec> v;
    for (std::size_t i = 0; i < config::SUBMODES.size(); ++i) {
        const auto& s = config::SUBMODES[i];
        v.push_back({s.name, s.code == "polar", s.k, s.coded_bits, s.bits_per_cu, s.crc_bits, s.payload_bytes,
                     tables::FORMATS[i].perm, tables::FORMATS[i].info_pos});
    }
    for (auto table : {tables::CPM_SPECS, tables::CPM_CTL})
        for (const auto& s : table) {
            const bool polar = s.code == "polar";
            const int crc = polar ? 24 : s.k >= 512 ? 32 : 16;  // codes.crc_bits
            int bits = 0;
            for (const auto& g : tables::CPM_GRIDS)
                if (g.name == s.grid) bits = g.bits;
            v.push_back({s.name, polar, s.k, s.coded_bits, bits, crc, (s.k - crc) / 8, s.perm,
                         polar ? polar::ga_info_pos(s.k, s.coded_bits) : std::span<const std::uint16_t>{}});
        }
    return v;
}

const std::vector<Spec>& specs() {
    static const std::vector<Spec> v = build_specs();
    return v;
}

// Codes and decoders per spec, built on first use. Entries are never
// replaced, so references stay valid; the lock covers construction only.
struct Codecs {
    std::optional<ldpc::Code> code, mother;
    std::map<int, std::unique_ptr<ldpc::Decoder>> ldpc;  // by extent, 0 = the code itself
    std::unique_ptr<polar::SCLDecoder> scl;
};
std::mutex cache_mutex;
std::map<const Spec*, Codecs> cache;

const ldpc::Code& mother_code(const Spec& s) {
    const auto& code = ldpc_code(s);
    std::lock_guard lock(cache_mutex);
    auto& c = cache[&s];
    if (!c.mother) c.mother = code.mother();
    return *c.mother;
}

// numpy's broadcast of a scalar-or-per-row argument.
template <typename T>
T at(std::span<const T> v, std::size_t row, std::size_t rows, T dflt) {
    if (v.empty()) return dflt;
    if (v.size() == 1) return v[0];
    if (v.size() != rows) throw std::invalid_argument("per-row argument: expected 1 or one per row");
    return v[row];
}

std::uint32_t mask_at(std::span<const std::uint32_t> m, std::size_t b, std::size_t rows) { return at(m, b, rows, 0u); }
int index_at(std::span<const int> i, std::size_t b, std::size_t rows) { return at(i, b, rows, static_cast<int>(b)); }

int crc_bytes(const Spec& s) { return s.crc_bits / 8; }
int framed_bytes(const Spec& s) { return s.payload_bytes + crc_bytes(s); }

// The first framed_bytes of info bits, XORed with pn (if given), packed MSB first.
std::vector<std::uint8_t> pack(const Spec& s, const std::uint8_t* bits, const std::uint8_t* pn) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(framed_bytes(s)), 0);
    for (std::size_t i = 0; i < out.size() * 8; ++i)
        out[i / 8] |= static_cast<std::uint8_t>(((bits[i] ^ (pn ? pn[i] : 0)) & 1) << (7 - i % 8));
    return out;
}

// pack()'s bytes are payload + CRC: the payload if the CRC matches under mask.
bool crc_matches(const Spec& s, const std::vector<std::uint8_t>& data, std::uint32_t mask) {
    const auto payload = std::span(data).first(static_cast<std::size_t>(s.payload_bytes));
    return with_crc(payload, s.crc_bits, mask) == data;
}

std::vector<std::uint8_t> pn_at(const Spec& s, int index) { return scrambler(s.k, scramble_seed(index)); }

// x[:, :cols] as float32: a buffer is in code order already
template <typename T>
Mat<float> cut(const Spec& s, const Mat<T>& x, std::size_t cols) {
    if (x.cols < cols) throw std::invalid_argument(std::string(s.name) + ": too few soft values per row");
    Mat<float> out(x.rows, cols);
    for (std::size_t b = 0; b < x.rows; ++b)
        for (std::size_t i = 0; i < cols; ++i) out[b][i] = static_cast<float>(x[b][i]);
    return out;
}

Mat<float> deinterleave_llr(const Spec& s, const Mat<float>& llr) {
    if (static_cast<int>(llr.cols) != s.coded_bits) throw std::invalid_argument(std::string(s.name) + ": expected (B, coded_bits)");
    Mat<float> out(llr.rows, llr.cols);
    for (std::size_t b = 0; b < llr.rows; ++b)
        for (std::size_t i = 0; i < llr.cols; ++i) out[b][s.perm[i]] = llr[b][i];
    return out;
}

// codes._decode_code_order: LLRs in the decoder's code order.
Info decode_code_order(const Spec& s, const ldpc::Decoder* dec, const Mat<float>& deint, int iters,
                       std::span<const std::uint32_t> masks, std::span<const int> index) {
    if (dec) {
        auto r = dec->decode(deint, iters);
        return {std::move(r.bits), std::move(r.ok)};
    }
    const auto& scl = polar_decoder(s);
    const auto r = scl.decode(deint);
    const std::size_t B = deint.rows, L = static_cast<std::size_t>(scl.list_size()), k = static_cast<std::size_t>(s.k);
    Info out{Mat<std::uint8_t>(B, k), std::vector<std::uint8_t>(B, 0)};
    for (std::size_t b = 0; b < B; ++b) {
        const auto pn = pn_at(s, index_at(index, b, B));
        const std::uint32_t m = mask_at(masks, b, B);
        std::size_t pick = 0;
        for (std::size_t l = 0; l < L; ++l)
            if (crc_matches(s, pack(s, r.paths[b] + l * k, pn.data()), m)) {
                pick = l;
                out.ok[b] = 1;
                break;
            }
        std::copy_n(r.paths[b] + pick * k, k, out.bits[b]);
    }
    return out;
}

}  // namespace

const Spec* spec(std::string_view name) {
    for (const auto& s : specs())
        if (s.name == name) return &s;
    return nullptr;
}

const Spec& spec(const config::Submode& s) { return specs().at(row(s)); }

const ldpc::Code& ldpc_code(const Spec& s) {
    if (s.polar) throw std::invalid_argument(std::string(s.name) + " is not LDPC");
    std::lock_guard lock(cache_mutex);
    auto& c = cache[&s];
    if (!c.code) c.code = ldpc::qc_code(s.k, s.coded_bits);
    return *c.code;
}

const ldpc::Decoder& ldpc_decoder(const Spec& s, int extent) {
    const auto& code = ldpc_code(s);
    std::lock_guard lock(cache_mutex);
    auto& d = cache[&s].ldpc[extent];
    if (!d) d = std::make_unique<ldpc::Decoder>(extent ? code.mother(extent) : code);
    return *d;
}

const polar::SCLDecoder& polar_decoder(const Spec& s) {
    if (!s.polar) throw std::invalid_argument(std::string(s.name) + " is not polar");
    std::lock_guard lock(cache_mutex);
    auto& d = cache[&s].scl;
    if (!d) d = std::make_unique<polar::SCLDecoder>(polar::PolarCode(s.k, s.coded_bits, s.info_pos), POLAR_LIST);
    return *d;
}

int rv_cycle(const Spec& s) { return s.polar ? 1 : 4; }

int buffer_len(const Spec& s) { return s.polar ? s.coded_bits : mother_code(s).n; }

std::vector<int> rv_positions(const Spec& s, int rv) {
    const int n = s.coded_bits, c = rv_cycle(s), L = buffer_len(s);
    std::vector<int> out(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) out[i] = s.polar ? i : (((rv % c) + c) % c * n + i) % L;
    return out;
}

std::vector<std::uint8_t> info_bits(const Spec& s, std::span<const std::uint8_t> payload, std::uint32_t crc_mask,
                                    int index) {
    if (static_cast<int>(payload.size()) != s.payload_bytes)
        throw std::invalid_argument(std::string(s.name) + " carries " + std::to_string(s.payload_bytes) + " bytes, got " +
                                    std::to_string(payload.size()));
    const auto framed = with_crc(payload, s.crc_bits, crc_mask);
    auto bits = pn_at(s, index);
    for (std::size_t i = 0; i < framed.size() * 8; ++i) bits[i] ^= (framed[i / 8] >> (7 - i % 8)) & 1;
    return bits;
}

Mat<std::uint8_t> encode_info(const Spec& s, const Mat<std::uint8_t>& bits, int rv) {
    Mat<std::uint8_t> coded;
    if (rv && !s.polar) {
        const auto full = mother_code(s).encode(bits);
        const auto pos = rv_positions(s, rv);
        coded = Mat<std::uint8_t>(bits.rows, pos.size());
        for (std::size_t b = 0; b < bits.rows; ++b)
            for (std::size_t j = 0; j < pos.size(); ++j) coded[b][j] = full[b][pos[j]];
    } else {
        coded = s.polar ? polar_decoder(s).code().encode(bits) : ldpc_code(s).encode(bits);
    }
    Mat<std::uint8_t> out(coded.rows, coded.cols);
    for (std::size_t b = 0; b < coded.rows; ++b)
        for (std::size_t i = 0; i < coded.cols; ++i) out[b][i] = coded[b][s.perm[i]];
    return out;
}

std::vector<std::uint8_t> encode(const Spec& s, std::span<const std::uint8_t> payload, int rv, std::uint32_t crc_mask,
                                 int index) {
    Mat<std::uint8_t> info(1, static_cast<std::size_t>(s.k));
    info.data = info_bits(s, payload, crc_mask, index);
    return encode_info(s, info, rv).data;
}

std::vector<double> flip(const Spec& s, int index, int rv) {
    Mat<std::uint8_t> info(1, static_cast<std::size_t>(s.k));
    info.data = pn_at(s, index);
    const auto coded = encode_info(s, info, rv);
    std::vector<double> out(coded.data.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = 1.0 - 2.0 * coded.data[i];
    return out;
}

void combine(const Spec& s, Mat<double>& buf, const Mat<double>& soft, std::span<const int> rvs) {
    if (static_cast<int>(soft.cols) != s.coded_bits) throw std::invalid_argument(std::string(s.name) + ": expected (B, coded_bits)");
    const std::size_t B = soft.rows, L = static_cast<std::size_t>(buffer_len(s));
    if (buf.rows == 0) buf = Mat<double>(B, L, 0.0);
    if (buf.rows < B || buf.cols != L) throw std::invalid_argument(std::string(s.name) + ": buffer shape");
    std::vector<double> deint(soft.cols);
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t i = 0; i < soft.cols; ++i) deint[s.perm[i]] = soft[b][i];
        const auto pos = rv_positions(s, at(rvs, b, B, 0));
        for (std::size_t j = 0; j < pos.size(); ++j) buf[b][pos[j]] += deint[j];  // np.add.at: in order
    }
}

std::vector<Payload> payloads(const Spec& s, const Mat<std::uint8_t>& bits, std::span<const std::uint8_t> converged,
                              std::span<const std::uint32_t> masks, std::span<const int> index) {
    if (static_cast<int>(bits.cols) < 8 * framed_bytes(s) || converged.size() != bits.rows)
        throw std::invalid_argument(std::string(s.name) + ": payloads shape");
    std::vector<Payload> out;
    for (std::size_t b = 0; b < bits.rows; ++b) {
        auto data = pack(s, bits[b], pn_at(s, index_at(index, b, bits.rows)).data());
        const bool ok = converged[b] && crc_matches(s, data, mask_at(masks, b, bits.rows));
        data.resize(static_cast<std::size_t>(s.payload_bytes));
        out.push_back({std::move(data), ok});
    }
    return out;
}

Info decode_llrs(const Spec& s, const Mat<float>& llr, int iters, std::span<const std::uint32_t> masks,
                 std::span<const int> index) {
    return decode_code_order(s, s.polar ? nullptr : &ldpc_decoder(s), deinterleave_llr(s, llr), iters, masks, index);
}

std::vector<Payload> decode_many(const Spec& s, const Mat<float>& soft, std::span<const std::uint32_t> masks,
                                 std::span<const int> index) {
    const auto d = decode_llrs(s, soft, ITERS, masks, index);
    return payloads(s, d.bits, d.ok, masks, index);
}

std::vector<Payload> decode_buffer(const Spec& s, const Mat<double>& buf, int max_rv, std::span<const std::uint32_t> masks,
                                   std::span<const int> index) {
    if (s.polar) {
        const auto d = decode_code_order(s, nullptr, cut(s, buf, static_cast<std::size_t>(s.coded_bits)), ITERS,
                                         masks, index);
        return payloads(s, d.bits, d.ok, masks, index);
    }
    const int extent = std::min(buffer_len(s), (std::min(max_rv, rv_cycle(s) - 1) + 1) * s.coded_bits);
    const auto d = decode_code_order(s, &ldpc_decoder(s, extent), cut(s, buf, static_cast<std::size_t>(extent)),
                                     ITERS, {}, {});
    return payloads(s, d.bits, d.ok, masks, index);
}

Raw decode_raw(const Spec& s, const Mat<float>& soft, std::span<const int> index) {
    const auto deint = deinterleave_llr(s, soft);
    const std::size_t B = soft.rows, k = static_cast<std::size_t>(s.k);
    Raw out;
    if (s.polar) {
        auto r = polar_decoder(s).decode(deint);
        out.list = POLAR_LIST;
        out.cands = std::move(r.paths);
        out.usable = Mat<std::uint8_t>(B, POLAR_LIST, 1);
    } else {
        auto r = ldpc_decoder(s).decode(deint, ITERS);
        out.list = 1;
        out.cands = std::move(r.bits);
        out.usable = Mat<std::uint8_t>(B, 1);
        out.usable.data = std::move(r.ok);
    }
    for (std::size_t b = 0; b < B; ++b) {
        const auto pn = pn_at(s, index_at(index, b, B));
        for (std::size_t l = 0; l < static_cast<std::size_t>(out.list); ++l)
            for (std::size_t j = 0; j < k; ++j) out.cands[b][l * k + j] ^= pn[j];
    }
    return out;
}

std::optional<std::vector<std::uint8_t>> check(const Spec& s, std::span<const std::uint8_t> cands,
                                               std::span<const std::uint8_t> usable, std::uint32_t crc_mask) {
    const std::size_t k = static_cast<std::size_t>(s.k);
    if (cands.size() != usable.size() * k) throw std::invalid_argument(std::string(s.name) + ": check shape");
    for (std::size_t l = 0; l < usable.size(); ++l)
        if (usable[l]) {
            auto data = pack(s, cands.data() + l * k, nullptr);
            if (crc_matches(s, data, crc_mask)) {
                data.resize(static_cast<std::size_t>(s.payload_bytes));
                return data;
            }
        }
    return std::nullopt;
}

std::vector<std::uint8_t> descramble(const Spec& s, std::span<const std::uint8_t> bits, int index) {
    if (static_cast<int>(bits.size()) > s.k) throw std::invalid_argument(std::string(s.name) + ": more than k bits");
    const auto pn = pn_at(s, index);
    std::vector<std::uint8_t> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) out[i] = bits[i] ^ pn[i];
    return out;
}

std::vector<std::uint8_t> crc_ok(const Spec& s, const Mat<std::uint8_t>& bits, std::span<const std::uint32_t> masks,
                                 std::span<const int> index) {
    if (static_cast<int>(bits.cols) < 8 * framed_bytes(s)) throw std::invalid_argument(std::string(s.name) + ": crc_ok shape");
    std::vector<std::uint8_t> out(bits.rows);
    for (std::size_t b = 0; b < bits.rows; ++b)
        out[b] = crc_matches(s, pack(s, bits[b], pn_at(s, index_at(index, b, bits.rows)).data()),
                             mask_at(masks, b, bits.rows));
    return out;
}

}  // namespace data2g::codes
