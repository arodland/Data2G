// data2g/codes.py: CRCs, the PN9 scrambler, interleavers, and the codec
// (encode, HARQ combining, batched decode) over ldpc:: and polar::.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "generated/config.hpp"
#include "ldpc/ldpc.hpp"
#include "polar/polar.hpp"
#include "util/mat.hpp"

namespace data2g::codes {

inline constexpr int PLAIN = -1;  // burst position meaning "not scrambled"

std::uint16_t crc16(std::span<const std::uint8_t> data);  // CRC-16/CCITT-FALSE (binascii.crc_hqx, 0xFFFF)
std::uint32_t crc24(std::span<const std::uint8_t> data, std::uint32_t crc = 0xFFFFFF);  // CRC-24C
std::uint32_t crc32(std::span<const std::uint8_t> data);  // zlib's

// payload + its n_crc-bit CRC (16, 24 or 32) XORed with mask, big-endian.
std::vector<std::uint8_t> with_crc(std::span<const std::uint8_t> payload, int n_crc, std::uint32_t mask = 0);

int scramble_seed(int index);
std::vector<std::uint8_t> scrambler(int k, int seed = 0x1FF);  // PN9, k bits

// Any spec with .code and .k (config::Submode, tables::CpmSpec).
template <typename Spec>
int crc_bits(const Spec& spec) {
    if (spec.code == "polar") return 24;
    return spec.k >= 512 ? 32 : 16;
}

template <typename Spec>
int payload_bytes(const Spec& spec) {
    return (spec.k - crc_bits(spec)) / 8;
}

// nullptr if there is no such submode.
const config::Submode* submode(std::string_view name);
// spec must be an element of config::SUBMODES.
std::span<const std::uint16_t> interleaver(const config::Submode& spec);
std::span<const std::uint16_t> info_pos(const config::Submode& spec);

// --- the codec (codes.encode / decode / combine) ------------------------------
//
// Batches are Mat rows. Per-row arguments (masks: CRC masks, index: burst
// positions, rvs) broadcast like numpy: empty means the default (mask 0,
// index = row number), one value means every row, else one per row.
// Everything is reentrant: one call never shares mutable state with another.

// A coded format: a submode, a CPM mode or a CPM control codeword
// (Python's SubmodeSpec / CpmSpec where data2g.codes reads them).
struct Spec {
    std::string_view name;
    bool polar;  // else LDPC
    int k, coded_bits, bits_per_cu, crc_bits, payload_bytes;
    std::span<const std::uint16_t> perm, info_pos;  // interleaver; polar info set
};

const Spec* spec(std::string_view name);  // nullptr if none
const Spec& spec(const config::Submode& s);

const ldpc::Code& ldpc_code(const Spec& s);  // LDPC specs
// codes._decoder (extent 0) or codes._ext_decoder(spec, extent).
const ldpc::Decoder& ldpc_decoder(const Spec& s, int extent = 0);
const polar::SCLDecoder& polar_decoder(const Spec& s);  // polar specs, list POLAR_LIST
const polar::SCLDecoder& polar_ir_decoder(const Spec& s);  // codes.polar_ir_code's, list POLAR_LIST

inline constexpr int POLAR_LIST = 8;
inline constexpr int ITERS = 40;  // LDPC decoder iterations

int rv_cycle(const Spec& s);
int buffer_len(const Spec& s);
std::vector<int> rv_positions(const Spec& s, int rv);

// payload (exactly payload_bytes) -> k info bits: payload, masked CRC, zero fill, scrambled.
std::vector<std::uint8_t> info_bits(const Spec& s, std::span<const std::uint8_t> payload, std::uint32_t crc_mask = 0,
                                    int index = 0);
// (B, k) info bits -> (B, coded_bits), interleaved; rv > 0 on LDPC: mother-code parity.
Mat<std::uint8_t> encode_info(const Spec& s, const Mat<std::uint8_t>& bits, int rv = 0);
std::vector<std::uint8_t> encode(const Spec& s, std::span<const std::uint8_t> payload, int rv = 0,
                                 std::uint32_t crc_mask = 0, int index = 0);
// +-1 per coded bit: soft bits sent at (index, rv) times this are the unscrambled codeword's.
std::vector<double> flip(const Spec& s, int index, int rv = 0);

// (n_cw, n) row-major -> burst order, every codeword's m-bit symbols dealt round-robin.
template <typename T>
std::vector<T> spread(std::span<const T> coded, int n_cw, int m) {
    const std::size_t n = coded.size() / n_cw, syms = n / m;
    std::vector<T> out(coded.size());
    for (int c = 0; c < n_cw; ++c)
        for (std::size_t j = 0; j < syms; ++j)
            for (int b = 0; b < m; ++b) out[(j * n_cw + c) * m + b] = coded[c * n + j * m + b];
    return out;
}

template <typename T>
std::vector<T> despread(std::span<const T> x, int n_cw, int m) {
    const std::size_t n = x.size() / n_cw, syms = n / m;
    std::vector<T> out(x.size());
    for (int c = 0; c < n_cw; ++c)
        for (std::size_t j = 0; j < syms; ++j)
            for (int b = 0; b < m; ++b) out[c * n + j * m + b] = x[(j * n_cw + c) * m + b];
    return out;
}

// Adds (B, coded_bits) soft bits in mapping order, sent at RVs `rvs`, into
// the (B, buffer_len) buffer; an empty buf (rows 0) starts as zeros.
void combine(const Spec& s, Mat<double>& buf, const Mat<double>& soft, std::span<const int> rvs);

struct Payload {
    std::vector<std::uint8_t> data;
    bool ok;  // CRC matches and the decoder converged
};

// Decoded (scrambled) info bits -> payloads (codes._payloads).
std::vector<Payload> payloads(const Spec& s, const Mat<std::uint8_t>& bits, std::span<const std::uint8_t> converged,
                              std::span<const std::uint32_t> masks = {}, std::span<const int> index = {});

struct Info {
    Mat<std::uint8_t> bits;     // (B, k), still scrambled
    std::vector<std::uint8_t> ok;  // LDPC: H satisfied; polar: a list path's CRC checked
};

// (B, coded_bits) LLRs in mapping order, decoded as one batch (LDPC: the
// batch stops together, so a row's result depends on its batch mates).
Info decode_llrs(const Spec& s, const Mat<float>& llr, int iters = ITERS, std::span<const std::uint32_t> masks = {},
                 std::span<const int> index = {});
std::vector<Payload> decode_many(const Spec& s, const Mat<float>& soft, std::span<const std::uint32_t> masks = {},
                                 std::span<const int> index = {});
// (B, buffer_len) combined buffers, RVs up to max_rv received.
std::vector<Payload> decode_buffer(const Spec& s, const Mat<double>& buf, int max_rv = 0,
                                   std::span<const std::uint32_t> masks = {}, std::span<const int> index = {});

struct Raw {
    int list;                  // candidates per row: 1 (LDPC) or POLAR_LIST
    Mat<std::uint8_t> cands;   // (B, list * k), descrambled, best first
    Mat<std::uint8_t> usable;  // (B, list)
};
// Decode once with the CRC mask open; then check() each mask in question.
Raw decode_raw(const Spec& s, const Mat<float>& soft, std::span<const int> index = {});
// One row of decode_raw -> the first usable candidate's payload whose CRC passes, or empty.
std::optional<std::vector<std::uint8_t>> check(const Spec& s, std::span<const std::uint8_t> cands,
                                               std::span<const std::uint8_t> usable, std::uint32_t crc_mask);

std::vector<std::uint8_t> descramble(const Spec& s, std::span<const std::uint8_t> bits, int index);
// (B, >= payload + CRC bits) scrambled info bits -> (B,) CRC matches.
std::vector<std::uint8_t> crc_ok(const Spec& s, const Mat<std::uint8_t>& bits,
                                 std::span<const std::uint32_t> masks = {}, std::span<const int> index = {});

}  // namespace data2g::codes
