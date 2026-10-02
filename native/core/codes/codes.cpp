#include "codes/codes.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

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

}  // namespace data2g::codes
