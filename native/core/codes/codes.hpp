// data2g/codes.py: CRCs, the PN9 scrambler, interleavers.
#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "generated/config.hpp"

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

}  // namespace data2g::codes
