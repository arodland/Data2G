// Frozen on-air data, compiled in. Definitions are generated
// (core/generated/*.cpp, tools/gen_native_tables.py).
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

#include "generated/config.hpp"

namespace data2g::tables {

// A submode's interleaver (perm[i] = code bit at mapping position i) and,
// on polar submodes, its info set. Empty info_pos: LDPC.
struct Format {
    std::string_view submode;
    std::span<const std::uint16_t> perm, info_pos;
};

// Row i belongs to config::SUBMODES[i].
extern const std::array<Format, config::SUBMODES.size()> FORMATS;

// scipy.stats.gamma.ppf(0.99, n) / n at index n - 1 (equalizer.per_carrier_noise).
inline constexpr int GAMMA_Q99_MAX = 2048;
extern const std::array<double, GAMMA_Q99_MAX> GAMMA_Q99;

}  // namespace data2g::tables
