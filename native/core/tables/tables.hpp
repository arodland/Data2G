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

// Polar info sets Python designs at run time (codes.polar_code with no
// frozen file: the CPM control codewords), frozen here by (k, e).
struct PolarDesign {
    int k, e;
    std::span<const std::uint16_t> info_pos;
};
extern const std::span<const PolarDesign> POLAR_GA;

}  // namespace data2g::tables
