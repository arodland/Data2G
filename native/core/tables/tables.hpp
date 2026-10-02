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

// A QC-LDPC base matrix for base graph bg lifted by z: rows x cols
// circulant shifts, row-major, -1 = zero block (data2g/ldpc.py).
struct ShiftTable {
    int bg, z, rows, cols;
    std::span<const std::int16_t> shift;
};

extern const std::span<const ShiftTable> LDPC_SHIFTS;

}  // namespace data2g::tables
