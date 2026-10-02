// Frozen on-air data, compiled in. Definitions are generated
// (core/generated/*.cpp, tools/gen_native_tables.py).
#pragma once

#include <array>
#include <complex>
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

// A constellation (data2g/constellation.py): 2^m points, point i carrying
// i's bits MSB first, and its ACE directions, row-major (2^m, 2): unit
// outward directions point i may move along (zero: none). Frozen because
// learned sets get theirs from scipy's ConvexHull.
struct Constellation {
    std::string_view name;
    int m;
    std::span<const std::complex<double>> points, ace;
};

// gray-qam4..256, then every data2g/constellations/*.npy by name.
extern const std::span<const Constellation> CONSTELLATIONS;

}  // namespace data2g::tables
