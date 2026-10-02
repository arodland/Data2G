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
// data2g/cpm.py. Grid row i owns CPM_CTL[i]. header_tones: 1024 rows of
// hdr_len, row v = (mode index + 2 x dup) << 6 | n_data (the header word
// before its CRC-6), as numpy's PCG64 drew them.
struct CpmGrid {
    std::string_view name;
    int m;
    double rate, center, bp, clip_db;
    int T, bits;
    double f0, sync_threshold, header_threshold;
    int hdr_len, costas_len;
    std::span<const std::uint8_t> preamble, mid_block, header_tones;
};

struct CpmSpec {
    std::string_view name, grid, code;
    int index, k, coded_bits, n_sym;
};

struct CpmParams {
    double preamble_s, block_s, spacing_s, hdr_s;
    int hdr_copies, max_data;
    double peak_ratio, ramp_s;
    int data_n, ctl_k, ctl_n;
};

extern const std::span<const CpmGrid> CPM_GRIDS;
extern const std::span<const CpmSpec> CPM_SPECS, CPM_CTL;
extern const CpmParams CPM;

}  // namespace data2g::tables
