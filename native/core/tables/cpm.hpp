// The CPM modes' grids, specs and parameters (data2g/cpm.py), compiled in:
// generated/cpm.cpp. Other areas read them through cpm::.
#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <string_view>

namespace data2g::tables {

// Grid row i owns CPM_CTL[i]. header_tones: 1024 rows of hdr_len, row v =
// (mode index + 2 x dup) << 6 | n_data (the header word before its CRC-6),
// as numpy's PCG64 drew them. tx: cpm.TX_FILTERS, by bandwidth cap code
// (0: 500 Hz, 1: 1200, 2: 2400).
struct CpmTxFilter {
    double bp, glide;
    int passes;
};

inline constexpr double NO_PRESENCE = std::numeric_limits<double>::infinity();

struct CpmGrid {
    std::string_view name;
    int m;
    double rate, center, bp, clip_db;
    int T, bits;
    // sync_threshold, mid_threshold: detect's floors (front, mid block); lock_llr1/2:
    // find's and late_lock's tone_llr score floors, wait_llr: a waiting front's; mid_llr: find_mid's;
    // presence_a/b: cpm.PRESENCE's a + b / sqrt(n) floor for data2g_on_air (NO_PRESENCE: no rule, never present).
    double f0, sync_threshold, mid_threshold, lock_llr1, lock_llr2, wait_llr, mid_llr, presence_a, presence_b;
    int hdr_len;
    std::span<const std::uint8_t> preamble, mid_block, header_tones;
    CpmTxFilter tx[3];
    // A wide-bin grid (cpm.WIDE_GRIDS): a symbol is L sub-windows of T / L samples, tones still `rate` apart,
    // a tone's energy the sum of the sub-windows' from the first `guard` on; noise_median and median_se are
    // that k = L - guard sum's (cpm._noise_median, _median_se). The on-air grids keep these defaults.
    int L = 1, guard = 0;
    double noise_median = 0.6931471805599453, median_se = 1.44;  // ln 2: an exponential's median over its mean
};

// A GF(M) data code's mother code (codes.gf_code, data2g.gfldpc.GFCode): n
// symbols, the codeword its first n0, rows of dmax slots (var -1: padding).
struct GfTable {
    int n = 0, n0 = 0, dmax = 0;
    std::span<const std::int16_t> var;
    std::span<const std::uint8_t> h;
};

struct CpmSpec {
    std::string_view name, grid, code;  // code: "gf" (data) | "polar" (control)
    int index, k, coded_bits, n_sym;
    std::span<const std::uint16_t> perm;  // interleaver (codes.interleaver; GF(M): of symbols)
    GfTable gf;  // code "gf" only
};

struct CpmParams {
    double preamble_s, block_s, spacing_s, hdr_s;
    int hdr_copies, max_data;
    double peak_ratio, ramp_s;
    int data_n, ctl_k, ctl_n;
    double hdr_margin1, hdr_margin2, llr_eps;  // cpm.HDR_MARGIN (one copy, both), LLR_EPS
    int refine_symbols;
};

extern const std::span<const CpmGrid> CPM_GRIDS;
extern const std::span<const CpmSpec> CPM_SPECS, CPM_CTL;
// The wide-bin toolbox grids, their data specs (GF codes another spec's: CpmSpec.gf holds that code) and control
// specs, row i's control CPM_WIDE_CTL[i] (generated/cpm_wide.cpp). On air only as DATA2G_CPM_ONAIR names them.
extern const std::span<const CpmGrid> CPM_WIDE_GRIDS;
extern const std::span<const CpmSpec> CPM_WIDE_SPECS, CPM_WIDE_CTL;
extern const CpmParams CPM;

}  // namespace data2g::tables
