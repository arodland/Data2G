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
// before its CRC-6), as numpy's PCG64 drew them. tx: cpm.TX_FILTERS, by
// bandwidth cap code (0: 500 Hz, 1: 1200, 2: 2400).
struct CpmTxFilter {
    double bp, glide;
    int passes;
};

struct CpmGrid {
    std::string_view name;
    int m;
    double rate, center, bp, clip_db;
    int T, bits;
    double f0, sync_threshold, header_threshold;
    int hdr_len, costas_len;
    std::span<const std::uint8_t> preamble, mid_block, header_tones;
    CpmTxFilter tx[3];
};

struct CpmSpec {
    std::string_view name, grid, code;
    int index, k, coded_bits, n_sym;
    std::span<const std::uint16_t> perm;  // interleaver (codes.interleaver)
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
// Polar info sets Python designs at run time (codes.polar_code with no
// frozen file: the CPM control codewords), frozen here by (k, e).
struct PolarDesign {
    int k, e;
    std::span<const std::uint16_t> info_pos;
};
extern const std::span<const PolarDesign> POLAR_GA;
// scipy.stats.gamma.ppf(0.99, n) / n at index n - 1 (equalizer.per_carrier_noise).
inline constexpr int GAMMA_Q99_MAX = 2048;
extern const std::array<double, GAMMA_Q99_MAX> GAMMA_Q99;
// A QC-LDPC base matrix for base graph bg lifted by z: rows x cols
// circulant shifts, row-major, -1 = zero block (data2g/ldpc.py).
struct ShiftTable {
    int bg, z, rows, cols;
    std::span<const std::int16_t> shift;
};

extern const std::span<const ShiftTable> LDPC_SHIFTS;
// Deflate's priming dictionary (data2g/arq/frames.py ZDICT, zdict.bin).
extern const std::span<const std::uint8_t> ZDICT;

// The gear shifter's outcome model (data2g/arq/predictor.py): bootstrap
// members, each an MLP (x - mean) / std -> tanh layers -> logits, W row-major
// (in, out). Outputs: P(burst usable) per OUTCOME_MODES, then P(codeword).
struct MlpLayer {
    int in, out;
    std::span<const double> W, b;
};
struct OutcomeMember {
    std::span<const double> mean, std;
    std::span<const MlpLayer> layers;
};
extern const std::span<const OutcomeMember> OUTCOME_MEMBERS;
// The gate's model (predictor.GATE_MODEL), its outputs in OUTCOME_MODES' order,
// and the gate (policy.GearShifter.gate): median spread_est of the last
// OUTCOME_GATE_HIST peer bursts under SPREAD_HZ and snr_est under SNR_DB.
extern const std::span<const OutcomeMember> OUTCOME_GATE_MEMBERS;
extern const double OUTCOME_GATE_SPREAD_HZ, OUTCOME_GATE_SNR_DB;
extern const int OUTCOME_GATE_HIST;
// whether each model takes the energy inputs (predictor.N_ENERGY)
extern const bool OUTCOME_ENERGY, OUTCOME_GATE_ENERGY;
// each mode's burst peak-to-average (arq/phy.py peak_db) by bandwidth cap
// code: the energy inputs' peak reference
struct ModePeak {
    std::string_view mode;
    double db[3];
};
extern const std::span<const ModePeak> MODE_PEAK_DB;
extern const std::span<const std::string_view> OUTCOME_MODES, OUTCOME_BANDS;
// The data ladder's robustness order (data2g/arq/policy.py MODE_THRESHOLDS):
// per mode, its 10% codeword failure SNR (dB) on awgn, mpg, mpp, mpd.
struct ModeThreshold {
    std::string_view name;
    std::array<double, 4> db;
};
extern const std::span<const ModeThreshold> MODE_THRESHOLDS;
// AWGN BICM capacity (bits per coded bit) over CAPACITY_GRID (dB), per
// constellation family (predictor.CONSTS order).
struct CapacityTable {
    std::string_view name;
    std::span<const double> mi;
};
extern const std::span<const double> CAPACITY_GRID;
extern const std::span<const CapacityTable> CAPACITY;

}  // namespace data2g::tables
