// data2g/cpm.py: the constant-envelope M-FSK modes (grids, specs, burst
// layout, modulation, Costas sync, header ML, noncoherent soft bits).
//
// Every on-air tone pattern (sync blocks, header tones) is a frozen table
// (tables::CPM_GRIDS); nothing here draws random numbers. The TX bandpass
// (cpm.bandpass, dsp.tx_condition) belongs to the dsp port: modulate()
// returns the audio before it.
#pragma once

#include <complex>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "tables/tables.hpp"
#include "util/mat.hpp"

namespace data2g::cpm {

using Grid = tables::CpmGrid;
using Spec = tables::CpmSpec;

// nullptr if there is no such grid / spec (data modes and control codewords).
const Grid* grid(std::string_view name);
const Spec* spec(std::string_view name);
const Grid& grid_of(const Spec& s);
const Spec& ctl(const Grid& g);

// The header's 10-bit value before its CRC-6: (mode index + 2 dup) << 6 | n_data.
int header_value(int index, int n_data, bool dup);
std::span<const std::uint8_t> header_symbols(const Grid& g, int value);

struct Layout {
    int n = 0;
    std::vector<int> sync_rows, sync_tones;
    std::vector<std::vector<int>> hdr_rows;
    std::vector<int> data_rows;
    int front = 0;
};

int stream_symbols(const Grid& g, int n_data, bool dup);
Layout layout(const Grid& g, int n_sym);
double burst_seconds(const Spec& s, int n_cw, bool dup = false);

std::vector<int> to_tones(const Grid& g, std::span<const std::uint8_t> bits);
std::vector<double> tones(const Grid& g, std::span<const int> sym);
// Every slot's coded bits (control first, twice if dup) -> audio before the TX bandpass.
std::vector<double> modulate(const Spec& s, const std::vector<std::vector<std::uint8_t>>& coded, bool dup);

// (n_sym, m + 2 extra) tone energies from `start`, CFO removed; shares per row.
Mat<double> energies(const Grid& g, std::span<const double> x, long start, int n_sym, double cfo, int extra = 0);
Mat<double> shares(const Mat<double>& E);

struct Detection {
    double score;
    long start;
    double cfo;
};
Detection detect(const Grid& g, std::span<const double> x, double reach_hz = 150.0, bool fine = true,
                 bool front_only = false, int n_sym = 0, double floor = -1.0);

// (n_sym, m) energies of one codeword -> n_sym * bits LLRs, mapping order.
std::vector<double> llrs(const Grid& g, const Mat<double>& E);

struct Header {
    const Spec* spec;
    int n_data;
    bool dup;
    double score, runner_up;
};
Header read_header(const Grid& g, std::span<const double> x, long s0, double cfo, int copies = 2);

struct Soft {
    std::vector<std::vector<double>> slots;  // per slot, mapping order
    Mat<double> E;                           // the stream's tone energies
};
Soft soft(const Grid& g, std::span<const double> x, long s0, double cfo, int n_data, bool dup);

double peak_ratio(const Grid& g, std::span<const double> x, long s0, double cfo);

struct Lock {
    const Spec* spec;
    int n_data;
    bool dup;
    long start;
    double cfo, score, header_score, header_margin;
    long end, header_end;
};
// threshold: nullopt = SYNC_THRESHOLD plus the header floor. hi: nullopt = len(x).
std::optional<Lock> find(const Grid& g, std::span<const double> x, std::optional<double> threshold = std::nullopt,
                         double reach_hz = 150.0, bool front_only = true, long lo = 0,
                         std::optional<long> hi = std::nullopt);

// cpm.measure without the effective-MI features (arq/predictor's, Phase 2):
// those are computed from `snr`, per symbol.
struct Measure {
    double snr_est, spread_est, frames;
    std::vector<double> snr;
};
Measure measure(const Grid& g, const Mat<double>& E, int n_sym);

// cpm.receive: an early lock's burst, whole in x (same sample origin) ->
// the per-slot soft bits and tone energies data2g.arq.phy reads.
struct Received {
    const Spec* spec;
    int n_cw, n_ctl_slots;  // slots, control included; control slots
    bool dup;
    std::vector<std::vector<double>> soft;
    Mat<double> E;
    double cfo;
    long preamble_start, header_end;
};
Received receive(std::span<const double> x, const Lock& lock);

}  // namespace data2g::cpm
