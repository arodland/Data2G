// data2g/modem.py: the burst modem (header codes and ML decode, burst
// modulation, preamble/header search, the header-copy search, receive up
// to soft bits, decode). Timing arithmetic is in modem/timing.hpp.
//
// Results are structs (Python's dicts). Positions are sample indices into
// the caller's buffer. Every function is reentrant: the only shared state
// is const tables built once (thread-safe statics).
#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "equalizer/equalizer.hpp"
#include "generated/config.hpp"
#include "generated/modem.hpp"
#include "modem/timing.hpp"
#include "util/mat.hpp"
#include "waveform/sync.hpp"

namespace data2g::modem {

using cd = std::complex<double>;
using waveform::SyncError;
using Spec = config::Submode;

double header_min_score(std::string_view band);  // HEADER_MIN_SCORE[band]; throws std::out_of_range
double pilot_noise(std::string_view band);       // PILOT_NOISE[band]
const Spec* by_index(std::string_view sync_band, int index);  // BY_INDEX; nullptr if none

// Which headers a receiver takes beyond validity (modem.Accept). nullptr
// where a function takes `const Accept*` is Python's None.
struct Accept {
    std::vector<std::pair<const Spec*, int>> max_cw;  // (submode, most codewords per burst)
    double min_score = 0.0;

    // names empty: every submode. A submode that can't fit one codeword in max_secs is left out.
    static Accept of(std::span<const std::string_view> names = {}, std::optional<double> max_secs = {},
                     double min_score = 0.0);
    std::vector<std::string_view> bands() const;  // sorted
};

// --- header ---------------------------------------------------------------

int crc6(int v);  // CRC-6 over 10 bits, seeded with PROTOCOL_VERSION
std::span<const std::uint16_t> header_cols(std::string_view band);  // the code's column masks
std::vector<std::uint8_t> codeword(int word, std::string_view band);  // 16-bit word -> coded header bits
// Throws std::invalid_argument unless 1 <= n_cw <= config::max_codewords(band) and the index fits.
std::vector<std::uint8_t> header_bits(int submode, int n_cw, std::string_view band = "w");
std::vector<int> valid_words(std::string_view band, const Accept* accept = nullptr);
// The ML correlation with each valid word, float32 as in Python: summed in
// double (a Walsh-Hadamard transform, the CRC being affine), then rounded.
// numpy's BLAS sums in float32 in its own order, so values differ by a few
// ulps. Reentrant; the cost no longer grows with the number of valid words.
std::vector<float> header_corr(std::span<const double> soft, std::string_view band, const Accept* accept = nullptr);

struct Header {
    int word = 0;
    const Spec* spec = nullptr;
    int n_cw = 0;
    double score = 0.0;
};
// ML over the valid words; the first of equal correlations wins (np.argmax).
Header decode_header(std::span<const double> soft, std::string_view band = "w", const Accept* accept = nullptr);

// --- transmit -------------------------------------------------------------

// Codeword payloads -> unit-RMS audio. rvs empty: all 0.
std::vector<double> modulate(std::span<const std::vector<std::uint8_t>> payloads, const Spec& spec,
                             std::span<const int> rvs = {});
// Coded bits in burst order (codes::spread) -> unit-RMS audio.
std::vector<double> modulate_bits(std::span<const std::uint8_t> bits, const Spec& spec);
// (n_f * 5, nc) data symbols -> the unclipped burst waveform.
std::vector<double> burst_waveform(const Mat<cd>& data, const Spec& spec);
// First sample of each data symbol of a burst of n_f data frames (ace_cells' `full`[:, 0]).
std::vector<std::int64_t> ace_cells(const Spec& spec, int n_f);

// --- receive --------------------------------------------------------------

double bin_phase_step(std::span<const cd> h);

struct Frames {
    Mat<cd> raw;  // (n_f * 6, nc): frame f, symbol s at row f * 6 + s
    Mat<cd> hp;   // (n_f + 1, nc) pilots, the closing one last
    std::vector<double> steps;  // (n_f + 1,)
};
// steps_in: replay a previous run's steps. Throws SyncError past the buffer.
Frames demod_frames(std::span<const cd> z, std::int64_t p, int n_f, int shift, double phi_ref,
                    std::span<const double> steps_in = {}, std::string_view band = "w");

struct HeaderRead {
    Header hdr;
    bool valid = false;  // false: under the floor (Python's hdr None)
    bool pending_copy = false;
    Mat<cd> y, y_all;   // header data symbols; every header symbol
    std::vector<cd> h_pre, h_first;
    std::int64_t p0 = 0, start = 0;
    std::string_view band;
    double n0_pre = 0.0;
    std::vector<double> n0_pre_k;
};
HeaderRead read_header(std::span<const cd> z, std::int64_t start, std::string_view band = "w",
                       const Accept* accept = nullptr);
// nullopt: the copy frame is not yet in z.
std::optional<std::vector<double>> copy_llr(std::span<const cd> z, std::int64_t p, std::string_view band, int n_hdr);

// A band's precomputed detection statistic (sync::StreamDetector) for the buffer's starts.
struct BandStat {
    std::string_view band;
    const Mat<double>* S;
};

struct BestHeader {
    HeaderRead hd;
    waveform::Acquisition acq;
    std::shared_ptr<const std::vector<cd>> z;  // z0 corrected by acq.freq_offset (shared: find_burst never copies it)
};
// bands empty: accept's (sorted) or SYNC_BANDS. Throws SyncError. final (with
// complete false): z0 is the head of a burst that has wholly arrived, so
// nothing waits (modem.py _best_header).
BestHeader best_header(std::span<const cd> z0, std::span<const std::string_view> bands = {}, bool complete = true,
                       const Accept* accept = nullptr, std::span<const BandStat> stats = {}, bool final = false);

struct CopyRef {
    int word = 0;
    std::int64_t pc = 0;  // the copy frame's first sample
};
// find_burst's / find_copy's dict: where a burst is.
struct Lock {
    const Spec* spec = nullptr;
    int n_cw = 0;
    std::int64_t start = 0, end = 0, p0 = 0;
    double score = 0.0;
    std::string_view band;
    double cfo = 0.0;
    std::optional<CopyRef> copy;  // set by find_copy
};
Lock find_burst(std::span<const double> x, std::span<const std::string_view> bands = {},
                const Accept* accept = nullptr, std::span<const BandStat> stats = {});
std::vector<double> pilot_coherence(std::span<const double> x, const Lock& lock, int n_max = 8, bool latest = false);
// C and level both given (StreamDetector.C rows for x's starts, its level) or
// neither. peak: the normalized pilot peak, if one was computed (find_copy.peak).
std::optional<Lock> find_copy(std::span<const double> x, std::string_view band, const Accept* accept = nullptr,
                              const Mat<cd>* C = nullptr, std::optional<double> level = {}, double* peak = nullptr);
// The same with C as row pointers (C_rows empty: compute C, as C == nullptr),
// each row `cols` long: StreamDetector.C rows, used in place.
std::optional<Lock> find_copy(std::span<const double> x, std::string_view band, const Accept* accept,
                              std::span<const cd* const> C_rows, std::size_t cols, std::optional<double> level,
                              double* peak = nullptr);
std::vector<double> cfo_aliases(cd d, double centre);
HeaderRead copy_header(std::span<const cd> z, const Lock& lock);

// equalizer::Estimate scaled to the data's channel (data_channel's dict).
struct DataEstimate : equalizer::Estimate {
    double clip_ratio = 0.0, gain = 1.0;
    std::string_view band;
};
// config.clip_consts: gain by frame count, the default, the clip-noise ratio.
struct ClipConsts {
    std::vector<std::pair<int, double>> gains;
    double gain = 1.0, ratio = 0.0;
};
ClipConsts clip_consts(const Spec& spec);       // receive()'s
ClipConsts clip_consts(std::string_view band);  // config.CLIP[band]
// clip nullptr: config.CLIP[band]. n_frames < 0: len(h_pilot) - 1.
DataEstimate data_channel(const Mat<cd>& h_pilot, equalizer::Support support, std::string_view band = "w",
                          double n0_pre = equalizer::INF, const ClipConsts* clip = nullptr,
                          std::span<const double> n0_pre_k = {}, int n_frames = -1);

struct Received {
    const Spec* spec = nullptr;
    int n_cw = 0;
    Mat<cd> raw;  // (n_f * 6, nc) data frames (the header copy's dropped), row f * 6 + s
    DataEstimate est;
    waveform::Acquisition acq;
    std::string_view band;  // the data band
    Mat<cd> hp;             // (frames on air + 1, nc), the copy frame's pilot included
    std::optional<int> kc;
    double cfo = 0.0;
    std::int64_t p0 = 0;
    int shift = 0;
    std::vector<double> steps;
    double phi_ref = 0.0;
    equalizer::Support support;
    std::int64_t preamble_start = 0;
    double score = 0.0;
};
// head: the preamble and header lie in x[:head]. copy: find_copy's lock. Throws SyncError.
Received receive(std::span<const double> x, std::span<const std::string_view> bands = {},
                 const Accept* accept = nullptr, std::optional<std::int64_t> head = {}, const Lock* copy = nullptr);
double resolve_alias(double fine, double coarse);

// var (rows, nc): n0_k (n0 if empty) per carrier + clip_ratio |h|^2.
Mat<double> noise_var(const Mat<cd>& h, const DataEstimate& est);
// LLRs of raw's data symbols (rows f * 6 + 1..5) given h, var (n_f * 5, nc).
std::vector<double> soft_bits(const Mat<cd>& raw, const Mat<cd>& h, const Mat<double>& var, const Spec& spec);

struct Burst {
    const Spec* spec = nullptr;
    std::vector<std::vector<std::uint8_t>> payloads;
    std::vector<bool> crc_ok;
    double freq_offset = 0.0;
    std::int64_t preamble_start = 0;
    double snr_db = 0.0;
    Mat<double> soft;  // (n_cw, coded_bits), mapping order
};
Burst decode_received(const Received& r);
Burst demodulate(std::span<const double> x, std::span<const std::string_view> bands = {},
                 const Accept* accept = nullptr);

}  // namespace data2g::modem
