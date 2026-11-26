// data2g/cpm.py: the constant-envelope M-FSK modes (grids, specs, burst
// layout, modulation, Costas sync, header ML, noncoherent soft bits).
//
// Every on-air tone pattern (sync blocks, header tones) is a frozen table
// (tables::CPM_GRIDS); nothing here draws random numbers. Unlike Python's,
// modulate() returns the audio before the TX bandpass: bandpass() is that
// step (cpm.modulate = bandpass(modulate)).
#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <numbers>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "tables/cpm.hpp"
#include "util/mat.hpp"

namespace data2g::cpm {

using Grid = tables::CpmGrid;
using Spec = tables::CpmSpec;

// nullptr if there is no such grid / spec (data modes and control codewords). By name: the wide-bin
// toolbox grids and their specs too (tables::CPM_WIDE_*, cpm.py's _Toolbox lookup), on air or not.
const Grid* grid(std::string_view name);
const Spec* spec(std::string_view name);
const Grid& grid_of(const Spec& s);
const Spec& ctl(const Grid& g);
// cpm.GRIDS' and cpm.SPECS' iteration: the on-air grids (tables::CPM_GRIDS, then the wide grids
// DATA2G_CPM_ONAIR names, cpm.ON_AIR_WIDE, read once) and their data specs (CPM_SPECS, then those wide
// grids' in CPM_WIDE_SPECS' order). Unset, master's tables.
const std::vector<const Grid*>& grids();
const std::vector<const Spec*>& specs();
// cpm.WIDE_ON (DATA2G_CPM_WIDE=1, read once): every wide spec in arq::modes() (studies).
bool wide_on();
// cpm.grid_specs: grid g's data specs (valid_words' order).
std::vector<const Spec*> grid_specs(const Grid& g);
// A wide grid's geometry (cpm.Grid's properties): samples per sub-window, symbols/s, sub-windows a tone
// energy sums. The on-air grids: T, rate, 1.
inline int nsub(const Grid& g) { return g.T / g.L; }
inline double baud(const Grid& g) { return g.rate / g.L; }
inline int kept(const Grid& g) { return g.L - g.guard; }
// cpm.shares_rows: wide grid g's coarse rows are sums of on-air grid h's (c8w50 on c8r50).
bool shares_rows(const Grid& g, const Grid& h);

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
// glide: the frequency trajectory smoothed over this fraction of a symbol (CpmTxFilter).
std::vector<double> tones(const Grid& g, std::span<const int> sym, double glide = 0.0);
// Every slot's coded bits (control first, twice if dup) -> audio before the
// TX bandpass, with the bandwidth cap's glide (tx_filter).
std::vector<double> modulate(const Spec& s, const std::vector<std::vector<std::uint8_t>>& coded, bool dup, int cap = 0);
// cpm.tx_filter: the grid's TX filter under bandwidth cap code `cap` (0..2).
const tables::CpmTxFilter& tx_filter(const Grid& g, int cap);
// cpm.TxFilter.overshoot: CLIP_OVERSHOOT, its last factor repeated to `passes`.
std::vector<double> overshoot(const tables::CpmTxFilter& f);
// cpm.bandpass: modulate()'s audio through the TX filter of cap code `cap`,
// clip and filter (waveform::tx_condition) over the tones +- its margin.
std::vector<double> bandpass(const Grid& g, std::span<const double> x, int cap = 0);

// (n_sym, m + 2 extra) tone energies from `start`, CFO removed (a wide grid: its kept sub-windows'
// summed); shares per row.
Mat<double> energies(const Grid& g, std::span<const double> x, std::int64_t start, int n_sym, double cfo, int extra = 0);
Mat<double> shares(const Mat<double>& E);

struct Detection {
    double score;
    std::int64_t start;
    double cfo;
};
// cpm.presence / present: Receiver.data2g_on_air's CPM reading (no sync or header needed).
inline constexpr double PRESENCE_MIN_S = 1.0;
inline constexpr int PRESENCE_PHASES = 4;
inline constexpr int PRESENCE_FRACS = 2;
inline constexpr double PRESENCE_C = 20.0;
double presence(const Grid& g, std::span<const double> x, double reach_hz = 150.0);
bool present(const Grid& g, std::span<const double> x);

struct Header {
    const Spec* spec;
    int n_data;
    bool dup;
    double score, runner_up;
    double front;  // the front block's part of score
};
// cpm.CoarseTrack: a streaming receiver's coarse detection rows of grid g as audio arrives, on the stream's
// T/4 timing grid, so each hop's searches (find, find_mid) read them instead of transforming their windows again.
// Also keeps the raw bin powers from the same FFTs for presence() / present() (q in 0, 2) and a sourced track.
class CoarseTrack {
public:
    // source: the track of a grid g shares rows with (shares_rows), fed the same audio first: g's rows are sums
    // of its symbols' energies, no transforms of its own (a start the source does not hold: g's own). It must
    // outlive this track.
    explicit CoarseTrack(const Grid& g, double reach_hz = 150.0, CoarseTrack* source = nullptr);
    // buf holds stream samples off..: rows for its new whole symbols; rows of samples it no longer holds dropped
    void feed(std::span<const double> buf, std::int64_t off);
    // detect's per-phase (first symbol's sample from a, rows n x 4 nb) of the whole symbols in [a, b)
    std::vector<std::pair<std::int64_t, Mat<double>>> rows(std::int64_t a, std::int64_t b) const;
    // presence() / present() over the whole symbols in stream samples [lo, hi)
    double presence(std::int64_t lo, std::int64_t hi) const;
    bool present(std::int64_t lo, std::int64_t hi) const;
    int extra;  // bins searched each side of the grid (detect's reach)

    // Native only, results unchanged: a stream re-searches the same candidate hop after hop on the same audio
    // (60 s of noise: 44% of c16r25's fine searches, 24% of c8r50's), so detect's fine search of a coarse
    // candidate (stream sample, CFO, which rows) and find's header reading of its result are kept while their
    // audio is. Only entries whose whole span was in the buffer: zero-filled edges would differ.
    struct FineMemo {
        std::int64_t s;
        double cfo;
        bool front_only, mid;
        Detection fine;  // start a stream sample
    };
    struct HeaderMemo {
        std::int64_t s;
        double cfo;
        Header h;
        bool ok;
    };
    mutable std::vector<FineMemo> fine_memo;
    mutable std::vector<HeaderMemo> header_memo;
    const Grid& grid() const { return *g_; }

private:
    bool sourced(std::int64_t nxt, std::int64_t n, std::vector<double>& out, std::vector<double>& raw) const;
    const Grid* g_;
    std::array<std::optional<std::int64_t>, 4> s_{};  // per phase: the stream sample of its first row
    std::array<std::vector<double>, 4> E_{};          // per phase: rows of 4 nb, flat (shares)
    std::array<std::vector<double>, 4> R_{};          // per phase: rows of 4 nb, flat (raw powers: presence, a sourced track)
    CoarseTrack* source_ = nullptr;
};

// mid: the mid block alone, its first sample for the start (find_mid). track: its rows for the coarse stage
// (x the stream from sample x0).
Detection detect(const Grid& g, std::span<const double> x, double reach_hz = 150.0, bool fine = true,
                 bool front_only = false, int n_sym = 0, double floor = -1.0, bool mid = false,
                 const CoarseTrack* track = nullptr, std::int64_t x0 = 0);

// (n_sym, m) energies of one codeword -> n_sym * bits LLRs, mapping order (the control codeword's).
std::vector<double> llrs(const Grid& g, const Mat<double>& E);
// cpm.bit_llrs: tone log-likelihoods metric * scale -> their coded-bit marginals (llrs: E at its metric's scale).
std::vector<double> bit_llrs(const Grid& g, const Mat<double>& metric, double scale = 1.0);
// -> n_sym * m symbol log-likelihoods, mapping order, value v's on tone gray(v) (a GF(M) data codeword's).
std::vector<double> symbol_ll(const Grid& g, const Mat<double>& E);

// cpm.tone_llr: per row and tone, that tone's log-likelihood ratio against a random one (floored by llr_eps).
Mat<double> tone_llr(const Grid& g, const Mat<double>& E);

// The leak model's study switches (cpm.py LEAK, LEAK_K, LEAK_Z, CTL_SQ,
// SPILL), read from DATA2G_CPM_LEAK, _LEAK_K, _LEAK_Z, _CTL_SQ and _SPILL
// once, with Python's defaults (off). Use: tone_metric's consumer, cpm.py's
// `use`: lock (tone_llr and refine: lock gates, header scores, nods), ctl
// (llrs), data (symbol_ll), dd (dd_metric).
enum class Use { lock, ctl, data, dd };
struct Study {
    std::array<bool, 4> leak{};  // by Use: that consumer on leak_metric where spill_significant
    bool ctlsq = false;          // "ctlsq" in DATA2G_CPM_LEAK: ctl_square
    double leak_k = 0.5, leak_z = 2.0, ctl_sq = 0.15;
    std::array<double, 4> spill{1.0, 1.0, 1.0, 1.0};  // by Use: spill()'s weight in the square law's N0
};
const Study& study();
void set_study(const Study& s);  // tests only (the bindings): never while a receive runs
// g: the grid whose energies E are (a wide grid's sum kept(g) sub-windows: every level is a sub-window's);
// nullptr: an on-air grid's (one window a symbol).
// cpm.tone_metric: (n_sym, m) tone energies -> log P(E | tone) up to a row constant, for `use`.
Mat<double> tone_metric(const Mat<double>& E, Use use, const Grid* g = nullptr);
// cpm.leak_metric: the Doppler spill as signal beside the sent tone (study().leak).
Mat<double> leak_metric(const Mat<double>& E, double k, const Grid* g = nullptr);
// cpm.spill_significant: the codeword's spill stands leak_z standard errors over 0.
bool spill_significant(const Mat<double>& E, const Grid* g = nullptr);
// cpm.ctl_square: control slot `slot` of a burst's data-row energies E stays on the square law (study().ctl_sq).
bool ctl_square(const Grid& g, const Mat<double>& E, int slot);

Header read_header(const Grid& g, std::span<const double> x, std::int64_t s0, double cfo, int copies = 2);

struct Soft {
    std::vector<std::vector<double>> slots;  // per slot, mapping order
    Mat<double> E;                           // the stream's tone energies
};
Soft soft(const Grid& g, std::span<const double> x, std::int64_t s0, double cfo, int n_data, bool dup);

double peak_ratio(const Grid& g, std::span<const double> x, std::int64_t s0, double cfo, bool mid = false);

struct Lock {
    const Spec* spec;
    int n_data;
    bool dup;
    std::int64_t start;
    double cfo, score, header_score, header_margin;
    std::int64_t end, header_end;
    // find's candidate (wait): a front whose first header copy fell short. Only
    // start (the detected one), cfo, score and header_end (where x must reach
    // for late_lock) are set; spec is nullptr.
    bool wait = false;
    int copies = 0;  // header copies the lock was read from (an early lock: 1)
};
// threshold: nullopt = SYNC_THRESHOLD plus the lock floors (lock_llr, hdr_margin). hi: nullopt = len(x).
// wait: a candidate instead of nullopt while the second header copy is still arriving.
std::optional<Lock> find(const Grid& g, std::span<const double> x, std::optional<double> threshold = std::nullopt,
                         double reach_hz = 150.0, bool front_only = true, std::int64_t lo = 0,
                         std::optional<std::int64_t> hi = std::nullopt, bool wait = false,
                         const CoarseTrack* track = nullptr, std::int64_t x0 = 0);
// find's candidate (x holding its header_end) -> its lock from both header copies
// (lock_llr2, hdr_margin2), or nullopt.
std::optional<Lock> late_lock(const Grid& g, std::span<const double> x, const Lock& cand);
// A burst whose front was missed, from its mid block and both header copies
// (cpm.find_mid): the mid block's start in [lo, hi). hi: nullopt = len(x).
std::optional<Lock> find_mid(const Grid& g, std::span<const double> x, std::int64_t lo = 0,
                             std::optional<std::int64_t> hi = std::nullopt, const CoarseTrack* track = nullptr,
                             std::int64_t x0 = 0);
// cpm.confirm: a lock's second look, tone_llr over its mid block and second
// header copy at the locked word, unfloored; under CONFIRM_LLR the burst is
// not there (receiver::Receiver lets an early lock go). confirm_end: one past
// its second header copy's last sample.
inline constexpr double CONFIRM_LLR = -2.0;
double confirm(const Grid& g, std::span<const double> x, const Lock& lock);
std::int64_t confirm_end(const Grid& g, const Lock& lock);
// Samples from a burst's start to the end of its first `copies` header copies,
// the latest any word puts them.
std::int64_t hdr_span(const Grid& g, int copies);
// cpm.prime_lock (docs/arq.md §3a): a burst of grid g starting in [lo, hi], its CFO near `cfo`, any valid
// word, scored over every row the word fixes; nullopt under PRIME_LLR. x must hold the second header copy
// of a start at hi.
inline constexpr double PRIME_LLR = 16.0;
inline constexpr int PRIME_STEP = 4;
inline constexpr double PRIME_CFO[] = {-1.0 / 16, 0.0, 1.0 / 16};
std::optional<Lock> prime_lock(const Grid& g, std::span<const double> x, std::int64_t lo, std::int64_t hi, double cfo);

// cpm.measure: arq/phy's measure makes the effective-MI features (arq/predictor's)
// of `snr`, per symbol.
struct Measure {
    double snr_est, spread_est, frames;
    std::vector<double> snr;
};
Measure measure(const Grid& g, const Mat<double>& E, int n_sym);

// cpm.receive: an early lock's burst, whole in x (same sample origin) ->
// the per-slot soft bits and tone energies data2g.arq.phy reads.
// A data slot's mask-free decode made while its burst arrived (arq::cpm_pass,
// phy.cpm_pass): candidates as codes::decode_raw's row, usable, and the
// decoded codeword's certain posterior (empty: it failed).
struct Pass {
    std::vector<std::uint8_t> cands, usable;
    std::vector<double> post;
};

struct Received {
    const Spec* spec;
    int n_cw, n_ctl_slots;  // slots, control included; control slots
    bool dup;
    std::vector<std::vector<double>> soft;
    Mat<double> E;
    double cfo;
    std::int64_t preamble_start, header_end;
    // (burst rows, m) complex tone bins at the refined start and CFO (coherent DD; a wide grid's sub-windows,
    // L rows a symbol; empty where the grid is not coherent())
    Mat<std::complex<double>> Z;
    std::map<int, Pass> passes;   // data slots' passes a streaming receiver made (arq::CpmStream); ModemRx makes the rest
};
// cpm.refine: a lock's start and CFO by the likelihood of the burst's first refine_symbols symbols.
std::pair<std::int64_t, double> refine(const Grid& g, std::span<const double> x, std::int64_t s0, double cfo,
                                       const Spec& spec, int n_data, bool dup);
// The control slots' soft bits are coherent (dd_soft of dd_metric with no
// posteriors) unless coh_ctl() is off.
Received receive(std::span<const double> x, const Lock& lock);
// receive with refine's result `at` (start, cfo) already made.
Received receive_refined(std::span<const double> x, const Lock& lock, std::pair<std::int64_t, double> at);

// --- coherent detection (cpm.py's section of that name) ---
// h = 1, continuous phase: every symbol starts at the same carrier phase. A
// channel estimate per row from its neighbours' known or decided tones makes
// the metric coherent; a poor estimate falls back to the square law.
inline constexpr double FLOOR_SIGMAS = 6.0;  // cpm.FLOOR_SIGMAS: tone_floors' bar over the common N0
inline constexpr int COH_W = 8;      // coh_metric's window for slots (+- rows, the row left out)
inline constexpr int COH_HDR_W = 4;  // the header's
// cpm._coh_w: coh_metric's window for grid g, in its rows (a wide grid's sub-windows: cpm.WIDE_COH_W).
int coh_w(const Grid& g, bool hdr = false);
// cpm.coherent: the coherent paths (lock, control, DD) read grid g: every on-air grid, a wide grid unless it
// drops sub-windows.
inline bool coherent(const Grid& g) { return g.L == 1 || g.guard == 0; }
// cpm.COH_LOCK: the grid's coherent lock gates (score one copy, both; margin
// one copy, both), nullopt for an uncalibrated grid or DATA2G_CPM_COH_LOCK != 1 (read once).
std::optional<std::array<double, 4>> coh_lock(const Grid& g);
// cpm.COH_PREGATE: the coherent reading runs only where the noncoherent score reaches this (one copy, both).
inline constexpr double COH_PREGATE[] = {12.0, 10.0};
// cpm.COH_CTL (DATA2G_CPM_COH_CTL, read once).
bool coh_ctl();

// (n_sym, m) complex tone bins from `start`, CFO removed (energies before the square). A wide grid:
// (n_sym L, m), each sub-window's, a symbol's L rows in turn.
Mat<std::complex<double>> bins(const Grid& g, std::span<const double> x, std::int64_t start, int n_sym, double cfo);
// (n, m) bins of consecutive rows and P (n, m) the tones' probabilities -> (n, m) coherent tone log-likelihoods.
// L > 1 (a wide grid): Z's rows are sub-windows, L a symbol, P a row per symbol, the result (n / L, m).
Mat<double> coh_metric(const Mat<std::complex<double>>& Z, const Mat<double>& P, int W, double n0, int L = 1);
// cpm._n0: the noise level of energies E (all but each row's strongest, over the median of a noise energy:
// exponential ln 2, a wide grid's k-sum Grid::noise_median).
double noise_n0(const Mat<double>& E, double noise_median = std::numbers::ln2);
// cpm._coh_run_llr: per-row tone_llr of known tones t over a run of consecutive rows (coh_metric, n0).
std::vector<double> coh_run_llr(const Grid& g, const Mat<std::complex<double>>& Z, std::span<const int> t, double n0);
// read_header on coh_metric (cpm.coh_read_header).
Header coh_read_header(const Grid& g, std::span<const double> x, std::int64_t s0, double cfo, int copies = 2);
// Per slot (control first), its rows in the burst.
std::vector<std::vector<int>> slot_rows(const Grid& g, int n_data, bool dup);
// cpm.dd_metric: (slot's rows, m) coh_metric over the burst rows from the
// slot's first less COH_W to its last plus COH_W; `post` {data slot: GF
// log-probabilities as sent, decode_posterior's}. It reads r.Z on those rows
// only, so a streaming receiver can run it once they are in.
using Posts = std::map<int, std::vector<double>>;
Mat<double> dd_metric(const Received& r, const Posts& post, int slot);
// cpm.dd_soft: a slot's soft values from its dd_metric (control: bit LLRs, data: symbol log-likelihoods).
std::vector<double> dd_soft(const Received& r, const Mat<double>& metric, int slot);

// A pending lock's receive, made as its burst arrives: refine once its
// window is in (refine_symbols, or the whole burst), then each data slot's
// pass (`pass`, arq::cpm_pass) once the slot and coh_w rows more are in,
// each on a Received of the audio so far (a slot's rows and its metric's
// window are complete then, and rows are computed alone, so it is bitwise
// the pass made on the whole burst). finish(): the burst's Received, with
// the passes made, as receive(x, lock) gives it.
// Not thread-safe: feed and finish from one thread at a time, in order.
using PassFn = std::function<Pass(const Received&, int slot, const Posts& earlier)>;
class Stream {
public:
    // x: the receiver's buffer at the lock (lock in its coordinates); pass: none = refine early only.
    Stream(std::vector<double> x, const Lock& lock, PassFn pass);
    void feed(std::span<const double> more);  // the audio after it, in order
    Received finish();  // once the burst is whole in what was fed
    int passes_made() const { return static_cast<int>(passes_.size()); }

private:
    void advance();
    std::vector<double> x_;
    Lock lock_;
    PassFn pass_;
    std::optional<std::pair<std::int64_t, double>> at_;
    std::int64_t refine_need_;  // x_ size refine needs
    std::vector<std::int64_t> slot_need_;  // per data slot (index 0 = first data slot): x_ size its pass needs
    std::map<int, Pass> passes_;
    Posts earlier_;
};

}  // namespace data2g::cpm
