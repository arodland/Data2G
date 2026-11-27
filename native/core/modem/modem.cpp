#include "modem/modem.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>

#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>

#include "codes/codes.hpp"
#include "constellation/constellation.hpp"
#include "dsp/dsp.hpp"
#include "modem/scattered.hpp"
#include "waveform/dsp.hpp"
#include "waveform/ofdm.hpp"

namespace data2g::modem {

using namespace config;
using std::int64_t;
using std::size_t;

namespace {

constexpr double PI = std::numbers::pi;
constexpr int HB = HEADER_BACKOFF;

const constellation::Constellation& qpsk() {
    static const auto* c = constellation::find("gray-qam4");
    return *c;
}

const constellation::Constellation& points_of(const Spec& spec) {
    const auto* c = constellation::find(spec.constellation);
    if (!c) throw std::out_of_range("no constellation " + std::string(spec.constellation));
    return *c;
}

std::vector<double> bb_of(std::string_view name) { return equalizer::bb(band(name)); }

double sq_abs(cd x) {
    const double a = std::abs(x);  // np.abs(x) ** 2
    return a * a;
}

// x / pilot, carrier by carrier
std::vector<cd> over_pilot(std::vector<cd> x, const waveform::Band& b) {
    for (size_t k = 0; k < x.size(); ++k) x[k] /= b.pilot[k];
    return x;
}

// np.sum(a * conj(b)) over carriers
cd dot_conj(std::span<const cd> a, std::span<const cd> b) {
    std::vector<cd> p(a.size());
    for (size_t k = 0; k < a.size(); ++k) p[k] = a[k] * std::conj(b[k]);
    return dsp::pairwise_sum(std::span<const cd>(p));
}

double energy(std::span<const cd> a) {
    std::vector<double> p(a.size());
    for (size_t k = 0; k < a.size(); ++k) p[k] = sq_abs(a[k]);
    return dsp::pairwise_sum(std::span<const double>(p));
}

// np.median of a list
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

// Python's floor modulo
int64_t pymod(int64_t a, int64_t b) { return ((a % b) + b) % b; }

int64_t clamp_slice(int64_t i, int64_t n) {
    if (i < 0) i += n;
    return std::clamp<int64_t>(i, 0, n);
}

// The header correlation as a Walsh-Hadamard transform. A valid word is
// v << 6 | crc6(v) for a 10-bit v, and the CRC is affine in v, so coded bit
// j of word(v) is parity(v & mask[j]) ^ flip[j]: the correlation of soft
// bits x with every word's signs is the 10-bit WHT of
// X[u] = sum over j with mask[j] = u of +-x[j]. ~10k operations for all
// 1024 values instead of (valid words) x N multiply-adds.
struct Wht {
    std::vector<std::uint16_t> mask;
    std::vector<double> sign;  // (-1)^flip[j]
};

const Wht& wht(std::string_view band) {
    static const auto tables = [] {
        std::array<Wht, HEADER_CODES.size()> out;
        const auto word = [](int v) { return v << 6 | crc6(v); };
        for (size_t b = 0; b < out.size(); ++b)
            for (const auto col : HEADER_CODES[b].second) {
                const auto parity = [col](int w) { return std::popcount(static_cast<unsigned>(w & col)) & 1; };
                std::uint16_t m = 0;
                for (int i = 0; i < 10; ++i) m |= static_cast<std::uint16_t>(parity(word(1 << i) ^ word(0)) << i);
                out[b].mask.push_back(m);
                out[b].sign.push_back(parity(word(0)) ? -1.0 : 1.0);
            }
        return out;
    }();
    for (size_t b = 0; b < HEADER_CODES.size(); ++b)
        if (HEADER_CODES[b].first == band) return tables[b];
    throw std::out_of_range("no header code for band " + std::string(band));
}

// valid_words' 10-bit values (word >> 6), in order
std::vector<int> valid_values(std::string_view band, const Accept* accept) {
    const int b = cw_bits(band);
    if (b < 5) throw std::logic_error("valid_values: lim covers 1024 >> 5 submode indices");
    std::array<int, 32> lim{};  // submode index -> codewords taken (0: none)
    if (accept) {
        for (const auto& [s, cw] : accept->max_cw)
            if (s->sync_band == band) lim[static_cast<size_t>(s->index)] = cw;
    } else {
        for (const auto& s : SUBMODES)
            if (s.sync_band == band) lim[static_cast<size_t>(s.index)] = max_codewords(band);
    }
    std::vector<int> out;
    for (int v = 0; v < 1024; ++v)
        if ((v & ((1 << b) - 1)) < lim[static_cast<size_t>(v >> b)]) out.push_back(v);
    return out;
}

// header_corr over the valid values `vs`
std::vector<float> corr_of(std::span<const double> soft, std::string_view band, std::span<const int> vs) {
    const auto& t = wht(band);
    if (soft.size() != t.mask.size()) throw std::invalid_argument("header: soft bits and the band's code differ in length");
    // soft.astype(np.float32), summed in double: each sum is the float32
    // correlation's exact value to far under a float32 ulp, in any order
    std::array<double, 1024> X{};
    for (size_t j = 0; j < soft.size(); ++j) X[t.mask[j]] += t.sign[j] * static_cast<float>(soft[j]);
    for (size_t h = 1; h < X.size(); h *= 2)
        for (size_t i = 0; i < X.size(); i += 2 * h)
            for (size_t k = i; k < i + h; ++k) {
                const double a = X[k], b = X[k + h];
                X[k] = a + b;
                X[k + h] = a - b;
            }
    std::vector<float> out(vs.size());
    for (size_t i = 0; i < vs.size(); ++i) out[i] = static_cast<float>(X[static_cast<size_t>(vs[i])]);
    return out;
}

double floor_of(std::string_view band, const Accept* accept) {
    return std::max(header_min_score(band), accept ? accept->min_score : 0.0);
}

std::vector<std::string_view> search_bands(std::span<const std::string_view> bands, const Accept* accept) {
    if (!bands.empty()) return {bands.begin(), bands.end()};
    if (accept) return accept->bands();
    return {SYNC_BANDS.begin(), SYNC_BANDS.end()};
}

// (1 - a) h0 + a h1 per carrier
std::vector<cd> lerp(std::span<const cd> h0, std::span<const cd> h1, double a) {
    std::vector<cd> out(h0.size());
    for (size_t k = 0; k < out.size(); ++k) out[k] = (1 - a) * h0[k] + a * h1[k];
    return out;
}

// LLRs of rows ys against channel rows hs (unit variance), QPSK
std::vector<double> qpsk_llr(const Mat<cd>& ys, const Mat<cd>& hs) {
    const std::vector<double> ones(ys.data.size(), 1.0);
    return constellation::llr(ys.data, hs.data, ones, qpsk());
}

Mat<cd> rows_of(const std::vector<std::vector<cd>>& v) {
    Mat<cd> m(v.size(), v.empty() ? 0 : v[0].size());
    for (size_t i = 0; i < v.size(); ++i) std::copy(v[i].begin(), v[i].end(), m[i]);
    return m;
}

// Python slice x[lo:hi] as a span
template <typename T>
std::span<const T> pyslice(std::span<const T> x, int64_t lo, int64_t hi) {
    const auto n = static_cast<int64_t>(x.size());
    const int64_t a = clamp_slice(lo, n), b = clamp_slice(hi, n);
    return b > a ? x.subspan(static_cast<size_t>(a), static_cast<size_t>(b - a)) : std::span<const T>();
}

}  // namespace

double header_min_score(std::string_view band) {
    for (const auto& [b, v] : HEADER_MIN_SCORE)
        if (b == band) return v;
    throw std::out_of_range("no HEADER_MIN_SCORE for band " + std::string(band));
}

double pilot_noise(std::string_view band) {
    for (const auto& [b, v] : PILOT_NOISE)
        if (b == band) return v;
    throw std::out_of_range("no PILOT_NOISE for band " + std::string(band));
}

double pilot_pattern(std::string_view band) {
    for (const auto& [b, v] : PILOT_PATTERN)
        if (b == band) return v;
    throw std::out_of_range("no PILOT_PATTERN for band " + std::string(band));
}

Accept Accept::of(std::span<const std::string_view> names, std::optional<double> max_secs, double min_score) {
    Accept a;
    a.min_score = min_score;
    std::vector<std::string_view> all;
    if (names.empty())
        for (const auto& s : SUBMODES) all.push_back(s.name);
    else
        all.assign(names.begin(), names.end());
    for (auto n : all) {
        const Spec* s = config::submode(n);
        if (!s) throw std::out_of_range("no submode " + std::string(n));
        int cw = max_codewords(s->sync_band);
        if (max_secs) {
            int fixed = preamble_samples(band(s->sync_band)) + header_samples(s->sync_band) + NSYM;
            fixed += FRAME_SAMPLES * copies(s->sync_band);
            cw = std::min<int>(cw, static_cast<int>(std::floor((*max_secs * FS - fixed) / (s->frames_per_cw * FRAME_SAMPLES))));
        }
        if (cw >= 1) a.max_cw.emplace_back(s, cw);
    }
    return a;
}

std::vector<std::string_view> Accept::bands() const {
    std::vector<std::string_view> out;
    for (const auto& [s, cw] : max_cw)
        if (std::find(out.begin(), out.end(), s->sync_band) == out.end()) out.push_back(s->sync_band);
    std::sort(out.begin(), out.end());
    return out;
}

// --- header -----------------------------------------------------------------

int crc6(int v) {
    int reg = (PROTOCOL_VERSION ^ PROFILE_TAG) & 0x3F;
    for (int i = 9; i >= 0; --i) {
        const int fb = ((reg >> 5) & 1) ^ ((v >> i) & 1);
        reg = ((reg << 1) & 0x3F) ^ (fb ? 0x3 : 0);
    }
    return reg;
}

std::span<const std::uint16_t> header_cols(std::string_view band) {
    for (const auto& [b, cols] : HEADER_CODES)
        if (b == band) return cols;
    throw std::out_of_range("no header code for band " + std::string(band));
}

std::vector<std::uint8_t> codeword(int word, std::string_view band) {
    const auto cols = header_cols(band);
    std::vector<std::uint8_t> out(cols.size());
    for (size_t j = 0; j < cols.size(); ++j)
        out[j] = static_cast<std::uint8_t>(std::popcount(static_cast<unsigned>(word & cols[j])) & 1);
    return out;
}

std::vector<std::uint8_t> header_bits(int submode, int n_cw, std::string_view band) {
    const int b = cw_bits(band);
    if (n_cw < 1 || n_cw > 1 << b)
        throw std::invalid_argument("1.." + std::to_string(1 << b) + " codewords per burst on " + std::string(band));
    if (submode < 0 || submode >= 1 << (10 - b))
        throw std::invalid_argument("submode index " + std::to_string(submode) + " does not fit " + std::string(band) +
                                    "'s header");
    const int v = submode << b | (n_cw - 1);
    return codeword(v << 6 | crc6(v), band);
}

std::vector<int> valid_words(std::string_view band, const Accept* accept) {
    auto out = valid_values(band, accept);
    for (int& v : out) v = v << 6 | crc6(v);
    return out;
}

std::vector<float> header_corr(std::span<const double> soft, std::string_view band, const Accept* accept) {
    return corr_of(soft, band, valid_values(band, accept));
}

Header decode_header(std::span<const double> soft, std::string_view band, const Accept* accept) {
    const auto vs = valid_values(band, accept);
    if (vs.empty()) throw std::invalid_argument("decode_header: no valid words");
    const auto corr = corr_of(soft, band, vs);
    size_t i = 0;  // np.argmax: the first maximum (a NaN, if any, wins)
    for (size_t k = 1; k < corr.size() && !std::isnan(corr[i]); ++k)
        if (corr[k] > corr[i] || std::isnan(corr[k])) i = k;
    std::vector<double> sq(soft.size());
    for (size_t k = 0; k < soft.size(); ++k) sq[k] = soft[k] * soft[k];
    const double norm = std::sqrt(dsp::pairwise_sum(std::span<const double>(sq)) * static_cast<double>(soft.size()));
    Header h;
    h.word = vs[i] << 6 | crc6(vs[i]);
    h.score = static_cast<double>(corr[i]) / (norm + 1e-12);
    const int v = h.word >> 6, b = cw_bits(band);
    h.spec = by_index(band, v >> b);
    if (!h.spec) throw std::logic_error("valid word without a submode");
    h.n_cw = (v & ((1 << b) - 1)) + 1;
    return h;
}

// --- transmit -----------------------------------------------------------------

std::vector<int64_t> ace_cells(const Spec& spec, int n_f) {
    const auto kc = copy_frame(spec.sync_band, n_f);
    const int64_t f0 = LEADIN_SAMPLES + preamble_samples(band(spec.sync_band)) + header_samples(spec.sync_band);
    std::vector<int64_t> out;
    for (int f = 0; f < n_f + kc.has_value(); ++f) {
        if (kc && f == *kc) continue;
        for (int s = !sc_spec(spec, n_f); s < SYMS_PER_FRAME; ++s)  // scattered pilots: the pilots among the cells
            out.push_back(f0 + static_cast<int64_t>(f * SYMS_PER_FRAME + s) * NSYM);
    }
    return out;
}

std::vector<double> burst_waveform(const Mat<cd>& data_in, const Spec& spec) {
    const auto& b = waveform::band(spec.band);
    const auto& sb = waveform::band(spec.sync_band);
    const size_t nc = static_cast<size_t>(b.nc()), snc = static_cast<size_t>(sb.nc());
    if (data_in.cols != nc || data_in.rows % DATA_SYMS_PER_FRAME)
        throw std::invalid_argument("burst_waveform: need (n_f * 5, nc) data symbols");
    int n_f = static_cast<int>(data_in.rows / DATA_SYMS_PER_FRAME);
    const int n_cw = n_f / spec.frames_per_cw;
    const auto hbits = header_bits(spec.index, n_cw, spec.sync_band);
    const auto hdr = constellation::modulate(hbits, qpsk());  // (header_syms * snc)
    const int hs = sb.spec->header_syms;
    const auto kc = copy_frame(spec.sync_band, n_f);
    if (kc && snc != nc) throw std::logic_error("header copy on a band other than its sync band");
    const bool scb = sc_band(spec.band);
    const auto open = copy_open(b);  // (a scattered-pilot band's copy frame opens and closes on it)
    Mat<cd> syms;
    if (sc_spec(spec, n_f)) {
        // each frame's pilots among its cells (sc_frame_cells); the copy's frame: copy_open, the 4 header symbols, copy_close
        const size_t n_air = static_cast<size_t>(n_f + kc.has_value()), per = DATA_SYMS_PER_FRAME * nc;
        syms = Mat<cd>(n_air * SYMS_PER_FRAME + 1, nc);
        for (size_t a = 0, f = 0; a < n_air; ++a) {
            cd* fr = syms[a * SYMS_PER_FRAME];
            if (kc && a == static_cast<size_t>(*kc)) {
                std::copy(open.begin(), open.end(), fr);
                std::copy(hdr.begin(), hdr.begin() + static_cast<std::ptrdiff_t>(hs * snc), fr + nc);
                std::copy(open.begin(), open.end(), fr + (SYMS_PER_FRAME - 1) * nc);
                continue;
            }
            const auto& cells = sc_frame_cells(b.nc(), static_cast<int>(a));
            for (size_t i = 0; i < nc; ++i) fr[cells.P[i]] = b.pilot[static_cast<size_t>(cells.PK[i])];
            const cd* d = data_in[f * DATA_SYMS_PER_FRAME];
            for (size_t j = 0; j < per; ++j) fr[cells.Dc[j]] = d[j];
            ++f;
        }
        std::copy(b.pilot.begin(), b.pilot.end(), syms[syms.rows - 1]);
    } else {
        Mat<cd> data = data_in;
        if (kc) {  // the header copy: a frame of its own (data band = sync band here)
            Mat<cd> d2(data.rows + DATA_SYMS_PER_FRAME, nc);
            const size_t at = static_cast<size_t>(*kc) * DATA_SYMS_PER_FRAME;
            std::copy(data.data.begin(), data.data.begin() + static_cast<std::ptrdiff_t>(at * nc), d2.data.begin());
            for (int s = 0; s < DATA_SYMS_PER_FRAME; ++s) {
                // a scattered-pilot band's copy frame ends on copy_close whatever the submode (copy_llr)
                if (scb && s == hs) {
                    std::copy(open.begin(), open.end(), d2[at + static_cast<size_t>(s)]);
                    continue;
                }
                const size_t src = static_cast<size_t>(s < hs ? s : s - hs);
                std::copy(hdr.begin() + static_cast<std::ptrdiff_t>(src * snc),
                          hdr.begin() + static_cast<std::ptrdiff_t>((src + 1) * snc), d2[at + static_cast<size_t>(s)]);
            }
            std::copy(data.data.begin() + static_cast<std::ptrdiff_t>(at * nc), data.data.end(),
                      d2.data.begin() + static_cast<std::ptrdiff_t>((at + DATA_SYMS_PER_FRAME) * nc));
            data = std::move(d2);
            ++n_f;
        }
        syms = Mat<cd>(static_cast<size_t>(n_f) * SYMS_PER_FRAME + 1, nc);
        for (size_t r = 0; r < syms.rows; ++r) {
            if (r % SYMS_PER_FRAME == 0) {
                std::copy(b.pilot.begin(), b.pilot.end(), syms[r]);
            } else {
                const size_t f = r / SYMS_PER_FRAME, s = r % SYMS_PER_FRAME - 1;
                std::copy(data[f * DATA_SYMS_PER_FRAME + s], data[f * DATA_SYMS_PER_FRAME + s] + nc, syms[r]);
            }
        }
        if (kc && scb) std::copy(open.begin(), open.end(), syms[static_cast<size_t>(*kc) * SYMS_PER_FRAME]);
    }
    const auto layout = header_layout(spec.sync_band);
    Mat<cd> hsyms(layout.size(), snc);
    size_t d = 0;
    for (size_t r = 0; r < layout.size(); ++r) {
        if (layout[r]) {
            std::copy(sb.pilot.begin(), sb.pilot.end(), hsyms[r]);
        } else {
            std::copy(hdr.begin() + static_cast<std::ptrdiff_t>(d * snc), hdr.begin() + static_cast<std::ptrdiff_t>((d + 1) * snc),
                      hsyms[r]);
            ++d;
        }
    }
    std::vector<double> out(LEADIN_SAMPLES, 0.0);
    const auto pre = sb.preamble_waveform();
    const auto hw = sb.modulate_symbols(hsyms);
    const auto dw = b.modulate_symbols(syms);
    out.insert(out.end(), pre.begin(), pre.end());
    out.insert(out.end(), hw.begin(), hw.end());
    out.insert(out.end(), dw.begin(), dw.end());
    out.insert(out.end(), LEADOUT_SAMPLES, 0.0);
    return out;
}

namespace {

// modem.ace_projector: each data cell of the clipped burst moved into its
// point's region (the point scaled by the carrier's clip gain).
waveform::Projector ace_projector(const Spec& spec, const Mat<cd>& data, std::vector<cd> dirs) {
    const auto& b = waveform::band(spec.band);
    const size_t nc = static_cast<size_t>(b.nc());
    const int n_f = static_cast<int>(data.rows / DATA_SYMS_PER_FRAME);
    auto cells = std::make_shared<std::vector<int64_t>>(ace_cells(spec, n_f));
    auto X = std::make_shared<Mat<cd>>(data);
    auto D = std::make_shared<std::vector<cd>>(std::move(dirs));
    auto hold = std::make_shared<std::vector<char>>();  // scattered pilots among the cells: put back on their points
    if (sc_spec(spec, n_f)) {
        const auto kc = copy_frame(spec.sync_band, n_f);
        std::vector<int> norm;
        for (int a = 0; a < n_f + kc.has_value(); ++a)
            if (!kc || a != *kc) norm.push_back(a);
        const size_t per = DATA_SYMS_PER_FRAME * nc, all = SYMS_PER_FRAME * nc;
        Mat<cd> X6(static_cast<size_t>(n_f) * SYMS_PER_FRAME, nc);
        std::vector<cd> D6(X6.data.size() * 2, 0.0);
        hold->assign(X6.data.size(), 0);
        for (size_t f = 0; f < static_cast<size_t>(n_f); ++f) {
            const auto& sc = sc_frame_cells(b.nc(), norm[f]);
            for (size_t i = 0; i < nc; ++i) {
                X6.data[f * all + static_cast<size_t>(sc.P[i])] = b.pilot[static_cast<size_t>(sc.PK[i])];
                (*hold)[f * all + static_cast<size_t>(sc.P[i])] = 1;
            }
            for (size_t j = 0; j < per; ++j) {
                const size_t c = f * all + static_cast<size_t>(sc.Dc[j]);
                X6.data[c] = data.data[f * per + j];
                D6[2 * c] = (*D)[2 * (f * per + j)];
                D6[2 * c + 1] = (*D)[2 * (f * per + j) + 1];
            }
        }
        *X = std::move(X6);
        *D = std::move(D6);
    }
    constexpr double restore = 1.0;  // scattered pilots restored onto their constellation points after ACE clipping
    return [&b, nc, cells, X, D, hold, restore](std::span<const double> x) {
        const size_t n = cells->size();
        Mat<cd> got(n, nc);
        for (size_t i = 0; i < n; ++i) {
            const double* w = x.data() + (*cells)[i] + NCP;
            for (size_t k = 0; k < nc; ++k) {
                cd acc = 0.0;
                for (size_t t = 0; t < static_cast<size_t>(M); ++t) acc += w[t] * std::conj(b.mod[NCP + t][k]);
                got[i][k] = (2.0 / M) * acc;
            }
        }
        std::vector<double> den(nc, 0.0);
        std::vector<cd> numc(nc, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < nc; ++k) {
                numc[k] += std::conj((*X)[i][k]) * got[i][k];
                den[k] += sq_abs((*X)[i][k]);
            }
        Mat<cd> want(n, nc);
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < nc; ++k) want[i][k] = (numc[k].real() / den[k]) * (*X)[i][k];
        const auto proj = constellation::ace_project(got.data, want.data, *D);
        std::vector<double> out(x.begin(), x.end());
        std::vector<cd> delta(nc);
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = 0; k < nc; ++k)
                delta[k] = !hold->empty() && (*hold)[i * nc + k] ? restore * (want[i][k] - got[i][k])
                                                                  : proj[i * nc + k] - got[i][k];
            double* o = out.data() + (*cells)[i];
            for (size_t t = 0; t < static_cast<size_t>(NSYM); ++t) {
                cd acc = 0.0;
                for (size_t k = 0; k < nc; ++k) acc += delta[k] * b.mod[t][k];
                o[t] += acc.real();
            }
        }
        return out;
    };
}

}  // namespace

namespace {
std::string tx_path_ = "mic";
std::mutex tx_path_mu_;
}  // namespace

std::string_view tx_path() {
    std::lock_guard lk(tx_path_mu_);
    return tx_path_ == "flat" ? "flat" : "mic";  // a view of a literal: safe after a later set
}

void set_tx_path(std::string_view path) {
    if (path != "mic" && path != "flat")
        throw std::invalid_argument("tx_path: '" + std::string(path) + "', not mic or flat");
    std::lock_guard lk(tx_path_mu_);
    tx_path_ = std::string(path);
}

std::vector<double> modulate_bits(std::span<const std::uint8_t> bits, const Spec& spec, std::string_view path) {
    if (path.empty()) path = tx_path();
    if (path != "mic" && path != "flat")
        throw std::invalid_argument("tx_path: '" + std::string(path) + "', not mic or flat");
    const bool fm = config::PROFILE == "fm";
    const auto& b = waveform::band(spec.band);
    const auto& pts = points_of(spec);
    const size_t nc = static_cast<size_t>(b.nc());
    const auto sym = constellation::modulate(bits, pts);
    if (sym.size() % (DATA_SYMS_PER_FRAME * nc)) throw std::invalid_argument("modulate_bits: not whole frames");
    Mat<cd> data(sym.size() / nc, nc);
    data.data = sym;
    const auto x = burst_waveform(data, spec);
    waveform::Projector project;
    if (!spec.ace.empty()) {
        const int m = pts.m;
        std::vector<cd> dirs(2 * sym.size());
        for (size_t s = 0; s < sym.size(); ++s) {
            size_t idx = 0;
            for (int j = 0; j < m; ++j) idx = idx << 1 | bits[s * static_cast<size_t>(m) + static_cast<size_t>(j)];
            dirs[2 * s] = pts.ace[2 * idx];
            dirs[2 * s + 1] = pts.ace[2 * idx + 1];
        }
        project = ace_projector(spec, data, std::move(dirs));
    }
    // FM flat path: sent unclipped from 16-QAM up (a headroom no burst reaches)
    const double headroom = fm && path == "flat" && pts.m >= config::FM_FLAT_NOCLIP_BITS ? 40.0 : spec.headroom;
    auto y = waveform::tx_condition(x, headroom, b.spec->clip_overshoot, LEADIN_SAMPLES, x.size() - LEADOUT_SAMPLES,
                                    waveform::band(spec.sync_band).tx_bandpass(), project, spec.ace);
    if (!fm) return y;
    if (path == "mic") {
        const double g = config::FM_MIC_GAIN[static_cast<size_t>(pts.m)];
        for (auto& v : y) v *= g;
        return y;
    }
    // the tilt: np.convolve(y, h, "same") (odd h: centred), then the rms over the burst kept
    std::span<const double> h;
    for (const auto& [name, taps] : TX_TILT)
        if (name == spec.sync_band) h = taps;
    if (h.empty()) throw std::logic_error("modulate_bits: no TX tilt for band " + std::string(spec.sync_band));
    const auto n = static_cast<std::ptrdiff_t>(y.size()), L = static_cast<std::ptrdiff_t>(h.size()), c = (L - 1) / 2;
    std::vector<double> t(y.size(), 0.0);
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        double acc = 0.0;
        for (std::ptrdiff_t j = 0; j < L; ++j) {
            const std::ptrdiff_t k = i + c - j;
            if (k >= 0 && k < n) acc += h[static_cast<size_t>(j)] * y[static_cast<size_t>(k)];
        }
        t[static_cast<size_t>(i)] = acc;
    }
    double py = 0.0, pt = 0.0;
    for (size_t i = LEADIN_SAMPLES; i < y.size() - LEADOUT_SAMPLES; ++i) py += y[i] * y[i], pt += t[i] * t[i];
    const double g = std::sqrt(py / pt);
    for (auto& v : t) v *= g;
    return t;
}

std::vector<double> modulate(std::span<const std::vector<std::uint8_t>> payloads, const Spec& spec,
                             std::span<const int> rvs, std::string_view tx_path) {
    if (payloads.empty() || payloads.size() > static_cast<size_t>(max_codewords(spec.sync_band)))
        throw std::invalid_argument("1.." + std::to_string(max_codewords(spec.sync_band)) + " codewords per burst, got " +
                                    std::to_string(payloads.size()));
    if (!rvs.empty() && rvs.size() != payloads.size()) throw std::invalid_argument("one rv per codeword");
    const auto& cs = codes::spec(spec);
    std::vector<std::uint8_t> coded;
    for (size_t i = 0; i < payloads.size(); ++i) {
        const auto c = codes::encode(cs, payloads[i], rvs.empty() ? 0 : rvs[i], 0, static_cast<int>(i));
        coded.insert(coded.end(), c.begin(), c.end());
    }
    const auto bits = codes::spread<std::uint8_t>(coded, static_cast<int>(payloads.size()), spec.bits_per_cu,
                                                  spread_group(spec));
    return modulate_bits(bits, spec, tx_path);
}

// --- receive ------------------------------------------------------------------

double bin_phase_step(std::span<const cd> h) {
    if (h.size() < 2) return std::arg(cd(0.0));
    return std::arg(dot_conj(h.subspan(1), h.first(h.size() - 1)));
}

namespace {

// modem._coherence: |<ref, h>| / (|ref| |h|)
double coherence(std::span<const cd> h, std::span<const cd> ref) {
    cd acc{};
    double eh = 0.0, er = 0.0;
    for (size_t k = 0; k < h.size(); ++k) {
        acc += h[k] * std::conj(ref[k]);
        eh += std::norm(h[k]);
        er += std::norm(ref[k]);
    }
    return std::abs(acc) / (std::sqrt(eh) * std::sqrt(er) + 1e-30);
}

// modem._find_slip. Every offset's pilot by a sliding DFT (one update per
// carrier a sample, not a window's demod): cheap enough for a Pi.
int64_t find_slip(std::span<const cd> z, int64_t p, int n_left, std::span<const cd> prev, const waveform::Band& b) {
    if (coherence(over_pilot(b.demod_window(z, p + NCP, DEMOD_BACKOFF), b), prev) >= SLIP_TRIGGER) return 0;
    const auto zlen = static_cast<int64_t>(z.size());
    const auto at_z = [&](int64_t i) { return i < zlen ? z[static_cast<size_t>(i)] : cd{}; };
    const int64_t lo = std::max<int64_t>(-SLIP_MAX, -p);
    const auto n_off = static_cast<size_t>(SLIP_MAX - lo + 1);
    const size_t nc = static_cast<size_t>(b.nc());
    std::vector<cd> rot(nc), x, h(nc);
    for (size_t k = 0; k < nc; ++k) rot[k] = waveform::phasor(b.bb[k], 1);
    std::vector<double> score(n_off, 0.0);
    std::vector<std::vector<double>> at;
    for (int j = 0; j <= std::min(2, n_left); ++j) {
        const int64_t a = p + j * FRAME_SAMPLES + NCP - DEMOD_BACKOFF + lo;  // offset lo's window
        x = b.demod_window(z, a, 0);
        auto& c = at.emplace_back(n_off);
        for (size_t i = 0; i < n_off; ++i) {
            if (i) {
                const int64_t s = a + static_cast<int64_t>(i) - 1;
                const cd d = (2.0 / M) * (at_z(s + M) - at_z(s));
                for (size_t k = 0; k < nc; ++k) x[k] = (x[k] + d) * rot[k];
            }
            for (size_t k = 0; k < nc; ++k) h[k] = x[k] / b.pilot[k];
            c[i] = coherence(h, prev);
            score[i] += c[i];
        }
    }
    size_t best = n_off;
    for (size_t i = 0; i < n_off; ++i)
        if (std::abs(lo + static_cast<int64_t>(i)) >= SLIP_MIN && (best == n_off || score[i] > score[best])) best = i;
    if (best == n_off) return 0;
    for (const auto& c : at)
        if (c[best] < std::max(2 * c[static_cast<size_t>(-lo)], SLIP_ACCEPT)) return 0;
    return lo + static_cast<int64_t>(best);
}

}  // namespace

Frames demod_frames(std::span<const cd> z, int64_t p, int n_f, int shift, double phi_ref,
                    std::span<const double> steps_in, std::string_view band, std::span<const std::int64_t> slips_in,
                    const Mat<int>* pil, std::optional<int> close_at, const Mat<int>* pil_k) {
    const auto& b = waveform::band(band);
    const size_t nc = static_cast<size_t>(b.nc());
    phi_ref = phi_ref + 2 * PI * RS * shift / FS;  // the shift's own slope
    p += shift;
    const auto zlen = static_cast<int64_t>(z.size());
    Frames out;
    Mat<cd> raw(static_cast<size_t>(n_f + 1) * SYMS_PER_FRAME, nc);
    std::vector<double> steps(static_cast<size_t>(n_f + 1), 0.0), powers;
    std::vector<std::int64_t> slips(static_cast<size_t>(n_f + 1), 0);
    if (pil && !pil->rows) pil = nullptr;
    const bool intra = pil && pil_k && pil_k->rows;
    const bool search = steps_in.empty() && slips_in.empty() && !pil;
    const auto open = copy_open(b);
    // each frame's pilot per carrier: its own symbol's on scattered pilots; copy_open on a scattered band's copy frame
    auto frame_pilot = [&](int f) {
        std::vector<cd> h(nc);
        const auto& ref = close_at && f == *close_at ? open : b.pilot;
        if (intra) {
            const int* ps = (*pil)[static_cast<size_t>(f)];
            const int* pk = (*pil_k)[static_cast<size_t>(f)];
            for (size_t i = 0; i < nc; ++i) {
                const size_t k = static_cast<size_t>(pk[i]);
                h[i] = raw[static_cast<size_t>(f * SYMS_PER_FRAME + ps[i])][k] / ref[k];
            }
        } else {
            for (size_t k = 0; k < nc; ++k)
                h[k] = raw[static_cast<size_t>(f * SYMS_PER_FRAME + (pil ? (*pil)[static_cast<size_t>(f)][k] : 0))][k] / ref[k];
        }
        return h;
    };
    // a symbol's neighbouring pilots (scattered): the drift's phase slope is read between them
    std::vector<size_t> k1, k2;
    if (pil && !intra) {
        size_t r0 = 0;
        while (r0 < pil->rows && std::all_of((*pil)[r0], (*pil)[r0] + nc, [](int v) { return v == 0; })) ++r0;
        if (r0 == pil->rows) r0 = 0;
        const int* row = (*pil)[r0];
        std::vector<int> syms(row, row + nc);
        std::sort(syms.begin(), syms.end());
        syms.erase(std::unique(syms.begin(), syms.end()), syms.end());
        for (int sy : syms) {
            size_t last = nc;
            for (size_t k = 0; k < nc; ++k)
                if (row[k] == sy) {
                    if (last != nc) k1.push_back(last), k2.push_back(k);
                    last = k;
                }
        }
    }
    double tau_ema = 0.0;
    int total = 0;
    for (int f = 0; f <= n_f; ++f) {
        const int n_s = f < n_f ? SYMS_PER_FRAME : 1;
        auto& slip = slips[static_cast<size_t>(f)];
        if (!slips_in.empty()) slip = slips_in[static_cast<size_t>(f)];
        else if (search && f > 0 && f < n_f) {
            const auto* prev = raw[static_cast<size_t>((f - 1) * SYMS_PER_FRAME)];
            slip = find_slip(z, p, n_f - f, over_pilot({prev, prev + nc}, b), b);
        }
        p += slip;
        if (p + n_s * NSYM > zlen) throw SyncError("burst truncated");
        for (int s = 0; s < n_s; ++s) {
            const auto w = b.demod_window(z, p + s * NSYM + NCP, DEMOD_BACKOFF);
            std::copy(w.begin(), w.end(), raw[static_cast<size_t>(f * SYMS_PER_FRAME + s)]);
        }
        steps[static_cast<size_t>(f)] = total;
        if (!steps_in.empty()) {
            if (f < n_f) {
                total = static_cast<int>(steps_in[static_cast<size_t>(f + 1)]);
                p += static_cast<int64_t>(steps_in[static_cast<size_t>(f + 1)] - steps_in[static_cast<size_t>(f)]);
            }
            p += FRAME_SAMPLES;
            continue;
        }
        const auto hf = frame_pilot(f);
        powers.push_back(energy(hf) / static_cast<double>(nc));
        if (powers.back() > 0.1 * median(powers)) {
            double d = 0.0;
            if (intra && (!close_at || f != *close_at) && f < n_f) {
                const int* ps = (*pil)[static_cast<size_t>(f)];
                const int* pk = (*pil_k)[static_cast<size_t>(f)];
                std::vector<int> syms(ps, ps + nc);
                std::sort(syms.begin(), syms.end());
                syms.erase(std::unique(syms.begin(), syms.end()), syms.end());
                cd acc = 0.0;
                double gap = 0.0;
                size_t cnt = 0;
                for (int sy : syms) {
                    size_t last = nc;
                    for (size_t i = 0; i < nc; ++i)
                        if (ps[i] == sy) {
                            if (last != nc) {
                                const double dk = static_cast<double>(pk[i] - pk[last]);
                                acc += hf[i] * std::conj(hf[last]) * std::exp(cd(0.0, -dk * phi_ref));
                                gap += dk;
                                ++cnt;
                            }
                            last = i;
                        }
                }
                d = cnt ? std::arg(acc) / (gap / static_cast<double>(cnt)) : 0.0;
            } else if (pil && !intra && std::any_of((*pil)[static_cast<size_t>(f)], (*pil)[static_cast<size_t>(f)] + nc, [](int v) { return v != 0; })) {
                cd acc = 0.0;
                double gap = 0.0;
                for (size_t i = 0; i < k1.size(); ++i) {
                    const double dk = static_cast<double>(k2[i]) - static_cast<double>(k1[i]);
                    acc += hf[k2[i]] * std::conj(hf[k1[i]]) * std::exp(cd(0.0, -dk * phi_ref));
                    gap += dk;
                }
                d = k1.empty() ? 0.0 : std::arg(acc) / (gap / static_cast<double>(k1.size()));
            } else {
                d = std::arg(std::exp(cd(0.0, bin_phase_step(hf) - phi_ref)));
            }
            tau_ema += 0.02 * (-d * FS / (2 * PI * RS) - tau_ema);
            if (std::abs(tau_ema) >= 2) {
                const int step = static_cast<int>(std::clamp(std::nearbyint(tau_ema), -2.0, 2.0));
                p += step;
                total += step;
                tau_ema -= step;
            }
        }
        p += FRAME_SAMPLES;
    }
    const auto ph = equalizer::time_shift_phase(steps, bb_of(band));
    for (size_t r = 0; r < raw.rows; ++r)
        for (size_t k = 0; k < nc; ++k) raw[r][k] *= ph[r / SYMS_PER_FRAME][k];
    out.hp = Mat<cd>(static_cast<size_t>(n_f + 1), nc);
    for (int f = 0; f <= n_f; ++f) {
        const auto h = frame_pilot(f);
        std::copy(h.begin(), h.end(), out.hp[static_cast<size_t>(f)]);
    }
    raw.rows = static_cast<size_t>(n_f) * SYMS_PER_FRAME;
    raw.data.resize(raw.rows * nc);
    out.raw = std::move(raw);
    out.steps = std::move(steps);
    out.slips = std::move(slips);
    return out;
}

std::optional<std::vector<double>> copy_llr(std::span<const cd> z, int64_t p, std::string_view band, int n_hdr) {
    const auto& b = waveform::band(band);
    if (p + FRAME_SAMPLES + NSYM > static_cast<int64_t>(z.size())) return std::nullopt;
    std::vector<std::vector<cd>> win;
    for (int s = 0; s <= SYMS_PER_FRAME; ++s) win.push_back(b.demod_window(z, p + s * NSYM + NCP, HB));
    if (sc_band(band)) {  // copy_open, the header's 4 symbols, copy_close: those 4 in order
        const auto open = copy_open(b);
        std::vector<cd> h0(win[0]), h1(win[SYMS_PER_FRAME - 1]);
        for (size_t k = 0; k < h0.size(); ++k) h0[k] /= open[k], h1[k] /= open[k];
        const size_t n = static_cast<size_t>(SYMS_PER_FRAME - 2), nc = h0.size();
        Mat<cd> ys(n, nc), hs(n, nc);
        for (size_t s = 1; s <= n; ++s) {
            std::copy(win[s].begin(), win[s].end(), ys[s - 1]);
            const auto h = lerp(h0, h1, static_cast<double>(s) / (SYMS_PER_FRAME - 1));
            std::copy(h.begin(), h.end(), hs[s - 1]);
        }
        return qpsk_llr(ys, hs);
    }
    const auto h0 = over_pilot(win[0], b), h1 = over_pilot(win[SYMS_PER_FRAME], b);
    const size_t nc = h0.size();
    Mat<cd> ys(DATA_SYMS_PER_FRAME, nc), hs(DATA_SYMS_PER_FRAME, nc);
    for (int s = 1; s < SYMS_PER_FRAME; ++s) {
        std::copy(win[static_cast<size_t>(s)].begin(), win[static_cast<size_t>(s)].end(), ys[static_cast<size_t>(s - 1)]);
        const auto h = lerp(h0, h1, static_cast<double>(s) / SYMS_PER_FRAME);
        std::copy(h.begin(), h.end(), hs[static_cast<size_t>(s - 1)]);
    }
    const auto llr = qpsk_llr(ys, hs);
    const size_t row = 2 * nc;
    std::vector<double> out(llr.begin(), llr.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(n_hdr) * row));
    for (int i = n_hdr; i < DATA_SYMS_PER_FRAME; ++i)
        for (size_t j = 0; j < row; ++j) out[static_cast<size_t>(i - n_hdr) * row + j] += llr[static_cast<size_t>(i) * row + j];
    return out;
}

HeaderRead read_header(std::span<const cd> z, int64_t start, std::string_view band, const Accept* accept) {
    const auto& b = waveform::band(band);
    const size_t nc = static_cast<size_t>(b.nc());
    const int R = b.spec->preamble_repeats, ref = std::min(R, REF_REPEATS);
    const int64_t u0 = start + PREAMBLE_CP + static_cast<int64_t>(R - ref) * M;
    Mat<cd> h_reps(static_cast<size_t>(R), nc);
    for (int r = 0; r < R; ++r) {
        const auto w = over_pilot(b.demod_window(z, start + PREAMBLE_CP + static_cast<int64_t>(r) * M, HB), b);
        std::copy(w.begin(), w.end(), h_reps[static_cast<size_t>(r)]);
    }
    HeaderRead out;
    out.h_pre.assign(nc, 0.0);
    for (int r = R - ref; r < R; ++r)
        for (size_t k = 0; k < nc; ++k) out.h_pre[k] += h_reps[static_cast<size_t>(r)][k];
    for (auto& v : out.h_pre) v /= static_cast<double>(ref);
    const int64_t h0 = start + b.preamble_samples();
    const int64_t p0 = h0 + header_samples(band);
    out.h_first = over_pilot(b.demod_window(z, p0 + NCP, HB), b);
    const auto layout = header_layout(band);
    const size_t L = layout.size();
    std::vector<double> t_sym(L);
    out.y_all = Mat<cd>(L, nc);
    for (size_t i = 0; i < L; ++i) {
        const int64_t s = h0 + static_cast<int64_t>(i) * NSYM + NCP;
        t_sym[i] = static_cast<double>(s) + M / 2.0;
        const auto w = b.demod_window(z, s, HB);
        std::copy(w.begin(), w.end(), out.y_all[i]);
    }
    // pilots: the preamble's reference, the header's own pilots, the first frame pilot unless hosting
    std::vector<double> t_p{static_cast<double>(u0) + ref * M / 2.0};
    std::vector<std::vector<cd>> hp_rows{out.h_pre};
    for (size_t i = 0; i < L; ++i)
        if (layout[i]) {
            t_p.push_back(t_sym[i]);
            hp_rows.push_back(over_pilot({out.y_all[i], out.y_all[i] + nc}, b));
        }
    if (!hosts(band)) {
        t_p.push_back(static_cast<double>(p0 + NCP) + M / 2.0);
        hp_rows.push_back(out.h_first);
    }
    const auto h_p = equalizer::freq_smooth(rows_of(hp_rows), {HB, HB + NCP}, bb_of(band)).hs;
    const int64_t np_ = static_cast<int64_t>(t_p.size());
    std::vector<size_t> data_rows;
    for (size_t i = 0; i < L; ++i)
        if (!layout[i]) data_rows.push_back(i);
    out.y = Mat<cd>(data_rows.size(), nc);
    Mat<cd> hs(data_rows.size(), nc);
    for (size_t d = 0; d < data_rows.size(); ++d) {
        const double t = t_sym[data_rows[d]];
        const int64_t ss = std::lower_bound(t_p.begin(), t_p.end(), t) - t_p.begin();  // searchsorted, left
        const auto j = static_cast<size_t>(std::clamp<int64_t>(ss - 1, 0, np_ - 2));
        const double a = (t - t_p[j]) / (t_p[j + 1] - t_p[j]);
        std::copy(out.y_all[data_rows[d]], out.y_all[data_rows[d]] + nc, out.y[d]);
        const auto h = lerp(h_p.row(j), h_p.row(j + 1), a);
        std::copy(h.begin(), h.end(), hs[d]);
    }
    const auto soft = qpsk_llr(out.y, hs);
    const double floor = floor_of(band, accept);
    out.hdr = decode_header(soft, band, accept);
    if (copies(band)) {
        // the second copy, at each frame it can be in: a decode counts only if
        // the burst it describes carries its copy there
        std::optional<Header> best;
        for (int kc = 1; kc <= HEADER_COPY_AFTER; ++kc) {
            const auto extra = copy_llr(z, p0 + static_cast<int64_t>(kc) * FRAME_SAMPLES, band, static_cast<int>(out.y.rows));
            if (!extra) {
                out.pending_copy = true;
                continue;
            }
            std::vector<double> sum(soft);
            for (size_t i = 0; i < sum.size(); ++i) sum[i] += (*extra)[i];
            const auto h2 = decode_header(sum, band, accept);
            if (copy_frame(band, h2.n_cw * h2.spec->frames_per_cw) == kc && h2.score >= floor &&
                (!best || h2.score > best->score))
                best = h2;
        }
        if (best) {
            out.hdr = *best;
            out.pending_copy = false;
        }
    }
    out.valid = out.hdr.score >= floor;
    out.p0 = p0;
    out.start = start;
    out.band = b.spec->name;
    out.n0_pre = equalizer::preamble_noise(h_reps);
    out.n0_pre_k = equalizer::preamble_noise_k(h_reps);
    return out;
}

BestHeader best_header(std::span<const cd> z0, std::span<const std::string_view> bands, bool complete,
                       const Accept* accept, std::span<const BandStat> stats, bool final) {
    struct Good {
        double rank;
        HeaderRead hd;
        waveform::Acquisition acq;
        std::shared_ptr<const std::vector<cd>> z;
    };
    std::vector<Good> good;
    int detected = 0;
    bool waiting = false;
    const auto len = static_cast<int64_t>(z0.size());
    for (const auto name : search_bands(bands, accept)) {
        const Mat<double>* S = nullptr;
        std::int64_t col0 = 0;
        bool quiet = false;
        for (const auto& st : stats)
            if (st.band == name) {
                S = st.S;
                col0 = st.col0;
                quiet = st.quiet;
            }
        if (quiet) continue;  // acquire would find nothing
        waveform::Acquisition acq_b;
        try {
            acq_b = waveform::acquire(z0, waveform::band(name), {}, ACQUIRE_REACH_HZ, {}, S, col0);
        } catch (const SyncError&) {
            continue;
        }
        ++detected;
        const int64_t hdr_end = preamble_samples(band(name)) + header_samples(name) + NSYM;
        std::vector<std::pair<int64_t, double>> hyps{{acq_b.preamble_start, acq_b.freq_offset}};
        hyps.insert(hyps.end(), acq_b.alternatives.begin(), acq_b.alternatives.end());
        for (size_t h = 0; h < hyps.size(); ++h) {
            const auto [start, f] = hyps[h];
            if (!complete && start + 2 * M > len - hdr_end) {
                waiting = waiting || !final;
                continue;
            }
            auto zb = std::make_shared<const std::vector<cd>>(waveform::freq_correct(z0, f));
            for (int k : {0, -1, 1, -2, 2}) {
                const int64_t s = start + static_cast<int64_t>(k) * M;
                if (!(0 <= s && s <= len - hdr_end)) continue;
                auto r = read_header(*zb, s, name, accept);
                if (!complete && !final && r.pending_copy && r.hdr.score < COPY_COMMIT_SCORE) {
                    waiting = true;
                    continue;
                }
                if (r.valid && (!complete || burst_end(r.p0, *r.hdr.spec, r.hdr.n_cw) <= len)) {
                    const double rank = r.hdr.score - (h || k ? ALT_PENALTY : 0.0);
                    auto acq = acq_b;
                    acq.preamble_start = start;
                    acq.freq_offset = f;
                    good.push_back({rank, std::move(r), std::move(acq), zb});
                }
            }
        }
    }
    if (good.empty()) throw SyncError(detected ? "header decode failed" : "no preamble found");
    size_t best = 0;  // max(): the first of equal ranks
    for (size_t i = 1; i < good.size(); ++i)
        if (good[i].rank > good[best].rank) best = i;
    if (waiting && good[best].rank < STREAM_COMMIT_SCORE) throw SyncError("a header is still arriving");
    return {std::move(good[best].hd), std::move(good[best].acq), good[best].z};
}

Lock find_burst(std::span<const double> x, std::span<const std::string_view> bands, const Accept* accept,
                std::span<const BandStat> stats) {
    // every band quiet: nothing to acquire, so no baseband copy of the whole buffer either
    const auto searched = search_bands(bands, accept);
    if (!searched.empty() && std::all_of(searched.begin(), searched.end(), [&](std::string_view name) {
            return std::any_of(stats.begin(), stats.end(), [&](const BandStat& st) { return st.band == name && st.quiet; });
        }))
        throw SyncError("no preamble found");
    const auto z0 = waveform::to_baseband(x);
    const auto b = best_header(z0, bands, false, accept, stats);
    Lock l;
    l.spec = b.hd.hdr.spec;
    l.n_cw = b.hd.hdr.n_cw;
    l.start = b.hd.start;
    l.score = b.hd.hdr.score;
    l.band = b.hd.band;
    l.end = burst_end(b.hd.p0, *l.spec, l.n_cw);
    l.p0 = b.hd.p0;
    l.cfo = b.acq.freq_offset;
    return l;
}

std::vector<double> pilot_coherence(std::span<const double> x, const Lock& lock, int n_max, bool latest, double* pattern) {
    const Spec& spec = *lock.spec;
    const auto& b = waveform::band(spec.band);
    const auto len = static_cast<int64_t>(x.size());
    const int total = frames_on_air(spec, lock.n_cw) + 1;
    int avail = 0;
    for (int f = 0; f < total; ++f) avail += lock.p0 + static_cast<int64_t>(f) * FRAME_SAMPLES + NCP + M <= len;
    const int f1 = latest ? avail : std::min(avail, n_max + 1);
    const int f0 = std::max(0, f1 - n_max - 1);
    if (pattern) *pattern = 0.0;
    if (f1 - f0 < 2) return {};
    const int64_t lo = std::max<int64_t>(0, lock.p0 + static_cast<int64_t>(f0) * FRAME_SAMPLES - 2 * NSYM);
    const int64_t hi = lock.p0 + static_cast<int64_t>(f1 - 1) * FRAME_SAMPLES + 3 * NSYM;
    const auto z = waveform::freq_correct(waveform::to_baseband(pyslice(x, lo, hi)), lock.cfo);
    std::vector<std::vector<cd>> y;
    for (int f = f0; f < f1; ++f)
        y.push_back(b.demod_window(z, lock.p0 - lo + static_cast<int64_t>(f) * FRAME_SAMPLES + NCP, DEMOD_BACKOFF));
    const auto kc = copy_frame(spec.sync_band, lock.n_cw * spec.frames_per_cw);
    const bool sc = sc_spec(spec, lock.n_cw * spec.frames_per_cw);
    const auto intra = sc ? sc_intra_spec(b.nc()) : std::nullopt;
    if (intra) {
        const int K = intra->K, ds = intra->syms[1] - intra->syms[0];
        const size_t nc = b.pilot.size();
        const auto open = copy_open(b);
        struct Seg {
            int sym;
            std::vector<int> ks;
            std::vector<cd> h;
        };
        std::vector<Seg> segs;
        std::vector<cd> a_lag(nc - static_cast<size_t>(K), cd{0, 0});
        for (int fa = f0; fa < f1; ++fa) {
            const int64_t p_fa = lock.p0 - lo + static_cast<int64_t>(fa) * FRAME_SAMPLES;
            if ((kc && fa == *kc) || fa == total - 1) {
                const auto& ref = (kc && fa == *kc) ? open : b.pilot;
                auto h0 = b.demod_window(z, p_fa + NCP, DEMOD_BACKOFF);
                for (size_t k = 0; k < nc; ++k) h0[k] /= ref[k];
                for (size_t k = 0; k + static_cast<size_t>(K) < nc; ++k) a_lag[k] += h0[k + static_cast<size_t>(K)] * std::conj(h0[k]);
                std::vector<int> allk(nc);
                std::iota(allk.begin(), allk.end(), 0);
                segs.push_back({fa * SYMS_PER_FRAME, std::move(allk), std::move(h0)});
            } else {
                const auto& fc = sc_frame_cells(b.nc(), fa);
                const size_t m = nc / static_cast<size_t>(K);
                std::vector<int> ks(fc.PK.begin(), fc.PK.begin() + static_cast<std::ptrdiff_t>(m));
                for (int si = 0; si < intra->n_syms; ++si) {
                    const int s_ = intra->syms[static_cast<size_t>(si)];
                    const auto w = b.demod_window(z, p_fa + static_cast<int64_t>(s_) * NSYM + NCP, DEMOD_BACKOFF);
                    std::vector<cd> hs(m);
                    for (size_t j = 0; j < m; ++j) hs[j] = w[static_cast<size_t>(ks[j])] / b.pilot[static_cast<size_t>(ks[j])];
                    for (size_t j = 0; j + 1 < m; ++j) a_lag[static_cast<size_t>(ks[j])] += hs[j + 1] * std::conj(hs[j]);
                    segs.push_back({fa * SYMS_PER_FRAME + s_, ks, std::move(hs)});
                }
            }
        }
        std::vector<std::pair<std::vector<cd>, std::vector<cd>>> pairs;
        for (size_t idx = 1; idx < segs.size(); ++idx) {
            const auto& s0 = segs[idx - 1];
            const auto& s1 = segs[idx];
            if (s1.sym - s0.sym != ds) continue;
            std::vector<cd> u, v;
            size_t i0 = 0, i1 = 0;
            while (i0 < s0.ks.size() && i1 < s1.ks.size()) {
                if (s0.ks[i0] == s1.ks[i1]) {
                    u.push_back(s0.h[i0++]);
                    v.push_back(s1.h[i1++]);
                } else if (s0.ks[i0] < s1.ks[i1]) {
                    ++i0;
                } else {
                    ++i1;
                }
            }
            if (!u.empty()) pairs.emplace_back(std::move(u), std::move(v));
        }
        if (pairs.empty()) return {};
        const size_t grp = std::min(static_cast<size_t>(K), pairs.size());
        std::vector<double> out;
        for (size_t g0 = 0; g0 + grp <= pairs.size(); ++g0) {
            cd dot = 0.0;
            double eu = 0.0, ev = 0.0;
            for (size_t j = g0; j < g0 + grp; ++j) {
                for (size_t t = 0; t < pairs[j].first.size(); ++t) {
                    dot += std::conj(pairs[j].first[t]) * pairs[j].second[t];
                    eu += std::norm(pairs[j].first[t]);
                    ev += std::norm(pairs[j].second[t]);
                }
            }
            out.push_back(std::abs(dot) / (std::sqrt(eu) * std::sqrt(ev) + 1e-30));
        }
        if (pattern) {
            cd s{0, 0};
            size_t act = 0;
            for (const cd& t : a_lag) {
                const double a = std::abs(t);
                if (a > 0.0) {
                    s += t / a;
                    ++act;
                }
            }
            *pattern = act ? std::abs(s) / static_cast<double>(act) : 0.0;
        }
        return out;
    }
    if (sc_band(spec.band) && kc && f0 <= *kc && *kc < f1) {  // its first symbol is copy_close
        const auto open = copy_open(b);
        auto& v = y[static_cast<size_t>(*kc - f0)];
        for (size_t k = 0; k < v.size(); ++k) v[k] = v[k] * b.pilot[k] / open[k];
    }
    if (sc) {  // each carrier's own pilot, but the copy's frame and the closing pilot
        const auto& cells = sc_cells(b.nc());
        const size_t nc = b.pilot.size();
        for (int f = f0; f < f1; ++f) {
            if ((kc && f == *kc) || f == total - 1) continue;
            auto& v = y[static_cast<size_t>(f - f0)];
            for (int j = 1; j < SYMS_PER_FRAME; ++j) {
                bool any = false;
                for (size_t k = 0; k < nc; ++k) any = any || cells.own[k] / b.nc() == j;
                if (!any) continue;
                const auto w = b.demod_window(z, lock.p0 - lo + static_cast<int64_t>(f) * FRAME_SAMPLES + j * NSYM + NCP,
                                              DEMOD_BACKOFF);
                for (size_t k = 0; k < nc; ++k)
                    if (cells.own[k] / b.nc() == j) v[k] = w[k];
            }
        }
    }
    std::vector<double> out;
    for (size_t f = 1; f < y.size(); ++f)
        out.push_back(std::abs(dot_conj(y[f], y[f - 1])) / (std::sqrt(energy(y[f]) * energy(y[f - 1])) + 1e-30));
    if (pattern) {  // adjacent carriers' products, pilot divided out, summed over the frames; unit weight each
        const std::size_t nc = b.pilot.size();
        const std::size_t lag = sc && sc_combs(b.nc()) ? 3 : 1;  // k3d2: neighbours in one symbol's comb
        std::vector<cd> a(nc - lag, cd{0, 0});
        for (const auto& v : y)
            for (std::size_t k = 0; k + lag < nc; ++k)
                a[k] += (v[k + lag] * std::conj(b.pilot[k + lag])) * std::conj(v[k] * std::conj(b.pilot[k]));
        cd s{0, 0};
        for (const cd& t : a) s += t / (std::abs(t) + 1e-30);
        *pattern = std::abs(s) / static_cast<double>(a.size());
    }
    return out;
}

void FoldTrack::acc(std::span<const cd* const> D_rows, int64_t a, int64_t u0, int64_t u1, bool add, size_t period) {
    const int64_t F = static_cast<int64_t>(period);
    for (int64_t u = u0; u < u1;) {
        const int64_t q = u % F, k = std::min(u1 - u, F - q);
        for (size_t f = 0; f < D_rows.size(); ++f) {
            // as doubles (std::complex is two, array-compatible): the same sums, and the loop vectorizes
            double* __restrict r = reinterpret_cast<double*>(R_[f] + q);
            const double* __restrict d = reinterpret_cast<const double*>(D_rows[f] + (u - a));
            if (add)
                for (int64_t i = 0; i < 2 * k; ++i) r[i] += d[i];
            else
                for (int64_t i = 0; i < 2 * k; ++i) r[i] -= d[i];
        }
        u += k;
    }
}

const Mat<cd>& FoldTrack::fold(std::span<const cd* const> D_rows, size_t dcols, int64_t a, int64_t lo, int64_t hi,
                               size_t period) {
    if (!have_ || n_ >= RENEW || R_.cols != period || !(lo_ <= lo && lo <= hi_ && hi_ <= hi) || lo_ < a ||
        hi > a + static_cast<int64_t>(dcols)) {
        R_ = Mat<cd>(D_rows.size(), period);
        acc(D_rows, a, lo, hi, true, period);
        have_ = true;
        n_ = 0;
    } else {
        acc(D_rows, a, hi_, hi, true, period);
        acc(D_rows, a, lo_, lo, false, period);
        ++n_;
    }
    lo_ = lo;
    hi_ = hi;
    return R_;
}

std::pair<int64_t, int> presence_window(size_t dcols, int64_t window, int decim, int pad_frames) {
    const int64_t FR_d = FRAME_SAMPLES / decim;
    const int64_t PF_d = static_cast<int64_t>(pad_frames) * FR_d;
    const int64_t n_d = static_cast<int64_t>(dcols) + PF_d;
    const int64_t W_d = std::min(n_d, window / decim);
    return {n_d - W_d, static_cast<int>((W_d - PF_d) / FR_d)};
}

std::pair<double, double> fold_presence(const Mat<cd>& fold, double level, int m, std::span<const double> row_w) {
    const size_t FR = fold.cols;
    std::vector<double> mag2(FR, 0.0);
    for (size_t f = 0; f < fold.rows; ++f) {
        const double w2 = row_w.empty() ? 1.0 : row_w[f] * row_w[f];
        const cd* row = fold[f];
        for (size_t ph = 0; ph < FR; ++ph) {
            const double p = (row[ph].real() * row[ph].real() + row[ph].imag() * row[ph].imag()) * w2;
            if (p > mag2[ph]) mag2[ph] = p;
        }
    }
    const double mx = std::sqrt(*std::max_element(mag2.begin(), mag2.end()));
    std::nth_element(mag2.begin(), mag2.begin() + FR / 2, mag2.end());
    const double hi = std::sqrt(mag2[FR / 2]);
    const double med = (FR % 2) ? hi : 0.5 * (std::sqrt(*std::max_element(mag2.begin(), mag2.begin() + FR / 2)) + hi);
    return {mx / (level * std::sqrt(static_cast<double>(m * COPY_PAIRS))), mx / (med + 1e-300)};
}

bool presence_verdict(double peak, double contrast, int m, std::string_view band) {
    if (m < PRESENCE_MIN_M || peak < PRESENCE_PEAK) return false;
    const auto idx = static_cast<size_t>(std::min(m - PRESENCE_MIN_M, static_cast<int>(PRESENCE_CONTRAST.size()) - 1));
    const double scale = (band == "n4" && config::sc_band(band)) ? PRESENCE_CONTRAST_N4_SCALE : 1.0;
    return contrast >= scale * PRESENCE_CONTRAST[idx];
}

namespace {

Mat<cd> pilot_fold(std::span<const cd* const> D_rows, size_t t0, size_t m, size_t FR_d = FRAME_SAMPLES) {
    Mat<cd> fold(D_rows.size(), FR_d);
    for (size_t f = 0; f < D_rows.size(); ++f) {
        cd* dst = fold[f];
        const cd* src = D_rows[f] + t0;
        for (size_t k = 0; k < m; ++k) {
            const cd* row = src + k * FR_d;
            for (size_t ph = 0; ph < FR_d; ++ph) dst[ph] += row[ph];
        }
    }
    return fold;
}

}  // namespace

std::pair<double, double> pilot_presence(std::span<const cd* const> D_rows, size_t dcols, double level, int64_t window) {
    const auto [t0, m] = presence_window(dcols, window);
    if (m < 1) return {0.0, 0.0};
    return fold_presence(pilot_fold(D_rows, static_cast<size_t>(t0), static_cast<size_t>(m)), level, m);
}

namespace {
std::vector<double> intra_freqs(int n_combs);
std::vector<double> comb_freqs();

bool verify_band(std::span<const cd> z0, int64_t ph, int n_frames, std::string_view band, double cfo) {
    const auto& b = waveform::band(band);
    const auto z = waveform::freq_correct(z0, cfo);
    const size_t nc = b.pilot.size();
    std::vector<std::vector<cd>> y;
    y.reserve(static_cast<size_t>(n_frames));
    if (config::sc_band(band)) {
        const auto& cells = sc_cells(b.nc());
        for (int k = 0; k < n_frames; ++k) {
            std::vector<cd> v(nc, cd{0, 0});
            for (int j = 0; j < SYMS_PER_FRAME; ++j) {
                bool any = false;
                for (size_t c = 0; c < nc; ++c) any = any || cells.own[c] / b.nc() == j;
                if (!any) continue;
                const auto w = b.demod_window(z, ph + static_cast<int64_t>(k) * FRAME_SAMPLES + j * NSYM, DEMOD_BACKOFF);
                for (size_t c = 0; c < nc; ++c)
                    if (cells.own[c] / b.nc() == j) v[c] = w[c];
            }
            y.push_back(std::move(v));
        }
    } else {
        for (int k = 0; k < n_frames; ++k)
            y.push_back(b.demod_window(z, ph + static_cast<int64_t>(k) * FRAME_SAMPLES, DEMOD_BACKOFF));
    }
    double c_sum = 0.0;
    for (size_t k = 1; k < y.size(); ++k)
        c_sum += std::abs(dot_conj(y[k], y[k - 1])) / (std::sqrt(energy(y[k]) * energy(y[k - 1])) + 1e-30);
    const double mc = c_sum / static_cast<double>(y.size() - 1);
    std::vector<cd> a(nc - 1, cd{0, 0});
    for (const auto& v : y)
        for (size_t k = 0; k + 1 < nc; ++k)
            a[k] += (v[k + 1] * std::conj(b.pilot[k + 1])) * std::conj(v[k] * std::conj(b.pilot[k]));
    cd s{0, 0};
    for (const cd& t : a) s += t / (std::abs(t) + 1e-30);
    const double pat = std::abs(s) / static_cast<double>(a.size());
    const double pat_floor = std::min(PRESENCE_PATTERN_CAP, PRESENCE_PATTERN_SCALE * pilot_pattern(band));
    return mc > pilot_noise(band) && pat > pat_floor;
}

bool verify_intra(std::span<const cd> z0, int64_t ph, int n_frames, std::string_view band, double cfo, int c_start,
                  const ScIntra& intra) {
    const auto& b = waveform::band(band);
    const size_t nc = b.pilot.size();
    const size_t K = static_cast<size_t>(intra.K);
    const auto z = waveform::freq_correct(z0, cfo);
    std::vector<std::pair<std::vector<cd>, std::vector<cd>>> pairs;
    std::vector<cd> a_lag(nc - K, cd{0, 0});
    for (int k = 0; k < n_frames; ++k) {
        const size_t c = static_cast<size_t>((c_start + k) % intra.n_combs);
        std::vector<size_t> ks;
        for (size_t j = c; j < nc; j += K) ks.push_back(j);
        std::vector<std::vector<cd>> hs;
        hs.reserve(static_cast<size_t>(intra.n_syms));
        for (int si = 0; si < intra.n_syms; ++si) {
            const auto w = b.demod_window(z, ph + static_cast<int64_t>(k) * FRAME_SAMPLES + intra.syms[si] * NSYM,
                                          DEMOD_BACKOFF);
            std::vector<cd> h(ks.size());
            for (size_t j = 0; j < ks.size(); ++j) h[j] = w[ks[j]] / b.pilot[ks[j]];
            for (size_t j = 0; j + 1 < ks.size(); ++j) a_lag[ks[j]] += h[j + 1] * std::conj(h[j]);
            hs.push_back(std::move(h));
        }
        for (size_t m_ = 0; m_ + 1 < hs.size(); ++m_) pairs.emplace_back(hs[m_], hs[m_ + 1]);
    }
    if (pairs.empty()) return false;
    const size_t grp = std::min(K, pairs.size());
    double c_sum = 0.0;
    size_t c_cnt = 0;
    for (size_t g0 = 0; g0 + grp <= pairs.size(); ++g0) {
        cd dot = 0.0;
        double eu = 0.0, ev = 0.0;
        for (size_t j = g0; j < g0 + grp; ++j)
            for (size_t t = 0; t < pairs[j].first.size(); ++t) {
                dot += std::conj(pairs[j].first[t]) * pairs[j].second[t];
                eu += std::norm(pairs[j].first[t]);
                ev += std::norm(pairs[j].second[t]);
            }
        c_sum += std::abs(dot) / (std::sqrt(eu) * std::sqrt(ev) + 1e-30);
        ++c_cnt;
    }
    cd s{0, 0};
    size_t act = 0;
    for (const cd& t : a_lag) {
        const double a = std::abs(t);
        if (a > 0.0) {
            s += t / a;
            ++act;
        }
    }
    const double pat = act ? std::abs(s) / static_cast<double>(act) : 0.0;
    const double pat_floor = std::min(PRESENCE_PATTERN_CAP, PRESENCE_PATTERN_SCALE * pilot_pattern(band));
    return (c_sum / static_cast<double>(c_cnt)) > pilot_noise(band) && pat > pat_floor;
}
}  // namespace

bool fold_verify(std::span<const double> x, std::string_view band, const Mat<cd>& fold, int64_t fold_phase0,
                 int decim, std::span<const double> row_w) {
    const size_t FR_d = fold.cols;
    size_t q = 0, fi = 0;
    double best = -1.0;
    for (size_t ph = 0; ph < FR_d; ++ph) {
        size_t bf = 0;
        double bm = -1.0;
        for (size_t f = 0; f < fold.rows; ++f) {
            const double w2 = row_w.empty() ? 1.0 : row_w[f] * row_w[f];
            const cd z = fold[f][ph];
            const double v = (z.real() * z.real() + z.imag() * z.imag()) * w2;
            if (v > bm) {
                bm = v;
                bf = f;
            }
        }
        if (bm > best) {
            best = bm;
            q = ph;
            fi = bf;
        }
    }
    const int64_t ph =
        ((static_cast<int64_t>(q) * decim - fold_phase0) % FRAME_SAMPLES + FRAME_SAMPLES) % FRAME_SAMPLES;
    const bool sc = config::sc_band(band);
    const auto intra = sc ? sc_intra_spec(config::band(band).nc) : std::nullopt;
    if (intra) {
        const int64_t avail = static_cast<int64_t>(x.size()) - ph - intra->syms[intra->n_syms - 1] * NSYM - M;
        if (avail < 0) return false;
        const int n_frames = static_cast<int>(avail / FRAME_SAMPLES) + 1;
        if (n_frames < 2) return false;
        const auto freqs = intra_freqs(intra->n_combs);
        const size_t n_cfo = waveform::cfo_grid().size();
        const int c0 = static_cast<int>(fi / n_cfo);
        const int64_t f_abs = (fold_phase0 + ph) / FRAME_SAMPLES;
        const int c_start = static_cast<int>(((c0 + f_abs) % intra->n_combs + intra->n_combs) % intra->n_combs);
        const auto z0 = waveform::to_baseband(x);
        const cd d = fold[fi][q];
        const int cfo_lag = (intra->syms[1] - intra->syms[0]) * NSYM;
        for (double f : cfo_aliases(d, freqs[fi], cfo_lag))
            if (verify_intra(z0, ph, n_frames, band, f, c_start, *intra)) return true;
        return false;
    }
    int max_j = 0;
    if (sc) {
        const int nc = config::band(band).nc;
        for (int p : sc_cells(nc).own) max_j = std::max(max_j, p / nc);
    }
    const int64_t avail = static_cast<int64_t>(x.size()) - ph - max_j * NSYM - M;
    if (avail < 0) return false;
    const int n_frames = static_cast<int>(avail / FRAME_SAMPLES) + 1;
    if (n_frames < 2) return false;
    const auto freqs = sc ? comb_freqs() : waveform::cfo_grid();
    const auto z0 = waveform::to_baseband(x);
    const cd d = fold[fi][q];
    const double centre = freqs[fi];
    for (double f : cfo_aliases(d, centre))
        if (verify_band(z0, ph, n_frames, band, f)) return true;
    const bool has_n4 = std::any_of(config::BANDS.begin(), config::BANDS.end(), [](const auto& b) { return b.name == "n4"; });
    if (!sc && band == "n10" && has_n4) {
        for (double df : {-100.0, 137.5}) {
            const double c4 = centre + df;
            if (std::abs(c4) <= config::ACQUIRE_REACH_HZ + 25.0)
                for (double f : cfo_aliases(d, c4))
                    if (verify_band(z0, ph, n_frames, "n4", f)) return true;
        }
    }
    return false;
}

bool pilot_present(std::span<const cd* const> D_rows, size_t dcols, double level, int64_t window,
                   std::span<const double> x, std::string_view band, int decim, int64_t stream_lo,
                   std::span<const double> row_w) {
    const bool sc = !band.empty() && config::sc_band(band);
    const auto intra = sc ? sc_intra_spec(config::band(band).nc) : std::nullopt;
    const int pad_fr = intra ? (COPY_PAIRS - 1) : COPY_PAIRS;
    const size_t FR_d = static_cast<size_t>(FRAME_SAMPLES / decim);
    const auto [t0_d, m] = presence_window(dcols, window, decim, pad_fr);
    if (m < PRESENCE_MIN_M) return false;
    Mat<cd> fold = pilot_fold(D_rows, static_cast<size_t>(t0_d), static_cast<size_t>(m), FR_d);
    if (sc) {
        for (size_t f = 0; f < fold.rows; ++f) {
            cd* dst = fold[f];
            cd mu{0, 0};
            for (size_t ph = 0; ph < FR_d; ++ph) mu += dst[ph];
            mu /= static_cast<double>(FR_d);
            for (size_t ph = 0; ph < FR_d; ++ph) dst[ph] -= mu;
        }
    }
    const auto [pk, ct] = fold_presence(fold, level, m, row_w);
    if (!presence_verdict(pk, ct, m, band)) return false;
    if (x.empty() || band.empty()) return true;
    const size_t off = static_cast<size_t>(t0_d * decim);
    if (off >= x.size()) return false;
    const size_t shift = static_cast<size_t>(((stream_lo / decim + t0_d) % static_cast<int64_t>(FR_d) + static_cast<int64_t>(FR_d)) % static_cast<int64_t>(FR_d));
    if (shift)
        for (size_t f = 0; f < fold.rows; ++f) std::rotate(fold[f], fold[f] + (FR_d - shift), fold[f] + FR_d);
    return fold_verify(x.subspan(off), band, fold, stream_lo + t0_d * decim, decim, row_w);
}

std::vector<double> cfo_aliases(cd d, double centre, int lag) {
    const double alias = static_cast<double>(FS) / lag;
    const double frac = std::arg(d) / (2 * PI) * alias;
    const double k0 = std::nearbyint((centre - frac) / alias);
    std::vector<double> out;
    for (double k : {k0 - 1, k0, k0 + 1})
        if (std::abs(frac + k * alias - centre) <= waveform::STEP_HZ / 2 + 0.5) out.push_back(frac + k * alias);
    return out;
}

namespace {

// find_copy from the pair products D on: steps 1 to 4 (freqs: D's rows' CFOs; comb: a scattered-pilot band's
// search, gated on its copy frame's rows)
std::optional<Lock> find_copy_from(std::span<const double> x, std::span<const cd> z0, std::string_view band,
                                   const Accept* accept, std::span<const cd* const> D_rows, size_t dcols, double level,
                                   double* peak, const Mat<cd>* fold_in, std::span<const double> freqs, bool comb,
                                   int cfo_lag = FRAME_SAMPLES, size_t fold_shift = 0, int decim = 1,
                                   std::span<const double> row_w = {});

// modem.SC_COPY_BIN_SPREAD (its reason there): past this spread of the CFO bins' noise levels find_copy weighs each
// bin's fold by the quietest bin's level over its own.
constexpr double SC_COPY_BIN_SPREAD = 1.25;

}  // namespace

// modem._copy_row_weights: an intra-frame comb layout's fold rows (n_combs x the CFO bins), each the quietest bin's
// level over its own; empty when the levels are within SC_COPY_BIN_SPREAD of each other.
std::vector<double> copy_row_weights(std::span<const double> row_level, int n_combs) {
    if (row_level.empty()) return {};
    const auto [lo_it, hi_it] = std::minmax_element(row_level.begin(), row_level.end());
    const double lo = *lo_it;
    if (!(lo > 0) || *hi_it <= SC_COPY_BIN_SPREAD * lo) return {};
    std::vector<double> w;
    w.reserve(row_level.size() * static_cast<size_t>(n_combs));
    for (int c = 0; c < n_combs; ++c)
        for (double v : row_level) w.push_back(lo / v);
    return w;
}

namespace {

// The per-bin noise level of each comb's outputs (sync.noise_level's before its min over bins), the least over combs.
std::vector<double> comb_row_levels(const std::vector<Mat<cd>>& Cs) {
    std::vector<double> out(Cs[0].rows, INFINITY);
    for (const auto& C : Cs)
        for (size_t i = 0; i < C.rows; ++i) {
            std::vector<double> p(C.cols);
            for (size_t j = 0; j < p.size(); ++j) p[j] = std::norm(C[i][j]);
            out[i] = std::min(out[i], dsp::quantile(std::move(p), waveform::NOISE_QUANTILE));
        }
    return out;
}

// modem._comb_filters: per symbol j (or comb phase c for intra-frame comb layouts) holding scattered pilots,
// the preamble template's carriers as a unit-norm template
std::vector<std::pair<int, std::vector<cd>>> comb_filters(std::string_view band) {
    const auto& b = waveform::band(band);
    const size_t nc = static_cast<size_t>(b.nc());
    // dem[k][n] = exp(-2j pi bb_k n / FS), n < M
    Mat<cd> dem(nc, static_cast<size_t>(M));
    for (size_t k = 0; k < nc; ++k)
        for (int n = 0; n < M; ++n) dem[k][static_cast<size_t>(n)] = std::polar(1.0, -2 * PI * static_cast<double>(b.bb[k]) * n / FS);
    std::vector<cd> proj(nc, 0.0);  // dem @ tpl
    for (size_t k = 0; k < nc; ++k)
        for (int n = 0; n < M; ++n) proj[k] += dem[k][static_cast<size_t>(n)] * b.preamble_template[static_cast<size_t>(PREAMBLE_CP + n)];
    std::vector<std::pair<int, std::vector<cd>>> out;
    if (const auto intra = sc_intra_spec(b.nc())) {
        for (int c = 0; c < intra->n_combs; ++c) {
            std::vector<cd> t(static_cast<size_t>(M), 0.0);
            for (size_t k = 0; k < nc; ++k)
                if (static_cast<int>(k) % intra->K == c)
                    for (int n = 0; n < M; ++n) t[static_cast<size_t>(n)] += std::conj(dem[k][static_cast<size_t>(n)]) * proj[k];
            double re = 0, im = 0;
            for (auto& v : t) {
                v /= M;
                re += v.real() * v.real();
                im += v.imag() * v.imag();
            }
            const double nrm = std::sqrt(re + im);
            for (auto& v : t) v /= nrm;
            out.emplace_back(c, std::move(t));
        }
        return out;
    }
    const auto& cells = sc_cells(b.nc());
    std::vector<int> js;
    for (size_t k = 0; k < nc; ++k) js.push_back(cells.own[k] / b.nc());
    std::sort(js.begin(), js.end());
    js.erase(std::unique(js.begin(), js.end()), js.end());
    for (int j : js) {
        std::vector<cd> t(static_cast<size_t>(M), 0.0);
        for (size_t k = 0; k < nc; ++k)
            if (cells.own[k] / b.nc() == j)
                for (int n = 0; n < M; ++n) t[static_cast<size_t>(n)] += std::conj(dem[k][static_cast<size_t>(n)]) * proj[k];
        double re = 0, im = 0;
        for (auto& v : t) {
            v /= M;
            re += v.real() * v.real();
            im += v.imag() * v.imag();
        }
        const double nrm = std::sqrt(re + im);
        for (auto& v : t) v /= nrm;
        out.emplace_back(j, std::move(t));
    }
    return out;
}

std::vector<double> intra_freqs(int n_combs) {
    const auto grid = waveform::cfo_grid();
    std::vector<double> out;
    out.reserve(grid.size() * static_cast<size_t>(n_combs));
    for (int c = 0; c < n_combs; ++c) out.insert(out.end(), grid.begin(), grid.end());
    return out;
}

Mat<cd> sum_frames(std::span<const cd* const> P_rows, size_t pcols, size_t F, size_t pairs) {
    const size_t need = (pairs - 1) * F;
    const size_t n = pcols > need ? pcols - need : 0;
    Mat<cd> D(P_rows.size(), n);
    if (!n) return D;
    for (size_t r = 0; r < P_rows.size(); ++r) {
        cd* dst = D[r];
        const cd* src = P_rows[r];
        for (size_t j = 0; j < pairs; ++j) {
            const cd* sj = src + j * F;
            for (size_t t = 0; t < n; ++t) dst[t] += sj[t];
        }
    }
    return D;
}

std::pair<Mat<cd>, double> intra_frame_prods(const std::vector<Mat<cd>>& Cs, const ScIntra& intra, int64_t stream_s0,
                                             bool need_level = true, int decim = 1) {
    const size_t D = static_cast<size_t>(decim);
    const size_t step = static_cast<size_t>(intra.syms[1] - intra.syms[0]) * NSYM / D;
    const size_t span = (intra.n_combs == 1 ? static_cast<size_t>(FRAME_SAMPLES)
                                            : static_cast<size_t>(intra.syms[static_cast<size_t>(intra.n_syms - 1)]) * NSYM) /
                        D;
    const size_t FR_d = FRAME_SAMPLES / D;
    const int64_t s0_d = stream_s0 / decim;
    size_t n = SIZE_MAX;
    for (const auto& c : Cs) n = std::min(n, c.cols > span ? c.cols - span : size_t{0});
    const size_t nb = Cs[0].rows;
    if (!n) return {Mat<cd>(nb * static_cast<size_t>(intra.n_combs), 0), 1.0};
    if (intra.n_combs == 1) {
        const Mat<cd>& C0 = Cs[0];
        Mat<cd> P(nb, n);
        for (size_t b = 0; b < nb; ++b) {
            const cd* r = C0[b];
            cd* dst = P[b];
            for (size_t t = 0; t < n; ++t)
                dst[t] = r[t + step] * std::conj(r[t]) + r[t + 2 * step] * std::conj(r[t + step]);
        }
        return {std::move(P), need_level ? std::sqrt(2.0) * waveform::noise_level(C0) : 1.0};
    }
    const size_t n_lags = static_cast<size_t>(intra.n_syms - 1);
    const size_t nc = static_cast<size_t>(intra.n_combs);
    std::vector<Mat<cd>> Qs(nc);
    double min_lvl = INFINITY;
    for (size_t c = 0; c < nc; ++c) {
        const Mat<cd>& Cc = Cs[c];
        if (need_level) min_lvl = std::min(min_lvl, waveform::noise_level(Cc));
        Mat<cd> Q(nb, n);
        for (size_t b = 0; b < nb; ++b) {
            const cd* r = Cc[b];
            cd* dst = Q[b];
            for (size_t m = 0; m < n_lags; ++m) {
                const cd* r0 = r + m * step;
                const cd* r1 = r + (m + 1) * step;
                for (size_t t = 0; t < n; ++t) {
                    const double ar = r1[t].real(), ai = r1[t].imag(), br = r0[t].real(), bi = r0[t].imag();
                    dst[t] += cd(ar * br + ai * bi, ai * br - ar * bi);
                }
            }
        }
        Qs[c] = std::move(Q);
    }
    Mat<cd> P(nc * nb, n);
    for (size_t t = 0; t < n; ++t) {
        const size_t f_idx = static_cast<size_t>((s0_d + static_cast<int64_t>(t)) / static_cast<int64_t>(FR_d));
        for (size_t c0 = 0; c0 < nc; ++c0) {
            const Mat<cd>& Q = Qs[(c0 + f_idx) % nc];
            for (size_t b = 0; b < nb; ++b) P[c0 * nb + b][t] = Q[b][t];
        }
    }
    return {std::move(P), need_level ? std::sqrt(static_cast<double>(n_lags)) * min_lvl : 1.0};
}

// modem._comb_freqs: the grid's bins at SUBBINS offsets inside each
std::vector<double> comb_freqs() {
    constexpr int SUBBINS = 5;  // modem.SC_COPY_SUBBINS
    std::vector<double> f2;
    for (double f : waveform::cfo_grid())
        for (int o = 0; o < SUBBINS; ++o) f2.push_back(f + (o - (SUBBINS - 1) / 2.0) * waveform::STEP_HZ / SUBBINS);
    return f2;
}

// modem._comb_coherent: the combs' outputs (Cs, symbols js) added within the frame at each of f2's CFOs
Mat<cd> comb_coherent(const std::vector<Mat<cd>>& Cs, const std::vector<int>& js, const std::vector<double>& f2,
                      int decim = 1) {
    const size_t k = f2.size() / Cs[0].rows;
    const size_t D = static_cast<size_t>(decim);
    size_t n = SIZE_MAX;
    for (size_t i = 0; i < js.size(); ++i) {
        const size_t off = static_cast<size_t>(js[i] * NSYM) / D;
        n = std::min(n, Cs[i].cols - std::min(Cs[i].cols, off));
    }
    Mat<cd> C(f2.size(), n);
    for (size_t i = 0; i < f2.size(); ++i)
        for (size_t c = 0; c < js.size(); ++c) {
            const cd rot = std::exp(cd(0.0, -2 * PI * f2[i] * js[c] * NSYM / FS));
            const cd* src = Cs[c][i / k] + static_cast<size_t>(js[c] * NSYM) / D;
            cd* dst = C[i];
            for (size_t t = 0; t < n; ++t) dst[t] += src[t] * rot;
        }
    return C;
}

std::optional<Lock> find_copy_comb(std::span<const double> x, std::string_view band, const Accept* accept, double* peak) {
    if (!copies(band)) {  // no copy to find: copy_llr reads a copy band's frame
        if (peak) *peak = 0.0;
        return std::nullopt;
    }
    const auto z0 = waveform::to_baseband(x);
    const auto freqs = waveform::cfo_grid();
    const auto filters = comb_filters(band);
    std::vector<int> js;
    std::vector<Mat<cd>> Cs;
    for (const auto& [j, t] : filters) {
        js.push_back(j);
        Cs.push_back(waveform::repeat_corrs(z0, t, freqs));
    }
    if (const auto intra = sc_intra_spec(config::band(band).nc)) {
        const auto f2 = intra_freqs(intra->n_combs);
        const auto [P1, level] = intra_frame_prods(Cs, *intra, 0);
        const Mat<cd> D = sum_frames(P1.row_ptrs(), P1.cols, FRAME_SAMPLES, COPY_PAIRS);
        const int cfo_lag = (intra->syms[1] - intra->syms[0]) * NSYM;
        const auto w = copy_row_weights(comb_row_levels(Cs), intra->n_combs);
        return find_copy_from(x, z0, band, accept, D.row_ptrs(), D.cols, level, peak, nullptr, f2, true, cfo_lag, 0, 1,
                              w);
    }
    const auto f2 = comb_freqs();
    const Mat<cd> C = comb_coherent(Cs, js, f2);
    const double level = waveform::noise_level(C);
    const Mat<cd> D = waveform::pair_products(C.row_ptrs(), C.cols, FRAME_SAMPLES, COPY_PAIRS);
    return find_copy_from(x, z0, band, accept, D.row_ptrs(), D.cols, level, peak, nullptr, f2, true);
}

}  // namespace

CombStream::CombStream(std::string_view band)
    : band_(band), intra_(sc_intra_spec(config::band(band).nc)), decim_(SC_COPY_DECIM) {
    for (auto& [j, t] : comb_filters(band)) {
        js_.push_back(j);
        filters_.push_back(std::move(t));
    }
    if (filters_.size() >= 2) {
        const auto tf = waveform::unit_template(waveform::band(band));
        const double sc = std::sqrt(static_cast<double>(filters_.size()));
        double max_err = 0.0;
        for (size_t n = 0; n < tf.size(); ++n) {
            cd s = 0.0;
            for (const auto& t : filters_) s += t[n];
            max_err = std::max(max_err, std::abs(s - sc * tf[n]));
        }
        can_use_full_ = max_err < 1e-12;
    }
    if (intra_) {
        freqs_ = intra_freqs(intra_->n_combs);
        lag = intra_->n_combs == 1 ? FRAME_SAMPLES
                                   : static_cast<std::int64_t>(intra_->syms[static_cast<size_t>(intra_->n_syms - 1)]) * NSYM;
    } else {
        freqs_ = comb_freqs();
        lag = static_cast<std::int64_t>(*std::max_element(js_.begin(), js_.end())) * NSYM;
    }
    reset();
}

void CombStream::reset() {
    tail_.clear();
    C_.assign(freqs_.size(), {});
    D_.assign(freqs_.size(), {});
    c_off_ = d_off_ = 0;
    c0 = d0 = fed = 0;
    levels_.clear();
}

void CombStream::restart_at(std::int64_t pos) {
    reset();
    fed = pos;
}

void CombStream::skip_to(std::int64_t pos) {  // as StreamDetector::skip_to: the level history kept
    tail_.clear();
    for (auto& r : C_) r.clear();
    for (auto& r : D_) r.clear();
    c_off_ = d_off_ = 0;
    c0 = d0 = fed = pos;
}

void CombStream::feed(std::span<const cd> z_new, const waveform::StreamDetector* full) {
    const auto added = static_cast<std::int64_t>(z_new.size());
    fed += added;
    std::vector<cd> z(std::move(tail_));
    z.insert(z.end(), z_new.begin(), z_new.end());
    const auto len = static_cast<std::int64_t>(z.size());
    const std::int64_t zs0 = fed - len;
    const std::int64_t D = static_cast<std::int64_t>(decim_);
    const std::int64_t lead = (D - (zs0 % D)) % D;
    const std::int64_t len_a = len - lead;
    const std::int64_t n_raw = len_a - M + 1 - lag;
    if (n_raw < D) {
        tail_ = std::move(z);
        return;
    }
    const std::int64_t n_d = n_raw / D;
    const std::int64_t n = n_d * D;
    const std::int64_t zs = zs0 + lead;
    const auto z_a = std::span<const cd>(z).subspan(static_cast<std::size_t>(lead));
    const auto grid = waveform::cfo_grid();
    const std::size_t n_c_raw = static_cast<std::size_t>(len_a - M + 1);
    const std::size_t n_c_d = (n_c_raw + static_cast<std::size_t>(D) - 1) / static_cast<std::size_t>(D);
    std::vector<Mat<cd>> Cs;
    if (can_use_full_ && full && full->bins() == grid.size()) {
        const auto fo = full->outputs_from(zs);
        const std::size_t need_full = (n_c_d - 1) * static_cast<std::size_t>(D) + 1;
        if (fo.start == zs && fo.cols >= need_full && fo.rows.size() == grid.size()) {
            const size_t K = filters_.size();
            Cs = waveform::repeat_corrs_multi(z_a, std::span<const std::vector<cd>>(filters_).first(K - 1), grid, decim_);
            const double sc = std::sqrt(static_cast<double>(K));
            Mat<cd> last(grid.size(), n_c_d);
            for (size_t b = 0; b < grid.size(); ++b) {
                const cd* frow = fo.rows[b];
                cd* dst = last[b];
                if (K == 3) {
                    const cd* r0 = Cs[0][b];
                    const cd* r1 = Cs[1][b];
                    for (size_t t = 0; t < n_c_d; ++t) dst[t] = sc * frow[t * static_cast<size_t>(D)] - r0[t] - r1[t];
                } else {
                    for (size_t t = 0; t < n_c_d; ++t) dst[t] = sc * frow[t * static_cast<size_t>(D)];
                    for (size_t c = 0; c + 1 < K; ++c) {
                        const cd* rc = Cs[c][b];
                        for (size_t t = 0; t < n_c_d; ++t) dst[t] -= rc[t];
                    }
                }
            }
            Cs.push_back(std::move(last));
        }
    }
    if (Cs.empty()) Cs = waveform::repeat_corrs_multi(z_a, filters_, grid, decim_);
    Mat<cd> C;
    std::vector<double> q;
    std::vector<double> p;
    const std::int64_t qwin_d = std::max<std::int64_t>(added, 2000) / D;
    if (intra_) {
        C = intra_frame_prods(Cs, *intra_, zs, false, decim_).first;
        const double scale = std::sqrt(static_cast<double>(intra_->n_combs == 1 ? 2 : intra_->n_syms - 1));
        const size_t nb = Cs[0].rows;
        q.assign(nb, INFINITY);
        const double inv_log = 1.0 / -std::log(1 - waveform::NOISE_QUANTILE);
        for (const auto& Cc : Cs) {
            const auto from = static_cast<std::size_t>(std::max<std::int64_t>(0, static_cast<int64_t>(Cc.cols) - qwin_d));
            const std::size_t np = Cc.cols - from;
            p.resize(np);
            const auto k = static_cast<std::size_t>(waveform::NOISE_QUANTILE * static_cast<double>(np - 1));
            for (std::size_t i = 0; i < nb; ++i) {
                const cd* row = Cc[i] + from;
                for (std::size_t t = 0; t < np; ++t) p[t] = row[t].real() * row[t].real() + row[t].imag() * row[t].imag();
                std::nth_element(p.begin(), p.begin() + static_cast<std::ptrdiff_t>(k), p.end());
                q[i] = std::min(q[i], scale * p[k] * inv_log);
            }
        }
    } else {
        C = comb_coherent(Cs, js_, freqs_, decim_);
        const auto from = static_cast<std::size_t>(std::max<std::int64_t>(0, static_cast<std::int64_t>(C.cols) - qwin_d));
        const std::size_t np = C.cols - from;
        p.resize(np);
        const auto k = static_cast<std::size_t>(waveform::NOISE_QUANTILE * static_cast<double>(np - 1));
        const double inv_log = 1.0 / -std::log(1 - waveform::NOISE_QUANTILE);
        q.resize(C.rows);
        for (std::size_t i = 0; i < C.rows; ++i) {
            const cd* row = C[i] + from;
            for (std::size_t t = 0; t < np; ++t) p[t] = row[t].real() * row[t].real() + row[t].imag() * row[t].imag();
            std::nth_element(p.begin(), p.begin() + static_cast<std::ptrdiff_t>(k), p.end());
            q[i] = p[k] * inv_log;
        }
    }
    const std::size_t take = std::min(C.cols, static_cast<std::size_t>(n_d));
    if (C_[0].size() == c_off_) {
        for (auto& r : C_) r.clear();
        c_off_ = 0;
        c0 = zs;
    }
    for (std::size_t i = 0; i < C.rows; ++i) C_[i].insert(C_[i].end(), C[i], C[i] + take);
    levels_.push_back(std::move(q));
    if (levels_.size() > static_cast<std::size_t>(CHUNKS)) levels_.erase(levels_.begin());
    tail_.assign(z.begin() + lead + n, z.end());
}

std::vector<double> CombStream::row_levels() const {
    if (levels_.empty()) return {};
    std::vector<double> v(levels_.size()), out(levels_[0].size());
    for (std::size_t b = 0; b < levels_[0].size(); ++b) {
        for (std::size_t k = 0; k < levels_.size(); ++k) v[k] = levels_[k][b];
        std::sort(v.begin(), v.end());
        const std::size_t h = v.size() / 2;
        out[b] = v.size() % 2 ? v[h] : (v[h - 1] + v[h]) / 2;
    }
    return out;
}

std::optional<double> CombStream::level() const {
    if (levels_.empty()) return std::nullopt;
    const auto r = row_levels();
    return *std::min_element(r.begin(), r.end());
}

void CombStream::trim(std::int64_t start) {
    const std::int64_t D = static_cast<std::int64_t>(decim_);
    const std::size_t nc = C_[0].size() - c_off_;
    const auto k = static_cast<std::size_t>(std::clamp<std::int64_t>((start - c0) / D, 0, static_cast<std::int64_t>(nc)));
    c_off_ += k;
    c0 += static_cast<std::int64_t>(k) * D;
    const std::size_t nd = D_[0].size() - d_off_;
    const auto kd = static_cast<std::size_t>(
        std::clamp<std::int64_t>((start - KEEP_BEHIND - d0) / D, 0, static_cast<std::int64_t>(nd)));
    d_off_ += kd;
    d0 += static_cast<std::int64_t>(kd) * D;
    auto compact = [](std::vector<std::vector<cd>>& rows, std::size_t& off) {
        if (off <= rows[0].size() / 2) return;
        for (auto& r : rows) r.erase(r.begin(), r.begin() + static_cast<std::ptrdiff_t>(off));
        off = 0;
    };
    compact(C_, c_off_);
    compact(D_, d_off_);
}

void CombStream::extend() {
    const std::int64_t D = static_cast<std::int64_t>(decim_);
    const std::size_t FR_d = static_cast<std::size_t>(FRAME_SAMPLES / decim_);
    if (d0 + static_cast<std::int64_t>(D_[0].size() - d_off_) * D < c0) {  // not extended across a trim: start at C's
        for (auto& r : D_) r.clear();
        d_off_ = 0;
        d0 = c0;
    }
    const auto have = static_cast<std::size_t>((d0 - c0) / D) + (D_[0].size() - d_off_);
    const std::size_t nc = C_[0].size() - c_off_;
    if (intra_) {
        const std::size_t PF_d = static_cast<std::size_t>(COPY_PAIRS - 1) * FR_d;
        if (nc > have + PF_d) {
            std::vector<const cd*> rows;
            for (const auto& r : C_) rows.push_back(r.data() + c_off_ + have);
            const Mat<cd> Dm = sum_frames(rows, nc - have, FR_d, COPY_PAIRS);
            for (std::size_t i = 0; i < D_.size(); ++i) D_[i].insert(D_[i].end(), Dm[i], Dm[i] + Dm.cols);
        }
        return;
    }
    const std::size_t PF_d = static_cast<std::size_t>(COPY_PAIRS) * FR_d;
    if (nc > have + PF_d) {
        std::vector<const cd*> rows;
        for (const auto& r : C_) rows.push_back(r.data() + c_off_ + have);
        const Mat<cd> Dm = waveform::pair_products(rows, nc - have, static_cast<int>(FR_d), COPY_PAIRS);
        for (std::size_t i = 0; i < D_.size(); ++i) D_[i].insert(D_[i].end(), Dm[i], Dm[i] + Dm.cols);
    }
}

CombStream::Products CombStream::products_from(std::int64_t start) {
    extend();
    const std::int64_t D = static_cast<std::int64_t>(decim_);
    const std::int64_t want = std::max(c0, start);
    const std::size_t nd = D_[0].size() - d_off_;
    const std::int64_t diff = want - d0;
    const auto k = static_cast<std::size_t>(
        std::clamp<std::int64_t>(diff > 0 ? (diff + D - 1) / D : 0, 0, static_cast<std::int64_t>(nd)));
    Products o{d0 + static_cast<std::int64_t>(k) * D, {}, nd - k};
    for (const auto& r : D_) o.rows.push_back(r.data() + d_off_ + k);
    return o;
}

CombStream::Products CombStream::kept_products() {
    extend();
    Products o{d0, {}, D_[0].size() - d_off_};
    for (const auto& r : D_) o.rows.push_back(r.data() + d_off_);
    return o;
}

std::optional<Lock> find_copy_comb_stream(std::span<const double> x, std::string_view band, const Accept* accept,
                                          std::span<const cd* const> D_rows, std::size_t dcols, double level,
                                          double* peak, const Mat<cd>* fold, std::size_t fold_shift, int decim,
                                          std::span<const double> row_level) {
    if (!copies(band)) {
        if (peak) *peak = 0.0;
        return std::nullopt;
    }
    if (const auto intra = sc_intra_spec(config::band(band).nc)) {
        const auto f2 = intra_freqs(intra->n_combs);
        const int cfo_lag = (intra->syms[1] - intra->syms[0]) * NSYM;
        const auto w = copy_row_weights(row_level, intra->n_combs);
        return find_copy_from(x, {}, band, accept, D_rows, dcols, level, peak, fold, f2, true, cfo_lag, fold_shift, decim,
                              w);
    }
    return find_copy_from(x, {}, band, accept, D_rows, dcols, level, peak, fold, comb_freqs(), true, FRAME_SAMPLES,
                          fold_shift, decim);
}

std::optional<Lock> find_copy(std::span<const double> x, std::string_view band, const Accept* accept,
                              const Mat<cd>* C_in, std::optional<double> level, double* peak) {
    if (!C_in && level) throw std::invalid_argument("find_copy: level without C");
    if (sc_band(band)) return find_copy_comb(x, band, accept, peak);
    if (!C_in) return find_copy(x, band, accept, std::span<const cd* const>{}, 0, level, peak);
    const Mat<cd> D = waveform::pair_products(C_in->row_ptrs(), C_in->cols, FRAME_SAMPLES, COPY_PAIRS);
    return find_copy(x, band, accept, D.row_ptrs(), D.cols, level, peak);
}

std::optional<Lock> find_copy(std::span<const double> x, std::string_view band, const Accept* accept,
                              std::span<const cd* const> D_rows, size_t dcols, std::optional<double> level,
                              double* peak, const Mat<cd>* fold_in, std::size_t fold_shift) {
    if (!D_rows.empty() && !level) throw std::invalid_argument("find_copy: C without level");
    if (sc_band(band)) return find_copy_comb(x, band, accept, peak);
    const auto& b = waveform::band(band);
    const auto freqs = waveform::cfo_grid();
    std::vector<cd> z0;
    Mat<cd> D_own;
    std::vector<const cd*> own_rows;
    if (D_rows.empty()) {
        if (level) throw std::invalid_argument("find_copy: level without C");
        z0 = waveform::to_baseband(x);
        const Mat<cd> C = waveform::repeat_corrs(z0, waveform::unit_template(b), freqs);
        level = waveform::noise_level(C);
        D_own = waveform::pair_products(C.row_ptrs(), C.cols, FRAME_SAMPLES, COPY_PAIRS);
        own_rows = D_own.row_ptrs();
        D_rows = own_rows;
        dcols = D_own.cols;
    } else if (!level) {
        throw std::invalid_argument("find_copy: C without level");
    }
    return find_copy_from(x, z0, band, accept, D_rows, dcols, *level, peak, fold_in, freqs, false, FRAME_SAMPLES, fold_shift);
}

namespace {

std::optional<Lock> find_copy_from(std::span<const double> x, std::span<const cd> z0, std::string_view band,
                                   const Accept* accept, std::span<const cd* const> D_rows, size_t dcols, double level,
                                   double* peak, const Mat<cd>* fold_in, std::span<const double> freqs, bool comb,
                                   int cfo_lag, size_t fold_shift, int decim, std::span<const double> row_w) {
    const size_t F = D_rows.size();
    const size_t FR = static_cast<size_t>(FRAME_SAMPLES / decim);
    const int64_t m = static_cast<int64_t>(dcols) / static_cast<int64_t>(FR);
    if (m < 1) return std::nullopt;
    Mat<cd> own;
    if (!fold_in) {
        own = Mat<cd>(F, FR);
        for (size_t f = 0; f < F; ++f) {
            const cd* d = D_rows[f];
            for (size_t i = 0; i < static_cast<size_t>(m); ++i)
                for (size_t ph = 0; ph < FR; ++ph) own[f][ph] = i ? own[f][ph] + d[i * FR + ph] : d[ph];
        }
    }
    const Mat<cd>& src = fold_in ? *fold_in : own;
    std::vector<cd> row_mean(F, cd(0.0, 0.0));
    std::vector<double> mag2(FR, 0.0);
    for (size_t f = 0; f < F; ++f) {
        const cd* row = src[f];
        if (comb) {  // a steady tone's products are the same at every phase: each CFO bin's mean out (modem.py's reason)
            double sr = 0.0, si = 0.0;
            for (size_t q = 0; q < FR; ++q) {
                sr += row[q].real();
                si += row[q].imag();
            }
            const double mr = sr / static_cast<double>(FR), mi = si / static_cast<double>(FR);
            row_mean[f] = cd(mr, mi);
            const double w2 = row_w.empty() ? 1.0 : row_w[f] * row_w[f];  // modem._copy_row_weights
            for (size_t q = 0; q < FR; ++q) {
                const double dr = row[q].real() - mr, di = row[q].imag() - mi;
                mag2[q] = std::max(mag2[q], w2 * (dr * dr + di * di));
            }
        } else {
            for (size_t q = 0; q < FR; ++q) {
                const double dr = row[q].real(), di = row[q].imag();
                mag2[q] = std::max(mag2[q], dr * dr + di * di);
            }
        }
    }
    const double pk = std::sqrt(*std::max_element(mag2.begin(), mag2.end())) /
                      (level * std::sqrt(static_cast<double>(m * COPY_PAIRS)));
    if (peak) *peak = pk;
    if (pk < COPY_DETECT) return std::nullopt;
    // Only convert x to baseband and unshift mag once the peak clears COPY_DETECT.
    std::vector<cd> z0_own;
    if (z0.empty()) {
        z0_own = waveform::to_baseband(x);
        z0 = z0_own;
    }
    std::vector<double> mag(FR);
    for (size_t ph = 0; ph < FR; ++ph) mag[ph] = std::sqrt(mag2[(fold_shift + ph) % FR]);
    std::vector<int> order(FR);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int c) { return -mag[static_cast<size_t>(a)] < -mag[static_cast<size_t>(c)]; });
    std::vector<int> grids;
    for (int ph_d : order) {
        const int ph = ph_d * decim;
        if (std::all_of(grids.begin(), grids.end(), [&](int g) {
                return std::min(std::abs(ph - g), FRAME_SAMPLES - std::abs(ph - g)) >= NCP;
            })) {
            grids.push_back(ph);
            if (static_cast<int>(grids.size()) == COPY_GRIDS) break;
        }
    }
    const auto& sb = config::band(band);
    const double floor = floor_of(band, accept);
    const int64_t len = static_cast<int64_t>(z0.size());
    std::vector<Lock> found;
    for (int ph : grids) {
        const size_t col = (fold_shift + static_cast<size_t>(ph / decim)) % FR;
        // the strongest bin, and weighed (row_w) the strongest after weighing first: modem.py's reason
        const auto best_bin = [&](bool weigh) {
            size_t i = 0;
            double best_n = -1.0;
            for (size_t f = 0; f < F; ++f) {
                const double cur_n = (weigh ? row_w[f] * row_w[f] : 1.0) * std::norm(src[f][col] - row_mean[f]);
                if (cur_n > best_n) {
                    best_n = cur_n;
                    i = f;
                }
            }
            return i;
        };
        std::vector<size_t> bins{best_bin(false)};
        if (!row_w.empty()) {
            const size_t j = best_bin(true);
            if (j != bins[0]) bins.insert(bins.begin(), j);
        }
        std::vector<double> cands;
        for (size_t i : bins)
            for (double f : cfo_aliases(src[i][col] - row_mean[i], freqs[i], cfo_lag)) cands.push_back(f);
        for (double f : cands) {
            const auto z = waveform::freq_correct(z0, f);
            for (int early : COPY_EARLIER) {
                for (int64_t pc = pymod(ph - early - NCP, FRAME_SAMPLES); pc < len - FRAME_SAMPLES; pc += FRAME_SAMPLES) {
                    const auto llr = copy_llr(z, pc, band, sb.header_syms);
                    if (!llr) continue;
                    const auto h = decode_header(*llr, band, accept);
                    const auto kc = copy_frame(band, h.n_cw * h.spec->frames_per_cw);
                    const int64_t p0 = pc - static_cast<int64_t>(kc.value_or(0)) * FRAME_SAMPLES;
                    const int64_t start = p0 - header_samples(band) - preamble_samples(sb);
                    if (kc && h.score >= floor && start >= 0) {
                        Lock l;
                        l.spec = h.spec;
                        l.n_cw = h.n_cw;
                        l.start = start;
                        l.score = h.score;
                        l.band = sb.name;
                        l.end = burst_end(p0, *h.spec, h.n_cw);
                        l.p0 = p0;
                        l.cfo = f;
                        l.copy = CopyRef{h.word, pc};
                        found.push_back(l);
                    }
                }
            }
        }
    }
    std::stable_sort(found.begin(), found.end(), [](const Lock& a, const Lock& c) { return -a.score < -c.score; });
    for (const auto& l : found) {
        double pat = 0.0;
        std::vector<double> c;
        if (comb) std::tie(c, pat) = sc::copy_rows(z0, l);
        else c = pilot_coherence(x, l, 10000, false, &pat);
        if (c.size() >= 2 && dsp::pairwise_sum(std::span<const double>(c)) / static_cast<double>(c.size()) >= COPY_COHERENCE &&
            pat >= COPY_PATTERN)
            return l;
    }
    return std::nullopt;
}

}  // namespace

HeaderRead copy_header(std::span<const cd> z, const Lock& lock) {
    const auto& b = waveform::band(lock.band);
    auto hd = read_header(z, lock.start, lock.band);
    hd.hdr = Header{lock.copy->word, lock.spec, lock.n_cw, lock.score};
    hd.valid = true;
    hd.h_pre = b.demod_window(z, lock.copy->pc + NCP, HB);
    const auto ref = sc_band(lock.band) ? copy_open(b) : b.pilot;  // (a scattered-pilot copy frame opens on copy_open)
    for (size_t k = 0; k < hd.h_pre.size(); ++k) hd.h_pre[k] /= ref[k];
    return hd;
}

ClipConsts clip_consts(const Spec& spec) {
    const auto& c = CLIP_SUBMODE[config::row(spec)];
    return {{{1, c.gain_1f}}, c.gain, c.ratio};
}

ClipConsts clip_consts(std::string_view name) {
    for (size_t i = 0; i < BANDS.size(); ++i)
        if (BANDS[i].name == name) return {{{1, CLIP_BAND[i].gain_1f}}, CLIP_BAND[i].gain, CLIP_BAND[i].ratio};
    throw std::out_of_range("no band " + std::string(name));
}

namespace {
DataEstimate data_channel_whole(const Mat<cd>& h_pilot, equalizer::Support support, std::string_view band,
                                double n0_pre, const ClipConsts* clip, std::span<const double> n0_pre_k,
                                int n_frames);
}  // namespace

DataEstimate data_channel(const Mat<cd>& h_pilot, equalizer::Support support, std::string_view band, double n0_pre,
                          const ClipConsts* clip, std::span<const double> n0_pre_k, int n_frames) {
    const int n_air = static_cast<int>(h_pilot.rows) - 1, W = config::EST_WINDOW;
    if (!(W > 0 && n_air > W)) return data_channel_whole(h_pilot, support, band, n0_pre, clip, n0_pre_k, n_frames);
    // each window of frames from its own pilots and one either side (Python's data_channel, EST_WINDOW)
    const size_t nc = h_pilot.cols, S = DATA_SYMS_PER_FRAME;
    DataEstimate est;
    std::vector<cd> h;
    std::vector<double> mse, n0_k;
    double n0 = 0, n0_snr = 0, p_sig = 0, spread = 0;
    int parts = 0;
    for (int a = 0; a < n_air; a += W) {
        const int lo = std::max(0, a - 1), hi = std::min(n_air, a + W + 1), count = std::min(W, n_air - a);
        Mat<cd> sub(static_cast<size_t>(hi - lo + 1), nc);
        std::copy(h_pilot[static_cast<size_t>(lo)], h_pilot[static_cast<size_t>(lo)] + sub.data.size(), sub.data.begin());
        auto e = data_channel_whole(sub, support, band, n0_pre, clip, n0_pre_k, n_frames < 0 ? n_air : n_frames);
        const size_t r0 = static_cast<size_t>(a - lo) * S * nc, r1 = r0 + static_cast<size_t>(count) * S * nc;
        h.insert(h.end(), e.h.data.begin() + static_cast<std::ptrdiff_t>(r0), e.h.data.begin() + static_cast<std::ptrdiff_t>(r1));
        mse.insert(mse.end(), e.mse.data.begin() + static_cast<std::ptrdiff_t>(r0),
                   e.mse.data.begin() + static_cast<std::ptrdiff_t>(r1));
        for (int f = 0; f < count; ++f)
            for (size_t k = 0; k < nc; ++k) n0_k.push_back(e.n0_k.empty() ? e.n0 : e.n0_k[k]);
        n0 += e.n0, n0_snr += e.n0_snr, p_sig += e.p_sig, spread += e.spread_hz, ++parts;
        if (a == 0) est = std::move(e);  // the rest of the fields: the first window's (Python's dict(parts[0]))
    }
    est.h = Mat<cd>(h.size() / nc, nc);
    est.h.data = std::move(h);
    est.mse = Mat<double>(mse.size() / nc, nc);
    est.mse.data = std::move(mse);
    est.n0_k = std::move(n0_k);
    est.n0 = n0 / parts, est.n0_snr = n0_snr / parts, est.p_sig = p_sig / parts, est.spread_hz = spread / parts;
    est.n_f = n_air;
    return est;
}

namespace {
DataEstimate data_channel_whole(const Mat<cd>& h_pilot, equalizer::Support support, std::string_view band,
                                double n0_pre, const ClipConsts* clip, std::span<const double> n0_pre_k,
                                int n_frames) {
    DataEstimate est;
    static_cast<equalizer::Estimate&>(est) = equalizer::estimate(h_pilot, support, bb_of(band), n0_pre, n0_pre_k);
    const ClipConsts own = clip ? ClipConsts{} : clip_consts(band);
    const ClipConsts& c = clip ? *clip : own;
    est.clip_ratio = c.ratio;
    const double g = c.gain_for(n_frames < 0 ? static_cast<int>(h_pilot.rows) - 1 : n_frames);
    for (auto& v : est.h.data) v *= g;
    for (auto& v : est.mse.data) v *= g * g;
    est.gain = g;
    est.band = config::band(band).name;
    return est;
}
}  // namespace

double resolve_alias(double fine, double coarse) {
    return fine + std::nearbyint((coarse - fine) * equalizer::FRAME_S) / equalizer::FRAME_S;
}

namespace {
Received receive_from(HeaderRead hd, waveform::Acquisition acq, std::vector<cd> z, int64_t len, bool partial, bool known);
}  // namespace

Received receive(std::span<const double> x, std::span<const std::string_view> bands, const Accept* accept,
                 std::optional<int64_t> head, const Lock* copy, bool partial) {
    const auto z0 = waveform::to_baseband(x);
    const auto len = static_cast<int64_t>(z0.size());
    HeaderRead hd;
    waveform::Acquisition acq;
    std::vector<cd> z;
    if (copy) {
        if (copy->end > len) throw SyncError("burst runs past the buffer");
        z = waveform::freq_correct(z0, copy->cfo);
        hd = copy_header(z, *copy);
        acq = {copy->start, copy->cfo, 0.0, {}};
    } else if (!head) {
        auto b = best_header(z0, bands, true, accept);
        hd = std::move(b.hd);
        acq = std::move(b.acq);
        z = *b.z;
    } else {
        auto b = best_header(pyslice(std::span<const cd>(z0), 0, *head), bands, false, accept, {}, true);
        hd = std::move(b.hd);
        acq = std::move(b.acq);
        z = waveform::freq_correct(z0, acq.freq_offset);
        if (!partial && burst_end(hd.p0, *hd.hdr.spec, hd.hdr.n_cw) > len) throw SyncError("burst runs past the buffer");
    }
    return receive_from(std::move(hd), std::move(acq), std::move(z), len, partial, false);
}

Received receive_known(std::span<const double> x, int64_t start, double cfo, const Spec& spec, int n_cw) {
    const auto z0 = waveform::to_baseband(x);
    const auto len = static_cast<int64_t>(z0.size());
    auto z = waveform::freq_correct(z0, cfo);
    auto hd = read_header(z, start, spec.sync_band);
    const int v = spec.index << cw_bits(spec.sync_band) | (n_cw - 1);
    hd.hdr.word = v << 6 | crc6(v);
    hd.hdr.spec = &spec;
    hd.hdr.n_cw = n_cw;
    hd.valid = true;
    if (burst_end(hd.p0, spec, n_cw) > len) throw SyncError("burst runs past the buffer");
    return receive_from(std::move(hd), {start, cfo, 0.0, {}}, std::move(z), len, false, true);
}

namespace {

// receive() once the header is read: hd, acq and z (the baseband corrected by acq's CFO). known: the burst's
// timing was known (receive(known=...)): the residual CFO's alias nearest no change.
Received receive_from(HeaderRead hd, waveform::Acquisition acq, std::vector<cd> z, int64_t len, bool partial, bool known) {
    const std::string_view sband = hd.band;  // the sync band; frames are on spec.band's carriers
    const auto& b = waveform::band(sband);
    const size_t nc = static_cast<size_t>(b.nc());
    const Spec& spec = *hd.hdr.spec;
    const int n_cw = hd.hdr.n_cw;
    const std::string_view db = spec.band;
    const int n_f = n_cw * spec.frames_per_cw;
    auto kc = copy_frame(sband, n_f);
    int n_air = n_f + kc.has_value();
    if (partial) {  // the frames in so far, each with its closing pilot (and room for the window's shift)
        n_air = static_cast<int>(std::min<int64_t>(n_air, std::max<int64_t>(0, (len - hd.p0 - NSYM - 2 * NCP) / FRAME_SAMPLES)));
        if (n_air < 1) throw SyncError("no whole frame yet");
        if (kc && *kc >= n_air) kc.reset();  // its copy not in yet
    }
    const int n_in = n_air - kc.has_value();  // data frames in
    const double phi_ref = bin_phase_step(hd.h_pre);
    const bool sc = sc_spec(spec, n_f), scb = sc_band(db);
    if (sc && partial)
        // a burst still arriving has no closing pilot symbol, and the frame after the last one in starts with
        // scattered cells, not a pilot row (Python's ponytail): decoding while it arrives waits for the end
        throw SyncError("scattered pilots: the whole burst only");
    const bool sc_intra = sc && sc_intra_spec(config::band(db).nc).has_value();
    Mat<int> pil, pil_k;
    if (sc_intra) {
        std::tie(pil, pil_k) = sc_pil_intra(db, n_air, kc);
    } else if (sc) {
        pil = sc_pil(db, n_air, kc);
    }
    const std::optional<int> close_at = scb ? kc : std::optional<int>{};

    // Residual CFO's alias from the known header symbols (modem.py's notes)
    const auto layout = header_layout(sband);
    const auto hsym = constellation::modulate(codeword(hd.hdr.word, sband), qpsk());
    Mat<cd> ref_syms(layout.size(), nc);
    std::vector<std::vector<cd>> hrows;
    for (size_t r = 0, d = 0; r < layout.size(); ++r) {
        if (layout[r]) {
            std::copy(b.pilot.begin(), b.pilot.end(), ref_syms[r]);
        } else {
            std::copy(hsym.begin() + static_cast<std::ptrdiff_t>(d * nc), hsym.begin() + static_cast<std::ptrdiff_t>((d + 1) * nc),
                      ref_syms[r]);
            hrows.emplace_back(ref_syms[r], ref_syms[r] + nc);
            ++d;
        }
    }
    auto times_conj = [&](std::span<const cd> y, std::span<const cd> r) {
        std::vector<cd> o(y.size());
        for (size_t k = 0; k < o.size(); ++k) o[k] = y[k] * std::conj(r[k]);
        return o;
    };
    std::vector<std::vector<std::vector<cd>>> runs(1);
    for (size_t r = 0; r < layout.size(); ++r) runs[0].push_back(times_conj(hd.y_all.row(r), ref_syms.row(r)));
    if (db == sband && !sc) runs[0].push_back(hd.h_first);  // (scattered: the first frame opens on data cells)
    if (kc) {
        const int64_t pc = hd.p0 + static_cast<int64_t>(*kc) * FRAME_SAMPLES;
        std::vector<std::vector<cd>> ref2;
        if (scb) {  // copy_open, the header's symbols, copy_close
            ref2.push_back(copy_open(b));
            ref2.insert(ref2.end(), hrows.begin(), hrows.end());
            ref2.push_back(copy_open(b));
        } else {
            ref2.push_back(b.pilot);
            for (int s = 0; s < DATA_SYMS_PER_FRAME; ++s)
                ref2.push_back(hrows[static_cast<size_t>(s) < hrows.size() ? static_cast<size_t>(s) : static_cast<size_t>(s) - hrows.size()]);
            ref2.push_back(b.pilot);
        }
        std::vector<std::vector<cd>> run;
        for (size_t s = 0; s < ref2.size(); ++s)
            run.push_back(times_conj(b.demod_window(z, pc + static_cast<int64_t>(s) * NSYM + NCP, HB), ref2[s]));
        runs.push_back(std::move(run));
    }
    // Leave carriers hit by a steady tone or its off-carrier sidelobes out of
    // coarse and residual_cfo (issue #330; modem.py's notes).
    const std::vector<char> noisy_sync = equalizer::noisy_carriers(hd.n0_pre_k);
    size_t hdr_rows = 0;
    for (const auto& run : runs) hdr_rows += run.size();
    Mat<cd> hdr_h(hdr_rows, nc);
    for (size_t r = 0; const auto& run : runs)
        for (const auto& row : run) {
            std::copy(row.begin(), row.end(), hdr_h[r]);
            ++r;
        }
    std::vector<char> hot_sync = equalizer::hot_carriers(hdr_h);
    if (noisy_sync.size() == nc)
        for (size_t c = 0; c < nc; ++c) hot_sync[c] = hot_sync[c] || noisy_sync[c];
    if (std::ranges::all_of(hot_sync, [](char x) { return x != 0; })) std::ranges::fill(hot_sync, 0);
    cd dsum = 0.0;
    for (const auto& run : runs)
        for (size_t i = 0; i + 1 < run.size(); ++i)
            for (size_t c = 0; c < nc; ++c)
                if (!hot_sync[c]) dsum += run[i + 1][c] * std::conj(run[i][c]);
    const double coarse = std::arg(dsum) / (2 * PI * NSYM / FS);
    // slips are searched for once, here (residual CFO left: the search is blind to a common phase), and replayed
    const Mat<int>* pp = sc ? &pil : nullptr;
    const Mat<int>* pk = sc_intra ? &pil_k : nullptr;
    auto first = demod_frames(z, hd.p0, n_air, 0, phi_ref, {}, db, {}, pp, close_at, pk);
    const auto& dbs = config::band(db);
    const auto& sbs = config::band(sband);
    const size_t k0_off = static_cast<size_t>(dbs.k0 - sbs.k0);
    const size_t nc_db = static_cast<size_t>(dbs.nc);
    std::vector<char> exclude_db(nc_db, 0);
    if (k0_off + nc_db <= noisy_sync.size()) {
        for (size_t c = 0; c < nc_db; ++c) {
            exclude_db[c] = noisy_sync[k0_off + c];
            if (db != sband) exclude_db[c] = exclude_db[c] || hot_sync[k0_off + c];
        }
    }
    double cfo_res = 0.0;
    if (sc_intra) {
        const auto [fine, cfo_dt] = sc::residual_cfo(first.raw, first.hp, close_at, n_air, db, exclude_db);
        const double ref_cfo = known ? 0.0 : coarse;
        cfo_res = fine + std::nearbyint((ref_cfo - fine) * cfo_dt) / cfo_dt;
    } else {
        const double fine = equalizer::residual_cfo(first.hp, exclude_db);
        cfo_res = resolve_alias(fine, known ? 0.0 : coarse);
    }
    const auto slips = std::move(first.slips);
    z = waveform::freq_correct(z, cfo_res);
    const auto bbd = bb_of(db);
    // the window and the reported delay on the support as before the noise
    // gate (the gear shifter was trained on it); the estimate on the gated one
    auto fr0 = demod_frames(z, hd.p0, n_air, 0, phi_ref, {}, db, slips, pp, close_at, pk);
    auto sup_rows = [&](Mat<cd> rows) {
        auto hot_s = equalizer::hot_carriers(rows);
        if (exclude_db.size() == rows.cols)
            for (size_t c = 0; c < rows.cols; ++c) hot_s[c] = hot_s[c] || exclude_db[c];
        const bool any_h = std::ranges::any_of(hot_s, [](char v) { return v != 0; });
        const bool all_h = std::ranges::all_of(hot_s, [](char v) { return v != 0; });
        if (any_h && !all_h)
            for (size_t r = 0; r < rows.rows; ++r)
                for (size_t c = 0; c < rows.cols; ++c)
                    if (hot_s[c]) rows[r][c] = 0.0;
        return rows;
    };
    const Mat<cd> hp0 = sc ? sup_rows(sc::full(z, hd, fr0.raw, fr0.hp, kc, 0, db).rows) : std::move(fr0.hp);
    auto support = equalizer::delay_support(hp0, bbd, -15.0, hd.n0_pre);
    auto seen = sc ? support : equalizer::delay_support(hp0, bbd);
    int shift = equalizer::window_shift(seen);
    auto fr = demod_frames(z, hd.p0, n_air, shift, phi_ref, {}, db, slips, pp, close_at, pk);
    support = {support.first - shift, support.second - shift};
    seen = {seen.first - shift, seen.second - shift};
    const auto clip = clip_consts(spec);
    Received out;
    if (sc) {
        support = equalizer::delay_support(sup_rows(sc::full(z, hd, fr.raw, fr.hp, kc, shift, db).rows), bbd, -15.0, hd.n0_pre);
        if (const int more = equalizer::window_shift(support)) {
            shift += more;
            fr = demod_frames(z, hd.p0, n_air, shift, phi_ref, {}, db, slips, pp, close_at, pk);
            support = {support.first - more, support.second - more};
        }
        auto e = sc::estimate(z, hd, fr.raw, fr.hp, kc, n_f, shift, support, spec);
        out.raw = std::move(e.raw);
        out.est = std::move(e.est);
        out.sc = std::move(e.info);
        seen = support;
    } else {
        out.est = data_channel(fr.hp, support, db, hd.n0_pre, &clip, db == sband ? std::span<const double>(hd.n0_pre_k)
                                                                                  : std::span<const double>(), n_f);
    }
    if (kc && !sc) {  // the copy frame's pilot stays in hp; its symbols leave the data
        auto drop = [](auto& m, size_t from, size_t count) {
            m.data.erase(m.data.begin() + static_cast<std::ptrdiff_t>(from * m.cols),
                         m.data.begin() + static_cast<std::ptrdiff_t>((from + count) * m.cols));
            m.rows -= count;
        };
        drop(fr.raw, static_cast<size_t>(*kc) * SYMS_PER_FRAME, SYMS_PER_FRAME);
        drop(out.est.h, static_cast<size_t>(*kc) * DATA_SYMS_PER_FRAME, DATA_SYMS_PER_FRAME);
        drop(out.est.mse, static_cast<size_t>(*kc) * DATA_SYMS_PER_FRAME, DATA_SYMS_PER_FRAME);
        if (out.est.n0_k.size() == static_cast<size_t>(out.est.n_f * out.est.nc)) {  // per frame (EST_WINDOW)
            const auto at = out.est.n0_k.begin() + static_cast<std::ptrdiff_t>(*kc * out.est.nc);
            out.est.n0_k.erase(at, at + out.est.nc);
        }
        out.est.n_f -= 1;
    }
    out.spec = &spec;
    out.n_cw = n_cw;
    if (!sc) out.raw = std::move(fr.raw);
    out.acq = std::move(acq);
    out.band = db;
    out.hp = std::move(fr.hp);
    out.kc = kc;
    out.cfo = out.acq.freq_offset + cfo_res;
    out.p0 = hd.p0;
    out.shift = shift;
    out.steps = std::move(fr.steps);
    out.slips = slips;
    out.phi_ref = phi_ref;
    out.support = support;
    out.seen_support = seen;
    out.preamble_start = hd.start;
    out.score = hd.hdr.score;
    out.frames_in = n_in;
    return out;
}

}  // namespace

Mat<double> noise_var(const Mat<cd>& h, const DataEstimate& est) {
    Mat<double> v(h.rows, h.cols);
    for (size_t r = 0; r < h.rows; ++r)
        for (size_t k = 0; k < h.cols; ++k)
            v[r][k] = est.n0_at(r, k) + est.clip_ratio * sq_abs(h[r][k]);
    return v;
}

std::vector<double> soft_bits(const Mat<cd>& raw, const Mat<cd>& h, const Mat<double>& var, const Spec& spec) {
    const size_t nc = raw.cols, n_f = raw.rows / SYMS_PER_FRAME;
    if (h.rows != n_f * DATA_SYMS_PER_FRAME || var.rows != h.rows || h.cols != nc || var.cols != nc)
        throw std::invalid_argument("soft_bits: raw, h, var shapes disagree");
    std::vector<cd> y;
    y.reserve(h.data.size());
    for (size_t f = 0; f < n_f; ++f)
        for (int s = 1; s < SYMS_PER_FRAME; ++s) y.insert(y.end(), raw[f * SYMS_PER_FRAME + static_cast<size_t>(s)],
                                                         raw[f * SYMS_PER_FRAME + static_cast<size_t>(s)] + nc);
    return constellation::llr(y, h.data, var.data, points_of(spec));
}

std::vector<double> symbol_noise_scale(const Received& r, const Mat<cd>& h, const Mat<double>& var) {
    const auto& pts = points_of(*r.spec).points;
    std::vector<double> ratio(h.rows);
    for (size_t row = 0; row < h.rows; ++row) {
        const cd* y = data_row(r.raw, row);
        double s = 0;
        for (size_t k = 0; k < h.cols; ++k) {
            double d = std::numeric_limits<double>::infinity();
            for (const cd& p : pts) d = std::min(d, std::abs(y[k] - h[row][k] * p));
            s += d * d / var[row][k];
        }
        ratio[row] = s / static_cast<double>(h.cols);
    }
    std::vector<double> sorted = ratio;  // numpy's median: the mean of the middle two for an even count
    std::sort(sorted.begin(), sorted.end());
    const size_t m = sorted.size() / 2;
    const double med = sorted.empty() ? 1.0 : sorted.size() % 2 ? sorted[m] : 0.5 * (sorted[m - 1] + sorted[m]);
    for (auto& x : ratio) x = std::max(1.0, x / std::max(med, 1e-12));
    return ratio;
}

Mat<double> data_var(const Received& r, const DataEstimate& est) {
    auto var = noise_var(est.h, est);
    for (size_t i = 0; i < var.data.size(); ++i) var.data[i] += est.mse.data[i];
    if constexpr (config::SYMBOL_NOISE) {
        const auto scale = symbol_noise_scale(r, est.h, var);
        for (size_t row = 0; row < var.rows; ++row)
            for (size_t k = 0; k < var.cols; ++k) var[row][k] *= scale[row];
    }
    return var;
}

Mat<double> burst_soft(const Received& r, const DataEstimate& est) {
    const auto var = data_var(r, est);
    Mat<double> out;
    out.data = codes::despread<double>(soft_bits(r.raw, est.h, var, *r.spec), r.n_cw, r.spec->bits_per_cu,
                                       spread_group(*r.spec));
    out.rows = static_cast<size_t>(r.n_cw);
    out.cols = out.data.size() / out.rows;
    return out;
}

std::vector<size_t> slot_cells(const Received& r, int slot) {
    const size_t n = static_cast<size_t>(r.n_cw), syms = n * static_cast<size_t>(r.spec->coded_bits / r.spec->bits_per_cu);
    std::vector<size_t> idx(syms);
    std::iota(idx.begin(), idx.end(), size_t{0});
    const auto cells = codes::despread<size_t>(idx, r.n_cw, 1, spread_group(*r.spec));
    const auto at = cells.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(slot) * (syms / n));
    return {at, at + static_cast<std::ptrdiff_t>(syms / n)};
}

std::vector<int> ready_slots(const Received& r, int margin) {
    const int frames = r.partial() ? r.frames_in - margin : r.n_frames();
    const auto have = static_cast<int64_t>(frames) * DATA_SYMS_PER_FRAME * waveform::band(r.spec->band).nc();
    const size_t n = static_cast<size_t>(r.n_cw), syms = n * static_cast<size_t>(r.spec->coded_bits / r.spec->bits_per_cu);
    std::vector<size_t> idx(syms);
    std::iota(idx.begin(), idx.end(), size_t{0});
    const auto cells = codes::despread<size_t>(idx, r.n_cw, 1, spread_group(*r.spec));  // slot by slot, as slot_cells
    std::vector<int> out;
    for (size_t s = 0; s < n; ++s) {
        const auto at = cells.begin() + static_cast<std::ptrdiff_t>(s * (syms / n));
        if (static_cast<int64_t>(*std::max_element(at, at + static_cast<std::ptrdiff_t>(syms / n))) < have)
            out.push_back(static_cast<int>(s));
    }
    return out;
}

std::vector<double> soft_scale(const Received& r, const DataEstimate& est) {
    if constexpr (!config::SYMBOL_NOISE) return {};
    auto full = noise_var(est.h, est);
    for (size_t i = 0; i < full.data.size(); ++i) full.data[i] += est.mse.data[i];
    return symbol_noise_scale(r, est.h, full);
}

std::vector<double> slot_soft(const Received& r, const DataEstimate& est, int slot, std::span<const double> scale) {
    const size_t nc = r.raw.cols;
    std::vector<cd> y, h;
    std::vector<double> var;
    // the slot's cells in its order (within what is in, on a partial burst)
    for (const size_t j : slot_cells(r, slot)) {
        const size_t row = j / nc, c = j % nc;
        const cd hj = est.h.data[j];
        y.push_back(data_row(r.raw, row)[c]);
        h.push_back(hj);
        const double v = est.n0_at(row, c) + est.clip_ratio * std::norm(hj) + est.mse.data[j];
        var.push_back(scale.empty() ? v : v * scale[row]);
    }
    return constellation::llr(y, h, var, points_of(*r.spec));
}

double snr_db(const equalizer::Estimate& est, std::string_view b) {
    return 10 * std::log10(est.p_sig / est.n0_snr * band(b).nc * RS / SNR_REF_BW_HZ);
}

SymbolTimes symbol_times(const Received& r) {
    const size_t S = DATA_SYMS_PER_FRAME, n_f = r.raw.rows / SYMS_PER_FRAME;
    SymbolTimes out{Mat<double>(n_f, S), std::vector<double>(r.hp.rows)};
    for (size_t f = 0; f < n_f; ++f) {
        const size_t air = f + (r.kc && static_cast<int>(f) >= *r.kc ? 1 : 0);
        for (size_t s = 0; s < S; ++s)
            out.rows[f][s] = static_cast<double>((air * SYMS_PER_FRAME + s + 1) * NSYM) / FS;
    }
    for (size_t i = 0; i < out.pilots.size(); ++i)
        out.pilots[i] = static_cast<double>(i * SYMS_PER_FRAME * NSYM) / FS;
    return out;
}

std::vector<cd> equalized(const Received& r) {
    const auto& h = r.est.h;
    const size_t nc = r.raw.cols, n_f = r.raw.rows / SYMS_PER_FRAME;
    std::vector<cd> out;
    out.reserve(h.data.size());
    for (size_t f = 0; f < n_f; ++f)
        for (size_t s = 1; s < SYMS_PER_FRAME; ++s)
            for (size_t k = 0; k < nc; ++k) {
                const cd hk = h[f * DATA_SYMS_PER_FRAME + s - 1][k];
                out.push_back(std::norm(hk) > 0 ? r.raw[f * SYMS_PER_FRAME + s][k] / hk : cd{});
            }
    return out;
}

std::vector<cd> preview_frame(std::span<const double> x, double cfo, const Spec& spec, int n_f, int air_frame) {
    // Fixed timing (no steps): x is exactly one frame and its closing pilot, so a timing step
    // would run past its end. Tracking free with no phase reference stepped 2 samples on FM
    // (FS / 2 pi RS is large) and threw "burst truncated" out of the engine thread.
    static constexpr double no_steps[2] = {0.0, 0.0};
    const auto fr = demod_frames(waveform::freq_correct(waveform::to_baseband(x), cfo), 0, 1, 0, 0.0, no_steps, spec.band);
    const size_t nc = fr.raw.cols;
    std::vector<cd> out;
    out.reserve(DATA_SYMS_PER_FRAME * nc);
    if (sc_spec(spec, n_f)) {
        // Scattered pilots sit inside the data symbols, so they share the data's clip gain (ratio 1).
        const auto& b = waveform::band(spec.band);
        const int nci = b.nc();
        const auto& cells = sc_frame_cells(nci, air_frame);
        Mat<cd> H(SYMS_PER_FRAME, nc);
        if (const auto intra = sc_intra_spec(nci)) {
            const int K = intra->K;
            const int c = ((air_frame % intra->n_combs) + intra->n_combs) % intra->n_combs;
            // Remove the carrier phase ramp (DEMOD_BACKOFF + any channel delay) before interpolating across K carriers.
            const double phi0 = -2 * PI * RS * DEMOD_BACKOFF / FS;
            cd dsum = 0.0;
            for (int pi = 0; pi < intra->n_syms; ++pi) {
                const int ps = intra->syms[static_cast<size_t>(pi)];
                for (int k = c; k < nci; k += K)
                    H[ps][k] = (fr.raw[ps][k] / b.pilot[static_cast<size_t>(k)]) * std::polar(1.0, -phi0 * k);
                for (int k = c; k + K < nci; k += K) dsum += H[ps][k + K] * std::conj(H[ps][k]);
            }
            const double dphi = std::arg(dsum) / K;
            const double phi = phi0 + dphi;
            const int last_p = c + ((nci - 1 - c) / K) * K;
            for (int pi = 0; pi < intra->n_syms; ++pi) {
                const int ps = intra->syms[static_cast<size_t>(pi)];
                for (int k = c; k < nci; k += K) H[ps][k] *= std::polar(1.0, -dphi * k);
                for (int k = 0; k < nci; ++k) {
                    if (k <= c) {
                        H[ps][k] = H[ps][c];
                    } else if (k >= last_p) {
                        H[ps][k] = H[ps][last_p];
                    } else if ((k - c) % K != 0) {
                        const int k0 = c + ((k - c) / K) * K;
                        const double w = static_cast<double>(k - k0) / K;
                        H[ps][k] = (1.0 - w) * H[ps][k0] + w * H[ps][k0 + K];
                    }
                }
                for (int k = 0; k < nci; ++k) H[ps][k] *= std::polar(1.0, phi * k);
            }
            for (int k = 0; k < nci; ++k) {
                H[1][k] = 0.5 * (H[0][k] + H[2][k]);
                H[3][k] = 0.5 * (H[2][k] + H[4][k]);
                H[5][k] = H[4][k];
            }
        } else {
            for (int k = 0; k < nci; ++k) {
                const int sp = cells.own[static_cast<size_t>(k)] / nci;
                const cd hk = fr.raw[sp][k] / b.pilot[static_cast<size_t>(k)];
                for (int s = 0; s < SYMS_PER_FRAME; ++s) H[s][k] = hk;
            }
        }
        for (int d : cells.Dc) {
            const size_t s = static_cast<size_t>(d / nci), k = static_cast<size_t>(d % nci);
            const cd h = H[s][k];
            out.push_back(std::norm(h) > 0 ? fr.raw[s][k] / h : cd{});
        }
        return out;
    }
    const double g = clip_consts(spec).gain_for(n_f);
    for (int s = 1; s < SYMS_PER_FRAME; ++s) {
        const double t = static_cast<double>(s) / SYMS_PER_FRAME;
        for (size_t k = 0; k < nc; ++k) {
            const cd h = g * ((1 - t) * fr.hp[0][k] + t * fr.hp[1][k]);
            out.push_back(std::norm(h) > 0 ? fr.raw[static_cast<size_t>(s)][k] / h : cd{});
        }
    }
    return out;
}

Burst decode_received(const Received& r) {
    const Spec& spec = *r.spec;
    Burst out;
    out.soft = burst_soft(r);
    Mat<float> sf(out.soft.rows, out.soft.cols);
    std::copy(out.soft.data.begin(), out.soft.data.end(), sf.data.begin());
    for (auto& p : codes::decode_many(codes::spec(spec), sf)) {
        out.payloads.push_back(std::move(p.data));
        out.crc_ok.push_back(p.ok);
    }
    out.spec = &spec;
    out.freq_offset = r.cfo;
    out.preamble_start = r.preamble_start;
    out.snr_db = snr_db(r.est, spec.band);
    return out;
}

Burst demodulate(std::span<const double> x, std::span<const std::string_view> bands, const Accept* accept) {
    return decode_received(receive(x, bands, accept));
}

}  // namespace data2g::modem
