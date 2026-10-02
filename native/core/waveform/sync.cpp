#include "waveform/sync.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <tuple>

#include "dsp/dsp.hpp"
#include "dsp/fft.hpp"

namespace data2g::waveform {
namespace {

using config::FS;
using config::M;
using config::PREAMBLE_CP;

constexpr double TWO_PI = 2.0 * std::numbers::pi;
// FS / STEP_HZ: every grid frequency is a whole number of bins of an FFT
// whose length is a multiple of this.
constexpr std::int64_t PER = 640;
static_assert(PER * STEP_HZ == FS);

std::size_t argmax(const double* v, std::size_t n) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

std::size_t argmax_col(const Mat<double>& S, std::size_t col) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < S.rows; ++i)
        if (S[i][col] > S[best][col]) best = i;
    return best;
}

// numpy's float64 remainder (npy_divmod): the sign of the divisor.
double py_mod(double a, double b) {
    double m = std::fmod(a, b);
    if (m != 0) {
        if ((b < 0) != (m < 0)) m += b;
    } else {
        m = std::copysign(0.0, b);
    }
    return m;
}

// |c|^2. Python's np.abs(c) ** 2 goes through hypot, 40% of this
// module's CPU in libm; c is FFT output, so tolerance-class either way.
double power(cdouble c) { return std::norm(c); }

// Python's max(0, ...) slice start, clamped into [0, n].
std::size_t clamp_to(std::int64_t i, std::size_t n) {
    return static_cast<std::size_t>(std::clamp<std::int64_t>(i, 0, static_cast<std::int64_t>(n)));
}

}  // namespace

std::vector<double> cfo_grid(double reach) {
    const double start = -reach, stop = reach + STEP_HZ / 2;
    const auto n = static_cast<std::size_t>(std::ceil((stop - start) / STEP_HZ));
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = start + static_cast<double>(i) * STEP_HZ;
    return out;
}

std::vector<cdouble> unit_template(const Band& band) {
    std::vector<cdouble> t(band.preamble_template.begin() + PREAMBLE_CP, band.preamble_template.begin() + PREAMBLE_CP + M);
    // np.linalg.norm: re.dot(re) + im.dot(im)
    double re = 0, im = 0;
    for (const auto& v : t) {
        re += v.real() * v.real();
        im += v.imag() * v.imag();
    }
    // numpy divides a complex array by a real as by complex(r, 0): x * (1 / r)
    const double inv = 1.0 / std::sqrt(re + im);
    for (auto& v : t) v = {v.real() * inv, v.imag() * inv};
    return t;
}

std::vector<cdouble> repeat_corr(std::span<const cdouble> z, std::span<const cdouble> t, double f) {
    // the template shifted by f, reversed and conjugated; the phase reduced
    // to one turn before exp (Python does not reduce: |theta| <= 19 rad)
    std::vector<cdouble> g(t.size());
    for (std::size_t k = 0; k < t.size(); ++k) {
        const double c = f * static_cast<double>(k) / FS;
        g[t.size() - 1 - k] = std::conj(t[k] * std::polar(1.0, TWO_PI * (c - std::floor(c))));
    }
    return dsp::fftconvolve_valid(z, g);
}

Mat<cdouble> repeat_corrs(std::span<const cdouble> z, std::span<const cdouble> t, std::span<const double> freqs) {
    const std::size_t m = t.size();
    if (z.size() < m) throw std::invalid_argument("repeat_corrs: signal shorter than the template");
    const std::size_t n = z.size() + m - 1, n_out = z.size() - m + 1;
    const std::size_t L = PER * dsp::next_fast_len((n + PER - 1) / PER), bins_per_step = L / PER;
    std::vector<cdouble> zp(L), gp(L);
    std::copy(z.begin(), z.end(), zp.begin());
    for (std::size_t k = 0; k < m; ++k) gp[k] = std::conj(t[m - 1 - k]);
    const std::vector<cdouble> Z = dsp::fft(zp, true), G0 = dsp::fft(gp, true);
    Mat<cdouble> out(freqs.size(), n_out);
    // independent per frequency: the parallel unit, if one is wanted
    for (std::size_t i = 0; i < freqs.size(); ++i) {
        const auto k = static_cast<std::int64_t>(std::llround(freqs[i] / STEP_HZ));
        const auto shift = static_cast<std::size_t>(((k * static_cast<std::int64_t>(bins_per_step)) % static_cast<std::int64_t>(L)
                                                     + static_cast<std::int64_t>(L)) % static_cast<std::int64_t>(L));
        std::vector<cdouble> prod(L);
        for (std::size_t j = 0; j < shift; ++j) prod[j] = Z[j] * G0[j + L - shift];  // np.roll(G0, shift)
        for (std::size_t j = shift; j < L; ++j) prod[j] = Z[j] * G0[j - shift];
        const std::vector<cdouble> c = dsp::fft(prod, false);
        // exp(-2j pi f (M-1) / FS) = exp(-2j pi k (M-1) / PER): an exact turn
        std::int64_t q = (k * static_cast<std::int64_t>(m - 1)) % PER;
        if (q < 0) q += PER;
        const cdouble ph = std::polar(1.0, -TWO_PI * static_cast<double>(q) / PER);
        for (std::size_t j = 0; j < n_out; ++j) out[i][j] = c[m - 1 + j] * ph;
    }
    return out;
}

RawStat raw_stat(std::span<const cdouble> z, const Band& band, double reach, int repeats,
                 std::optional<std::size_t> levels_from, bool keep_outs) {
    if (repeats <= 0) repeats = band.spec->preamble_repeats;
    const std::vector<cdouble> t = unit_template(band);
    RawStat r;
    r.freqs = cfo_grid(reach);
    const auto n_out_s = static_cast<std::int64_t>(z.size()) - PREAMBLE_CP - static_cast<std::int64_t>(repeats) * M + 1;
    if (n_out_s < 1) throw std::invalid_argument("raw_stat: signal shorter than a preamble");
    const auto n_out = static_cast<std::size_t>(n_out_s);
    std::vector<double> all(r.freqs);
    all.insert(all.end(), NOISE_REF_HZ.begin(), NOISE_REF_HZ.end());
    const Mat<cdouble> cs = repeat_corrs(z, t, all);
    r.S = Mat<double>(r.freqs.size(), n_out);
    r.q.resize(all.size());
    if (keep_outs) r.outs = Mat<cdouble>(r.freqs.size(), cs.cols);
    const double expo_mean = -std::log(1 - NOISE_QUANTILE);  // quantile -> mean of an exponential
    // independent per bin, like repeat_corrs
    for (std::size_t i = 0; i < all.size(); ++i) {
        const cdouble* c = cs[i];
        std::vector<double> p(cs.cols);
        for (std::size_t j = 0; j < p.size(); ++j) p[j] = power(c[j]);
        double q;
        if (!levels_from) {
            q = dsp::quantile(p, NOISE_QUANTILE) / expo_mean;
        } else {
            std::vector<double> pn(p.begin() + static_cast<std::ptrdiff_t>(std::min(*levels_from, p.size())), p.end());
            const auto k = static_cast<std::size_t>(NOISE_QUANTILE * (static_cast<double>(pn.size()) - 1));
            std::nth_element(pn.begin(), pn.begin() + static_cast<std::ptrdiff_t>(k), pn.end());
            q = pn[k] / expo_mean;
        }
        r.q[i] = std::max(q, 1e-12 * (dsp::pairwise_sum(p) / static_cast<double>(p.size())) + 1e-300);  // silence (tests)
        if (i >= r.freqs.size()) continue;
        if (keep_outs) std::copy(c, c + cs.cols, r.outs[i]);
        // each window against the one before it, summed over repeats in
        // Python's order; repeat-major so the loop over starts vectorizes.
        // c[a + M] * conj(c[a]) spelled out: the same products without
        // std::complex's NaN fallback
        std::vector<double> re(n_out), im(n_out);
        for (int rr = 1; rr < repeats; ++rr) {
            const cdouble* x = c + PREAMBLE_CP + static_cast<std::size_t>(rr) * M;
            const cdouble* y = x - M;
            for (std::size_t k = 0; k < n_out; ++k) {
                re[k] += x[k].real() * y[k].real() + x[k].imag() * y[k].imag();
                im[k] += x[k].imag() * y[k].real() - x[k].real() * y[k].imag();
            }
        }
        double* S = r.S[i];
        for (std::size_t k = 0; k < n_out; ++k) S[k] = std::sqrt(re[k] * re[k] + im[k] * im[k]);
    }
    return r;
}

std::pair<Mat<double>, std::vector<double>> detection_stat(std::span<const cdouble> z, const Band& band, double reach,
                                                           int repeats) {
    RawStat r = raw_stat(z, band, reach, repeats);
    const double q = *std::min_element(r.q.begin(), r.q.end());
    for (double& v : r.S.data) v /= q;
    return {std::move(r.S), std::move(r.freqs)};
}

std::size_t first_path(std::span<const double> power, std::size_t peak, int search, double frac, bool cyclic) {
    const auto n = static_cast<std::int64_t>(power.size());
    const double thr = frac * power[peak];
    for (int d = search; d > 0; --d) {
        std::int64_t i = static_cast<std::int64_t>(peak) - d;
        if (cyclic) {
            i = ((i % n) + n) % n;
        } else if (i < 1 || i + 1 >= n) {
            continue;
        }
        const auto u = static_cast<std::size_t>(i);
        const auto lo = static_cast<std::size_t>(((i - 1) % n + n) % n), hi = static_cast<std::size_t>((i + 1) % n);
        if (power[u] >= thr && power[u] >= power[lo] && power[u] >= power[hi]) return u;
    }
    return peak;
}

std::pair<std::int64_t, double> refine(std::span<const cdouble> z, const Band& band, std::int64_t n, double f) {
    const std::vector<cdouble> t = unit_template(band);
    const int R = band.spec->preamble_repeats;
    const std::int64_t n_pre = band.preamble_samples();
    const std::int64_t lo = std::max<std::int64_t>(0, n - config::FIRST_PATH_SEARCH - 8);
    const std::int64_t hi = std::min<std::int64_t>(static_cast<std::int64_t>(z.size()) - n_pre, n + 8);
    if (hi <= lo) return {n, f};
    const std::vector<cdouble> c = repeat_corr(z.subspan(static_cast<std::size_t>(lo), static_cast<std::size_t>(hi + n_pre - lo)), t, f);
    std::vector<double> prof(static_cast<std::size_t>(hi - lo + 1));
    for (std::size_t m = 0; m < prof.size(); ++m) {
        double acc = 0;
        for (int r = 0; r < R; ++r) acc += power(c[m + PREAMBLE_CP + static_cast<std::size_t>(r) * M]);
        prof[m] = acc;
    }
    const std::int64_t start = lo + static_cast<std::int64_t>(first_path(prof, argmax(prof.data(), prof.size())));
    std::vector<cdouble> d(static_cast<std::size_t>(R - 1));
    const auto base = static_cast<std::size_t>(start - lo + PREAMBLE_CP);
    for (std::size_t r = 1; r < static_cast<std::size_t>(R); ++r) d[r - 1] = c[base + r * M] * std::conj(c[base + (r - 1) * M]);
    const cdouble s = dsp::pairwise_sum(d);
    if (std::abs(s) == 0) return {start, f};
    // c_f removes f within a window only, so successive outputs advance by
    // the whole CFO, measured modulo FS / M; the grid picks the alias
    const double period = static_cast<double>(FS) / M;
    const double res = py_mod(std::arg(s) / TWO_PI * period - f + period / 2, period) - period / 2;
    return {start, f + res};
}

std::vector<std::size_t> crossings(std::span<const double> D, double threshold, std::size_t span, std::size_t limit) {
    std::vector<std::size_t> out;
    std::size_t n = 0;
    while (out.size() < limit) {
        std::size_t c = n;
        while (c < D.size() && !(D[c] >= threshold)) ++c;
        if (c >= D.size()) break;
        out.push_back(c + argmax(D.data() + c, std::min(span, D.size() - c)));
        n = c + span;
    }
    return out;
}

Acquisition acquire(std::span<const cdouble> z, const Band& band, std::optional<double> threshold_in, double reach,
                    std::optional<std::pair<std::int64_t, std::int64_t>> search, const Mat<double>* S_in) {
    const double threshold = threshold_in ? *threshold_in : band.preamble_threshold();
    const std::size_t n_pre = static_cast<std::size_t>(band.preamble_samples());
    if (z.size() < n_pre + 2 * M) throw SyncError("signal too short");
    Mat<double> S;
    std::vector<double> freqs;
    if (S_in) {
        S = *S_in;
        freqs = cfo_grid(reach);
    } else {
        std::tie(S, freqs) = detection_stat(z, band, reach);
    }
    if (search) {
        const auto cols = static_cast<std::int64_t>(S.cols);
        const std::int64_t s0 = std::max<std::int64_t>(0, search->first), s1 = std::min(cols, search->second);
        if (s1 - s0 < 1)
            throw SyncError("empty search window (" + std::to_string(search->first) + ", " + std::to_string(search->second) + ")");
        for (std::size_t i = 0; i < S.rows; ++i)
            for (std::int64_t j = 0; j < cols; ++j)
                if (j < s0 || j >= s1) S[i][static_cast<std::size_t>(j)] = -1.0;
    }
    std::vector<double> D(S.cols, -INFINITY);
    for (std::size_t i = 0; i < S.rows; ++i)
        for (std::size_t j = 0; j < S.cols; ++j) D[j] = std::max(D[j], S[i][j]);
    const double best = D.empty() ? -INFINITY : *std::max_element(D.begin(), D.end());
    if (best < threshold) {
        char msg[96];
        std::snprintf(msg, sizeof msg, "no preamble found (peak %.1f < %g)", best, threshold);
        throw SyncError(msg);
    }
    const std::vector<std::size_t> peaks = crossings(D, threshold, n_pre, 1 + TIME_ALTERNATIVES);
    std::size_t n = peaks[0];
    const std::size_t i = argmax_col(S, n);
    Acquisition acq;
    acq.metric = S[i][n];
    std::tie(acq.preamble_start, acq.freq_offset) = refine(z, band, static_cast<std::int64_t>(n), freqs[i]);

    // runner-ups: other CFO bins at this start more than a grid step either
    // side of the winner's, then the later crossings
    std::vector<double> col(S.rows);
    for (std::size_t r = 0; r < S.rows; ++r) col[r] = S[r][n];
    const auto blank = [&](std::size_t j) {
        for (std::size_t r = j < 2 ? 0 : j - 2; r < std::min(col.size(), j + 3); ++r) col[r] = -1;
    };
    blank(i);
    for (int a = 0; a < ALTERNATIVES; ++a) {
        const std::size_t j = argmax(col.data(), col.size());
        if (col[j] < threshold) break;
        acq.alternatives.push_back(refine(z, band, static_cast<std::int64_t>(n), freqs[j]));
        blank(j);
    }
    for (std::size_t p = 1; p < peaks.size(); ++p) {
        n = peaks[p];
        acq.alternatives.push_back(refine(z, band, static_cast<std::int64_t>(n), freqs[argmax_col(S, n)]));
    }
    return acq;
}

StreamDetector::StreamDetector(const Band& b, double r)
    : band(b), reach(r), span(PREAMBLE_CP + static_cast<std::int64_t>(b.spec->preamble_repeats) * M) {
    reset();
}

void StreamDetector::reset() {
    const std::size_t n = cfo_grid(reach).size();
    tail.clear();
    S_.assign(n, {});
    C_.assign(n, {});
    s_off_ = c_off_ = 0;
    s0 = c0 = fed = 0;
    levels.clear();
}

void StreamDetector::feed(std::span<const cdouble> z_new) {
    if (S(0).empty() && tail.empty()) s0 = fed;
    fed += static_cast<std::int64_t>(z_new.size());
    const auto added = static_cast<std::int64_t>(z_new.size());
    std::vector<cdouble> z(std::move(tail));
    z.insert(z.end(), z_new.begin(), z_new.end());
    const auto len = static_cast<std::int64_t>(z.size());
    if (len < span + M) {
        tail = std::move(z);
        return;
    }
    // the level from the new outputs, at least 2000 (0.25 s) of the latest
    const auto levels_from = static_cast<std::size_t>(std::max<std::int64_t>(0, len - M + 1 - std::max<std::int64_t>(added, 2000)));
    RawStat r = raw_stat(z, band, reach, 0, levels_from, true);
    const std::int64_t zs = fed - len;  // stream index of z[0], so of c[0]
    if (C(0).empty()) c0 = zs;
    const std::int64_t have = c0 + static_cast<std::int64_t>(C(0).size());  // outputs overlap the last chunk's
    const std::size_t skip = clamp_to(have - zs, r.outs.cols);
    for (std::size_t i = 0; i < bins(); ++i) {
        C_[i].insert(C_[i].end(), r.outs[i] + skip, r.outs[i] + r.outs.cols);
        S_[i].insert(S_[i].end(), r.S[i], r.S[i] + r.S.cols);
    }
    levels.push_back(std::move(r.q));
    if (levels.size() > static_cast<std::size_t>(CHUNKS)) levels.erase(levels.begin());
    tail.assign(z.begin() + static_cast<std::ptrdiff_t>(r.S.cols), z.end());
}

namespace {
template <typename T>
void drop_front(std::vector<std::vector<T>>& rows, std::size_t& off, std::size_t k) {
    off += k;
    if (off <= rows[0].size() / 2) return;
    for (auto& row : rows) row.erase(row.begin(), row.begin() + static_cast<std::ptrdiff_t>(off));
    off = 0;
}
}  // namespace

void StreamDetector::trim(std::int64_t start) {
    const std::size_t k = clamp_to(start - s0, S(0).size());
    drop_front(S_, s_off_, k);
    s0 += static_cast<std::int64_t>(k);
    const std::size_t kc = clamp_to(start - c0, C(0).size());
    drop_front(C_, c_off_, kc);
    c0 += static_cast<std::int64_t>(kc);
}

std::optional<double> StreamDetector::level() const {
    if (levels.empty()) return std::nullopt;
    double best = INFINITY;
    std::vector<double> v(levels.size());
    for (std::size_t b = 0; b < levels[0].size(); ++b) {
        for (std::size_t k = 0; k < levels.size(); ++k) v[k] = levels[k][b];
        std::sort(v.begin(), v.end());
        const std::size_t h = v.size() / 2;
        best = std::min(best, v.size() % 2 ? v[h] : (v[h - 1] + v[h]) / 2);
    }
    return best;
}

Mat<double> StreamDetector::stat(std::int64_t lo, std::int64_t hi) const {
    Mat<double> out(bins(), static_cast<std::size_t>(std::max<std::int64_t>(0, hi - lo)), -1.0);
    const std::int64_t a = std::max(lo, s0), b = std::min(hi, s0 + static_cast<std::int64_t>(S(0).size()));
    const auto q = level();
    if (b > a && q) {
        for (std::size_t i = 0; i < bins(); ++i) {
            const double* src = S(i).data() + (a - s0);
            double* dst = out[i] + (a - lo);
            for (std::int64_t j = 0; j < b - a; ++j) dst[j] = src[j] / *q;
        }
    }
    return out;
}

}  // namespace data2g::waveform
