#include "cpm/cpm.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

#include "dsp/dsp.hpp"
#include "dsp/fft.hpp"
#include "util/pool.hpp"
#include "waveform/dsp.hpp"

namespace data2g::cpm {

namespace {

using cdouble = std::complex<double>;
constexpr double FS = config::FS;
constexpr double PI = std::numbers::pi;

// numpy's pairwise summation (add.reduce over a contiguous run), so sums,
// means and shares round as the reference's do.
double np_sum(const double* a, std::size_t n) { return dsp::pairwise_sum(std::span<const double>(a, n)); }

double np_sum(const std::vector<double>& v) { return np_sum(v.data(), v.size()); }
double np_mean(const std::vector<double>& v) { return np_sum(v) / static_cast<double>(v.size()); }

// numpy's npy_logaddexp.
double logaddexp(double x, double y) {
    if (x == y) return x + std::numbers::ln2;
    const double d = x - y;
    if (d > 0) return x + std::log1p(std::exp(-d));
    if (d <= 0) return y + std::log1p(std::exp(d));
    return d;
}

// rows x T complex, contiguous -> each row's forward FFT (numpy's fft, axis=1).
std::vector<cdouble> fft_rows(const std::vector<cdouble>& z, std::size_t rows, std::size_t T) {
    std::vector<cdouble> out(z.size());
    if (z.empty()) return out;
    const pocketfft::shape_t shape{rows, T};
    const pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(T * sizeof(cdouble)),
                                     static_cast<std::ptrdiff_t>(sizeof(cdouble))};
    pocketfft::c2c(shape, stride, stride, pocketfft::shape_t{1}, true, z.data(), out.data(), 1.0);
    return out;
}

// `rows` symbols of T real samples from x, each zero-padded to 4T -> |rfft|^2 (numpy's rfft(n=4T),
// re^2 + im^2): rows x (2T + 1), bins R / 4 apart (detect's coarse stage).
Mat<double> padded_power(const double* x, std::size_t rows, std::size_t T) {
    const std::size_t N = 4 * T, nf = 2 * T + 1;
    Mat<double> P(rows, nf);
    if (!rows) return P;
    std::vector<double> in(rows * N, 0.0);
    for (std::size_t r = 0; r < rows; ++r) std::copy(x + r * T, x + (r + 1) * T, in.begin() + static_cast<std::ptrdiff_t>(r * N));
    std::vector<cdouble> out(rows * nf);
    pocketfft::r2c(pocketfft::shape_t{rows, N},
                   pocketfft::stride_t{static_cast<std::ptrdiff_t>(N * sizeof(double)), static_cast<std::ptrdiff_t>(sizeof(double))},
                   pocketfft::stride_t{static_cast<std::ptrdiff_t>(nf * sizeof(cdouble)), static_cast<std::ptrdiff_t>(sizeof(cdouble))},
                   pocketfft::shape_t{1}, true, in.data(), out.data(), 1.0);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t k = 0; k < nf; ++k) {
            const cdouble z = out[r * nf + k];
            P[r][k] = z.real() * z.real() + z.imag() * z.imag();
        }
    return P;
}

// |Z[:, :nb]| ** 2 as re^2 + im^2 (cpm.py's _sq).
Mat<double> power(const std::vector<cdouble>& Z, std::size_t rows, std::size_t T, std::size_t nb) {
    Mat<double> E(rows, nb);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t k = 0; k < nb; ++k) {
            const cdouble z = Z[r * T + k];
            E[r][k] = z.real() * z.real() + z.imag() * z.imag();
        }
    return E;
}

// Python's round() for the half-way cases (banker's), as the layout uses it.
int py_round(double v) { return static_cast<int>(std::nearbyint(v)); }

std::int64_t floor_div(std::int64_t a, std::int64_t b) { return a / b - ((a % b != 0) && ((a < 0) != (b < 0))); }

std::vector<int> gray(int m) {
    std::vector<int> g(m);
    for (int i = 0; i < m; ++i) g[i] = i ^ (i >> 1);
    return g;
}

// The median of each row's energies without its largest, over all rows (np.median of the flattened rest).
double median_rest(const Mat<double>& E) {
    std::vector<double> v;
    v.reserve(E.rows * (E.cols - 1));
    std::vector<double> row(E.cols);
    for (std::size_t r = 0; r < E.rows; ++r) {
        std::copy(E[r], E[r] + E.cols, row.begin());
        std::sort(row.begin(), row.end());
        v.insert(v.end(), row.begin(), row.end() - 1);
    }
    const std::size_t n = v.size(), h = n / 2;
    std::nth_element(v.begin(), v.begin() + h, v.end());
    if (n % 2) return v[h];
    const double hi = v[h], lo = *std::max_element(v.begin(), v.begin() + h);
    return (lo + hi) / 2;
}

std::vector<double> row_max(const Mat<double>& E) {
    std::vector<double> top(E.rows);
    for (std::size_t r = 0; r < E.rows; ++r) top[r] = *std::max_element(E[r], E[r] + E.cols);
    return top;
}

Mat<double> take_rows(const Mat<double>& E, std::span<const int> rows) {
    Mat<double> out(rows.size(), E.cols);
    for (std::size_t i = 0; i < rows.size(); ++i) std::copy(E[rows[i]], E[rows[i]] + E.cols, out[i]);
    return out;
}

}  // namespace

const Grid* grid(std::string_view name) {
    for (const auto& g : tables::CPM_GRIDS)
        if (g.name == name) return &g;
    return nullptr;
}

const Spec* spec(std::string_view name) {
    for (auto table : {tables::CPM_SPECS, tables::CPM_CTL})
        for (const auto& s : table)
            if (s.name == name) return &s;
    return nullptr;
}

const Grid& grid_of(const Spec& s) { return *grid(s.grid); }

const Spec& ctl(const Grid& g) { return tables::CPM_CTL[&g - tables::CPM_GRIDS.data()]; }

int header_value(int index, int n_data, bool dup) {
    const int v = ((index + 2 * dup) << 6) | n_data;
    if (index < 0 || n_data < 0 || n_data >= 64 || v >= 1024) throw std::invalid_argument("cpm header out of range");
    return v;
}

std::span<const std::uint8_t> header_symbols(const Grid& g, int value) {
    return g.header_tones.subspan(static_cast<std::size_t>(value) * g.hdr_len, g.hdr_len);
}

int stream_symbols(const Grid& g, int n_data, bool dup) {
    return (1 + dup) * ctl(g).n_sym + n_data * tables::CPM.data_n / g.bits;
}

Layout layout(const Grid& g, int n_sym) {
    const int H = g.hdr_len, D = std::max(1, py_round(tables::CPM.spacing_s * g.rate));
    Layout L;
    int r = 0, i = 0;
    auto block = [&](std::span<const std::uint8_t> b) {
        for (auto t : b) {
            L.sync_rows.push_back(r++);
            L.sync_tones.push_back(t);
        }
    };
    auto header = [&] {
        L.hdr_rows.emplace_back(H);
        for (int k = 0; k < H; ++k) L.hdr_rows.back()[k] = r++;
    };
    for (int s = 0; s < std::max(n_sym, 1); s += D, ++i) {
        if (i < tables::CPM.hdr_copies) {  // a sync block before each header copy, none after
            block(i == 0 ? g.preamble : g.mid_block);
            header();
        }
        for (int d = std::min(D, n_sym - s); d > 0; --d) L.data_rows.push_back(r++);
    }
    while (static_cast<int>(L.hdr_rows.size()) < tables::CPM.hdr_copies) {
        block(g.mid_block);
        header();
    }
    L.n = r;
    L.front = static_cast<int>(g.preamble.size());
    return L;
}

double burst_seconds(const Spec& s, int n_cw, bool dup) {
    const Grid& g = grid_of(s);
    const int n_data = std::max(0, n_cw - 1 - static_cast<int>(dup));
    return static_cast<double>(layout(g, stream_symbols(g, n_data, dup)).n * g.T) / FS + 2 * tables::CPM.ramp_s;
}

std::vector<int> to_tones(const Grid& g, std::span<const std::uint8_t> bits) {
    if (bits.size() % g.bits) throw std::invalid_argument("bits not a whole number of symbols");
    const auto gr = gray(g.m);
    std::vector<int> out(bits.size() / g.bits);
    for (std::size_t i = 0; i < out.size(); ++i) {
        int idx = 0;
        for (int b = 0; b < g.bits; ++b) idx = (idx << 1) | (bits[i * g.bits + b] & 1);
        out[i] = gr[idx];
    }
    return out;
}

std::vector<double> tones(const Grid& g, std::span<const int> sym, double glide) {
    const int T = g.T;
    std::vector<double> x(sym.size() * T);
    const int ng = static_cast<int>(glide * T);  // samples of glide
    double acc = 0.0;  // np.cumsum: sequential
    if (ng > 1) {
        // the tone index per sample, smoothed: np.convolve(np.pad(a, ng, "edge"),
        // hanning(ng + 2)[1:-1] normalized, "same")[ng:-ng], whose output i is
        // the full convolution's i + (ng - 1) / 2
        std::vector<double> w(ng);
        double sum = 0.0;
        for (int j = 0; j < ng; ++j) sum += w[j] = 0.5 - 0.5 * std::cos(2 * PI * (j + 1) / (ng + 1));
        for (auto& v : w) v /= sum;
        const auto L = static_cast<std::int64_t>(x.size());
        const auto at = [&](std::int64_t k) {  // the edge-padded sequence
            return static_cast<double>(sym[std::clamp<std::int64_t>(k - ng, 0, L - 1) / T]);
        };
        for (std::int64_t i = 0; i < L; ++i) {
            const std::int64_t c = i + ng + (ng - 1) / 2;
            double a = 0.0;
            for (int j = 0; j < ng; ++j) a += at(c - j) * w[j];
            acc += g.f0 + a * g.rate;
            x[i] = std::numbers::sqrt2 * std::cos(2 * PI * acc / FS);
        }
    } else {
        for (std::size_t i = 0; i < sym.size(); ++i) {
            const double f = g.f0 + static_cast<double>(sym[i]) * g.rate;
            for (int k = 0; k < T; ++k) {
                acc += f;
                x[i * T + k] = std::numbers::sqrt2 * std::cos(2 * PI * acc / FS);
            }
        }
    }
    const int n_ramp = static_cast<int>(tables::CPM.ramp_s * FS);
    const std::size_t n = x.size();
    for (int i = 0; i < n_ramp && static_cast<std::size_t>(i) < n; ++i) {
        const double ramp = (1 - std::cos(PI * (i + 0.5) / n_ramp)) / 2;
        x[i] *= ramp;
        x[n - 1 - i] *= ramp;
    }
    return x;
}

const tables::CpmTxFilter& tx_filter(const Grid& g, int cap) {
    if (cap < 0 || cap > 2) throw std::out_of_range("no bandwidth cap " + std::to_string(cap));
    return g.tx[cap];
}

std::vector<double> overshoot(const tables::CpmTxFilter& f) {
    std::vector<double> o(config::CLIP_OVERSHOOT.begin(), config::CLIP_OVERSHOOT.end());
    o.resize(static_cast<std::size_t>(f.passes), config::CLIP_OVERSHOOT.back());
    return o;
}

std::vector<double> bandpass(const Grid& g, std::span<const double> x, int cap) {
    const auto& f = tx_filter(g, cap);
    return waveform::tx_condition(x, g.clip_db, overshoot(f), 0, std::numeric_limits<std::size_t>::max(),
                                  {g.f0 - f.bp, g.f0 + (g.m - 1) * g.rate + f.bp});
}

std::vector<double> modulate(const Spec& s, const std::vector<std::vector<std::uint8_t>>& coded, bool dup, int cap) {
    const Grid& g = grid_of(s);
    const int n_data = static_cast<int>(coded.size()) - 1 - dup;
    std::vector<int> stream;
    for (const auto& c : coded) {
        const auto t = to_tones(g, c);
        stream.insert(stream.end(), t.begin(), t.end());
    }
    const Layout L = layout(g, static_cast<int>(stream.size()));
    std::vector<int> sym(L.n);
    for (std::size_t i = 0; i < L.sync_rows.size(); ++i) sym[L.sync_rows[i]] = L.sync_tones[i];
    const auto h = header_symbols(g, header_value(s.index, n_data, dup));
    for (const auto& rows : L.hdr_rows)
        for (std::size_t i = 0; i < rows.size(); ++i) sym[rows[i]] = h[i];
    for (std::size_t i = 0; i < L.data_rows.size(); ++i) sym[L.data_rows[i]] = stream[i];
    return tones(g, sym, tx_filter(g, cap).glide);
}

Mat<double> energies(const Grid& g, std::span<const double> x, std::int64_t start, int n_sym, double cfo, int extra) {
    const std::size_t T = g.T, N = static_cast<std::size_t>(n_sym) * T;
    const double w = (-2 * PI) * (g.f0 - extra * g.rate + cfo);
    std::vector<cdouble> z(N);
    const std::int64_t len = static_cast<std::int64_t>(x.size());
    for (std::size_t i = 0; i < N; ++i) {
        const std::int64_t s = start + static_cast<std::int64_t>(i);
        if (s < 0 || s >= len) continue;
        const double arg = w * (static_cast<double>(s) / FS);
        z[i] = {x[s] * std::cos(arg), x[s] * std::sin(arg)};
    }
    return power(fft_rows(z, n_sym, T), n_sym, T, g.m + 2 * extra);
}

Mat<double> shares(const Mat<double>& E) {
    Mat<double> out(E.rows, E.cols);
    for (std::size_t r = 0; r < E.rows; ++r) {
        const double tot = np_sum(E[r], E.cols) + 1e-30;
        for (std::size_t k = 0; k < E.cols; ++k) out[r][k] = E[r][k] / tot;
    }
    return out;
}

// cpm.coarse_rows: x's whole symbols -> n x (4 nb): symbol k's tones from `extra` bins below the grid at CFO
// fraction q / 4, as shares, at [k][q nb ..]. One rfft per symbol zero-padded to 4T, bins R / 4 apart: a tone at
// f0 + k R is bin 4 (f0 / R + k) (f0 a whole number of bins), so fraction q / 4 is that bin + q, what mixing by
// it and a T-point FFT would give.
Mat<double> coarse_rows(const Grid& g, std::span<const double> x, int extra) {
    const std::size_t T = static_cast<std::size_t>(g.T), n = x.size() / T, nb = g.m + 2 * extra;
    Mat<double> out(n, 4 * nb);
    if (!n) return out;
    const Mat<double> P = padded_power(x.data(), n, T);
    const std::int64_t bin0 = 4 * (static_cast<std::int64_t>(std::lround(g.f0 / g.rate)) - extra);
    Mat<double> E(n, nb);
    for (int q = 0; q < 4; ++q) {
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t b = 0; b < nb; ++b) E[r][b] = P[r][bin0 + q + 4 * static_cast<std::int64_t>(b)];
        const Mat<double> S = shares(E);
        for (std::size_t r = 0; r < n; ++r) std::copy(S[r], S[r] + nb, out[r] + q * nb);
    }
    return out;
}

CoarseTrack::CoarseTrack(const Grid& g, double reach_hz)
    : extra(static_cast<int>(std::ceil(reach_hz / g.rate))), g_(&g) {}

void CoarseTrack::feed(std::span<const double> buf, std::int64_t off) {
    const std::int64_t T = g_->T, end = off + static_cast<std::int64_t>(buf.size());
    std::erase_if(fine_memo, [&](const FineMemo& m) { return m.s < off + T; });  // (fine reaches T/8 before)
    std::erase_if(header_memo, [&](const HeaderMemo& m) { return m.s < off; });
    const std::size_t w = 4 * (g_->m + 2 * extra);
    for (int ph = 0; ph < 4; ++ph) {
        auto& s = s_[ph];
        auto& E = E_[ph];
        if (!s || *s < off) {
            const std::int64_t first = off + ((ph * T / 4 - off) % T + T) % T;
            if (s) {  // the rows still in buf
                const auto drop = std::min<std::size_t>(E.size() / w, static_cast<std::size_t>((first - *s) / T));
                E.erase(E.begin(), E.begin() + static_cast<std::ptrdiff_t>(drop * w));
            }
            s = first;
        }
        const std::int64_t nxt = *s + static_cast<std::int64_t>(E.size() / w) * T;
        if (end - nxt < T) continue;
        const Mat<double> R = coarse_rows(*g_, buf.subspan(static_cast<std::size_t>(nxt - off)), extra);
        E.insert(E.end(), R.data.begin(), R.data.end());
    }
}

std::vector<std::pair<std::int64_t, Mat<double>>> CoarseTrack::rows(std::int64_t a, std::int64_t b) const {
    const std::int64_t T = g_->T;
    const std::size_t w = 4 * (g_->m + 2 * extra);
    std::vector<std::pair<std::int64_t, Mat<double>>> out;
    for (int ph = 0; ph < 4; ++ph) {
        if (!s_[ph]) continue;
        const std::int64_t s = *s_[ph], have = static_cast<std::int64_t>(E_[ph].size() / w);
        const std::int64_t k0 = std::max<std::int64_t>(0, -floor_div(s - a, T));  // the first row starting at or after a
        const std::int64_t k1 = std::max(k0, std::min(have, floor_div(b - s, T)));  // rows ending by b
        Mat<double> M(static_cast<std::size_t>(k1 - k0), w);
        std::copy(E_[ph].begin() + static_cast<std::ptrdiff_t>(k0 * w), E_[ph].begin() + static_cast<std::ptrdiff_t>(k1 * w),
                  M.data.begin());
        out.emplace_back(s + k0 * T - a, std::move(M));
    }
    return out;
}

Detection detect(const Grid& g, std::span<const double> x, double reach_hz, bool fine, bool front_only, int n_sym,
                 double floor, bool mid, const CoarseTrack* track, std::int64_t x0) {
    const Layout L = layout(g, n_sym ? n_sym : stream_symbols(g, 0, false));
    std::span<const int> rows = L.sync_rows, tones_ = L.sync_tones;
    if (front_only) {
        rows = rows.first(L.front);
        tones_ = tones_.first(L.front);
    }
    std::vector<int> mid_rows_, mid_tones;
    if (mid) {
        mid_tones.assign(g.mid_block.begin(), g.mid_block.end());
        for (std::size_t i = 0; i < mid_tones.size(); ++i) mid_rows_.push_back(static_cast<int>(i));
        rows = mid_rows_;
        tones_ = mid_tones;
    }
    const std::size_t R = rows.size();
    const std::int64_t span = rows.back() + 1, T = g.T;
    const int extra = static_cast<int>(std::ceil(reach_hz / g.rate));
    const std::size_t nb = g.m + 2 * extra, ndk = 2 * extra + 1;
    Detection best{-1.0, 0, 0.0};
    const std::int64_t len = static_cast<std::int64_t>(x.size());
    // per timing phase: its first symbol's sample in x, coarse_rows of its symbols (cpm.py's detect)
    std::vector<std::pair<std::int64_t, Mat<double>>> C;
    if (track) {
        if (track->extra != extra) throw std::invalid_argument("detect: track's reach differs");
        C = track->rows(x0, x0 + len);
    } else {
        for (int ph = 0; ph < 4; ++ph) {
            const std::int64_t off = std::min(len, ph * T / 4);
            C.emplace_back(off, coarse_rows(g, x.subspan(static_cast<std::size_t>(off)), extra));
        }
    }
    for (int q = 0; q < 4; ++q) {
        const double frac = 0.25 * q;
        for (const auto& [off, E4] : C) {
            const std::int64_t n = static_cast<std::int64_t>(E4.rows);
            if (n < span) continue;
            Mat<double> E(static_cast<std::size_t>(n), nb);
            for (std::int64_t r = 0; r < n; ++r) std::copy(E4[r] + q * nb, E4[r] + (q + 1) * nb, E[r]);
            const std::int64_t J = n - span + 1;
            std::vector<double> S(ndk * J, 0.0);  // S[dk, j] = sum over r of E[rows_r + j, tone_r + dk]
            for (std::size_t dk = 0; dk < ndk; ++dk)
                for (std::int64_t j = 0; j < J; ++j) {
                    double acc = 0.0;
                    for (std::size_t r = 0; r < R; ++r) acc += E[rows[r] + j][tones_[r] + dk];
                    S[dk * J + j] = acc;
                }
            const std::size_t k = std::max_element(S.begin(), S.end()) - S.begin();
            if (S[k] / R > best.score) {
                const std::int64_t dk = static_cast<std::int64_t>(k) / J, j = static_cast<std::int64_t>(k) % J;
                best = {S[k] / R, off + j * T, (dk - extra) * g.rate + frac * g.rate};
            }
        }
    }
    if (fine && best.score >= floor) {  // timing to T/32, CFO to R/16
        // the fine search's samples: [best.start - T/8, best.start + span T + T/8]
        const bool whole = best.start + floor_div(-T, 8) >= 0 && best.start + T / 8 + span * T <= len;
        if (track && whole) {
            const std::int64_t s_abs = x0 + best.start;
            for (const auto& m : track->fine_memo)
                if (m.s == s_abs && m.cfo == best.cfo && m.front_only == front_only && m.mid == mid)
                    return {best.score, m.fine.start - x0, m.fine.cfo};
        }
        const Detection coarse = best;
        double top = 0;
        std::int64_t s_best = 0;
        double c_best = 0;
        bool first = true;
        std::vector<double> on(R), all(R * g.m);
        const std::int64_t step = std::max<std::int64_t>(1, T / 32), lo = floor_div(-T, 8), hi = T / 8;
        // energies() mixes by absolute sample, so one mix per CFO covers every timing offset (a slice of it
        // is what energies(x, s, ...) builds, bit for bit): 5 mixes, not 45
        const std::int64_t a0 = best.start + lo, N = span * T, M = (hi - lo) + N;
        std::vector<cdouble> zq(M), seg(N);
        for (int q = -2; q <= 2; ++q) {
            const double df = (0.0625 * q) * g.rate;
            const double c = best.cfo + df;
            const double w = (-2 * PI) * (g.f0 + c);
            for (std::int64_t i = 0; i < M; ++i) {
                const std::int64_t s = a0 + i;
                if (s < 0 || s >= len) {
                    zq[i] = 0.0;
                    continue;
                }
                const double arg = w * (static_cast<double>(s) / FS);
                zq[i] = {x[s] * std::cos(arg), x[s] * std::sin(arg)};
            }
            for (std::int64_t dt = lo; dt < hi + 1; dt += step) {
                const std::int64_t s = best.start + dt;
                std::copy(zq.begin() + (dt - lo), zq.begin() + (dt - lo) + N, seg.begin());
                const Mat<double> E = power(fft_rows(seg, span, T), span, T, g.m);
                for (std::size_t r = 0; r < R; ++r) {
                    on[r] = E[rows[r]][tones_[r]];
                    std::copy(E[rows[r]], E[rows[r]] + g.m, all.begin() + r * g.m);
                }
                const double v = np_sum(on) / np_sum(all);
                // Python's max over (score, start, cfo) tuples
                if (first || v > top || (v == top && (s > s_best || (s == s_best && c > c_best)))) {
                    top = v, s_best = s, c_best = c, first = false;
                }
            }
        }
        best.start = s_best;
        best.cfo = c_best;
        if (track && whole)
            track->fine_memo.push_back({x0 + coarse.start, coarse.cfo, front_only, mid, {best.score, x0 + best.start, best.cfo}});
    }
    return best;
}

namespace {

// cpm.spill: the share of Es each neighbour of a row's strongest tone holds over N0, 0..1 (#244).
double spill(const Mat<double>& E, double n0, double es) {
    std::vector<double> nb(E.rows);
    for (std::size_t r = 0; r < E.rows; ++r) {
        const auto a = static_cast<std::size_t>(std::max_element(E[r], E[r] + E.cols) - E[r]);  // np.argmax: first
        nb[r] = (a > 0 ? E[r][a - 1] : n0) + (a + 1 < E.cols ? E[r][a + 1] : n0) - 2 * n0;
    }
    return std::clamp(np_mean(nb) / 2 / es, 0.0, 1.0);
}

// np.median of v (reordered).
double np_median(std::vector<double>& v) {
    const std::size_t n = v.size(), h = n / 2;
    std::nth_element(v.begin(), v.begin() + h, v.end());
    if (n % 2) return v[h];
    const double hi = v[h], lo = *std::max_element(v.begin(), v.begin() + h);
    return (lo + hi) / 2;
}

// cpm.tone_floors: per-tone noise floors, N0 times the tone's median over the tones' median where it stands
// FLOOR_SIGMAS standard errors over that (a steady carrier), N0 elsewhere; empty where none does.
std::vector<double> tone_floors(const Mat<double>& E, double n0) {
    if (!E.rows) return {};
    std::vector<double> med(E.cols), col(E.rows);
    for (std::size_t k = 0; k < E.cols; ++k) {
        for (std::size_t r = 0; r < E.rows; ++r) col[r] = E[r][k];
        med[k] = np_median(col);
    }
    std::vector<double> sorted = med;
    const double ref = np_median(sorted);
    const double bar = ref * (1 + FLOOR_SIGMAS * 1.44 / std::sqrt(static_cast<double>(E.rows)));
    std::vector<double> nb(E.cols, n0);
    bool any = false;
    for (std::size_t k = 0; k < E.cols; ++k)
        if (med[k] > bar) nb[k] = n0 * (med[k] / ref), any = true;
    return any ? nb : std::vector<double>{};
}

// cpm._tone_gain: tone_metric as E * gain - offset. One N0 (gain, off empty): the square law's
// scale = Es / (N0 (N0 + Es)), the codeword's Es, N0 plus the spill. Per-tone floors (tone_floors):
// gain[k] = Es / (N_k (N_k + Es)), off[k] = ln(1 + Es / N_k).
struct ToneGain {
    double scale = 0.0;
    std::vector<double> gain, off;
    bool flat() const { return gain.empty(); }
    double at(double e, std::size_t k) const { return flat() ? e * scale : e * gain[k] - off[k]; }
};

ToneGain tone_gain(const Mat<double>& E) {
    const auto top = row_max(E);
    const double mtop = np_mean(top);
    double n0 = median_rest(E) / std::numbers::ln2;  // exponential: median = N0 ln 2
    n0 = std::max(n0, 1e-12 * mtop + 1e-300);
    ToneGain out;
    auto nb = tone_floors(E, n0);
    if (!nb.empty()) {
        Mat<double> Ew(E.rows, E.cols);  // whitened, for Es and the spill
        for (std::size_t r = 0; r < E.rows; ++r)
            for (std::size_t k = 0; k < E.cols; ++k) Ew[r][k] = E[r][k] * (n0 / nb[k]);
        const double es = std::max(np_mean(row_max(Ew)) - n0, 1e-3 * n0);
        const double sp = spill(Ew, n0, es) * es;
        out.gain.resize(E.cols);
        out.off.resize(E.cols);
        for (std::size_t k = 0; k < E.cols; ++k) {
            const double n = nb[k] + sp;
            out.gain[k] = es / (n + es) / n;
            out.off[k] = std::log1p(es / n);
        }
        return out;
    }
    const double es = std::max(mtop - n0, 1e-3 * n0);
    n0 += spill(E, n0, es) * es;
    out.scale = es / (n0 + es) / n0;
    return out;
}

// cpm.tone_metric where a tone has a floor of its own (one N0: E * scale, left to the callers).
Mat<double> tone_metric(const Mat<double>& E, const ToneGain& tg) {
    Mat<double> G(E.rows, E.cols);
    for (std::size_t r = 0; r < E.rows; ++r)
        for (std::size_t k = 0; k < E.cols; ++k) G[r][k] = tg.at(E[r][k], k);
    return G;
}

}  // namespace

std::vector<double> symbol_ll(const Grid& g, const Mat<double>& E) {
    const ToneGain tg = tone_gain(E);
    const auto gr = gray(g.m);
    std::vector<double> out(E.rows * g.m);
    for (std::size_t r = 0; r < E.rows; ++r)
        for (int v = 0; v < g.m; ++v) out[r * g.m + v] = tg.at(E[r][gr[v]], gr[v]);
    return out;
}

std::vector<double> llrs(const Grid& g, const Mat<double>& E) {
    const ToneGain tg = tone_gain(E);
    return tg.flat() ? bit_llrs(g, E, tg.scale) : bit_llrs(g, tone_metric(E, tg));
}

// cpm.bit_llrs: (n_sym, m) tone log-likelihoods (metric * scale) -> llrs's coded-bit LLRs (their marginals).
std::vector<double> bit_llrs(const Grid& g, const Mat<double>& metric, double scale) {
    const Mat<double>& E = metric;
    const auto gr = gray(g.m);
    std::vector<int> label(g.m);  // tone -> the bit group it carries
    for (int i = 0; i < g.m; ++i) label[gr[i]] = i;
    std::vector<double> out(E.rows * g.bits);
    for (std::size_t r = 0; r < E.rows; ++r)
        for (int b = 0; b < g.bits; ++b) {
            double l[2] = {-INFINITY, -INFINITY};
            bool any[2] = {false, false};
            for (int t = 0; t < g.m; ++t) {
                const int bit = (label[t] >> (g.bits - 1 - b)) & 1;
                const double m = E[r][t] * scale;
                l[bit] = any[bit] ? logaddexp(l[bit], m) : m;
                any[bit] = true;
            }
            out[r * g.bits + b] = l[0] - l[1];
        }
    return out;
}

namespace {

// cpm.header_rows: per grid, per number of copies (1..HDR_COPIES), each valid
// word's header rows (valid_words' order), and the span they reach (symbols).
struct WordRows {
    std::vector<std::vector<std::vector<int>>> rows;  // [copies - 1][word]
    std::vector<int> span;                           // [copies - 1]
    std::vector<int> mid;                            // [word]: the row its mid block starts at (cpm.mid_rows)
};

const WordRows& word_rows(const Grid& g) {
    static const auto cache = [] {
        std::vector<std::pair<std::string_view, WordRows>> out;
        for (const auto& gg : tables::CPM_GRIDS) {
            WordRows w{std::vector<std::vector<std::vector<int>>>(tables::CPM.hdr_copies),
                       std::vector<int>(tables::CPM.hdr_copies, 0), {}};
            for (const auto& s : tables::CPM_SPECS) {
                if (s.grid != gg.name) continue;
                for (bool d : {false, true})
                    for (int n = 0; n <= tables::CPM.max_data; ++n) {
                        const Layout L = layout(gg, stream_symbols(gg, n, d));
                        w.mid.push_back(L.sync_rows[L.front]);
                        std::vector<int> r;
                        for (int c = 0; c < tables::CPM.hdr_copies; ++c) {
                            r.insert(r.end(), L.hdr_rows[c].begin(), L.hdr_rows[c].end());
                            w.span[c] = std::max(w.span[c], r.back() + 1);
                            w.rows[c].push_back(r);
                        }
                    }
            }
            out.emplace_back(gg.name, std::move(w));
        }
        return out;
    }();
    for (const auto& [name, w] : cache)
        if (name == g.name) return w;
    throw std::out_of_range("no CPM grid " + std::string(g.name));
}

Lock make_lock(const Grid& g, std::int64_t s0, double cfo, double score, const Header& h, int copies) {
    const Layout L = layout(g, stream_symbols(g, h.n_data, h.dup));
    Lock l{h.spec, h.n_data, h.dup, s0, cfo, score, h.score, h.score - h.runner_up,
           s0 + L.n * g.T, s0 + static_cast<std::int64_t>(L.hdr_rows[copies - 1].back() + 1) * g.T};
    l.copies = copies;
    return l;
}

}  // namespace

std::int64_t hdr_span(const Grid& g, int copies) {
    return static_cast<std::int64_t>(word_rows(g).span.at(copies - 1)) * g.T;
}

Mat<double> tone_llr(const Grid& g, const Mat<double>& E) {
    const ToneGain tg = tone_gain(E);
    const double lm = std::log(static_cast<double>(g.m));
    const double keep = std::log1p(-tables::CPM.llr_eps), floor_ = std::log(tables::CPM.llr_eps);
    Mat<double> L(E.rows, E.cols);
    for (std::size_t r = 0; r < E.rows; ++r) {
        double tot = tg.at(E[r][0], 0);  // np.logaddexp.reduce: left to right
        for (std::size_t k = 1; k < E.cols; ++k) tot = logaddexp(tot, tg.at(E[r][k], k));
        for (std::size_t k = 0; k < E.cols; ++k) L[r][k] = logaddexp(keep + (tg.at(E[r][k], k) - tot + lm), floor_);
    }
    return L;
}

namespace {

double front_llr(const Grid& g, const Mat<double>& L) {
    std::vector<double> v(g.preamble.size());
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = L[i][g.preamble[i]];
    return np_sum(v);
}

// cpm._segments: tone_llr over header copies of hdr_len rows each, each copy's sum floored at 0.
double segments(const Grid& g, const Mat<double>& L, const std::vector<int>& rows, std::span<const std::uint8_t> t) {
    double out = 0.0;
    std::vector<double> v(g.hdr_len);
    for (std::size_t i = 0; i < rows.size(); i += g.hdr_len) {
        for (int j = 0; j < g.hdr_len; ++j) v[j] = L[rows[i + j]][t[j]];
        out += std::max(np_sum(v), 0.0);
    }
    return out;
}

}  // namespace

Header read_header(const Grid& g, std::span<const double> x, std::int64_t s0, double cfo, int copies) {
    const WordRows& W = word_rows(g);
    const auto& rows = W.rows.at(copies - 1);
    const Mat<double> L = tone_llr(g, energies(g, x, s0, W.span[copies - 1], cfo));
    const double front = front_llr(g, L);
    Header best{nullptr, 0, false, -INFINITY, -INFINITY, front};
    std::size_t k = 0;
    for (const auto& s : tables::CPM_SPECS) {  // valid_words' order
        if (s.grid != g.name) continue;
        for (bool d : {false, true})
            for (int n = 0; n <= tables::CPM.max_data; ++n) {
                const auto t = header_symbols(g, header_value(s.index, n, d));
                const double score = std::max(front, 0.0) + segments(g, L, rows[k++], t);
                // ties: the later word, as argsort(...)[::-1] over a stable sort
                if (score >= best.score) {
                    best.runner_up = best.score;
                    best = {&s, n, d, score, best.runner_up, front};
                } else if (score > best.runner_up) {
                    best.runner_up = score;
                }
            }
    }
    return best;
}

Soft soft(const Grid& g, std::span<const double> x, std::int64_t s0, double cfo, int n_data, bool dup) {
    const Layout L = layout(g, stream_symbols(g, n_data, dup));
    Soft out;
    out.E = take_rows(energies(g, x, s0, L.n, cfo), L.data_rows);
    std::size_t i = 0;
    for (int k = 0; k < 1 + dup + n_data; ++k) {
        const std::size_t n = k < 1 + dup ? ctl(g).n_sym : tables::CPM.data_n / g.bits;
        Mat<double> part(n, out.E.cols);
        std::copy(out.E[i], out.E[i] + n * out.E.cols, part.data.begin());
        out.slots.push_back(k < 1 + dup ? llrs(g, part) : symbol_ll(g, part));
        i += n;
    }
    return out;
}

namespace {

// cpm._lock_header: (the noncoherent reading if it passes; else, the grid having
// coh_lock gates and the noncoherent score reaching COH_PREGATE, the coherent
// reading if it passes them; else the noncoherent one; passed). copies 1: the
// first gates.
std::pair<Header, bool> lock_header(const Grid& g, std::span<const double> x, std::int64_t s0, double cfo, int copies) {
    const bool one = copies == 1;
    const Header h = read_header(g, x, s0, cfo, copies);
    if (h.score >= (one ? g.lock_llr1 : g.lock_llr2) &&
        h.score - h.runner_up >= (one ? tables::CPM.hdr_margin1 : tables::CPM.hdr_margin2))
        return {h, true};
    if (const auto gates = coh_lock(g); gates && h.score >= COH_PREGATE[one ? 0 : 1]) {
        const Header c = coh_read_header(g, x, s0, cfo, copies);
        if (c.score >= (*gates)[one ? 0 : 1] && c.score - c.runner_up >= (*gates)[one ? 2 : 3])
            return {c, true};
    }
    return {h, false};
}

}  // namespace

double peak_ratio(const Grid& g, std::span<const double> x, std::int64_t s0, double cfo, bool mid) {
    const auto f = mid ? g.mid_block : g.preamble;
    const Mat<double> E = shares(energies(g, x, s0, static_cast<int>(f.size()), cfo));
    std::vector<double> on(f.size());
    for (std::size_t i = 0; i < f.size(); ++i) on[i] = E[i][f[i]];
    return np_mean(on) / np_mean(row_max(E));
}

std::optional<Lock> find(const Grid& g, std::span<const double> x, std::optional<double> threshold, double reach_hz,
                         bool front_only, std::int64_t lo, std::optional<std::int64_t> hi, bool wait,
                         const CoarseTrack* track, std::int64_t x0) {
    const double floor = threshold.value_or(g.sync_threshold);
    const std::int64_t len = static_cast<std::int64_t>(x.size()), T = g.T;
    const Layout L0 = layout(g, stream_symbols(g, 0, false));
    Detection d;
    if (lo || hi) {
        const std::int64_t span = ((front_only ? L0.sync_rows[L0.front - 1] : L0.sync_rows.back()) + 2) * T;
        const std::int64_t a = std::min(len, std::max<std::int64_t>(0, lo - T)), b = hi ? std::min(len, *hi + span + T) : len;
        d = detect(g, x.subspan(a, std::max<std::int64_t>(0, b - a)), reach_hz, true, front_only, 0, floor, false, track,
                   x0 + a);
        d.start += a;
        if (!(lo <= d.start && d.start < hi.value_or(len))) return std::nullopt;
    } else {
        d = detect(g, x, reach_hz, true, front_only, 0, floor, false, track, x0);
    }
    if (d.score < floor) return std::nullopt;
    if (d.start + hdr_span(g, 1) > len) return std::nullopt;  // its header copy is still arriving
    if (d.start < 0) return std::nullopt;  // its front is cut off
    if (threshold) return make_lock(g, d.start, d.cfo, d.score, read_header(g, x, d.start, d.cfo, 1), 1);
    std::optional<std::pair<Header, bool>> got;
    if (track)
        for (const auto& m : track->header_memo)
            if (m.s == x0 + d.start && m.cfo == d.cfo) got = {m.h, m.ok};
    if (!got) {
        got = lock_header(g, x, d.start, d.cfo, 1);
        if (track) track->header_memo.push_back({x0 + d.start, d.cfo, got->first, got->second});
    }
    const auto [h, ok] = *got;
    if (ok) return make_lock(g, d.start, d.cfo, d.score, h, 1);
    if (h.front < g.wait_llr) return std::nullopt;  // noise, or a burst's data or a later sync block, its front missed
    // a front, its first header copy faded: both copies, once the second is in
    Lock cand{};
    cand.wait = true;
    cand.start = d.start;
    cand.cfo = d.cfo;
    cand.score = d.score;
    cand.header_end = cand.end = d.start + hdr_span(g, 2);
    if (cand.header_end <= len) return late_lock(g, x, cand);
    if (wait) return cand;
    return std::nullopt;
}

std::optional<Lock> find_mid(const Grid& g, std::span<const double> x, std::int64_t lo,
                             std::optional<std::int64_t> hi, const CoarseTrack* track, std::int64_t x0) {
    const std::int64_t T = g.T, nb = static_cast<std::int64_t>(g.mid_block.size()),
                       len = static_cast<std::int64_t>(x.size());
    const std::int64_t a = std::min(len, std::max<std::int64_t>(0, lo - T)),
                       b = hi ? std::min(len, *hi + (nb + 1) * T) : len;
    Detection d = detect(g, x.subspan(a, std::max<std::int64_t>(0, b - a)), 150.0, true, false, 0, g.mid_threshold, true,
                         track, x0 + a);
    d.start += a;
    if (d.score < g.mid_threshold || !(lo <= d.start && d.start < hi.value_or(len))) return std::nullopt;
    if (d.start + (nb + g.hdr_len) * T > len) return std::nullopt;  // its second header copy is still arriving
    // each word puts its mid block at its own row: read at the start it implies (cpm.read_mid)
    const WordRows& W = word_rows(g);
    const int copies = tables::CPM.hdr_copies;
    const auto& rows = W.rows[copies - 1];
    std::vector<std::pair<std::int64_t, Mat<double>>> L;
    Header best{nullptr, 0, false, -INFINITY, -INFINITY, 0.0};
    std::int64_t best_s = 0;
    std::vector<double> vm(g.mid_block.size());
    std::size_t k = 0;
    for (const auto& s : tables::CPM_SPECS) {  // valid_words' order
        if (s.grid != g.name) continue;
        for (bool dup : {false, true})
            for (int n = 0; n <= tables::CPM.max_data; ++n, ++k) {
                const std::int64_t b1 = W.mid[k], s0 = d.start - b1 * T;
                if (s0 < 0) continue;
                auto it = std::find_if(L.begin(), L.end(), [&](const auto& e) { return e.first == s0; });
                if (it == L.end()) {
                    L.emplace_back(s0, tone_llr(g, energies(g, x, s0, W.span[copies - 1], d.cfo)));
                    it = L.end() - 1;
                }
                const Mat<double>& Ls = it->second;
                const auto t = header_symbols(g, header_value(s.index, n, dup));
                for (std::size_t i = 0; i < vm.size(); ++i) vm[i] = Ls[b1 + i][g.mid_block[i]];
                const double score = std::max(front_llr(g, Ls), 0.0) + np_sum(vm) + segments(g, Ls, rows[k], t);
                if (score >= best.score) {  // ties: the later word (read_header)
                    best.runner_up = best.score;
                    best = {&s, n, dup, score, best.runner_up, 0.0};
                    best_s = s0;
                } else if (score > best.runner_up) {
                    best.runner_up = score;
                }
            }
    }
    if (!best.spec || best.score < g.mid_llr || best.score - best.runner_up < tables::CPM.hdr_margin2)
        return std::nullopt;
    return make_lock(g, best_s, d.cfo, d.score, best, copies);
}

std::optional<Lock> prime_lock(const Grid& g, std::span<const double> x, std::int64_t lo, std::int64_t hi, double cfo) {
    const WordRows& W = word_rows(g);
    const int copies = tables::CPM.hdr_copies;
    const auto& rows = W.rows[copies - 1];
    // a start a whole symbol later reads the same symbols one row on: per phase and CFO, one tone_llr over
    // the window and the rows after it (its scale from all of them), sliced per start (cpm.prime_lock)
    const std::int64_t step = std::max<std::int64_t>(1, g.T / PRIME_STEP);
    const std::size_t n_rows = static_cast<std::size_t>(W.span[copies - 1]);
    const std::int64_t n_pos = (hi - lo) / g.T + 1;
    std::vector<std::int64_t> phases;
    for (std::int64_t p = 0; p < g.T; p += step) phases.push_back(p);
    constexpr std::size_t NC = std::size(PRIME_CFO);
    std::vector<Mat<double>> LL;
    for (auto p : phases)
        for (double df : PRIME_CFO)
            LL.push_back(tone_llr(g, energies(g, x, lo + p, static_cast<int>(n_rows + n_pos), cfo + df * g.rate)));
    double best = -INFINITY, best_cfo = 0.0;
    std::int64_t best_s = 0;
    std::size_t best_k = 0;
    std::vector<double> vm(g.mid_block.size());
    for (std::int64_t m = 0; m < n_pos; ++m) {
        for (std::size_t pi = 0; pi < phases.size(); ++pi) {
            const std::int64_t s0 = lo + m * g.T + phases[pi];
            if (s0 > hi) continue;
            for (std::size_t ci = 0; ci < NC; ++ci) {
                const Mat<double>& all = LL[pi * NC + ci];
                Mat<double> L(n_rows, all.cols);
                std::copy(all[static_cast<std::size_t>(m)], all[static_cast<std::size_t>(m)] + n_rows * all.cols, L.data.begin());
                const double front = std::max(front_llr(g, L), 0.0);
                std::size_t k = 0;
                for (const auto& s : tables::CPM_SPECS) {  // valid_words' order
                    if (s.grid != g.name) continue;
                    for (bool dup : {false, true})
                        for (int nd = 0; nd <= tables::CPM.max_data; ++nd, ++k) {
                            const std::size_t b1 = static_cast<std::size_t>(W.mid[k]);
                            for (std::size_t i = 0; i < vm.size(); ++i) vm[i] = L[b1 + i][g.mid_block[i]];
                            const auto t = header_symbols(g, header_value(s.index, nd, dup));
                            const double sc = front + std::max(np_sum(vm), 0.0) + segments(g, L, rows[k], t);
                            if (sc > best) best = sc, best_s = s0, best_cfo = cfo + PRIME_CFO[ci] * g.rate, best_k = k;
                        }
                }
            }
        }
    }
    if (best < PRIME_LLR) return std::nullopt;
    std::size_t k = 0;
    for (const auto& s : tables::CPM_SPECS) {
        if (s.grid != g.name) continue;
        for (bool dup : {false, true})
            for (int nd = 0; nd <= tables::CPM.max_data; ++nd, ++k)
                if (k == best_k) return make_lock(g, best_s, best_cfo, best, Header{&s, nd, dup, best, best, 0.0}, copies);
    }
    return std::nullopt;
}

std::int64_t confirm_end(const Grid& g, const Lock& l) {
    const Layout lay = layout(g, stream_symbols(g, l.n_data, l.dup));
    return l.start + static_cast<std::int64_t>(lay.hdr_rows[1].back() + 1) * g.T;
}

double confirm(const Grid& g, std::span<const double> x, const Lock& l) {
    const Layout lay = layout(g, stream_symbols(g, l.n_data, l.dup));
    const Mat<double> L = tone_llr(g, energies(g, x, l.start, lay.hdr_rows[1].back() + 1, l.cfo));
    const auto t = header_symbols(g, header_value(l.spec->index, l.n_data, l.dup));
    const int b1 = lay.sync_rows[lay.front];
    std::vector<double> vm(g.mid_block.size()), vh(lay.hdr_rows[1].size());
    for (std::size_t i = 0; i < vm.size(); ++i) vm[i] = L[b1 + i][g.mid_block[i]];
    for (std::size_t i = 0; i < vh.size(); ++i) vh[i] = L[lay.hdr_rows[1][i]][t[i]];
    return np_sum(vm) + np_sum(vh);
}

std::optional<Lock> late_lock(const Grid& g, std::span<const double> x, const Lock& cand) {
    const auto [h, ok] = lock_header(g, x, cand.start, cand.cfo, tables::CPM.hdr_copies);
    if (!ok) return std::nullopt;
    return make_lock(g, cand.start, cand.cfo, cand.score, h, tables::CPM.hdr_copies);
}

// data2g.cpm.spread: Doppler spread from the winning tone's energy, a
// least-squares fit of log c(k) = a - b k^2 over the lags 1.. whose
// autocovariance holds up; with none, too fast or no fading by the excess
// variance, unless the symbols on the same or adjacent winning tones (`tones`)
// are correlated: slow selective fading, unmeasured (constants: cpm.py's SPREAD_*).
static double spread(const Grid& g, const std::vector<double>& top, double n0, const std::vector<int>& tones) {
    constexpr int LAGS = 8;
    constexpr double NEAR_Z = 2.0, NEAR_S0 = 0.08, NEAR_S1 = 0.16;  // cpm.SPREAD_NEAR_S
    constexpr std::size_t MIN_SYMS = 16;
    constexpr double KEEP = 0.05, MAX_HZ = 12.0, UNMEASURED_HZ = 2.0, EXCESS_MIN = 0.2, SIG = 2.5, EXCESS_FIT = 0.05;
    const std::size_t n = top.size();
    if (n < MIN_SYMS) return UNMEASURED_HZ;
    const double m0 = np_mean(top);
    std::vector<double> t(n), c(LAGS);
    for (std::size_t i = 0; i < n; ++i) t[i] = top[i] - m0;
    for (int k = 1; k <= LAGS; ++k) {
        std::vector<double> p(n - k);
        for (std::size_t i = 0; i + k < n; ++i) p[i] = t[i + k] * t[i];
        c[k - 1] = np_mean(p);
    }
    std::vector<double> sq(n);
    for (std::size_t i = 0; i < n; ++i) sq[i] = t[i] * t[i];
    const double v = np_mean(sq);
    const double s = std::max(m0 - n0, 1e-12);
    const double excess = (v - (2 * n0 * s + n0 * n0)) / (s * s);
    int m = 0;
    while (excess >= EXCESS_FIT && c[0] > SIG / std::sqrt(static_cast<double>(n)) * v && m < LAGS &&
           c[m] > KEEP * c[0])
        ++m;
    if (m < 2) {
        if (excess < EXCESS_MIN) return 0.0;
        // cpm._slow_by_near_tones
        double num = 0.0;
        std::size_t pairs = 0;
        // cpm._near_lags: the pairs NEAR_S0..NEAR_S1 apart in time
        const auto lag0 = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(NEAR_S0 * g.rate - 1e-9)));
        const auto lag1 = static_cast<std::size_t>(std::floor(NEAR_S1 * g.rate + 1e-9));
        for (std::size_t lag = lag0; lag <= lag1 && lag < n; ++lag) {
            std::vector<double> p;
            for (std::size_t i = 0; i + lag < n; ++i)
                if (std::abs(tones[i + lag] - tones[i]) <= 1) p.push_back(t[i + lag] * t[i]);
            num += np_sum(p);
            pairs += p.size();
        }
        const double np_ = static_cast<double>(pairs);
        return pairs > 0 && v > 0 && num / np_ / v * std::sqrt(np_) > NEAR_Z ? UNMEASURED_HZ : MAX_HZ;
    }
    std::vector<double> x(m), y(m);
    for (int i = 0; i < m; ++i) {
        x[i] = static_cast<double>((i + 1) * (i + 1));
        y[i] = std::log(c[i]);
    }
    const double mx = np_mean(x), my = np_mean(y);
    std::vector<double> sxy(m), sxx(m);
    for (int i = 0; i < m; ++i) {
        sxy[i] = (x[i] - mx) * (y[i] - my);
        sxx[i] = (x[i] - mx) * (x[i] - mx);
    }
    const double b = -np_sum(sxy) / np_sum(sxx);
    return std::min(std::sqrt(std::max(b, 0.0)) * g.rate / PI, MAX_HZ);
}

Measure measure(const Grid& g, const Mat<double>& E, int n_sym) {
    const auto top = row_max(E);
    const double n0 = std::max(median_rest(E) / std::numbers::ln2, 1e-300);
    Measure out;
    out.snr.resize(top.size());
    for (std::size_t i = 0; i < top.size(); ++i) out.snr[i] = std::max(top[i] / n0 - 1, 1e-6);
    double es_n0 = std::max(np_mean(top) - n0, 1e-12) / n0;
    // a carrier: for snr_est only, every tone scaled down to the quietest quarter's floor and Es back up
    // by the share kept (cpm.measure); the per-symbol snr and the spread stay as read
    if (!tone_floors(E, n0).empty()) {
        std::vector<double> med(E.cols), col(E.rows);
        for (std::size_t k = 0; k < E.cols; ++k) {
            for (std::size_t r = 0; r < E.rows; ++r) col[r] = E[r][k];
            med[k] = np_median(col);
        }
        std::vector<double> s = med;  // np.quantile(med, 0.25), linear
        std::sort(s.begin(), s.end());
        const double pos = 0.25 * static_cast<double>(s.size() - 1);
        const std::size_t lo = static_cast<std::size_t>(pos);
        const double q = lo + 1 < s.size() ? s[lo] + (pos - static_cast<double>(lo)) * (s[lo + 1] - s[lo]) : s[lo];
        std::vector<double> sc(E.cols);
        for (std::size_t k = 0; k < E.cols; ++k) sc[k] = q / std::max(med[k], q);
        Mat<double> Ew(E.rows, E.cols);
        for (std::size_t r = 0; r < E.rows; ++r)
            for (std::size_t k = 0; k < E.cols; ++k) Ew[r][k] = E[r][k] * sc[k];
        const double n0w = std::max(median_rest(Ew) / std::numbers::ln2, 1e-300);
        es_n0 = std::max((np_mean(row_max(Ew)) - n0w) / np_mean(sc), 1e-12) / n0w;
    }
    std::vector<int> tones(E.rows);  // np.argmax: the first of equals
    for (std::size_t r = 0; r < E.rows; ++r) {
        std::size_t k = 0;
        for (std::size_t j = 1; j < E.cols; ++j)
            if (E[r][j] > E[r][k]) k = j;
        tones[r] = static_cast<int>(k);
    }
    out.spread_est = spread(g, top, n0, tones);
    out.snr_est = 10 * std::log10(es_n0 * g.rate / config::SNR_REF_BW_HZ);
    out.frames = std::max(1.0, static_cast<double>(n_sym * g.T) / config::FRAME_SAMPLES);
    return out;
}

std::pair<std::int64_t, double> refine(const Grid& g, std::span<const double> x, std::int64_t s0, double cfo,
                                       const Spec& spec, int n_data, bool dup) {
    const Layout L = layout(g, stream_symbols(g, n_data, dup));
    const auto h = header_symbols(g, header_value(spec.index, n_data, dup));
    const int n = std::min(L.n, tables::CPM.refine_symbols);
    std::vector<int> known(n, -1);
    for (std::size_t i = 0; i < L.sync_rows.size(); ++i)
        if (L.sync_rows[i] < n) known[L.sync_rows[i]] = L.sync_tones[i];
    for (const auto& rows : L.hdr_rows)
        for (std::size_t i = 0; i < rows.size(); ++i)
            if (rows[i] < n) known[rows[i]] = h[i];
    std::vector<int> kr, ur;
    for (int r = 0; r < n; ++r) (known[r] >= 0 ? kr : ur).push_back(r);
    // the metric's weight fixed at the lock's placement: tone_metric(E0).sum() / E0.sum(), as Python reads it
    const Mat<double> E0 = energies(g, x, s0, n, cfo);
    const double tot = np_sum(E0.data.data(), E0.data.size());
    double w = 0.0;
    ToneGain tg;  // per-tone floors: the tones' own gains and offsets in place of w
    if (tot > 0) {
        tg = tone_gain(E0);
        if (tg.flat()) {
            std::vector<double> G0(E0.data.size());
            for (std::size_t i = 0; i < G0.size(); ++i) G0[i] = E0.data[i] * tg.scale;
            w = np_sum(G0) / tot;
        }
    }
    const auto sc = [&](double e, int k) { return tg.flat() ? e * w : e * tg.gain[k] - tg.off[k]; };
    const std::int64_t len = static_cast<std::int64_t>(x.size()), T = g.T, N = static_cast<std::int64_t>(n) * T;
    // the likelihood of mixed samples z (n symbols); the log-sum over tones max-shifted (numpy's
    // logaddexp.reduce rounds otherwise, within 1e-15: no tie it could break was seen)
    const auto ll = [&](const cdouble* z) {
        const std::vector<cdouble> seg(z, z + N);
        const Mat<double> E = power(fft_rows(seg, n, T), n, T, g.m);
        std::vector<double> vk(kr.size()), vu(ur.size());
        for (std::size_t i = 0; i < kr.size(); ++i) vk[i] = sc(E[kr[i]][known[kr[i]]], known[kr[i]]);
        for (std::size_t i = 0; i < ur.size(); ++i) {
            const double* e = E[ur[i]];
            double mx = *std::max_element(e, e + g.m) * w;
            if (!tg.flat()) {
                mx = sc(e[0], 0);
                for (int k = 1; k < g.m; ++k) mx = std::max(mx, sc(e[k], k));
            }
            double acc = 0.0;
            for (int k = 0; k < g.m; ++k) acc += std::exp(sc(e[k], k) - mx);
            vu[i] = mx + std::log(acc);
        }
        return np_sum(vk) + np_sum(vu);
    };
    std::int64_t bs = s0;
    double bc = cfo, bv = 0.0;
    const std::pair<std::int64_t, double> steps[] = {{std::max(1, g.T / 16), g.rate / 16},
                                                     {std::max(1, g.T / 64), g.rate / 64}};
    bool centre_known = false;
    for (const auto& [st, sf] : steps) {
        const std::int64_t s = bs, a0 = s - 2 * st, M = 4 * st + N;
        const double f = bc;
        // energies() mixes by absolute sample, so one mix per CFO serves every start. The stage mixes once
        // at its centre CFO; each other CFO is that times a rotation, e^(-2 pi i sf t / FS) to the power j,
        // whose phase repeats every FS / sf samples: one sincos per table entry, not per sample per CFO.
        std::vector<cdouble> z[5];
        z[2].assign(M, 0.0);
        const double w0 = (-2 * PI) * (g.f0 + f);
        for (std::int64_t q = 0; q < M; ++q) {
            const std::int64_t t = a0 + q;
            if (t < 0 || t >= len) continue;
            const double arg = w0 * (static_cast<double>(t) / FS);
            z[2][q] = {x[t] * std::cos(arg), x[t] * std::sin(arg)};
        }
        const double P = FS / sf;
        const std::int64_t Lt = (P == std::floor(P) && P <= static_cast<double>(M)) ? static_cast<std::int64_t>(P) : M;
        std::vector<cdouble> tab(Lt);
        for (std::int64_t q = 0; q < Lt; ++q) {
            const double arg = (-2 * PI) * (sf * static_cast<double>(q) / FS);
            tab[q] = {std::cos(arg), std::sin(arg)};
        }
        const double cyc = (-2 * PI) * std::fmod(sf * static_cast<double>(a0) / FS, 1.0);
        const cdouble c0{std::cos(cyc), std::sin(cyc)};
        for (int j : {-2, -1, 1, 2}) z[j + 2].resize(M);
        for (std::int64_t q = 0; q < M; ++q) {
            const cdouble r1 = c0 * tab[q % Lt], r2 = r1 * r1, b = z[2][q];
            z[0][q] = b * std::conj(r2);
            z[1][q] = b * std::conj(r1);
            z[3][q] = b * r1;
            z[4][q] = b * r2;
        }
        // the 25 placements are independent: one pool task each (bit-identical for any pool size); the
        // second stage's centre is the first stage's best, already scored
        double v[5][5];
        pool::parallel_for(25, [&](std::size_t k) {
            const int i = static_cast<int>(k / 5), j = static_cast<int>(k % 5);
            if (centre_known && i == 2 && j == 2) return;
            v[i][j] = ll(z[j].data() + i * st);
        });
        if (centre_known) v[2][2] = bv;
        double top = -INFINITY;
        for (int i = -2; i <= 2; ++i)
            for (int j = -2; j <= 2; ++j)
                if (v[i + 2][j + 2] > top)  // Python's max: the first best
                    top = v[i + 2][j + 2], bs = s + i * st, bc = f + j * sf;
        bv = top;
        centre_known = true;
    }
    return {bs, bc};
}

Received receive(std::span<const double> x, const Lock& lock) {
    const Grid& g = grid_of(*lock.spec);
    return receive_refined(x, lock, refine(g, x, lock.start, lock.cfo, *lock.spec, lock.n_data, lock.dup));
}

Received receive_refined(std::span<const double> x, const Lock& lock, std::pair<std::int64_t, double> at) {
    const Grid& g = grid_of(*lock.spec);
    const auto [s0, cfo] = at;
    Soft s = soft(g, x, s0, cfo, lock.n_data, lock.dup);
    const int n = static_cast<int>(s.slots.size());
    Received r{lock.spec, n, 1 + lock.dup, lock.dup, std::move(s.slots), std::move(s.E), lock.cfo, lock.start,
               lock.header_end, bins(g, x, s0, layout(g, stream_symbols(g, lock.n_data, lock.dup)).n, cfo), {}};
    if (coh_ctl())  // the control slots on the coherent metric (sync and header rows known, the rest square-law decided)
        for (int k = 0; k < r.n_ctl_slots; ++k) r.soft[k] = dd_soft(r, dd_metric(r, {}, k), k);
    return r;
}

Stream::Stream(std::vector<double> x, const Lock& lock, PassFn pass)
    : x_(std::move(x)), lock_(lock), pass_(std::move(pass)) {
    const Grid& g = grid_of(*lock.spec);
    const Layout L = layout(g, stream_symbols(g, lock.n_data, lock.dup));
    const std::int64_t N = static_cast<std::int64_t>(std::min(L.n, tables::CPM.refine_symbols)) * g.T;
    // refine reads up to two steps of each stage past its window (its two grids)
    refine_need_ = lock.start + N + 2 * std::max(1, g.T / 16) + 2 * std::max(1, g.T / 64) + 1;
    advance();
}

void Stream::feed(std::span<const double> more) {
    x_.insert(x_.end(), more.begin(), more.end());
    advance();
}

void Stream::advance() {
    const Grid& g = grid_of(*lock_.spec);
    const auto len = static_cast<std::int64_t>(x_.size());
    if (!at_ && len >= refine_need_) at_ = refine(g, x_, lock_.start, lock_.cfo, *lock_.spec, lock_.n_data, lock_.dup);
    if (!at_ || !pass_) return;
    const int n_ctl = 1 + lock_.dup;
    const auto rows = slot_rows(g, lock_.n_data, lock_.dup);
    const int n = layout(g, stream_symbols(g, lock_.n_data, lock_.dup)).n;
    std::optional<Received> r;
    for (int k = n_ctl; k < n_ctl + lock_.n_data; ++k) {
        if (passes_.count(k)) continue;
        const std::int64_t b = std::min(n, rows[k].back() + 1 + COH_W);
        if (len < at_->first + b * g.T) break;  // its rows and its metric's window are not all in yet
        if (!r) r = receive_refined(x_, lock_, *at_);  // rows past the audio read as zeros; this slot's don't
        Pass p = pass_(*r, k, earlier_);
        if (!p.post.empty()) earlier_[k] = p.post;
        passes_[k] = std::move(p);
    }
}

Received Stream::finish() {
    const Grid& g = grid_of(*lock_.spec);
    if (!at_) at_ = refine(g, x_, lock_.start, lock_.cfo, *lock_.spec, lock_.n_data, lock_.dup);
    Received r = receive_refined(x_, lock_, *at_);
    r.passes = std::move(passes_);
    return r;
}

// --- coherent detection ---

namespace {

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return !v || std::strcmp(v, "1") == 0;
}

double sq_abs(cdouble z) {  // cpm.py's _sq: re^2 + im^2
    return z.real() * z.real() + z.imag() * z.imag();
}

Mat<double> power_of(const Mat<cdouble>& Z) {
    Mat<double> E(Z.rows, Z.cols);
    for (std::size_t i = 0; i < Z.data.size(); ++i) E.data[i] = sq_abs(Z.data[i]);
    return E;
}

Mat<cdouble> take_crows(const Mat<cdouble>& Z, std::span<const int> rows) {
    Mat<cdouble> out(rows.size(), Z.cols);
    for (std::size_t i = 0; i < rows.size(); ++i) std::copy(Z[rows[i]], Z[rows[i]] + Z.cols, out[i]);
    return out;
}

}  // namespace

// cpm._coh_run_llr: per-row tone_llr of known tones t over a run of consecutive rows, on coh_metric.
std::vector<double> coh_run_llr(const Grid& g, const Mat<cdouble>& Z, std::span<const int> t, double n0) {
    Mat<double> P(Z.rows, Z.cols, 0.0);
    for (std::size_t i = 0; i < t.size(); ++i) P[i][t[i]] = 1.0;
    const Mat<double> L = coh_metric(Z, P, COH_HDR_W, n0);
    const double lm = std::log(static_cast<double>(g.m));
    const double keep = std::log1p(-tables::CPM.llr_eps), floor_ = std::log(tables::CPM.llr_eps);
    std::vector<double> out(t.size());
    for (std::size_t r = 0; r < L.rows; ++r) {
        double tot = L[r][0];  // np.logaddexp.reduce: left to right
        for (std::size_t k = 1; k < L.cols; ++k) tot = logaddexp(tot, L[r][k]);
        out[r] = logaddexp(keep + (L[r][t[r]] - tot + lm), floor_);
    }
    return out;
}

std::optional<std::array<double, 4>> coh_lock(const Grid& g) {
    static const bool on = env_on("DATA2G_CPM_COH_LOCK");
    if (!on) return std::nullopt;
    const double m1 = tables::CPM.hdr_margin1, m2 = tables::CPM.hdr_margin2;
    if (g.name == "c16r25") return std::array{34.0, 38.0, m1, m2};
    // c8r50's one-copy margin: early locks read another word where two of its bursts overlap (cpm.COH_LOCK)
    if (g.name == "c8r50") return std::array{36.0, 40.0, 14.0, m2};
    if (g.name == "c32r62") return std::array{36.0, 42.0, m1, m2};
    return std::nullopt;
}

bool coh_ctl() {
    static const bool on = env_on("DATA2G_CPM_COH_CTL");
    return on;
}

Mat<cdouble> bins(const Grid& g, std::span<const double> x, std::int64_t start, int n_sym, double cfo) {
    const std::size_t T = g.T, N = static_cast<std::size_t>(n_sym) * T;
    const double w = (-2 * PI) * (g.f0 + cfo);
    std::vector<cdouble> z(N);
    const std::int64_t len = static_cast<std::int64_t>(x.size());
    for (std::size_t i = 0; i < N; ++i) {
        const std::int64_t s = start + static_cast<std::int64_t>(i);
        if (s < 0 || s >= len) continue;
        const double arg = (w * static_cast<double>(s)) / FS;  // cpm.bins: (-2 pi i (f0 + cfo) idx) / FS
        z[i] = {x[s] * std::cos(arg), x[s] * std::sin(arg)};
    }
    const auto F = fft_rows(z, n_sym, T);
    Mat<cdouble> out(n_sym, g.m);
    for (int r = 0; r < n_sym; ++r) std::copy_n(&F[r * T], g.m, out[r]);
    return out;
}

Mat<double> coh_metric(const Mat<cdouble>& Z0, const Mat<double>& P, int W, double n0) {
    const std::size_t n = Z0.rows, m = Z0.cols;
    Mat<double> out(n, m);
    if (!n) return out;
    std::vector<int> t(n);
    std::vector<double> c(n);
    std::vector<cdouble> a(n);
    for (std::size_t r = 0; r < n; ++r) {
        t[r] = static_cast<int>(std::max_element(P[r], P[r] + m) - P[r]);
        c[r] = P[r][t[r]];
        a[r] = Z0[r][t[r]];
    }
    // consecutive pairs ~ exp(i (nu - 2 pi d delta)): the timing ramp's delta on a grid, then nu
    std::vector<cdouble> p(n - 1);
    std::vector<int> d(n - 1);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        p[i] = a[i + 1] * std::conj(a[i]) * c[i + 1] * c[i];
        d[i] = t[i + 1] - t[i];
    }
    // the pairs summed per tone step d first (cpm.py: np.unique, np.add.at): the grid's phase depends on d alone
    std::vector<cdouble> qd(2 * m - 1);
    std::vector<char> seen(2 * m - 1, 0);
    for (std::size_t i = 0; i < p.size(); ++i) {
        const auto j = static_cast<std::size_t>(d[i] + static_cast<int>(m) - 1);
        qd[j] += p[i];
        seen[j] = 1;
    }
    std::vector<int> dv;
    std::vector<cdouble> q;
    for (std::size_t j = 0; j < qd.size(); ++j)
        if (seen[j]) dv.push_back(static_cast<int>(j) - static_cast<int>(m) + 1), q.push_back(qd[j]);
    double delta = 0.0, best = -1.0;
    for (int k = 0; k < 401; ++k) {
        const double gk = k == 400 ? 0.5 : k * (1.0 / 400) + -0.5;  // np.linspace(-0.5, 0.5, 401)
        cdouble s = 0.0;
        for (std::size_t i = 0; i < q.size(); ++i) {
            const double th = (2 * PI) * (gk * dv[i]);
            s += cdouble(std::cos(th), std::sin(th)) * q[i];
        }
        const double v = std::abs(s);
        if (v > best) best = v, delta = gk;
    }
    cdouble s = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const double th = (2 * PI) * delta * d[i];
        s += p[i] * cdouble(std::cos(th), std::sin(th));
    }
    const double nu = std::arg(s);
    Mat<cdouble> Z(n, m);
    for (std::size_t r = 0; r < n; ++r) {
        const double tr = -nu * static_cast<double>(r);
        const cdouble rot(std::cos(tr), std::sin(tr));
        for (std::size_t k = 0; k < m; ++k) {
            const double tk = (2 * PI) * delta * static_cast<double>(k);
            Z[r][k] = Z0[r][k] * cdouble(std::cos(tk), std::sin(tk)) * rot;
        }
    }
    for (std::size_t r = 0; r < n; ++r) {
        cdouble v = 0.0;
        for (std::size_t k = 0; k < m; ++k) v += P[r][k] * Z[r][k];
        a[r] = v;
    }
    // h_i: the mean over rows j != i within +-W (cpm.coh_metric's convolve "same")
    std::vector<cdouble> h(n);
    for (std::size_t i = 0; i < n; ++i) {
        cdouble v = 0.0;
        int cnt = 0;
        const std::size_t lo = i >= static_cast<std::size_t>(W) ? i - W : 0, hi = std::min(n, i + W + 1);
        for (std::size_t j = lo; j < hi; ++j)
            if (j != i) v += a[j], ++cnt;
        h[i] = v / static_cast<double>(std::max(cnt, 1));
    }
    std::vector<double> res(n), row(m);
    {
        Mat<double> E(n, m);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t k = 0; k < m; ++k) E[r][k] = sq_abs(Z[r][k]);
        const auto nb = tone_floors(E, n0);
        if (!nb.empty()) {  // per-tone floors: Z_k = h_i + CN(0, N_k + s2) on the sent tone, CN(0, N_k) elsewhere
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t k = 0; k < m; ++k) row[k] = P[r][k] * (sq_abs(Z[r][k] - h[r]) - nb[k]);
                res[r] = np_sum(row);
            }
            const double s2 = std::max(np_mean(res), n0 * 1e-3);
            for (std::size_t r = 0; r < n; ++r)
                for (std::size_t k = 0; k < m; ++k)
                    out[r][k] = E[r][k] / nb[k] - sq_abs(Z[r][k] - h[r]) / (nb[k] + s2) - std::log1p(s2 / nb[k]);
            return out;
        }
    }
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t k = 0; k < m; ++k) row[k] = P[r][k] * sq_abs(Z[r][k] - h[r]);
        res[r] = np_sum(row);
    }
    const double var = std::max(np_mean(res), n0 * (1 + 1e-3));
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t k = 0; k < m; ++k) out[r][k] = sq_abs(Z[r][k]) / n0 - sq_abs(Z[r][k] - h[r]) / var;
    return out;
}

double noise_n0(const Mat<double>& E) { return std::max(median_rest(E) / std::numbers::ln2, 1e-30); }

Header coh_read_header(const Grid& g, std::span<const double> x, std::int64_t s0, double cfo, int copies) {
    const WordRows& WR = word_rows(g);
    const auto& rows = WR.rows.at(copies - 1);
    const Mat<cdouble> Z = bins(g, x, s0, WR.span[copies - 1], cfo);
    const double n0 = noise_n0(power_of(Z));
    const int F = static_cast<int>(g.preamble.size()), H = g.hdr_len, nb = static_cast<int>(g.mid_block.size());
    Header best{nullptr, 0, false, -INFINITY, -INFINITY, 0.0};
    double front = 0.0;
    std::size_t k = 0;
    for (const auto& s : tables::CPM_SPECS) {  // valid_words' order
        if (s.grid != g.name) continue;
        for (bool d : {false, true})
            for (int n = 0; n <= tables::CPM.max_data; ++n) {
                const auto t = header_symbols(g, header_value(s.index, n, d));
                const auto& r = rows[k++];
                std::vector<int> ra, ta;
                for (int i = 0; i < F; ++i) ra.push_back(i), ta.push_back(g.preamble[i]);
                for (int i = 0; i < H; ++i) ra.push_back(r[i]), ta.push_back(t[i]);
                const auto la = coh_run_llr(g, take_crows(Z, ra), ta, n0);
                front = std::max(front, np_sum(la.data(), F));
                double score = std::max(np_sum(la), 0.0);
                if (copies > 1) {
                    std::vector<int> rb, tb;
                    for (int i = 0; i < nb; ++i) rb.push_back(r[H] - nb + i), tb.push_back(g.mid_block[i]);
                    for (int i = 0; i < H; ++i) rb.push_back(r[H + i]), tb.push_back(t[i]);
                    score += std::max(np_sum(coh_run_llr(g, take_crows(Z, rb), tb, n0)), 0.0);
                }
                // ties: the later word, as read_header
                if (score >= best.score) {
                    best.runner_up = best.score;
                    best = {&s, n, d, score, best.runner_up, 0.0};
                } else if (score > best.runner_up) {
                    best.runner_up = score;
                }
            }
    }
    best.front = front;
    return best;
}

std::vector<std::vector<int>> slot_rows(const Grid& g, int n_data, bool dup) {
    const Layout L = layout(g, stream_symbols(g, n_data, dup));
    std::vector<std::vector<int>> out;
    std::size_t i = 0;
    for (int k = 0; k < 1 + dup + n_data; ++k) {
        const std::size_t n = k < 1 + dup ? ctl(g).n_sym : tables::CPM.data_n / g.bits;
        out.emplace_back(L.data_rows.begin() + i, L.data_rows.begin() + i + n);
        i += n;
    }
    return out;
}

Mat<double> dd_metric(const Received& r, const Posts& post, int slot) {
    const Grid& g = grid_of(*r.spec);
    const int n_ctl = 1 + r.dup, n_data = static_cast<int>(r.soft.size()) - n_ctl;
    const Layout L = layout(g, stream_symbols(g, n_data, r.dup));
    const auto rows = slot_rows(g, n_data, r.dup);
    const auto& sr = rows.at(slot);
    const int a = std::max(0, sr.front() - COH_W), b = std::min(L.n, sr.back() + 1 + COH_W);
    Mat<cdouble> Z(b - a, g.m);
    std::copy(r.Z[a], r.Z[b - 1] + g.m, Z.data.begin());
    const Mat<double> E = power_of(Z);
    const ToneGain tg = tone_gain(E);
    Mat<double> P(E.rows, E.cols);
    for (std::size_t i = 0; i < E.rows; ++i) {  // the square law's probabilities
        double mx = -INFINITY;
        for (int k = 0; k < g.m; ++k) mx = std::max(mx, tg.at(E[i][k], k));
        for (int k = 0; k < g.m; ++k) P[i][k] = std::exp(tg.at(E[i][k], k) - mx);
        const double s = np_sum(P[i], g.m);
        for (int k = 0; k < g.m; ++k) P[i][k] /= s;
    }
    const auto h = header_symbols(g, header_value(r.spec->index, n_data, r.dup));
    const auto known = [&](int row, int tone) {
        if (row < a || row >= b) return;
        std::fill(P[row - a], P[row - a] + g.m, 0.0);
        P[row - a][tone] = 1.0;
    };
    for (std::size_t i = 0; i < L.sync_rows.size(); ++i) known(L.sync_rows[i], L.sync_tones[i]);
    for (const auto& hr : L.hdr_rows)
        for (std::size_t i = 0; i < hr.size(); ++i) known(hr[i], h[i]);
    const auto gr = gray(g.m);
    for (const auto& [s, v] : post) {
        const auto& rr = rows.at(s);
        for (std::size_t i = 0; i < rr.size(); ++i) {
            if (rr[i] < a || rr[i] >= b) continue;
            const double* lp = &v[i * g.m];
            const double mx = *std::max_element(lp, lp + g.m);
            std::vector<double> q(g.m);
            for (int k = 0; k < g.m; ++k) q[k] = std::exp(lp[k] - mx);
            const double tot = np_sum(q);
            for (int k = 0; k < g.m; ++k) P[rr[i] - a][gr[k]] = q[k] / tot;  // value k on tone gray(k)
        }
    }
    const Mat<double> M = coh_metric(Z, P, COH_W, noise_n0(E));
    Mat<double> out(sr.size(), g.m);
    for (std::size_t i = 0; i < sr.size(); ++i) std::copy(M[sr[i] - a], M[sr[i] - a] + g.m, out[i]);
    return out;
}

std::vector<double> dd_soft(const Received& r, const Mat<double>& metric, int slot) {
    const Grid& g = grid_of(*r.spec);
    if (slot < 1 + r.dup) {
        auto v = bit_llrs(g, metric);
        for (auto& x : v) x = std::clamp(x, -30.0, 30.0);
        return v;
    }
    const auto gr = gray(g.m);
    std::vector<double> out(metric.rows * g.m);
    for (std::size_t i = 0; i < metric.rows; ++i) {
        const double mx = *std::max_element(metric[i], metric[i] + g.m);
        for (int v = 0; v < g.m; ++v) out[i * g.m + v] = std::clamp(metric[i][gr[v]] - mx, -60.0, 0.0);
    }
    return out;
}

}  // namespace data2g::cpm
