#include "cpm/cpm.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include "dsp/fft.hpp"

namespace data2g::cpm {

namespace {

using cdouble = std::complex<double>;
constexpr double FS = config::FS;
constexpr double PI = std::numbers::pi;

// numpy's pairwise summation (add.reduce over a contiguous run), so sums,
// means and shares round as the reference's do.
double np_sum(const double* a, std::size_t n) {
    if (n < 8) {
        double r = 0.0;
        for (std::size_t i = 0; i < n; ++i) r += a[i];
        return r;
    }
    if (n <= 128) {
        double r[8];
        std::copy(a, a + 8, r);
        std::size_t i = 8;
        for (; i < n - n % 8; i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_sum(a, n2) + np_sum(a + n2, n - n2);
}

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

// |Z[:, :nb]| ** 2 (hypot, then squared, as numpy's abs and power).
Mat<double> power(const std::vector<cdouble>& Z, std::size_t rows, std::size_t T, std::size_t nb) {
    Mat<double> E(rows, nb);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t k = 0; k < nb; ++k) {
            const double a = std::hypot(Z[r * T + k].real(), Z[r * T + k].imag());
            E[r][k] = a * a;
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
        block(i == 0 ? g.preamble : g.mid_block);
        if (i < tables::CPM.hdr_copies) header();
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

std::vector<double> tones(const Grid& g, std::span<const int> sym) {
    const int T = g.T;
    std::vector<double> x(sym.size() * T);
    double acc = 0.0;  // np.cumsum: sequential
    for (std::size_t i = 0; i < sym.size(); ++i) {
        const double f = g.f0 + static_cast<double>(sym[i]) * g.rate;
        for (int k = 0; k < T; ++k) {
            acc += f;
            x[i * T + k] = std::numbers::sqrt2 * std::cos(2 * PI * acc / FS);
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

std::vector<double> modulate(const Spec& s, const std::vector<std::vector<std::uint8_t>>& coded, bool dup) {
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
    return tones(g, sym);
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

Detection detect(const Grid& g, std::span<const double> x, double reach_hz, bool fine, bool front_only, int n_sym,
                 double floor) {
    const Layout L = layout(g, n_sym ? n_sym : stream_symbols(g, 0, false));
    std::span<const int> rows = L.sync_rows, tones_ = L.sync_tones;
    if (front_only) {
        rows = rows.first(L.front);
        tones_ = tones_.first(L.front);
    }
    const std::size_t R = rows.size();
    const std::int64_t span = rows.back() + 1, T = g.T;
    const int extra = static_cast<int>(std::ceil(reach_hz / g.rate));
    const std::size_t nb = g.m + 2 * extra, ndk = 2 * extra + 1;
    Detection best{-1.0, 0, 0.0};
    const std::int64_t len = static_cast<std::int64_t>(x.size());
    std::vector<cdouble> zf(x.size());
    for (double frac : {0.0, 0.25, 0.5, 0.75}) {
        // mixed once per CFO fraction as numpy does: (w * n) / FS, the
        // division a complex one (times 1 / FS)
        const double w = (-2 * PI) * (g.f0 - extra * g.rate + frac * g.rate), inv = 1.0 / FS;
        for (std::int64_t i = 0; i < len; ++i) {
            const double arg = (w * static_cast<double>(i)) * inv;
            zf[i] = {x[i] * std::cos(arg), x[i] * std::sin(arg)};
        }
        for (int ph = 0; ph < 4; ++ph) {
            const std::int64_t off = ph * T / 4, n = (len - off) / T;
            if (n < span) continue;
            std::vector<cdouble> seg(zf.begin() + off, zf.begin() + off + n * T);
            const Mat<double> E = shares(power(fft_rows(seg, n, T), n, T, nb));
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
        double top = 0;
        std::int64_t s_best = 0;
        double c_best = 0;
        bool first = true;
        std::vector<double> on(R), all(R * g.m);
        const std::int64_t step = std::max(1L, T / 32), lo = floor_div(-T, 8), hi = T / 8;
        for (std::int64_t dt = lo; dt < hi + 1; dt += step)
            for (int q = -2; q <= 2; ++q) {
                const double df = (0.0625 * q) * g.rate;
                const std::int64_t s = best.start + dt;
                const double c = best.cfo + df;
                const Mat<double> E = energies(g, x, s, static_cast<int>(span), c);
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
        best.start = s_best;
        best.cfo = c_best;
    }
    return best;
}

std::vector<double> llrs(const Grid& g, const Mat<double>& E) {
    const auto top = row_max(E);
    const double mtop = np_mean(top);
    double n0 = median_rest(E) / std::numbers::ln2;  // exponential: median = N0 ln 2
    n0 = std::max(n0, 1e-12 * mtop + 1e-300);
    const double es = std::max(mtop - n0, 1e-3 * n0);
    const double scale = es / (n0 + es) / n0;
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

Header read_header(const Grid& g, std::span<const double> x, std::int64_t s0, double cfo, int copies) {
    const Layout L = layout(g, stream_symbols(g, 0, false));
    std::vector<int> rows;
    for (int c = 0; c < copies; ++c) rows.insert(rows.end(), L.hdr_rows[c].begin(), L.hdr_rows[c].end());
    const Mat<double> E = take_rows(shares(energies(g, x, s0, rows.back() + 1, cfo)), rows);
    Header best{nullptr, 0, false, -INFINITY, -INFINITY};
    std::vector<double> v(rows.size());
    for (const auto& s : tables::CPM_SPECS) {  // valid_words' order
        if (s.grid != g.name) continue;
        for (bool d : {false, true})
            for (int n = 0; n <= tables::CPM.max_data; ++n) {
                const auto t = header_symbols(g, header_value(s.index, n, d));
                for (std::size_t i = 0; i < rows.size(); ++i) v[i] = E[i][t[i % t.size()]];
                const double score = np_mean(v);
                // ties: the later word, as argsort(...)[::-1] over a stable sort
                if (score >= best.score) {
                    best.runner_up = best.score;
                    best = {&s, n, d, score, best.runner_up};
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
        out.slots.push_back(llrs(g, part));
        i += n;
    }
    return out;
}

double peak_ratio(const Grid& g, std::span<const double> x, std::int64_t s0, double cfo) {
    const auto f = g.preamble;
    const Mat<double> E = shares(energies(g, x, s0, static_cast<int>(f.size()), cfo));
    std::vector<double> on(f.size());
    for (std::size_t i = 0; i < f.size(); ++i) on[i] = E[i][f[i]];
    return np_mean(on) / np_mean(row_max(E));
}

std::optional<Lock> find(const Grid& g, std::span<const double> x, std::optional<double> threshold, double reach_hz,
                         bool front_only, std::int64_t lo, std::optional<std::int64_t> hi) {
    const double floor = threshold.value_or(g.sync_threshold);
    const std::int64_t len = static_cast<std::int64_t>(x.size()), T = g.T;
    const Layout L0 = layout(g, stream_symbols(g, 0, false));
    Detection d;
    if (lo || hi) {
        const std::int64_t span = ((front_only ? L0.sync_rows[L0.front - 1] : L0.sync_rows.back()) + 2) * T;
        const std::int64_t a = std::min(len, std::max(0L, lo - T)), b = hi ? std::min(len, *hi + span + T) : len;
        d = detect(g, x.subspan(a, std::max(0L, b - a)), reach_hz, true, front_only, 0, floor);
        d.start += a;
        if (!(lo <= d.start && d.start < hi.value_or(len))) return std::nullopt;
    } else {
        d = detect(g, x, reach_hz, true, front_only, 0, floor);
    }
    if (d.score < floor) return std::nullopt;
    const std::int64_t hdr_end = (L0.hdr_rows[0].back() + 1) * T;
    if (d.start + hdr_end > len) return std::nullopt;  // its header copy is still arriving
    // a tiled front also matches whole periods late: the alignment whose header reads best
    const std::int64_t period = g.costas_len * T, tiles = static_cast<std::int64_t>(g.preamble.size()) * T / period;
    std::optional<std::pair<std::int64_t, Header>> best;
    for (std::int64_t k = 0; k < tiles; ++k) {
        const std::int64_t s = d.start - k * period;
        if (s < 0) continue;
        const Header h = read_header(g, x, s, d.cfo, 1);
        if (!best || h.score > best->second.score) best.emplace(s, h);
    }
    if (!best) return std::nullopt;  // its front is cut off
    const auto [s0, h] = *best;
    if (peak_ratio(g, x, s0, d.cfo) < tables::CPM.peak_ratio) return std::nullopt;
    if (!threshold && h.score < g.header_threshold) return std::nullopt;
    const Layout L = layout(g, stream_symbols(g, h.n_data, h.dup));
    return Lock{h.spec, h.n_data, h.dup, s0, d.cfo, d.score, h.score, h.score - h.runner_up,
                s0 + L.n * T, s0 + hdr_end};
}

Measure measure(const Grid& g, const Mat<double>& E, int n_sym) {
    const auto top = row_max(E);
    const double n0 = std::max(median_rest(E) / std::numbers::ln2, 1e-300);
    Measure out;
    out.snr.resize(top.size());
    for (std::size_t i = 0; i < top.size(); ++i) out.snr[i] = std::max(top[i] / n0 - 1, 1e-6);
    const double mtop = np_mean(top);
    const double es = std::max(mtop - n0, 1e-12);
    double rho = 1.0;
    const std::size_t n = top.size();
    double var = 0;
    for (double t : top) var += (t - mtop) * (t - mtop);
    if (n > 3 && var > 0) {  // Pearson of top[:-1] against top[1:] (np.corrcoef, clipped)
        std::vector<double> a(top.begin(), top.end() - 1), b(top.begin() + 1, top.end());
        const double ma = np_mean(a), mb = np_mean(b);
        double sab = 0, saa = 0, sbb = 0;
        for (std::size_t i = 0; i + 1 < n; ++i) {
            sab += (a[i] - ma) * (b[i] - mb);
            saa += (a[i] - ma) * (a[i] - ma);
            sbb += (b[i] - mb) * (b[i] - mb);
        }
        rho = std::clamp(sab / std::sqrt(saa * sbb), -1.0, 1.0);
    }
    out.spread_est = std::clamp(std::sqrt(std::max(-std::log(std::max(rho, 1e-3)), 0.0)) * g.rate / PI, 0.0, 3.0);
    out.snr_est = 10 * std::log10(es / n0 * g.rate / config::SNR_REF_BW_HZ);
    out.frames = std::max(1.0, static_cast<double>(n_sym * g.T) / config::FRAME_SAMPLES);
    return out;
}

Received receive(std::span<const double> x, const Lock& lock) {
    Soft s = soft(grid_of(*lock.spec), x, lock.start, lock.cfo, lock.n_data, lock.dup);
    const int n = static_cast<int>(s.slots.size());
    return {lock.spec, n, 1 + lock.dup, lock.dup, std::move(s.slots), std::move(s.E), lock.cfo, lock.start,
            lock.header_end};
}

}  // namespace data2g::cpm
