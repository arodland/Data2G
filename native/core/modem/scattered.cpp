// data2g/modem.py's scattered pilots (config.SC_BANDS, the DATA2G_PILOTS study flag): the layout, the joint 2-D LMMSE
// estimate and its decision-directed pass, the transmitter's pilot calibration, find_copy's copy-row gates. modem.cpp
// has the rest (transmit, demod_frames, receive, find_copy, pilot_coherence), beside the pilot-row code.
#include "modem/scattered.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <numbers>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "codes/codes.hpp"
#include "config/config.hpp"
#include "constellation/constellation.hpp"
#include "tables/gamma_q99.hpp"
#include "util/linalg.hpp"
#include "util/pool.hpp"
#include "util/simd.hpp"
#include "waveform/dsp.hpp"

namespace data2g::modem {

using namespace config;
using std::int64_t;
using std::size_t;

namespace {

constexpr double PI = std::numbers::pi;
constexpr double TSYM = static_cast<double>(NSYM) / FS;
constexpr double FRAME_S = equalizer::FRAME_S;
// modem.py's constants (their reasons are there)
constexpr std::array<std::pair<int, int>, 4> SC_STRIDES = {{{24, 49}, {48, 97}, {10, 37}, {4, 17}}};
constexpr std::array<double, 5> SC_SPREADS_SLOW = {0.1, 0.25, 0.5, 1.0, 1.5};
constexpr std::array<double, 6> SC_SPREADS = {2.0, 3.0, 4.5, 7.0, 10.0, 15.0};
constexpr double SC_LOO_TOL = 0.1;
constexpr double SC_FAST_HZ = 3.0;
constexpr double SC_SLOW_RHO = 0.35;
constexpr double SC_WIDEN = 1.25;
constexpr int SC_FRAMES = 4;
constexpr int SC_HOT_PASSES = 1;
constexpr int SC_SEARCH_MIN = 4;  // modem.SC_SEARCH_MIN (its DATA2G_SC_SEARCH_MIN study switch is Python's)

// modem._sc_interior: the spread search's frames, those whose widest reach holds every frame's pilots and no whole
// pilot symbol (whole: their symbols); every frame when fewer than SC_SEARCH_MIN do
std::vector<int> sc_interior(const std::vector<int>& frames, const std::vector<double>& whole, int n_air) {
    const double reach = SYMS_PER_FRAME * SC_FRAMES + 3;
    std::vector<int> out;
    for (int f : frames) {
        const double c = f * SYMS_PER_FRAME + 2.5;
        bool in = c - reach >= 0 && c + reach <= n_air * SYMS_PER_FRAME;
        for (double w : whole) in = in && !(std::abs(w - c) < reach);
        if (in) out.push_back(f);
    }
    return static_cast<int>(out.size()) >= SC_SEARCH_MIN ? out : frames;
}
constexpr double SC_HOT_EXCESS = 4.0;
constexpr double SC_HOT_POWER = 8.0;

bool env_on(const char* name) {  // Python's os.environ.get(name, "1") == "1"
    const char* v = std::getenv(name);
    return !v || std::string_view(v) == "1";
}

ScCells make_cells(int nc) {
    const int N = SYMS_PER_FRAME * nc;
    const auto layout = sc_layout();
    int g = 2 * nc + 1;  // k3d2's
    if (layout == "mix") {
        g = 0;
        for (const auto& [n, s] : SC_STRIDES)
            if (n == nc) g = s;
        if (!g) throw std::logic_error("scattered pilots: no stride for " + std::to_string(nc) + " carriers");
    } else if (layout == "gold") {
        const int g0 = static_cast<int>(std::nearbyint(N / 1.6180339887));
        g = 0;
        for (int d = 0; d < N && !g; ++d)
            for (int v : {g0 + d, g0 - d})
                if (std::gcd(v, N) == 1) {
                    g = v;
                    break;
                }
    }
    std::vector<char> pil(static_cast<size_t>(N), 0);
    for (int i = 0; i < nc; ++i) pil[static_cast<size_t>((static_cast<int64_t>(i) * g % N + N) % N)] = 1;
    ScCells c;
    c.own.assign(static_cast<size_t>(nc), -1);
    for (int i = 0; i < N; ++i) {
        if (pil[static_cast<size_t>(i)]) {
            c.P.push_back(i);
            c.PK.push_back(i % nc);
            if (c.own[static_cast<size_t>(i % nc)] >= 0) throw std::logic_error("a pilot layout must give each carrier one pilot a frame");
            c.own[static_cast<size_t>(i % nc)] = i;
        } else {
            c.Dc.push_back(i);
        }
    }
    if (std::count(c.own.begin(), c.own.end(), -1)) throw std::logic_error("a pilot layout must give each carrier one pilot a frame");
    return c;
}

std::vector<cd> over(std::vector<cd> x, std::span<const cd> by) {
    for (size_t k = 0; k < x.size(); ++k) x[k] /= by[k];
    return x;
}

// modem._sc_slack: the joint estimate's delay-support padding, samples either side
int sc_slack(int nc) { return nc >= 24 ? equalizer::SUPPORT_SLACK : static_cast<int>(std::nearbyint(static_cast<double>(FS) / (nc * RS))); }

// Observations for the joint estimate, sorted by symbol (stable, as np.argsort(kind="stable")).
struct Obs {
    std::vector<double> sym;
    std::vector<int> k, kind;
    std::vector<cd> val;
};

// _sc_joint's cached weights for one pattern of observations about a frame
struct Weights {
    Mat<cd> W;                // (rows, n): h = W @ values; rows: the frame's 6 nc cells, or (pilots) its nc pilots'
    std::vector<double> m;    // (6 nc,): mse / p_sig (not kept for pilots alone)
    std::vector<size_t> col;  // (nc,): each scattered pilot's own observation
    std::vector<double> own;  // (nc,): real(W[P, col])
    size_t bytes() const { return W.data.size() * sizeof(cd) + 64 * m.size(); }
};

// Python keeps 256 patterns; a w48 pattern's W is up to ~3 MB, so the C++ also caps the bytes.
// ponytail: a full cache starts over (Python's rule); an LRU if sessions thrash it.
constexpr size_t CACHE_ENTRIES = 256, CACHE_BYTES = size_t{192} << 20;
std::mutex cache_mu;
std::unordered_map<std::string, std::shared_ptr<const Weights>> cache;
size_t cache_bytes = 0;

int quant(double v, int n) { return static_cast<int>(std::nearbyint(std::log2(std::max(v, 1e-9)) * n)); }

// The solves on split real and imaginary parts (row-major), so the inner loops vectorize: std::complex's
// multiply checks for NaN and does not. a: Hermitian positive definite n x n -> its lower Cholesky factor in
// place (the upper triangle untouched); false if not positive definite.
DATA2G_SIMD_CLONES bool cholesky_split(double* ar, double* ai, size_t n) {
    std::vector<double> cr(n), ci(n);  // column j of L below the diagonal, contiguous
    for (size_t j = 0; j < n; ++j) {
        const double d = ar[j * n + j];
        if (!(d > 0)) return false;
        const double ljj = std::sqrt(d), inv = 1.0 / ljj;
        ar[j * n + j] = ljj;
        ai[j * n + j] = 0.0;
        for (size_t i = j + 1; i < n; ++i) {
            cr[i] = ar[i * n + j] *= inv;
            ci[i] = ai[i * n + j] *= inv;
        }
        // the trailing lower triangle, row by row: a[i][k] -= L[i][j] conj(L[k][j]) (axpy form: no reduction)
        for (size_t i = j + 1; i < n; ++i) {
            const double xr = cr[i], xi = ci[i];
            double* rr = ar + i * n;
            double* ri = ai + i * n;
            for (size_t k = j + 1; k <= i; ++k) {
                rr[k] -= xr * cr[k] + xi * ci[k];
                ri[k] -= xi * cr[k] - xr * ci[k];
            }
        }
    }
    return true;
}

// L Y = X, then (with back) L^H Z = Y, in place on X (n x m, split)
DATA2G_SIMD_CLONES void tri_solves(const double* lr, const double* li, size_t n, double* xr, double* xi, size_t m, bool back) {
    for (size_t i = 0; i < n; ++i) {
        double* yr = xr + i * m;
        double* yi = xi + i * m;
        for (size_t k = 0; k < i; ++k) {
            const double cr = lr[i * n + k], ci = li[i * n + k];
            const double* zr = xr + k * m;
            const double* zi = xi + k * m;
            for (size_t c = 0; c < m; ++c) {
                yr[c] -= cr * zr[c] - ci * zi[c];
                yi[c] -= cr * zi[c] + ci * zr[c];
            }
        }
        const double inv = 1.0 / lr[i * n + i];
        for (size_t c = 0; c < m; ++c) yr[c] *= inv, yi[c] *= inv;
    }
    if (!back) return;
    for (size_t k = n; k-- > 0;) {
        double* yr = xr + k * m;
        double* yi = xi + k * m;
        for (size_t i = k + 1; i < n; ++i) {
            const double cr = lr[i * n + k], ci = -li[i * n + k];  // conj(L[i][k])
            const double* zr = xr + i * m;
            const double* zi = xi + i * m;
            for (size_t c = 0; c < m; ++c) {
                yr[c] -= cr * zr[c] - ci * zi[c];
                yi[c] -= cr * zi[c] + ci * zr[c];
            }
        }
        const double inv = 1.0 / lr[k * n + k];
        for (size_t c = 0; c < m; ++c) yr[c] *= inv, yi[c] *= inv;
    }
}

// pilots: the weights at the frame's scattered pilots alone (the spread search's leave-one-out needs no more)
std::shared_ptr<const Weights> make_weights(std::span<const double> rel, std::span<const int> k, std::span<const int> kd,
                                            const std::vector<cd>& Rf, int nc, double spread,
                                            const std::array<double, 3>& nr, std::span<const double> nk, bool pilots,
                                            const Mat<cd>& Bd) {
    const auto& cells = sc_cells(nc);
    const size_t un = rel.size(), unc = static_cast<size_t>(nc);
    std::vector<size_t> rows;  // the frame's cells weighed
    if (pilots) rows.assign(cells.P.begin(), cells.P.end());
    else for (size_t c = 0; c < static_cast<size_t>(SYMS_PER_FRAME) * unc; ++c) rows.push_back(c);
    const size_t R = rows.size();
    std::vector<double> Ar(un * un), Ai(un * un);
    for (size_t i = 0; i < un; ++i) {
        for (size_t j = 0; j <= i; ++j) {  // the lower triangle: all Cholesky reads
            const cd v = equalizer::doppler_corr((rel[i] - rel[j]) * TSYM, spread) *
                         Rf[static_cast<size_t>(k[i]) * unc + static_cast<size_t>(k[j])];
            Ar[i * un + j] = v.real();
            Ai[i * un + j] = v.imag();
        }
        Ar[i * un + i] += nr[static_cast<size_t>(kd[i])] * (nk.empty() ? 1.0 : nk[static_cast<size_t>(k[i])]);
    }
    // X = Bm^H (n x R): Bm[r][j] = doppler((sym_r - rel_j) Ts) Rf[k_r, k_j]
    Mat<cd> Bm(R, un);
    for (size_t r = 0; r < R; ++r) {
        const double cs = static_cast<double>(rows[r] / unc);  // np.divmod: the cell's symbol, its carrier
        const size_t ck = rows[r] % unc;
        for (size_t j = 0; j < un; ++j)
            Bm[r][j] = equalizer::doppler_corr((cs - rel[j]) * TSYM, spread) * Rf[ck * unc + static_cast<size_t>(k[j])];
    }
    // Factored (the whole frame's cells, a support narrower than the band): Rf = Bd Bd^H / nd, so Bm = E G with
    // G[(s, d), j] = doppler((s - rel_j) Ts) conj(Bd[k_j, d]) / nd and E's row r Bd[k_r, :] in symbol s_r's block:
    // the solves take G^H's 6 nd columns, not Bm^H's 6 nc. Exact (the same algebra); 1.6-2x on w48's slow frames.
    const size_t nd = Bd.cols, S = static_cast<size_t>(SYMS_PER_FRAME);
    const bool factored = !pilots && S * nd < R;
    const size_t RHS = factored ? S * nd : R;
    std::vector<double> Xr(un * RHS), Xi(un * RHS);
    for (size_t j = 0; j < un; ++j) {
        if (factored) {
            const cd* bk = Bd[static_cast<size_t>(k[j])];
            for (size_t sy = 0; sy < S; ++sy) {
                const double dp = equalizer::doppler_corr((static_cast<double>(sy) - rel[j]) * TSYM, spread) / static_cast<double>(nd);
                for (size_t d = 0; d < nd; ++d) {  // conj(G[(s, d), j]) = dp Bd[k_j, d]
                    Xr[j * RHS + sy * nd + d] = dp * bk[d].real();
                    Xi[j * RHS + sy * nd + d] = dp * bk[d].imag();
                }
            }
        } else {
            for (size_t r = 0; r < R; ++r) {
                Xr[j * RHS + r] = Bm[r][j].real();
                Xi[j * RHS + r] = -Bm[r][j].imag();
            }
        }
    }
    // A is Hermitian positive definite (a Gaussian in time times a delay prior, plus noise): Cholesky, else LU
    if (cholesky_split(Ar.data(), Ai.data(), un)) {
        tri_solves(Ar.data(), Ai.data(), un, Xr.data(), Xi.data(), RHS, true);
    } else {
        std::vector<cd> A(un * un), X(un * RHS);
        for (size_t i = 0; i < un; ++i)
            for (size_t j = 0; j < un; ++j)
                A[i * un + j] = equalizer::doppler_corr((rel[i] - rel[j]) * TSYM, spread) *
                                Rf[static_cast<size_t>(k[i]) * unc + static_cast<size_t>(k[j])];
        for (size_t i = 0; i < un; ++i)
            A[i * un + i] += nr[static_cast<size_t>(kd[i])] * (nk.empty() ? 1.0 : nk[static_cast<size_t>(k[i])]);
        for (size_t q = 0; q < X.size(); ++q) X[q] = cd(Xr[q], Xi[q]);
        linalg::lu_solve(A.data(), static_cast<int>(un), X.data(), static_cast<int>(RHS));
        for (size_t q = 0; q < X.size(); ++q) Xr[q] = X[q].real(), Xi[q] = X[q].imag();
    }
    if (factored) {  // W[r][j] = sum_d Bd[k_r, d] conj(X[j][(s_r, d)]), back into the unfactored layout
        std::vector<double> Wr(un * R), Wi(un * R);
        for (size_t r = 0; r < R; ++r) {
            const size_t sy = rows[r] / unc, ck = rows[r] % unc;
            const cd* bk = Bd[ck];
            for (size_t j = 0; j < un; ++j) {
                const double* xr = Xr.data() + j * RHS + sy * nd;
                const double* xi = Xi.data() + j * RHS + sy * nd;
                double a = 0.0, b = 0.0;  // sum_d bk[d] conj(x[d])
                for (size_t d = 0; d < nd; ++d) {
                    a += bk[d].real() * xr[d] + bk[d].imag() * xi[d];
                    b += bk[d].imag() * xr[d] - bk[d].real() * xi[d];
                }
                Wr[j * R + r] = a;
                Wi[j * R + r] = -b;  // stored as X = W^H, as the unfactored solve leaves it
            }
        }
        Xr.swap(Wr);
        Xi.swap(Wi);
    }
    auto w = std::make_shared<Weights>();
    w->W = Mat<cd>(R, un);
    if (!pilots) w->m.resize(R);
    for (size_t r = 0; r < R; ++r) {
        double s = 0.0;
        for (size_t j = 0; j < un; ++j) {
            const cd v(Xr[j * R + r], -Xi[j * R + r]);  // W = X^H
            w->W[r][j] = v;
            s += (v * std::conj(Bm[r][j])).real();
        }
        if (!pilots) w->m[r] = std::max(1.0 - s, 0.0);
    }
    for (size_t i = 0; i < unc; ++i) {
        const int p = cells.P[i], p_sym = p / nc;  // Python's c // nc
        size_t at = un;
        for (size_t j = 0; j < un && at == un; ++j)
            if (kd[j] == 0 && k[j] == cells.PK[i] && rel[j] == static_cast<double>(p_sym)) at = j;
        if (at == un) throw std::logic_error("_sc_joint: a frame's pilot outside its own window");
        w->col.push_back(at);
        w->own.push_back(w->W[pilots ? i : static_cast<size_t>(p)][at].real());
    }
    return w;
}

struct Joint {
    Mat<cd> h;  // (frames * 6, nc), the pilots' units
    Mat<double> mse;
    std::vector<double> loo, model;  // per carrier: the leave-one-out error summed over frames, the model's
};

// modem._sc_joint: 2-D LMMSE of the channel at every cell of `frames` (on-air indices) from the observations in reach.
// pilots: loo and model alone (the spread search), from the weights at the pilots. kw (nc, or empty): each carrier's
// noise over the rest's.
Joint joint(const Obs& o, std::span<const int> frames, std::string_view band, equalizer::Support support, double spread,
            double nr_sc, double nr_full, double p_sig, bool pilots = false, std::span<const double> kw = {}) {
    const int nc = config::band(band).nc;
    const size_t unc = static_cast<size_t>(nc), C = static_cast<size_t>(SYMS_PER_FRAME) * unc;
    const int qs = quant(spread, 8), q1 = quant(nr_sc, 6), q2 = quant(nr_full, 6);
    const double sq = std::pow(2.0, qs / 8.0);
    const std::array<double, 3> nr = {std::pow(2.0, q1 / 6.0), std::pow(2.0, q2 / 6.0), std::pow(2.0, q2 / 6.0) / REF_REPEATS};
    const double reach = SYMS_PER_FRAME * std::clamp(std::ceil(0.7 / sq / FRAME_S), 1.0, static_cast<double>(SC_FRAMES)) + 3;
    std::vector<int> kq;  // to 1.5 dB
    std::vector<double> nk;
    for (double v : kw) {
        kq.push_back(quant(v, 2));
        nk.push_back(std::pow(2.0, kq.back() / 2.0));
    }
    const int slack = sc_slack(nc);
    const auto& cells = sc_cells(nc);
    const size_t F = frames.size();
    std::vector<size_t> lo(F), hi(F);
    std::vector<std::string> keys(F);
    std::vector<std::vector<double>> rels(F);
    for (size_t i = 0; i < F; ++i) {
        const double c = frames[i] * SYMS_PER_FRAME + 2.5;
        lo[i] = static_cast<size_t>(std::lower_bound(o.sym.begin(), o.sym.end(), c - reach) - o.sym.begin());
        hi[i] = static_cast<size_t>(std::lower_bound(o.sym.begin(), o.sym.end(), c + reach) - o.sym.begin());
        auto& rel = rels[i];
        for (size_t j = lo[i]; j < hi[i]; ++j) rel.push_back(o.sym[j] - frames[i] * SYMS_PER_FRAME);
        auto& key = keys[i];
        const int head[] = {support.first, support.second, slack, qs, q1, q2, pilots};
        key.assign(band);
        key.push_back('\0');
        key.append(reinterpret_cast<const char*>(head), sizeof head);
        key.append(reinterpret_cast<const char*>(rel.data()), rel.size() * sizeof(double));
        key.append(reinterpret_cast<const char*>(o.kind.data() + lo[i]), (hi[i] - lo[i]) * sizeof(int));
        key.append(reinterpret_cast<const char*>(kq.data()), kq.size() * sizeof(int));
    }
    std::vector<std::shared_ptr<const Weights>> got(F);
    std::vector<size_t> missing;  // the first frame of each pattern not cached
    {
        std::lock_guard lk(cache_mu);
        std::set<std::string_view> seen;
        for (size_t i = 0; i < F; ++i) {
            if (auto it = cache.find(keys[i]); it != cache.end()) got[i] = it->second;
            else if (seen.insert(keys[i]).second) missing.push_back(i);
        }
    }
    if (!missing.empty()) {
        const auto bbv = equalizer::bb(config::band(band));
        std::vector<cd> Rf = equalizer::gram(bbv, support.first, support.second, slack);
        const double width = support.second - support.first + 2 * slack + 1;
        for (auto& v : Rf) v /= width;
        Mat<cd> Bd(unc, static_cast<size_t>(width));
        for (size_t c = 0; c < unc; ++c)
            for (int d = support.first - slack; d <= support.second + slack; ++d)
                Bd[c][static_cast<size_t>(d - support.first + slack)] = std::polar(1.0, (-2.0 * PI * (bbv[c] * d)) * (1.0 / FS));
        std::vector<std::shared_ptr<const Weights>> made(missing.size());
        pool::parallel_for(missing.size(), [&](size_t m) {
            const size_t i = missing[m];
            made[m] = make_weights(rels[i], std::span<const int>(o.k).subspan(lo[i], hi[i] - lo[i]),
                                   std::span<const int>(o.kind).subspan(lo[i], hi[i] - lo[i]), Rf, nc, sq, nr, nk, pilots, Bd);
        });
        std::lock_guard lk(cache_mu);
        for (size_t m = 0; m < missing.size(); ++m) {
            if (cache.size() >= CACHE_ENTRIES || cache_bytes + made[m]->bytes() > CACHE_BYTES) {
                cache.clear();
                cache_bytes = 0;
            }
            if (cache.emplace(keys[missing[m]], made[m]).second) cache_bytes += made[m]->bytes();
        }
        for (size_t i = 0; i < F; ++i)
            if (!got[i])
                for (size_t m = 0; m < missing.size(); ++m)
                    if (keys[missing[m]] == keys[i]) got[i] = made[m];
    }
    Joint out;
    out.loo.assign(unc, 0.0);
    out.model.assign(unc, 0.0);
    if (!pilots) {
        out.h = Mat<cd>(F * SYMS_PER_FRAME, unc);
        out.mse = Mat<double>(F * SYMS_PER_FRAME, unc);
    }
    std::vector<cd> hp(unc);
    for (size_t i = 0; i < F; ++i) {
        const Weights& w = *got[i];
        const size_t n = hi[i] - lo[i];
        const cd* v = o.val.data() + lo[i];
        auto dot = [&](size_t r) {
            cd s = 0.0;
            const cd* wr = w.W[r];
            for (size_t j = 0; j < n; ++j) s += wr[j] * v[j];
            return s;
        };
        if (pilots) {
            for (size_t p = 0; p < unc; ++p) hp[p] = dot(p);
        } else {
            cd* h = out.h[i * SYMS_PER_FRAME];  // the frame's 6 rows, contiguous
            double* m = out.mse[i * SYMS_PER_FRAME];
            for (size_t c = 0; c < C; ++c) {
                h[c] = dot(c);
                m[c] = p_sig * w.m[c];
            }
            for (size_t p = 0; p < unc; ++p) hp[p] = h[static_cast<size_t>(cells.P[p])];
        }
        for (size_t p = 0; p < unc; ++p) {
            const double own = std::min(w.own[p], 0.99);
            const auto k = static_cast<size_t>(cells.PK[p]);
            out.loo[k] += std::norm((v[w.col[p]] - hp[p]) / (1 - own));
            out.model[k] += p_sig * nr[0] * (nk.empty() ? 1.0 : nk[k]) / (1 - own);
        }
    }
    return out;
}

double sum(std::span<const double> v, const std::vector<char>* skip = nullptr) {
    double s = 0.0;
    for (size_t i = 0; i < v.size(); ++i)
        if (!skip || !(*skip)[i]) s += v[i];
    return s;
}

// modem._sc_hot: carriers whose pilots' leave-one-out error is over the 99th percentile of n_f frames' scatter about
// the median carrier's, and by at least SC_HOT_EXCESS x the pilots' noise nn -> (hot, their error past the model per
// pilot, 0 elsewhere)
std::pair<std::vector<char>, std::vector<double>> sc_hot(const std::vector<double>& loo, const std::vector<double>& model,
                                                         int n_f, double nn) {
    const size_t n = loo.size();
    std::vector<double> t(n), extra(n);
    for (size_t k = 0; k < n; ++k) {
        t[k] = loo[k] / std::max(model[k], 1e-30);
        extra[k] = std::max(loo[k] - model[k], 0.0) / n_f;
    }
    std::vector<double> s = t;
    std::ranges::sort(s);
    const double med = n % 2 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
    const double lim = med * n_f / (n_f - 1.0 / 3) * tables::GAMMA_Q99[static_cast<size_t>(n_f - 1)];
    std::vector<char> hot(n);
    for (size_t k = 0; k < n; ++k) hot[k] = t[k] > lim && extra[k] >= SC_HOT_EXCESS * nn;
    if (std::ranges::all_of(hot, [](char x) { return x != 0; })) std::ranges::fill(hot, 0);
    for (size_t k = 0; k < n; ++k)
        if (!hot[k]) extra[k] = 0.0;
    return {std::move(hot), std::move(extra)};
}

// modem._sc_cal's second value: the submode's scattered pilots' distortion over their power, as its transmitter leaves
// them, measured once on two clean bursts (the first value, their gain, is not used). While measuring, the default
// clip ratio (Python's placeholder).
double sc_cal_ratio(const Spec& spec) {
    static std::mutex mu;
    static std::map<size_t, double> done;
    thread_local std::set<size_t> measuring;
    const size_t key = config::row(spec);
    {
        std::lock_guard lk(mu);
        if (auto it = done.find(key); it != done.end()) return it->second;
    }
    if (measuring.count(key)) return clip_consts(spec).ratio;
    measuring.insert(key);
    struct Unmark {
        size_t k;
        ~Unmark() { measuring.erase(k); }
    } unmark{key};
    const auto& b = waveform::band(spec.band);
    const size_t nc = static_cast<size_t>(b.nc());
    const auto& cells = sc_cells(b.nc());
    const auto& cs = codes::spec(spec);
    const auto& pts = *constellation::find(spec.constellation);
    const int n_cw = std::min(4, max_codewords(spec.sync_band));  // past the short ones (DATA2G_SC_SHORT_FRAMES 0)
    std::uint64_t state = 0;
    std::vector<cd> num_d(nc, 0.0), num_p(nc, 0.0);
    std::vector<double> den_d(nc, 0.0);
    double den_p = 0;
    std::vector<Mat<cd>> parts;
    for (int burst = 0; burst < 2; ++burst) {
        std::vector<std::uint8_t> coded;
        for (int i = 0; i < n_cw; ++i) {
            const auto c = codes::encode(cs, sc_payload(spec.payload_bytes, state), 0, 0, i);
            coded.insert(coded.end(), c.begin(), c.end());
        }
        const auto bits = codes::spread<std::uint8_t>(coded, n_cw, spec.bits_per_cu, 0);
        const auto x = modulate_bits(bits, spec);
        std::vector<double> padded(800, 0.0);
        padded.insert(padded.end(), x.begin(), x.end());
        padded.insert(padded.end(), 800, 0.0);
        const auto r = receive_known(padded, 800 + LEADIN_SAMPLES, 0.0, spec, n_cw);
        const auto X = constellation::modulate(bits, pts);
        const size_t n_f = r.raw.rows / SYMS_PER_FRAME, per = DATA_SYMS_PER_FRAME * nc;
        for (size_t j = 0; j < per; ++j) {
            cd s = 0.0;
            double e = 0.0;
            for (size_t f = 0; f < n_f; ++f) {
                const cd y = data_row(r.raw, f * DATA_SYMS_PER_FRAME + j / nc)[j % nc], xx = X[f * per + j];
                s += y * std::conj(xx);
                e += std::norm(xx);
            }
            const size_t k = static_cast<size_t>(cells.Dc[j]) % nc;
            num_d[k] += s;
            den_d[k] += e;
        }
        const Mat<cd>& ls = r.sc->ls;
        for (size_t i = 0; i < nc; ++i) {
            cd s = 0.0;
            for (size_t f = 0; f < ls.rows; ++f) s += ls[f][i];
            num_p[static_cast<size_t>(cells.PK[i])] += s;
        }
        den_p += static_cast<double>(ls.rows);
        parts.push_back(ls);
    }
    std::vector<cd> gp(nc);
    double gp2 = 0.0, rows = 0.0, err = 0.0;
    for (size_t k = 0; k < nc; ++k) {
        gp[k] = num_p[k] / den_p;
        gp2 += std::norm(gp[k]);
    }
    for (const auto& ls : parts) {
        rows += static_cast<double>(ls.rows);
        for (size_t f = 0; f < ls.rows; ++f)
            for (size_t i = 0; i < nc; ++i) err += std::norm(ls[f][i] - gp[static_cast<size_t>(cells.PK[i])]);
    }
    const double ratio = err / (rows * gp2);
    std::lock_guard lk(mu);
    done.emplace(key, ratio);
    return ratio;
}

// The thermal noise from the preamble's repeats about a cubic in time per carrier (modem._sc_estimate's n0):
// the least-squares residual's power per carrier, by an orthonormal basis of the (R, 4) Vandermonde matrix.
std::vector<double> cubic_residual_noise(const Mat<cd>& reps) {
    const size_t R = reps.rows, nc = reps.cols;
    std::array<std::vector<double>, 4> Q;
    for (size_t c = 0; c < 4; ++c) {
        Q[c].resize(R);
        for (size_t r = 0; r < R; ++r) Q[c][r] = std::pow(static_cast<double>(r) / static_cast<double>(R), 3 - static_cast<int>(c));
        for (int pass = 0; pass < 2; ++pass)  // Gram-Schmidt, twice
            for (size_t p = 0; p < c; ++p) {
                double d = 0;
                for (size_t r = 0; r < R; ++r) d += Q[p][r] * Q[c][r];
                for (size_t r = 0; r < R; ++r) Q[c][r] -= d * Q[p][r];
            }
        double nrm = 0;
        for (double v : Q[c]) nrm += v * v;
        nrm = std::sqrt(nrm);
        for (double& v : Q[c]) v /= nrm;
    }
    std::vector<double> out(nc);
    for (size_t k = 0; k < nc; ++k) {
        double s = 0.0;
        std::vector<cd> res(R);
        for (size_t r = 0; r < R; ++r) res[r] = reps[r][k];
        for (size_t c = 0; c < 4; ++c) {
            cd d = 0.0;
            for (size_t r = 0; r < R; ++r) d += Q[c][r] * reps[r][k];
            for (size_t r = 0; r < R; ++r) res[r] -= d * Q[c][r];
        }
        for (const cd& v : res) s += std::norm(v);
        out[k] = s / static_cast<double>(R - 4);
    }
    return out;
}

}  // namespace

std::vector<std::uint8_t> sc_payload(int n, std::uint64_t& state) {
    std::vector<std::uint8_t> out;
    while (static_cast<int>(out.size()) < n) {
        state += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        for (int b = 0; b < 8; ++b) out.push_back(static_cast<std::uint8_t>(z >> (8 * b)));
    }
    out.resize(static_cast<size_t>(n));
    return out;
}

const ScFlags& sc_flags() {
    static const ScFlags f = [] {
        ScFlags out;
        out.spread_loo = env_on("DATA2G_SC_SPREAD_LOO");
        out.copy_coherent = env_on("DATA2G_SC_COPY_COHERENT");
        out.copy_rows = env_on("DATA2G_SC_COPY_ROWS");
        if (const char* v = std::getenv("DATA2G_SC_RESTORE_ACE")) out.restore_ace = std::stod(v);
        return out;
    }();
    return f;
}

bool sc_spec(const Spec& spec, int n_f) { return sc_band(spec.band) && n_f > 0; }

const ScCells& sc_cells(int nc) {
    static const auto all = [] {
        std::map<int, ScCells> m;
        for (const auto& b : BANDS) m.emplace(b.nc, make_cells(b.nc));
        return m;
    }();
    return all.at(nc);
}

bool sc_combs(int nc) {
    const auto& c = sc_cells(nc);
    if (nc % 3) return false;
    for (size_t i = 0; i < c.P.size(); ++i)
        if (c.P[i] / nc != 2 * (c.PK[i] % 3)) return false;
    return true;
}

std::vector<cd> copy_open(const waveform::Band& b) {
    std::vector<cd> out(b.pilot.size());
    for (size_t k = 0; k < out.size(); ++k) out[k] = std::conj(b.pilot[k]);
    return out;
}

Mat<int> sc_pil(std::string_view band, int n_air, std::optional<int> kc) {
    if (!sc_band(band)) return {};
    const int nc = config::band(band).nc;
    const auto& cells = sc_cells(nc);
    Mat<int> pil(static_cast<size_t>(n_air + 1), static_cast<size_t>(nc));
    for (int f = 0; f < n_air; ++f)
        if (!kc || f != *kc)
            for (int k = 0; k < nc; ++k) pil[static_cast<size_t>(f)][k] = cells.own[static_cast<size_t>(k)] / nc;
    return pil;
}

std::pair<Mat<cd>, Mat<double>> sc_refine(const Received& r, const Mat<cd>& z, const Mat<double>& w) {
    const ScInfo& i = *r.sc;
    const auto& b = waveform::band(r.band);
    const size_t nc = static_cast<size_t>(b.nc()), n_f = i.norm.size(), per = DATA_SYMS_PER_FRAME * nc;
    const auto& cells = sc_cells(b.nc());
    const size_t n_air = static_cast<size_t>(i.n_air);
    if (z.data.size() != n_f * per || w.data.size() != n_f * per) throw std::invalid_argument("sc_refine: shapes disagree");
    Mat<cd> z6(n_air * SYMS_PER_FRAME, nc);
    Mat<double> w6(n_air * SYMS_PER_FRAME, nc);
    auto at = [&](size_t f, size_t j) {  // data cell j (sending order) of data frame f, in the (n_air * 6, nc) rows
        const size_t c = static_cast<size_t>(cells.Dc[j]);
        return (static_cast<size_t>(i.norm[f]) * SYMS_PER_FRAME + c / nc) * nc + c % nc;
    };
    double msum = 0.0;
    for (double v : i.mse.data) msum += v;
    for (size_t f = 0; f < n_f; ++f)
        for (size_t j = 0; j < per; ++j) {
            const size_t q = f * per + j;
            if (w.data[q] > 0) z6.data[at(f, j)] = z.data[q] - i.h.data[q];
            w6.data[at(f, j)] = w.data[q];
        }
    Mat<double> t_rows(n_air, SYMS_PER_FRAME);
    for (size_t a = 0; a < n_air; ++a)
        for (size_t s = 0; s < static_cast<size_t>(SYMS_PER_FRAME); ++s)
            t_rows[a][s] = static_cast<double>((a * SYMS_PER_FRAME + s) * NSYM) / FS;
    const auto [dh, dm] = equalizer::refine(Mat<cd>(0, nc), {}, z6, w6, t_rows, r.support,
                                            msum / static_cast<double>(i.mse.data.size()), r.est.filter_spread_hz,
                                            r.est.n0, equalizer::bb(config::band(r.band)), r.est.slack);
    Mat<cd> h(n_f * DATA_SYMS_PER_FRAME, nc);
    Mat<double> mse(h.rows, nc);
    for (size_t f = 0; f < n_f; ++f)
        for (size_t j = 0; j < per; ++j) {
            const size_t q = f * per + j;
            h.data[q] = i.h.data[q] + dh.data[at(f, j)];
            mse.data[q] = dm.data[at(f, j)];
        }
    return {std::move(h), std::move(mse)};
}

void sc_clear_weights() {
    std::lock_guard lk(cache_mu);
    cache.clear();
    cache_bytes = 0;
}

namespace sc {

Full full(std::span<const cd> z, const HeaderRead& hd, const Mat<cd>& raw, const Mat<cd>& hp, std::optional<int> kc,
          int shift, std::string_view db) {
    const auto& b = waveform::band(hd.band);
    const int R = b.spec->preamble_repeats, ref = std::min(R, REF_REPEATS);
    const auto& dbs = config::band(db);
    const size_t ncd = static_cast<size_t>(dbs.nc), col0 = static_cast<size_t>(dbs.k0 - b.spec->k0);
    Full out{Mat<cd>(0, ncd), {}};
    auto add = [&](std::span<const cd> row, double t) {
        out.rows.data.insert(out.rows.data.end(), row.begin(), row.end());
        ++out.rows.rows;
        out.t.push_back(t);
    };
    std::vector<cd> mean(static_cast<size_t>(b.nc()), 0.0);
    double usum = 0.0;
    for (int r = R - ref; r < R; ++r) {
        const int64_t u = hd.start + PREAMBLE_CP + static_cast<int64_t>(r) * M;
        usum += static_cast<double>(u);
        const auto w = b.demod_window(z, u + shift, DEMOD_BACKOFF);
        for (size_t k = 0; k < mean.size(); ++k) mean[k] += w[k];
    }
    std::vector<cd> row(ncd);
    for (size_t k = 0; k < ncd; ++k) row[k] = mean[col0 + k] / static_cast<double>(ref) / b.pilot[col0 + k];
    add(row, (usum / ref - NCP - static_cast<double>(hd.p0)) / FS);
    const auto hw = b.demod_window(z, hd.p0 - NSYM + NCP + shift, DEMOD_BACKOFF);  // the header's own pilot
    for (size_t k = 0; k < ncd; ++k) row[k] = hw[col0 + k] / b.pilot[col0 + k];
    add(row, -static_cast<double>(NSYM) / FS);
    if (kc) {  // (a copy frame: data band = sync band)
        add(hp.row(static_cast<size_t>(*kc)), static_cast<double>(*kc * FRAME_SAMPLES) / FS);
        const auto open = copy_open(b);
        const cd* last = raw[static_cast<size_t>(*kc * SYMS_PER_FRAME + SYMS_PER_FRAME - 1)];
        for (size_t k = 0; k < ncd; ++k) row[k] = last[k] / open[k];
        add(row, static_cast<double>(*kc * FRAME_SAMPLES + (SYMS_PER_FRAME - 1) * NSYM) / FS);
    }
    add(hp.row(hp.rows - 1), static_cast<double>((hp.rows - 1) * FRAME_SAMPLES) / FS);
    return out;
}

Estimated estimate(std::span<const cd> z, const HeaderRead& hd, const Mat<cd>& raw, const Mat<cd>& hp,
                   std::optional<int> kc, int n_f, int shift, equalizer::Support support, const Spec& spec) {
    const std::string_view db = spec.band;
    const auto& b = waveform::band(db);
    const int nci = b.nc();
    const size_t nc = static_cast<size_t>(nci), n_air = hp.rows - 1, uf = static_cast<size_t>(n_f);
    const auto& cells = sc_cells(nci);
    const auto clip = clip_consts(spec);
    const double g = clip.gain;  // the pilots clip with the data at any length
    std::vector<int> norm;
    for (size_t f = 0; f < n_air; ++f)
        if (!kc || static_cast<int>(f) != *kc) norm.push_back(static_cast<int>(f));
    auto cell = [&](size_t f, int c) {  // data frame f's cell c (of 6 nc)
        return raw[static_cast<size_t>(norm[f]) * SYMS_PER_FRAME + static_cast<size_t>(c) / nc][static_cast<size_t>(c) % nc];
    };
    Mat<cd> ls(uf, nc);
    for (size_t f = 0; f < uf; ++f)
        for (size_t i = 0; i < nc; ++i) ls[f][i] = cell(f, cells.P[i]) / b.pilot[static_cast<size_t>(cells.PK[i])];
    const double ratio = sc_cal_ratio(spec);
    // thermal noise from the preamble's repeats about a cubic in time per carrier
    const auto& sb = waveform::band(hd.band);
    const int R = sb.spec->preamble_repeats;
    double n0 = hd.n0_pre;
    if (R > 4) {
        Mat<cd> reps(static_cast<size_t>(R), static_cast<size_t>(sb.nc()));
        for (int r = 0; r < R; ++r) {
            const auto w = sb.demod_window(z, hd.start + PREAMBLE_CP + static_cast<int64_t>(r) * M, HEADER_BACKOFF);
            std::copy(w.begin(), w.end(), reps[static_cast<size_t>(r)]);
        }
        // the clean carriers' mean (equalizer.per_carrier_noise's band level): a tone between carriers stays in the
        // residual
        const auto nk = equalizer::per_carrier_noise(cubic_residual_noise(reps), R - 4);
        n0 = *std::ranges::min_element(nk);
    }
    // the pilots' power, a tone's carriers out (mean power over SC_HOT_POWER x the median carrier's)
    double ls2 = 0.0;
    size_t n_cool = 0;
    {
        std::vector<double> pk(nc, 0.0);
        for (size_t f = 0; f < uf; ++f)
            for (size_t i = 0; i < nc; ++i) pk[i] += std::norm(ls[f][i]);
        for (double& v : pk) v /= static_cast<double>(uf);
        std::vector<double> srt = pk;
        std::ranges::sort(srt);
        const double med = nc % 2 ? srt[nc / 2] : (srt[nc / 2 - 1] + srt[nc / 2]) / 2;
        std::vector<char> hot(nc);
        for (size_t i = 0; i < nc; ++i) hot[i] = pk[i] > SC_HOT_POWER * med;
        for (size_t f = 0; f < uf; ++f)
            for (size_t i = 0; i < nc; ++i)
                if (!hot[i]) ls2 += std::norm(ls[f][i]), ++n_cool;
    }
    const double p_y = std::max(ls2 / static_cast<double>(n_cool) - n0, 1e-12) / (1 + ratio);
    double n0_y = n0 + ratio * p_y;  // the scattered pilots': thermal + clip
    const double p_sig = p_y / (g * g);
    DataEstimate est;
    est.n_f = n_f;
    est.nc = nci;
    est.clip_ratio = clip.ratio;
    est.gain = clip.gain_for(n_f);
    est.band = config::band(db).name;
    const int width = support.second - support.first;
    const int slack = std::clamp(static_cast<int>(std::floor((52.0 - width) / 2)), 0, equalizer::SUPPORT_SLACK);
    est.slack = nc >= 24 ? slack : sc_slack(nci);
    // slow or fast by the same carrier's pilots a frame apart
    size_t pairs = 0;
    cd acc = 0.0;
    for (size_t f = 0; f + 1 < uf; ++f)
        if (norm[f + 1] - norm[f] == 1) {
            ++pairs;
            for (size_t k = 0; k < nc; ++k)
                acc += std::conj(hp[static_cast<size_t>(norm[f])][k]) * hp[static_cast<size_t>(norm[f + 1])][k];
        }
    const double rho = pairs ? std::abs(acc) / (static_cast<double>(pairs * nc) * p_y) : 0.0;
    const bool loo_rule = sc_flags().spread_loo && nc >= 24;
    std::optional<double> spread;
    if (!loo_rule && rho > SC_SLOW_RHO)
        spread = std::clamp(SC_WIDEN * std::sqrt(-2 * std::log(std::min(rho, 0.99999))) / (PI * FRAME_S), 0.05, SC_FAST_HZ);
    const auto fl = full(z, hd, raw, hp, kc, shift, db);
    // observations: (symbol from the first frame's start, carrier, kind 0 scattered | 1 whole | 2 the preamble's mean)
    Obs o;
    for (size_t f = 0; f < uf; ++f)
        for (size_t i = 0; i < nc; ++i) {
            const int sym = norm[f] * SYMS_PER_FRAME + cells.P[i] / nci;  // Python's P // nc
            o.sym.push_back(static_cast<double>(sym));
            o.k.push_back(cells.PK[i]);
            o.kind.push_back(0);
            o.val.push_back(ls[f][i] / g);
        }
    for (size_t r = 0; r < fl.rows.rows; ++r)
        for (size_t k = 0; k < nc; ++k) {
            o.sym.push_back(std::nearbyint(fl.t[r] * FS / NSYM * 100) / 100);  // np.round(.., 2)
            o.k.push_back(static_cast<int>(k));
            o.kind.push_back(r == 0 ? 2 : 1);
            o.val.push_back(fl.rows[r][k]);
        }
    {  // sorted by symbol, stable
        std::vector<size_t> idx(o.sym.size());
        std::iota(idx.begin(), idx.end(), size_t{0});
        std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t c) { return o.sym[a] < o.sym[c]; });
        Obs s;
        for (size_t j : idx) {
            s.sym.push_back(o.sym[j]);
            s.k.push_back(o.k[j]);
            s.kind.push_back(o.kind[j]);
            s.val.push_back(o.val[j]);
        }
        o = std::move(s);
    }
    double nr_sc = n0_y / (g * g) / p_sig, nr_full = n0 / p_sig;
    if (!spread) {
        // the spread whose model predicts each pilot from the others with the least error (the leave-one-out)
        std::vector<double> cand;
        if (loo_rule) cand.assign(SC_SPREADS_SLOW.begin(), SC_SPREADS_SLOW.end());
        for (double v : SC_SPREADS)
            if (loo_rule || v >= (pairs ? SC_FAST_HZ : equalizer::DEFAULT_SPREAD_HZ)) cand.push_back(v);
        std::vector<Joint> fits;
        std::vector<double> loo;
        std::vector<double> whole;
        for (size_t j = 0; j < o.sym.size(); ++j)
            if (o.kind[j] > 0) whole.push_back(o.sym[j]);
        const auto search = sc_interior(norm, whole, static_cast<int>(n_air));
        for (double v : cand) fits.push_back(joint(o, search, db, support, v, nr_sc, nr_full, p_sig, true));
        // a tone's carriers miss by far more than the rest whatever the spread: scored without them
        for (const auto& f : fits) loo.push_back(sum(f.loo));
        size_t i = static_cast<size_t>(std::min_element(loo.begin(), loo.end()) - loo.begin());
        const auto hot0 = sc_hot(fits[i].loo, fits[i].model, static_cast<int>(search.size()), n0_y / (g * g)).first;
        for (size_t c = 0; c < fits.size(); ++c) loo[c] = sum(fits[c].loo, &hot0);
        i = static_cast<size_t>(std::min_element(loo.begin(), loo.end()) - loo.begin());
        if (loo_rule) {  // the widest within SC_LOO_TOL of the best's error past the noise
            const double tol = loo[i] + SC_LOO_TOL * std::max(loo[i] - sum(fits[i].model, &hot0), 0.0);
            double best = 0.0;
            for (size_t c = 0; c < cand.size(); ++c)
                if (loo[c] <= tol) best = std::max(best, cand[c]);
            spread = best;
        } else {
            spread = cand[i];
        }
    }
    est.spread_hz = std::min(*spread, 4.0);
    est.filter_spread_hz = *spread;
    auto J = joint(o, norm, db, support, *spread, nr_sc, nr_full, p_sig);
    // carriers whose pilots miss by far more than the rest (a tone): their own noise, their pilots weighed by it, and
    // left out of the band-wide excess
    auto [hot, extra] = sc_hot(J.loo, J.model, n_f, n0_y / (g * g));
    const bool any_hot = std::ranges::any_of(hot, [](char x) { return x != 0; });
    std::vector<double> kw;
    auto weigh = [&] {
        kw.resize(nc);
        for (size_t k = 0; k < nc; ++k) kw[k] = 1 + extra[k] / (n0_y / (g * g));
    };
    for (int pass = 0; pass < (any_hot ? SC_HOT_PASSES : 0); ++pass) {
        weigh();
        J = joint(o, norm, db, support, *spread, nr_sc, nr_full, p_sig, false, kw);
        for (size_t k = 0; k < nc; ++k) extra[k] = hot[k] ? std::max(extra[k] + (J.loo[k] - J.model[k]) / n_f, 0.0) : 0.0;
    }
    std::vector<char> warm(nc, 0);  // the carriers left out of the band's excess: the hot ones, unless all are
    if (!std::ranges::all_of(hot, [](char x) { return x != 0; })) warm = hot;
    double d = 0.0;
    size_t n_rest = 0;
    for (size_t k = 0; k < nc; ++k)
        if (!warm[k]) d += J.loo[k] - J.model[k], ++n_rest;
    const double excess = std::max(d, 0.0) / static_cast<double>(static_cast<size_t>(n_f) * n_rest);
    if (excess > 0.25 * n0_y / (g * g)) {
        // the pilots miss their neighbours' prediction by more than the model's noise: noise to the data too
        n0 += excess * g * g;
        n0_y += excess * g * g;
        nr_sc = n0_y / (g * g) / p_sig;
        nr_full = n0 / p_sig;
        for (double& e : extra) e = std::max(e - excess, 0.0);
        if (!kw.empty()) weigh();
        J = joint(o, norm, db, support, *spread, nr_sc, nr_full, p_sig, false, kw);
    }
    est.n0 = est.n0_snr = n0;
    est.n0_k.assign(nc, n0);
    if (any_hot) {  // per data cell, as h: a frame's cells are in sending order, not by carrier
        est.n0_k.resize(uf * DATA_SYMS_PER_FRAME * nc);
        for (size_t f = 0; f < uf; ++f)
            for (size_t j = 0; j < DATA_SYMS_PER_FRAME * nc; ++j)
                est.n0_k[f * DATA_SYMS_PER_FRAME * nc + j] = n0 + extra[static_cast<size_t>(cells.Dc[j]) % nc] * g * g;
    }
    est.p_sig = p_sig;
    const size_t per = DATA_SYMS_PER_FRAME * nc;
    auto info = std::make_shared<ScInfo>();
    info->norm = norm;
    info->n_air = static_cast<int>(n_air);
    info->h = Mat<cd>(uf * DATA_SYMS_PER_FRAME, nc);
    info->mse = Mat<double>(uf * DATA_SYMS_PER_FRAME, nc);
    est.h = Mat<cd>(uf * DATA_SYMS_PER_FRAME, nc);
    est.mse = Mat<double>(uf * DATA_SYMS_PER_FRAME, nc);
    Estimated out;
    out.raw = Mat<cd>(uf * SYMS_PER_FRAME, nc);
    for (size_t f = 0; f < uf; ++f) {
        for (size_t k = 0; k < nc; ++k) out.raw[f * SYMS_PER_FRAME][k] = hp[static_cast<size_t>(norm[f])][k] * b.pilot[k];
        for (size_t j = 0; j < per; ++j) {
            const size_t c = static_cast<size_t>(cells.Dc[j]);
            const size_t q = f * per + j;
            info->h.data[q] = J.h[f * SYMS_PER_FRAME + c / nc][c % nc];
            info->mse.data[q] = J.mse[f * SYMS_PER_FRAME + c / nc][c % nc];
            est.h.data[q] = g * info->h.data[q];
            est.mse.data[q] = g * g * info->mse.data[q];
            out.raw[f * SYMS_PER_FRAME + 1 + j / nc][j % nc] = cell(f, cells.Dc[j]);
        }
    }
    info->full = fl.rows;
    info->ls = std::move(ls);
    info->n0_y = n0_y;
    out.est = std::move(est);
    out.info = std::move(info);
    return out;
}

std::pair<std::vector<double>, double> copy_rows(std::span<const cd> z0, const Lock& lock) {
    const auto& b = waveform::band(lock.band);
    const int64_t pc = lock.copy->pc;
    if (pc + SYMS_PER_FRAME * NSYM > static_cast<int64_t>(z0.size())) return {{}, 0.0};
    const auto z = waveform::freq_correct(z0, lock.cfo);
    const auto open = copy_open(b);
    const auto r0 = over(b.demod_window(z, pc + NCP, DEMOD_BACKOFF), open);
    const auto r5 = over(b.demod_window(z, pc + (SYMS_PER_FRAME - 1) * NSYM + NCP, DEMOD_BACKOFF), open);
    cd d = 0.0;
    double e0 = 0.0, e5 = 0.0;
    for (size_t k = 0; k < r0.size(); ++k) {
        d += std::conj(r5[k]) * r0[k];
        e0 += std::norm(r0[k]);
        e5 += std::norm(r5[k]);
    }
    const double c = std::abs(d) / (std::sqrt(e0) * std::sqrt(e5) + 1e-30);
    cd s = 0.0;
    for (size_t k = 0; k + 1 < r0.size(); ++k) {
        const cd a = r0[k + 1] * std::conj(r0[k]) + r5[k + 1] * std::conj(r5[k]);
        s += a / (std::abs(a) + 1e-30);
    }
    return {{c, c}, std::abs(s) / static_cast<double>(r0.size() - 1)};
}

}  // namespace sc

}  // namespace data2g::modem
