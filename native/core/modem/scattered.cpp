// data2g/modem.py's scattered pilots (config.SC_BANDS): the layout, the joint 2-D LMMSE
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
constexpr size_t SC_SOLVE_LANES = 2;  // joint()'s weight solves at once (its reason there)
constexpr double SC_SEP_HZ = 1.5;  // modem.SC_SEP_HZ
constexpr double SC_SEP_REPORT_HZ = 1.0;  // modem.SC_SEP_REPORT_HZ
constexpr double SC_JOINT_REACH2_HZ = 2.0;  // modem.SC_JOINT_REACH2_HZ: joint()'s reach at most 2 frames from here up
constexpr int SC_SEARCH_MIN = 4;  // modem.SC_SEARCH_MIN

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
constexpr double SC_HOT_EXCESS = 1.0;  // modem.SC_HOT_EXCESS (its reasons there)
constexpr double SC_HOT_POWER = 8.0;
constexpr std::array<double, 8> GAMMA_Q999 = {
    6.907755278982, 4.616706738226, 3.742957414138, 3.265560194797,
    2.958829844507, 2.742457533947, 2.580233834314, 2.453272174423,
};

ScCells make_cells(int nc) {
    const int N = SYMS_PER_FRAME * nc;
    int g = 0;
    for (const auto& [n, s] : SC_STRIDES)
        if (n == nc) g = s;
    if (!g) throw std::logic_error("scattered pilots: no stride for " + std::to_string(nc) + " carriers");
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

ScCells make_intra_cells(int nc, const ScIntra& sp, int f_mod) {
    const int N = SYMS_PER_FRAME * nc;
    const int c = ((f_mod % sp.n_combs) + sp.n_combs) % sp.n_combs;
    ScCells out;
    for (int i = 0; i < N; ++i) {
        const int sym = i / nc, k = i % nc;
        bool is_sym = false;
        for (int s = 0; s < sp.n_syms; ++s) is_sym = is_sym || sp.syms[static_cast<size_t>(s)] == sym;
        if (is_sym && k % sp.K == c) {
            out.P.push_back(i);
            out.PK.push_back(k);
        } else {
            out.Dc.push_back(i);
        }
    }
    if (static_cast<int>(out.P.size()) != nc) throw std::logic_error("intra pilot count mismatch");
    return out;
}

std::vector<cd> over(std::vector<cd> x, std::span<const cd> by) {
    for (size_t k = 0; k < x.size(); ++k) x[k] /= by[k];
    return x;
}

// modem._sc_slack: the joint estimate's delay-support padding, samples either side
int sc_slack(int nc, std::optional<equalizer::Support> support = {}) {
    if (nc < 24) return static_cast<int>(std::nearbyint(static_cast<double>(FS) / (nc * RS)));
    if (support) {
        const auto intra = sc_intra_spec(nc);
        const int max_d = (intra && intra->K == 2) ? 79 : 52;
        return std::clamp(static_cast<int>(std::floor((max_d - (support->second - support->first)) / 2.0)), 0,
                          equalizer::SUPPORT_SLACK);
    }
    return equalizer::SUPPORT_SLACK;
}

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
    // forward-only (the whole frame's cells, Cholesky): W not formed. h = Y^H (L^-1 values) with Y = L^-1 Bm^H, or
    // factored (Y = L^-1 G^H, nd > 0) h_r = sum_d Bd[k_r, d] (Y^H L^-1 values)[(s_r, d)]
    bool fwd = false;
    size_t n = 0, nd = 0, rhs = 0;
    std::vector<double> Lr, Li;  // the Cholesky factor (lower, row-major n x n)
    std::vector<double> Yr, Yi;  // (n x rhs)
    Mat<cd> Bd;                  // factored: the support's steering vectors (nc x nd)
    size_t bytes() const {
        return W.data.size() * sizeof(cd) + 64 * m.size() + 8 * (Lr.size() + Li.size() + Yr.size() + Yi.size());
    }
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
// place (the upper triangle untouched); false if not positive definite. cr/ci: caller-allocated length-n scratch
// (no std::vector inside DATA2G_SIMD_CLONES: Android NDK libc++ marks its ctor/dtor hidden across target_clones).
DATA2G_SIMD_CLONES bool cholesky_split(double* ar, double* ai, double* cr, double* ci, size_t n) {
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
DATA2G_SIMD_CLONES void tri_solves(const double* lr, const double* li, size_t n, double* xr, double* xi, size_t m, bool back,
                                   bool forward = true) {
    for (size_t i = 0; i < (forward ? n : 0); ++i) {
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
                                            const Mat<cd>& Bd, const ScCells& cells) {
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
    std::vector<double> cr(un), ci(un);
    const bool chol = cholesky_split(Ar.data(), Ai.data(), cr.data(), ci.data(), un);
    if (chol && !pilots) {
        // the whole frame's weights, forward only: Y = L^-1 RHS, kept with L. The back substitution (half the
        // solves) and W are not needed: each frame's h is Y^H (L^-1 its values), one forward solve a frame,
        // and mse_r = 1 - |L^-1 Bm^H e_r|^2, own_p = (L^-1 Bm^H e_p)^H (L^-1 e_col)
        tri_solves(Ar.data(), Ai.data(), un, Xr.data(), Xi.data(), RHS, false);
        auto w = std::make_shared<Weights>();
        w->fwd = true;
        w->n = un;
        w->rhs = RHS;
        w->m.resize(R);
        // z_r = L^-1 Bm^H e_r as a combination of Y's columns: unfactored Y[:, r]; factored sum_d Y[:, (s, d)]
        // conj(Bd[k_r, d])
        auto z_of = [&](size_t r, std::vector<cd>& z) {
            z.assign(un, 0.0);
            if (!factored) {
                for (size_t j = 0; j < un; ++j) z[j] = cd(Xr[j * RHS + r], Xi[j * RHS + r]);
                return;
            }
            const size_t sy = rows[r] / unc, ck = rows[r] % unc;
            const cd* bk = Bd[ck];
            for (size_t j = 0; j < un; ++j) {
                cd acc = 0.0;
                for (size_t d = 0; d < nd; ++d) acc += cd(Xr[j * RHS + sy * nd + d], Xi[j * RHS + sy * nd + d]) * std::conj(bk[d]);
                z[j] = acc;
            }
        };
        if (factored) {  // M_s = Y_s^H Y_s per symbol (nd x nd), then mse_r = 1 - b_r^T M_s conj(b_r)
            std::vector<cd> Ms(S * nd * nd, 0.0);
            for (size_t sy = 0; sy < S; ++sy)
                for (size_t j = 0; j < un; ++j) {
                    const double* yr = Xr.data() + j * RHS + sy * nd;
                    const double* yi = Xi.data() + j * RHS + sy * nd;
                    for (size_t d = 0; d < nd; ++d) {
                        const cd a(yr[d], -yi[d]);
                        cd* row = Ms.data() + (sy * nd + d) * nd;
                        for (size_t e = 0; e < nd; ++e) row[e] += a * cd(yr[e], yi[e]);
                    }
                }
            for (size_t r = 0; r < R; ++r) {
                const size_t sy = rows[r] / unc;
                const cd* b = Bd[rows[r] % unc];
                cd q = 0.0;
                for (size_t d = 0; d < nd; ++d) {
                    cd t = 0.0;
                    const cd* row = Ms.data() + (sy * nd + d) * nd;
                    for (size_t e = 0; e < nd; ++e) t += row[e] * std::conj(b[e]);
                    q += b[d] * t;
                }
                w->m[r] = std::max(1.0 - q.real(), 0.0);
            }
        } else {
            for (size_t r = 0; r < R; ++r) {
                double s2 = 0.0;
                for (size_t j = 0; j < un; ++j) s2 += Xr[j * RHS + r] * Xr[j * RHS + r] + Xi[j * RHS + r] * Xi[j * RHS + r];
                w->m[r] = std::max(1.0 - s2, 0.0);
            }
        }
        // own_p = W[P_p, col_p] = (L^-H z_p)[col_p]: the pilots' z as nc columns, one back substitution for all
        std::vector<cd> z;
        std::vector<double> Zr(un * unc), Zi(un * unc);
        for (size_t i = 0; i < unc; ++i) {
            const int p = cells.P[i], p_sym = p / nc;
            size_t at = un;
            for (size_t j = 0; j < un && at == un; ++j)
                if (kd[j] == 0 && k[j] == cells.PK[i] && rel[j] == static_cast<double>(p_sym)) at = j;
            if (at == un) throw std::logic_error("_sc_joint: a frame's pilot outside its own window");
            w->col.push_back(at);
            z_of(static_cast<size_t>(p), z);
            for (size_t j = 0; j < un; ++j) Zr[j * unc + i] = z[j].real(), Zi[j * unc + i] = z[j].imag();
        }
        tri_solves(Ar.data(), Ai.data(), un, Zr.data(), Zi.data(), unc, true, false);
        for (size_t i = 0; i < unc; ++i) w->own.push_back(Zr[w->col[i] * unc + i]);
        w->Lr.swap(Ar);
        w->Li.swap(Ai);
        w->Yr.swap(Xr);
        w->Yi.swap(Xi);
        if (factored) {
            w->nd = nd;
            w->Bd = Bd;
        }
        return w;
    }
    if (chol) {
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
    const double reach =
        SYMS_PER_FRAME * std::clamp(std::ceil(0.7 / sq / FRAME_S), 1.0, sq >= SC_JOINT_REACH2_HZ ? 2.0 : SC_FRAMES) + 3;
    std::vector<int> kq;  // to 1.5 dB
    std::vector<double> nk;
    for (double v : kw) {
        kq.push_back(quant(v, 2));
        nk.push_back(std::pow(2.0, kq.back() / 2.0));
    }
    const int slack = sc_slack(nc);
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
        key.append(reinterpret_cast<const char*>(o.k.data() + lo[i]), (hi[i] - lo[i]) * sizeof(int));
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
        const size_t lanes = std::min(missing.size(), SC_SOLVE_LANES);
        pool::parallel_for(lanes, [&](size_t lane) {
            for (size_t m = lane; m < missing.size(); m += lanes) {
                const size_t i = missing[m];
                made[m] = make_weights(rels[i], std::span<const int>(o.k).subspan(lo[i], hi[i] - lo[i]),
                                       std::span<const int>(o.kind).subspan(lo[i], hi[i] - lo[i]), Rf, nc, sq, nr, nk,
                                       pilots, Bd, sc_frame_cells(nc, frames[i]));
            }
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
        const auto& cells = sc_frame_cells(nc, frames[i]);
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
        } else if (w.fwd) {
            cd* h = out.h[i * SYMS_PER_FRAME];
            double* m = out.mse[i * SYMS_PER_FRAME];
            std::vector<double> ur(n), ui(n);
            for (size_t r = 0; r < n; ++r) {
                double ar = v[r].real(), ai = v[r].imag();
                const double* lr = w.Lr.data() + r * n;
                const double* li = w.Li.data() + r * n;
                for (size_t c = 0; c < r; ++c) {
                    ar -= lr[c] * ur[c] - li[c] * ui[c];
                    ai -= lr[c] * ui[c] + li[c] * ur[c];
                }
                ur[r] = ar / lr[r];
                ui[r] = ai / lr[r];
            }
            std::vector<double> tr(w.rhs, 0.0), ti(w.rhs, 0.0);
            for (size_t j = 0; j < n; ++j) {
                const double* yr = w.Yr.data() + j * w.rhs;
                const double* yi = w.Yi.data() + j * w.rhs;
                const double a = ur[j], b = ui[j];
                for (size_t q = 0; q < w.rhs; ++q) {
                    tr[q] += yr[q] * a + yi[q] * b;
                    ti[q] += yr[q] * b - yi[q] * a;
                }
            }
            for (size_t c = 0; c < C; ++c) {
                if (w.nd) {
                    const size_t sy = c / unc;
                    const cd* bk = w.Bd[c % unc];
                    cd s = 0.0;
                    for (size_t d = 0; d < w.nd; ++d) s += bk[d] * cd(tr[sy * w.nd + d], ti[sy * w.nd + d]);
                    h[c] = s;
                } else {
                    h[c] = cd(tr[c], ti[c]);
                }
                m[c] = p_sig * w.m[c];
            }
            for (size_t p = 0; p < unc; ++p) hp[p] = h[static_cast<size_t>(cells.P[p])];
        } else {
            cd* h = out.h[i * SYMS_PER_FRAME];
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

// modem._sc_separable: per carrier in time (its own observations), then per symbol across carriers through Rf.
Joint separable(const Obs& o, std::span<const int> frames, std::string_view band, equalizer::Support support,
                double spread, double nr_sc, double nr_full, double p_sig, bool pilots = false,
                std::span<const double> kw = {}) {
    const int nci = config::band(band).nc;
    const size_t nc = static_cast<size_t>(nci), S = static_cast<size_t>(SYMS_PER_FRAME), F = frames.size();
    const int qs = quant(spread, 8), q1 = quant(nr_sc, 6), q2 = quant(nr_full, 6);
    const double sq = std::pow(2.0, qs / 8.0);
    const std::array<double, 3> nr = {std::pow(2.0, q1 / 6.0), std::pow(2.0, q2 / 6.0), std::pow(2.0, q2 / 6.0) / REF_REPEATS};
    std::vector<int> kq(nc, 0);
    std::vector<double> nk(nc, 1.0);
    for (size_t k = 0; k < kw.size(); ++k) {
        kq[k] = quant(kw[k], 2);
        nk[k] = std::pow(2.0, kq[k] / 2.0);
    }
    const double reach = SYMS_PER_FRAME * std::clamp(std::ceil(0.7 / sq / FRAME_S), 1.0, static_cast<double>(SC_FRAMES)) + 3;
    const int slack = sc_slack(nci);
    std::vector<cd> Rf = equalizer::gram(equalizer::bb(config::band(band)), support.first, support.second, slack);
    const double width = support.second - support.first + 2 * slack + 1;
    for (auto& v : Rf) v /= width;
    const int per = sc_period(nci);
    std::vector<std::vector<std::vector<int>>> psyms_by_mod(static_cast<size_t>(per), std::vector<std::vector<int>>(nc));
    for (int f_ = 0; f_ < per; ++f_) {
        const auto& cells = sc_frame_cells(nci, f_);
        for (size_t i = 0; i < nc; ++i)
            psyms_by_mod[static_cast<size_t>(f_)][static_cast<size_t>(cells.PK[i])].push_back(cells.P[i] / nci);
    }
    std::vector<std::vector<size_t>> by(nc);
    for (size_t j = 0; j < o.sym.size(); ++j) by[static_cast<size_t>(o.k[j])].push_back(j);
    struct TW {
        std::vector<double> W;  // (S, n) unbiased time weights
        std::vector<double> nu;  // (S,) their noise
        std::vector<size_t> cols;
    };
    struct PilObs {
        size_t i, k;
        int s_p;
        double ownt;
        cd yp;
    };
    std::map<std::string, TW> tcache;
    Mat<cd> xh(F * S, nc);
    Mat<double> ev(F * S, nc);
    std::ranges::fill(ev.data, 1e8);
    std::vector<PilObs> pil_obs;
    std::vector<double> rel;
    std::vector<int> kk;
    for (size_t k = 0; k < nc; ++k) {
        const auto& idx = by[k];
        for (size_t i = 0; i < F; ++i) {
            const double c = frames[i] * SYMS_PER_FRAME + 2.5;
            const auto lo = static_cast<size_t>(std::lower_bound(idx.begin(), idx.end(), c - reach,
                                                                 [&](size_t j, double v) { return o.sym[j] < v; }) - idx.begin());
            const auto hi = static_cast<size_t>(std::lower_bound(idx.begin(), idx.end(), c + reach,
                                                                 [&](size_t j, double v) { return o.sym[j] < v; }) - idx.begin());
            const size_t n = hi - lo;
            if (!n) continue;
            rel.resize(n);
            kk.resize(n);
            for (size_t m = 0; m < n; ++m) {
                rel[m] = o.sym[idx[lo + m]] - frames[i] * SYMS_PER_FRAME;
                kk[m] = o.kind[idx[lo + m]];
            }
            const auto& psyms = psyms_by_mod[static_cast<size_t>(((frames[i] % per) + per) % per)][k];
            std::string key;
            const int head[] = {qs, q1, q2, kq[k], static_cast<int>(psyms.size())};
            key.append(reinterpret_cast<const char*>(head), sizeof head);
            key.append(reinterpret_cast<const char*>(psyms.data()), psyms.size() * sizeof(int));
            key.append(reinterpret_cast<const char*>(rel.data()), n * sizeof(double));
            key.append(reinterpret_cast<const char*>(kk.data()), n * sizeof(int));
            auto it = tcache.find(key);
            if (it == tcache.end()) {
                std::vector<double> A(n * n), X(n * S), Bm(S * n);
                for (size_t a = 0; a < n; ++a) {
                    for (size_t bb = 0; bb < n; ++bb) A[a * n + bb] = equalizer::doppler_corr((rel[a] - rel[bb]) * TSYM, sq);
                    A[a * n + a] += nr[static_cast<size_t>(kk[a])] * nk[k];
                }
                for (size_t sy = 0; sy < S; ++sy)
                    for (size_t a = 0; a < n; ++a) {
                        Bm[sy * n + a] = equalizer::doppler_corr((static_cast<double>(sy) - rel[a]) * TSYM, sq);
                        X[a * S + sy] = Bm[sy * n + a];
                    }
                linalg::lu_solve(A.data(), static_cast<int>(n), X.data(), static_cast<int>(S));
                TW t{std::vector<double>(S * n), std::vector<double>(S), {}};
                for (size_t sy = 0; sy < S; ++sy) {
                    double d = 0.0;
                    for (size_t a = 0; a < n; ++a) d += X[a * S + sy] * Bm[sy * n + a];
                    const double e = std::clamp(1 - d, 1e-9, 0.999);
                    for (size_t a = 0; a < n; ++a) t.W[sy * n + a] = X[a * S + sy] / (1 - e);
                    t.nu[sy] = e / (1 - e);
                }
                for (int s_p : psyms) {
                    size_t col = n;
                    for (size_t a = 0; a < n && col == n; ++a)
                        if (rel[a] == static_cast<double>(s_p) && kk[a] == 0) col = a;
                    if (col == n) throw std::logic_error("_sc_separable: a frame's pilot outside its own window");
                    t.cols.push_back(col);
                }
                it = tcache.emplace(std::move(key), std::move(t)).first;
            }
            const TW& t = it->second;
            for (size_t sy = 0; sy < S; ++sy) {
                cd acc = 0.0;
                for (size_t a = 0; a < n; ++a) acc += t.W[sy * n + a] * o.val[idx[lo + a]];
                xh[i * S + sy][k] = acc;
                ev[i * S + sy][k] = t.nu[sy];
            }
            for (size_t p = 0; p < psyms.size(); ++p) {
                const int s_p = psyms[p];
                const size_t col = t.cols[p];
                pil_obs.push_back({i, k, s_p, t.W[static_cast<size_t>(s_p) * n + col], o.val[idx[lo + col]]});
            }
        }
    }
    struct FW {
        std::vector<cd> G;  // (nc, nc) Rf (Rf + E)^-1
        std::vector<double> m;
    };
    std::map<std::vector<double>, FW> fcache;
    auto fweights = [&](const double* e) -> const FW& {
        std::vector<double> key(e, e + nc);
        auto it = fcache.find(key);
        if (it != fcache.end()) return it->second;
        std::vector<cd> A(Rf), X(Rf);
        for (size_t k = 0; k < nc; ++k) A[k * nc + k] += e[k];
        linalg::lu_solve(A.data(), nci, X.data(), nci);
        FW w{std::vector<cd>(nc * nc), std::vector<double>(nc)};
        for (size_t r = 0; r < nc; ++r) {
            double s2 = 0.0;
            for (size_t c = 0; c < nc; ++c) {
                const cd g = std::conj(X[c * nc + r]);
                w.G[r * nc + c] = g;
                s2 += (g * std::conj(Rf[r * nc + c])).real();
            }
            w.m[r] = std::max(Rf[r * nc + r].real() - s2, 0.0);
        }
        return fcache.emplace(std::move(key), std::move(w)).first->second;
    };
    Joint out;
    out.loo.assign(nc, 0.0);
    out.model.assign(nc, 0.0);
    out.h = Mat<cd>(F * S, nc);
    if (!pilots) out.mse = Mat<double>(F * S, nc);
    std::vector<std::vector<char>> sym_need;
    if (pilots) {
        sym_need.assign(F, std::vector<char>(S, 0));
        for (const auto& po : pil_obs) sym_need[po.i][static_cast<size_t>(po.s_p)] = 1;
    }
    for (size_t i = 0; i < F; ++i) {
        for (size_t sy = 0; sy < S; ++sy) {
            if (pilots && !sym_need[i][sy]) continue;
            const FW& w = fweights(ev[i * S + sy]);
            const cd* x = xh[i * S + sy];
            for (size_t k = 0; k < nc; ++k) {
                cd acc = 0.0;
                for (size_t c = 0; c < nc; ++c) acc += w.G[k * nc + c] * x[c];
                out.h[i * S + sy][k] = acc;
                if (!pilots) out.mse[i * S + sy][k] = p_sig * w.m[k];
            }
        }
    }
    for (const auto& po : pil_obs) {
        const FW& w = fweights(ev[po.i * S + static_cast<size_t>(po.s_p)]);
        const double own = std::min(w.G[po.k * nc + po.k].real() * po.ownt, 0.99);
        out.loo[po.k] += std::norm((po.yp - out.h[po.i * S + static_cast<size_t>(po.s_p)][po.k]) / (1 - own));
        out.model[po.k] += p_sig * nr[0] * nk[po.k] / (1 - own);
    }
    if (pilots) out.h = Mat<cd>();
    return out;
}

// --- Slepian-mode separable 2-D LMMSE (_slepian_basis, _sc_slepian_prep, _sc_slepian) ---

struct SlepianZero {
    size_t nc = 0, r = 0;
    std::vector<cd> Rf0;       // (nc, nc) at d = 0..W-1
    Mat<cd> Uf0;               // (nc, r)
    std::vector<double> lam;   // (r,)
    Mat<double> Uf_abs2;       // (nc, r)
    std::vector<double> rem;   // (nc,)
};

std::shared_ptr<const SlepianZero> slepian_zero(std::string_view band, int W) {
    static std::mutex mu;
    static std::map<std::pair<std::string_view, int>, std::shared_ptr<const SlepianZero>> tbl;
    const auto key = std::make_pair(band, W);
    {
        std::lock_guard lk(mu);
        if (auto it = tbl.find(key); it != tbl.end()) return it->second;
    }
    const auto bbv = equalizer::bb(config::band(band));
    const size_t nc = bbv.size();
    std::vector<cd> Rf0 = equalizer::gram(bbv, 0, W - 1, 0);
    for (auto& v : Rf0) v /= static_cast<double>(W);
    const double df = nc > 1 ? bbv[1] - bbv[0] : static_cast<double>(RS);
    const double M_eff = df > 0.0 ? static_cast<double>(FS) / df : 0.0;
    bool uniform = nc > 1 && static_cast<double>(W) < M_eff;
    for (size_t k = 1; uniform && k + 1 < nc; ++k)
        if (std::abs((bbv[k + 1] - bbv[k]) - df) > 1e-6) uniform = false;
    std::vector<double> evals(nc, 0.0);
    std::vector<cd> full_vecs(nc * nc, 0.0);
    std::vector<size_t> order(nc);
    std::iota(order.begin(), order.end(), size_t{0});
    if (uniform) {
        // Grunbaum's (1981) commuting tridiagonal matrix: strictly simple eigenvalues (no degenerate-subspace rotation).
        std::vector<cd> T(nc * nc, 0.0), vecs;
        const double cw = std::cos(PI * static_cast<double>(W) / M_eff);
        for (size_t k = 0; k < nc; ++k)
            T[k * nc + k] = -cw * std::cos(PI * (2.0 * static_cast<double>(k) - static_cast<double>(nc - 1)) / M_eff);
        for (size_t k = 0; k + 1 < nc; ++k) {
            const double off = std::sin(PI * static_cast<double>(k + 1) / M_eff) *
                               std::sin(PI * static_cast<double>(nc - 1 - k) / M_eff);
            T[k * nc + k + 1] = off;
            T[(k + 1) * nc + k] = off;
        }
        const auto ev_t = linalg::hermitian_eigen(T, vecs, static_cast<int>(nc));
        std::stable_sort(order.begin(), order.end(), [&](size_t i, size_t j) { return ev_t[i] > ev_t[j]; });
        const double d_mid = 0.5 * static_cast<double>(W - 1);
        std::vector<cd> ph(nc);
        for (size_t k = 0; k < nc; ++k) ph[k] = std::polar(1.0, -2.0 * PI * bbv[k] * d_mid / FS);
        for (size_t m = 0; m < nc; ++m) {
            const size_t col = order[m];
            for (size_t k = 0; k < nc; ++k) full_vecs[k * nc + m] = ph[k] * vecs[k * nc + col].real();
            double lam_m = 0.0;
            for (size_t i = 0; i < nc; ++i) {
                cd acc = 0.0;
                for (size_t j = 0; j < nc; ++j) acc += Rf0[i * nc + j] * full_vecs[j * nc + m];
                lam_m += (std::conj(full_vecs[i * nc + m]) * acc).real();
            }
            evals[m] = lam_m;
        }
        std::iota(order.begin(), order.end(), size_t{0});
    } else {
        std::vector<cd> a = Rf0;
        evals = linalg::hermitian_eigen(a, full_vecs, static_cast<int>(nc));
        std::stable_sort(order.begin(), order.end(), [&](size_t i, size_t j) { return evals[i] > evals[j]; });
    }
    const double thr = 1e-4 * evals[order[0]];
    size_t r = 0;
    while (r < nc && evals[order[r]] > thr) ++r;
    auto z = std::make_shared<SlepianZero>();
    z->nc = nc;
    z->r = r;
    z->Rf0 = std::move(Rf0);
    z->Uf0 = Mat<cd>(nc, r);
    z->lam.resize(r);
    z->Uf_abs2 = Mat<double>(nc, r);
    z->rem.assign(nc, 0.0);
    for (size_t m = 0; m < r; ++m) {
        z->lam[m] = evals[order[m]];
        for (size_t k = 0; k < nc; ++k) {
            const cd u = full_vecs[k * nc + order[m]];
            z->Uf0[k][m] = u;
            z->Uf_abs2[k][m] = std::norm(u);
        }
    }
    for (size_t k = 0; k < nc; ++k) {
        double s = 0.0;
        for (size_t m = 0; m < r; ++m) s += z->Uf_abs2[k][m] * z->lam[m];
        z->rem[k] = std::max(z->Rf0[k * nc + k].real() - s, 0.0);
    }
    std::lock_guard lk(mu);
    return tbl.emplace(key, std::move(z)).first->second;
}

void back_solve_real(const double* l, int n, double* b, int nrhs) {
    for (int i = n - 1; i >= 0; --i) {
        double* bi = b + static_cast<size_t>(i) * static_cast<size_t>(nrhs);
        for (int j = i + 1; j < n; ++j) {
            const double lji = l[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i)];
            const double* bj = b + static_cast<size_t>(j) * static_cast<size_t>(nrhs);
            for (int c = 0; c < nrhs; ++c) bi[c] -= lji * bj[c];
        }
        const double inv = 1.0 / l[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(i)];
        for (int c = 0; c < nrhs; ++c) bi[c] *= inv;
    }
}

struct SlepianPrep {
    size_t nc = 0, r = 0, Q = 0;
    std::array<double, 3> nr{};
    std::vector<double> nk;
    Mat<cd> Uf;  // (nc, r)
    std::shared_ptr<const SlepianZero> zero;
    std::vector<double> u_syms;
    std::vector<size_t> starts, ends;
    Mat<cd> y_tilde;      // (Q, r)
    Mat<double> nu_row;   // (Q, r)
    std::vector<std::shared_ptr<const Mat<cd>>> Wu_list;  // each (r, M_q)
};

SlepianPrep slepian_prep(const Obs& o, std::string_view band, equalizer::Support support, double nr_sc, double nr_full,
                         std::span<const double> kw = {}) {
    const auto& bs = config::band(band);
    const int nci = bs.nc;
    const size_t nc = static_cast<size_t>(nci);
    const int q1 = quant(nr_sc, 6), q2 = quant(nr_full, 6);
    SlepianPrep p;
    p.nc = nc;
    p.nr = {std::pow(2.0, q1 / 6.0), std::pow(2.0, q2 / 6.0), std::pow(2.0, q2 / 6.0) / REF_REPEATS};
    std::vector<int> kq;
    p.nk.assign(nc, 1.0);
    for (size_t k = 0; k < kw.size(); ++k) {
        kq.push_back(quant(kw[k], 2));
        p.nk[k] = std::pow(2.0, kq.back() / 2.0);
    }
    int d0 = support.first, d1 = support.second;
    const auto intra = sc_intra_spec(nci);
    const int max_d = (intra && intra->K == 2) ? 79 : 52;
    if (d1 - d0 > max_d) {
        const int mid = (d0 + d1) >= 0 ? (d0 + d1) / 2 : (d0 + d1 - 1) / 2;
        d0 = mid - max_d / 2;
        d1 = d0 + max_d;
    }
    const int slack = sc_slack(nci, equalizer::Support{d0, d1});
    const int W = (d1 - d0) + 2 * slack + 1;
    p.zero = slepian_zero(bs.name, W);
    const size_t r = p.zero->r;
    p.r = r;
    const auto bbv = equalizer::bb(bs);
    const int d_shift = d0 - slack;
    std::vector<cd> phi(nc);
    for (size_t k = 0; k < nc; ++k) phi[k] = std::polar(1.0, -2.0 * PI * bbv[k] * d_shift / FS);
    p.Uf = Mat<cd>(nc, r);
    for (size_t k = 0; k < nc; ++k)
        for (size_t m = 0; m < r; ++m) p.Uf[k][m] = phi[k] * p.zero->Uf0[k][m];
    std::vector<cd> Rf(nc * nc);
    for (size_t a = 0; a < nc; ++a)
        for (size_t b_ = 0; b_ < nc; ++b_) Rf[a * nc + b_] = phi[a] * std::conj(phi[b_]) * p.zero->Rf0[a * nc + b_];

    // Group sorted observations by unique symbol time
    for (size_t j = 0; j < o.sym.size();) {
        const double s = o.sym[j];
        size_t e = j + 1;
        while (e < o.sym.size() && o.sym[e] == s) ++e;
        p.u_syms.push_back(s);
        p.starts.push_back(j);
        p.ends.push_back(e);
        j = e;
    }
    p.Q = p.u_syms.size();
    p.y_tilde = Mat<cd>(p.Q, r);
    p.nu_row = Mat<double>(p.Q, r);
    p.Wu_list.resize(p.Q);

    struct RowGot {
        std::shared_ptr<const Mat<cd>> Wu;
        std::vector<double> nu;
    };
    std::unordered_map<std::string, RowGot> rcache;
    for (size_t q_idx = 0; q_idx < p.Q; ++q_idx) {
        const size_t a = p.starts[q_idx], b_ = p.ends[q_idx], Mq = b_ - a;
        std::string rkey;
        rkey.append(reinterpret_cast<const char*>(o.k.data() + a), Mq * sizeof(int));
        rkey.append(reinterpret_cast<const char*>(o.kind.data() + a), Mq * sizeof(int));
        auto it = rcache.find(rkey);
        if (it == rcache.end()) {
            bool uniform_full = kq.empty() && Mq == nc;
            const int kd0 = o.kind[a];
            if (uniform_full)
                for (size_t u = 0; u < nc; ++u)
                    if (o.k[a + u] != static_cast<int>(u) || o.kind[a + u] != kd0) {
                        uniform_full = false;
                        break;
                    }
            auto Wu = std::make_shared<Mat<cd>>(r, Mq);
            std::vector<double> nu(r);
            if (uniform_full) {
                for (size_t m = 0; m < r; ++m) {
                    for (size_t u = 0; u < nc; ++u) (*Wu)[m][u] = std::conj(p.Uf[u][m]);
                    nu[m] = p.nr[static_cast<size_t>(kd0)];
                }
            } else {
                // B = Rf[ks, :] @ Uf (Mq x r)
                Mat<cd> B(Mq, r);
                for (size_t u = 0; u < Mq; ++u) {
                    const size_t ku = static_cast<size_t>(o.k[a + u]);
                    for (size_t m = 0; m < r; ++m) {
                        cd acc = 0.0;
                        for (size_t j = 0; j < nc; ++j) acc += Rf[ku * nc + j] * p.Uf[j][m];
                        B[u][m] = acc;
                    }
                }
                std::vector<double> Ar(Mq * Mq), Ai(Mq * Mq), Xr(Mq * r), Xi(Mq * r);
                for (size_t u = 0; u < Mq; ++u) {
                    const size_t ku = static_cast<size_t>(o.k[a + u]);
                    for (size_t v = 0; v <= u; ++v) {
                        const cd val = Rf[ku * nc + static_cast<size_t>(o.k[a + v])];
                        Ar[u * Mq + v] = val.real();
                        Ai[u * Mq + v] = val.imag();
                    }
                    Ar[u * Mq + u] += p.nr[static_cast<size_t>(o.kind[a + u])] * p.nk[ku];
                    for (size_t m = 0; m < r; ++m) {
                        Xr[u * r + m] = B[u][m].real();
                        Xi[u * r + m] = B[u][m].imag();
                    }
                }
                std::vector<double> cr(Mq), ci(Mq);
                if (cholesky_split(Ar.data(), Ai.data(), cr.data(), ci.data(), Mq)) {
                    tri_solves(Ar.data(), Ai.data(), Mq, Xr.data(), Xi.data(), r, true);
                } else {
                    std::vector<cd> A(Mq * Mq), X(B.data);
                    for (size_t u = 0; u < Mq; ++u) {
                        const size_t ku = static_cast<size_t>(o.k[a + u]);
                        for (size_t v = 0; v < Mq; ++v) A[u * Mq + v] = Rf[ku * nc + static_cast<size_t>(o.k[a + v])];
                        A[u * Mq + u] += p.nr[static_cast<size_t>(o.kind[a + u])] * p.nk[ku];
                    }
                    linalg::lu_solve(A.data(), static_cast<int>(Mq), X.data(), static_cast<int>(r));
                    for (size_t idx = 0; idx < Mq * r; ++idx) Xr[idx] = X[idx].real(), Xi[idx] = X[idx].imag();
                }
                // G = X^H (r x Mq); c_m = Re(sum_u G[m, u] * B[u, m])
                for (size_t m = 0; m < r; ++m) {
                    double cm = 0.0;
                    for (size_t u = 0; u < Mq; ++u) {
                        const cd gm(Xr[u * r + m], -Xi[u * r + m]);
                        cm += (gm * B[u][m]).real();
                    }
                    const double lm = p.zero->lam[m];
                    cm = std::clamp(cm, 1e-8 * lm, (1.0 - 1e-7) * lm);
                    const double scale = lm / cm;
                    for (size_t u = 0; u < Mq; ++u) (*Wu)[m][u] = scale * cd(Xr[u * r + m], -Xi[u * r + m]);
                    nu[m] = std::min(lm * (lm - cm) / cm, 1e8 * lm);
                }
            }
            it = rcache.emplace(std::move(rkey), RowGot{std::move(Wu), std::move(nu)}).first;
        }
        p.Wu_list[q_idx] = it->second.Wu;
        const Mat<cd>& Wu = *it->second.Wu;
        const cd* yq = o.val.data() + a;
        for (size_t m = 0; m < r; ++m) {
            cd acc = 0.0;
            const cd* wm = Wu[m];
            for (size_t u = 0; u < Mq; ++u) acc += wm[u] * yq[u];
            p.y_tilde[q_idx][m] = acc;
            p.nu_row[q_idx][m] = it->second.nu[m];
        }
    }
    return p;
}

Joint slepian(const Obs& o, const SlepianPrep& p, std::span<const int> frames, double spread, double p_sig,
              bool pilots = false) {
    const size_t nc = p.nc, r = p.r, S = static_cast<size_t>(SYMS_PER_FRAME), F = frames.size();
    const int qs = quant(spread, 8);
    const double sq = std::pow(2.0, qs / 8.0);
    const double reach = SYMS_PER_FRAME * std::clamp(std::ceil(0.7 / sq / FRAME_S), 1.0, static_cast<double>(SC_FRAMES)) + 3;
    struct TGot {
        std::vector<double> W;  // (r, S, J)
        Mat<double> mse;        // (S, nc)
    };
    std::unordered_map<std::string, TGot> tcache;
    Joint out;
    out.loo.assign(nc, 0.0);
    out.model.assign(nc, 0.0);
    if (!pilots) {
        out.h = Mat<cd>(F * S, nc);
        out.mse = Mat<double>(F * S, nc);
    }
    std::vector<double> rel;
    std::vector<cd> a_hat(S * r);
    for (size_t i = 0; i < F; ++i) {
        const double c = frames[i] * SYMS_PER_FRAME + 2.5;
        const auto lo = static_cast<size_t>(std::lower_bound(p.u_syms.begin(), p.u_syms.end(), c - reach) - p.u_syms.begin());
        const auto hi = static_cast<size_t>(std::lower_bound(p.u_syms.begin(), p.u_syms.end(), c + reach) - p.u_syms.begin());
        const size_t J = hi - lo;
        rel.resize(J);
        for (size_t j = 0; j < J; ++j) rel[j] = p.u_syms[lo + j] - frames[i] * SYMS_PER_FRAME;
        std::string tkey;
        tkey.append(reinterpret_cast<const char*>(rel.data()), J * sizeof(double));
        tkey.append(reinterpret_cast<const char*>(p.nu_row[lo]), J * r * sizeof(double));
        auto it = tcache.find(tkey);
        if (it == tcache.end()) {
            std::vector<double> Kt(J * J), Kdt(S * J);
            for (size_t a = 0; a < J; ++a)
                for (size_t b_ = 0; b_ < J; ++b_) Kt[a * J + b_] = equalizer::doppler_corr((rel[a] - rel[b_]) * TSYM, sq);
            for (size_t sy = 0; sy < S; ++sy)
                for (size_t j = 0; j < J; ++j) Kdt[sy * J + j] = equalizer::doppler_corr((static_cast<double>(sy) - rel[j]) * TSYM, sq);
            TGot tg;
            tg.W.resize(r * S * J);
            std::vector<double> e_mode(r * S);
            std::vector<double> A(J * J), X(J * S);
            for (size_t m = 0; m < r; ++m) {
                A = Kt;
                const double inv_lam = 1.0 / p.zero->lam[m];
                for (size_t j = 0; j < J; ++j) A[j * J + j] += p.nu_row[lo + j][m] * inv_lam;
                for (size_t j = 0; j < J; ++j)
                    for (size_t sy = 0; sy < S; ++sy) X[j * S + sy] = Kdt[sy * J + j];
                if (linalg::cholesky(A.data(), static_cast<int>(J))) {
                    linalg::forward_solve(A.data(), static_cast<int>(J), X.data(), static_cast<int>(S));
                    back_solve_real(A.data(), static_cast<int>(J), X.data(), static_cast<int>(S));
                } else {
                    A = Kt;
                    for (size_t j = 0; j < J; ++j) A[j * J + j] += p.nu_row[lo + j][m] * inv_lam;
                    for (size_t j = 0; j < J; ++j)
                        for (size_t sy = 0; sy < S; ++sy) X[j * S + sy] = Kdt[sy * J + j];
                    linalg::lu_solve(A.data(), static_cast<int>(J), X.data(), static_cast<int>(S));
                }
                for (size_t sy = 0; sy < S; ++sy) {
                    double dot = 0.0;
                    double* wdst = tg.W.data() + (m * S + sy) * J;
                    const double* krow = Kdt.data() + sy * J;
                    for (size_t j = 0; j < J; ++j) {
                        const double wv = X[j * S + sy];
                        wdst[j] = wv;
                        dot += wv * krow[j];
                    }
                    e_mode[m * S + sy] = p.zero->lam[m] * std::max(1.0 - dot, 0.0);
                }
            }
            if (!pilots) {
                tg.mse = Mat<double>(S, nc);
                for (size_t sy = 0; sy < S; ++sy)
                    for (size_t k = 0; k < nc; ++k) {
                        double s = p.zero->rem[k];
                        const double* u2 = p.zero->Uf_abs2[k];
                        for (size_t m = 0; m < r; ++m) s += u2[m] * e_mode[m * S + sy];
                        tg.mse[sy][k] = p_sig * s;
                    }
            }
            it = tcache.emplace(std::move(tkey), std::move(tg)).first;
        }
        const TGot& tg = it->second;
        if (!pilots) {
            std::ranges::fill(a_hat, cd(0.0));
            for (size_t m = 0; m < r; ++m)
                for (size_t sy = 0; sy < S; ++sy) {
                    cd acc = 0.0;
                    const double* wm = tg.W.data() + (m * S + sy) * J;
                    for (size_t j = 0; j < J; ++j) acc += wm[j] * p.y_tilde[lo + j][m];
                    a_hat[sy * r + m] = acc;
                }
            for (size_t sy = 0; sy < S; ++sy) {
                cd* hrow = out.h[i * S + sy];
                double* mrow = out.mse[i * S + sy];
                const double* ms = tg.mse[sy];
                const cd* ah = a_hat.data() + sy * r;
                for (size_t k = 0; k < nc; ++k) {
                    cd acc = 0.0;
                    const cd* uk = p.Uf[k];
                    for (size_t m = 0; m < r; ++m) acc += uk[m] * ah[m];
                    hrow[k] = acc;
                    mrow[k] = ms[k];
                }
            }
        }
        for (size_t j = 0; j < J; ++j) {
            const double rj = rel[j];
            if (rj < 0.0 || rj >= static_cast<double>(S) || rj != std::floor(rj)) continue;
            const size_t q_idx = lo + j;
            const size_t a = p.starts[q_idx], b_ = p.ends[q_idx], Mq = b_ - a;
            if (o.kind[a] != 0) continue;
            const size_t s_p = static_cast<size_t>(rj);
            const Mat<cd>& Wu = *p.Wu_list[q_idx];
            if (pilots) {
                for (size_t m = 0; m < r; ++m) {
                    cd acc = 0.0;
                    const double* wm = tg.W.data() + (m * S + s_p) * J;
                    for (size_t j2 = 0; j2 < J; ++j2) acc += wm[j2] * p.y_tilde[lo + j2][m];
                    a_hat[s_p * r + m] = acc;
                }
            }
            for (size_t u = 0; u < Mq; ++u) {
                const size_t ku = static_cast<size_t>(o.k[a + u]);
                const cd* uk = p.Uf[ku];
                cd hv;
                if (pilots) {
                    hv = 0.0;
                    const cd* ah = a_hat.data() + s_p * r;
                    for (size_t m = 0; m < r; ++m) hv += uk[m] * ah[m];
                } else {
                    hv = out.h[i * S + s_p][ku];
                }
                double own = 0.0;
                for (size_t m = 0; m < r; ++m) own += (uk[m] * tg.W[(m * S + s_p) * J + j] * Wu[m][u]).real();
                own = std::min(own, 0.99);
                out.loo[ku] += std::norm((o.val[a + u] - hv) / (1.0 - own));
                out.model[ku] += p_sig * p.nr[0] * p.nk[ku] / (1.0 - own);
            }
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

std::vector<int> sc_pilot_counts(int nc, std::span<const int> frames) {
    if (sc_period(nc) == 1 && !sc_intra_spec(nc)) return std::vector<int>(static_cast<size_t>(nc), static_cast<int>(frames.size()));
    std::vector<int> out(static_cast<size_t>(nc), 0);
    for (int f : frames) {
        const auto& cells = sc_frame_cells(nc, f);
        for (int k : cells.PK) ++out[static_cast<size_t>(k)];
    }
    return out;
}

// modem._sc_hot: carriers whose pilots' leave-one-out error is over the 99th percentile of n_p_k observations' scatter
std::pair<std::vector<char>, std::vector<double>> sc_hot(const std::vector<double>& loo, const std::vector<double>& model,
                                                         std::span<const int> n_p_k, double nn) {
    const size_t n = loo.size();
    std::vector<double> t(n, 0.0), extra(n, 0.0), act_t;
    for (size_t k = 0; k < n; ++k) {
        if (n_p_k[k] > 0) {
            t[k] = loo[k] / std::max(model[k], 1e-30);
            extra[k] = std::max(loo[k] - model[k], 0.0) / static_cast<double>(n_p_k[k]);
            act_t.push_back(t[k]);
        }
    }
    if (act_t.empty()) return {std::vector<char>(n, 0), std::vector<double>(n, 0.0)};
    std::ranges::sort(act_t);
    const size_t na = act_t.size();
    const double med = na % 2 ? act_t[na / 2] : (act_t[na / 2 - 1] + act_t[na / 2]) / 2;
    std::vector<char> hot(n, 0);
    size_t n_hot = 0;
    for (size_t k = 0; k < n; ++k) {
        if (n_p_k[k] <= 0) continue;
        const double nf = static_cast<double>(n_p_k[k]);
        const double q = tables::GAMMA_Q99[static_cast<size_t>(std::clamp(n_p_k[k] - 1, 0, tables::GAMMA_Q99_MAX - 1))];
        const double lim = med * nf / std::max(nf - 1.0 / 3, 0.5) * q;
        if (t[k] > lim && extra[k] >= SC_HOT_EXCESS * nn) hot[k] = 1, ++n_hot;
    }
    if (n_hot == na) std::ranges::fill(hot, 0);
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
    const auto& cs = codes::spec(spec);
    const auto& pts = *constellation::find(spec.constellation);
    const int n_cw = std::min(4, max_codewords(spec.sync_band));
    std::uint64_t state = 0;
    std::vector<cd> num_d(nc, 0.0), num_p(nc, 0.0);
    std::vector<double> den_d(nc, 0.0), den_p(nc, 0.0);
    struct Part {
        Mat<cd> ls;
        std::vector<int> norm;
    };
    std::vector<Part> parts;
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
        const Mat<cd>& ls = r.sc->ls;
        const auto& norm = r.sc->norm;
        for (size_t f = 0; f < n_f; ++f) {
            const auto& cells = sc_frame_cells(b.nc(), norm[f]);
            for (size_t j = 0; j < per; ++j) {
                const cd y = data_row(r.raw, f * DATA_SYMS_PER_FRAME + j / nc)[j % nc], xx = X[f * per + j];
                const size_t k = static_cast<size_t>(cells.Dc[j]) % nc;
                num_d[k] += y * std::conj(xx);
                den_d[k] += std::norm(xx);
            }
            for (size_t i = 0; i < nc; ++i) {
                const size_t k = static_cast<size_t>(cells.PK[i]);
                num_p[k] += ls[f][i];
                den_p[k] += 1.0;
            }
        }
        parts.push_back({ls, norm});
    }
    std::vector<cd> gp(nc, 0.0);
    for (size_t k = 0; k < nc; ++k)
        if (den_p[k] > 0.0) gp[k] = num_p[k] / den_p[k];
    double err = 0.0, tot_gp = 0.0;
    for (const auto& pt : parts) {
        for (size_t f = 0; f < pt.ls.rows; ++f) {
            const auto& cells = sc_frame_cells(b.nc(), pt.norm[f]);
            for (size_t i = 0; i < nc; ++i) {
                const cd gpk = gp[static_cast<size_t>(cells.PK[i])];
                err += std::norm(pt.ls[f][i] - gpk);
                tot_gp += std::norm(gpk);
            }
        }
    }
    const double ratio = err / std::max(tot_gp, 1e-30);
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

bool sc_spec(const Spec& spec, int n_f) { return sc_band(spec.band) && n_f > 0; }

std::optional<ScIntra> sc_intra_spec(int nc) {
    if (nc < 24) return std::nullopt;
    return ScIntra{3, {0, 2, 4}, 3, 3};  // fk3d2: K=3, symbols 0/2/4, 3 frame-staggered combs
}

int sc_period(int nc) {
    const auto sp = sc_intra_spec(nc);
    return sp ? sp->n_combs : 1;
}

const ScCells& sc_cells(int nc) {
    static const auto all = [] {
        std::map<int, ScCells> m;
        for (const auto& b : BANDS)
            if (sc_band(b.name)) m.emplace(b.nc, make_cells(b.nc));
        return m;
    }();
    return all.at(nc);
}

const ScCells& sc_frame_cells(int nc, int f) {
    const auto sp = sc_intra_spec(nc);
    if (!sp) return sc_cells(nc);
    static const auto all = [] {
        std::map<std::pair<int, int>, ScCells> m;
        for (const auto& b : BANDS) {
            if (!sc_band(b.name)) continue;
            if (const auto isp = sc_intra_spec(b.nc)) {
                for (int c = 0; c < isp->n_combs; ++c) m.emplace(std::make_pair(b.nc, c), make_intra_cells(b.nc, *isp, c));
            }
        }
        return m;
    }();
    const int fm = ((f % sp->n_combs) + sp->n_combs) % sp->n_combs;
    return all.at({nc, fm});
}

bool sc_combs(int nc) {
    if (sc_intra_spec(nc)) return false;
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

std::pair<Mat<int>, Mat<int>> sc_pil_intra(std::string_view band, int n_air, std::optional<int> kc) {
    const int nc = config::band(band).nc;
    const size_t unc = static_cast<size_t>(nc);
    Mat<int> ps(static_cast<size_t>(n_air + 1), unc), pk(static_cast<size_t>(n_air + 1), unc);
    for (int f = 0; f <= n_air; ++f)
        for (int k = 0; k < nc; ++k) pk[static_cast<size_t>(f)][static_cast<size_t>(k)] = k;
    for (int f = 0; f < n_air; ++f) {
        if (kc && f == *kc) continue;
        const auto& fc = sc_frame_cells(nc, f);
        for (size_t i = 0; i < unc; ++i) {
            ps[static_cast<size_t>(f)][i] = fc.P[i] / nc;
            pk[static_cast<size_t>(f)][i] = fc.PK[i];
        }
    }
    return {std::move(ps), std::move(pk)};
}

std::pair<Mat<cd>, Mat<double>> sc_refine(const Received& r, const Mat<cd>& z, const Mat<double>& w) {
    const ScInfo& i = *r.sc;
    const auto& b = waveform::band(r.band);
    const size_t nc = static_cast<size_t>(b.nc()), n_f = i.norm.size(), per = DATA_SYMS_PER_FRAME * nc;
    const size_t n_air = static_cast<size_t>(i.n_air);
    if (z.data.size() != n_f * per || w.data.size() != n_f * per) throw std::invalid_argument("sc_refine: shapes disagree");
    Mat<cd> z6(n_air * SYMS_PER_FRAME, nc);
    Mat<double> w6(n_air * SYMS_PER_FRAME, nc);
    auto at = [&](size_t f, size_t j) {
        const auto& cells = sc_frame_cells(b.nc(), i.norm[f]);
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

std::pair<double, double> residual_cfo(const Mat<cd>& raw0, const Mat<cd>& hp, std::optional<int> kc, int n_air,
                                       std::string_view db, std::span<const char> exclude_db) {
    const auto& b = waveform::band(db);
    const int nci = b.nc();
    const size_t nc = static_cast<size_t>(nci);
    const auto intra = sc_intra_spec(nci);
    if (!intra) throw std::logic_error("sc::residual_cfo: non-intra layout");
    const int K = intra->K, ds = intra->syms[1] - intra->syms[0];
    const double dt = static_cast<double>(ds * NSYM) / FS;
    const size_t m_comb = nc / static_cast<size_t>(K);
    std::vector<cd> u0, u1;
    std::vector<int> kk;
    for (int fa = 0; fa < n_air; ++fa) {
        if (kc && fa == *kc) continue;
        const auto& fc = sc_frame_cells(nci, fa);
        std::vector<std::vector<cd>> rows(static_cast<size_t>(intra->n_syms), std::vector<cd>(m_comb));
        for (int si = 0; si < intra->n_syms; ++si) {
            const size_t ridx = static_cast<size_t>(fa * SYMS_PER_FRAME + intra->syms[static_cast<size_t>(si)]);
            for (size_t j = 0; j < m_comb; ++j) {
                const size_t k = static_cast<size_t>(fc.PK[j]);
                rows[static_cast<size_t>(si)][j] = raw0[ridx][k] / b.pilot[k];
            }
        }
        for (int si = 0; si + 1 < intra->n_syms; ++si) {
            for (size_t j = 0; j < m_comb; ++j) {
                u0.push_back(rows[static_cast<size_t>(si)][j]);
                u1.push_back(rows[static_cast<size_t>(si + 1)][j]);
                kk.push_back(fc.PK[j]);
            }
        }
        if ((kc && fa + 1 == *kc) || fa + 1 == n_air) {
            const cd* nxt = hp[static_cast<size_t>(fa + 1)];
            for (size_t j = 0; j < m_comb; ++j) {
                const size_t k = static_cast<size_t>(fc.PK[j]);
                u0.push_back(rows.back()[j]);
                u1.push_back(nxt[k]);
                kk.push_back(static_cast<int>(k));
            }
        } else if (intra->n_combs == 1 && fa + 1 < n_air) {
            const cd* nxt = raw0[static_cast<size_t>((fa + 1) * SYMS_PER_FRAME)];
            for (size_t j = 0; j < m_comb; ++j) {
                const size_t k = static_cast<size_t>(fc.PK[j]);
                u0.push_back(rows.back()[j]);
                u1.push_back(nxt[k] / b.pilot[k]);
                kk.push_back(static_cast<int>(k));
            }
        }
    }
    if (u0.empty()) return {0.0, dt};
    std::vector<int> cnt(nc, 0);
    std::vector<double> pk(nc, 0.0);
    std::vector<cd> prod_k(nc, 0.0);
    for (size_t i = 0; i < u0.size(); ++i) {
        const size_t k = static_cast<size_t>(kk[i]);
        ++cnt[k];
        pk[k] += 0.5 * (std::norm(u0[i]) + std::norm(u1[i]));
        prod_k[k] += u1[i] * std::conj(u0[i]);
    }
    std::vector<double> act_pk;
    int max_cnt = 0;
    for (size_t k = 0; k < nc; ++k)
        if (cnt[k] > 0) {
            pk[k] /= static_cast<double>(cnt[k]);
            act_pk.push_back(pk[k]);
            max_cnt = std::max(max_cnt, cnt[k]);
        }
    std::ranges::sort(act_pk);
    const size_t na = act_pk.size();
    const double med_pk = na % 2 ? act_pk[na / 2] : (act_pk[na / 2 - 1] + act_pk[na / 2]) / 2;
    std::vector<char> hot(nc, 0);
    size_t n_hot_act = 0;
    for (size_t k = 0; k < nc; ++k) {
        if (cnt[k] == 0) {
            hot[k] = 1;
            continue;
        }
        bool h = pk[k] > equalizer::HOT_CARRIER * med_pk;
        if (exclude_db.size() == nc && exclude_db[k]) h = true;
        if (h) hot[k] = 1, ++n_hot_act;
    }
    if (n_hot_act == na)
        for (size_t k = 0; k < nc; ++k) hot[k] = cnt[k] == 0 ? 1 : 0;
    cd d = 0.0;
    for (size_t k = 0; k < nc; ++k)
        if (!hot[k]) d += prod_k[k];
    if (max_cnt >= equalizer::HOT_INNOV_MIN_PILOTS - 1 && std::abs(d) > 0.0) {
        const cd ph = d / std::abs(d);
        std::vector<double> innov(nc, 0.0), act_in;
        for (size_t i = 0; i < u0.size(); ++i) innov[static_cast<size_t>(kk[i])] += std::norm(u1[i] - ph * u0[i]);
        for (size_t k = 0; k < nc; ++k)
            if (cnt[k] > 0) {
                innov[k] /= static_cast<double>(cnt[k]);
                act_in.push_back(innov[k]);
            }
        std::ranges::sort(act_in);
        const double med_i = na % 2 ? act_in[na / 2] : (act_in[na / 2 - 1] + act_in[na / 2]) / 2;
        std::vector<char> hot2 = hot;
        size_t n_h2 = 0;
        for (size_t k = 0; k < nc; ++k) {
            if (cnt[k] > 0 && innov[k] > equalizer::HOT_INNOV * med_i) hot2[k] = 1;
            if (cnt[k] > 0 && hot2[k]) ++n_h2;
        }
        if (n_h2 < na && hot2 != hot) {
            d = 0.0;
            for (size_t k = 0; k < nc; ++k)
                if (!hot2[k]) d += prod_k[k];
        }
    }
    return {std::abs(d) > 0.0 ? std::arg(d) / (2 * PI * dt) : 0.0, dt};
}

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
    const auto intra = sc_intra_spec(nci);
    const int per_f = sc_period(nci);
    const auto clip = clip_consts(spec);
    const double g = clip.gain;
    std::vector<int> norm;
    for (size_t f = 0; f < n_air; ++f)
        if (!kc || static_cast<int>(f) != *kc) norm.push_back(static_cast<int>(f));
    auto cell = [&](size_t f, int c) {
        return raw[static_cast<size_t>(norm[f]) * SYMS_PER_FRAME + static_cast<size_t>(c) / nc][static_cast<size_t>(c) % nc];
    };
    Mat<cd> ls(uf, nc);
    for (size_t f = 0; f < uf; ++f) {
        const auto& cells = sc_frame_cells(nci, norm[f]);
        for (size_t i = 0; i < nc; ++i) ls[f][i] = cell(f, cells.P[i]) / b.pilot[static_cast<size_t>(cells.PK[i])];
    }
    const double ratio = sc_cal_ratio(spec);
    const auto& sb = waveform::band(hd.band);
    const int R = sb.spec->preamble_repeats;
    double n0 = hd.n0_pre;
    if (R > 4) {
        Mat<cd> reps(static_cast<size_t>(R), static_cast<size_t>(sb.nc()));
        for (int r = 0; r < R; ++r) {
            const auto w = sb.demod_window(z, hd.start + PREAMBLE_CP + static_cast<int64_t>(r) * M, HEADER_BACKOFF);
            std::copy(w.begin(), w.end(), reps[static_cast<size_t>(r)]);
        }
        const auto nk = equalizer::per_carrier_noise(cubic_residual_noise(reps), R - 4);
        n0 = *std::ranges::min_element(nk);
    }
    const auto n_p_norm = sc_pilot_counts(nci, norm);
    std::vector<double> pk(nc, 0.0), act_pk;
    for (size_t f = 0; f < uf; ++f) {
        const auto& cells = sc_frame_cells(nci, norm[f]);
        for (size_t i = 0; i < nc; ++i) pk[static_cast<size_t>(cells.PK[i])] += std::norm(ls[f][i]);
    }
    for (size_t k = 0; k < nc; ++k)
        if (n_p_norm[k] > 0) {
            pk[k] /= static_cast<double>(n_p_norm[k]);
            act_pk.push_back(pk[k]);
        }
    std::ranges::sort(act_pk);
    const size_t na = act_pk.size();
    const double med_pk = na % 2 ? act_pk[na / 2] : (act_pk[na / 2 - 1] + act_pk[na / 2]) / 2;
    std::vector<char> hot_pow(nc, 0);
    for (size_t k = 0; k < nc; ++k)
        if (n_p_norm[k] > 0 && pk[k] > SC_HOT_POWER * med_pk) hot_pow[k] = 1;
    double ls2 = 0.0;
    size_t n_cool = 0;
    for (size_t f = 0; f < uf; ++f) {
        const auto& cells = sc_frame_cells(nci, norm[f]);
        for (size_t i = 0; i < nc; ++i)
            if (!hot_pow[static_cast<size_t>(cells.PK[i])]) ls2 += std::norm(ls[f][i]), ++n_cool;
    }
    const double p_y = std::max(ls2 / static_cast<double>(std::max<size_t>(n_cool, 1)) - n0, 1e-12) / (1 + ratio);
    double n0_y = n0 + ratio * p_y;
    const double p_sig = p_y / (g * g);
    DataEstimate est;
    est.n_f = n_f;
    est.nc = nci;
    est.clip_ratio = clip.ratio;
    est.gain = clip.gain_for(n_f);
    est.band = config::band(db).name;
    const int max_comb_d = (intra && intra->K == 2) ? 79 : 52;
    const int width = support.second - support.first;
    const int slack = std::clamp(static_cast<int>(std::floor((max_comb_d - width) / 2.0)), 0, equalizer::SUPPORT_SLACK);
    est.slack = nc >= 24 ? slack : sc_slack(nci);
    // slow or fast by same-carrier pilots per_f frames apart
    size_t pairs = 0, pairs_p = 0;
    cd acc = 0.0;
    for (size_t f = 0; f + 1 < uf; ++f)
        if (norm[f + 1] - norm[f] == 1) ++pairs;
    for (size_t f = 0; f + static_cast<size_t>(per_f) < uf; ++f)
        if (norm[f + static_cast<size_t>(per_f)] - norm[f] == per_f) {
            ++pairs_p;
            for (size_t k = 0; k < nc; ++k)
                acc += std::conj(hp[static_cast<size_t>(norm[f])][k]) *
                       hp[static_cast<size_t>(norm[f + static_cast<size_t>(per_f)])][k];
        }
    double rho = pairs_p ? std::abs(acc) / (static_cast<double>(pairs_p * nc) * p_y) : 0.0;
    double rho_dt = static_cast<double>(per_f) * FRAME_S;
    if (intra && rho <= SC_SLOW_RHO && uf > 0) {
        const size_t w_in = nc / static_cast<size_t>(intra->K);
        cd acc_in = 0.0;
        for (size_t f = 0; f < uf; ++f) {
            const cd* hr = hp[static_cast<size_t>(norm[f])];
            for (size_t j = 0; j < w_in; ++j) acc_in += std::conj(hr[j]) * hr[nc - w_in + j];
        }
        const double rho_in = std::abs(acc_in) / (static_cast<double>(uf * w_in) * p_y);
        if (rho_in > 0.60) {
            rho = rho_in;
            rho_dt = static_cast<double>((intra->syms[static_cast<size_t>(intra->n_syms - 1)] - intra->syms[0]) * NSYM) / FS;
        }
    }
    const bool loo_rule = nc >= 24;
    std::optional<double> spread;
    if (!loo_rule && rho > SC_SLOW_RHO)
        spread = std::clamp(SC_WIDEN * std::sqrt(-2 * std::log(std::min(rho, 0.99999))) / (PI * rho_dt), 0.05, SC_FAST_HZ);
    const auto fl = full(z, hd, raw, hp, kc, shift, db);
    Obs o;
    for (size_t f = 0; f < uf; ++f) {
        const auto& cells = sc_frame_cells(nci, norm[f]);
        for (size_t i = 0; i < nc; ++i) {
            const int sym = norm[f] * SYMS_PER_FRAME + cells.P[i] / nci;
            o.sym.push_back(static_cast<double>(sym));
            o.k.push_back(cells.PK[i]);
            o.kind.push_back(0);
            o.val.push_back(ls[f][i] / g);
        }
    }
    for (size_t r = 0; r < fl.rows.rows; ++r)
        for (size_t k = 0; k < nc; ++k) {
            o.sym.push_back(std::nearbyint(fl.t[r] * FS / NSYM * 100) / 100);
            o.k.push_back(static_cast<int>(k));
            o.kind.push_back(r == 0 ? 2 : 1);
            o.val.push_back(fl.rows[r][k]);
        }
    {
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
    const bool slep = nc >= 24;
    const bool slep_all = slep && intra.has_value() && width <= max_comb_d;

    // detect steady tones on carriers with zero scattered pilots via whole pilot rows
    bool any_u = false;
    for (size_t k = 0; k < nc; ++k) any_u = any_u || (n_p_norm[k] == 0);
    std::vector<char> hot_u(nc, 0);
    std::vector<double> extra_u(nc, 0.0);
    bool any_hot_u = false;
    if (any_u && fl.rows.rows > 1) {
        const size_t n_fr = fl.rows.rows - 1;
        Mat<cd> f_rows(n_fr, nc);
        std::copy(fl.rows[1], fl.rows[1] + n_fr * nc, f_rows.data.begin());
        const auto fs = equalizer::freq_smooth(f_rows, support, equalizer::bb(config::band(db)), est.slack);
        if (nci - fs.r >= 2) {
            std::vector<double> pk_f(nc, 0.0);
            for (size_t k = 0; k < nc; ++k) {
                double s = 0.0;
                for (size_t r = 0; r < n_fr; ++r) s += std::norm(f_rows[r][k] - fs.hs[r][k]);
                pk_f[k] = (s / static_cast<double>(n_fr)) / std::max(fs.keep[k], 1e-3);
            }
            std::vector<double> srt = pk_f;
            std::ranges::sort(srt);
            const double med_f = nc % 2 ? srt[nc / 2] : (srt[nc / 2 - 1] + srt[nc / 2]) / 2;
            const double qf = GAMMA_Q999[std::clamp(n_fr - 1, size_t{0}, GAMMA_Q999.size() - 1)];
            const double lim_f = med_f * static_cast<double>(n_fr) / std::max(static_cast<double>(n_fr) - 1.0 / 3, 0.5) * qf;
            for (size_t k = 0; k < nc; ++k) {
                if (n_p_norm[k] != 0) continue;
                const double ex_f = std::max(pk_f[k] - med_f, 0.0) / (g * g);
                if (pk_f[k] > lim_f && ex_f >= SC_HOT_EXCESS * (n0_y / (g * g))) {
                    hot_u[k] = 1;
                    extra_u[k] = ex_f;
                    any_hot_u = true;
                }
            }
        }
    }
    std::vector<double> kw_init, kw_pick;
    if (any_hot_u) {
        kw_init.resize(nc);
        for (size_t k = 0; k < nc; ++k) kw_init[k] = 1.0 + extra_u[k] / (n0_y / (g * g));
    }
    const bool any_hot_pow = std::ranges::any_of(hot_pow, [](char x) { return x != 0; });
    if (any_hot_pow) {
        kw_pick.resize(nc);
        for (size_t k = 0; k < nc; ++k) {
            const double ex = hot_pow[k] ? std::max(pk[k] - med_pk, 0.0) / (g * g) : extra_u[k];
            kw_pick[k] = 1.0 + ex / (n0_y / (g * g));
        }
    } else {
        kw_pick = kw_init;
    }

    double nr_sc = n0_y / (g * g) / p_sig, nr_full = n0 / p_sig;
    if (!spread) {
        std::vector<double> cand;
        if (loo_rule) cand.assign(SC_SPREADS_SLOW.begin(), SC_SPREADS_SLOW.end());
        for (double v : SC_SPREADS)
            if (loo_rule || v >= (pairs ? SC_FAST_HZ : equalizer::DEFAULT_SPREAD_HZ)) cand.push_back(v);
        std::vector<double> whole;
        for (size_t j = 0; j < o.sym.size(); ++j)
            if (o.kind[j] > 0) whole.push_back(o.sym[j]);
        const auto search = sc_interior(norm, whole, static_cast<int>(n_air));
        const auto n_p_search = sc_pilot_counts(nci, search);
        enum class Mode { Joint, Sep, Slep };
        auto pick = [&](Mode mode, const std::vector<double>& cs) {
            std::vector<Joint> fits;
            std::optional<SlepianPrep> prep;
            if (mode == Mode::Slep) prep = slepian_prep(o, db, support, nr_sc, nr_full, kw_pick);
            for (double v : cs) {
                if (mode == Mode::Slep) fits.push_back(slepian(o, *prep, search, v, p_sig, true));
                else if (mode == Mode::Sep) fits.push_back(separable(o, search, db, support, v, nr_sc, nr_full, p_sig, true, kw_pick));
                else fits.push_back(joint(o, search, db, support, v, nr_sc, nr_full, p_sig, true, kw_pick));
            }
            std::vector<char> skip0(nc, 0);
            for (size_t k = 0; k < nc; ++k) skip0[k] = (n_p_search[k] == 0 || hot_pow[k]) ? 1 : 0;
            std::vector<double> loo;
            for (const auto& f : fits) loo.push_back(sum(f.loo, &skip0));
            size_t i = static_cast<size_t>(std::min_element(loo.begin(), loo.end()) - loo.begin());
            auto skip = sc_hot(fits[i].loo, fits[i].model, n_p_search, n0_y / (g * g)).first;
            bool any_keep = false;
            for (size_t k = 0; k < nc; ++k) {
                if (n_p_search[k] == 0 || hot_pow[k]) skip[k] = 1;
                any_keep = any_keep || !skip[k];
            }
            if (!any_keep)
                for (size_t k = 0; k < nc; ++k) skip[k] = n_p_search[k] == 0 ? 1 : 0;
            for (size_t c = 0; c < fits.size(); ++c) loo[c] = sum(fits[c].loo, &skip);
            i = static_cast<size_t>(std::min_element(loo.begin(), loo.end()) - loo.begin());
            if (!loo_rule) return cs[i];
            const double tol = loo[i] + SC_LOO_TOL * std::max(loo[i] - sum(fits[i].model, &skip), 0.0);
            double best = 0.0;
            for (size_t c = 0; c < cs.size(); ++c)
                if (loo[c] <= tol) best = std::max(best, cs[c]);
            return best;
        };
        if (slep_all) {
            spread = pick(Mode::Slep, cand);
        } else if (SC_SEP_HZ > 0 && loo_rule) {
            spread = pick(Mode::Sep, cand);
            if (*spread > SC_SEP_HZ) {
                std::vector<double> fast;
                for (double v : cand)
                    if (v > SC_SEP_HZ) fast.push_back(v);
                spread = pick(slep ? Mode::Slep : Mode::Joint, fast);
            }
        } else {
            spread = pick(slep ? Mode::Slep : Mode::Joint, cand);
        }
    }
    const bool sep = loo_rule && *spread <= SC_SEP_HZ;
    est.spread_hz = std::min(*spread, sep ? SC_SEP_REPORT_HZ : 4.0);
    est.filter_spread_hz = *spread;
    auto est_fn = [&](std::span<const double> w) {
        if (slep_all || (slep && !sep)) {
            const auto prep = slepian_prep(o, db, support, nr_sc, nr_full, w);
            return slepian(o, prep, norm, *spread, p_sig, false);
        }
        return sep ? separable(o, norm, db, support, *spread, nr_sc, nr_full, p_sig, false, w)
                   : joint(o, norm, db, support, *spread, nr_sc, nr_full, p_sig, false, w);
    };
    auto J = est_fn(kw_init);
    auto [hot, extra] = sc_hot(J.loo, J.model, n_p_norm, n0_y / (g * g));
    if (any_hot_u) {
        for (size_t k = 0; k < nc; ++k)
            if (hot_u[k]) hot[k] = 1, extra[k] = extra_u[k];
    }
    bool any_hot_act = false;
    for (size_t k = 0; k < nc; ++k) any_hot_act = any_hot_act || (hot[k] && n_p_norm[k] > 0);
    const bool any_hot = any_hot_act || any_hot_u;
    std::vector<double> kw = kw_init;
    auto weigh = [&] {
        kw.resize(nc);
        for (size_t k = 0; k < nc; ++k) kw[k] = 1 + extra[k] / (n0_y / (g * g));
    };
    for (int pass = 0; pass < (any_hot_act ? SC_HOT_PASSES : 0); ++pass) {
        weigh();
        J = est_fn(kw);
        for (size_t k = 0; k < nc; ++k)
            if (hot[k] && n_p_norm[k] > 0)
                extra[k] = std::max(extra[k] + (J.loo[k] - J.model[k]) / static_cast<double>(n_p_norm[k]), 0.0);
    }
    double d = 0.0;
    size_t n_rest = 0;
    for (size_t k = 0; k < nc; ++k)
        if (!hot[k] && n_p_norm[k] > 0) d += J.loo[k] - J.model[k], n_rest += static_cast<size_t>(n_p_norm[k]);
    if (!n_rest) {
        d = 0.0;
        for (size_t k = 0; k < nc; ++k)
            if (n_p_norm[k] > 0) d += J.loo[k] - J.model[k], n_rest += static_cast<size_t>(n_p_norm[k]);
    }
    const double excess = std::max(d, 0.0) / static_cast<double>(std::max<size_t>(n_rest, 1));
    if (excess > 0.25 * n0_y / (g * g)) {
        n0 += excess * g * g;
        n0_y += excess * g * g;
        nr_sc = n0_y / (g * g) / p_sig;
        nr_full = n0 / p_sig;
        for (double& e : extra) e = std::max(e - excess, 0.0);
        if (!kw.empty()) weigh();
        J = est_fn(kw);
    }
    est.n0 = est.n0_snr = n0;
    est.n0_k.assign(nc, n0);
    if (any_hot) {
        est.n0_k.resize(uf * DATA_SYMS_PER_FRAME * nc);
        for (size_t f = 0; f < uf; ++f) {
            const auto& cells = sc_frame_cells(nci, norm[f]);
            for (size_t j = 0; j < DATA_SYMS_PER_FRAME * nc; ++j)
                est.n0_k[f * DATA_SYMS_PER_FRAME * nc + j] = n0 + extra[static_cast<size_t>(cells.Dc[j]) % nc] * g * g;
        }
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
        const auto& cells = sc_frame_cells(nci, norm[f]);
        if (intra) {
            for (size_t i = 0; i < nc; ++i)
                out.raw[f * SYMS_PER_FRAME][i] = hp[static_cast<size_t>(norm[f])][i] * b.pilot[static_cast<size_t>(cells.PK[i])];
        } else {
            for (size_t k = 0; k < nc; ++k) out.raw[f * SYMS_PER_FRAME][k] = hp[static_cast<size_t>(norm[f])][k] * b.pilot[k];
        }
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
