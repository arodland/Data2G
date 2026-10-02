#include "equalizer/equalizer.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>

#include "tables/tables.hpp"
#include "util/linalg.hpp"

namespace data2g::equalizer {

using config::FS;
using config::NCP;
using std::size_t;

namespace {

constexpr double PI = std::numbers::pi;

// exp(-2j*pi*x/FS) the way numpy evaluates `np.exp(-2j * np.pi * x / FS)`:
// the complex division by FS is a multiplication by 1/FS.
cd phasor(double x) { return std::polar(1.0, (-2.0 * PI * x) * (1.0 / FS)); }

double sq_abs(const cd& x) {
    const double a = std::abs(x);  // np.abs(x) ** 2
    return a * a;
}

// h @ conj(U) @ U.T, row by row.
Mat<cd> project(const Mat<cd>& h, const Basis& b) {
    const size_t nc = h.cols, r = static_cast<size_t>(b.r);
    Mat<cd> out(h.rows, nc);
    std::vector<cd> t(r);
    for (size_t p = 0; p < h.rows; ++p) {
        std::fill(t.begin(), t.end(), cd(0.0));
        for (size_t k = 0; k < nc; ++k)
            for (size_t j = 0; j < r; ++j) t[j] += h[p][k] * std::conj(b.u[k][j]);
        for (size_t c = 0; c < nc; ++c) {
            cd s = 0.0;
            for (size_t j = 0; j < r; ++j) s += t[j] * b.u[c][j];
            out[p][c] = s;
        }
    }
    return out;
}

// B over the support plus slack (nc x nd): B[k, j] = phasor(bb_k * d_j).
std::vector<cd> steering(std::span<const double> bb, int d0, int d1) {
    std::vector<cd> B;
    for (size_t k = 0; k < bb.size(); ++k)
        for (int d = d0 - SUPPORT_SLACK; d <= d1 + SUPPORT_SLACK; ++d) B.push_back(phasor(bb[k] * d));
    return B;
}

// B B^H
std::vector<cd> gram(std::span<const double> bb, int d0, int d1) {
    const size_t nc = bb.size();
    const auto B = steering(bb, d0, d1);
    const size_t nd = B.size() / nc;
    std::vector<cd> g(nc * nc);
    for (size_t k = 0; k < nc; ++k)
        for (size_t c = k; c < nc; ++c) {
            cd s = 0.0;
            for (size_t j = 0; j < nd; ++j) s += B[k * nd + j] * std::conj(B[c * nd + j]);
            g[k * nc + c] = s;
            g[c * nc + k] = std::conj(s);
        }
    return g;
}

double mean_sq_diff(const Mat<cd>& a, const Mat<cd>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) s += sq_abs(a.data[i] - b.data[i]);
    return s / static_cast<double>(a.data.size());
}

}  // namespace

std::vector<double> bb(const config::Band& band) {
    std::vector<double> out(static_cast<size_t>(band.nc));
    for (int k = 0; k < band.nc; ++k)
        out[static_cast<size_t>(k)] = config::CARRIER0 + config::RS * (band.k0 + k) - config::FCENTER;
    return out;
}

double residual_cfo(const Mat<cd>& h) {
    cd d = 0.0;
    for (size_t p = 1; p < h.rows; ++p)
        for (size_t c = 0; c < h.cols; ++c) d += h[p][c] * std::conj(h[p - 1][c]);
    return std::abs(d) > 0 ? std::arg(d) / (2 * PI * FRAME_S) : 0.0;
}

std::vector<double> delay_profile(const Mat<cd>& h, std::span<const double> bb) {
    const size_t nc = bb.size(), nd = DELAY_MAX - DELAY_MIN + 1;
    if (h.cols != nc) throw std::invalid_argument("delay_profile: h_pilot and bb disagree on carriers");
    // np.hanning(nc + 2)[1:-1]
    const double m = static_cast<double>(nc + 2);
    std::vector<double> w(nc);
    for (size_t k = 0; k < nc; ++k) w[k] = 0.5 + 0.5 * std::cos(PI * (1.0 - m + 2.0 * static_cast<double>(k + 1)) / (m - 1));
    Mat<cd> steer(nc, nd);  // conj(steer)
    for (size_t k = 0; k < nc; ++k)
        for (size_t d = 0; d < nd; ++d) steer[k][d] = std::conj(phasor(bb[k] * (DELAY_MIN + static_cast<int>(d))));
    std::vector<double> p(nd, 0.0);
    std::vector<cd> g(nd), hw(nc);
    for (size_t f = 0; f < h.rows; ++f) {
        for (size_t k = 0; k < nc; ++k) hw[k] = h[f][k] * w[k];
        std::fill(g.begin(), g.end(), cd(0.0));
        for (size_t k = 0; k < nc; ++k)
            for (size_t d = 0; d < nd; ++d) g[d] += hw[k] * steer[k][d];
        for (size_t d = 0; d < nd; ++d) p[d] += sq_abs(g[d]);
    }
    for (double& x : p) x /= static_cast<double>(h.rows);
    return p;
}

Support delay_support(const Mat<cd>& h, std::span<const double> bb, double floor_db) {
    const auto p = delay_profile(h, bb);
    const double thr = *std::max_element(p.begin(), p.end()) * std::pow(10.0, floor_db / 10);
    int first = -1, last = -1;
    for (size_t i = 1; i + 1 < p.size(); ++i)
        if (p[i] >= thr && p[i] >= p[i - 1] && p[i] >= p[i + 1]) {
            if (first < 0) first = static_cast<int>(i);
            last = static_cast<int>(i);
        }
    if (first < 0) first = last = static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
    return {DELAY_MIN + first, DELAY_MIN + last};
}

int window_shift(Support s) {
    // Python's round(): half to even, as nearbyint in the default mode.
    return static_cast<int>(std::nearbyint((s.first + s.second - NCP) / 2.0));
}

std::shared_ptr<const Basis> support_basis(std::span<const double> bb, int d0, int d1) {
    using Key = std::tuple<std::vector<double>, int, int>;
    static std::mutex mutex;
    static std::map<Key, std::shared_ptr<const Basis>> cache;
    Key key{std::vector<double>(bb.begin(), bb.end()), d0, d1};
    {
        std::lock_guard lock(mutex);
        if (auto it = cache.find(key); it != cache.end()) return it->second;
    }
    const int nc = static_cast<int>(bb.size());
    if (nc < 3 || d1 < d0 - 2 * SUPPORT_SLACK) throw std::invalid_argument("support_basis: bad band or support");
    // The SVD from an eigen-solve of the smaller Gram matrix: B B^H's
    // eigenvectors are U, B^H B's are V (then U = B V / s); s = sqrt(lambda).
    const auto B = steering(bb, d0, d1);
    const size_t NC = static_cast<size_t>(nc), nd = B.size() / NC;
    const bool left = NC <= nd;
    const size_t n = left ? NC : nd;
    std::vector<cd> g(n * n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i; j < n; ++j) {
            cd s = 0.0;
            if (left)
                for (size_t m = 0; m < nd; ++m) s += B[i * nd + m] * std::conj(B[j * nd + m]);
            else
                for (size_t m = 0; m < NC; ++m) s += std::conj(B[m * nd + i]) * B[m * nd + j];
            g[i * n + j] = s;
            g[j * n + i] = std::conj(s);
        }
    std::vector<cd> v;
    const auto lam = linalg::hermitian_eigen(g, v, static_cast<int>(n));
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), size_t{0});
    std::ranges::stable_sort(order, [&](size_t a, size_t b) { return lam[a] > lam[b]; });
    auto sv = [&](size_t i) { return std::sqrt(std::max(lam[order[i]], 0.0)); };
    auto b = std::make_shared<Basis>();
    for (size_t i = 0; i < n; ++i) b->r_full += sv(i) > sv(0) * 1e-2;
    b->r = std::min(b->r_full, nc - 2);
    const size_t r = static_cast<size_t>(b->r);
    b->u = Mat<cd>(NC, r);
    for (size_t j = 0; j < r; ++j)
        for (size_t k = 0; k < NC; ++k) {
            if (left) {
                b->u[k][j] = v[k * n + order[j]];
            } else {
                cd s = 0.0;
                for (size_t m = 0; m < nd; ++m) s += B[k * nd + m] * v[m * n + order[j]];
                b->u[k][j] = s / sv(j);
            }
        }
    b->keep.assign(NC, 1.0);
    for (size_t k = 0; k < NC; ++k) {
        double lev = 0.0;
        for (size_t j = 0; j < r; ++j) lev += std::norm(b->u[k][j]);
        b->keep[k] = 1 - lev;
    }
    std::lock_guard lock(mutex);
    // ponytail: wholesale clear at Python's lru_cache size; supports per band are few
    if (cache.size() >= 1024) cache.clear();
    return cache.emplace(std::move(key), std::move(b)).first->second;
}

Smoothed freq_smooth(const Mat<cd>& h, Support support, std::span<const double> bb) {
    const auto basis = support_basis(bb, support.first, support.second);
    const int nc = static_cast<int>(bb.size());
    if (h.cols != bb.size()) throw std::invalid_argument("freq_smooth: h_pilot and bb disagree on carriers");
    Smoothed out{project(h, *basis), INF, basis->r, basis->keep};
    if (nc - basis->r_full >= 2) out.n0 = mean_sq_diff(h, out.hs) * nc / (nc - basis->r);
    return out;
}

std::vector<double> preamble_noise_k(const Mat<cd>& h) {
    std::vector<double> out(h.cols, 0.0);
    for (size_t p = 1; p < h.rows; ++p)
        for (size_t c = 0; c < h.cols; ++c) out[c] += sq_abs(h[p][c] - h[p - 1][c]);
    for (double& x : out) x = x / static_cast<double>(h.rows - 1) / 2;
    return out;
}

double preamble_noise(const Mat<cd>& h) {
    const auto k = preamble_noise_k(h);
    return std::accumulate(k.begin(), k.end(), 0.0) / static_cast<double>(k.size());
}

std::vector<double> per_carrier_noise(std::span<const double> power, int samples) {
    if (samples < 1 || samples > tables::GAMMA_Q99_MAX)
        throw std::out_of_range("per_carrier_noise: samples " + std::to_string(samples) + " outside the gamma table");
    const double q = tables::GAMMA_Q99[static_cast<size_t>(samples - 1)];
    std::vector<double> s(power.begin(), power.end());
    std::ranges::sort(s);
    const size_t n = s.size();
    const double med = n % 2 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
    const double lim = med * samples / (samples - 1.0 / 3) * q;
    std::vector<char> hot(n);
    for (size_t i = 0; i < n; ++i) hot[i] = power[i] > lim;
    if (std::ranges::all_of(hot, [](char x) { return x != 0; })) std::ranges::fill(hot, 0);
    double sum = 0.0;
    int cnt = 0;
    for (size_t i = 0; i < n; ++i)
        if (!hot[i]) sum += power[i], ++cnt;
    const double band = sum / cnt;
    std::vector<double> out(n, band);
    for (size_t i = 0; i < n; ++i)
        if (hot[i]) out[i] = std::max((samples * power[i] + NOISE_SHAPE_PRIOR * band) / (samples + NOISE_SHAPE_PRIOR), band);
    return out;
}

double doppler_corr(double dt, double spread_hz) {
    const double x = PI * (spread_hz / 2) * dt;
    return std::exp(-2 * (x * x));
}

double measure_spread(const Mat<cd>& hs, double n0_s) {
    if (hs.rows < 8) return DEFAULT_SPREAD_HZ;
    double pw = 0.0;
    for (const cd& x : hs.data) pw += sq_abs(x);
    const double p = pw / static_cast<double>(hs.data.size()) - n0_s;
    if (p <= 0) return DEFAULT_SPREAD_HZ;
    cd c = 0.0;
    for (size_t f = 1; f < hs.rows; ++f)
        for (size_t k = 0; k < hs.cols; ++k) c += hs[f][k] * std::conj(hs[f - 1][k]);
    const double rho = std::clamp(std::abs(c / static_cast<double>((hs.rows - 1) * hs.cols)) / p, 1e-3, 0.9999);
    const double sigma = std::sqrt(-std::log(rho) / 2) / (PI * FRAME_S);
    return std::clamp(2 * sigma, 0.02, 4.0);
}

Estimate estimate(const Mat<cd>& h_pilot, Support support, std::span<const double> bb, double n0_pre,
                  std::span<const double> n0_pre_k) {
    constexpr int S = config::SYMS_PER_FRAME - 1;
    const int n_p = static_cast<int>(h_pilot.rows), n_f = n_p - 1, nc = static_cast<int>(bb.size());
    if (n_p < 2) throw std::invalid_argument("estimate: need at least two pilots");
    auto sm = freq_smooth(h_pilot, support, bb);
    const Mat<cd>& hs = sm.hs;
    Estimate e;
    e.n_f = n_f;
    e.nc = nc;
    double n0 = sm.n0;
    if (std::isfinite(n0)) {
        std::vector<double> pk(static_cast<size_t>(nc), 0.0);
        for (int p = 0; p < n_p; ++p)
            for (int c = 0; c < nc; ++c) pk[static_cast<size_t>(c)] += sq_abs(h_pilot[static_cast<size_t>(p)][c] - hs[static_cast<size_t>(p)][c]);
        for (int c = 0; c < nc; ++c)
            pk[static_cast<size_t>(c)] = pk[static_cast<size_t>(c)] / n_p / std::max(sm.keep[static_cast<size_t>(c)], 1e-3);
        e.n0_k = per_carrier_noise(pk, n_p);
    } else if (static_cast<int>(n0_pre_k.size()) == nc) {
        e.n0_k = per_carrier_noise(n0_pre_k, 7);
    }
    if (!std::isfinite(n0)) n0 = n0_pre;
    if (!std::isfinite(n0)) throw std::invalid_argument("no noise estimate: pass n0_pre on a band this narrow");
    const double n0_s = n0 * sm.r / nc;
    const double spread = measure_spread(hs, n0_s);
    double pw = 0.0;
    for (const cd& x : hs.data) pw += sq_abs(x);
    const double p_sig = std::max(pw / static_cast<double>(hs.data.size()) - n0_s, 1e-12);

    e.h = Mat<cd>(static_cast<size_t>(n_f * S), static_cast<size_t>(nc));
    e.mse = Mat<double>(static_cast<size_t>(n_f * S), static_cast<size_t>(nc));
    std::vector<double> a, x, rdp;
    for (int f = 0; f < n_f; ++f) {
        const int lo = std::max(0, std::min(f - TIME_TAPS + 1, n_p - 2 * TIME_TAPS));
        const int J = std::min(n_p, lo + 2 * TIME_TAPS) - lo;
        a.assign(static_cast<size_t>(J * J), 0.0);
        for (int i = 0; i < J; ++i)
            for (int j = 0; j < J; ++j)
                a[static_cast<size_t>(i * J + j)] =
                    p_sig * doppler_corr((lo + i) * FRAME_S - (lo + j) * FRAME_S, spread) + (i == j ? n0_s : 0.0);
        rdp.assign(static_cast<size_t>(S * J), 0.0);
        x.assign(static_cast<size_t>(J * S), 0.0);
        for (int s = 0; s < S; ++s) {
            const double t = f * FRAME_S + (s + 1) / static_cast<double>(config::SYMS_PER_FRAME) * FRAME_S;
            for (int j = 0; j < J; ++j) {
                const double v = p_sig * doppler_corr(t - (lo + j) * FRAME_S, spread);
                rdp[static_cast<size_t>(s * J + j)] = v;
                x[static_cast<size_t>(j * S + s)] = v;
            }
        }
        linalg::lu_solve(a.data(), J, x.data(), S);  // x[j, s] = W[s, j]
        for (int s = 0; s < S; ++s) {
            double m = 0.0;
            for (int j = 0; j < J; ++j) m += x[static_cast<size_t>(j * S + s)] * rdp[static_cast<size_t>(s * J + j)];
            auto* hrow = e.h[static_cast<size_t>(f * S + s)];
            for (int j = 0; j < J; ++j) {
                const double wsj = x[static_cast<size_t>(j * S + s)];
                const cd* hp = hs[static_cast<size_t>(lo + j)];
                for (int c = 0; c < nc; ++c) hrow[c] += wsj * hp[c];
            }
            std::fill_n(e.mse[static_cast<size_t>(f * S + s)], nc, std::max(p_sig - m, 0.0));
        }
    }
    if (e.n0_k.empty() || !PER_CARRIER_NOISE) e.n0_k.assign(static_cast<size_t>(nc), n0);
    e.n0 = n0;
    e.p_sig = p_sig;
    e.spread_hz = spread;
    return e;
}

Mat<cd> time_shift_phase(std::span<const double> shift, std::span<const double> bb) {
    Mat<cd> out(shift.size(), bb.size());
    for (size_t i = 0; i < shift.size(); ++i)
        for (size_t k = 0; k < bb.size(); ++k) out[i][k] = phasor(shift[i] * bb[k]);
    return out;
}

std::pair<Mat<cd>, Mat<double>> refine(const Mat<cd>& h_pilot, std::span<const double> t_pilot, const Mat<cd>& z,
                                       const Mat<double>& w, const Mat<double>& t_rows, Support support,
                                       double p_sig, double spread, double n0, std::span<const double> bb) {
    const size_t nc = bb.size(), F = t_rows.rows, S = t_rows.cols, P = h_pilot.rows;
    if (h_pilot.cols != nc || z.cols != nc || w.cols != nc || z.rows != F * S || w.rows != F * S || t_pilot.size() != P)
        throw std::invalid_argument("refine: shapes disagree");
    const auto basis = support_basis(bb, support.first, support.second);
    const Mat<cd> hs_p = project(h_pilot, *basis);
    std::vector<cd> rf = gram(bb, support.first, support.second);
    const double scale = p_sig / (support.second - support.first + 2 * SUPPORT_SLACK + 1);
    for (cd& x : rf) x = scale * x;
    auto Rf = [&](size_t r, size_t c) { return rf[r * nc + c]; };

    // Observations: the smoothed pilots, then each data row with anything
    // known, LMMSE-smoothed across carriers from its known ones.
    std::vector<double> times(t_pilot.begin(), t_pilot.end());
    Mat<cd> vals(P, nc);
    Mat<double> var(P, nc, n0 * basis->r / static_cast<double>(nc));
    vals.data = hs_p.data;
    std::vector<size_t> known;
    std::vector<cd> a, x;
    for (size_t i = 0; i < F * S; ++i) {
        known.clear();
        for (size_t c = 0; c < nc; ++c)
            if (w[i][c] > 0) known.push_back(c);
        if (known.empty()) continue;
        const size_t K = known.size(), nx = nc + 1;
        // S = Rf[k, k] + diag(1 / w), x = [Rf[k, :] | z[k]]
        a.assign(K * K, cd(0.0));
        x.assign(K * nx, cd(0.0));
        for (size_t u = 0; u < K; ++u) {
            for (size_t v = 0; v < K; ++v) a[u * K + v] = Rf(known[u], known[v]);
            a[u * K + u] += 1 / w[i][known[u]];
            for (size_t c = 0; c < nc; ++c) x[u * nx + c] = Rf(known[u], c);
            x[u * nx + nc] = z[i][known[u]];
        }
        times.push_back(t_rows.data[i]);
        std::vector<cd> vrow(nc);
        std::vector<double> varrow(nc);
        // Python's G = solve(S, Rf[k]).conj().T. S is Hermitian positive
        // definite, so with S = L L^H and Y = L^-1 x: vals = Y[:, :nc]^H Y[:, nc]
        // and G Rf[k, c] = |Y[:, c]|^2, forward solves only (2.5x faster).
        if (linalg::cholesky(a.data(), static_cast<int>(K))) {
            linalg::forward_solve(a.data(), static_cast<int>(K), x.data(), static_cast<int>(nx));
            for (size_t c = 0; c < nc; ++c) {
                cd hv = 0.0;
                double m = 0.0;
                for (size_t u = 0; u < K; ++u) {
                    const cd y = x[u * nx + c], yz = x[u * nx + nc];
                    hv += cd(y.real() * yz.real() + y.imag() * yz.imag(), y.real() * yz.imag() - y.imag() * yz.real());
                    m += y.real() * y.real() + y.imag() * y.imag();
                }
                vrow[c] = hv;
                varrow[c] = std::max(p_sig - m, 1e-9 * p_sig);
            }
        } else {  // not numerically positive definite: LU, as numpy
            for (size_t u = 0; u < K; ++u) {
                for (size_t v = 0; v < K; ++v) a[u * K + v] = Rf(known[u], known[v]);
                a[u * K + u] += 1 / w[i][known[u]];
            }
            linalg::lu_solve(a.data(), static_cast<int>(K), x.data(), static_cast<int>(nx));  // G[c, u] = conj(x[u, c])
            for (size_t c = 0; c < nc; ++c) {
                cd hv = 0.0, m = 0.0;
                for (size_t u = 0; u < K; ++u) {
                    const cd g = std::conj(x[u * nx + c]);
                    hv += g * z[i][known[u]];
                    m += g * std::conj(Rf(c, known[u]));
                }
                vrow[c] = hv;
                varrow[c] = std::max(p_sig - m.real(), 1e-9 * p_sig);
            }
        }
        vals.data.insert(vals.data.end(), vrow.begin(), vrow.end());
        var.data.insert(var.data.end(), varrow.begin(), varrow.end());
    }
    const size_t O = times.size();

    // Time: per frame, every observation within DD_TAPS frames of its middle.
    Mat<cd> h(F * S, nc);
    Mat<double> mse(F * S, nc);
    std::vector<size_t> obs;
    std::vector<double> rt, rdp, ar, xr;
    for (size_t f = 0; f < F; ++f) {
        // t_rows.mean(axis=1) bit for bit (a sequential sum: measured), since
        // an observation exactly DD_TAPS frames away sits on the boundary
        double mid = 0.0;
        for (size_t s = 0; s < S; ++s) mid += t_rows[f][s];
        mid /= static_cast<double>(S);
        obs.clear();
        for (size_t o = 0; o < O; ++o)
            if (std::abs(times[o] - mid) <= DD_TAPS * FRAME_S) obs.push_back(o);
        const size_t J = obs.size();
        rt.assign(J * J, 0.0);
        for (size_t u = 0; u < J; ++u)
            for (size_t v = 0; v < J; ++v) rt[u * J + v] = p_sig * doppler_corr(times[obs[u]] - times[obs[v]], spread);
        rdp.assign(S * J, 0.0);
        for (size_t s = 0; s < S; ++s)
            for (size_t u = 0; u < J; ++u) rdp[s * J + u] = p_sig * doppler_corr(t_rows[f][s] - times[obs[u]], spread);
        for (size_t c = 0; c < nc; ++c) {
            ar = rt;
            for (size_t u = 0; u < J; ++u) ar[u * J + u] += var[obs[u]][c];
            xr.resize(J * S);
            for (size_t u = 0; u < J; ++u)
                for (size_t s = 0; s < S; ++s) xr[u * S + s] = rdp[s * J + u];
            if (J) linalg::lu_solve(ar.data(), static_cast<int>(J), xr.data(), static_cast<int>(S));
            for (size_t s = 0; s < S; ++s) {
                cd hv = 0.0;
                double m = 0.0;
                for (size_t u = 0; u < J; ++u) {
                    hv += xr[u * S + s] * vals[obs[u]][c];
                    m += xr[u * S + s] * rdp[s * J + u];
                }
                h[f * S + s][c] = hv;
                mse[f * S + s][c] = std::max(p_sig - m, 0.0);
            }
        }
    }
    return {std::move(h), std::move(mse)};
}

}  // namespace data2g::equalizer
