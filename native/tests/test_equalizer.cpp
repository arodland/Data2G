// Python-free checks of the equalizer port. Parity with data2g/equalizer.py
// is tests/test_native_equalizer.py's job.

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <random>
#include <vector>

#include "check.hpp"
#include "equalizer/equalizer.hpp"
#include "util/linalg.hpp"

using namespace data2g;
using cd = std::complex<double>;

namespace {

// Braces, not cd(g(rng), g(rng)): the order a call's arguments are evaluated
// in is unspecified (GCC goes right to left on x86-64, left to right on
// aarch64), so the two platforms drew different noise. A braced list is
// evaluated left to right everywhere.
//
// A static two-path channel's frequency response on the band's carriers,
// plus complex Gaussian noise of variance n0, at every pilot.
Mat<cd> two_path(const std::vector<double>& bb, int n_p, int d_a, int d_b, double n0, std::mt19937_64& rng) {
    std::normal_distribution<double> g(0.0, std::sqrt(n0 / 2));
    Mat<cd> h(static_cast<std::size_t>(n_p), bb.size());
    for (int p = 0; p < n_p; ++p)
        for (std::size_t k = 0; k < bb.size(); ++k) {
            const double w = -2 * std::numbers::pi * bb[k] / config::FS;
            h[static_cast<std::size_t>(p)][k] =
                std::polar(1.0, w * d_a) + 0.7 * std::polar(1.0, w * d_b + 1.0) + cd{g(rng), g(rng)};
        }
    return h;
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog dog(120, "test_equalizer");
    std::mt19937_64 rng(1);

    // LU against a known product, real and complex.
    {
        std::vector<double> a = {4, 1, 2, 1, 3, 0, 2, 0, 5}, x = {1, -2, 3};
        std::vector<double> b = {4 - 2 + 6, 1 - 6, 2 + 15};
        linalg::lu_solve(a.data(), 3, b.data(), 1);
        check::close(b, x, 1e-12, "real lu_solve");
        std::vector<cd> ac = {{0, 1}, {2, 0}, {1, -1}, {0, 0}}, bc = {{0, 1}, {1, -1}};  // x = (1, 0)
        linalg::lu_solve(ac.data(), 2, bc.data(), 1);
        check::close(bc, std::vector<cd>{{1, 0}, {0, 0}}, 1e-12, "complex lu_solve with pivoting");
    }

    // Hermitian Jacobi: A v = lambda v for every pair.
    {
        const int n = 6;
        std::normal_distribution<double> g;
        std::vector<cd> m(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = i; j < n; ++j) {
                m[i * n + j] = i == j ? cd(g(rng)) : cd{g(rng), g(rng)};
                m[j * n + i] = std::conj(m[i * n + j]);
            }
        auto a = m;
        std::vector<cd> v;
        const auto lam = linalg::hermitian_eigen(a, v, n);
        double worst = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                cd s = 0;
                for (int k = 0; k < n; ++k) s += m[i * n + k] * v[k * n + j];
                worst = std::max(worst, std::abs(s - lam[j] * v[i * n + j]));
            }
        check::is_true(worst < 1e-12, "hermitian_eigen residual");
    }

    const auto w = equalizer::bb(config::BANDS[0]);
    check::equal(w.front(), -550.0, "wide bb[0]");
    check::equal(w.back(), 600.0, "wide bb[-1]");

    // Two paths 24 samples apart are found, the window centred on them, and
    // the smoothed estimate is closer to the truth than the raw pilots.
    {
        const auto clean = two_path(w, 33, 3, 27, 0.0, rng);
        auto noisy = clean;
        std::normal_distribution<double> g(0.0, std::sqrt(0.01 / 2));
        for (auto& x : noisy.data) x += cd{g(rng), g(rng)};
        const auto sup = equalizer::delay_support(noisy, w);
        check::is_true(std::abs(sup.first - 3) <= 1 && std::abs(sup.second - 27) <= 1, "two-path support");
        check::equal(equalizer::window_shift({0, 33}), 0, "window_shift rounds half to even (0.5 -> 0)");
        check::equal(equalizer::window_shift({0, 35}), 2, "window_shift rounds half to even (1.5 -> 2)");
        const auto est = equalizer::estimate(noisy, sup, w);
        check::is_true(std::abs(est.n0 - 0.01) < 0.003, "noise from the pilot residual");
        check::is_true(est.p_sig > 1.3 && est.p_sig < 1.7, "signal power 1 + 0.49");
        // Noise alone moves a static channel's estimate: over 400 seeds of
        // this channel the median is 0.03 Hz, the max 0.11, and this draw
        // gives 0.12. 0.2 still tells static from fading (DEFAULT_SPREAD_HZ
        // is 2). C++-only: parity is test_native_equalizer.py's.
        check::is_true(est.spread_hz <= 0.2, "static channel: small spread");
        double err = 0;
        for (std::size_t r = 0; r < est.h.rows; ++r)
            for (std::size_t k = 0; k < w.size(); ++k) err += std::norm(est.h[r][k] - clean[0][k]);
        err /= static_cast<double>(est.h.data.size());
        check::is_true(err < 0.01 / 4, "interpolated estimate beats the raw pilots");

        // refine with every data cell known exactly (z = the channel, high
        // weight) lands on the channel.
        const std::size_t F = 32, S = 5;
        Mat<cd> z(F * S, w.size());
        Mat<double> wt(F * S, w.size(), 1e3), tr(F, S);
        std::vector<double> tp(33);
        for (std::size_t p = 0; p < 33; ++p) tp[p] = static_cast<double>(p) * equalizer::FRAME_S;
        for (std::size_t f = 0; f < F; ++f)
            for (std::size_t s = 0; s < S; ++s) {
                tr[f][s] = (static_cast<double>(f) * 6 + static_cast<double>(s) + 1) * config::NSYM / config::FS;
                for (std::size_t k = 0; k < w.size(); ++k) z[f * S + s][k] = clean[0][k];
            }
        const auto [h, mse] = equalizer::refine(noisy, tp, z, wt, tr, sup, est.p_sig, est.spread_hz, est.n0, w);
        double e2 = 0;
        for (std::size_t r = 0; r < h.rows; ++r)
            for (std::size_t k = 0; k < w.size(); ++k) e2 += std::norm(h[r][k] - clean[0][k]);
        check::is_true(e2 / static_cast<double>(h.data.size()) < 1e-4, "refine on known cells");
    }

    // Narrow band: the residual has no room, so the preamble's noise is used.
    {
        const auto n4 = equalizer::bb(config::BANDS[2]);
        const auto h = two_path(n4, 9, 0, 10, 0.01, rng);
        const auto sm = equalizer::freq_smooth(h, {0, 10}, n4);
        check::is_true(std::isinf(sm.n0) && sm.r == 2, "n4: rank capped at nc - 2, no residual noise");
        bool threw = false;
        try {
            equalizer::estimate(h, {0, 10}, n4);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check::is_true(threw, "n4 without n0_pre throws");
        const std::vector<double> pre = {0.01, 0.01, 0.01, 0.01};
        check::equal(equalizer::estimate(h, {0, 10}, n4, 0.02, pre).n0, 0.02, "n4 takes n0_pre");
    }

    // per_carrier_noise: one hot carrier gets its own (shrunk) level.
    {
        std::vector<double> p(24, 1.0);
        p[5] = 10.0;
        const auto n = equalizer::per_carrier_noise(p, 33);
        check::equal(n[0], 1.0, "clean carrier keeps the band level");
        check::close(std::vector<double>{n[5]}, std::vector<double>{(33 * 10.0 + 8) / 41}, 1e-12, "hot carrier");
        bool threw = false;
        try {
            equalizer::per_carrier_noise(p, 0);
        } catch (const std::out_of_range&) {
            threw = true;
        }
        check::is_true(threw, "samples outside the gamma table throws");
    }
    return check::report("test_equalizer");
}
