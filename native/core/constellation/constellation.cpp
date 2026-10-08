#include "constellation/constellation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

#include "dsp/dsp.hpp"

namespace data2g::constellation {

namespace {

// numpy's pairwise sum, so per-axis LLRs sum in the reference's order.
double np_sum(const double* a, int n) { return dsp::pairwise_sum(std::span<const double>(a, static_cast<std::size_t>(n))); }

// numpy's complex multiply on FMA hardware (its SIMD loop; measured on
// numpy 2.5, AVX-512): both parts rounded through one fma. Matching it makes
// products bitwise equal to the reference there. a conj(b) is mul(a, conj(b)).
cd mul(cd a, cd b) {
    return {std::fma(a.real(), b.real(), -(a.imag() * b.imag())), std::fma(a.real(), b.imag(), a.imag() * b.real())};
}

bool gray_square(const Constellation& c) { return c.name.starts_with("gray-qam"); }

// Gray QPSK: each bit rides one axis, so the exact LLR is linear.
void llr_qpsk(std::span<const cd> y, std::span<const cd> h, std::span<const double> var, double* out) {
    const double s0 = -2 * std::sqrt(2.0);
    for (std::size_t i = 0; i < y.size(); ++i) {
        const cd p = mul(y[i], std::conj(h[i]));
        const double s = s0 / var[i];
        out[2 * i] = p.real() * s;
        out[2 * i + 1] = p.imag() * s;
    }
}

// Gray square QAM, one axis at a time: the first m/2 bits ride I, the rest
// Q, so each bit's LLR needs only its axis's 2^(m/2) levels.
void llr_square(std::span<const cd> y, std::span<const cd> h, std::span<const double> var, const Constellation& c,
                double* out) {
    const int k = c.m / 2, n = 1 << k;
    constexpr double ninf = -std::numeric_limits<double>::infinity();
    std::array<double, 16> amp{}, d{}, e{};
    for (int l = 0; l < n; ++l) amp[l] = c.points[std::size_t(l) << k].real();
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double ah = std::hypot(h[i].real(), h[i].imag()), g2 = ah * ah, w = g2 / var[i];  // numpy's abs
        double zr = 0, zi = 0;
        if (g2 > 0) {  // y conj(h) / g2, as numpy divides complex by complex (Smith)
            const cd p = mul(y[i], std::conj(h[i]));
            const double pr = p.real(), pi = p.imag();
            const double rat = 0.0 / g2, scl = 1.0 / (g2 + 0.0 * rat);
            zr = (pr + pi * rat) * scl;
            zi = (pi - pr * rat) * scl;
        }
        double* o = out + std::size_t(i) * c.m;
        for (const double t : {zr, zi}) {
            for (int l = 0; l < n; ++l) d[l] = -w * ((t - amp[l]) * (t - amp[l]));
            for (int j = 0; j < k; ++j) {
                double lse[2];
                for (int b = 0; b < 2; ++b) {
                    double mx = ninf;
                    for (int l = 0; l < n; ++l)
                        if ((l >> (k - 1 - j) & 1) == b) mx = std::max(mx, d[l]);
                    for (int l = 0; l < n; ++l) e[l] = (l >> (k - 1 - j) & 1) == b ? std::exp(d[l] - mx) : 0.0;
                    lse[b] = mx + std::log(np_sum(e.data(), n));
                }
                *o++ = lse[0] - lse[1];
            }
        }
    }
}

void llr_general(std::span<const cd> y, std::span<const cd> h, std::span<const double> var, const Constellation& c,
                 double* out) {
    const std::size_t M = c.points.size();
    // The sums over the points whose bit j is 0 / 1, from a tree of partial sums: each pass pairs
    // neighbouring entries (labels differing in their last remaining bit), so the even and odd entries
    // summed on the way are bit j's two halves, 2 M adds in all instead of M c.m data-dependent ones.
    std::vector<double> d(M), ping(M), pong(M), s0(c.m), s1(c.m);
    for (std::size_t i = 0; i < y.size(); ++i) {
        double mx = -std::numeric_limits<double>::infinity();
        for (std::size_t p = 0; p < M; ++p) {
            const cd hx = mul(h[i], c.points[p]);
            const double dr = y[i].real() - hx.real(), di = y[i].imag() - hx.imag();
            d[p] = -(dr * dr + di * di) / var[i];  // numpy's hypot, squared, to rounding
            mx = std::max(mx, d[p]);
        }
        for (std::size_t p = 0; p < M; ++p) ping[p] = std::exp(d[p] - mx);
        double *a = ping.data(), *b = pong.data();
        std::size_t n = M;
        for (int j = c.m - 1; j >= 0; --j) {
            double zeros = 0.0, ones = 0.0;
            for (std::size_t q = 0; q < n; q += 2) {
                b[q / 2] = a[q] + a[q + 1];
                zeros += a[q];
                ones += a[q + 1];
            }
            s0[j] = zeros;
            s1[j] = ones;
            n /= 2;
            std::swap(a, b);
        }
        double* o = out + i * c.m;
        // A half underflowing (|LLR| > ~690) is floored, as in the reference.
        for (int j = 0; j < c.m; ++j) o[j] = std::log(std::max(s0[j], 1e-300)) - std::log(std::max(s1[j], 1e-300));
    }
}

}  // namespace

const Constellation* find(std::string_view name) {
    for (const auto& c : tables::CONSTELLATIONS)
        if (c.name == name) return &c;
    return nullptr;
}

std::vector<cd> modulate(std::span<const std::uint8_t> bits, const Constellation& c) {
    if (bits.size() % c.m) throw std::invalid_argument("bit count is not a multiple of bits per symbol");
    std::vector<cd> out(bits.size() / c.m);
    for (std::size_t s = 0; s < out.size(); ++s) {
        std::size_t idx = 0;
        for (int j = 0; j < c.m; ++j) {
            const auto b = bits[s * c.m + j];
            if (b > 1) throw std::invalid_argument("bits must be 0 or 1");
            idx = idx << 1 | b;
        }
        out[s] = c.points[idx];
    }
    return out;
}

std::vector<double> llr(std::span<const cd> y, std::span<const cd> h, std::span<const double> var,
                        const Constellation& c) {
    if (h.size() != y.size() || var.size() != y.size()) throw std::invalid_argument("y, h, var sizes differ");
    std::vector<double> out(y.size() * c.m);
    if (gray_square(c) && c.m == 2)
        llr_qpsk(y, h, var, out.data());
    else if (gray_square(c) && c.m <= 8)
        llr_square(y, h, var, c, out.data());
    else
        llr_general(y, h, var, c, out.data());
    return out;
}

std::vector<cd> ace_project(std::span<const cd> got, std::span<const cd> want, std::span<const cd> dirs) {
    if (want.size() != got.size() || dirs.size() != 2 * got.size())
        throw std::invalid_argument("ace_project: got, want, dirs sizes differ");
    std::vector<cd> out(got.size());
    for (std::size_t n = 0; n < got.size(); ++n) {
        const double er = got[n].real() - want[n].real(), ei = got[n].imag() - want[n].imag();
        double outr = want[n].real(), outi = want[n].imag();
        for (int i = 0; i < 2; ++i) {
            const double dr = dirs[2 * n + i].real(), di = dirs[2 * n + i].imag();
            const double a = mul({er, ei}, {dr, -di}).real();  // component along the direction
            const double s = a * (a > 0);  // as the reference: NaN and -0 alike
            outr += dr * s;
            outi += di * s;
        }
        out[n] = {outr, outi};
    }
    return out;
}

}  // namespace data2g::constellation
