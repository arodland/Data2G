#include "waveform/dsp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>

#include "dsp/dsp.hpp"
#include "util/simd.hpp"

namespace data2g::waveform {
namespace {

using config::FS;

constexpr double TWO_PI = 2.0 * std::numbers::pi;
constexpr int HET_G = std::gcd(config::FCENTER, FS);
constexpr int HET_PERIOD = FS / HET_G;  // 16 at 8 kHz, 32 at 16
constexpr int HET_STEP = config::FCENTER / HET_G;  // 3

// k / period is exact for a power-of-two period: as accurate as exp() can be
const std::array<cdouble, HET_PERIOD>& het_table() {
    static const auto t = [] {
        std::array<cdouble, HET_PERIOD> out;
        for (int k = 0; k < HET_PERIOD; ++k) out[static_cast<std::size_t>(k)] = std::polar(1.0, -TWO_PI * k / HET_PERIOD);
        return out;
    }();
    return t;
}

double mean_square(std::span<const double> x) {
    std::vector<double> sq(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) sq[i] = x[i] * x[i];
    return dsp::pairwise_sum(sq) / static_cast<double>(x.size());
}

}  // namespace

std::vector<cdouble> to_baseband(std::span<const double> x, std::int64_t n0) {
    const auto& h = het_table();
    std::int64_t k = (HET_STEP * (n0 % HET_PERIOD)) % HET_PERIOD;
    if (k < 0) k += HET_PERIOD;
    std::vector<cdouble> out(x.size());
    for (std::size_t n = 0; n < x.size(); ++n) {
        out[n] = x[n] * h[static_cast<std::size_t>(k)];
        k += HET_STEP;
        if (k >= HET_PERIOD) k -= HET_PERIOD;
    }
    return out;
}

namespace {

// Vectorized exp(-2*pi*i * wrap_cycles(f_hz * n / FS)) * z[n]: exact same cycle rounding as
// data2g.waveform.dsp.freq_correct, with quarter-turn reduction + degree-16/15 Taylor on [-pi/4, pi/4]
// (< 1e-16 truncation error) so the whole loop auto-vectorizes in DATA2G_SIMD_CLONES.
DATA2G_SIMD_CLONES void freq_correct_kernel(const cdouble* __restrict z, cdouble* __restrict out, std::size_t n,
                                            double f_hz) {
    const double* src = reinterpret_cast<const double*>(z);
    double* dst = reinterpret_cast<double*>(out);
    constexpr double fs = static_cast<double>(FS);
    for (std::size_t i = 0; i < n; ++i) {
        const double c = (f_hz * static_cast<double>(i)) / fs;
        const double f = c - std::floor(c);  // [0, 1)
        const double qf = std::floor(4.0 * f + 0.5);  // nearest quarter-turn: 0..4
        const double t = -TWO_PI * (f - 0.25 * qf);   // [-pi/4, pi/4]
        const double t2 = t * t;
        const double cv =
            1.0 +
            t2 * (-1.0 / 2.0 +
                  t2 * (1.0 / 24.0 +
                        t2 * (-1.0 / 720.0 +
                              t2 * (1.0 / 40320.0 +
                                    t2 * (-1.0 / 3628800.0 +
                                          t2 * (1.0 / 479001600.0 +
                                                t2 * (-1.0 / 87178291200.0 + t2 * (1.0 / 20922789888000.0))))))));
        const double sv =
            t * (1.0 +
                 t2 * (-1.0 / 6.0 +
                       t2 * (1.0 / 120.0 +
                             t2 * (-1.0 / 5040.0 +
                                   t2 * (1.0 / 362880.0 +
                                         t2 * (-1.0 / 39916800.0 +
                                               t2 * (1.0 / 6227020800.0 + t2 * (-1.0 / 1307674368000.0))))))));
        const int q = static_cast<int>(qf);
        double pr = (q & 1) ? sv : cv;
        double pi = (q & 1) ? -cv : sv;
        if (q & 2) {
            pr = -pr;
            pi = -pi;
        }
        const double zr = src[2 * i], zi = src[2 * i + 1];
        dst[2 * i] = zr * pr - zi * pi;
        dst[2 * i + 1] = zr * pi + zi * pr;
    }
}

}  // namespace

std::vector<cdouble> freq_correct(std::span<const cdouble> z, double f_hz) {
    if (f_hz == 0.0) return std::vector<cdouble>(z.begin(), z.end());
    std::vector<cdouble> out(z.size());
    freq_correct_kernel(z.data(), out.data(), z.size(), f_hz);
    return out;
}

std::vector<double> tx_condition(std::span<const double> x, double clip_headroom_db, std::span<const double> overshoot,
                                 std::size_t active_lo, std::size_t active_hi, std::pair<double, double> bandpass,
                                 const Projector& project, std::span<const double> closing) {
    active_hi = std::min(active_hi, x.size());
    active_lo = std::min(active_lo, active_hi);
    const auto active = [&](std::span<const double> v) { return v.subspan(active_lo, active_hi - active_lo); };
    std::vector<double> out(x.begin(), x.end());
    const double power = mean_square(active(x));
    if (!(power > 0)) return out;  // silence, or (NaN) an empty active range
    // mean envelope power is 2x mean real power
    const double thresh = std::sqrt(2 * power) * std::pow(10.0, clip_headroom_db / 20);
    const std::vector<double> taps = dsp::firwin_bandpass(config::TX_FILTER_TAPS, bandpass.first, bandpass.second, FS);
    std::vector<double> ks(overshoot.begin(), overshoot.end());
    if (project) ks.insert(ks.end(), closing.begin(), closing.end());
    for (std::size_t i = 0; i < ks.size(); ++i) {
        const double k = ks[i];
        const std::vector<cdouble> z = dsp::hilbert(out);
        for (std::size_t j = 0; j < out.size(); ++j) {
            double scale = std::min(1.0, thresh / std::max(std::abs(z[j]), 1e-12));
            // numpy computes `** 2.0` as a square, any other power with pow
            if (k == 2.0) scale = scale * scale;
            else if (k != 1.0) scale = std::pow(scale, k);
            out[j] = z[j].real() * scale;
        }
        out = dsp::convolve_same(out, taps);
        if (project && i < overshoot.size()) out = project(out);
    }
    const double rms = std::sqrt(mean_square(active(out)));
    for (double& v : out) v /= rms;
    return out;
}

double papr_db(std::span<const double> x) {
    if (x.empty()) throw std::invalid_argument("papr_db of no samples");
    const std::vector<cdouble> z = dsp::hilbert(x);
    std::vector<double> env2(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) {
        const double a = std::abs(z[i]);
        env2[i] = a * a;
    }
    const double mean = dsp::pairwise_sum(env2) / static_cast<double>(env2.size());
    return 10 * std::log10(*std::max_element(env2.begin(), env2.end()) / mean);
}

}  // namespace data2g::waveform
