#include "audio/filters.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "dsp/dsp.hpp"
#include "generated/config.hpp"

namespace data2g::audio {

namespace {

int ratio(int rate) {
    if (rate < config::FS || rate % config::FS) throw std::invalid_argument("sample rate must be a multiple of FS");
    return rate / config::FS;
}

std::vector<double> design(int k, int rate, double gain) {
    if (k == 1) return {1.0};
    std::vector<double> h = dsp::firwin_lowpass(32 * k + 1, 0.9 * config::FS / 2, rate);
    for (double& v : h) v *= gain;
    return h;
}

double mean(std::span<const double> v) { return dsp::pairwise_sum(v) / static_cast<double>(v.size()); }

// np.median
double median(std::span<const double> v) {
    std::vector<double> s(v.begin(), v.end());
    const std::size_t h = s.size() / 2;
    std::nth_element(s.begin(), s.begin() + static_cast<std::ptrdiff_t>(h), s.end());
    if (s.size() % 2) return s[h];
    const double lo = *std::max_element(s.begin(), s.begin() + static_cast<std::ptrdiff_t>(h));
    return (lo + s[h]) / 2.0;
}

}  // namespace

std::vector<double> lfilter_fir(std::span<const double> b, std::span<const double> x, std::vector<double>& zi) {
    const std::size_t nb = b.size();
    if (zi.size() + 1 != nb) throw std::invalid_argument("lfilter_fir: zi must be len(b) - 1");
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double xn = x[i];
        if (nb == 1) {
            y[i] = xn * b[0];
            continue;
        }
        const double yn = xn * b[0] + zi[0];
        for (std::size_t n = 1; n < nb - 1; ++n) zi[n - 1] = zi[n] + xn * b[n];
        zi[nb - 2] = xn * b[nb - 1];
        y[i] = yn;
    }
    return y;
}

Decimator::Decimator(int rate) : d_(ratio(rate)), taps_(design(d_, rate, 1.0)), zi_(taps_.size() - 1, 0.0) {}

std::vector<double> Decimator::operator()(std::span<const double> x) {
    const std::vector<double> y = lfilter_fir(taps_, x, zi_);
    std::vector<double> out;
    out.reserve(y.size() / static_cast<std::size_t>(d_) + 1);
    for (std::size_t i = static_cast<std::size_t>(phase_); i < y.size(); i += static_cast<std::size_t>(d_))
        out.push_back(y[i]);
    phase_ = ((phase_ - static_cast<std::int64_t>(x.size())) % d_ + d_) % d_;
    return out;
}

Interpolator::Interpolator(int rate)
    : u_(ratio(rate)), taps_(design(u_, rate, static_cast<double>(u_))), zi_(taps_.size() - 1, 0.0) {}

std::vector<double> Interpolator::operator()(std::span<const double> x) {
    std::vector<double> up(x.size() * static_cast<std::size_t>(u_), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) up[i * static_cast<std::size_t>(u_)] = x[i];
    return lfilter_fir(taps_, up, zi_);
}

std::vector<double> Blanker::operator()(std::span<const double> x) {
    std::vector<double> y(x.begin(), x.end());
    std::vector<double> m, clipped;
    std::vector<char> zero, hit;
    for (std::size_t i = 0; i < y.size(); i += BLOCK) {
        const std::span<double> b(y.data() + i, std::min<std::size_t>(BLOCK, y.size() - i));
        const std::size_t n = b.size();
        m.resize(n);
        clipped.resize(n);
        for (std::size_t j = 0; j < n; ++j) m[j] = std::abs(b[j]);
        const double med = median(m);
        if (med > RESYNC * env) {
            for (std::size_t j = 0; j < n; ++j) clipped[j] = std::min(m[j], 3 * med);
            env = mean(clipped);
        }
        if (env <= 0) continue;  // digital silence
        hit.assign(n, 0);
        zero.assign(n, 0);
        bool any_hit = false, any_zero = false;
        for (std::size_t j = 0; j < n; ++j) {
            hit[j] = m[j] > LIMIT * env;
            zero[j] = m[j] > ZERO * env;
            any_hit |= hit[j] != 0;
            any_zero |= zero[j] != 0;
        }
        if (any_hit) {
            if (GUARD && any_zero) {  // dilate by GUARD each side
                std::vector<char> grown(n, 0);
                for (std::size_t j = 0; j < n; ++j) {
                    if (!zero[j]) continue;
                    const std::size_t lo = j >= GUARD ? j - GUARD : 0, hi = std::min(n, j + GUARD + 1);
                    std::fill(grown.begin() + static_cast<std::ptrdiff_t>(lo), grown.begin() + static_cast<std::ptrdiff_t>(hi), 1);
                }
                zero.swap(grown);
                for (std::size_t j = 0; j < n; ++j) hit[j] |= zero[j];
            }
            for (std::size_t j = 0; j < n; ++j) {
                if (!hit[j]) continue;
                b[j] *= zero[j] ? 0.0 : LIMIT * env / std::max(m[j], 1e-30);
                ++n_blanked;
            }
        }
        for (std::size_t j = 0; j < n; ++j) clipped[j] = std::min(m[j], 3 * env);
        env += (mean(clipped) - env) * std::min(1.0, static_cast<double>(n) / TAU);
    }
    return y;
}

}  // namespace data2g::audio
