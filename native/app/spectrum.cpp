#include "spectrum.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "dsp/fft.hpp"

namespace data2g::app::spectrum {

namespace {

constexpr double DB_FLOOR = -95.0, DB_CEIL = -20.0;
// TX audio is full scale (a burst's peak at 1.0): its own range, so it neither
// saturates nor reads as quieter than the band.
constexpr double TX_DB_FLOOR = -80.0, TX_DB_CEIL = -15.0;

struct Stop {
    double at;
    Rgb color;
};
using Ramp = std::array<Stop, 5>;

// 256-entry colour ramp through the stops.
std::array<Rgb, 256> ramp(const Ramp& stops) {
        std::array<Rgb, 256> out{};
        for (int i = 0; i < 256; ++i) {
            const double x = i / 255.0;
            std::size_t k = 0;
            while (k + 2 < stops.size() && x > stops[k + 1].at) ++k;
            const Stop& lo = stops[k];
            const Stop& hi = stops[k + 1];
            const double t = (x - lo.at) / (hi.at - lo.at);
            for (int ch = 0; ch < 3; ++ch)
                out[i][ch] = static_cast<std::uint8_t>(lo.color[ch] + t * (hi.color[ch] - lo.color[ch]));
        }
        return out;
}

// RX: black -> blue -> green -> yellow -> white.
const std::array<Rgb, 256>& colormap() {
    static const auto lut =
        ramp({{{0.00, {0, 0, 0}}, {0.25, {0, 0, 140}}, {0.50, {0, 170, 90}}, {0.75, {245, 235, 40}}, {1.00, {255, 255, 255}}}});
    return lut;
}

// TX: black -> purple -> magenta -> pink -> white, apart from every RX hue.
const std::array<Rgb, 256>& tx_colormap() {
    static const auto lut =
        ramp({{{0.00, {0, 0, 0}}, {0.25, {70, 0, 90}}, {0.50, {200, 30, 140}}, {0.75, {255, 140, 200}}, {1.00, {255, 255, 255}}}});
    return lut;
}

}  // namespace

std::vector<double> spectrum_db(const std::vector<double>& block) {
    static const std::vector<double> window = [] {
        std::vector<double> w(NFFT);
        for (int i = 0; i < NFFT; ++i) w[i] = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * i / (NFFT - 1));
        return w;
    }();
    std::vector<dsp::cdouble> buf(NFFT);
    for (int i = 0; i < NFFT; ++i) buf[i] = block[i] * window[i];
    const auto spec = dsp::fft(buf, true);
    std::vector<double> out(BINS);
    for (int i = 0; i < BINS; ++i) out[i] = 20.0 * std::log10(std::abs(spec[i]) / NFFT + 1e-12);
    return out;
}

std::vector<double> reduce_to_width(const std::vector<double>& v, int width) {
    const int n = static_cast<int>(v.size());
    if (width <= 0 || n == 0) return {};
    if (width == n) return v;
    std::vector<double> out(width);
    if (width > n) {
        for (int i = 0; i < width; ++i) {
            const double at = width == 1 ? 0.0 : static_cast<double>(i) * (n - 1) / (width - 1);
            const int lo = static_cast<int>(at), hi = std::min(lo + 1, n - 1);
            out[i] = v[lo] + (at - lo) * (v[hi] - v[lo]);
        }
        return out;
    }
    for (int i = 0; i < width; ++i) {
        const int start = static_cast<int>(static_cast<long long>(i) * n / width);
        const int stop = i + 1 == width ? n : static_cast<int>(static_cast<long long>(i + 1) * n / width);
        out[i] = *std::max_element(v.begin() + start, v.begin() + stop);
    }
    return out;
}

std::vector<Rgb> rgb_row(const std::vector<double>& block, bool tx, int width) {
    if (block.size() < static_cast<std::size_t>(NFFT)) return {};
    const std::vector<double> newest(block.end() - NFFT, block.end());
    const std::vector<double> db = reduce_to_width(spectrum_db(newest), width);
    const auto& lut = tx ? tx_colormap() : colormap();
    const double floor = tx ? TX_DB_FLOOR : DB_FLOOR, ceil = tx ? TX_DB_CEIL : DB_CEIL;
    std::vector<Rgb> out(db.size());
    for (std::size_t x = 0; x < db.size(); ++x)
        out[x] = lut[static_cast<std::size_t>(std::clamp((db[x] - floor) / (ceil - floor), 0.0, 1.0) * 255.0)];
    return out;
}

double peak(const std::vector<double>& block, std::size_t n) {
    double p = 0.0;
    for (auto it = block.end() - static_cast<std::ptrdiff_t>(std::min(n, block.size())); it != block.end(); ++it)
        p = std::max(p, std::abs(*it));
    return p;
}

}  // namespace data2g::app::spectrum
