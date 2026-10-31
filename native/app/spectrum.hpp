// The waterfall's rows: a block of the 8 kHz input as one row of colour, and
// the level meter's peak. Shared by data2g-gui's waterfall (gui/waterfall.cpp)
// and the Android app's; no Qt. Lifted from SSTVAE's dsp::spectrum_db and
// dsp::reduce_to_width.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "generated/config.hpp"

namespace data2g::app::spectrum {

constexpr int NFFT = 1024;           // samples per row
constexpr int BINS = NFFT / 2;       // 0 .. FS/2
constexpr double DISPLAY_HZ = config::FS / 2.0;
constexpr double CLIP = 0.99;        // a peak at or above it is clipping

using Rgb = std::array<std::uint8_t, 3>;

// Hann-windowed magnitude spectrum of NFFT samples in dB, scaled by 1/NFFT.
std::vector<double> spectrum_db(const std::vector<double>& block);
// Onto exactly `width` columns: interpolate when widening, peak-hold when
// shrinking so a narrow carrier can't be sampled away.
std::vector<double> reduce_to_width(const std::vector<double>& v, int width);
// The newest NFFT samples of `block` as `width` pixels. RX: black, blue, green,
// yellow, white over -95..-20 dB; `tx` (our own audio, full scale): black,
// purple, magenta, pink, white over -80..-15 dB, apart from every RX hue.
std::vector<Rgb> rgb_row(const std::vector<double>& block, bool tx, int width);
// The largest |sample| among the newest `n` of `block`.
double peak(const std::vector<double>& block, std::size_t n);

}  // namespace data2g::app::spectrum
