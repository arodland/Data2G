// The receiver's measurements of a burst, as plain data: what phy::measure
// (data2g/arq/phy.py measure's dict) and cpm measure produce, the predictor
// reads and a Policy observes. Apart from predictor.hpp so the link layer
// (link.hpp's Policy::observe) needs no outcome model.
#pragma once

#include <array>
#include <optional>
#include <string_view>

namespace data2g::arq {

inline constexpr std::array<std::string_view, 4> CONSTS = {"gray-qam4", "gray-qam16", "c64-snr18", "c256-snr26"};

// The receiver's noise profile (receiver::NoiseProfile's snapshot), per NOISE_BANDS_HZ.
inline constexpr std::array<std::array<double, 2>, 5> NOISE_BANDS_HZ = {
    {{350, 950}, {950, 1300}, {1300, 1750}, {1750, 2100}, {2100, 2700}}};
struct NoiseLevels {
    std::array<double, 5> db{}, tail_db{};
    double impulses_per_min = 0.0;
};

// The receiver's measurements of a burst (phy.measure, cpm.measure);
// mi in CONSTS order. headroom and frames: Python's .get defaults.
struct Measured {
    double snr_est = 0.0, spread_est = 0.0, delay_est_ms = 0.0;
    std::array<double, CONSTS.size()> mi{};
    double headroom = 0.0, frames = 16.0;
    // std over frames of each frame's noise, dB (phy.py n0_spread_db: FM's regime input; 0 on HF).
    // GearShifter::observe replaces it with the median over the peer's last bursts.
    double n0_spread_db = 0.0;
    // Post-decode data-cell MER (dB, SNR_REF_BW_HZ convention; issue #290), nullopt before data_mer.
    std::optional<double> mer_db;
    std::optional<NoiseLevels> noise;  // the receiver's, with it (the gear shifter's noise rule)
    // the energy inputs (GearShifter::energy_features: mean dB, fill, 1); nullopt: none yet
    std::optional<std::array<double, 3>> energy;
};

}  // namespace data2g::arq
