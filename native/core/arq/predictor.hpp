// data2g/arq/predictor.py's live path: the MIESM features (effective MI
// over the AWGN BICM capacity tables) and the outcome model, the frozen
// bootstrap ensemble (tables::OUTCOME_MEMBERS) with the installed
// LOGIT_OFFSETS. The study toggles (DATA2G_OUTCOME_MODEL, _OUTCOME_LCB,
// _LOGIT_OFFSETS) and the link abstraction are Python only.
#pragma once

#include <array>
#include <complex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tables/tables.hpp"

namespace data2g::arq {

inline constexpr std::array<std::string_view, 4> CONSTS = {"gray-qam4", "gray-qam16", "c64-snr18", "c256-snr26"};

// A submode's constellation -> the capacity table standing in for it.
std::string_view const_family(std::string_view name);
// np.interp over the family's table (clamped at the ends).
double capacity(double snr_db, std::string_view constellation);
// Mean capacity at |h|^2 / var over the channel uses (same length).
double effective_mi(std::span<const std::complex<double>> h, std::span<const double> var,
                    std::string_view constellation);

// The receiver's measurements of a burst (phy.measure, cpm.measure);
// mi in CONSTS order. headroom and frames: Python's .get defaults.
// The receiver's noise profile (tnc::NoiseProfile's snapshot), per NOISE_BANDS_HZ.
inline constexpr std::array<std::array<double, 2>, 5> NOISE_BANDS_HZ = {
    {{350, 950}, {950, 1300}, {1300, 1750}, {1750, 2100}, {2100, 2700}}};
struct NoiseLevels {
    std::array<double, 5> db{}, tail_db{};
    double impulses_per_min = 0.0;
};

struct Measured {
    double snr_est = 0.0, spread_est = 0.0, delay_est_ms = 0.0;
    std::array<double, CONSTS.size()> mi{};
    double headroom = 0.0, frames = 16.0;
    std::optional<NoiseLevels> noise;  // the receiver's, with it (the gear shifter's noise rule)
};

// What a sync band (OFDM band or CPM grid) occupies, Hz (predictor.band_span_hz).
std::array<double, 2> band_span_hz(std::string_view band);
// How much worse a burst in `band` should fare than the one measured in
// `measured_band`, from the noise profile, as an SNR drop (predictor.noise_shift_db).
double noise_shift_db(const std::optional<NoiseLevels>& noise, std::string_view measured_band, std::string_view band,
                      double tail_weight = 1.0, double deadband_db = 1.0);
// The measurements as if the SNR were shift_db lower (predictor.shifted).
Measured shifted(const Measured& m, double shift_db);

// The burst before the last: its measurements, band, and age (s).
struct Prev {
    Measured m;
    std::string band;
    double age = 0.0;
};

std::vector<double> outcome_inputs(const Measured& m, std::string_view band, double gap, double seconds,
                                   const Prev* prev = nullptr,
                                   std::span<const std::string_view> bands = tables::OUTCOME_BANDS);
// The ensemble's logits (mean member probability), LOGIT_OFFSETS not applied.
std::vector<double> outcome_logits(std::span<const double> x);

int outcome_index(std::string_view submode);  // its row in OUTCOME_MODES, -1: unknown
bool outcome_knows(std::string_view submode);

struct Outcome {
    double burst;  // P(burst usable)
    double cw;     // P(a codeword decodes | usable)
};
// Indexed by outcome_index, LOGIT_OFFSETS applied.
std::vector<Outcome> predict_outcome(const Measured& m, std::string_view band, double gap, double seconds,
                                     const Prev* prev = nullptr);

}  // namespace data2g::arq
