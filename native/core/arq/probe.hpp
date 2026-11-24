// FM sounding probe (data2g/arq/probe.py, normative): known multitones behind a ping's identity burst, measured
// without decoding: a sounding of every tone (passband, noise, band edges) and a level staircase in the data band
// (SINAD per step: the drive that suits the path).
#pragma once
#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include "arq/frames.hpp"
#include "arq/ping.hpp"
namespace data2g::arq::probe {
inline constexpr double TONE_LO_HZ = 100.0, TONE_HI_HZ = 7000.0;
inline constexpr int SETTLE = 2, MEAS = 10, SOUND_MEAS = 20, GUARD = 1;
inline constexpr double SOUND_DB = -12.0, GAP_S = 0.05;
inline constexpr std::array<double, 6> STEPS_DB = {4.0, 0.0, -6.0, -12.0, -18.0, 0.0};
inline constexpr int REF = 1;
inline constexpr double CAP_DB = 28.0, TIE_DB = 1.5, EDGE_DB = 10.0, WIDE_HZ = 5400.0, VARIED_DB = 2.0;
inline constexpr int REPORT_BYTES = 21;
inline const std::string STAIR_BAND = "f55";

int period();  // samples: 50 Hz bins
double bin_hz();
const std::vector<int>& tones();
const std::vector<int>& holes();
std::size_t n_samples();
// The probe (silence GAP_S, the sounding, the staircase), unit rms x `level` at STEPS_DB 0, through the TX path.
std::vector<double> waveform(std::string_view tx_path, double level);

struct Report {
    double lo_hz = 0, hi_hz = 0, snr_nb = 0, snr_top = 0, snr_tilt = 0, noise_tilt = 0;
    std::array<double, STEPS_DB.size()> snr{}, sinad{};
    double rms_dbfs = -64, peak_dbfs = -64;
    int clip_permille = 0;
};
// The probe as received (`y` from where waveform() starts), and its input level -> the numbers a report carries.
Report measure(std::span<const double> y, const ping::Level& lvl);
Bytes pack_report(const Report& m);
Report unpack_report(ByteView b);  // throws std::invalid_argument if short
std::string fmt(const Report& m);
double best_step(const Report& m);
std::vector<std::string> verdict(const Report& m, std::string_view tx_path);
}  // namespace data2g::arq::probe
