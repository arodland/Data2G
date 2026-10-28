// cwid/: the CW ID's audio and CwSense (tests/test_cwid.py; parity in tests/test_native_cwid.py).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "cwid/cwid.hpp"

using namespace data2g;

namespace {

void audio() {
    check::current_step = "audio";
    check::equal(cwid::id_text("w1aw-7"), std::string("DE W1AW"), "DE and the call less its SSID");
    const auto x = cwid::audio("DE W1AW", 20);
    check::equal(static_cast<double>(x.size()), std::round(cwid::seconds("DE W1AW", 20) * config::FS), "length");
    double peak = 0;
    for (double v : x) peak = std::max(peak, std::abs(v));
    check::is_true(std::abs(peak - 1.0) < 1e-3 && std::abs(x.front()) < 1e-3 && std::abs(x.back()) < 1e-3,
                   "peak 1, silent ends");
    // PARIS: 50 dots a word
    check::is_true(std::abs(cwid::seconds("PARIS", 20) - (43 * 0.06 + cwid::RAMP / 8000.0)) < 1e-9, "PARIS timing");
    bool threw = false;
    try {
        cwid::audio("DE W1AW", 60);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "60 wpm refused");
}

// Whether a CwSense fed `x` in 10 ms blocks was ever active.
bool heard(const std::vector<double>& x) {
    cwid::CwSense s;
    bool any = false;
    for (std::size_t i = 0; i < x.size(); i += 80)
        any |= s.feed(std::span(x).subspan(i, std::min<std::size_t>(80, x.size() - i)));
    return any;
}

void sense() {
    check::current_step = "sense";
    // noise: unit variance; the ID at key-down SNR 0 dB in 2500 Hz
    std::mt19937_64 g(7);  // uniform-based noise, the same on every platform
    std::uniform_real_distribution<double> u(-std::sqrt(3.0), std::sqrt(3.0));
    std::vector<double> noise(60 * config::FS);
    for (double& v : noise) v = u(g);
    check::is_true(!heard(noise), "60 s of noise: never");
    const double a = std::sqrt(2 * 2500.0 / (config::FS / 2.0));  // a tone at 0 dB: its power a^2 / 2, the noise in 2500 Hz
    auto x = cwid::audio("DE W1AW", 20);
    std::vector<double> y(noise.begin(), noise.begin() + static_cast<std::ptrdiff_t>(x.size() + 2 * config::FS));
    for (std::size_t i = 0; i < x.size(); ++i) y[i + config::FS] += a * x[i];
    check::is_true(heard(y), "an ID at 0 dB: heard");
}

}  // namespace

int main() {
    audio();
    sense();
    return check::report("cwid");
}
