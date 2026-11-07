// Hand burst round trips that need no Python. Parity with data2g/hand.py is
// tests/test_native_hand.py's job.

#include <cmath>
#include <random>
#include <vector>

#include "check.hpp"
#include "hand/hand.hpp"

using namespace data2g;

int main() {
    check::report_crashes_instead_of_prompting();
    check::equal(hand::channel_freq(0), 330.0, "channel 0");
    check::equal(hand::channel_freq(39), 2670.0, "channel 39");
    const std::uint64_t p = 0x5A5A123456789ABCull & ((1ull << hand::PAYLOAD) - 1);
    check::equal(hand::payload_int(hand::payload_bits(p)), p, "payload bits round trip");

    // two Hands 2 channels apart, landing late, at 6 dB; a third channel empty
    const int tag = 0x5A;
    const std::int64_t late = 1234;
    std::vector<double> y(static_cast<std::size_t>(hand::WINDOW_SAMPLES), 0.0);
    for (auto [c, payload] : {std::pair{20, p}, std::pair{22, std::uint64_t{42}}}) {
        const auto x = hand::modulate(tag, hand::payload_bits(payload), hand::channel_freq(c));
        for (std::size_t i = 0; i < x.size(); ++i) y[static_cast<std::size_t>(hand::SEARCH_SAMPLES + late) + i] += x[i];
    }
    std::mt19937 rng(3);
    std::normal_distribution<double> noise(0.0, std::sqrt(4000.0 / 2500.0 / std::pow(10.0, 0.6)));
    for (double& v : y) v += noise(rng);
    const std::vector<int> channels{20, 21, 22};
    const auto got = hand::decode_window(y, tag, channels, hand::SEARCH_SAMPLES);
    check::equal(got.size(), std::size_t{2}, "two Hands decoded");
    if (got.size() != 2) return check::report("hand");
    check::equal(got[0].c, 20, "first channel");
    check::equal(got[0].payload, p, "first payload");
    check::equal(got[1].c, 22, "second channel");
    check::equal(got[1].payload, std::uint64_t{42}, "second payload");
    check::is_true(got[0].snr > 0, "snr");
    check::is_true(std::abs(got[0].dt_ms - 1000.0 * late / 8000) < 30, "dt");
    // another net's tag: nothing passes the CRC
    check::equal(hand::decode_window(y, 0x5B, channels, hand::SEARCH_SAMPLES).size(), std::size_t{0}, "wrong tag");
    // --hand-round cheap's grids find the same two Hands
    const auto cheap = hand::decode_window(y, tag, channels, hand::SEARCH_SAMPLES, hand::ROUND_CHEAP);
    check::equal(cheap.size(), std::size_t{2}, "the cheap round finds them too");
    if (cheap.size() == 2) check::equal(cheap[0].payload, p, "the cheap round's payload");
    return check::report("hand");
}
