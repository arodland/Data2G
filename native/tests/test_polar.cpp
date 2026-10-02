// Polar round trips that need no Python. Parity with data2g/polar.py is
// tests/test_native_polar.py's job.

#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "polar/polar.hpp"

using namespace data2g;

namespace {

// Encode random info words, send BPSK over AWGN at sigma (0: noiseless), decode;
// returns how many rows came back as path 0.
int round_trip(const polar::PolarCode& code, int rows, double sigma, unsigned seed, const std::string& what) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, sigma > 0 ? sigma : 1.0);
    Mat<std::uint8_t> bits(rows, code.k);
    for (auto& b : bits.data) b = rng() & 1;
    const auto x = code.encode(bits);
    check::equal(x.cols, static_cast<std::size_t>(code.e), what + " coded length");
    Mat<float> llr(rows, code.e);
    for (std::size_t i = 0; i < x.data.size(); ++i) {
        const double y = 1.0 - 2.0 * x.data[i] + (sigma > 0 ? noise(rng) : 0.0);
        llr.data[i] = static_cast<float>(sigma > 0 ? 2 * y / (sigma * sigma) : 8 * y);
    }
    const polar::SCLDecoder dec(code, 8);
    const auto r = dec.decode(llr);
    int ok = 0;
    for (int b = 0; b < rows; ++b) {
        ok += std::equal(bits[b], bits[b] + code.k, r.paths[b]);
        check::is_true(std::is_sorted(r.metric[b], r.metric[b] + 8), what + " metrics ascending");
    }
    return ok;
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();

    std::vector<std::uint8_t> u(64);
    std::mt19937 rng(0);
    for (auto& b : u) b = rng() & 1;
    auto v = u;
    polar::transform(v);
    polar::transform(v);
    check::is_true(v == u, "transform is an involution");
    std::vector<std::uint8_t> unit(8, 0);
    unit[7] = 1;
    polar::transform(unit);
    check::is_true(std::ranges::all_of(unit, [](auto b) { return b == 1; }), "last row of G is all ones");

    // Puncturing: n - e of the bit-reversed order, sorted; sent is the rest.
    const std::uint16_t info3[] = {3, 5, 6, 7};
    const polar::PolarCode small(4, 6, info3);
    check::equal(small.n, 8, "mother length");
    check::is_true(small.punctured == std::vector<std::uint16_t>{0, 4}, "punctured = bitrev(3)[:2], sorted");
    check::is_true(small.sent == std::vector<std::uint16_t>{1, 2, 3, 5, 6, 7}, "sent");

    int n_codes = 0;
    for (const auto& s : config::SUBMODES) {
        if (s.code != "polar") continue;
        ++n_codes;
        const auto code = polar::polar_code(s);
        const std::string name(s.name);
        check::equal(code.n, static_cast<int>(std::bit_ceil(static_cast<unsigned>(s.coded_bits))), name + " n");
        check::equal(round_trip(code, 16, 0.0, 1, name), 16, name + " noiseless round trip");
    }
    check::is_true(n_codes > 0, "some polar submodes");

    const polar::PolarCode ctl(184, 360, polar::ga_info_pos(184, 360));
    check::equal(round_trip(ctl, 16, 0.0, 2, "ctl"), 16, "CPM ctl noiseless round trip");
    bool threw = false;
    try {
        polar::ga_info_pos(10, 20);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    check::is_true(threw, "no GA design for an unknown (k, e)");

    // tests/test_polar.py's noisy case: k 48 of e 480, sigma 0.8, > 95% at path 0.
    const auto ack = polar::polar_code(*std::ranges::find_if(config::SUBMODES, [](auto& s) { return s.name == "ack-4f"; }));
    check::is_true(round_trip(ack, 64, 0.8, 3, "ack-4f noisy") > 60, "ack-4f decodes at sigma 0.8");
    return check::report("polar");
}
