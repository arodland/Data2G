// Known answers and self-consistency that need no Python. Parity with
// data2g/constellation.py is tests/test_native_constellation.py's job.

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "constellation/constellation.hpp"

using namespace data2g;
using constellation::cd;

int main() {
    check::report_crashes_instead_of_prompting();
    std::mt19937 rng(1);
    std::normal_distribution<double> gauss;

    check::is_true(constellation::find("gray-qam16") && !constellation::find("gray-qam8"), "find");
    for (const auto& c : tables::CONSTELLATIONS) {
        const std::string n(c.name);
        check::equal(c.points.size(), std::size_t(1) << c.m, n + " has 2^m points");
        check::equal(c.ace.size(), 2 * c.points.size(), n + " has 2 ACE directions per point");
        double p = 0;
        for (auto x : c.points) p += std::norm(x);
        // Learned sets were trained in float32: unit power to ~1e-7.
        check::is_true(std::abs(p / c.points.size() - 1) < 1e-6, n + " unit average power");
        bool unit_or_zero = true, outward = true;
        for (std::size_t i = 0; i < c.ace.size(); ++i) {
            const double a = std::abs(c.ace[i]);
            unit_or_zero &= a == 0 || std::abs(a - 1) < 1e-12;
            outward &= (c.ace[i] * std::conj(c.points[i / 2])).real() >= 0;
        }
        check::is_true(unit_or_zero && outward, n + " ACE directions unit or zero, outward");

        // modulate: every label, MSB first, maps to its point.
        std::vector<std::uint8_t> bits;
        for (std::size_t i = 0; i < c.points.size(); ++i)
            for (int j = c.m - 1; j >= 0; --j) bits.push_back(i >> j & 1);
        const auto x = constellation::modulate(bits, c);
        check::is_true(std::equal(x.begin(), x.end(), c.points.begin()), n + " modulate maps label i to point i");

        // Noiseless, var far below dmin^2 (c256-snr26's is 1.5e-7): every
        // LLR's sign is its bit's, through a random phase.
        std::vector<cd> y, h;
        for (auto s : x) {
            h.push_back(std::polar(1.0, gauss(rng)));
            y.push_back(h.back() * s);
        }
        const auto l = constellation::llr(y, h, std::vector<double>(y.size(), 1e-9), c);
        bool signs = true;
        for (std::size_t i = 0; i < l.size(); ++i) signs &= (l[i] < 0) == (bits[i] == 1);
        check::is_true(signs, n + " LLR signs at high SNR");

        // The Gray fast paths are the exact LLR: the same points under another
        // name take the general path. 1e-9 as the reference's own test.
        if (c.name.starts_with("gray-qam")) {
            const tables::Constellation general{"general", c.m, c.points, c.ace};
            std::vector<double> var;
            y.clear();
            h.clear();
            for (int i = 0; i < 300; ++i) {
                y.emplace_back(gauss(rng), gauss(rng));
                h.emplace_back(i < 3 ? 0.0 : gauss(rng), i < 3 ? 0.0 : gauss(rng));  // a dead carrier: LLR 0
                var.push_back(0.05 + i / 200.0);
            }
            check::close(constellation::llr(y, h, var, c), constellation::llr(y, h, var, general), 1e-9,
                         n + " fast LLR path equals the exact one");
        }
    }
    check::is_true(std::abs(constellation::find("gray-qam16")->points[0] - cd(-3, -3) / std::sqrt(10.0)) < 1e-15,
                   "gray-qam16 label 0 is (-3, -3) / sqrt(10)");

    bool threw = false;
    try {
        constellation::modulate(std::vector<std::uint8_t>{0, 1, 1}, *constellation::find("gray-qam4"));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "modulate rejects a partial symbol");

    // ACE: the outward part of the error along a direction stays, the rest goes.
    const std::vector<cd> want{{1, 1}}, dirs{{1, 0}, {0, 1}};
    check::is_true(constellation::ace_project(std::vector<cd>{{1.5, 0.7}}, want, dirs)[0] == cd(1.5, 1), "ACE keeps outward");
    check::is_true(constellation::ace_project(std::vector<cd>{{0.2, 3}}, want, std::vector<cd>{{0, 0}, {0, 0}})[0] ==
                       cd(1, 1), "ACE with no directions returns want");
    return check::report("constellation");
}
