// Gear-shifter decision cost: observe + recommend on random measurements,
// recommend timed (single thread). Median, p95, max over 400 decisions.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "arq/policy.hpp"
#include "arq/predictor.hpp"

using namespace data2g;

int main() {
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> U(0, 1);
    arq::GearShifter g;
    std::vector<double> ms;
    double now = 0;
    for (int i = 0; i < 400; ++i) {
        arq::Measured m;
        m.snr_est = -8 + 33 * U(rng);
        m.spread_est = U(rng) < 0.5 ? 0.0 : 3 * U(rng);
        m.delay_est_ms = 4 * U(rng);
        m.frames = 16;
        for (std::size_t c = 0; c < arq::CONSTS.size(); ++c) m.mi[c] = arq::capacity(m.snr_est - 3, arq::CONSTS[c]);
        now += 5;
        g.observe(m, "qpsk-r1/2", now);
        g.observe_energy(-12 + 42 * U(rng), now);
        arq::StationView st;
        st.cap = i % 3;
        st.chat = U(rng) < 0.2;
        const auto t = std::chrono::steady_clock::now();
        g.recommend(st);
        const double ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        if (i >= 20) ms.push_back(ms_);
    }
    std::sort(ms.begin(), ms.end());
    std::printf("recommend median %.2f ms, p95 %.2f ms, max %.2f ms\n", ms[ms.size() / 2], ms[ms.size() * 95 / 100], ms.back());
}
