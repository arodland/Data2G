#include "rx_plot.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

#include "arq/engine.hpp"
#include "constellation/constellation.hpp"
#include "cpm/cpm.hpp"
#include "modem/modem.hpp"
#include "util/mat.hpp"

namespace data2g::app {

namespace {

RxConstellation iq_plot(const std::string& mode, std::span<const std::complex<double>> y, const modem::Spec& spec) {
    RxConstellation plot;
    plot.mode = mode;
    const std::size_t step = y.size() / RxConstellation::MAX_SYMBOLS + 1;
    for (std::size_t i = 0; i < y.size(); i += step) plot.symbols.emplace_back(y[i]);
    if (const auto* c = constellation::find(spec.constellation))
        for (const auto& p : c->points) plot.points.emplace_back(p);
    return plot;
}

// Above 1, the strongest tone's excess counts for more: 32 tones' clusters
// part at +3 dB, and still blur at -3 (runs/tone-wheel/compare-p.png; 1.5 and
// up pull a poor signal onto the rim).
constexpr double EXCESS_POWER = 1.25;

RxConstellation tone_plot(const std::string& mode, const Mat<double>& shares) {
    RxConstellation plot;
    plot.mode = mode;
    plot.tones = true;
    std::vector<std::complex<double>> wheel;
    for (std::size_t k = 0; k < shares.cols; ++k) wheel.push_back(std::polar(1.0, 2 * std::numbers::pi * k / shares.cols));
    for (const auto& w : wheel) plot.points.emplace_back(w);
    const std::size_t step = shares.rows / RxConstellation::MAX_SYMBOLS + 1;
    std::vector<double> row(shares.cols);
    for (std::size_t i = 0; i < shares.rows; i += step) {
        // the median tone is the symbol's noise floor: only the excess counts, or a
        // clean 32-tone symbol's noise in the other 31 pulls it halfway in
        row.assign(shares[i], shares[i] + shares.cols);
        const auto m = row.begin() + row.size() / 2;
        std::nth_element(row.begin(), m, row.end());
        const double floor = row.size() % 2 ? *m : (*m + *std::max_element(row.begin(), m)) / 2;
        std::complex<double> z = 0;
        double total = 0;
        for (std::size_t k = 0; k < shares.cols; ++k) {
            const double e = std::pow(std::max(shares[i][k] - floor, 0.0), EXCESS_POWER);
            z += e * wheel[k];
            total += e;
        }
        plot.symbols.emplace_back(total > 0 ? z / total : z);
    }
    return plot;
}

}  // namespace

bool plot_visible(const RxConstellation& c) {
    if (c.tones) return !c.symbols.empty();
    float reach = 0;
    for (const auto& q : c.points) reach = std::max({reach, std::abs(q.real()), std::abs(q.imag())});
    const float edge = 1.5f * (reach > 0 ? reach : 1.0f);
    return std::any_of(c.symbols.begin(), c.symbols.end(),
                       [&](const auto& z) { return std::abs(z.real()) <= edge && std::abs(z.imag()) <= edge; });
}

std::optional<RxConstellation> rx_constellation(const arq::BurstHeard& b) {
    if (b.heard.ofdm) return iq_plot(b.submode, modem::equalized(*b.heard.ofdm), *b.heard.ofdm->spec);
    if (b.heard.cpm) return tone_plot(b.submode, cpm::shares(b.heard.cpm->E));
    return std::nullopt;
}

RxConstellation rx_constellation(const arq::BurstPreview& p) {
    auto plot = p.ofdm ? iq_plot(p.submode, p.iq, *p.ofdm) : tone_plot(p.submode, p.shares);
    plot.live = true;
    return plot;
}

}  // namespace data2g::app
