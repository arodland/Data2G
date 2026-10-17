#include "rx_plot.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <utility>

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

RxConstellation tone_plot(const std::string& mode, const Mat<double>& shares) {
    RxConstellation plot;
    plot.mode = mode;
    plot.tones = true;
    plot.points = {{1.0f, 1.0f}};
    const std::size_t step = shares.rows / RxConstellation::MAX_SYMBOLS + 1;
    for (std::size_t i = 0; i < shares.rows; i += step) {
        double top = 0, second = 0;
        for (std::size_t k = 0; k < shares.cols; ++k) {
            const double e = shares[i][k];
            if (e > top) second = std::exchange(top, e);
            else if (e > second) second = e;
        }
        plot.symbols.emplace_back(static_cast<float>(1 - second), static_cast<float>(top));
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
