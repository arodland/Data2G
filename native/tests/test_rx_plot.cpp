// The constellation plot's points (app/rx_plot.*), without the GUI: a clean
// CPM burst on the tone view, and which plots are worth showing.

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "app/rx_plot.hpp"
#include "arq/engine.hpp"
#include "arq/modes.hpp"
#include "arq/phy.hpp"
#include "check.hpp"
#include "generated/config.hpp"
#include "tnc/tnc.hpp"

using namespace data2g;

namespace {

// Most symbols' strongest tone holds nearly all the energy.
void test_cpm() {
    check::current_step = "constellation: CPM";
    const auto& m = arq::mode_at("fsk8r50-r1/2");
    arq::TxBurst b{std::string(m.name), {}, 0};
    b.slots.push_back({{7, 0, 0}, 0, arq::Bytes(static_cast<std::size_t>(arq::payload_bytes(m)), 0x55)});
    b.slots.push_back({{7, 0, 1}, 0, arq::Bytes(static_cast<std::size_t>(arq::payload_bytes(m)), 0xa3)});
    std::vector<double> y(config::FS / 2, 0.0);
    const auto x = arq::tx_audio(b);
    y.insert(y.end(), x.begin(), x.end());
    y.insert(y.end(), config::FS / 2, 0.0);
    auto rx = tnc::receive_any(y, config::FS);
    check::is_true(rx.has_value(), "the CPM burst received");
    if (!rx) return;
    arq::BurstHeard heard;
    heard.submode = b.submode;
    heard.heard = arq::heard_of(std::move(*rx));
    const auto plot = app::rx_constellation(heard);
    check::is_true(plot && plot->tones && !plot->symbols.empty(), "a tone plot");
    if (!plot) return;
    std::size_t clean = 0;
    for (const auto& z : plot->symbols) clean += z.real() > 0.9f && z.imag() > 0.8f;
    check::is_true(clean > plot->symbols.size() * 9 / 10, "symbols near the ideal (1, 1)");
    std::size_t wedge = 0;  // shares sum to 1: under the diagonal, over the tie line
    for (const auto& z : plot->symbols) wedge += z.imag() <= z.real() + 1e-6f && z.imag() >= 1 - z.real() - 1e-6f;
    check::equal(wedge, plot->symbols.size(), "every symbol between the diagonal and the tie line");
}

void test_visible() {
    check::current_step = "constellation: visible";
    app::RxConstellation c;
    c.points = {{1, 1}, {-1, -1}};
    check::is_true(!app::plot_visible(c), "no symbols: nothing to show");
    c.symbols = {{3e7f, -2e7f}, {std::nanf(""), 0}};
    check::is_true(!app::plot_visible(c), "a broken estimate's symbols, all off the plot");
    c.symbols.push_back({0.9f, -1.1f});
    check::is_true(app::plot_visible(c), "one on the plot is enough");
}

}  // namespace

int main() {
    test_cpm();
    test_visible();
    return check::report("rx_plot");
}
