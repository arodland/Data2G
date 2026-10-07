#include "arq/predictor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "cpm/cpm.hpp"
#include "dsp/dsp.hpp"
#include "waveform/ofdm.hpp"

namespace data2g::arq {

namespace {

// predictor.LOGIT_OFFSETS for the installed model.
constexpr std::pair<std::string_view, double> LOGIT_OFFSETS[] = {{"w48-16qam-r1/2", -1.0}, {"n10-256l-r3/4", -0.5}};

const tables::CapacityTable& table(std::string_view constellation) {
    const auto family = const_family(constellation);
    for (const auto& t : tables::CAPACITY)
        if (t.name == family) return t;
    throw std::out_of_range("no capacity table for " + std::string(constellation));
}

// np.interp, one point.
double interp(double x, std::span<const double> xp, std::span<const double> fp) {
    const std::size_t n = xp.size();
    if (x > xp[n - 1]) return fp[n - 1];
    if (x < xp[0]) return fp[0];
    const std::size_t j = static_cast<std::size_t>(std::upper_bound(xp.begin(), xp.end(), x) - xp.begin()) - 1;
    if (j == n - 1 || xp[j] == x) return fp[j];
    const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    return slope * (x - xp[j]) + fp[j];
}

double sigmoid(double z) { return 1 / (1 + std::exp(-z)); }

}  // namespace

std::string_view const_family(std::string_view name) {
    if (name.starts_with("c64")) return "c64-snr18";
    if (name.starts_with("c256")) return "c256-snr26";
    return name;
}

double capacity(double snr_db, std::string_view constellation) {
    return interp(snr_db, tables::CAPACITY_GRID, table(constellation).mi);
}

double peak_db(std::string_view mode) {
    for (const auto& p : tables::MODE_PEAK_DB)
        if (p.mode == mode) return p.db;
    throw std::out_of_range("no peak for mode " + std::string(mode));
}

std::array<double, 2> band_span_hz(std::string_view band) {
    if (const auto* g = cpm::grid(band)) return {g->f0 - g->bp, g->f0 + (g->m - 1) * g->rate + g->bp};
    auto f = waveform::band(band).freqs;
    std::sort(f.begin(), f.end());
    const double half = f.size() > 1 ? (f[1] - f[0]) / 2.0 : 25.0;
    return {static_cast<double>(f.front()) - half, static_cast<double>(f.back()) + half};
}

namespace {
double band_level(const std::array<double, 5>& db, std::string_view band) {
    const auto [lo, hi] = band_span_hz(band);
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < NOISE_BANDS_HZ.size(); ++i) {
        const double w = std::max(0.0, std::min(hi, NOISE_BANDS_HZ[i][1]) - std::max(lo, NOISE_BANDS_HZ[i][0]));
        num += w * std::pow(10.0, db[i] / 10);
        den += w;
    }
    if (den <= 0) return dsp::quantile(std::vector<double>(db.begin(), db.end()), 0.5);
    return 10 * std::log10(num / den);
}
}  // namespace

double noise_shift_db(const std::optional<NoiseLevels>& noise, std::string_view measured_band, std::string_view band,
                      double tail_weight, double deadband_db) {
    if (!noise) return 0.0;
    std::array<double, 5> loud{};
    for (std::size_t i = 0; i < loud.size(); ++i) loud[i] = noise->db[i] + noise->tail_db[i];
    const double med = band_level(noise->db, band) - band_level(noise->db, measured_band);
    const double l = band_level(loud, band) - band_level(loud, measured_band);
    const double shift = std::max(0.0, med) + tail_weight * std::max(0.0, l - std::max(0.0, med));
    return shift >= deadband_db ? shift : 0.0;
}

Measured shifted(const Measured& m, double shift_db) {
    if (shift_db == 0.0) return m;
    Measured out = m;
    out.snr_est -= shift_db;
    for (std::size_t i = 0; i < CONSTS.size(); ++i) {
        const auto& t = table(CONSTS[i]);
        const double snr = interp(m.mi[i], t.mi, tables::CAPACITY_GRID);  // the curve is increasing
        out.mi[i] = interp(snr - shift_db, tables::CAPACITY_GRID, t.mi);
    }
    return out;
}

double effective_mi(std::span<const std::complex<double>> h, std::span<const double> var,
                    std::string_view constellation) {
    if (h.size() != var.size() || h.empty()) throw std::invalid_argument("effective_mi: h and var must match");
    const auto& t = table(constellation);
    std::vector<double> mi(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) {
        const double a = std::abs(h[i]);
        const double snr = 10 * std::log10(std::max(a * a / var[i], 1e-6));
        mi[i] = interp(snr, tables::CAPACITY_GRID, t.mi);
    }
    return dsp::pairwise_sum(mi) / static_cast<double>(mi.size());
}

std::vector<double> outcome_inputs(const Measured& m, std::string_view band, double gap, double seconds,
                                   const Prev* prev, std::span<const std::string_view> bands, bool energy) {
    std::vector<double> x = {m.snr_est, std::log(0.05 + m.spread_est), m.delay_est_ms};
    x.insert(x.end(), m.mi.begin(), m.mi.end());
    x.push_back(m.headroom);
    x.push_back(std::log2(m.frames));
    for (const auto b : bands) x.push_back(b == band ? 1.0 : 0.0);
    const Measured& pm = prev ? prev->m : m;
    const std::string_view pb = prev ? std::string_view(prev->band) : band;
    const double age = prev ? prev->age : 0.0;
    x.push_back(prev ? 1.0 : 0.0);
    x.insert(x.end(), pm.mi.begin(), pm.mi.end());
    x.push_back(pm.snr_est);
    x.push_back(std::log(0.05 + pm.spread_est));
    x.push_back(std::log2(1 + age));
    x.push_back(pb == band ? 1.0 : 0.0);
    x.push_back(std::log2(pm.frames));
    x.push_back(gap);
    x.push_back(std::log2(seconds));
    if (energy) {
        const std::array<double, 3> e = m.energy.value_or(std::array<double, 3>{});
        x.insert(x.end(), e.begin(), e.end());
    }
    return x;
}

std::vector<double> outcome_logits(std::span<const double> x, bool gate) {
    const auto& members = gate ? tables::OUTCOME_GATE_MEMBERS : tables::OUTCOME_MEMBERS;
    std::vector<double> psum;
    for (const auto& mem : members) {
        if (x.size() != mem.mean.size()) throw std::invalid_argument("outcome_logits: wrong input size");
        std::vector<double> h(x.size());
        for (std::size_t i = 0; i < h.size(); ++i) h[i] = (x[i] - mem.mean[i]) / mem.std[i];
        for (std::size_t l = 0; l < mem.layers.size(); ++l) {
            const auto& L = mem.layers[l];
            std::vector<double> o(static_cast<std::size_t>(L.out));
            for (int j = 0; j < L.out; ++j) {
                double acc = 0.0;
                for (int i = 0; i < L.in; ++i) acc += h[i] * L.W[static_cast<std::size_t>(i) * L.out + j];
                o[j] = acc + L.b[j];
                if (l + 1 < mem.layers.size()) o[j] = std::tanh(o[j]);
            }
            h = std::move(o);
        }
        if (psum.empty()) psum.assign(h.size(), 0.0);
        for (std::size_t j = 0; j < h.size(); ++j) psum[j] += sigmoid(std::clamp(h[j], -40.0, 40.0));
    }
    for (auto& v : psum) {
        const double p = std::clamp(v / static_cast<double>(members.size()), 1e-9, 1 - 1e-9);
        v = std::log(p / (1 - p));
    }
    return psum;
}

int outcome_index(std::string_view submode) {
    const auto& names = tables::OUTCOME_MODES;
    const auto it = std::find(names.begin(), names.end(), submode);
    return it == names.end() ? -1 : static_cast<int>(it - names.begin());
}

bool outcome_knows(std::string_view submode) { return outcome_index(submode) >= 0; }

std::vector<Outcome> predict_outcome(const Measured& m, std::string_view band, double gap, double seconds,
                                     const Prev* prev, bool gate) {
    const bool energy = gate ? tables::OUTCOME_GATE_ENERGY : tables::OUTCOME_ENERGY;
    auto z = outcome_logits(outcome_inputs(m, band, gap, seconds, prev, tables::OUTCOME_BANDS, energy), gate);
    const std::size_t n = tables::OUTCOME_MODES.size();
    for (const auto& [name, off] : LOGIT_OFFSETS)
        if (const int i = outcome_index(name); i >= 0) z[i] += off;
    std::vector<Outcome> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = {sigmoid(std::clamp(z[i], -40.0, 40.0)), sigmoid(std::clamp(z[n + i], -40.0, 40.0))};
    return out;
}

}  // namespace data2g::arq
