// waveform/{ofdm,dsp,sync} and dsp/: properties that need no Python.
// Parity with data2g/waveform is tests/test_native_waveform.py's job.
// Prints one live-receiver hop's StreamDetector time per band.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "check.hpp"
#include "dsp/dsp.hpp"
#include "waveform/dsp.hpp"
#include "waveform/ofdm.hpp"
#include "waveform/sync.hpp"

using namespace data2g;
using waveform::cdouble;

namespace {

std::vector<cdouble> noise(std::mt19937_64& rng, std::size_t n, double sigma) {
    std::normal_distribution<double> g(0.0, sigma);
    std::vector<cdouble> z(n);
    for (auto& v : z) v = {g(rng), g(rng)};
    return z;
}

// a preamble at `lead`, shifted by f_hz, in silence, at baseband
std::vector<cdouble> preamble_at(const waveform::Band& b, std::size_t lead, double f_hz, std::size_t len) {
    std::vector<double> x(len);
    const auto w = b.preamble_waveform();
    for (std::size_t i = 0; i < w.size(); ++i) x[lead + i] = w[i];
    auto z = waveform::to_baseband(x);
    return waveform::freq_correct(z, -f_hz);
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(300, "test_waveform");
    std::mt19937_64 rng(1);
    constexpr double PI = std::numbers::pi;

    check::current_step = "numpy primitives";
    check::equal(dsp::pairwise_sum(std::vector<double>{1, 2, 3}), 6.0, "pairwise_sum");
    check::equal(dsp::quantile({5, 1, 4, 2, 3}, 0.2), 1.8, "quantile linear");
    check::equal(dsp::quantile({7}, 0.2), 7.0, "quantile of one");
    check::equal(dsp::next_fast_len(641), std::size_t{648}, "next_fast_len");
    {
        const auto h = dsp::firwin_bandpass(201, 850, 2200, config::FS);
        double sym = 0, gain = 0;
        for (std::size_t i = 0; i < h.size(); ++i) {
            sym = std::max(sym, std::abs(h[i] - h[h.size() - 1 - i]));
            gain += h[i] * std::cos(PI * (static_cast<double>(i) - 100) * (1525.0 / 4000));
        }
        check::is_true(sym < 1e-15, "firwin symmetric (to rounding: linspace is not)");
        check::is_true(std::abs(gain - 1) < 1e-12, "firwin unit gain at the band centre");
    }
    {
        std::vector<double> x(800);
        std::vector<cdouble> want(800);
        for (std::size_t n = 0; n < x.size(); ++n) {
            x[n] = std::cos(2 * PI * 50.0 * static_cast<double>(n) / 800);
            want[n] = std::polar(1.0, 2 * PI * 50.0 * static_cast<double>(n) / 800);
        }
        check::close(dsp::hilbert(x), want, 1e-12, "hilbert of a whole-period cosine");
    }

    check::current_step = "ofdm";
    check::close(std::vector{waveform::phasor(2000)}, std::vector{cdouble{0, 1}}, 1e-15, "phasor quarter turn");
    check::close(std::vector{waveform::phasor(-8001)}, std::vector{waveform::phasor(7999)}, 0.0, "phasor reduces exactly");
    check::equal(waveform::band("w").freqs.front(), std::int64_t{950}, "w first carrier");
    check::equal(waveform::band("w48").nc(), 48, "w48 carriers");
    check::equal(waveform::band("n4").preamble_samples(), 1344, "preamble samples");
    for (const auto& spec : config::BANDS) {
        const auto& b = waveform::band(spec.name);
        const std::string name(spec.name);
        // modulate -> baseband -> demodulate recovers the symbols
        Mat<cdouble> s(6, static_cast<std::size_t>(b.nc()));
        std::uniform_real_distribution<double> u(0, 1);
        for (auto& v : s.data) v = std::polar(1.0, 2 * PI * u(rng));
        const auto z = waveform::to_baseband(b.modulate_symbols(s));
        double err = 0;
        for (std::size_t i = 1; i < 5; ++i) {
            const auto got = b.demod_window(z, static_cast<std::int64_t>(i) * config::NSYM + config::NCP, 6);
            // a 6-sample backoff turns each carrier by its baseband frequency
            for (std::size_t k = 0; k < got.size(); ++k)
                err = std::max(err, std::abs(got[k] * waveform::phasor(6 * b.bb[k]) - s[i][k]));
        }
        check::is_true(err < 1e-9, name + " symbol loopback");
        const auto w = b.preamble_waveform();
        double per = 0;
        for (std::size_t n = config::M; n < w.size(); ++n) per = std::max(per, std::abs(w[n] - w[n - config::M]));
        check::is_true(per < 1e-9, name + " preamble M-periodic");
    }

    check::current_step = "dsp";
    {
        std::vector<double> x(5000);
        std::normal_distribution<double> g;
        for (auto& v : x) v = g(rng);
        const auto whole = waveform::to_baseband(x);
        const auto part = waveform::to_baseband(std::span(x).subspan(1234), 1234);
        check::close(part, std::vector<cdouble>(whole.begin() + 1234, whole.end()), 0.0, "to_baseband n0 continues the phase");
        const auto back = waveform::freq_correct(waveform::freq_correct(whole, 37.3), -37.3);
        check::close(back, whole, 1e-12, "freq_correct inverts");
        // the clipper: unit RMS over the active part, lower PAPR
        const auto& b = waveform::band("w");
        Mat<cdouble> s(20, 24);
        std::uniform_real_distribution<double> u(0, 1);
        for (auto& v : s.data) v = std::polar(1.0, 2 * PI * u(rng));
        std::vector<double> tx(800, 0.0);
        const auto body = b.modulate_symbols(s);
        tx.insert(tx.end(), body.begin(), body.end());
        tx.insert(tx.end(), 800, 0.0);
        const auto y = waveform::tx_condition(tx, 1.0, config::CLIP_OVERSHOOT, 800, tx.size() - 800);
        double ms = 0;
        for (std::size_t i = 800; i < y.size() - 800; ++i) ms += y[i] * y[i];
        check::is_true(std::abs(ms / static_cast<double>(y.size() - 1600) - 1) < 1e-12, "tx_condition unit RMS");
        check::is_true(waveform::papr_db(y) < waveform::papr_db(tx) - 3, "tx_condition cuts PAPR");
        int calls = 0;
        waveform::tx_condition(tx, 1.0, config::CLIP_OVERSHOOT, 800, tx.size() - 800, b.tx_bandpass(),
                               [&](std::span<const double> v) { ++calls; return std::vector<double>(v.begin(), v.end()); },
                               std::vector<double>{1.0});
        check::equal(calls, 3, "ACE projector after each overshoot pass, not the closing ones");
    }

    check::current_step = "acquire";
    for (const char* name : {"w", "n10", "w48"}) {  // n4 bursts sync on n10
        const auto& b = waveform::band(name);
        for (double f : {0.0, 6.0, -37.5, 143.0}) {
            auto z = preamble_at(b, 3000, f, 9000);
            const auto n = noise(rng, z.size(), 0.01);
            for (std::size_t i = 0; i < z.size(); ++i) z[i] += n[i];
            const auto acq = waveform::acquire(z, b);
            const std::string what = std::string(name) + " at " + std::to_string(f) + " Hz";
            check::is_true(std::abs(acq.preamble_start - 3000) <= (b.nc() >= 24 ? 1 : 4), what + ": timing");
            check::is_true(std::abs(acq.freq_offset - f) < 0.5, what + ": CFO");
            check::is_true(acq.metric > b.preamble_threshold(), what + ": metric");
        }
    }
    {
        bool threw = false;
        try {
            waveform::acquire(noise(rng, 32000, 1.0), waveform::band("w"));
        } catch (const waveform::SyncError&) {
            threw = true;
        }
        check::is_true(threw, "noise alone: SyncError");
    }

    check::current_step = "StreamDetector";
    {
        const auto& b = waveform::band("n10");
        const auto z = noise(rng, 12000, 1.0);
        const auto whole = waveform::raw_stat(z, b);
        waveform::StreamDetector d(b);
        for (std::size_t a = 0; a < z.size(); a += 1777) d.feed(std::span(z).subspan(a, std::min<std::size_t>(1777, z.size() - a)));
        bool same = d.s0 == 0 && d.S(0).size() == whole.S.cols;
        double worst = 0;
        for (std::size_t i = 0; same && i < d.bins(); ++i)
            for (std::size_t j = 0; j < whole.S.cols; ++j) worst = std::max(worst, std::abs(d.S(i)[j] - whole.S[i][j]));
        check::is_true(same && worst < 1e-9, "chunked statistic equals the whole signal's");
    }
    // one live-receiver hop (tnc: FS / 4 new samples, ~2.5 s kept): feed, trim, stat
    for (const char* name : {"w", "n10", "w48"}) {
        waveform::StreamDetector d(waveform::band(name));
        const auto z = noise(rng, 2000 * 200, 1.0);
        std::vector<double> times;
        for (std::size_t k = 0; k < 200; ++k) {
            const auto t0 = std::chrono::steady_clock::now();
            d.feed(std::span(z).subspan(k * 2000, 2000));
            const auto fed = static_cast<std::int64_t>((k + 1) * 2000);
            d.trim(fed - 20000);
            const auto S = d.stat(fed - 18000, fed - d.span + 1);
            times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            check::is_true(!S.data.empty(), "hop stat");
        }
        std::sort(times.begin() + 20, times.end());
        std::printf("StreamDetector hop, %s: %.2f ms median\n", name, times[110]);
    }
    return check::report("test_waveform");
}
