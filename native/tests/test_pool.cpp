// The shared pool (core/util/pool) changes timing, never results: batched
// LDPC (soft values included) and polar decodes, the CFO-grid sync, the DD
// re-estimate and a whole receive + DD decode, bit for bit at pool sizes 1,
// 2, 4 and 8. Also the pool itself: every index once, exceptions, nesting,
// two callers at once. Run under TSan too (docs/native-port-plan.md).

#include <cstring>
#include <random>
#include <thread>
#include <algorithm>

#include "arq/phy.hpp"
#include "check.hpp"
#include "equalizer/equalizer.hpp"
#include "util/pool.hpp"
#include "waveform/sync.hpp"

using namespace data2g;

namespace {

template <typename T>
std::string bits(const std::vector<T>& v) {
    std::string s(v.size() * sizeof(T), '\0');
    if (!v.empty()) std::memcpy(s.data(), v.data(), s.size());
    return s;
}

template <typename T>
std::string bits(const Mat<T>& m) {
    return bits(m.data);
}

// Everything the pool touches, as bytes per part.
std::vector<std::string> run(const std::vector<double>& y_fail, const std::vector<double>& y_mixed) {
    std::vector<std::string> out;
    std::mt19937 rng(5);
    std::normal_distribution<float> n01(0.0f, 1.0f);

    // LDPC: 64 codewords, sum-product with posteriors and min-sum, half
    // converging (scaled noise around +-8), half pure noise
    const auto& big = codes::spec(*codes::submode("w48-16qam-r1/2"));
    Mat<float> llr(64, static_cast<std::size_t>(big.coded_bits));
    for (std::size_t b = 0; b < llr.rows; ++b)
        for (std::size_t j = 0; j < llr.cols; ++j) llr[b][j] = b % 2 ? 2.0f * n01(rng) : 3.0f + 2.5f * n01(rng);
    const auto& dec = codes::ldpc_decoder(big);
    const auto bp = dec.decode(llr, codes::ITERS, {}, true);
    out.push_back(bits(bp.bits) + bits(bp.ok) + bits(bp.posterior));
    const float alpha = 0.75f;
    const auto ms = dec.decode(llr, codes::ITERS, std::span(&alpha, 1), true);
    out.push_back(bits(ms.bits) + bits(ms.ok) + bits(ms.posterior));

    // polar: 16 rows
    const auto& pol = codes::spec(*codes::submode("polar-k192-f8"));
    Mat<float> pl(16, static_cast<std::size_t>(pol.coded_bits));
    for (auto& v : pl.data) v = 1.0f + 2.0f * n01(rng);
    const auto sr = codes::polar_decoder(pol).decode(pl);
    out.push_back(bits(sr.paths) + bits(sr.metric));

    // sync on a signal long enough for the pool
    const auto& band = waveform::band("w48");
    std::vector<waveform::cdouble> z(9000);
    for (auto& v : z) v = {n01(rng), n01(rng)};
    const auto rs = waveform::raw_stat(z, band, config::ACQUIRE_REACH_HZ, 0, std::nullopt, true);
    out.push_back(bits(rs.S) + bits(rs.q) + bits(rs.outs));

    // equalizer::refine on random observations, some carriers unknown
    {
        const auto bb = equalizer::bb(*band.spec);
        const std::size_t nc = bb.size(), F = 12, S = 5, P = F + 1;
        Mat<equalizer::cd> hp(P, nc), zz(F * S, nc);
        Mat<double> w(F * S, nc), t_rows(F, S);
        std::vector<double> t_pilot(P);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        for (auto& v : hp.data) v = {n01(rng), n01(rng)};
        for (auto& v : zz.data) v = {n01(rng), n01(rng)};
        for (auto& v : w.data) v = u01(rng) < 0.2 ? 0.0 : 1.0 + 4.0 * u01(rng);
        for (std::size_t r = 0; r < nc; ++r) w[7][r] = 0.0;  // a row with nothing known
        for (std::size_t i = 0; i < P; ++i) t_pilot[i] = static_cast<double>(i) * equalizer::FRAME_S;
        for (std::size_t f = 0; f < F; ++f)
            for (std::size_t s = 0; s < S; ++s) t_rows[f][s] = (static_cast<double>(f) + (s + 1) / 6.0) * equalizer::FRAME_S;
        const auto [h, mse] = equalizer::refine(hp, t_pilot, zz, w, t_rows, {-3, 5}, 1.0, 1.5, 0.1, bb);
        out.push_back(bits(h) + bits(mse));
    }

    // a whole-buffer receive, then every slot decoded with DD, and the
    // soft bits stored for the ones that fail
    for (const auto* y : {&y_fail, &y_mixed}) {
        const arq::Heard heard{std::make_shared<const modem::Received>(modem::receive(*y)), nullptr};
        const auto& r = *heard.ofdm;
        std::string soft;
        const auto slots = arq::soft_bits(heard);
        for (const auto& s : *slots) soft += bits(s);
        out.push_back(bits(r.raw) + bits(r.est.h) + bits(r.est.mse) + bits(r.hp) + soft);
        arq::SoftStore store;
        arq::ModemRx rx(heard, &store, std::nullopt, nullptr, true);
        std::string flags, got;
        for (int i = 0; i < rx.n_cw(); ++i) {
            const arq::SoftKey k{false, 0, i, 0};
            const auto p = rx.decode(i, {7, 0, i}, 0, &k);
            flags += p ? "1" : "0";
            if (p) got += std::string(p->begin(), p->end());
        }
        for (const auto& [k, e] : store) got += bits(e.buf);
        out.push_back(flags + got);
    }
    return out;
}

std::vector<double> on_air(const std::vector<double>& x, double sigma, unsigned seed) {
    std::vector<double> y(2400, 0.0);
    y.insert(y.end(), x.begin(), x.end());
    y.insert(y.end(), 2400, 0.0);
    std::mt19937 rng(seed);
    std::normal_distribution<double> n(0.0, sigma);
    for (double& v : y) v += n(rng);
    return y;
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(600, "test_pool");

    check::current_step = "pool basics";
    for (int n : {1, 3, 8}) {
        pool::set_threads(n);
        check::equal(pool::threads(), n, "set_threads");
        std::vector<int> hit(1000, 0);
        pool::parallel_for(hit.size(), [&](std::size_t i) { ++hit[i]; });
        check::is_true(std::all_of(hit.begin(), hit.end(), [](int v) { return v == 1; }), "every index once");
        bool threw = false;
        try {
            pool::parallel_for(100, [](std::size_t i) {
                if (i == 37) throw std::runtime_error("37");
            });
        } catch (const std::runtime_error& e) {
            threw = std::string(e.what()) == "37";
        }
        check::is_true(threw, "an exception reaches the caller");
        std::vector<int> nested(64, 0);
        pool::parallel_for(8, [&](std::size_t i) { pool::parallel_for(8, [&](std::size_t j) { ++nested[i * 8 + j]; }); });
        check::is_true(std::all_of(nested.begin(), nested.end(), [](int v) { return v == 1; }), "nested runs inline");
        std::vector<int> a(500, 0), b(500, 0);
        std::thread other([&] {
            for (int k = 0; k < 20; ++k) pool::parallel_for(b.size(), [&](std::size_t i) { ++b[i]; });
        });
        for (int k = 0; k < 20; ++k) pool::parallel_for(a.size(), [&](std::size_t i) { ++a[i]; });
        other.join();
        check::is_true(std::all_of(a.begin(), a.end(), [](int v) { return v == 20; }) &&
                           std::all_of(b.begin(), b.end(), [](int v) { return v == 20; }),
                       "two callers at once");
    }

    check::current_step = "bitwise across pool sizes";
    const std::string name = "w48-qpsk-r1/2";
    const auto& spec = *codes::submode(name);
    arq::TxBurst burst{name, {}, 0};
    for (int i = 0; i < 16; ++i) {
        arq::Bytes p(static_cast<std::size_t>(spec.payload_bytes));
        for (std::size_t j = 0; j < p.size(); ++j) p[j] = static_cast<std::uint8_t>(i * 31 + j);
        burst.slots.push_back({{7, 0, i}, 0, p});
    }
    const auto x = arq::tx_audio(burst);
    const auto y_fail = on_air(x, 1.6, 11), y_mixed = on_air(x, 0.97, 12);
    pool::set_threads(1);
    const auto want = run(y_fail, y_mixed);
    for (int n : {2, 4, 8}) {
        pool::set_threads(n);
        const auto got = run(y_fail, y_mixed);
        for (std::size_t i = 0; i < want.size(); ++i)
            check::is_true(got[i] == want[i], "part " + std::to_string(i) + " at " + std::to_string(n) + " threads");
    }
    // the DD part must have had something to do: some slots fail, some decode
    const auto flags = want.back().substr(0, 16), none = want[want.size() - 3].substr(0, 16);
    check::is_true(flags.find('0') != std::string::npos && flags.find('1') != std::string::npos,
                   "mixed burst: some slots decode, DD runs on the rest (" + flags + ")");
    check::equal(none, std::string(16, '0'), "failing burst: every slot fails");
    return check::report("test_pool");
}
