// gfldpc:: on every CPM data code: the field, the encoder against the
// checks, a noisy decode (Rayleigh per tone, as scripts/gf_study.py's
// ideal channel), and decode time per codeword (printed: the CPU budget a
// burst's data costs; DATA2G_GF_BENCH=1 for more codewords).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "check.hpp"
#include "codes/codes.hpp"
#include "gfldpc/gfldpc.hpp"
#include "tables/cpm.hpp"

using namespace data2g;

namespace {

// B codewords of `code` through noncoherent M-FSK at es (Rayleigh per tone, N0 = 1) -> the square-law
// symbol log-likelihoods (the genie's: es known), value v on tone v (the mapping does not matter here).
Mat<float> channel(const gfldpc::Code& code, const Mat<std::uint8_t>& x, double es_db, std::mt19937& rng) {
    const int q = code.q;
    const double es = std::pow(10.0, es_db / 10.0), g = es / (1 + es);
    std::normal_distribution<double> nd(0.0, std::sqrt(0.5));
    Mat<float> ll(x.rows, x.cols * q);
    for (std::size_t b = 0; b < x.rows; ++b)
        for (std::size_t i = 0; i < x.cols; ++i) {
            const double hr = nd(rng) * std::sqrt(es), hi = nd(rng) * std::sqrt(es);
            for (int a = 0; a < q; ++a) {
                double re = nd(rng), im = nd(rng);
                if (a == x[b][i]) re += hr, im += hi;
                ll[b][i * q + a] = static_cast<float>((re * re + im * im) * g);
            }
        }
    return ll;
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    for (int p : {3, 4, 5}) {
        const gfldpc::GF gf(p);
        bool ok = true;
        for (int a = 1; a < gf.q; ++a) ok = ok && gf.mul[a * gf.q + gf.inv[a]] == 1;
        check::is_true(ok, "GF(2^" + std::to_string(p) + ") inverses");
    }
    const bool bench = std::getenv("DATA2G_GF_BENCH") != nullptr;
    std::mt19937 rng(7);
    for (const auto& t : tables::CPM_SPECS) {
        const auto& s = codes::spec(t);
        const std::string name(s.name);
        const auto& mother = codes::gf_code(s);
        const auto code = mother.cut(codes::n_sym(s));
        check::equal(code.k(), mother.k(), name + " the mother keeps the information set");
        check::equal(code.k(), (s.k + s.bits_per_cu - 1) / s.bits_per_cu, name + " k");
        const int B = bench ? 200 : 20;
        Mat<std::uint8_t> u(static_cast<std::size_t>(B), static_cast<std::size_t>(code.k()));
        for (auto& v : u.data) v = static_cast<std::uint8_t>(rng() % code.q);
        const auto x = mother.encode(u);
        bool sy = true;
        for (int b = 0; b < B; ++b) sy = sy && mother.syndrome_ok(x[b]);
        check::is_true(sy, name + " encode satisfies every check");
        Mat<std::uint8_t> xc(x.rows, static_cast<std::size_t>(code.n));
        for (std::size_t b = 0; b < x.rows; ++b) std::copy_n(x[b], code.n, xc[b]);
        const auto& dec = codes::gf_decoder(s);
        // 3 dB over the r1/2 codes' 10% Rayleigh point (gf_study): nearly every codeword decodes
        const double good = s.k == 320 ? 9.5 : 12.0, hopeless = 0.0;
        const auto d = dec.decode(channel(code, xc, good, rng), codes::GF_ITERS);
        int right = 0;
        for (int b = 0; b < B; ++b) right += d.ok[b] && std::equal(xc[b], xc[b] + code.n, d.hard[b]);
        check::is_true(right >= B * 9 / 10, name + " decodes at " + std::to_string(good) + " dB: " + std::to_string(right));
        for (double es : {good, hopeless}) {  // decode time per codeword: converging, and every iteration run
            const auto ll = channel(code, xc, es, rng);
            const auto t0 = std::chrono::steady_clock::now();
            const auto r = dec.decode(ll, codes::GF_ITERS);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / B;
            int conv = 0;
            for (auto o : r.ok) conv += o;
            std::printf("%-15s %5.1f dB: %6.2f ms a codeword (%d of %d converged)\n", name.c_str(), es, ms, conv, B);
        }
    }
    return check::report("gfldpc");
}
