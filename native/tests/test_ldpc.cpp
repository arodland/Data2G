// QC-LDPC checks that need no Python: codewords satisfy H, decoding
// clean and noisy words, early stop, phi's accuracy, reentrancy. Parity with
// data2g/ldpc.py is tests/test_native_ldpc.py's job.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "check.hpp"
#include "ldpc/ldpc.hpp"

using namespace data2g;

namespace {

Mat<std::uint8_t> random_bits(std::size_t rows, int k, std::mt19937& rng) {
    Mat<std::uint8_t> m(rows, k);
    for (auto& b : m.data) b = rng() & 1;
    return m;
}

// BPSK + AWGN LLRs, 2y / sigma^2.
Mat<float> noisy(const Mat<std::uint8_t>& cw, double sigma, std::mt19937& rng) {
    std::normal_distribution<double> g(0.0, sigma);
    Mat<float> llr(cw.rows, cw.cols);
    for (std::size_t i = 0; i < cw.data.size(); ++i)
        llr.data[i] = static_cast<float>(2 * ((1.0 - 2.0 * cw.data[i]) + g(rng)) / (sigma * sigma));
    return llr;
}

bool rows_equal(const Mat<std::uint8_t>& a, const Mat<std::uint8_t>& b, std::size_t r) {
    return std::equal(a[r], a[r] + a.cols, b[r]);
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(300, "ldpc");
    std::mt19937 rng(1);

    check::current_step = "every LDPC submode";
    for (const auto& s : config::SUBMODES) {
        if (s.code != "ldpc") continue;
        const std::string name(s.name);
        const auto code = ldpc::qc_code(s.k, s.coded_bits);
        const auto bits = random_bits(3, s.k, rng);
        const auto full = code.encode_full(bits);
        check::is_true(std::ranges::all_of(code.syndrome_ok(full), [](auto v) { return v == 1; }), name + " H c = 0");
        auto broken = full;
        broken[1][5] ^= 1;
        const auto bad = code.syndrome_ok(broken);
        check::is_true(bad[0] && !bad[1] && bad[2], name + " one flipped bit breaks H");
        const auto cw = code.encode(bits);
        check::equal(static_cast<int>(cw.cols), s.coded_bits, name + " n");
        bool systematic = true;
        for (int j = 0; j < s.coded_bits && code.sent()[j] < s.k; ++j)
            systematic = systematic && cw[0][j] == bits[0][code.sent()[j]];
        check::is_true(systematic, name + " sent info bits are the info bits");
        const auto mother = code.mother();
        const auto mcw = mother.encode(bits);
        check::is_true(std::ranges::all_of(mother.syndrome_ok(mother.encode_full(bits)), [](auto v) { return v == 1; }),
                       name + " mother H c = 0");
        bool prefix = true;
        for (std::size_t r = 0; r < bits.rows; ++r) prefix = prefix && std::equal(cw[r], cw[r] + cw.cols, mcw[r]);
        check::is_true(prefix, name + " mother code starts with the codeword");

        ldpc::Decoder dec(code);
        Mat<float> clean(cw.rows, cw.cols);
        for (std::size_t i = 0; i < cw.data.size(); ++i) clean.data[i] = cw.data[i] ? -8.0f : 8.0f;
        const auto r = dec.decode(clean, 40);
        check::is_true(r.ok == std::vector<std::uint8_t>(3, 1) && r.bits.data == bits.data, name + " decodes clean words");
    }

    check::current_step = "noisy decode";
    {
        // As test_ldpc.py: rate 1/2, BPSK at Es/N0 ~3 dB.
        const auto code = ldpc::qc_code(500, 1000);
        const ldpc::Decoder dec(code);
        const auto bits = random_bits(8, 500, rng);
        auto llr = noisy(code.encode(bits), 0.7, rng);
        for (const auto& alpha : {std::vector<float>{}, std::vector<float>{0.8f}}) {
            const auto r = dec.decode(llr, 30, alpha, true);
            check::is_true(r.ok == std::vector<std::uint8_t>(8, 1) && r.bits.data == bits.data,
                           alpha.empty() ? "BP corrects noise" : "min-sum corrects noise");
            check::equal(r.posterior.cols, std::size_t{1000}, "posterior per sent bit");
        }

        // One hopeless row: the batch runs every iteration, the others
        // still decode, and that row reports failure.
        std::normal_distribution<float> g(0.0f, 1.0f);
        for (std::size_t j = 0; j < llr.cols; ++j) llr[3][j] = g(rng);
        const auto r = dec.decode(llr, 40);
        bool others = true;
        for (std::size_t b = 0; b < 8; ++b)
            if (b != 3) others = others && r.ok[b] && rows_equal(r.bits, bits, b);
        check::is_true(others && !r.ok[3], "a failed row does not disturb the rest");

        // Reentrant: two threads at once give what one gives.
        ldpc::Decoded a, b;
        std::thread t1([&] { a = dec.decode(llr, 40, {}, true); });
        std::thread t2([&] { b = dec.decode(llr, 40, {}, true); });
        t1.join();
        t2.join();
        check::is_true(a.bits.data == r.bits.data && b.bits.data == r.bits.data && a.ok == r.ok &&
                       a.posterior.data == b.posterior.data, "concurrent decodes agree");
    }

    check::current_step = "phi";
    {
        // Against the same float32 pipeline through libm in double: each
        // stage correctly rounded but in rare double-rounding cases.
        std::vector<float> x;
        for (float v = 1e-9f; v < 32.0f; v = std::nextafter(v * 1.0007f, 64.0f)) x.push_back(v);
        x.push_back(0.0f);
        x.push_back(1e4f);
        auto got = x;
        ldpc::phi(got);
        int exact = 0, far = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const float c = std::fmin(std::fmax(x[i], 1e-7f), 30.0f);
            const float t = static_cast<float>(std::tanh(static_cast<double>(c * 0.5f)));
            const float want = static_cast<float>(-std::log(static_cast<double>(t)));
            exact += got[i] == want;
            const auto ulps = std::bit_cast<std::int32_t>(got[i]) - std::bit_cast<std::int32_t>(want);
            far += std::abs(ulps) > 1;
        }
        check::equal(far, 0, "phi within an ULP of the libm pipeline");
        check::is_true(exact >= static_cast<int>(x.size()) - static_cast<int>(x.size()) / 10000,
                       "phi exact in all but 1 case in 10^4 (" + std::to_string(x.size() - exact) + " of " +
                           std::to_string(x.size()) + " differ)");
        check::equal(got[got.size() - 2], 16.811243f, "phi floor at x = 1e-7");
        check::is_true(got.back() == 0.0f, "phi(BIG) = 0");
    }

    check::current_step = "errors";
    {
        bool threw = false;
        try {
            ldpc::qc_code(8000, 24000);
        } catch (const std::out_of_range&) {
            threw = true;
        }
        check::is_true(threw, "no shift table: out_of_range");
        check::equal(ldpc::layout(2400, 3840).second, 240, "layout of w48-256l-r5/8");
    }
    return check::report("ldpc");
}
