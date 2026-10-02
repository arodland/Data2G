// The burst modem with no Python: known answers from the reference (CRC-6,
// header bits, Accept caps), header round trips, and loopbacks through
// modulate -> find_burst / receive / demodulate, the header copy included.
// Parity: tests/test_native_modem.py.

#include <algorithm>
#include <random>
#include <string>

#include "check.hpp"
#include "codes/codes.hpp"
#include "modem/modem.hpp"

using namespace data2g;

namespace {

std::vector<std::vector<std::uint8_t>> payloads(const config::Submode& s, int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<std::vector<std::uint8_t>> out(static_cast<std::size_t>(n));
    for (auto& p : out)
        for (int i = 0; i < s.payload_bytes; ++i) p.push_back(static_cast<std::uint8_t>(rng() & 0xFF));
    return out;
}

// pad, burst, pad, plus white noise at `sigma` per sample
std::vector<double> on_air(const std::vector<double>& x, std::size_t pad, double sigma, unsigned seed) {
    std::vector<double> y(pad, 0.0);
    y.insert(y.end(), x.begin(), x.end());
    y.insert(y.end(), pad, 0.0);
    std::mt19937 rng(seed);
    std::normal_distribution<double> n(0.0, sigma);
    for (double& v : y) v += n(rng);
    return y;
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(300, "test_modem");

    check::current_step = "header";
    check::equal(modem::crc6(0), 15, "crc6(0)");
    check::equal(modem::crc6(100), 37, "crc6(100)");
    check::equal(modem::crc6(1023), 62, "crc6(1023)");
    const std::vector<std::uint8_t> want{1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1};
    const auto hb = modem::header_bits(7, 3, "w");
    check::is_true(std::vector<std::uint8_t>(hb.begin(), hb.begin() + 16) == want, "header_bits(7, 3) first 16");
    for (auto band : modem::SYNC_BANDS) {
        for (const auto& s : config::SUBMODES) {
            if (s.sync_band != band) continue;
            for (int n : {1, 64}) {
                std::vector<double> soft;
                for (auto b : modem::header_bits(s.index, n, band)) soft.push_back(1.0 - 2.0 * b);
                const auto h = modem::decode_header(soft, band);
                check::is_true(h.spec == &s && h.n_cw == n && h.score > 0.999, std::string(s.name) + " header roundtrip");
            }
        }
    }
    const auto acc = modem::Accept::of({}, 4.0);
    check::equal(acc.max_cw[0].second, 6, "Accept.of(None, 4): ack-4f cap");
    check::equal(acc.max_cw[1].second, 3, "Accept.of(None, 4): polar-k96-f8 cap");
    check::is_true(acc.bands() == std::vector<std::string_view>{"n10", "w", "w48"}, "Accept bands sorted");

    check::current_step = "loopback";
    for (const char* name : {"ack-1f", "qpsk-r1/2", "n4-ack-2f", "n10-qpsk-r1/3", "w48-16qam-r1/2", "w48-64l-r1/2"}) {
        const auto& s = *codes::submode(name);
        const auto sent = payloads(s, 2, 7);
        const auto y = on_air(modem::modulate(sent, s), 3000, 0.05, 3);
        const auto b = modem::demodulate(y);
        check::is_true(b.spec == &s && b.payloads == sent && b.crc_ok == std::vector<bool>{true, true},
                       std::string(name) + " loopback");
        // the streaming search: from a prefix holding the head
        const std::size_t head = 3000 + config::LEADIN_SAMPLES + modem::head_samples(s.sync_band) + 2 * config::NSYM;
        const auto lock = modem::find_burst(std::span(y).first(head));
        check::is_true(lock.spec == &s && lock.n_cw == 2, std::string(name) + " find_burst");
        check::equal(lock.end, modem::burst_end(lock.p0, s, 2), std::string(name) + " burst end");
        check::is_true(lock.start >= 3000 + config::LEADIN_SAMPLES - 40 && lock.start <= 3000 + config::LEADIN_SAMPLES + 40,
                       std::string(name) + " preamble start");
        const auto c = modem::pilot_coherence(y, lock, 8);
        check::is_true(c.size() >= 2 && c[0] > 0.9, std::string(name) + " pilots coherent");
    }

    check::current_step = "header copy";
    {
        const auto& s = *codes::submode("qpsk-r1/5");
        const auto sent = payloads(s, 1, 9);
        auto x = modem::modulate(sent, s);
        const int wipe = modem::preamble_samples(modem::band("w")) + modem::header_samples("w");
        std::fill(x.begin() + config::LEADIN_SAMPLES, x.begin() + config::LEADIN_SAMPLES + wipe, 0.0);
        const auto y = on_air(x, 2000, 0.05, 5);
        bool threw = false;
        try {
            modem::receive(y);
        } catch (const modem::SyncError&) {
            threw = true;
        }
        check::is_true(threw, "no preamble: receive fails");
        double peak = 0;
        const auto lock = modem::find_copy(y, "w", nullptr, nullptr, {}, &peak);
        check::is_true(lock.has_value() && lock->spec == &s && lock->copy.has_value() && peak > modem::COPY_DETECT,
                       "find_copy locks");
        if (lock) {
            const auto b = modem::decode_received(modem::receive(y, {}, nullptr, {}, &*lock));
            check::is_true(b.payloads == sent && b.crc_ok[0], "copy lock decodes");
        }
    }

    check::current_step = "noise";
    {
        const auto y = on_air(std::vector<double>(16000, 0.0), 0, 1.0, 11);
        bool threw = false;
        try {
            modem::find_burst(y);
        } catch (const modem::SyncError&) {
            threw = true;
        }
        check::is_true(threw, "noise: no burst");
        check::is_true(!modem::find_copy(y, "w48").has_value(), "noise: no copy lock");
    }

    return check::report("test_modem");
}
