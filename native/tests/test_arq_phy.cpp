// Known answers and Python-free round trips through the modem: phy's
// masked decodes, resend combining, the DD budget's clock, and two KISS
// links shifting modes from each other's reports. Parity with
// data2g/arq/phy.py and data2g/kisslink.py is tests/test_native_arq_phy.py's.

#include <algorithm>
#include <random>

#include "arq/phy.hpp"
#include "check.hpp"
#include "kisslink/kisslink.hpp"

using namespace data2g;
using namespace data2g::arq;

namespace {

std::vector<double> on_air(const std::vector<double>& x, double sigma, unsigned seed) {
    std::vector<double> y(2400, 0.0);
    y.insert(y.end(), x.begin(), x.end());
    y.insert(y.end(), 2400, 0.0);
    std::mt19937 rng(seed);
    std::normal_distribution<double> n(0.0, sigma);
    for (double& v : y) v += n(rng);
    return y;
}

Heard hear(const std::vector<double>& y) {
    return Heard{std::make_shared<const modem::Received>(modem::receive(y)), nullptr};
}

Bytes payload(int n, unsigned seed) {
    std::mt19937 rng(seed);
    Bytes b(static_cast<std::size_t>(n));
    for (auto& v : b) v = static_cast<std::uint8_t>(rng() & 255);
    return b;
}

Bytes frame(const std::string& dst, const std::string& src, int ctrl, const std::string& info) {
    Bytes out;
    auto addr = [&](const std::string& c, bool last) {
        for (std::size_t i = 0; i < 6; ++i) out.push_back(static_cast<std::uint8_t>((i < c.size() ? c[i] : ' ') << 1));
        out.push_back(static_cast<std::uint8_t>(0x60 | (last ? 1 : 0)));
    };
    addr(dst, false);
    addr(src, true);
    out.push_back(static_cast<std::uint8_t>(ctrl));
    if ((ctrl & 1) == 0 || (ctrl & 0xEF) == 0x03) {
        out.push_back(0xF0);
        out.insert(out.end(), info.begin(), info.end());
    }
    return out;
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(300, "test_arq_phy");

    check::current_step = "known answers";
    check::equal(mask_value({7, 0, 2}), 3389368501u, "mask_value(7, 0, 2)");
    check::equal(mask_value(ctl_mask(0, 0, kisslink::KISS_KEY)), 3994227689u, "mask_value(KISS control 0)");
    check::equal(mask_value({0, 1, 9}), 0u, "key 0: mask 0");
    check::equal(kisslink::station_hash("W1AW"), 8033, "station_hash(W1AW)");
    check::equal(kisslink::station_hash("kc2g-7"), 26660, "station_hash(kc2g-7)");
    {
        const auto a = kisslink::parse_ax25(frame("KC2G", "W1AW", 0x00, "hi"));
        check::is_true(a && a->dst == "KC2G" && a->src == "W1AW" && a->next_hop == "KC2G" && a->connected, "AX.25 I frame");
        check::is_true(!kisslink::parse_ax25(frame("APRS", "W1AW", 0x13, "!"))->connected, "UI with P/F");
        const std::string junk = "not an ax.25 frame at all";
        check::is_true(!kisslink::parse_ax25(Bytes(junk.begin(), junk.end())), "not AX.25");
    }

    const std::string name = "w48-qpsk-r1/2";
    const auto& spec = *codes::submode(name);
    std::vector<Bytes> pl;
    for (unsigned i = 0; i < 3; ++i) pl.push_back(payload(spec.payload_bytes, i));
    auto burst = [&](int rv) {
        TxBurst b{name, {}, 0};
        for (int i = 0; i < 3; ++i) b.slots.push_back({{7, 0, i}, rv, pl[static_cast<std::size_t>(i)]});
        return b;
    };

    check::current_step = "masked decodes";
    {
        ModemRx rx(hear(on_air(tx_audio(burst(0)), 0.02, 1)));
        check::equal(rx.n_cw(), 3, "n_cw");
        check::is_true(rx.decode_plain(0, {7, 0, 0}) == pl[0], "slot 0 under its mask");
        check::is_true(!rx.decode_plain(1, {7, 0, 0}), "another seq");
        check::is_true(!rx.decode_plain(1, {8, 0, 1}), "another session");
        check::is_true(rx.decode_plain(1, {7, 0, 1}) == pl[1], "slot 1 under its mask");
        check::is_true(!rx.decode_plain(5, {7, 0, 5}), "past the burst");
        const auto m = measure(hear(on_air(tx_audio(burst(0)), 0.02, 1)));
        check::is_true(m.snr_est > 15 && m.frames == 12 && m.mi[0] > 0.9, "measure: a clean burst");
    }

    check::current_step = "resend combining";
    {
        // noise where one transmission mostly fails: what fails alone is
        // stored, and the RV 1 resend decodes from the pair
        SoftStore store;
        int calls = 0;
        auto counting = [&] { return ++calls, 0.0; };
        std::vector<bool> failed;
        {
            ModemRx rx(hear(on_air(tx_audio(burst(0)), 1.2, 3)), &store, 10.0, nullptr, true, counting);
            for (int i = 0; i < 3; ++i) {
                SoftKey k{false, 0, i, 0};
                failed.push_back(!rx.decode(i, {7, 0, i}, 0, &k));
            }
        }
        const auto n_failed = std::count(failed.begin(), failed.end(), true);
        check::equal(static_cast<long>(store.size()), static_cast<long>(n_failed), "a failed decode stores its soft bits");
        check::is_true(n_failed >= 1, "some slot fails alone");
        check::is_true(calls > 1, "DD looked at the clock");
        ModemRx again(hear(on_air(tx_audio(burst(1)), 1.2, 4)), &store);
        int got = 0;
        for (int i = 0; i < 3; ++i) {
            SoftKey k{false, 0, i, 0};
            if (failed[static_cast<std::size_t>(i)]) got += again.decode(i, {7, 0, i}, 1, &k) == pl[static_cast<std::size_t>(i)];
        }
        check::equal(got, static_cast<int>(n_failed), "resends combine");
        bool threw = false;
        store[SoftKey{false, 0, 9, 0}] = SoftEntry{std::vector<double>(10), 0, "qpsk-r1/2", 0, 0, {}};
        try {
            SoftKey k{false, 0, 9, 0};
            again.decode(0, {7, 0, 9}, 1, &k);
        } catch (const StoreMismatch&) {
            threw = true;
        }
        check::is_true(threw, "a resend in another submode");
    }

    check::current_step = "kiss links";
    {
        double t = 0.0;
        auto clock = [&] { return t; };
        kisslink::KissLink a(2, "", clock), b(2, "", clock);
        auto over_air = [&](kisslink::KissLink& tx, kisslink::KissLink& rx, unsigned seed) {
            const auto bu = tx.next_burst();
            auto frames = rx.on_burst(hear(on_air(tx_audio(*bu), 0.02, seed)));
            t += 3;
            return std::make_pair(bu, frames ? *frames : std::vector<Bytes>{});
        };
        const auto i1 = frame("KC2G", "W1AW", 0x00, "connected hello");
        a.enqueue(i1);
        auto [b1, f1] = over_air(a, b, 1);
        check::is_true(b1->submode == "qpsk-r1/5" && f1 == std::vector<Bytes>{i1}, "no report yet: broadcast");
        b.enqueue(frame("W1AW", "KC2G", 0x21, ""));
        auto [b2, f2] = over_air(b, a, 2);
        check::equal(static_cast<int>(f2.size()), 1, "the RR crosses with its report");
        const auto i2 = frame("KC2G", "W1AW", 0x22, std::string(300, 'x'));
        a.enqueue(i2);
        const auto b3 = a.next_burst();
        check::is_true(b3 && b3->submode != "qpsk-r1/5", "shifted from the report: " + (b3 ? b3->submode : ""));
        check::is_true(a.queue.empty() && a.n_sent == 2, "queue drained");
        a.command(3, Bytes{200});
        check::is_true(a.slot_s == 2.0, "SLOTTIME");
    }

    return check::report("test_arq_phy");
}
