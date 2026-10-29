// Primed reception (docs/arq.md §3a) without Python: known bits and the score, a primed CPM control decode on
// erased soft bits, two stations whose control slots fail blind, the expected slot's lock and its timing.
// Everything here is deterministic integer arithmetic or std::mt19937's raw output (no
// std::normal_distribution, which differs across standard libraries), so every platform takes the same path.
// Parity with Python is tests/test_primed.py's.

#include <cmath>
#include <cstdint>
#include <random>

#include "arq/phy.hpp"
#include "check.hpp"
#include "receiver/receiver.hpp"

using namespace data2g;
using namespace data2g::arq;

namespace {

// A deterministic share `keep` (percent) of a codeword's soft bits kept as +-4, the rest erased.
std::vector<double> erased(const std::vector<std::uint8_t>& c, int keep, std::uint32_t salt) {
    std::vector<double> x(c.size());
    for (std::size_t i = 0; i < c.size(); ++i) {
        const std::uint32_t h = (static_cast<std::uint32_t>(i) + salt) * 2654435761u;
        x[i] = (h >> 16) % 100 < static_cast<std::uint32_t>(keep) ? 4.0 * (1 - 2.0 * c[i]) : 0.0;
    }
    return x;
}

ModemRx cpm_rx(const std::vector<double>& soft) {
    auto r = std::make_shared<cpm::Received>();
    r->spec = cpm::spec("fsk16r25-r1/3");
    r->n_cw = 1;
    r->n_ctl_slots = 1;
    r->dup = false;
    r->soft = {soft};
    r->cfo = 0.0;
    r->preamble_start = r->header_end = 0;
    return ModemRx(Heard{nullptr, r});
}

// Approximately Gaussian, from mt19937's raw output alone (twelve uniforms).
double gauss(std::mt19937& rng) {
    double s = 0;
    for (int i = 0; i < 12; ++i) s += static_cast<double>(rng()) / 4294967296.0;
    return s - 6.0;
}

class FixedPolicy : public Policy {
public:
    std::pair<std::string, int> choose(Station&, int) override { return {"m22", 6}; }
    int payload_bytes(const std::string&) override { return 22; }
    int rv_cycle(const std::string&) override { return 4; }
    std::string connect_mode(int, int) override { return "m22"; }
};

// Control slots fail blind on demand; a primed retry passes when the hypothesis agrees with what was sent.
class PrimeRx : public RxBurst {
public:
    PrimeRx(const TxBurst& b, bool blind_ok) : burst(b), blind_ok(blind_ok) {}
    const TxBurst& burst;
    bool blind_ok;
    int primed = 0;
    const std::string& submode() const override { return burst.submode; }
    int n_cw() const override { return static_cast<int>(burst.slots.size()); }
    std::optional<Bytes> decode(int slot, const MaskId& mask, int, const SoftKey*) override {
        const Slot& s = burst.slots.at(static_cast<std::size_t>(slot));
        if (!(mask == s.mask_id) || (mask.seq >= SEQ_MOD && !blind_ok)) return std::nullopt;
        return s.payload;
    }
    void forget(const SoftKey&) override {}
    bool can_prime() const override { return true; }
    std::optional<Bytes> decode_known(int slot, const MaskId& mask, ByteView t, ByteView m, bool) override {
        const Slot& s = burst.slots.at(static_cast<std::size_t>(slot));
        if (!(mask == s.mask_id) || t.size() != s.payload.size()) return std::nullopt;
        for (std::size_t i = 0; i < t.size(); ++i)
            if ((t[i] ^ s.payload[i]) & m[i]) return std::nullopt;
        ++primed;
        return s.payload;
    }
};

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(300, "test_primed");

    const codes::Spec& ctl = codes::spec(cpm::ctl(cpm::grid_of(*cpm::spec("fsk16r25-r1/3"))));
    check::current_step = "known_bits, score";
    {
        const Bytes t{0xA5, 0x0F}, m{0xFF, 0x0F};
        const auto k = codes::known_bits(ctl, t, m);
        check::is_true(k[0] == 1 && k[1] == 0 && k[7] == 1 && k[8] == -1 && k[12] == 1 && k[16] == -1, "known_bits");
        check::is_true(k[static_cast<std::size_t>(8 * ctl.payload_bytes)] == -1, "the CRC is never known");
        const Bytes p(static_cast<std::size_t>(ctl.payload_bytes), 7);
        const auto c = codes::encode(ctl, p, 0, 5, 0);
        std::vector<double> x(c.size());
        for (std::size_t i = 0; i < c.size(); ++i) x[i] = 30.0 * (1 - 2.0 * c[i]);
        check::is_true(std::abs(codes::score(ctl, x, p, 5, 0) - ctl.coded_bits * std::log(2.0)) < 1e-6, "score: certain");
        check::equal(codes::score(ctl, std::vector<double>(c.size(), 0.0), p, 5, 0), 0.0, "score: erased");
    }

    check::current_step = "decode_known";
    {
        const MaskId mid = ctl_mask(1, 0, 0x5A5A), other = ctl_mask(1, 0, 0x1234);
        Core core;
        core.burst_seq = 3, core.acted_on = 2, core.cum = 77, core.recommend = 50, core.size_hint = 2;
        const Bytes p = Control{core, {{T_REPLY, Bytes{50}}, {T_DUPCTL, Bytes{}}}}.pack(ctl.payload_bytes)[0];
        Bytes known(p.size(), 0xFF);
        known[1] &= 0xC0;  // the cumulative ACK, core bits 21..15
        known[2] &= 0x7F;
        const auto soft = erased(codes::encode(ctl, p, 0, mask_value(mid), 0), 30, 11);
        check::is_true(!cpm_rx(soft).decode(0, mid, 0, nullptr), "premise: no blind decode (30% kept)");
        check::is_true(cpm_rx(soft).decode_known(0, mid, p, known, false) == p, "primed: the control");
        Bytes wrong = p;
        wrong[6] ^= 1;  // another T_REPLY value, pinned
        check::is_true(!cpm_rx(soft).decode_known(0, mid, wrong, known, false), "a wrong hypothesis fails its CRC");
        check::is_true(!cpm_rx(soft).decode_known(0, other, p, known, false), "another session's mask fails");
        check::is_true(!cpm_rx(std::vector<double>(soft.size(), 0.0)).decode_known(0, mid, p, Bytes(p.size(), 0xFF), false),
                       "nothing there: nothing primed");
    }

    check::current_step = "stations";
    {
        std::mt19937 rng(5);
        Bytes up(3000);
        for (auto& v : up) v = static_cast<std::uint8_t>(rng());
        Station a(0, std::make_shared<FixedPolicy>(), true), b(1, std::make_shared<FixedPolicy>());
        a.write(up);
        Bytes got;
        TxBurstPtr burst = a.build();
        Station* sender = &a;
        int primed = 0, turn = 0;
        for (; turn < 400 && got != up; ++turn) {
            Station* receiver = sender == &a ? &b : &a;
            PrimeRx rx(*burst, turn < 2 || turn % 2 == 0);  // the first exchange blind: something to predict from
            check::is_true(receiver->handle(rx), "every control read, blind or primed");
            primed += rx.primed;
            const Bytes r = b.read();
            got.insert(got.end(), r.begin(), r.end());
            burst = receiver->build();
            receiver->answered();
            sender = receiver;
        }
        check::is_true(got == up, "delivered");
        check::is_true(primed > turn / 3, "primed half the controls");
        check::equal(a.stats["rx_primed"] + b.stats["rx_primed"], static_cast<std::int64_t>(primed), "rx_primed counts them");
    }

    check::current_step = "prime_lock";
    {
        const auto& g = *cpm::grid("c16r25");
        const cpm::Spec& sp = *cpm::spec("fsk16r25-r1/3");
        const Bytes p(static_cast<std::size_t>(ctl.payload_bytes), 0x3C);
        const auto burst = cpm::modulate(sp, {codes::encode(ctl, p, 0, 0, 0)}, false);
        std::mt19937 rng(9);
        double pw = 0;
        for (double v : burst) pw += v * v;
        const double sigma = std::sqrt(pw / static_cast<double>(burst.size()) * 4.0);  // -6 dB in the audio band
        std::vector<double> y(2400 + burst.size() + 16000);
        for (std::size_t i = 0; i < y.size(); ++i) y[i] = sigma * gauss(rng) + (i >= 2400 && i < 2400 + burst.size() ? burst[i - 2400] : 0.0);
        const auto lock = cpm::prime_lock(g, y, 2400 - 1200, 2400 + 1200, 0.0);
        check::is_true(lock && lock->start == 2400 && lock->spec == &sp && lock->n_data == 0 && !lock->dup, "prime_lock: the burst");
        std::vector<double> z(y.size());
        for (double& v : z) v = gauss(rng);
        check::is_true(!cpm::prime_lock(g, z, 1200, 3600, 0.0), "prime_lock: noise");
    }

    check::current_step = "ReplySlot";
    {
        receiver::ReplySlot s;
        check::is_true(!s.window(10.0), "nothing learned");
        s.heard(0.3, 4.5);
        s.heard(0.4, std::nullopt);
        s.heard(5.0, std::nullopt);  // not a reply
        const auto w = s.window(10.0);
        check::is_true(w && std::abs(w->first - (10.35 - s.TOL_S)) < 1e-12 && std::abs(w->second - (10.35 + s.TOL_S)) < 1e-12,
                       "the median gap, either side");
    }

    return check::report("test_primed");
}
