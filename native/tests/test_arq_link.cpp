// Known answers and a Python-free run of two stations and two sessions
// through a lossy fake channel. Parity with data2g/arq is
// tests/test_native_arq.py's job.

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <string>

#include "arq/session.hpp"
#include "check.hpp"

using namespace data2g;
using namespace data2g::arq;

namespace {

std::string hex(const Bytes& b) {
    static const char* d = "0123456789abcdef";
    std::string out;
    for (auto x : b) out += {d[x >> 4], d[x & 15]};
    return out;
}

Bytes bytes(const std::string& s) { return {s.begin(), s.end()}; }

// test_arq's MODES: name -> payload bytes; one 4-RV code.
class TestPolicy : public Policy {
public:
    explicit TestPolicy(unsigned seed) : rng(seed) {}
    std::mt19937 rng;
    std::pair<std::string, int> choose(Station&, int escalation) override {
        if (escalation) return {"m46", 2};
        static const char* modes[] = {"m4", "m22", "m46"};
        if (rng() % 10 == 0) mode = modes[rng() % 3];
        return {mode, static_cast<int>(1 + rng() % 12)};
    }
    int payload_bytes(const std::string& m) override { return m == "m4" ? 4 : m == "m22" ? 22 : 46; }
    int rv_cycle(const std::string& m) override { return m == "m4" ? 1 : 4; }
    double airtime(const std::string&, int n_cw, bool) override { return 0.4 + 0.12 * n_cw; }
    std::string connect_mode(int, int) override { return "m46"; }
    std::string mode = "m22";
};

// Slots lost at random, the rest decode if the mask and RV match.
class FakeRx : public RxBurst {
public:
    FakeRx(const TxBurst& b, std::mt19937& rng, double p_cw) : burst(b) {
        for (std::size_t i = 0; i < b.slots.size(); ++i) good.push_back(std::uniform_real_distribution<>(0, 1)(rng) >= p_cw);
    }
    const TxBurst& burst;
    std::vector<bool> good;
    int mismatch = 0;
    const std::string& submode() const override { return burst.submode; }
    int n_cw() const override { return static_cast<int>(burst.slots.size()); }
    std::optional<Bytes> decode(int slot, const MaskId& mask, int rv, const SoftKey*) override {
        const Slot& s = burst.slots.at(static_cast<std::size_t>(slot));
        if (!(mask == s.mask_id) || (mask.seq < SEQ_MOD && rv != s.rv)) {
            mismatch += mask.seq < SEQ_MOD;
            return std::nullopt;
        }
        return good[static_cast<std::size_t>(slot)] ? std::optional<Bytes>(s.payload) : std::nullopt;
    }
    void forget(const SoftKey&) override {}
};

Bytes text(std::mt19937& rng, std::size_t n) {
    static const char* words[] = {"the ", "of ", "and ", "burst ", "codeword ", "ACK ", "resend ", "73 ", "QTH ", "\r\n"};
    Bytes out;
    while (out.size() < n) {
        const std::string w = words[rng() % 10];
        out.insert(out.end(), w.begin(), w.end());
    }
    out.resize(n);
    return out;
}

void stations(unsigned seed, double p_burst, double p_cw) {
    std::mt19937 rng(seed);
    const Bytes up = text(rng, 5000), down = text(rng, 1500);
    Station a(0, std::make_shared<TestPolicy>(seed + 1), true), b(1, std::make_shared<TestPolicy>(seed + 2));
    a.write(up);
    b.write(down);
    Bytes got_a, got_b;
    TxBurstPtr burst = a.build();
    Station* sender = &a;
    int mismatch = 0;
    std::string result = "stuck";
    for (int turn = 0; turn < 4000; ++turn) {
        Station* receiver = sender == &a ? &b : &a;
        bool ok = false;
        if (std::uniform_real_distribution<>(0, 1)(rng) >= p_burst) {
            FakeRx rx(*burst, rng, p_cw);
            ok = receiver->handle(rx);
            mismatch += rx.mismatch;
        }
        for (auto [st, got] : {std::pair{&a, &got_a}, std::pair{&b, &got_b}}) {
            const Bytes r = st->read();
            got->insert(got->end(), r.begin(), r.end());
        }
        check::is_true(Bytes(up.begin(), up.begin() + static_cast<std::ptrdiff_t>(got_b.size())) == got_b, "delivered prefix exact");
        if (a.state == LinkState::FAILED || b.state == LinkState::FAILED) {
            result = "failed: " + a.fail_reason + b.fail_reason;
            break;
        }
        if (got_a == down && got_b == up && !a.tx.pending() && !b.tx.pending()) {
            result = "done";
            break;
        }
        if (ok) {
            burst = receiver->build();
            receiver->answered();
            sender = receiver;
        } else {
            burst = a.on_timeout();
            sender = &a;
            if (!burst) {
                result = "failed: " + a.fail_reason;
                break;
            }
        }
    }
    const bool lossy_ok = p_burst >= 0.2 && result == "failed: link lost";
    check::is_true(result == "done" || lossy_ok, "stations seed " + std::to_string(seed) + ": " + result);
    check::equal(mismatch, 0, "no accounting mismatch");
    if (result == "done") check::is_true(a.stats["cw_comp"] > 0, "text goes compressed");
}

void sessions(unsigned seed, double p_burst) {
    std::mt19937 rng(seed);
    Session a("W1AW", std::make_shared<TestPolicy>(seed + 1), 1.0, std::make_shared<DefaultRng>(seed + 2));
    Session b("K2XYZ", std::make_shared<TestPolicy>(seed + 3), 1.0, std::make_shared<DefaultRng>(seed + 4));
    const Bytes up = text(rng, 3000), down = text(rng, 700);
    b.listen();
    a.write(up);
    b.write(down);
    a.connect("K2XYZ", 2, 0.0);
    // half duplex, one burst in flight: deliver at its end
    Bytes got_a, got_b;
    double t = 0;
    bool asked = false;
    while (t < 2000 && !(a.state == SessionState::CLOSED && b.state == SessionState::CLOSED)) {
        bool sent = false;
        for (auto [me, other] : {std::pair{&a, &b}, std::pair{&b, &a}}) {
            auto burst = me->poll(t);
            if (!burst) continue;
            const double end = t + 0.1 + me->policy->airtime(burst->submode, static_cast<int>(burst->slots.size()), false);
            me->on_tx_end(end);
            if (std::uniform_real_distribution<>(0, 1)(rng) >= p_burst) {
                other->on_header(burst->submode, static_cast<int>(burst->slots.size()), t + 0.35);
                FakeRx rx(*burst, rng, 0.05);
                other->on_rx(rx, end + 0.2);
            }
            t = end + 0.2;
            sent = true;
            break;
        }
        const Bytes ra = a.read(), rb = b.read();
        got_a.insert(got_a.end(), ra.begin(), ra.end());
        got_b.insert(got_b.end(), rb.begin(), rb.end());
        if (got_a == down && got_b == up && !asked) {
            a.disconnect();
            asked = true;
        }
        if (!sent) {
            auto n = std::min(a.next_event().value_or(1e9), b.next_event().value_or(1e9));
            t = std::max(t, n);
        }
    }
    check::is_true(got_b == up && got_a == down, "sessions seed " + std::to_string(seed) + " deliver both ways");
    check::is_true(a.state == SessionState::CLOSED && b.state == SessionState::CLOSED, "both close");
    check::equal(a.events.front(), std::string("CONNECTED K2XYZ"), "caller's CONNECTED");
    check::equal(b.events.front(), std::string("CONNECTED W1AW"), "callee's CONNECTED");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    Core c{1, 2, 5, 3, 100, true, 10, 33, 2};
    check::equal(hex(c.pack()), std::string("5af24a86"), "core word (Python's value)");
    check::is_true(Core::unpack(c.pack()) == c, "core round trip");
    check::equal(hex(pack_call("w1aw")), std::string("05dc057000000000"), "callsign packing");
    check::equal(unpack_call(pack_call("VK2ABC-15")), std::string("VK2ABC-15"), "callsign round trip");
    check::equal(session_key("W1AW", "K2XYZ-7", 1234), 53747, "session key");
    check::equal(hex(pack_bitmap({1, 3, 64}, 0)), std::string("a000000000000001"), "bitmap");
    check::equal(hex(pack_rv({0, 1, 2, 3, 3})), std::string("1bc0"), "rv packing");
    check::equal(unwrap(2, 127), std::int64_t{130}, "unwrap past the wrap");
    check::equal(unwrap(126, 130), std::int64_t{126}, "unwrap behind");

    const Bytes hist = bytes("history of the stream so far "), data = bytes(std::string(400, 'x') + "the quick brown fox");
    Bytes z = deflate(hist, data);
    z.resize(z.size() + 9, 0);  // zero padded, as in a codeword
    check::is_true(inflate(hist, z) == data, "inflate(deflate) with padding");
    check::equal(deflate(Bytes{}, bytes(std::string(20 * 20, 'a'))).size() < 20, true, "deflate compresses");
    bool threw = false;
    try {
        inflate(hist, Bytes(30, 0xff));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "garbage fails to inflate");

    RecordReader r;
    const Bytes recs = to_records(Bytes(600, 7));
    check::equal(recs.size(), std::size_t{603}, "records of 255");
    check::equal(r.feed(recs).size(), std::size_t{600}, "records read back");

    for (unsigned seed = 0; seed < 12; ++seed) stations(seed, (seed % 3) * 0.1, (seed / 3 % 3) * 0.1);
    for (unsigned seed = 0; seed < 6; ++seed) sessions(seed, (seed % 3) * 0.05);
    return check::report("arq");
}
