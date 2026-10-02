// host::Host with no Python, over two C++ engines (GearPolicy, so BUFFER's
// credit comes from the C++ shifter): command replies, a session driven by
// VARA commands, data both ways, notifications, IAMALIVE. Parity with
// host.py is tests/test_native_host.py's.

#include <algorithm>
#include <cmath>
#include <random>

#include "check.hpp"
#include "host/host.hpp"

using namespace data2g;
using arq::Engine;

namespace {

constexpr int BLOCK = config::FS / 10;

bool has(const std::vector<std::string>& v, const std::string& s) { return std::find(v.begin(), v.end(), s) != v.end(); }

struct Pair {
    Engine ea, eb;
    host::Host a, b;
    std::vector<std::string> seen_a, seen_b;
    std::mt19937 rng{1};
    std::vector<double> ao = std::vector<double>(BLOCK), bo = std::vector<double>(BLOCK);

    Pair(arq::EngineConfig ca, arq::EngineConfig cb) : ea("NOCALL", ca), eb("NOCALL", cb), a(ea), b(eb) {}

    // Both step through a 12 dB channel; each host's after_step, as the app does.
    bool run(double seconds, const std::function<bool()>& until) {
        std::normal_distribution<double> noise(0.0, std::sqrt((config::FS / 2.0) / config::SNR_REF_BW_HZ / std::pow(10.0, 1.2)));
        std::vector<double> x(BLOCK);
        for (int i = 0; i < static_cast<int>(seconds * 10); ++i) {
            for (int j = 0; j < BLOCK; ++j) x[j] = 2.2 * bo[j] + noise(rng);
            auto oa = ea.step(x);
            a.after_step(oa.ptt);
            ao = oa.audio;
            for (int j = 0; j < BLOCK; ++j) x[j] = 2.2 * ao[j] + noise(rng);
            auto ob = eb.step(x);
            b.after_step(ob.ptt);
            bo = ob.audio;
            take();
            if (until()) return true;
        }
        return false;
    }
    void take() {
        seen_a.insert(seen_a.end(), a.out_cmd.begin(), a.out_cmd.end());
        seen_b.insert(seen_b.end(), b.out_cmd.begin(), b.out_cmd.end());
        a.out_cmd.clear();
        b.out_cmd.clear();
    }
};

arq::EngineConfig seeded(std::uint64_t s) {
    arq::EngineConfig c;
    c.seed = s;
    return c;
}

void commands() {
    check::current_step = "commands";
    Engine e("NOCALL", seeded(3));
    host::Host h(e);
    for (const char* line : {"VERSION", "mycall k2xyz K2XYZ-1", "LISTEN CQ", "LISTEN on", "BW1200", "CHAT OFF", "PUBLIC ON",
                             "FOO", "   ", "LISTEN", "CQFRAME K2XYZ 9999", "CONNECT W1AW"})
        h.command(line);
    const std::vector<std::string> want = {"VERSION Data2G 0.1", "OK", "OK", "OK", "OK", "OK", "OK", "WRONG", "WRONG", "WRONG", "WRONG"};
    check::is_true(h.out_cmd == want, "replies");
    check::equal(e.call(), std::string("K2XYZ"), "MYCALL: first call");
    check::is_true(e.aliases() == std::vector<std::string>{"K2XYZ-1"}, "MYCALL: aliases");
    check::is_true(h.listening && e.session().state == arq::SessionState::LISTEN, "listening");
    check::equal(h.cap, 1, "BW1200");
    h.out_cmd.clear();
    h.command("ABORT");
    check::is_true(h.out_cmd == std::vector<std::string>{"DISCONNECTED", "OK"}, "ABORT");
    check::is_true(e.session().state == arq::SessionState::LISTEN, "ABORT keeps listening");
    h.client_gone();
    check::is_true(!h.listening && e.session().state != arq::SessionState::LISTEN, "client gone: no longer listening");

    int alive = 0;
    h.out_cmd.clear();
    for (int i = 0; i < static_cast<int>(2.5 * host::ALIVE_S * 10); ++i) {
        e.step(std::vector<double>(BLOCK, 0.0));
        h.after_step(false);
        alive += static_cast<int>(std::count(h.out_cmd.begin(), h.out_cmd.end(), "IAMALIVE"));
        h.out_cmd.clear();
    }
    check::equal(alive, 2, "IAMALIVE every 60 s");
}

void session() {
    check::current_step = "session";
    Pair p(seeded(11), seeded(12));
    for (const char* line : {"MYCALL K2XYZ", "LISTEN ON"}) p.b.command(line);
    p.a.command("CONNECT W1AW K2XYZ");
    p.take();
    check::is_true(p.run(60, [&] { return has(p.seen_a, "CONNECTED W1AW K2XYZ 2300") && has(p.seen_b, "CONNECTED W1AW K2XYZ 2300"); }),
                   "CONNECTED on both");
    check::is_true(has(p.seen_a, "PTT ON") && has(p.seen_b, "BUSY ON"), "PTT and BUSY");
    check::is_true(std::any_of(p.seen_a.begin(), p.seen_a.end(), [](const std::string& s) { return s.starts_with("MODE "); }), "MODE");
    std::mt19937 rng(5);
    arq::Bytes up(3000);
    for (auto& v : up) v = static_cast<std::uint8_t>(rng());
    const arq::Bytes down = {'h', 'e', 'l', 'l', 'o'};
    p.seen_a.clear();
    p.a.data_in(up);
    p.b.data_in(down);
    check::is_true(p.run(120, [&] { return p.b.out_data.size() >= up.size() && p.a.out_data.size() >= down.size(); }), "data both ways");
    check::is_true(p.b.out_data == up && p.a.out_data == down, "data intact");
    // BUFFER: nonzero while the 3000 bytes were queued, 0 once acked
    std::vector<long> buffers;
    for (const auto& s : p.seen_a)
        if (s.starts_with("BUFFER ")) buffers.push_back(std::stol(s.substr(7)));
    check::is_true(!buffers.empty() && buffers.front() > 0, "BUFFER while queued");
    p.run(10, [&] { return has(p.seen_a, "BUFFER 0"); });
    check::is_true(has(p.seen_a, "BUFFER 0"), "BUFFER 0 once acked");
    p.a.command("DISCONNECT");
    check::is_true(p.run(60, [&] { return has(p.seen_a, "DISCONNECTED") && has(p.seen_b, "DISCONNECTED"); }), "DISCONNECTED on both");
    check::is_true(p.eb.session().state == arq::SessionState::LISTEN, "the callee listens again");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog dog(600, "test_host");
    commands();
    session();
    return check::report("test_host");
}
