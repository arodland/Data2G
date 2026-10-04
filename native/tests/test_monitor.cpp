// monitor::Monitor: the dump formats, then a third station hearing two
// engines: a session (CONNECT, both streams, compressed text included,
// DISC, the trailing ID) and a KISS broadcast frame.

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>

#include "check.hpp"
#include "monitor/monitor.hpp"

using namespace data2g;
using namespace data2g::arq;
using data2g::monitor::Dump;

namespace {

constexpr int BLOCK = config::FS / 10;

Bytes text(const std::string& s) { return Bytes(s.begin(), s.end()); }

void units() {
    check::current_step = "units";
    check::equal(monitor::text_dump(text("Hi\n\xff")), std::string("Hi<0A>\n<FF>"), "text_dump");
    check::equal(monitor::hex_dump(text("Hello, world\n\x01\x02\x03XY")),
                 std::string("00000000: 4865 6c6c 6f2c 2077 6f72 6c64 0a01 0203  Hello, world....\n"
                             "00000010: 5859                                     XY\n"),
                 "hex_dump");
    Dump d{"head\n", {{"p", text("a\nb")}}};
    check::equal(monitor::render(d, monitor::Format::TEXT), std::string("head\n  p, 3 B:\n    a<0A>\n    b\n"), "render");
}

// Both engines step, each hearing the other, the monitor hearing both,
// until until() or `seconds` pass.
double g_snr_db = 12.0;

bool run(Engine& a, Engine& b, monitor::Monitor& m, std::vector<Dump>& dumps, double seconds,
         const std::function<bool()>& until, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, std::sqrt((config::FS / 2.0) / config::SNR_REF_BW_HZ / std::pow(10.0, g_snr_db / 10)));
    std::vector<double> ao(BLOCK, 0.0), bo(BLOCK, 0.0), x(BLOCK), y(BLOCK);
    for (int i = 0; i < static_cast<int>(seconds * 10); ++i) {
        for (int j = 0; j < BLOCK; ++j) y[j] = 2.2 * (ao[j] + bo[j]) + noise(rng);
        for (auto& dump : m.feed(y, [](double t) {
                 char s[16];
                 std::snprintf(s, sizeof s, "%.1f", t);
                 return std::string(s);
             }))
            dumps.push_back(std::move(dump));
        for (int j = 0; j < BLOCK; ++j) x[j] = 2.2 * bo[j] + noise(rng);
        ao = a.step(x).audio;
        for (int j = 0; j < BLOCK; ++j) x[j] = 2.2 * ao[j] + noise(rng);
        bo = b.step(x).audio;
        if (until()) return true;
    }
    return false;
}

// Every payload whose label starts with `label`, concatenated.
Bytes stream(const std::vector<Dump>& dumps, const std::string& label) {
    Bytes out;
    for (const auto& d : dumps)
        for (const auto& [l, p] : d.payloads)
            if (l.rfind(label, 0) == 0) out.insert(out.end(), p.begin(), p.end());
    return out;
}

std::string all(const std::vector<Dump>& dumps) {
    std::string out;
    for (const auto& d : dumps) out += monitor::render(d, monitor::Format::TEXT);
    return out;
}

void session() {
    check::current_step = "session";
    EngineConfig ca, cb;
    ca.seed = 31;
    cb.seed = 32;
    Engine a("W1AW", ca), b("K2XYZ", cb);
    monitor::Monitor m;
    std::vector<Dump> dumps;
    b.listen();
    a.connect("K2XYZ", 2);
    auto both = [&](SessionState s) { return a.session().state == s && b.session().state == s; };
    check::is_true(run(a, b, m, dumps, 60, [&] { return both(SessionState::CONNECTED); }, 1), "connected");
    std::string up;
    while (up.size() < 3000) up += "The quick brown fox jumps over the lazy dog, line " + std::to_string(up.size()) + ".\n";
    const std::string down = "73 de K2XYZ\n";
    a.session().write(text(up));
    b.session().write(text(down));
    std::size_t got_b = 0, got_a = 0;
    check::is_true(run(a, b, m, dumps, 240, [&] {
        got_b += b.session().read().size();
        got_a += a.session().read().size();
        return got_b >= up.size() && got_a >= down.size();
    }, 2), "data both ways");
    a.session().disconnect();
    check::is_true(run(a, b, m, dumps, 60, [&] { return both(SessionState::CLOSED); }, 3), "closed");
    run(a, b, m, dumps, 8, [] { return false; }, 4);  // the IDs after it

    const std::string out = all(dumps);
    check::is_true(out.find("CONNECT W1AW>K2XYZ") != std::string::npos, "CONNECT named");
    check::is_true(out.find("CONNECT_ACK K2XYZ>W1AW") != std::string::npos, "CONNECT_ACK named by its nonce");
    check::is_true(out.find("ARQ W1AW>K2XYZ key") != std::string::npos, "A's bursts named");
    check::is_true(out.find("ARQ K2XYZ>W1AW key") != std::string::npos, "B's bursts named");
    check::is_true(out.find("ok(z)") != std::string::npos, "compressed codewords read");
    check::is_true(out.find("DISC") != std::string::npos, "DISC");
    check::is_true(out.find("ID W1AW") != std::string::npos || out.find("ID K2XYZ") != std::string::npos, "ID frame");
    check::is_true(stream(dumps, "W1AW>K2XYZ stream") == text(up), "A's stream as sent");
    check::is_true(stream(dumps, "K2XYZ>W1AW stream") == text(down), "B's stream as sent");
    if (check::failures || std::getenv("SHOW")) std::fputs(out.c_str(), stderr);
}

void broadcast() {
    check::current_step = "broadcast";
    kisslink::KissLink ka;
    EngineConfig ca, cb;
    ca.seed = 41;
    ca.kiss = &ka;
    cb.seed = 42;
    Engine a("W1AW", ca), b("K2XYZ", cb);
    Bytes ui;  // APRS <- W1AW, UI
    for (const auto& [c, last] : {std::pair{std::string("APRS"), false}, {std::string("W1AW"), true}}) {
        for (std::size_t i = 0; i < 6; ++i) ui.push_back(static_cast<std::uint8_t>((i < c.size() ? c[i] : ' ') << 1));
        ui.push_back(static_cast<std::uint8_t>(0x60 | last));
    }
    ui.push_back(0x03);
    ui.push_back(0xF0);
    const Bytes info = text("!beacon");
    ui.insert(ui.end(), info.begin(), info.end());
    ka.enqueue(ui);
    monitor::Monitor m;
    std::vector<Dump> dumps;
    check::is_true(run(a, b, m, dumps, 30, [&] { return !dumps.empty() && ka.queue.empty(); }, 5), "heard");
    run(a, b, m, dumps, 3, [] { return false; }, 6);
    const std::string out = all(dumps);
    check::is_true(out.find("BCAST \"KISS 0\": 1 frame(s)") != std::string::npos, "group named");
    check::is_true(stream(dumps, "frame 1 (AX.25 W1AW>APRS, UI)") == ui, "the frame, AX.25 addresses read");
    if (check::failures) std::fputs(out.c_str(), stderr);
}

}  // namespace

// Random bytes (no compression, many codewords) on a link losing some:
// resends, and the monitor's own losses.
void lossy(double snr_db, unsigned seed) {
    check::current_step = "lossy";
    g_snr_db = snr_db;
    EngineConfig ca, cb;
    ca.seed = seed;
    cb.seed = seed + 1;
    Engine a("W1AW", ca), b("K2XYZ", cb);
    monitor::Monitor m;
    std::vector<Dump> dumps;
    b.listen();
    a.connect("K2XYZ", 2);
    check::is_true(run(a, b, m, dumps, 120, [&] { return a.session().state == SessionState::CONNECTED; }, seed), "connected");
    std::mt19937 rng(seed);
    Bytes up(6000);
    for (auto& v : up) v = static_cast<std::uint8_t>(rng());
    a.session().write(up);
    std::size_t got = 0;
    check::is_true(run(a, b, m, dumps, 600, [&] { return (got += b.session().read().size()) >= up.size(); }, seed + 2), "delivered");
    run(a, b, m, dumps, 3, [] { return false; }, seed + 3);  // the monitor is a block behind
    const std::string out = all(dumps);
    const Bytes seen = stream(dumps, "W1AW>K2XYZ stream");
    const bool gap = out.find("not all heard") != std::string::npos;
    std::fprintf(stderr, "lossy %.0f dB: %zu bursts, resends %s, gap %d, %zu of %zu B shown\n", snr_db, dumps.size(),
                 out.find("r ok") != std::string::npos ? "read" : "none read", gap, seen.size(), up.size());
    check::is_true(gap || seen == up, "no gap: the stream as sent");
    check::is_true(std::search(up.begin(), up.end(), seen.begin(), seen.begin() + std::min<std::size_t>(seen.size(), 200)) != up.end(),
                   "what was shown is the stream's");
    if (check::failures || std::getenv("SHOW")) std::fputs(out.c_str(), stderr);
    g_snr_db = 12.0;
}

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog dog(900, "test_monitor");
    units();
    session();
    broadcast();
    for (double snr : {2.0, 0.0, -2.0}) lossy(snr, static_cast<unsigned>(50 + snr));
    return check::report("test_monitor");
}
