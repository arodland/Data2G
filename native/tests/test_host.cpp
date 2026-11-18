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
    bool a_deaf = false;  // a hears only the noise
    bool a_unserved = false;  // a's after_step is not called: its events stay undrained

    Pair(arq::EngineConfig ca, arq::EngineConfig cb) : ea("NOCALL", ca), eb("NOCALL", cb), a(ea), b(eb) {}

    // Both step through a 12 dB channel; each host's after_step, as the app does.
    bool run(double seconds, const std::function<bool()>& until) {
        std::normal_distribution<double> noise(0.0, std::sqrt((config::FS / 2.0) / config::SNR_REF_BW_HZ / std::pow(10.0, 1.2)));
        std::vector<double> x(BLOCK);
        for (int i = 0; i < static_cast<int>(seconds * 10); ++i) {
            for (int j = 0; j < BLOCK; ++j) x[j] = (a_deaf ? 0.0 : 2.2 * bo[j]) + noise(rng);
            auto oa = ea.step(x);
            if (!a_unserved) a.after_step(oa.ptt);
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
    const std::vector<std::string> want = {"VERSION Data2G " DATA2G_VERSION, "OK", "REGISTERED K2XYZ", "REGISTERED K2XYZ-1", "OK", "OK", "OK",
                                           "OK", "OK", "WRONG", "WRONG", "WRONG", "WRONG"};
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

void accept() {
    check::current_step = "accept";
    check::is_true(!host::parse_addr("8301") && !host::parse_addr(":x") && !host::parse_addr(":0") && !host::parse_addr(":70000"),
                   "bad addresses");
    const auto local = host::parse_addr(":8400"), named = host::parse_addr("[::1]:22");
    check::is_true(local && local->host == "127.0.0.1" && local->port == 8400, ":port is localhost");
    check::is_true(named && named->host == "::1" && named->port == 22, "[host]:port");
    const auto from = host::parse_addr(":8400", "192.0.2.7");
    check::is_true(from && from->host == "192.0.2.7", ":port is the command client's host");

    Pair p(seeded(21), seeded(22));
    std::vector<int> dialed;
    bool up = false;
    p.b.dial = [&](const host::Addr& a) {
        dialed.push_back(a.port);
        return up;
    };
    for (const char* line : {"MYCALL K2XYZ", "ACCEPT :8400"}) p.b.command(line);
    check::is_true(p.b.listening && p.eb.session().state == arq::SessionState::LISTEN, "ACCEPT listens");
    p.a.command("CONNECT W1AW K2XYZ");
    p.take();
    check::is_true(p.run(60, [&] { return has(p.seen_a, "DISCONNECTED"); }), "no connection: refused");
    check::equal(p.ea.session().close_reason, std::string("REFUSED"), "the NAK's reason");
    check::is_true(dialed == std::vector<int>{8400} && !has(p.seen_b, "CONNECTED W1AW K2XYZ 2300"), "dialed once, never connected");
    up = true;
    p.a.command("CONNECT W1AW K2XYZ");
    check::is_true(p.run(60, [&] { return has(p.seen_a, "CONNECTED W1AW K2XYZ 2300"); }), "connected once dialed");
    p.b.command("LISTEN ON");
    check::is_true(!p.b.accept_addr, "LISTEN ON ends ACCEPT");
}

void dial_given_up() {
    check::current_step = "dial given up";
    Engine e("NOCALL", seeded(4));
    host::Host h(e);
    for (const char* line : {"MYCALL K2XYZ", "LISTEN ON"}) h.command(line);
    const std::vector<double> quiet(BLOCK, 0.0);
    h.out_cmd.clear();
    h.command("CONNECT K2XYZ W1AW");
    h.command("DISCONNECT");
    check::is_true(h.out_cmd == std::vector<std::string>{"OK", "OK"}, "DISCONNECT before the CONNECT is on air: OK first");
    h.command("CONNECT K2XYZ W1AW");
    h.after_step(false);
    check::is_true(has(h.out_cmd, "DISCONNECTED") && !e.tx(), "... the call ends unsent, its DISCONNECTED not lost to the next CONNECT");
    for (int i = 0; i < 50 && !e.tx(); ++i) h.after_step(e.step(quiet).ptt);
    check::is_true(e.tx().has_value() && e.session().state == arq::SessionState::CONNECTING, "a CONNECT on air");
    h.out_cmd.clear();
    h.command("DISCONNECT");
    check::is_true(h.out_cmd == std::vector<std::string>{"OK"} && e.tx().has_value(), "DISCONNECT while dialing: OK, the CONNECT not cut");
    int keyed = 0, blocks = 0;
    bool closed_at_key = true;
    for (bool was = true; blocks < 300; ++blocks) {
        h.after_step(e.step(quiet).ptt);
        if (e.tx() && !was) {
            ++keyed;
            closed_at_key = has(h.out_cmd, "DISCONNECTED");
        }
        was = e.tx().has_value();
    }
    check::is_true(keyed == 1 && !closed_at_key && has(h.out_cmd, "DISCONNECTED"),
                   "... its answer waited for, then one DISC, and DISCONNECTED once it is out");
    check::is_true(e.session().state == arq::SessionState::LISTEN, "... listening again");
    h.command("CONNECT K2XYZ W1AW");
    for (int i = 0; i < 300 && e.session().tries == 0; ++i) h.after_step(e.step(quiet).ptt);
    check::is_true(e.session().tries == 1 && !e.tx(), "between tries");
    h.out_cmd.clear();
    h.command("DISCONNECT");
    for (int i = 0; i < 10 && !e.tx(); ++i) h.after_step(e.step(quiet).ptt);
    check::is_true(e.tx().has_value() && !has(h.out_cmd, "DISCONNECTED"), "DISCONNECT between tries: the DISC at once");
    for (int i = 0; i < 60 && !has(h.out_cmd, "DISCONNECTED"); ++i) h.after_step(e.step(quiet).ptt);
    check::is_true(has(h.out_cmd, "DISCONNECTED") && !e.tx(), "... and DISCONNECTED once it is out");

    h.command("CONNECT K2XYZ W1AW");
    for (int i = 0; i < 50 && !e.tx(); ++i) h.after_step(e.step(quiet).ptt);
    for (int i = 0; i < 50 && e.tx(); ++i) h.after_step(e.step(quiet).ptt);
    {
        auto& s = e.session();
        const std::string narrow = s.policy->connect_mode(0, 0);
        check::is_true(s.deadline.has_value() && narrow != s.policy->connect_mode(2, 0), "awaiting the answer");
        s.on_header(narrow, 1, e.now());
        check::is_true(*s.deadline == e.now() + s.policy->airtime(narrow, 1, false) + s.t_turn + s.tune.reply_start_s,
                       "an ID in a narrower cap's connect mode holds the reply window as one in mine");
    }
    for (int i = 0; i < 300 && e.session().tries == 0; ++i) h.after_step(e.step(quiet).ptt);
    {
        auto& s = e.session();
        const std::string mode = s.policy->connect_mode(2, 0);
        const double air = s.policy->airtime(mode, 4, false);
        check::is_true(s.tries == 1 && !e.tx(), "between tries again");
        s.on_header(mode, 4, e.now());
        const double clear = e.now() + air + s.t_turn;
        h.command("DISCONNECT");
        check::is_true(s.bye && s.out == s.bye && !s.deadline && *s.due == clear,
                       "DISCONNECT between tries queues the DISC, clear of a burst heard before it");
        s.on_header(mode, 4, e.now() + 1.0);
        check::is_true(*s.due == e.now() + 1.0 + air + s.t_turn, "... and of one heard while it is queued");
    }
    h.command("ABORT");

    Pair q(seeded(43), seeded(44));
    for (const char* line : {"MYCALL K2XYZ", "LISTEN ON"}) q.b.command(line);
    q.a.command("CONNECT W1AW K2XYZ");
    check::is_true(q.run(10, [&] { return q.ea.tx().has_value(); }), "a CONNECT on air to a listening callee");
    q.a.command("DISCONNECT");
    int ups = 0;
    bool up = true;
    check::is_true(q.run(60, [&] {
        ups += q.ea.tx().has_value() && !up;
        up = q.ea.tx().has_value();
        return has(q.seen_a, "DISCONNECTED") && has(q.seen_b, "DISCONNECTED");
    }) && ups == 1 && has(q.seen_a, "CONNECTED W1AW K2XYZ 2300"), "a callee that answers a dial given up gets the DISC, no data burst ahead of it");

    Pair r(seeded(45), seeded(46));
    for (const char* line : {"MYCALL K2XYZ", "LISTEN ON"}) r.b.command(line);
    r.a.command("CONNECT W1AW K2XYZ");
    r.a_unserved = true;
    check::is_true(r.run(20, [&] { return r.ea.session().state == arq::SessionState::CONNECTED; }), "answered, CONNECTED not yet told");
    r.a.out_cmd.clear();
    r.a.command("ABORT");
    r.a.after_step(false);
    {
        const auto& o = r.a.out_cmd;
        const auto c = std::find(o.begin(), o.end(), "CONNECTED W1AW K2XYZ 2300"), d = std::find(o.begin(), o.end(), "DISCONNECTED");
        check::is_true(c != o.end() && c < d && std::count(o.begin(), o.end(), "DISCONNECTED") == 1,
                       "ABORT tells the CONNECTED it overtook, then one DISCONNECTED");
    }

    Pair p(seeded(41), seeded(42));
    for (const char* line : {"MYCALL K2XYZ", "LISTEN ON"}) p.b.command(line);
    p.a.command("CONNECT W1AW K2XYZ");
    check::is_true(p.run(10, [&] { return p.ea.tx().has_value(); }), "a CONNECT on air to a listening callee");
    p.a.command("DISCONNECT");
    p.a_deaf = true;
    check::is_true(p.run(30, [&] { return has(p.seen_b, "CONNECTED W1AW K2XYZ 2300") && has(p.seen_b, "DISCONNECTED"); }),
                   "a callee whose CONNECT_ACK was lost is freed by the DISC");
}

void operator_abort() {
    check::current_step = "operator abort";
    Pair p(seeded(31), seeded(32));
    for (const char* line : {"MYCALL K2XYZ", "LISTEN ON"}) p.b.command(line);
    p.b.out_cmd.clear();
    check::is_true(!p.b.abort_session() && p.b.out_cmd.empty(), "only listening: nothing to abort, nothing sent");
    p.a.command("CONNECT W1AW K2XYZ");
    p.take();
    check::is_true(p.run(60, [&] { return has(p.seen_a, "CONNECTED W1AW K2XYZ 2300") && has(p.seen_b, "CONNECTED W1AW K2XYZ 2300"); }),
                   "connected");
    check::is_true(p.b.abort_session(), "aborted");
    check::is_true(p.b.out_cmd == std::vector<std::string>{"DISCONNECTED"}, "the client told DISCONNECTED (no OK: not its command)");
    check::is_true(p.eb.session().state == arq::SessionState::LISTEN, "still listening");
    p.b.out_cmd.clear();
    check::is_true(!p.b.abort_session() && p.b.out_cmd.empty(), "again: nothing");
}

void txpower() {
    check::current_step = "TXPOWER";
    Engine e("W1AW", seeded(5));
    host::Host h(e);
    h.command("TXPOWER");  // Data2G only
    h.command("DATA2G ON");
    for (const char* line : {"TXPOWER", "TXPOWER -1.2", "TXPOWER", "TXPOWER -60.1", "TXPOWER 0.1", "TXPOWER abc", "TXPOWER nan",
                             "TXPOWER 1 2", "TXPOWER 1x", "TXPOWER"})
        h.command(line);
    const std::vector<std::string> want = {"WRONG", "OK", "TXPOWER 0.0", "OK", "TXPOWER -1.2", "WRONG", "WRONG", "WRONG", "WRONG",
                                           "WRONG", "WRONG", "TXPOWER -1.2"};
    check::is_true(h.out_cmd == want, "replies");
    check::equal(h.output_volume, -1.2, "volume");
}

void cwid_command() {
    check::current_step = "CWID";
    Engine e("W1AW", seeded(6));
    host::Host h(e);
    h.command("CWID ON");  // a VARA client's: ignored
    check::is_true(e.cw_id == "off", "ignored without DATA2G ON");
    h.command("DATA2G ON");
    for (const char* line : {"CWID ONLY 25", "CWID ON 60", "CWID ON x", "CWID MAYBE", "CWID ON"}) h.command(line);
    const std::vector<std::string> want = {"OK", "OK", "OK", "WRONG", "WRONG", "WRONG", "OK"};
    check::is_true(h.out_cmd == want, "replies (host.py's)");
    check::is_true(e.cw_id == "add" && e.cw_wpm == 25.0, "set");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog dog(600, "test_host");
    commands();
    session();
    accept();
    dial_given_up();
    operator_abort();
    txpower();
    cwid_command();
    return check::report("test_host");
}
