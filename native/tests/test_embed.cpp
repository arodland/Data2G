// The embedding library through its C ABI (embed/include/CData2G/data2g.h, docs/EMBEDDING.md), as an
// application drives it: two modems back to back over a noisy channel, each with an "application" that
// plays what it is handed a tick at a time and reports what it played. A session with data both ways; the
// radio keyed and released around what was really played; ABORT, the watchdog, a fault and d2g_free, each
// leaving nothing queued and the PTT off; commands from inside a callback.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "CData2G/data2g.h"
#include "arq/modes.hpp"
#include "check.hpp"
#include "embed/modem.hpp"

using namespace data2g;

namespace {

// the engine's TX scale on HF (arq/phy full_scale), as test_host.cpp's channel
constexpr double LINK_G = 2.2;

bool starts(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

// An application: owns the audio and the PTT. One tick of its clock plays a tick of what is queued.
struct App {
    D2gModem* m = nullptr;
    int rate = 48000;
    std::size_t tick = 960;  // samples a tick: 20 ms
    // what the modem asked for
    std::vector<float> queue;  // handed over, not yet played
    std::uint64_t handed = 0, played = 0;
    bool ptt = false;
    int keys = 0, unkeys = 0, flushes = 0, bursts = 0;
    float peak = 0;
    std::vector<std::string> lines, log, events;  // events: every callback's name, in order
    std::vector<std::uint8_t> data;
    struct Tx {
        std::string mode;
        D2gTxReport r;
    };
    struct Rx {
        std::string mode;
        D2gRxReport r;
    };
    std::vector<Tx> tx;
    std::vector<Rx> rx;
    // what it noticed
    bool audio_unkeyed = false;     // tx_audio with the PTT off
    bool unkey_with_audio = false;  // ptt(false) with audio still queued
    std::uint64_t fed = 0;          // samples given to the modem: its clock
    std::uint64_t drained_at = 0;   // fed when the last of a transmission was played
    std::uint64_t min_tail = UINT64_MAX, max_tail = 0;  // from there to ptt(false)
    int hold = 0, holding = 0;      // ticks from tx_audio to the first sample played: the output's latency
    bool never_play = false;
    bool report_in_flush = false;   // tx_played from inside tx_flush
    std::function<void(App&, const std::string&)> on_line;
    std::function<void(App&)> on_tx_audio;

    static App& of(void* ctx) { return *static_cast<App*>(ctx); }

    D2gCallbacks callbacks() {
        D2gCallbacks c{};
        c.tx_audio = [](void* ctx, const float* s, std::size_t n) {
            App& a = of(ctx);
            a.events.emplace_back("tx_audio");
            if (!a.ptt) a.audio_unkeyed = true;
            if (a.queue.empty()) a.holding = a.hold;
            a.queue.insert(a.queue.end(), s, s + n);
            a.handed += n;
            ++a.bursts;
            for (std::size_t i = 0; i < n; ++i) a.peak = std::max(a.peak, std::abs(s[i]));
            if (a.on_tx_audio) a.on_tx_audio(a);
        };
        c.tx_flush = [](void* ctx) {
            App& a = of(ctx);
            a.events.emplace_back("tx_flush");
            ++a.flushes;
            a.queue.clear();
            if (a.report_in_flush && a.m) d2g_tx_played(a.m, a.played);
        };
        c.ptt = [](void* ctx, bool on) {
            App& a = of(ctx);
            a.events.emplace_back(on ? "ptt on" : "ptt off");
            if (on) {
                ++a.keys;
            } else if (a.ptt) {
                ++a.unkeys;
                if (!a.queue.empty()) a.unkey_with_audio = true;
                a.min_tail = std::min(a.min_tail, a.fed - a.drained_at);
                a.max_tail = std::max(a.max_tail, a.fed - a.drained_at);
            }
            a.ptt = on;
        };
        c.host_line = [](void* ctx, const char* line) {
            App& a = of(ctx);
            a.events.emplace_back("host_line");
            a.lines.emplace_back(line);
            if (a.on_line) a.on_line(a, a.lines.back());
        };
        c.host_data = [](void* ctx, const std::uint8_t* d, std::size_t n) {
            App& a = of(ctx);
            a.events.emplace_back("host_data");
            a.data.insert(a.data.end(), d, d + n);
        };
        c.log = [](void* ctx, const char* line) {
            App& a = of(ctx);
            a.events.emplace_back("log");
            a.log.emplace_back(line);
        };
        c.tx_report = [](void* ctx, const D2gTxReport* r) {
            App& a = of(ctx);
            a.events.emplace_back("tx_report");
            a.tx.push_back({r->mode, *r});
        };
        c.rx_report = [](void* ctx, const D2gRxReport* r) {
            App& a = of(ctx);
            a.events.emplace_back("rx_report");
            a.rx.push_back({r->mode, *r});
        };
        return c;
    }

    void open(std::uint32_t seed, const std::function<void(D2gConfig&)>& tune = {}) {
        D2gConfig cfg;
        d2g_config_default(&cfg);
        cfg.sample_rate = static_cast<std::uint32_t>(rate);
        cfg.seed = seed;
        if (tune) tune(cfg);
        tick = static_cast<std::size_t>(rate) / 50;
        const D2gCallbacks c = callbacks();
        m = d2g_new(&cfg, &c, this);
    }
    void close() {
        if (m) d2g_free(m);
        m = nullptr;
    }
    void command(const char* line) { d2g_command(m, line); }
    bool has(const std::string& line) const { return std::find(lines.begin(), lines.end(), line) != lines.end(); }
    bool line_starting(const char* p) const {
        return std::any_of(lines.begin(), lines.end(), [&](const std::string& s) { return starts(s, p); });
    }
    int count(const std::string& line) const { return static_cast<int>(std::count(lines.begin(), lines.end(), line)); }
    bool logged(const char* part) const {
        return std::any_of(log.begin(), log.end(), [&](const std::string& s) { return s.find(part) != std::string::npos; });
    }

    // A tick's worth out of the queue to the air, and the running total to the modem.
    std::vector<float> play() {
        std::vector<float> air(tick, 0.0f);
        if (ptt && !never_play && !queue.empty()) {
            if (holding > 0) {
                --holding;
            } else {
                const std::size_t n = std::min(tick, queue.size());
                std::copy_n(queue.begin(), n, air.begin());
                queue.erase(queue.begin(), queue.begin() + static_cast<std::ptrdiff_t>(n));
                played += n;
                if (queue.empty()) drained_at = fed;
            }
        }
        if (m) d2g_tx_played(m, played);
        return air;
    }
    void hear(const std::vector<float>& x) {
        if (!m) return;
        fed += x.size();
        d2g_rx_audio(m, x.data(), x.size());
    }
    void wait() {  // a tick with no capture
        if (!m) return;
        fed += tick;
        d2g_advance(m, tick);
    }
};

// Two applications on one channel at `snr_db` (in config::SNR_REF_BW_HZ, as the engine tests').
struct Pair {
    App a, b;
    std::mt19937 rng{1};
    double noise_std;

    Pair(int rate, double snr_db) {
        a.rate = b.rate = rate;
        // white at the application's rate with the density the engine tests' has at FS
        noise_std = std::sqrt((rate / 2.0) / config::SNR_REF_BW_HZ / std::pow(10.0, snr_db / 10));
    }
    ~Pair() {
        a.close();
        b.close();
    }

    bool run(double seconds, const std::function<bool()>& until) {
        std::normal_distribution<float> noise(0.0f, static_cast<float>(noise_std));
        std::vector<float> x(a.tick);
        for (int i = 0; i < static_cast<int>(seconds * 50); ++i) {
            const auto air_a = a.play(), air_b = b.play();
            // a: a radio that delivers nothing while it is keyed. b: one that hears itself (a monitor), which
            // the modem must not
            if (a.ptt) {
                a.wait();
            } else {
                for (std::size_t j = 0; j < x.size(); ++j) x[j] = static_cast<float>(LINK_G) * air_b[j] + noise(rng);
                a.hear(x);
            }
            for (std::size_t j = 0; j < x.size(); ++j)
                x[j] = static_cast<float>(LINK_G) * (b.ptt ? air_b[j] : air_a[j]) + noise(rng);
            b.hear(x);
            if (until()) return true;
        }
        return false;
    }
};

// One application and nobody to hear it.
bool run_alone(App& a, double seconds, const std::function<bool()>& until) {
    for (int i = 0; i < static_cast<int>(seconds * 50); ++i) {
        a.play();
        a.wait();
        if (until()) return true;
    }
    return false;
}

void keying(const App& x, const char* who) {
    const std::string w = std::string(who) + ": ";
    check::is_true(x.keys > 0 && x.keys == x.unkeys && !x.ptt, w + "keyed, and released as often");
    check::equal(x.count("PTT ON"), x.keys, w + "PTT ON with each key");
    check::equal(x.count("PTT OFF"), x.unkeys, w + "PTT OFF with each release");
    check::is_true(!x.audio_unkeyed, w + "audio only while keyed");
    check::is_true(!x.unkey_with_audio, w + "never released with audio queued");
    check::equal(x.flushes, 0, w + "nothing flushed");
    check::equal(static_cast<long long>(x.played), static_cast<long long>(x.handed), w + "everything handed over was played");
    // ptt_off_delay_s (50 ms) on the sample clock, from the last sample played; noticed within a tick or two
    const auto off = static_cast<std::uint64_t>(0.05 * x.rate);
    check::is_true(x.min_tail >= off && x.max_tail <= off + 3 * x.tick, w + "released the off delay after the last sample");
    check::is_true(x.peak > 0.1f && x.peak <= 1.0f, w + "audio within full scale");
}

void session() {
    check::current_step = "session";
    Pair p(48000, 12);
    p.a.open(11);
    p.b.open(12);
    check::is_true(p.a.m && p.b.m, "two modems");
    if (!p.a.m || !p.b.m) return;
    p.a.tick = p.b.tick = 1000;  // not a whole number of the engine's samples: the resampler keeps its place across calls
    const std::string connected = "CONNECTED W1AW K2XYZ " + arq::bw_name(2);

    std::mt19937 rng(5);
    std::vector<std::uint8_t> up(3000), down(700);
    for (auto& v : up) v = static_cast<std::uint8_t>(rng());
    for (auto& v : down) v = static_cast<std::uint8_t>(rng());
    // from inside a callback: data and a command are taken and run before the outer call returns; audio is not
    bool nested = false;
    p.b.on_line = [&](App& b, const std::string& line) {
        if (line != connected || nested) return;
        nested = true;
        d2g_send(b.m, down.data(), down.size());
        d2g_command(b.m, "VERSION");
        const std::vector<float> zeros(480, 0.0f);
        d2g_rx_audio(b.m, zeros.data(), zeros.size());
        d2g_advance(b.m, 480);
    };

    p.a.command("VERSION");
    check::equal(p.a.lines.empty() ? std::string() : p.a.lines.back(), std::string("VERSION ") + d2g_version(),
                 "a reply before d2g_command returns");
    p.a.command("ACCEPT :8400");
    check::equal(p.a.lines.back(), std::string("WRONG"), "ACCEPT refused: no port to dial");
    p.a.command("MYC\xC3\x85LL W1AW");
    check::equal(p.a.lines.back(), std::string("WRONG"), "non-ASCII replaced, not passed on");
    for (const char* line : {"MYCALL K2XYZ", "LISTEN ON"}) p.b.command(line);
    check::is_true(p.b.has("REGISTERED K2XYZ"), "MYCALL");
    p.a.command("MYCALL W1AW");
    p.a.command("CONNECT W1AW K2XYZ");
    check::is_true(p.run(60, [&] { return p.a.has(connected) && p.b.has(connected); }), "CONNECTED on both");
    check::is_true(p.a.has("PTT ON") && p.b.has("BUSY ON"), "PTT and BUSY");
    check::is_true(nested && p.b.has(std::string("VERSION ") + d2g_version()), "a command from inside a callback ran");
    check::is_true(p.b.logged("d2g_rx_audio from inside a callback") && p.b.logged("d2g_advance from inside a callback"),
                   "audio from inside a callback: refused, logged");

    d2g_send(p.a.m, up.data(), up.size());
    check::is_true(p.run(180, [&] { return p.b.data.size() >= up.size() && p.a.data.size() >= down.size(); }), "data both ways");
    check::is_true(p.b.data == up, "data intact, a to b");
    check::is_true(p.a.data == down, "data intact, b to a (sent from inside a callback)");
    std::vector<long> buffers;
    for (const auto& s : p.a.lines)
        if (starts(s, "BUFFER ")) buffers.push_back(std::stol(s.substr(7)));
    check::is_true(!buffers.empty() && *std::max_element(buffers.begin(), buffers.end()) > 0, "BUFFER while queued");
    p.run(20, [&] { return p.a.has("BUFFER 0"); });
    check::is_true(p.a.has("BUFFER 0"), "BUFFER 0 once acked");

    p.a.command("DISCONNECT");
    check::is_true(p.run(60, [&] { return p.a.has("DISCONNECTED") && p.b.has("DISCONNECTED"); }), "DISCONNECTED on both");
    p.run(15, [&] { return !p.a.ptt && !p.b.ptt && p.a.queue.empty() && p.b.queue.empty() && p.a.keys == p.a.unkeys && p.b.keys == p.b.unkeys; });
    keying(p.a, "a");
    keying(p.b, "b");

    // reports
    const float top = d2g_max_net_bps(D2G_BW_2300);
    // 1200's fastest is a 500 Hz mode's on HF, so the two may be equal
    check::is_true(top > d2g_max_net_bps(D2G_BW_1200) && d2g_max_net_bps(D2G_BW_1200) >= d2g_max_net_bps(D2G_BW_500) &&
                       d2g_max_net_bps(D2G_BW_500) > 0 && d2g_max_net_bps(3) == 0,
                   "max net bps by bandwidth");
    long tx_bytes = 0, rx_bytes = 0;
    bool tx_ok = !p.a.tx.empty(), rx_ok = !p.b.rx.empty(), snr = false;
    for (const auto& t : p.a.tx) {
        tx_bytes += t.r.bytes;
        tx_ok = tx_ok && !t.mode.empty() && t.r.codewords >= 1 && t.r.data_codewords <= t.r.codewords && t.r.air_s > 0 &&
                t.r.width_hz > 0 && t.r.net_bps >= 0 && t.r.net_bps <= top;
    }
    for (const auto& r : p.b.rx) {
        rx_bytes += r.r.bytes;
        rx_ok = rx_ok && !r.mode.empty() && r.r.codewords >= 1 && r.r.codewords_ok + r.r.codewords_failed <= r.r.codewords &&
                r.r.air_s > 0 && r.r.net_bps <= top && (r.r.lost ? std::isnan(r.r.snr_db) && r.r.codewords_ok == 0 : true);
        snr = snr || (!r.r.lost && std::isfinite(r.r.snr_db) && r.r.snr_db > 0 && r.r.snr_db < 40);
    }
    check::is_true(tx_ok, "TX reports: mode, codewords, time on air, a rate within the bandwidth's best");
    check::is_true(rx_ok && snr, "RX reports: mode, codewords, S/N");
    // the stream's bytes (records framed: a little over what was sent), each counted once on either side
    check::is_true(tx_bytes >= static_cast<long>(up.size()) && tx_bytes <= static_cast<long>(up.size()) + 64,
                   "TX reports count the data sent, once");
    check::is_true(rx_bytes >= tx_bytes && rx_bytes < tx_bytes + 1024, "RX reports count the data taken, once (and its padding)");
    check::is_true(std::any_of(p.a.tx.begin(), p.a.tx.end(), [](const App::Tx& t) { return t.r.data_codewords > 0 && t.r.net_bps > 0; }),
                   "a data burst's rate");
    check::is_true(std::any_of(p.a.rx.begin(), p.a.rx.end(), [](const App::Rx& r) { return !r.r.lost; }), "the caller heard its peer");
    check::is_true(!p.a.log.empty(), "the log reaches the application");

    p.a.close();
    check::equal(p.a.flushes, 1, "free: a last tx_flush");
    check::is_true(p.a.events.size() >= 2 && p.a.events[p.a.events.size() - 2] == "tx_flush" && p.a.events.back() == "ptt off",
                   "free: tx_flush and ptt(false) are the last callbacks");
}

// The engine's own rate, no resampling: a connection and its end.
void rate_8000() {
    check::current_step = "8000 Hz";
    Pair p(8000, 12);
    p.a.open(21);
    p.b.open(22);
    check::is_true(p.a.m && p.b.m, "two modems at 8000 Hz");
    if (!p.a.m || !p.b.m) return;
    const std::string connected = "CONNECTED W1AW K2XYZ " + arq::bw_name(2);
    for (const char* line : {"MYCALL K2XYZ", "LISTEN ON"}) p.b.command(line);
    p.a.command("MYCALL W1AW");
    p.a.command("CONNECT W1AW K2XYZ");
    check::is_true(p.run(60, [&] { return p.a.has(connected) && p.b.has(connected); }), "CONNECTED on both");
    const std::vector<std::uint8_t> hello = {'h', 'e', 'l', 'l', 'o'};
    d2g_send(p.b.m, hello.data(), hello.size());
    check::is_true(p.run(60, [&] { return p.a.data.size() >= hello.size(); }) && p.a.data == hello, "data");
    p.b.command("DISCONNECT");
    check::is_true(p.run(60, [&] { return p.a.has("DISCONNECTED") && p.b.has("DISCONNECTED"); }), "DISCONNECTED on both");
    p.run(15, [&] { return !p.a.ptt && !p.b.ptt && p.a.keys == p.a.unkeys && p.b.keys == p.b.unkeys; });
    keying(p.a, "a");
    keying(p.b, "b");
}

// A poor channel: bursts lost and sent again, and the keying still exact.
void weak() {
    check::current_step = "weak channel";
    Pair p(48000, -6);
    p.a.open(81);
    p.b.open(82);
    if (!p.a.m || !p.b.m) return check::is_true(false, "two modems");
    const std::string connected = "CONNECTED W1AW K2XYZ " + arq::bw_name(0);
    for (const char* line : {"MYCALL K2XYZ", "LISTEN ON"}) p.b.command(line);
    for (const char* line : {"MYCALL W1AW", "BW500", "CONNECT W1AW K2XYZ"}) p.a.command(line);
    check::is_true(p.run(120, [&] { return p.a.has(connected) && p.b.has(connected); }), "CONNECTED on both, 500 Hz");
    std::mt19937 rng(9);
    std::vector<std::uint8_t> up(600);
    for (auto& v : up) v = static_cast<std::uint8_t>(rng());
    d2g_send(p.a.m, up.data(), up.size());
    check::is_true(p.run(600, [&] { return p.b.data.size() >= up.size(); }) && p.b.data == up, "data intact");
    const float top = d2g_max_net_bps(D2G_BW_500);
    check::is_true(std::all_of(p.a.tx.begin(), p.a.tx.end(), [&](const App::Tx& t) { return t.r.width_hz <= 500 && t.r.net_bps <= top; }),
                   "BW500: every burst within 500 Hz and its best rate");
    check::is_true(std::all_of(p.b.rx.begin(), p.b.rx.end(), [&](const App::Rx& r) { return r.r.net_bps <= top; }),
                   "... heard so too");
    p.a.command("DISCONNECT");
    check::is_true(p.run(120, [&] { return p.a.has("DISCONNECTED") && p.b.has("DISCONNECTED"); }), "DISCONNECTED on both");
    p.run(15, [&] { return !p.a.ptt && !p.b.ptt && p.a.keys == p.a.unkeys && p.b.keys == p.b.unkeys; });
    keying(p.a, "a");
    keying(p.b, "b");
}

// ABORT stops a transmission at once: flushed, the radio released, and the count of what was played still
// right for the next one.
void abort_flushes() {
    check::current_step = "abort";
    App a;
    a.open(31, [](D2gConfig& c) { c.tx_gain = 0.5f; });
    check::is_true(a.m != nullptr, "a modem");
    if (!a.m) return;
    a.command("MYCALL W1AW");
    a.command("CONNECT W1AW K2XYZ");
    check::is_true(run_alone(a, 10, [&] { return a.ptt && a.played > 9600; }), "calling: keyed, playing");
    check::is_true(!a.queue.empty(), "mid-burst");
    a.command("ABORT");
    check::equal(a.flushes, 1, "ABORT: flushed before d2g_command returns");
    check::is_true(!a.ptt && a.unkeys == 1, "ABORT: PTT off before d2g_command returns");
    check::is_true(a.lines.size() >= 3 && std::vector<std::string>(a.lines.end() - 3, a.lines.end()) ==
                                                  std::vector<std::string>{"DISCONNECTED", "OK", "PTT OFF"},
                   "ABORT: DISCONNECTED, OK, PTT OFF");
    check::is_true(a.played < a.handed, "part of it was never played");
    check::is_true(a.peak > 0.05f && a.peak <= 0.5f, "tx_gain applied");
    run_alone(a, 2, [] { return false; });
    check::is_true(a.keys == 1 && a.flushes == 1, "quiet after it");

    // again, with the output half a second behind: the engine's burst is over while its end is still
    // playing, and an ABORT there is the next block's to act on. The count is reported from inside tx_flush.
    a.hold = 25;
    a.report_in_flush = true;
    a.command("CONNECT W1AW K2XYZ");
    check::is_true(run_alone(a, 20, [&] { return a.ptt && !a.queue.empty() && a.queue.size() < 0.3 * a.rate; }), "the tail of a late burst");
    a.command("ABORT");
    run_alone(a, 0.3, [] { return false; });
    check::is_true(a.flushes == 2 && !a.ptt && a.unkeys == 2, "ABORT in the tail: flushed and released within a block");
    a.hold = 0;

    // and a whole burst after the two cut ones: released when it, not they, has been played
    a.command("CONNECT W1AW K2XYZ");
    check::is_true(run_alone(a, 10, [&] { return a.keys == 3; }), "calling again");
    const std::uint64_t before = a.played;
    check::is_true(run_alone(a, 30, [&] { return a.unkeys == 3; }), "released");
    check::is_true(a.queue.empty() && a.played > before && !a.unkey_with_audio && a.flushes == 2, "after its last sample");
    a.command("ABORT");
    a.close();
}

// A transmission the application never plays: flushed and the radio released WATCHDOG_S past its length.
void watchdog() {
    check::current_step = "watchdog";
    App a;
    a.never_play = true;
    a.open(41);
    if (!a.m) return check::is_true(false, "a modem");
    a.command("MYCALL W1AW");
    a.command("CONNECT W1AW K2XYZ");
    check::is_true(run_alone(a, 10, [&] { return a.ptt; }), "keyed");
    const std::uint64_t at = a.fed, len = a.handed;
    check::is_true(run_alone(a, 60, [&] { return !a.ptt; }), "released");
    const double waited = static_cast<double>(a.fed - at) / a.rate, want = static_cast<double>(len) / a.rate + 3.0;
    check::is_true(waited >= want - 0.05 && waited <= want + 0.2, "its length and 3 s after it was handed over");
    check::is_true(a.flushes == 1 && a.unkeys == 1 && a.played == 0, "flushed, nothing played");
    check::is_true(a.logged("transmit audio not played"), "logged");
    a.command("ABORT");
    a.close();
}

// d2g_free from inside a callback: nothing after it but tx_flush and ptt(false), when the outer call returns.
void free_inside() {
    check::current_step = "free inside a callback";
    App a;
    a.open(51);
    if (!a.m) return check::is_true(false, "a modem");
    std::size_t at = 0;
    a.on_tx_audio = [&](App& x) {
        at = x.events.size();
        D2gModem* m = x.m;
        x.m = nullptr;
        d2g_free(m);
        d2g_free(m);  // twice from in here: once
    };
    a.command("MYCALL W1AW");
    a.command("CONNECT W1AW K2XYZ");
    D2gModem* m = a.m;
    for (int i = 0; i < 500 && a.m; ++i) {  // not App::wait: m is gone once the call returns
        a.fed += a.tick;
        d2g_advance(m, a.tick);
    }
    check::is_true(a.m == nullptr, "freed from tx_audio");
    const std::vector<std::string> after(a.events.begin() + static_cast<std::ptrdiff_t>(at), a.events.end());
    check::is_true(after == std::vector<std::string>{"tx_flush", "ptt off"}, "then only tx_flush and ptt(false)");
    check::is_true(!a.ptt && a.flushes == 1, "the radio released");
}

// d2g_free while keyed, and a fault: an exception that reaches the boundary (here, one of the application's
// own) flushes, releases the radio and says so; later calls do nothing.
void free_keyed_and_fault() {
    check::current_step = "free while keyed";
    {
        App a;
        a.open(61);
        if (!a.m) return check::is_true(false, "a modem");
        a.command("MYCALL W1AW");
        a.command("CONNECT W1AW K2XYZ");
        check::is_true(run_alone(a, 10, [&] { return a.ptt && a.played > 0; }), "keyed");
        a.close();
        check::is_true(!a.ptt && a.flushes == 1 && a.queue.empty(), "freed while keyed: flushed, PTT off");
    }
    check::current_step = "fault";
    App a;
    a.open(62);
    if (!a.m) return check::is_true(false, "a modem");
    a.command("MYCALL W1AW");
    a.command("CONNECT W1AW K2XYZ");
    check::is_true(run_alone(a, 10, [&] { return a.ptt && a.played > 0; }), "keyed");
    a.on_line = [](App&, const std::string& line) {
        if (line == "OK") throw std::runtime_error("boom");
    };
    a.command("LISTEN ON");
    check::is_true(!a.ptt && a.flushes == 1 && a.queue.empty(), "fault: flushed, PTT off");
    check::is_true(a.logged("modem failed (boom)"), "fault: logged");
    a.on_line = nullptr;
    const std::size_t n = a.events.size();
    a.command("VERSION");
    run_alone(a, 1, [] { return false; });
    check::equal(a.events.size(), n, "fault: later calls ignored");
    a.close();
    check::equal(a.events.size(), n, "fault: the free has nothing left to release");
}

// embed::Modem itself, at another rate: what the C ABI does not show. The engine is told how far its clock
// runs ahead of the air (Engine::set_tx_latency), measured from when the application says a burst was played;
// TXPOWER scales the audio.
void direct() {
    check::current_step = "embed::Modem";
    struct Radio : embed::Sink {
        std::vector<float> queue;
        std::uint64_t played = 0;
        bool keyed = false;
        int hold = 0, unkeys = 0;
        float peak = 0;
        std::vector<std::string> lines;
        void tx_audio(std::span<const float> s) override {
            hold = 25;  // half a second of 20 ms ticks before the first sample is out
            queue.insert(queue.end(), s.begin(), s.end());
            for (const float v : s) peak = std::max(peak, std::abs(v));
        }
        void tx_flush() override { queue.clear(); }
        void ptt(bool on) override {
            unkeys += keyed && !on;
            keyed = on;
        }
        void host_line(const std::string& l) override { lines.push_back(l); }
        void host_data(std::span<const std::uint8_t>) override {}
    } radio;
    embed::Config cfg;
    cfg.sample_rate = 16000;
    cfg.seed = 71;
    cfg.log = false;
    {
        embed::Config bad = cfg;
        bad.sample_rate = 12000;
        bool threw = false;
        try {
            embed::Modem m(bad, radio);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check::is_true(threw, "12000 Hz: not a multiple of the engine's rate");
    }
    embed::Modem m(cfg, radio);
    const std::size_t tick = 320;
    for (const char* line : {"MYCALL W1AW", "DATA2G ON", "TXPOWER -6", "CONNECT W1AW K2XYZ"}) m.command(line);
    check::equal(m.engine().tx_latency(), 0.0, "no latency known before a burst");
    for (int i = 0; i < 1000 && radio.unkeys == 0; ++i) {
        if (radio.keyed && !radio.queue.empty() && radio.hold-- <= 0) {
            const std::size_t n = std::min(tick, radio.queue.size());
            radio.queue.erase(radio.queue.begin(), radio.queue.begin() + static_cast<std::ptrdiff_t>(n));
            radio.played += n;
        }
        m.tx_played(radio.played);
        m.advance(tick);
    }
    check::equal(radio.unkeys, 1, "a burst played and the radio released");
    // the half second, the block the engine keyed it in (0.1 s), and a tick or two of noticing
    check::is_true(m.engine().tx_latency() > 0.55 && m.engine().tx_latency() < 0.75, "the output's latency reaches the engine");
    const float want = static_cast<float>(std::pow(10.0, -6.0 / 20));
    check::is_true(radio.peak <= want * 1.001f && radio.peak > 0.2f * want, "TXPOWER -6 dB on the audio");
    check::is_true(std::find(radio.lines.begin(), radio.lines.end(), "PTT OFF") != radio.lines.end(), "PTT OFF");
    m.command("ABORT");
}

void refused() {
    check::current_step = "refused";
    App a;
    D2gConfig cfg;
    d2g_config_default(&cfg);
    check::is_true(cfg.sample_rate == 48000 && cfg.ptt_on_delay_s > 0 && cfg.tx_gain == 1.0f && cfg.dd_budget_s == 1.0f,
                   "defaults");
    D2gCallbacks c = a.callbacks();
    check::is_true(d2g_new(nullptr, &c, &a) == nullptr && d2g_new(&cfg, nullptr, &a) == nullptr, "null config or callbacks");
    for (const auto& bad : std::vector<std::function<void(D2gConfig&)>>{
             [](D2gConfig& x) { x.sample_rate = 0; },
             [](D2gConfig& x) { x.sample_rate = 44100; },
             [](D2gConfig& x) { x.ptt_on_delay_s = -1; },
             [](D2gConfig& x) { x.ptt_off_delay_s = NAN; },
             [](D2gConfig& x) { x.tx_gain = INFINITY; },
             [](D2gConfig& x) { x.dd_budget_s = -0.5f; },
         }) {
        D2gConfig x = cfg;
        bad(x);
        check::is_true(d2g_new(&x, &c, &a) == nullptr, "a bad config is refused");
    }
    c.host_data = nullptr;
    check::is_true(d2g_new(&cfg, &c, &a) == nullptr, "a required callback missing");
    // null handles and arguments do nothing
    d2g_command(nullptr, "VERSION");
    d2g_send(nullptr, nullptr, 0);
    d2g_rx_audio(nullptr, nullptr, 0);
    d2g_advance(nullptr, 1);
    d2g_tx_played(nullptr, 1);
    d2g_free(nullptr);
    d2g_config_default(nullptr);
    check::is_true(a.events.empty(), "no callback from a refusal");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog dog(600, "test_embed");
    refused();
    session();
    rate_8000();
    weak();
    abort_flushes();
    watchdog();
    free_inside();
    free_keyed_and_fault();
    direct();
    return check::report("test_embed");
}
