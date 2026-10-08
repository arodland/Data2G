// arq::Engine with no Python: two engines through a noisy channel in sync
// mode (a session, KISS, the recorder's files), the worker's header
// latency with a 1 s decode in flight, the worker's bounded queue (a stalled
// decode: blocks dropped, logged, the engine recovers), and two worker-mode
// engines in real time. Parity with data2g/arq/engine.py is tests/test_native_engine.py's
// (a C++ engine against a Python one) and the --native substitution's.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <cstdlib>

#include "arq/engine.hpp"
#include "check.hpp"

using namespace data2g;
using namespace data2g::arq;
using clk = std::chrono::steady_clock;

namespace {

constexpr int BLOCK = config::FS / 10;
// Sanitizers slow everything several-fold: wall-clock latency bounds are off.
#if defined(__SANITIZE_THREAD__) || defined(__SANITIZE_ADDRESS__)
constexpr bool TSAN = true;
#else
constexpr bool TSAN = false;
#endif

// Both engines step, each hearing the other's last block plus noise, until
// until() or `seconds` pass; paced: at real time.
bool link(Engine& a, Engine& b, double snr_db, double seconds, const std::function<bool()>& until, unsigned seed,
          bool paced = false) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, std::sqrt((config::FS / 2.0) / config::SNR_REF_BW_HZ / std::pow(10.0, snr_db / 10)));
    std::vector<double> ao(BLOCK, 0.0), bo(BLOCK, 0.0), x(BLOCK);
    const auto t0 = clk::now();
    for (int i = 0; i < static_cast<int>(seconds * 10); ++i) {
        for (int j = 0; j < BLOCK; ++j) x[j] = 2.2 * bo[j] + noise(rng);
        ao = a.step(x).audio;
        for (int j = 0; j < BLOCK; ++j) x[j] = 2.2 * ao[j] + noise(rng);
        bo = b.step(x).audio;
        if (until()) return true;
        if (paced) std::this_thread::sleep_until(t0 + std::chrono::milliseconds(100 * (i + 1)));
    }
    return false;
}

Bytes bytes(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    Bytes b(n);
    for (auto& v : b) v = static_cast<std::uint8_t>(rng() & 255);
    return b;
}

Bytes ui_frame(const std::string& info) {  // APRS <- W1AW, UI
    Bytes out;
    for (const auto& [c, last] : {std::pair{std::string("APRS"), false}, {std::string("W1AW"), true}}) {
        for (std::size_t i = 0; i < 6; ++i) out.push_back(static_cast<std::uint8_t>((i < c.size() ? c[i] : ' ') << 1));
        out.push_back(static_cast<std::uint8_t>(0x60 | last));
    }
    out.push_back(0x03);
    out.push_back(0xF0);
    out.insert(out.end(), info.begin(), info.end());
    return out;
}

bool both(Engine& a, Engine& b, SessionState s) { return a.session().state == s && b.session().state == s; }

void sync_session(const std::filesystem::path& dir) {
    check::current_step = "sync: session";
    EngineConfig ca;
    ca.seed = 1;
    ca.record_dir = dir.string();
    EngineConfig cb;
    cb.seed = 2;
    Engine a("w1aw", ca), b("K2XYZ", cb);
    check::equal(a.call(), std::string("W1AW"), "call upper-cased");
    b.listen();
    a.connect("K2XYZ", 2);
    check::is_true(link(a, b, 12, 60, [&] { return both(a, b, SessionState::CONNECTED); }, 0), "connected");
    const Bytes up = bytes(1500, 3), down = bytes(400, 4);
    a.session().write(up);
    b.session().write(down);
    Bytes got_a, got_b;
    check::is_true(link(a, b, 12, 240, [&] {
        const auto ra = a.session().read(), rb = b.session().read();
        got_a.insert(got_a.end(), ra.begin(), ra.end());
        got_b.insert(got_b.end(), rb.begin(), rb.end());
        return got_b.size() >= up.size() && got_a.size() >= down.size();
    }, 1), "data both ways");
    check::is_true(got_b == up && got_a == down, "data intact");
    a.session().disconnect();
    check::is_true(link(a, b, 12, 60, [&] { return both(a, b, SessionState::CLOSED); }, 2), "closed");
    const auto ev = a.events();
    check::is_true(!ev.empty() && ev.front() == "CONNECTED K2XYZ" && ev.back().rfind("DISCONNECTED", 0) == 0, "events");

    check::current_step = "sync: recorder";
    std::ifstream log(dir / "events.jsonl");
    std::string line;
    int n_rx = 0, n_tx = 0, n = 0, lines = 0;
    while (std::getline(log, line)) {
        n_rx += line.rfind("{\"kind\": \"rx\"", 0) == 0;
        n_tx += line.rfind("{\"kind\": \"tx\"", 0) == 0;
        n += line.front() == '{' && line.back() == '}';
        ++lines;
    }
    check::is_true(n_rx > 0 && n_tx > 0, "rx and tx events");
    check::equal(n, lines, "one JSON object per line");
    std::ifstream rx(dir / "rx_00000.npz", std::ios::binary);
    char sig[4] = {};
    rx.read(sig, 4);
    check::is_true(std::string(sig, 4) == std::string("PK\x03\x04", 4), "rx_00000.npz is a zip");
    check::equal(static_cast<long long>(std::filesystem::file_size(dir / "audio_in.f16")),
                 static_cast<long long>(2 * a.n()), "audio_in.f16: every sample");
}

// An async recorder writes what a synchronous one does, in the same order, with nothing lost at flush() or
// at destruction.
void recorder_async(const std::filesystem::path& base) {
    check::current_step = "recorder: async";
    auto slurp = [](const std::filesystem::path& p) {
        std::ifstream f(p, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    };
    auto drive = [](Recorder& r) {
        for (int i = 0; i < 300; ++i) {
            r.audio(std::vector<double>(BLOCK, 0.001 * (i % 50)));
            if (i % 6 == 0) r.event("tx", {{"t", json_num(0.1 * i)}, {"n", std::to_string(i)}});
        }
    };
    const auto sync_dir = base / "sync", async_dir = base / "async";
    {
        Recorder r(sync_dir.string(), "W1AW");
        drive(r);
    }
    {
        Recorder r(async_dir.string(), "W1AW", true);
        drive(r);
        r.flush();
        check::equal(static_cast<long long>(std::filesystem::file_size(async_dir / "audio_in.f16")),
                     static_cast<long long>(2 * 300 * BLOCK), "async: flush() leaves every sample on disk");
        r.event("tx", {{"t", "1.0"}});  // after the flush: the destructor must write it
    }
    check::is_true(slurp(sync_dir / "audio_in.f16") == slurp(async_dir / "audio_in.f16"), "async: audio bytes match");
    auto lines = [&](const std::filesystem::path& p) {  // all but the start event, whose wall clock differs
        std::vector<std::string> out;
        std::ifstream f(p);
        for (std::string l; std::getline(f, l);) out.push_back(l);
        out.erase(out.begin());
        return out;
    };
    const auto a = lines(sync_dir / "events.jsonl"), b = lines(async_dir / "events.jsonl");
    check::equal(b.size(), a.size() + 1, "async: events, plus the one after the flush");
    check::is_true(std::equal(a.begin(), a.end(), b.begin()), "async: events in order");
}

void sync_kiss() {
    check::current_step = "sync: KISS";
    kisslink::KissLink ka, kb;
    EngineConfig ca, cb;
    ca.seed = 21;
    ca.kiss = &ka;
    cb.seed = 22;
    cb.kiss = &kb;
    Engine a("W1AW", ca), b("K2XYZ", cb);
    const Bytes ui = ui_frame("!beacon");
    ka.enqueue(ui);
    check::is_true(link(a, b, 12, 30, [&] {
        return std::find(b.kiss_rx().begin(), b.kiss_rx().end(), std::pair<int, Bytes>{0, ui}) != b.kiss_rx().end();
    }, 5), "a UI frame crosses");
}

// EngineConfig::noise_rule reaches each session's gear shifter; 0 turns it off.
void noise_rule_config() {
    check::current_step = "noise rule";
    const auto rule = [](double w) {
        EngineConfig c;
        c.noise_rule = w;
        Engine e("W1AW", c);
        return dynamic_cast<GearPolicy&>(*e.session().policy).shifter.noise_rule;
    };
    check::is_true(EngineConfig{}.noise_rule == NOISE_RULE && rule(NOISE_RULE) == std::optional(NOISE_RULE), "default");
    check::is_true(rule(0.4) == std::optional(0.4), "a weight");
    check::is_true(!rule(0.0), "0: off");
}

void units() {
    check::current_step = "units";
    check::equal(static_cast<int>(to_half(1.0)), 0x3C00, "half 1");
    check::equal(static_cast<int>(to_half(-2.0)), 0xC000, "half -2");
    check::equal(static_cast<int>(to_half(65504.0)), 0x7BFF, "half max");
    check::equal(static_cast<int>(to_half(65520.0)), 0x7C00, "half: rounds to inf");
    check::equal(static_cast<int>(to_half(std::ldexp(1.0, -24))), 1, "half least subnormal");
    check::equal(static_cast<int>(to_half(1.0 + std::ldexp(1.0, -11))), 0x3C00, "half: tie to even");
    check::equal(json_num(60.0), std::string("60.0"), "json 60.0");
    check::equal(json_num(1e-5), std::string("1e-05"), "json 1e-05");
    check::equal(json_str("a\"b"), std::string("\"a\\\"b\""), "json string");
}

// A burst decode that takes 1 s (a DD pass), on the session stage.
class SlowEngine : public Engine {
public:
    using Engine::Engine;
    ~SlowEngine() override { stop(); }

protected:
    void hear_burst(tnc::BurstEvent& ev, double t) override {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        Engine::hear_burst(ev, t);
    }
};

// Two bursts 0.1 s apart in noise.
std::vector<double> two_bursts() {
    auto burst = std::make_shared<TxBurst>();
    burst->submode = "qpsk-r1/5";
    const auto pb = static_cast<std::size_t>(payload_bytes(mode_at(burst->submode)));
    for (int i = 0; i < 2; ++i) burst->slots.push_back({ctl_mask(0, i, 0), 0, bytes(pb, 7 + i)});
    const auto x = tx_audio(*burst);
    std::vector<double> y(2 * config::FS, 0.0);
    y.insert(y.end(), x.begin(), x.end());
    y.insert(y.end(), config::FS / 10, 0.0);
    y.insert(y.end(), x.begin(), x.end());
    y.insert(y.end(), 3 * config::FS, 0.0);
    std::mt19937 rng(9);
    std::normal_distribution<double> noise(0.0, 0.05);
    for (double& v : y) v += noise(rng);
    return y;
}

// The block in which BUSY rises for the second burst, and how late (wall
// seconds past that block's arrival at real time) step() returned it.
template <typename E>
std::pair<int, double> second_header(E& e, const std::vector<double>& y, bool paced, int want = -1) {
    const auto t0 = clk::now();
    int edges = 0;
    bool was = false;
    for (std::size_t i = 0; (i + 1) * BLOCK <= y.size(); ++i) {
        const auto due = t0 + std::chrono::milliseconds(100 * (i + 1));
        if (paced) std::this_thread::sleep_until(due);
        e.step(std::span<const double>(y).subspan(i * BLOCK, BLOCK));
        const bool busy = e.busy();
        if (busy && !was && ++edges == 2) {
            const int at = want >= 0 ? want : static_cast<int>(i);
            const auto arrived = t0 + std::chrono::milliseconds(100 * (at + 1));
            return {static_cast<int>(i), std::chrono::duration<double>(clk::now() - arrived).count()};
        }
        was = busy;
    }
    return {-1, 0.0};
}

void worker_latency() {
    check::current_step = "worker: header latency";
    const auto y = two_bursts();
    Engine ref("W1AW");
    const int j = second_header(ref, y, false).first;
    check::is_true(j > 0, "two headers heard");
    EngineConfig cs, cw;
    cw.worker = true;
    SlowEngine sync_e("W1AW", cs), worker_e("W1AW", cw);
    const auto [js, late_sync] = second_header(sync_e, y, true, j);
    const auto [jw, late_worker] = second_header(worker_e, y, true, j);
    std::printf("second header, 1 s decode in flight: %.3f s late in sync mode, %.3f s with the worker\n", late_sync,
                late_worker);
    check::equal(js, j, "sync: same block");
    check::equal(jw, j, "worker: same block");
    check::is_true(late_worker < late_sync, "the worker hears it sooner");
    // Absolute wall-clock bounds: not on shared CI runners (macOS under
    // Rosetta missed 0.15 s); the relative check above runs everywhere.
    if (!TSAN && !std::getenv("CI")) {
        check::is_true(late_worker < 0.15, "worker: on time");
        check::is_true(late_sync > 0.15, "sync: held by the decode");
    }
}

// A burst decode that hangs until released (a stalled worker).
class StallEngine : public Engine {
public:
    using Engine::Engine;
    ~StallEngine() override {
        open = true;
        stop();
    }
    std::atomic<bool> open{false};

protected:
    void hear_burst(tnc::BurstEvent& ev, double t) override {
        while (!open) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        Engine::hear_burst(ev, t);
    }
};

void worker_backlog() {
    check::current_step = "worker: bounded queue";
    std::mutex log_mu;
    std::vector<std::string> warnings;
    set_log_sink({[](const char*, int level) { return level >= 30; },
                  [&](const char* name, int, const std::string& msg) {
                      std::lock_guard lock(log_mu);
                      if (std::string(name) == "data2g.engine") warnings.push_back(msg);
                  }});
    auto burst = std::make_shared<TxBurst>();
    burst->submode = "qpsk-r1/5";
    const auto pb = static_cast<std::size_t>(payload_bytes(mode_at(burst->submode)));
    for (int i = 0; i < 2; ++i) burst->slots.push_back({ctl_mask(0, i, 0), 0, bytes(pb, 7 + i)});
    const auto x = tx_audio(*burst);
    // burst, 6 s with the decode stalled, release, 1 s, burst, 3 s
    std::vector<double> y(2 * config::FS, 0.0);
    y.insert(y.end(), x.begin(), x.end());
    y.insert(y.end(), 6 * config::FS, 0.0);
    const std::size_t release = y.size() / BLOCK;
    y.insert(y.end(), config::FS, 0.0);
    y.insert(y.end(), x.begin(), x.end());
    y.insert(y.end(), 3 * config::FS, 0.0);
    y.resize(y.size() / BLOCK * BLOCK);
    std::mt19937 rng(10);
    std::normal_distribution<double> noise(0.0, 0.05);
    for (double& v : y) v += noise(rng);

    EngineConfig c;
    c.worker = true;
    c.max_backlog_s = 2.0;
    StallEngine e("W1AW", c);
    std::atomic<int> processed{0}, heard{0}, decoded{0};
    std::atomic<std::int64_t> n_seen{0};
    e.set_after_block([&](bool) {
        n_seen = e.n();
        ++processed;
    });
    e.set_on_burst([&](const BurstHeard& b) {
        ++heard;
        decoded += !b.lost;
    });
    const std::size_t steps = y.size() / BLOCK;
    for (std::size_t i = 0; i < steps; ++i) {
        if (i == release) {
            check::is_true(e.decode_dropped() > 0, "stalled: blocks dropped");
            e.open = true;
            // let the worker drain what it holds before more audio comes
            const auto t0 = clk::now();
            while (processed < static_cast<int>(i - e.decode_dropped() / BLOCK) && clk::now() - t0 < std::chrono::seconds(60))
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        e.step(std::span<const double>(y).subspan(i * BLOCK, BLOCK));
    }
    const double dropped_s = static_cast<double>(e.decode_dropped()) / config::FS;
    std::printf("stalled decode, 2 s bound: %.1f s dropped\n", dropped_s);
    check::equal(e.decode_dropped() % BLOCK, std::uint64_t{0}, "whole blocks dropped");
    check::is_true(dropped_s > 2.0 && dropped_s < 6.0, "dropped what was past the bound");
    {
        std::lock_guard lock(log_mu);
        check::equal(warnings.size(), std::size_t{1}, "one warning per episode");
        const std::string want = format("%.1f s of audio dropped", dropped_s);
        check::is_true(!warnings.empty() && warnings[0].find(want) != std::string::npos, "the warning says how much: " + want);
    }
    check::equal(heard.load(), 2, "both bursts heard");
    check::equal(decoded.load(), 2, "the burst after the hole decoded");
    check::equal(n_seen.load(), static_cast<std::int64_t>(y.size()), "dropped time still passed on the session stage");
    set_log_sink({});
}

void worker_session() {
    check::current_step = "worker: real-time session";
    EngineConfig ca, cb;
    ca.seed = 31;
    ca.worker = cb.worker = true;
    cb.seed = 32;
    Engine a("W1AW", ca), b("K2XYZ", cb);
    // the session is the worker's: watched from after_block, driven by post()
    std::atomic<int> sa{0}, sb{0};
    std::mutex mu;
    Bytes got;
    a.set_after_block([&](bool) { sa = static_cast<int>(a.session().state); });
    b.set_after_block([&](bool) {
        sb = static_cast<int>(b.session().state);
        const auto r = b.session().read();
        std::lock_guard lock(mu);
        got.insert(got.end(), r.begin(), r.end());
    });
    b.post([&] { b.listen(); });
    a.post([&] { a.connect("K2XYZ", 2); });
    const int up = static_cast<int>(SessionState::CONNECTED);
    check::is_true(link(a, b, 15, 60, [&] { return sa == up && sb == up; }, 11, true), "worker: connected");
    const Bytes data = bytes(300, 12);
    a.post([&] { a.session().write(data); });
    check::is_true(link(a, b, 15, 60, [&] {
        std::lock_guard lock(mu);
        return got.size() >= data.size();
    }, 12, true), "worker: data arrives");
    std::lock_guard lock(mu);
    check::is_true(got == data, "worker: data intact");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog dog(TSAN ? 1800 : 600, "test_engine");
    const auto dir = std::filesystem::temp_directory_path() / ("data2g_test_engine_" + std::to_string(clk::now().time_since_epoch().count()));
    units();
    recorder_async(dir / "rec");
    noise_rule_config();
    sync_session(dir);
    sync_kiss();
    worker_latency();
    worker_backlog();
    worker_session();
    std::filesystem::remove_all(dir);
    return check::report("test_engine");
}
