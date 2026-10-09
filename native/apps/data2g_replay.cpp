// data2g-replay: run a recording's receive audio through the engine, offline and as fast as it will go. For
// timing the receive path on a slow machine with no sound card, rig or client: the log (the engine's DEBUG
// "RX decode ..." lines carry each burst's decode time, stage by stage) is the same as a live run's.
//
//   data2g-replay recordings/20261008-200708 --call KC2G --log-level DEBUG
//
// The recording is a data2g-host --record-dir (audio_in.f16: float16 mono at 8 kHz). The engine listens as
// --call, so a session in the recording is answered and its bursts go through the session as they did live.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

#include "arq/engine.hpp"
#include "host/host.hpp"
#include "util/pool.hpp"

using namespace data2g;

namespace {

constexpr std::size_t BLOCK = config::FS / 10;

double half_to_double(std::uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    double v;
    if (e == 0) v = std::ldexp(m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(1024 + m, e - 25);
    return (h >> 15) ? -v : v;
}

[[noreturn]] void usage(int code) {
    std::fprintf(code ? stderr : stdout,
                 "usage: data2g-replay DIR [--call CALL] [--seconds S] [--worker] [--threads N] [--dd-budget S]\n"
                 "                     [--log-level DEBUG|INFO|WARNING] [--repeat N]\n"
                 "\n"
                 "Runs DIR/audio_in.f16 through the engine listening as CALL (default NOCALL).\n"
                 "  --seconds     only the first S seconds of audio\n"
                 "  --worker      the session stage on its own thread, as the host runs it (default: inline)\n"
                 "  --threads     the decode pool's size (default: the host's)\n"
                 "  --dd-budget   seconds of DD per burst (default 1; 0: none), as data2g-host's\n"
                 "  --slow-ms     warn about an engine step slower than this (default 200)\n"
                 "  --host        run data2g-host's per-step work too (the VARA host layer: BUFFER, events)\n"
                 "  --log-level   default INFO\n"
                 "  --repeat      run N times; the log is of the first run, the others report their time only\n");
    std::exit(code);
}

int g_level = 20;

const char* level_name(int level) {
    return level >= 40 ? "ERROR" : level >= 30 ? "WARNING" : level >= 20 ? "INFO" : "DEBUG";
}

void log_line(int level, const std::string& msg) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    char ts[24];
    std::strftime(ts, sizeof ts, "%H:%M:%S", std::localtime(&t));
    std::fprintf(stderr, "%s.%03d %s %s\n", ts, static_cast<int>(ms), level_name(level), msg.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir, call = "NOCALL";
    double seconds = 1e18;
    bool worker = false, quiet = false;
    int threads = 0, repeat = 1;
    double dd_budget = arq::DD_BUDGET_S, slow_ms = 200;
    bool with_host = false;
    bool mono = std::getenv("DATA2G_REPLAY_MONO") != nullptr;  // slow steps' CLOCK_MONOTONIC span, to match perf samples
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc) usage(2);
            return argv[++i];
        };
        if (a == "-h" || a == "--help") usage(0);
        else if (a == "--call") call = value();
        else if (a == "--seconds") seconds = std::atof(value());
        else if (a == "--worker") worker = true;
        else if (a == "--threads") threads = std::atoi(value());
        else if (a == "--dd-budget") dd_budget = std::atof(value());
        else if (a == "--slow-ms") slow_ms = std::atof(value());
        else if (a == "--host") with_host = true;
        else if (a == "--repeat") repeat = std::max(1, std::atoi(value()));
        else if (a == "--log-level") {
            const std::string l = value();
            g_level = l == "DEBUG" ? 10 : l == "INFO" ? 20 : l == "WARNING" ? 30 : 20;
        } else if (!a.empty() && a[0] != '-' && dir.empty()) dir = a;
        else usage(2);
    }
    if (dir.empty()) usage(2);

    std::ifstream in(dir + "/audio_in.f16", std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "data2g-replay: cannot read %s/audio_in.f16\n", dir.c_str());
        return 1;
    }
    std::vector<std::uint16_t> raw(static_cast<std::size_t>(in.tellg()) / 2);  // little-endian halves, as numpy writes them
    in.seekg(0);
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * 2));
    std::vector<double> audio(static_cast<std::size_t>(std::min(static_cast<double>(raw.size()), seconds * config::FS)));
    for (std::size_t i = 0; i < audio.size(); ++i) audio[i] = half_to_double(raw[i]);

    if (threads > 0) pool::set_threads(threads);
    arq::set_log_sink({[&quiet](const char*, int level) { return !quiet && level >= g_level; },
                       [&quiet](const char*, int level, const std::string& msg) {
                           if (!quiet) log_line(level, msg);
                       }});

    for (int run = 0; run < repeat; ++run) {
        quiet = run > 0;
        arq::EngineConfig cfg;
        cfg.worker = worker;
        cfg.dd_budget_s = dd_budget;
        cfg.seed = 1;
        arq::Engine eng(call, cfg, {});
        eng.listen(true);
        // the host's per-step work, as the live Station runs it inside every engine step (BUFFER report etc.)
        std::unique_ptr<host::Host> host;
        if (with_host) {
            host = std::make_unique<host::Host>(eng);
            host->listening = true;
            eng.set_after_block([&host](bool ptt) {
                host->after_step(ptt);
                host->out_cmd.clear();
                host->out_data.clear();
            });
        }
        std::vector<double> step_ms;
        std::vector<bool> ptt_trace;
        const auto t_all = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i + BLOCK <= audio.size(); i += BLOCK) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto o = eng.step(std::span<const double>(audio).subspan(i, BLOCK));
            ptt_trace.push_back(o.ptt);
            if (mono && std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() >= slow_ms)
                std::fprintf(stderr, "MONO %.6f %.6f\n", std::chrono::duration<double>(t0.time_since_epoch()).count(),
                             std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
            step_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            if (step_ms.back() >= slow_ms && !quiet)
                log_line(30, "slow step at audio t=" + std::to_string(static_cast<double>(i) / config::FS).substr(0, 6) + " s: " +
                                 std::to_string(static_cast<int>(step_ms.back())) + " ms");
        }
        const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count();
        if (std::getenv("DATA2G_REPLAY_TX")) {  // the steps around each end of our own transmissions
            for (std::size_t k = 1; k < ptt_trace.size(); ++k)
                if (ptt_trace[k - 1] && !ptt_trace[k]) {
                    std::fprintf(stderr, "TX ends at audio t=%.1f s; steps (ms) from 2 before:", static_cast<double>(k * BLOCK) / config::FS);
                    for (std::size_t j = k >= 2 ? k - 2 : 0; j < std::min(k + 6, step_ms.size()); ++j) std::fprintf(stderr, " %.0f", step_ms[j]);
                    std::fprintf(stderr, "\n");
                }
        }
        double sum = 0, worst = 0;
        for (const double m : step_ms) {
            sum += m;
            worst = std::max(worst, m);
        }
        std::fprintf(stderr, "run %d: %.1f s of audio in %.2f s (%.1fx real time), engine steps: mean %.2f ms, worst %.0f ms\n",
                     run + 1, static_cast<double>(audio.size()) / config::FS, total,
                     static_cast<double>(audio.size()) / config::FS / total, sum / static_cast<double>(step_ms.size()), worst);
    }
    return 0;
}
