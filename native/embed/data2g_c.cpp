// The C ABI over embed::Modem (include/CData2G/data2g.h, docs/EMBEDDING.md): what an application in another
// language links. Three things are this file's alone:
//
//  - calls from inside a callback: a command or data is queued and runs before the outermost call returns
//    (embed::Modem is not re-entrant); audio and time from inside one are refused;
//  - no C++ exception crosses the boundary: a fault flushes, un-keys, says so in the log, and every later
//    call is ignored;
//  - d2g_free from inside a callback: no callback after it but the last two, and the free itself waits for
//    the outermost call to return.
#include "CData2G/data2g.h"

#include <cmath>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "embed/modem.hpp"
#include "host/host.hpp"

namespace embed = data2g::embed;

struct D2gModem final : embed::Sink {
    struct Pending {
        bool command;  // false: data
        std::string line;
        std::vector<std::uint8_t> data;
    };

    D2gCallbacks cbs{};
    void* ctx = nullptr;
    std::unique_ptr<embed::Modem> modem;
    std::deque<Pending> pending;  // asked for from inside a callback
    int depth = 0;                // calls under way: over 0 inside a callback
    bool dead = false;            // failed or being freed: no callback but release()'s
    bool released = false;        // the last tx_flush and ptt(false) are done
    bool free_requested = false;
    std::string fault;

    // embed::Sink: the application's callbacks, while the modem lives
    void tx_audio(std::span<const float> s) override {
        if (!dead) cbs.tx_audio(ctx, s.data(), s.size());
    }
    void tx_flush() override {
        if (!dead) cbs.tx_flush(ctx);
    }
    void ptt(bool on) override {
        if (!dead) cbs.ptt(ctx, on);
    }
    void host_line(const std::string& line) override {
        if (!dead) cbs.host_line(ctx, line.c_str());
    }
    void host_data(std::span<const std::uint8_t> d) override {
        if (!dead) cbs.host_data(ctx, d.data(), d.size());
    }
    void log(const std::string& line) override {
        if (!dead && cbs.log) cbs.log(ctx, line.c_str());
    }
    void tx_report(const embed::TxReport& r) override {
        if (dead || !cbs.tx_report) return;
        const D2gTxReport c{r.time_s,
                            r.mode.c_str(),
                            static_cast<std::uint32_t>(r.width_hz),
                            static_cast<std::uint16_t>(r.codewords),
                            static_cast<std::uint16_t>(r.data_codewords),
                            static_cast<std::uint32_t>(r.bytes),
                            static_cast<float>(r.air_s),
                            static_cast<float>(r.net_bps)};
        cbs.tx_report(ctx, &c);
    }
    void rx_report(const embed::RxReport& r) override {
        if (dead || !cbs.rx_report) return;
        const D2gRxReport c{r.time_s,
                            r.mode.c_str(),
                            static_cast<std::uint32_t>(r.width_hz),
                            static_cast<std::uint16_t>(r.codewords),
                            static_cast<std::uint16_t>(r.codewords_ok),
                            static_cast<std::uint16_t>(r.codewords_failed),
                            static_cast<std::uint32_t>(r.bytes),
                            r.lost,
                            r.snr_db ? static_cast<float>(*r.snr_db) : std::numeric_limits<float>::quiet_NaN(),
                            static_cast<float>(r.air_s),
                            static_cast<float>(r.net_bps)};
        cbs.rx_report(ctx, &c);
    }
};

namespace {

void kill(D2gModem* m, std::string why) {
    if (m->dead) return;
    m->dead = true;
    m->fault = std::move(why);
    if (m->modem) m->modem->halt();
}

// The outermost call is returning. A modem that failed or is being freed leaves the radio as the contract
// says, once: nothing queued, PTT off. Then the free, if one was asked for.
void leave(D2gModem* m) {
    if (--m->depth > 0) return;
    if (m->dead && !m->released) {
        m->released = true;
        ++m->depth;  // a d2g_free from inside these two waits for them
        m->cbs.tx_flush(m->ctx);
        m->cbs.ptt(m->ctx, false);
        if (!m->free_requested && m->cbs.log)
            m->cbs.log(m->ctx, ("error: modem failed (" + m->fault + "): further calls are ignored").c_str());
        --m->depth;
    }
    if (m->free_requested) delete m;
}

// One outermost call: f, then whatever its callbacks asked for, in order.
template <class F>
void run(D2gModem* m, F&& f) {
    ++m->depth;
    try {
        f();
        while (!m->dead && !m->pending.empty()) {
            const D2gModem::Pending p = std::move(m->pending.front());
            m->pending.pop_front();
            if (p.command)
                m->modem->command(p.line);
            else
                m->modem->send(p.data);
        }
    } catch (const std::exception& e) {
        kill(m, e.what());
    } catch (...) {
        kill(m, "unknown error");
    }
    leave(m);
}

void refused(D2gModem* m, const char* what) {
    if (!m->dead && m->cbs.log) m->cbs.log(m->ctx, (std::string("warning: ") + what + " from inside a callback: ignored").c_str());
}

}  // namespace

extern "C" {

void d2g_config_default(D2gConfig* cfg) {
    if (!cfg) return;
    const embed::Config d;
    *cfg = D2gConfig{};
    cfg->sample_rate = D2G_RATE_DEFAULT;
    cfg->ptt_on_delay_s = static_cast<float>(d.ptt_on_delay_s);
    cfg->ptt_off_delay_s = static_cast<float>(d.ptt_off_delay_s);
    cfg->tx_gain = static_cast<float>(d.tx_gain);
    cfg->decode_threads = 0;
    cfg->decoder_effort = 25;  // codes::DEFAULT_ITERS
    cfg->dd_budget_s = static_cast<float>(*d.dd_budget_s);
    cfg->seed = 0;
}

D2gModem* d2g_new(const D2gConfig* cfg, const D2gCallbacks* cbs, void* ctx) {
    if (!cfg || !cbs) return nullptr;
    if (!cbs->tx_audio || !cbs->tx_flush || !cbs->ptt || !cbs->host_line || !cbs->host_data) return nullptr;
    if (cfg->sample_rate > 384000) return nullptr;  // embed::Modem refuses the rest
    D2gModem* m = nullptr;
    try {
        embed::Config c;
        c.sample_rate = static_cast<int>(cfg->sample_rate);
        c.ptt_on_delay_s = cfg->ptt_on_delay_s;
        c.ptt_off_delay_s = cfg->ptt_off_delay_s;
        c.tx_gain = cfg->tx_gain;
        c.decode_threads = cfg->decode_threads;
        c.decoder_effort = cfg->decoder_effort;
        c.dd_budget_s = static_cast<double>(cfg->dd_budget_s);
        if (cfg->seed) c.seed = cfg->seed;
        c.log = cbs->log != nullptr;
        m = new D2gModem;
        m->cbs = *cbs;
        m->ctx = ctx;
        m->modem = std::make_unique<embed::Modem>(c, *m);
        return m;
    } catch (...) {
        delete m;
        return nullptr;
    }
}

void d2g_free(D2gModem* m) {
    if (!m) return;
    m->free_requested = true;
    kill(m, "freed");
    if (m->depth > 0) return;  // inside a callback: the call that fired it finishes the job (leave)
    ++m->depth;
    leave(m);
}

void d2g_command(D2gModem* m, const char* line) {
    if (!m || !line || m->dead) return;
    if (m->depth > 0) {
        m->pending.push_back({true, line, {}});
        return;
    }
    run(m, [&] { m->modem->command(line); });
}

void d2g_send(D2gModem* m, const uint8_t* data, size_t n) {
    if (!m || !data || !n || m->dead) return;
    if (m->depth > 0) {
        m->pending.push_back({false, {}, std::vector<std::uint8_t>(data, data + n)});
        return;
    }
    run(m, [&] { m->modem->send(std::span<const std::uint8_t>(data, n)); });
}

void d2g_rx_audio(D2gModem* m, const float* samples, size_t n) {
    if (!m || !samples || m->dead) return;
    if (m->depth > 0) return refused(m, "d2g_rx_audio");
    run(m, [&] { m->modem->rx_audio(std::span<const float>(samples, n)); });
}

void d2g_advance(D2gModem* m, uint64_t samples) {
    if (!m || m->dead) return;
    if (m->depth > 0) return refused(m, "d2g_advance");
    run(m, [&] { m->modem->advance(samples); });
}

void d2g_tx_played(D2gModem* m, uint64_t total_samples) {
    if (!m || m->dead) return;
    m->modem->tx_played(total_samples);  // a count: no callback comes of it, so it is safe wherever it is called
}

float d2g_max_net_bps(uint8_t bandwidth) {
    try {
        return static_cast<float>(embed::max_net_bps(bandwidth));
    } catch (...) {
        return 0.0f;
    }
}

const char* d2g_version(void) {
    static const std::string v(data2g::host::VERSION);
    return v.c_str();
}

}  // extern "C"
