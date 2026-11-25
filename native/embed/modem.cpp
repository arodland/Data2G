#include "embed/modem.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "arq/burst.hpp"
#include "arq/modes.hpp"
#include "arq/nod.hpp"
#include "arq/policy.hpp"
#include "codes/codes.hpp"
#include "generated/config.hpp"
#include "util/pool.hpp"

namespace data2g::embed {

namespace {

constexpr const char* LOG = "data2g.embed";
constexpr int INFO = 20, WARNING = 30;  // Python's logging levels (arq/link.hpp); 40 is an error
constexpr std::size_t BLOCK = config::FS / 10;       // the engine's step, as the station's
constexpr std::size_t CHUNK = config::FS / 100;      // what is looked at between two looks at the transmitter
// audio::Interpolator's delay, in its input's samples (a 32 u + 1 tap filter at u times the rate): that much
// silence after a burst brings its last samples out
constexpr std::size_t INTERP_DELAY = 16;
constexpr std::size_t LOG_KEPT = 2000;  // lines held for a caller that isn't calling

// The core's log (arq::set_log_sink, process-wide), from whichever thread writes it (the pool's decode
// threads too), kept for the modem's own thread to hand to the application.
struct LogQueue {
    std::mutex mu;
    std::vector<std::string> lines;
    std::atomic<bool> owned{false};
};

LogQueue& log_queue() {
    static LogQueue q;
    return q;
}

void log_put(const char*, int level, const std::string& msg) {
    auto& q = log_queue();
    std::lock_guard lock(q.mu);
    if (q.lines.size() >= LOG_KEPT) return;
    q.lines.push_back(level >= 40 ? "error: " + msg : level >= WARNING ? "warning: " + msg : msg);
}

const Config& checked(const Config& c) {
    if (c.sample_rate < config::FS || c.sample_rate % config::FS)
        throw std::invalid_argument("sample rate must be a multiple of " + std::to_string(config::FS));
    for (const double v : {c.ptt_on_delay_s, c.ptt_off_delay_s, c.tx_gain})
        if (!std::isfinite(v) || v < 0) throw std::invalid_argument("delays and gain must be finite and not negative");
    if (c.dd_budget_s && (std::isnan(*c.dd_budget_s) || *c.dd_budget_s < 0))
        throw std::invalid_argument("dd_budget_s must not be negative");
    return c;
}

arq::EngineConfig engine_config(const Config& c) {
    arq::EngineConfig e;
    e.ptt_delay_s = c.ptt_on_delay_s;
    e.dd_budget_s = c.dd_budget_s;
    e.seed = c.seed;
    e.worker = false;  // everything on the caller's thread
    return e;
}

double seconds(const arq::Mode& m, int n_cw, bool dup) {
    try {
        return arq::burst_seconds(m, n_cw, dup);
    } catch (const std::exception&) {  // a count the mode has no burst for (a lost burst's header)
        return 0.0;
    }
}

double nod_seconds() { return static_cast<double>(arq::nod::span()) / config::FS; }

std::string first_word(std::string_view line) {
    std::string w;
    for (const char c : line) {
        if (c == ' ' || c == '\t') {
            if (w.empty()) continue;
            break;
        }
        w += static_cast<char>(c >= 'a' && c <= 'z' ? c - 'a' + 'A' : c);
    }
    return w;
}

}  // namespace

double mode_net_bps(const arq::Mode& m) {
    // the shifter's largest burst (policy.hpp SIZE_S): one control codeword, taken as full of inline data
    // (frames T_INLINE; CPM's is the grid's short one), and the rest data
    const int n = arq::slots_for(m, arq::SIZE_S.back());
    const double s = seconds(m, n, false);
    return s > 0 ? 8.0 * ((n - 1) * arq::payload_bytes(m) + arq::ctl_payload_bytes(m)) / s : 0.0;
}

double max_net_bps(int cap) {
    if (cap < 0 || cap > 2) return 0.0;
    double best = 0.0;
    for (const arq::Mode* m : arq::allowed(cap)) best = std::max(best, mode_net_bps(*m));
    return best;
}

Modem::Modem(const Config& cfg, Sink& sink)
    : cfg_(checked(cfg)), sink_(sink), engine_("NOCALL", engine_config(cfg_)), host_(engine_),
      ratio_(static_cast<std::size_t>(cfg_.sample_rate / config::FS)) {
    if (ratio_ > 1) down_.emplace(cfg_.sample_rate);
    if (cfg_.decode_threads > 0) pool::set_threads(cfg_.decode_threads);
    if (cfg_.decoder_effort > 0) codes::set_ldpc_iters(cfg_.decoder_effort);
    if (cfg_.log && !log_queue().owned.exchange(true)) {
        log_ = true;
        arq::set_log_sink({[](const char*, int level) { return level >= INFO; }, log_put});
    }
    host_.announce_ptt = false;  // PTT ON / OFF go out as the radio is keyed and released (unkey, start_tx)
    // the hooks run inside Engine::step, on its session stage: they only take notes, and never throw
    engine_.set_on_burst([this](const arq::BurstHeard& h) {
        try {
            on_burst(h);
        } catch (...) {
        }
    });
    engine_.set_on_decoded([this](int decoded, int failed) {
        try {
            on_decoded(decoded, failed);
        } catch (...) {
        }
    });
    engine_.set_on_send([this](const arq::TxBurstPtr& b) {
        try {
            on_send(b);
        } catch (...) {
        }
    });
    engine_.set_on_nod([this](const arq::TxBurstPtr& b) {
        try {
            on_nod(b);
        } catch (...) {
        }
    });
}

Modem::~Modem() {
    engine_.stop();
    if (!log_) return;
    arq::set_log_sink({});
    auto& q = log_queue();
    {
        std::lock_guard lock(q.mu);
        q.lines.clear();
    }
    q.owned = false;
}

// --- the application's calls -----------------------------------------------------------

void Modem::command(std::string_view line) {
    if (halted_) return;
    std::string text;  // ascii, as host.py decodes it (others replaced); one line
    for (const unsigned char c : line) {
        if (c == '\r' || c == '\n') break;
        text += c < 128 ? static_cast<char>(c) : '?';
    }
    const bool on_air = engine_.tx().has_value();
    if (first_word(text) == "ACCEPT")  // a data connection of its own per session: there is no port here to dial
        host_.out_cmd.emplace_back("WRONG");
    else
        host_.command(text);
    // ABORT while the engine was sending: stopped now, not at the next block (whose tx_abort finds nothing left)
    if (on_air && !engine_.tx() && (keyed_ || consumed() < queued_)) {
        flush();
        unkey();
    }
    deliver();
}

void Modem::send(std::span<const std::uint8_t> data) {
    if (halted_) return;
    host_.data_in(data);
    deliver();
}

void Modem::rx_audio(std::span<const float> samples) { feed(samples.data(), samples.size()); }

void Modem::advance(std::uint64_t samples) {
    // a step at a time, so a count from a confused clock cannot ask for more than one allocation's worth
    const std::uint64_t step = static_cast<std::uint64_t>(cfg_.sample_rate);
    do {
        const auto n = static_cast<std::size_t>(std::min(samples, step));
        feed(nullptr, n);
        samples -= n;
    } while (samples && !halted_);
}

void Modem::tx_played(std::uint64_t total_samples) {
    // what was flushed was not played: the total can never pass what is left of what was handed over
    played_ = std::max(played_, std::min(total_samples, queued_ - flushed_));
}

// --- receive ---------------------------------------------------------------------------

void Modem::feed(const float* x, std::size_t n) {
    const std::size_t chunk = CHUNK * ratio_;
    while (n && !halted_) {
        const std::size_t m = std::min(n, chunk);
        clock_ += m;
        // the radio's receive audio is muted while it transmits (station.cpp's engine loop): from the key to
        // the release, whatever the application captured
        in_.assign(m, 0.0);
        if (x && !keyed_) std::copy_n(x, m, in_.begin());
        if (down_) {
            const auto y = (*down_)(in_);
            pending_.insert(pending_.end(), y.begin(), y.end());
        } else {
            pending_.insert(pending_.end(), in_.begin(), in_.end());
        }
        while (pending_.size() >= BLOCK && !halted_) block();
        service_tx();
        deliver();
        if (x) x += m;
        n -= m;
    }
    if (halted_) return;
    service_tx();  // advance(0), after tx_played
    deliver();
}

void Modem::block() {
    const std::int64_t n0 = engine_.n();
    auto out = engine_.step(std::span<const double>(pending_.data(), BLOCK));
    pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(BLOCK));
    host_.after_step(out.ptt);
    if (out.tx_abort && (keyed_ || consumed() < queued_)) {  // whatever is playing stops
        flush();
        unkey();
    }
    if (out.tx_burst) start_tx(*out.tx_burst, n0);
    deliver();
}

// --- transmit --------------------------------------------------------------------------

std::uint64_t Modem::consumed() const { return std::min(played_ + flushed_, queued_); }

// The engine keyed a transmission in the block that began at its sample n0: the whole of it to the
// application, the radio keyed first. One that comes while the last is still out follows it on the same key.
void Modem::start_tx(const std::vector<double>& burst, std::int64_t n0) {
    if (!keyed_) {
        keyed_ = true;
        sink_.ptt(true);
        host_.out_cmd.emplace_back("PTT ON");
    }
    unkey_at_.reset();
    const double gain = cfg_.tx_gain * std::pow(10.0, host_.output_volume / 20);  // TXPOWER, dB
    const auto put = [&](std::span<const double> y) {
        for (const double v : y) out_.push_back(static_cast<float>(std::clamp(v * gain, -1.0, 1.0)));
    };
    out_.clear();
    out_.reserve((burst.size() + INTERP_DELAY) * ratio_);
    if (ratio_ > 1) {
        audio::Interpolator up(cfg_.sample_rate);
        put(up(burst));
        put(up(std::vector<double>(INTERP_DELAY, 0.0)));
    } else {
        put(burst);
    }
    queued_ += out_.size();
    watchdog_ = clock_ + (queued_ - consumed()) + static_cast<std::uint64_t>(WATCHDOG_S * cfg_.sample_rate);
    engine_end_ = (static_cast<std::uint64_t>(n0) + burst.size()) * ratio_;
    sink_.tx_audio(out_);
}

// The radio's release: once everything handed over is out (played, or flushed) and the off delay has gone by
// on the sample clock. A transmission the application never finishes playing is cut by the watchdog.
void Modem::service_tx() {
    if (!keyed_) return;
    if (consumed() >= queued_) {
        if (!unkey_at_) {
            unkey_at_ = clock_ + static_cast<std::uint64_t>(std::llround(cfg_.ptt_off_delay_s * cfg_.sample_rate));
            // the engine's clock had the burst end at engine_end_; it ended on air about now. Hand timing and
            // the decoder's budget for a reply allow for the difference (station.cpp's on_latency average)
            const double s = std::clamp((static_cast<double>(clock_) - static_cast<double>(engine_end_)) / cfg_.sample_rate,
                                        0.0, WATCHDOG_S);
            latency_ = latency_ < 0 ? s : 0.7 * latency_ + 0.3 * s;
            engine_.set_tx_latency(latency_);
        }
        if (clock_ >= *unkey_at_) unkey();
    } else if (clock_ >= watchdog_) {
        arq::log_write(LOG, WARNING,
                       arq::format("transmit audio not played %.0f s past its length: flushed, PTT off", WATCHDOG_S));
        flush();
        unkey();
    }
}

void Modem::flush() {
    sink_.tx_flush();
    flushed_ = queued_ - played_;  // after the call: a tx_played from inside it is the exact count
}

void Modem::unkey() {
    unkey_at_.reset();
    if (!keyed_) return;
    keyed_ = false;
    sink_.ptt(false);
    host_.out_cmd.emplace_back("PTT OFF");
}

// --- to the application ----------------------------------------------------------------

// What the engine and the host produced, once they have returned: the log, the bursts' reports, the command
// port's lines, then the data port's bytes.
void Modem::deliver() {
    drain_log();
    if (rx_open_) {  // heard, and its read never came (it threw): as heard
        rx_reports_.push_back(std::move(*rx_open_));
        rx_open_.reset();
    }
    for (const auto& r : std::exchange(tx_reports_, {})) {
        if (halted_) return;
        sink_.tx_report(r);
    }
    for (const auto& r : std::exchange(rx_reports_, {})) {
        if (halted_) return;
        sink_.rx_report(r);
    }
    const auto lines = std::exchange(host_.out_cmd, {});
    const auto data = std::exchange(host_.out_data, {});
    host_.out_kiss_refs.clear();
    for (const auto& l : lines) {
        if (halted_) return;
        sink_.host_line(l);
    }
    if (!data.empty() && !halted_) sink_.host_data(data);
}

void Modem::drain_log() {
    if (!log_) return;
    std::vector<std::string> lines;
    {
        auto& q = log_queue();
        std::lock_guard lock(q.mu);
        lines.swap(q.lines);
    }
    for (const auto& l : lines) {
        if (halted_) return;
        sink_.log(l);
    }
}

// --- reports ---------------------------------------------------------------------------

void Modem::on_send(const arq::TxBurstPtr& b) {
    TxReport r;
    r.time_s = engine_.now();
    const arq::Mode* m = arq::mode(b->submode);
    if (m) r.width_hz = static_cast<int>(arq::width_hz(*m));
    if (b->nod) {
        r.mode = "nod";
        r.codewords = 1;
        r.air_s = nod_seconds();
    } else {
        r.mode = b->submode;
        r.codewords = static_cast<int>(b->slots.size());
        bool session = false;  // an ID or CQ frame's key is 0 (burst.hpp): not the session's
        for (const auto& s : b->slots) {
            r.data_codewords += s.mask_id.seq < arq::SEQ_MOD;  // a control codeword's is SEQ_MOD + its index
            session = session || s.mask_id.key != 0;
        }
        if (session) r.bytes = static_cast<int>(tx_new());
        if (m) r.air_s = seconds(*m, r.codewords, arq::dup_ctl(*b));
    }
    if (r.air_s > 0) r.net_bps = 8.0 * r.bytes / r.air_s;
    tx_reports_.push_back(std::move(r));
}

void Modem::on_burst(const arq::BurstHeard& h) {
    if (rx_open_) rx_reports_.push_back(std::move(*rx_open_));
    rx_open_.reset();
    RxReport r;
    r.time_s = h.t;
    r.mode = h.submode;
    r.codewords = h.n_cw;
    r.lost = h.lost;
    r.snr_db = h.snr_db;
    if (const arq::Mode* m = arq::mode(h.submode)) {
        r.width_hz = static_cast<int>(arq::width_hz(*m));
        r.air_s = seconds(*m, h.n_cw, !h.lost && h.heard.cpm && h.heard.cpm->dup);
    }
    if (h.lost)
        rx_reports_.push_back(std::move(r));
    else
        rx_open_ = std::move(r);  // on_decoded's, after the session has read it
}

void Modem::on_decoded(int decoded, int failed) {
    if (!rx_open_) return;
    RxReport r = std::move(*rx_open_);
    rx_open_.reset();
    r.codewords_ok = decoded;
    r.codewords_failed = failed;
    r.bytes = static_cast<int>(rx_new());
    if (r.air_s > 0) r.net_bps = 8.0 * r.bytes / r.air_s;
    rx_reports_.push_back(std::move(r));
}

void Modem::on_nod(const arq::TxBurstPtr& b) {
    RxReport r;
    r.time_s = engine_.now();
    r.mode = "nod";
    if (const arq::Mode* m = arq::mode(b->submode)) r.width_hz = static_cast<int>(arq::width_hz(*m));
    r.codewords = r.codewords_ok = 1;
    r.air_s = nod_seconds();
    rx_reports_.push_back(std::move(r));
}

// The session's stream, counted where it meets the air (a session's Station; a new session's counts start
// over). Sent: the bytes put into codewords so far (TxSide::stream_end), inline ones too; a resend adds none.
std::int64_t Modem::tx_new() {
    const auto& st = engine_.session().station;
    if (st != station_.lock()) {
        station_ = st;
        tx_stream_ = rx_stream_ = 0;
    }
    if (!st) return 0;
    const std::int64_t n = st->tx.stream_end, d = std::max<std::int64_t>(0, n - tx_stream_);
    tx_stream_ = n;
    return d;
}

// Received: the payloads of the codewords taken, delivered in order (RxSide::wire) or held above a hole (its
// buf). A payload is the mode's size whatever it carries, so a message's last codeword counts whole.
std::int64_t Modem::rx_new() {
    const auto& st = engine_.session().station;
    if (st != station_.lock()) {
        station_ = st;
        tx_stream_ = rx_stream_ = 0;
    }
    if (!st) return 0;
    std::int64_t n = st->rx.wire;
    for (const auto& [seq, held] : st->rx.buf) n += static_cast<std::int64_t>(held.first.size());
    const std::int64_t d = std::max<std::int64_t>(0, n - rx_stream_);
    rx_stream_ = n;
    return d;
}

}  // namespace data2g::embed
