// The modem for an application that owns the audio and the PTT (docs/EMBEDDING.md): an arq::Engine in sync
// mode and a host::Host, run as app/station.cpp's engine loop and app/transmitter.cpp run them, but by the
// caller instead of by threads. No Qt, no thread of its own, no socket, no file.
//
// The caller is the clock: rx_audio() and advance() count samples at the application's rate, and everything
// the modem does (decoding, the session's timers, keying and releasing the radio) happens inside those two
// calls, with the Sink called back before they return. A burst to send is handed over whole (Sink::tx_audio);
// the application plays it and says how far it has got (tx_played), and the radio is released once all of it
// went out and the off delay has passed on the sample clock.
//
// One call at a time, all from one thread or serial queue. A Sink method may call tx_played(); nothing else
// (embed/data2g_c.cpp queues what an application asks for from inside a callback). Process-wide settings
// (the log sink, the pool's threads, the LDPC iteration cap) make it one Modem per process.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "arq/engine.hpp"
#include "arq/modes.hpp"
#include "audio/filters.hpp"
#include "host/host.hpp"

namespace data2g::embed {

inline constexpr double WATCHDOG_S = 3.0;  // a burst not played this long past its length: flushed, the radio released

struct Config {
    int sample_rate = 48000;        // the application's, both ways: a multiple of config::FS
    double ptt_on_delay_s = 0.1;    // silence at the head of each transmission (EngineConfig::ptt_delay_s)
    double ptt_off_delay_s = 0.05;  // from the last sample played to the radio released
    double tx_gain = 1.0;           // on the transmit audio (a burst's peak is 1.0), with the host's TXPOWER
    int decode_threads = 0;         // pool::set_threads; 0: as it is (pool::default_threads)
    int decoder_effort = 0;         // codes::set_ldpc_iters; 0: as it is (codes::DEFAULT_ITERS)
    std::optional<double> dd_budget_s = arq::DD_BUDGET_S;  // EngineConfig's
    std::optional<std::uint64_t> seed;                     // EngineConfig's (nullopt: random)
    bool log = true;                // the core's log (INFO and up) to Sink::log
};

// A burst sent, as it is keyed.
struct TxReport {
    double time_s = 0.0;     // engine time
    std::string mode;        // the submode ("nod": a nod, docs/arq.md §3b)
    int width_hz = 0;
    int codewords = 0;       // in the burst
    int data_codewords = 0;  // ... of them data, new or sent again
    int bytes = 0;           // of the session's stream, first sent in this burst (none of a resend's)
    double air_s = 0.0;
    double net_bps = 0.0;    // bytes over air_s
};

// A burst heard, once the session has read it.
struct RxReport {
    double time_s = 0.0;
    std::string mode;
    int width_hz = 0;
    int codewords = 0;         // the header's count
    int codewords_ok = 0;      // decoded
    int codewords_failed = 0;  // tried, not decoded
    int bytes = 0;             // the data codewords new to the session's stream: their payload, padding too
    bool lost = false;         // the header heard, the burst not
    std::optional<double> snr_db;
    double air_s = 0.0;
    double net_bps = 0.0;
};

// The most a burst in this mode carries over its time on air: its longest burst (the shifter's largest size),
// every codeword full. No TxReport's net_bps passes it.
double mode_net_bps(const arq::Mode& m);
// The fastest mode's within a bandwidth cap (0: 500 Hz, 1: 1200, 2: 2300); 0 for an unknown cap. A narrower
// cap's modes are all within a wider one's, so it never falls as the cap widens.
double max_net_bps(int cap);

// The application's side. Called on the caller's thread, from inside the Modem call that caused it.
class Sink {
public:
    virtual ~Sink() = default;
    virtual void tx_audio(std::span<const float> samples) = 0;  // one transmission, whole, at Config::sample_rate
    virtual void tx_flush() = 0;                                // drop what is queued: it was not played
    virtual void ptt(bool on) = 0;
    virtual void host_line(const std::string& line) = 0;        // the command port's, without CR
    virtual void host_data(std::span<const std::uint8_t> data) = 0;
    virtual void log(const std::string& /*line*/) {}
    virtual void tx_report(const TxReport&) {}
    virtual void rx_report(const RxReport&) {}
};

class Modem {
public:
    // Throws std::invalid_argument for a sample rate that is not a multiple of config::FS, a negative or
    // non-finite delay or gain.
    Modem(const Config& cfg, Sink& sink);
    ~Modem();
    Modem(const Modem&) = delete;
    Modem& operator=(const Modem&) = delete;

    // A VARA command line (host/host.hpp), without CR; its replies before it returns. ACCEPT is refused:
    // there is no port to dial.
    void command(std::string_view line);
    void send(std::span<const std::uint8_t> data);  // the data port's
    // Received audio at Config::sample_rate, and that many samples of time. While a transmission of ours is
    // out or the radio keyed, silence is heard in its place.
    void rx_audio(std::span<const float> samples);
    // Time without audio (the radio keyed and no capture, or a capture that stalled). advance(0) only looks
    // at what is due: the radio's release after tx_played.
    void advance(std::uint64_t samples);
    // Samples of tx_audio the application has played, a running total; never more than were handed over and
    // not flushed. May be called from a Sink method. The radio is released by the next rx_audio or advance.
    void tx_played(std::uint64_t total_samples);
    // Stops the call under way at its next step, and every later one at once: nothing more is decoded.
    // From a Sink method (the application is going away).
    void halt() { halted_ = true; }

    bool keyed() const { return keyed_; }
    arq::Engine& engine() { return engine_; }
    host::Host& host() { return host_; }

private:
    void feed(const float* x, std::size_t n);  // x null: silence
    void block();
    void start_tx(const std::vector<double>& burst, std::int64_t n0);
    void service_tx();
    void flush();
    void unkey();
    void deliver();
    void drain_log();
    std::uint64_t consumed() const;

    void on_burst(const arq::BurstHeard& h);
    void on_decoded(int decoded, int failed);
    void on_send(const arq::TxBurstPtr& b);
    void on_nod(const arq::TxBurstPtr& b);
    std::int64_t tx_new();  // stream bytes first sent since the last asked, this session
    std::int64_t rx_new();  // ... first received

    Config cfg_;
    Sink& sink_;
    arq::Engine engine_;
    host::Host host_;
    std::size_t ratio_;                      // sample_rate / FS
    std::optional<audio::Decimator> down_;   // nullopt: ratio 1
    std::vector<double> in_, pending_;       // a chunk at the application's rate; decimated, not yet a block
    std::vector<float> out_;
    bool log_ = false;                       // the core's log sink is ours
    bool halted_ = false;

    std::uint64_t clock_ = 0;  // samples heard or advanced, at the application's rate
    // transmit accounting, in samples at the application's rate
    bool keyed_ = false;
    std::uint64_t queued_ = 0, played_ = 0, flushed_ = 0;
    std::uint64_t watchdog_ = 0;             // clock_ by which what is queued must be out
    std::optional<std::uint64_t> unkey_at_;  // all out: the radio is released at this clock_
    std::uint64_t engine_end_ = 0;           // where the engine's clock puts the last burst's end
    double latency_ = -1.0;                  // the engine's clock to the air, averaged (Engine::set_tx_latency)

    std::vector<TxReport> tx_reports_;
    std::vector<RxReport> rx_reports_;
    std::optional<RxReport> rx_open_;  // heard, not yet read
    std::weak_ptr<arq::Station> station_;  // the session these two count
    std::int64_t tx_stream_ = 0, rx_stream_ = 0;
};

}  // namespace data2g::embed
