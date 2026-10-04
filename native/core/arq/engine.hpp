// data2g/arq/engine.py: one station's live stack, clocked by audio samples.
//
// step(x) -> (audio out, PTT) once per block at FS: what was heard goes to
// the streaming receiver (tnc::Receiver), headers and bursts to the ARQ
// session and the gear shifter (and to KISS, when the engine serves it
// too), and whatever the session sends is played out, half duplex. Its
// clock is the sample count, so the same code runs behind a sound card or
// back to back with another Engine through a simulated channel.
//
// Threads. A step has two stages:
// - the receiver stage (search, BUSY: Receiver::feed_deferred) always runs
//   on the thread that calls step();
// - the session stage (receiving a burst, ModemRx with its DD pass, KISS,
//   CQ frames, the session, TX audio, the recorder) runs in step() too by
//   default (sync): step(x) returns this block's output, exactly as Python
//   does, deterministically (parity tests, two engines in a simulation).
// With `worker`, the session stage runs on a worker thread, fed blocks in
// order, so a burst's decode and DD don't hold the next blocks' search and
// BUSY. Every session-stage time is the one sync mode gives (a burst is
// handled at the block it ended in, so reply deadlines don't move); what
// varies, with the decode's wall time, is how far output trails input, as
// Python's backlog does behind a sound card. Not deterministic: it is for
// real time. In worker mode the session stage's state (session(), the
// KissLink, the soft-bit store) is the worker's: reach it only through
// post() and the after_block callback, which run there between blocks.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "arq/phy.hpp"
#include "arq/policy.hpp"
#include "arq/session.hpp"
#include "kisslink/kisslink.hpp"
#include "tnc/tnc.hpp"

namespace data2g::arq {

inline constexpr double MAX_BURST_S = 16.0;  // longest burst accepted from a header
// ID frames (docs/arq.md §7a): in a session, one goes ahead of this station's
// turn at least this often (FCC 97.119: every 10 minutes), and one more once
// the session is over: after its last burst (a DISC_ACK), else ID_GUARD_S
// after it closed, when the peer's own ID (after its DISC_ACK) has been heard
inline constexpr double ID_INTERVAL_S = 600.0;
inline constexpr double ID_GUARD_S = REPLY_START_S;

// engine.Recorder (--record-dir), in the same formats: events.jsonl (one
// JSON object per line, as json.dumps writes it), rx_NNNNN.npz per burst
// heard (an "audio" float32 array; stored, not deflated), and audio_in.f16:
// everything the receiver was fed, float16 at FS (zeros while transmitting).
class Recorder {
public:
    Recorder(const std::string& dir, const std::string& call);  // throws std::runtime_error
    void audio(std::span<const double> x);
    // fields: (key, JSON value) pairs after "kind"
    void event(std::string_view kind, const std::vector<std::pair<std::string, std::string>>& fields);
    std::string rx(double t, std::span<const double> audio, const tnc::Pending& header, bool lost,
                   const std::optional<Measured>& meas, const std::optional<tnc::NoiseSnapshot>& noise = std::nullopt);

private:
    std::string dir_;
    std::ofstream log_, audio_;
    int n_ = 0;
};

// JSON values as Python's json.dumps writes them.
std::string json_str(std::string_view s);
std::string json_num(double v);
std::string json_measured(const Measured& m);  // phy.measure's dict
std::string json_noise(const tnc::NoiseSnapshot& n);  // tnc.NoiseProfile.snapshot()'s dict
// The bytes of a .npz holding one float32 array (np.load reads it).
std::vector<std::uint8_t> npz_f32(const std::string& name, std::span<const float> a);
std::uint16_t to_half(double x);  // numpy's float64 -> float16: round to nearest even

struct EngineConfig {
    double ptt_delay_s = 0.1;
    std::string record_dir;  // empty: no recording
    std::optional<std::uint64_t> seed;  // the engine's DefaultRng (nullopt: random)
    double min_header_score = 0.0;
    kisslink::KissLink* kiss = nullptr;  // served too (the KISS personality); not owned
    double stats_interval_s = 60.0;
    bool dd = dd_default();
    std::optional<double> dd_budget_s = DD_BUDGET_S;  // nullopt: no limit
    bool worker = false;  // the session stage on a worker thread (see the top)
    // Worker: audio queued for the session stage, at most. Past it blocks
    // are dropped (counted, logged once per episode) and the receiver
    // reset; their time still passes on the session stage, as silence.
    double max_backlog_s = 60.0;
    // The gear shifter's noise rule, its tail weight (GearShifter::noise_rule); 0: off.
    double noise_rule = NOISE_RULE;
};

// How a binding swaps in its own objects; every one optional.
struct EngineHooks {
    std::function<std::shared_ptr<Policy>()> policy;  // default: a GearPolicy per session
    std::shared_ptr<Rng> rng;  // the engine's draws (default: DefaultRng(seed))
    std::function<std::shared_ptr<Rng>(double seed)> session_rng;  // default: DefaultRng
    std::function<std::shared_ptr<Session>(const std::string& call, std::shared_ptr<Policy>, std::shared_ptr<Rng>,
                                           const std::vector<std::string>& aliases, double stats_interval_s)>
        session;
};

// A burst heard (Engine::set_on_burst): what a front end's burst log shows.
struct BurstHeard {
    double t = 0.0;  // engine time, s
    std::string submode;
    int n_cw = 0;
    bool lost = false;  // header heard, burst not (the recorder's "lost")
    std::optional<double> snr_db;  // phy.measure's snr_est; nullopt when lost
};

class Engine {
public:
    struct Out {
        std::vector<double> audio;  // a burst's peak at 1.0
        bool ptt = false;
    };
    struct Tx {
        TxBurstPtr burst;
        std::vector<double> audio;
        std::size_t pos = 0;
    };

    explicit Engine(std::string call, EngineConfig cfg = {}, EngineHooks hooks = {});
    virtual ~Engine();  // stop()s
    // Joins the worker after the block in hand; no step() after it. A
    // subclass overriding a seam calls it from its own destructor, before
    // its part is gone.
    void stop();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // -- audio side (the engine thread)
    Out step(std::span<const double> x);
    bool busy() const { return busy_now_; }  // a burst arriving (the receiver stage's, latest)
    bool channel_busy() const { return channel_busy_now_; }  // the host's BUSY ON/OFF
    // Samples dropped because the worker fell max_backlog_s behind (any thread).
    std::uint64_t decode_dropped() const { return dropped_.load(std::memory_order_relaxed); }

    // -- the session stage: the engine thread in sync mode; in worker mode
    // through post() / after_block only
    void post(std::function<void()> f);  // runs before the next block's session stage
    void set_after_block(std::function<void(bool ptt)> f) { after_block_ = std::move(f); }
    // Every burst heard, on the session stage. Set it before the first step().
    void set_on_burst(std::function<void(const BurstHeard&)> f) { on_burst_ = std::move(f); }

    const std::string& call() const { return call_; }
    const std::vector<std::string>& aliases() const { return aliases_; }
    Session& session() { return *session_; }
    const std::shared_ptr<Session>& session_ptr() const { return session_; }
    double now() const { return static_cast<double>(n_) / config::FS; }
    std::int64_t n() const { return n_; }
    void set_n(std::int64_t n) { n_ = n; }
    void listen(bool on = true);
    void set_call(const std::string& call, const std::vector<std::string>& aliases = {});
    void abort();
    void connect(const std::string& peer, int cap);  // throws std::runtime_error while a session is under way
    void set_chat(bool on);
    std::vector<std::string> events();  // CONNECTED ..., DISCONNECTED ..., CQFRAME call cap, ID call key
    void send_cq(const std::string& call, int cap);  // throws std::runtime_error, std::invalid_argument (call)
    // A control-only burst anyone can read (frame type SESSION, mask 0), in the cap's connect mode.
    TxBurstPtr open_frame(int cap, int ext, const Bytes& body) const;
    double id_interval_s = ID_INTERVAL_S;
    std::optional<double> next_event() { return session_->next_event(); }
    const std::optional<Tx>& tx() const { return tx_; }
    const std::deque<TxBurstPtr>& extra() const { return extra_; }
    std::vector<Bytes>& kiss_rx() { return kiss_rx_; }  // frames heard for KISS clients; the host clears it
    kisslink::KissLink* kiss() const { return cfg_.kiss; }
    const modem::Accept& accept() const { return accept_; }
    SoftStore& store() { return store_; }
    const EngineConfig& config() const { return cfg_; }

protected:
    // Seams for a binding (a Python receiver, a patched phy.tx_audio) and
    // for tests (a slow decode).
    virtual std::vector<tnc::Receiver::Item> receiver_feed(std::span<const double> x) {
        return receiver_.feed_deferred(x);
    }
    virtual bool receiver_busy() { return receiver_.busy(); }
    virtual bool receiver_channel_busy() { return receiver_.channel_busy(); }
    virtual void receiver_reset() { receiver_.reset(); }
    virtual std::vector<double> tx_audio(const TxBurst& b) { return arq::tx_audio(b); }
    virtual void hear_burst(tnc::BurstEvent& ev, double t);

private:
    struct Block {
        std::vector<double> x;
        std::vector<tnc::Receiver::Item> items;
        bool busy = false;
        std::uint64_t gen = 0, seq = 0;
        std::int64_t gap = 0;  // samples dropped just before this block
    };
    struct Done {
        Out out;
        bool sound = false;  // anything played (else silence that can be dropped to catch up)
    };

    void new_session();
    bool idle() const;
    Done process(Block& b);  // the session stage
    void hear(std::vector<tnc::Receiver::Item>& items, double t);
    TxBurstPtr kiss_burst(std::int64_t k, bool busy);
    bool cq(ModemRx& rx);
    TxBurstPtr id_frame(const Session& s) const;
    void id_check(double t);
    std::vector<TxBurstPtr> with_id(const TxBurstPtr& burst, double t);
    // Bursts back to back on one PTT; `main` is the one the session is told has gone.
    void start_tx(const std::vector<TxBurstPtr>& bursts, double t, const TxBurstPtr& main);
    void request_reset();
    void apply_reset();
    void run_posted();
    void work();
    Out next_out(std::size_t k);  // under mu_

    std::string call_;
    std::vector<std::string> aliases_;
    EngineConfig cfg_;
    EngineHooks hooks_;
    std::shared_ptr<Rng> rng_;
    modem::Accept accept_;
    tnc::Receiver receiver_;
    tnc::NoiseProfile noise_;  // the passband's noise between bursts (recorded with each burst heard)
    std::int64_t ptt_delay_;
    std::optional<Recorder> rec_;
    std::int64_t n_ = 0;
    std::optional<Tx> tx_;
    std::shared_ptr<Session> session_;
    SoftStore store_;
    bool chat_ = false;
    std::deque<TxBurstPtr> extra_;  // bursts outside any session (CQ frames)
    std::vector<std::string> events_;  // host notifications from outside the session
    std::vector<Bytes> kiss_rx_;
    std::int64_t kiss_busy_ = 0;  // samples of unbroken BUSY a queued KISS burst has waited
    bool kiss_deferred_ = false;  // ... and it has waited on BUSY
    std::int64_t kiss_slot_ = 0;  // next p-persistence slot, samples
    std::shared_ptr<Session> id_for_;      // the session ID frames are being sent for
    std::optional<double> id_due_;         // its next ID (nullopt: closed, its last ID pending or sent)
    std::shared_ptr<Session> id_pending_;  // its last ID, after it closed, from id_pending_t_
    double id_pending_t_ = 0.0;
    double hold_ = 0.0;  // nothing new goes before this (the peer's last ID may follow its DISC_ACK)
    std::function<void(bool)> after_block_;
    std::function<void(const BurstHeard&)> on_burst_;

    // between the stages
    std::atomic<bool> transmitting_{false}, busy_now_{false}, channel_busy_now_{false};
    std::atomic<std::uint64_t> want_gen_{0};  // the session stage asks for a receiver reset
    std::uint64_t gen_ = 0, seq_ = 0;          // the receiver stage's
    std::int64_t gap_ = 0;                     // the receiver stage's: samples dropped since the last block queued
    std::atomic<std::size_t> queued_{0};       // samples in in_
    std::atomic<std::uint64_t> dropped_{0};
    std::mutex mu_;
    std::condition_variable wake_, done_cv_;
    std::deque<Block> in_;
    std::deque<Done> out_;
    std::deque<std::function<void()>> posted_;
    std::uint64_t n_done_ = 0;
    int slow_ = 0;  // blocks with a burst queued or in the session stage
    bool stop_ = false;
    std::thread worker_;
};

}  // namespace data2g::arq
