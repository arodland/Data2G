// The station runtime both front ends share (data2g-host, data2g-gui):
// host.py main()'s options and its server. One radio, two personalities,
// each on or off: VARA (ARQ sessions on a command port and the next one, as
// VARA HF) and KISS (frames on a KISS port, modes shifted per station).
//
// Threads:
// - the owner's (a Qt event loop): the TCP servers (command, data, KISS),
//   start(), stop() and every poll below;
// - engine (the receive thread): reads the capture FIFO a block (0.1 s) at a
//   time and steps the Engine; it never touches the transmitter;
// - transmit (app/transmitter.hpp): keys PTT (rig::Keyer), plays each burst
//   the engine hands over on the sound card's own clock and releases PTT, so
//   nothing the receive side computes, or the rig's slow answers, can leave
//   a hole in a burst;
// - decode worker (decode_worker, the default): the engine's session stage,
//   so a burst's decode and DD don't delay search and BUSY;
// - capture and playback (Qt Multimedia QThreads, or PipeIo's two);
// The transmit, capture and playback threads ask for real-time scheduling
// (--realtime, app/realtime.hpp): they do a little work and block. The
// engine and decode threads compute for long stretches and are left alone.
// - the rig controller's worker (Hamlib).
// The Host (VARA semantics) lives on the session stage: the owner reaches it
// with Engine::post(); what it says comes back as queued calls.
//
// QtCore and QtNetwork only: no Widgets, no desktop assumptions (a mobile
// front end could own one too).
//
// The rest of data2g_station, each its own file: app/options.hpp (the
// command line and settings, Args), app/log.hpp (the log and the quit
// signals), app/tcp_servers.hpp (the command, data and KISS listeners),
// app/rx_plot.hpp (constellation-plot points), app/transmitter.hpp (the
// transmit thread), app/realtime.hpp.
#pragma once

#include <QObject>
class QTcpSocket;

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cstddef>
#include <span>

#include "app/log.hpp"
#include "app/options.hpp"
#include "app/rx_plot.hpp"

namespace data2g {
namespace arq {
class Engine;
enum class SessionState;  // arq/session.hpp
}
namespace audio {
class CaptureFifo;
class PlaybackFifo;
class PipeIo;
}  // namespace audio
namespace host {
class Host;
}
namespace kisslink {
class KissLink;
}
namespace monitor {
struct Dump;
}
namespace rig {
class RigController;
}  // namespace rig
}  // namespace data2g

namespace data2g::app {

class Port;
class KissServer;
class Transmitter;

// Bad options found only at start() (argparse would have exited with 2).
struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The link as the session stage last saw it (after every block).
struct LinkStatus {
    arq::SessionState state{};  // IDLE
    std::string peer;
    int cap = 2;                    // arq::bw_name(cap): 500, 1200, 2300
    std::string mode;               // the last burst's submode, heard or sent
    std::int64_t rx_bytes = 0, tx_bytes = 0;  // this connection's, delivered / acked
};

struct BurstLogEntry {
    std::chrono::system_clock::time_point when;
    std::string mode;
    int n_cw = 0;
    bool lost = false;
    std::optional<double> snr_db;
    int decoded = 0, failed = 0;  // slots (Engine::set_on_decoded)
};

struct AudioCounters {
    std::uint64_t overflows = 0, underruns = 0, late = 0, dropped = 0;
    double backlog_s = 0.0;
    std::uint64_t decode_dropped = 0;  // samples the decode worker's queue dropped (Engine::decode_dropped)
};

class Station {
public:
    explicit Station(Args a);
    ~Station();  // stop()s
    Station(const Station&) = delete;
    Station& operator=(const Station&) = delete;

    // Listens, opens audio and the rig, starts the threads. Throws
    // UsageError, std::invalid_argument (no such audio device) or
    // std::runtime_error (a port taken, a device that won't open); what it
    // had started is stopped again.
    void start();
    // PTT down, threads joined, ports closed, devices released. Idempotent.
    void stop();
    bool running() const { return running_; }
    // The engine thread died (logged CRITICAL); stop() and report it.
    bool failed() const { return failed_; }
    const Args& args() const { return a_; }

    // -- polled by the owner, never blocking on the engine
    LinkStatus link() const;
    std::vector<BurstLogEntry> take_bursts();  // since the last call (the newest 256 kept)
    // The newest since the last call. The first call starts the live view (--live-constellation).
    std::optional<RxConstellation> take_constellation();
    bool ptt() const;  // the rig keyed (or, with no rig, would be)
    bool busy() const;  // the host's BUSY
    AudioCounters counters() const;
    // The newest `n` samples (FS) the engine read, and how many it has read
    // in all (unchanged: nothing new).
    // `tx`: the newest block is our own TX audio (8 kHz, peak 1.0), not input.
    std::vector<double> input_tail(std::size_t n, std::uint64_t* total = nullptr, bool* tx = nullptr) const;
    // The dial frequency the rig last reported (--rig-poll-interval > 0), or nothing.
    std::optional<double> rig_frequency() const;
    // Monitoring: while on, every burst heard, and every one we send, is
    // also decoded as a packet dump (core/monitor), whoever it is for, on a
    // thread of its own. Its sessions are followed from one turning on to
    // the next stop().
    void set_monitor(bool on) { monitor_on_ = on; }
    // The operator's Abort: any session under way is dropped at once and the
    // command clients told DISCONNECTED (host::Host::abort_session).
    void abort_session();
    std::vector<monitor::Dump> take_dumps();  // since the last call (the newest 256 kept)

private:
    struct Outbox;
    struct SoundCard;
    struct MonitorThread;
    struct Dial;
    // session stage
    void flush(std::uint64_t asker = 0);
    void note_link();
    void log_burst();  // reading_ into bursts_; status_mu_ held
    // owner's thread
    void deliver(const Outbox& o);
    void dial(const std::string& host, int port, unsigned gen, std::shared_ptr<Dial> d);  // ACCEPT's connection
    void hang_up();
    // transmit thread (Transmitter's on_ptt)
    void announce_ptt(bool on);
    // engine (receive) thread
    void engine_loop();
    void watch_counters();
    void tap(std::span<const double> x, bool tx);

    Args a_;
    QObject ctx_;  // queued calls from other threads land here, on the owner's thread
    std::unique_ptr<kisslink::KissLink> link_;
    std::unique_ptr<arq::Engine> engine_;
    std::unique_ptr<host::Host> host_;
    std::unique_ptr<Port> cmd_, data_;
    unsigned dial_gen_ = 0, dial_live_ = 0;  // session stage: the last dial; the live session's (0: the data port)
    QTcpSocket* dialed_ = nullptr;           // owner's thread: ACCEPT's connection, dial generation dialed_gen_
    unsigned dialed_gen_ = 0;
    std::unique_ptr<KissServer> kiss_;
    int rate_;
    std::unique_ptr<audio::CaptureFifo> cap_;
    std::unique_ptr<audio::PlaybackFifo> play_;
    std::unique_ptr<audio::PipeIo> pipe_;
    std::unique_ptr<SoundCard> card_;
    std::unique_ptr<rig::RigController> rig_;
    std::unique_ptr<MonitorThread> monitor_;  // made in start(), before the engine runs
    std::atomic<bool> monitor_on_{false};
    std::unique_ptr<Transmitter> tx_;  // made last in start(), its thread with it; the engine thread submits to it
    std::thread engine_thread_;
    std::atomic<bool> stop_{false}, failed_{false};
    std::atomic<bool> rt_warned_{false};
    bool running_ = false;
    std::uint64_t overflows_ = 0, underruns_ = 0, late_ = 0, dropped_ = 0;  // engine thread

    std::string mode_;  // session stage
    mutable std::mutex status_mu_;
    LinkStatus link_status_;
    std::deque<BurstLogEntry> bursts_;
    BurstLogEntry reading_;  // heard, awaiting on_decoded; engine's thread
    std::optional<RxConstellation> constellation_;
    std::optional<RxConstellation> live_;  // the burst arriving's, kept for when it is lost
    bool live_asked_ = false;              // owner's thread
    mutable std::mutex tap_mu_;
    std::vector<double> tap_;  // ring of the newest input
    std::uint64_t tapped_ = 0;
    bool tap_tx_ = false;  // the newest tapped block was TX audio
};

}  // namespace data2g::app
