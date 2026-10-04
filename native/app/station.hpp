// The station runtime both front ends share (data2g-host, data2g-gui):
// host.py main()'s options and its server. One radio, two personalities,
// each on or off: VARA (ARQ sessions on a command port and the next one, as
// VARA HF) and KISS (frames on a KISS port, modes shifted per station).
//
// Threads:
// - the owner's (a Qt event loop): the TCP servers (command, data, KISS),
//   start(), stop() and every poll below;
// - engine: reads the capture FIFO a block (0.1 s) at a time, steps the
//   Engine, keys PTT and queues TX audio (rig::Keyer);
// - decode worker (decode_worker, the default): the engine's session stage,
//   so a burst's decode and DD don't delay search and BUSY;
// - capture and playback (Qt Multimedia QThreads, or PipeIo's two);
// - the rig controller's worker (Hamlib).
// The Host (VARA semantics) lives on the session stage: the owner reaches it
// with Engine::post(); what it says comes back as queued calls.
//
// QtCore and QtNetwork only: no Widgets, no desktop assumptions (a mobile
// front end could own one too).
#pragma once

#include <QObject>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <cstddef>
#include <span>

#include "rig/hamlib/hamlib.hpp"  // types only: no libhamlib needed

namespace data2g {
namespace arq {
class Engine;
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
namespace rig {
class RigController;
class Keyer;
}  // namespace rig
}  // namespace data2g

namespace data2g::app {

// --- logging: Python's basicConfig(format="%(asctime)s %(levelname)s %(message)s") ---

enum Level { DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40, CRITICAL = 50 };
extern std::atomic<int> g_level;
void log_line(int level, const std::string& msg);
[[gnu::format(printf, 2, 3)]] void logf(int level, const char* fmt, ...);
// --log-level's value -> a level; nullopt: not one.
std::optional<int> parse_level(const std::string& s);
// Routes the arq log (sessions, engine) through log_line at g_level.
void install_arq_log();

// --- the options: host.py main()'s argparse ---------------------------------------------

std::string default_record_dir();  // recordings/YYYYmmdd-HHMMSS

struct Args {
    bool vara = true, kiss = true;
    int kiss_port = 8100;
    std::string kiss_address = "127.0.0.1";
    double kiss_busy_limit = 60.0;
    int kiss_bw = 2400;
    std::optional<std::string> broadcast_mode, mycall, input_device, output_device;
    std::string host = "127.0.0.1";
    int command_port = 8300;
    bool list_audio_devices = false, list_modes = false;
    int sample_rate = 48000;
    double output_volume = 0.0;
    std::string rigctld_host = "localhost";
    int rigctld_port = 4532;
    // Hamlib (SSTVAE's rig settings). Model 2 with no rig_device is rigctld
    // at rigctld_host:rigctld_port, so the two flags above work as before.
    // Enumerated values in SSTVAE's lowercase spellings; "default": leave
    // the backend's own. See rig_choices().
    bool rig = true;  // false: no rig (so no PTT)
    int rig_model = 2;
    std::string rig_device;  // serial device or host:port; empty: model 2's rigctld, else Hamlib's
    int rig_baud = 0;        // 0: the backend's
    std::string rig_data_bits = "default", rig_stop_bits = "default", rig_parity = "default",
                rig_handshake = "default", rig_dtr = "default", rig_rts = "default";
    std::string ptt_method = "cat", ptt_device, ptt_audio = "mic";
    std::string rig_mode = "none";
    int rig_timeout_ms = 1000, rig_retries = 1;
    double rig_poll_interval = 0.0;  // s; 0: key only (frequency not read)
    bool rig_debug = false;          // Hamlib's trace into the log
    bool list_rigs = false;
    int ptt_on_delay_ms = 100, ptt_off_delay_ms = 50, tx_lead_ms = 100;
    double min_header_score = 0.0;
    int buffer_credit = -1;
    std::string record_dir = default_record_dir();
    std::string log_level = "INFO";
    double stats_interval = 60.0;
    double noise_rule = 1.0;  // the gear shifter's noise rule, its tail weight; 0: off
    // additions
    bool decode_worker = true;
    std::string audio_io;  // "pipe:IN,OUT"; empty: the sound card

    bool operator==(const Args&) const = default;
};

// The command line over `a` (defaults, or what a front end saved). Exits as
// argparse does on -h and on a bad flag. `prog` names the binary in errors.
Args parse(int argc, char** argv, Args a = {}, const char* prog = "data2g-host");
[[noreturn]] void usage_error(const std::string& msg, const char* prog = "data2g-host");
// What main() checks once parsed; the message, or nullopt when fine.
std::optional<std::string> check(const Args& a);
int kiss_cap(int hz);
// The rig settings' allowed values, by Args field name ("rig_parity", ...).
const std::vector<std::string>& rig_choices(std::string_view field);
// Whether `a` asks for a rig at all, and the device Hamlib is given.
bool rig_enabled(const Args& a);
std::string rig_device(const Args& a);
// The one mapping from the options to Hamlib's (the GUI's test buttons use it
// too). `a` must have passed check().
rig::HamlibConfig hamlib_config(const Args& a);
// --list-rigs: Hamlib's models (number, manufacturer, model, status). False
// when built without Hamlib.
bool list_rigs();
void list_modes(int kiss_bw);

// --- the runtime -------------------------------------------------------------------------

// Bad options found only at start() (argparse would have exited with 2).
struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The link as the session stage last saw it (after every block).
struct LinkStatus {
    std::string state = "idle";  // arq::state_name: idle, listen, connecting, connected, disconnecting, closed
    std::string peer;
    int cap = 2;                    // 0: 500 Hz, 1: 1200, 2: 2300
    std::string mode;               // the last burst's submode, heard or sent
    std::int64_t rx_bytes = 0, tx_bytes = 0;  // this connection's, delivered / acked
};

struct BurstLogEntry {
    std::chrono::system_clock::time_point when;
    std::string mode;
    int n_cw = 0;
    bool lost = false;
    std::optional<double> snr_db;
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
    bool ptt() const { return ptt_; }
    bool busy() const;  // the host's BUSY
    AudioCounters counters() const;
    // The newest `n` samples (FS) the engine read, and how many it has read
    // in all (unchanged: nothing new).
    // `tx`: the newest block is our own TX audio (8 kHz, peak 1.0), not input.
    std::vector<double> input_tail(std::size_t n, std::uint64_t* total = nullptr, bool* tx = nullptr) const;
    // The dial frequency the rig last reported (--rig-poll-interval > 0), or nothing.
    std::optional<double> rig_frequency() const;

private:
    struct Outbox;
    struct Port;
    struct KissServer;
    struct SoundCard;
    // session stage
    void flush();
    void note_link();
    // owner's thread
    void deliver(const Outbox& o);
    // engine thread
    void engine_loop();
    void watch_counters();
    void tap(std::span<const double> x, bool tx);

    Args a_;
    QObject ctx_;  // queued calls from other threads land here, on the owner's thread
    std::unique_ptr<kisslink::KissLink> link_;
    std::unique_ptr<arq::Engine> engine_;
    std::unique_ptr<host::Host> host_;
    std::unique_ptr<Port> cmd_, data_;
    std::unique_ptr<KissServer> kiss_;
    int rate_;
    std::unique_ptr<audio::CaptureFifo> cap_;
    std::unique_ptr<audio::PlaybackFifo> play_;
    std::unique_ptr<audio::PipeIo> pipe_;
    std::unique_ptr<SoundCard> card_;
    std::unique_ptr<rig::RigController> rig_;
    std::unique_ptr<rig::Keyer> keyer_;
    std::thread engine_thread_;
    std::atomic<bool> stop_{false}, failed_{false}, ptt_{false};
    bool running_ = false;
    std::uint64_t overflows_ = 0, underruns_ = 0, late_ = 0, dropped_ = 0;  // engine thread

    std::string mode_;  // session stage
    mutable std::mutex status_mu_;
    LinkStatus link_status_;
    std::deque<BurstLogEntry> bursts_;
    mutable std::mutex tap_mu_;
    std::vector<double> tap_;  // ring of the newest input
    std::uint64_t tapped_ = 0;
    bool tap_tx_ = false;  // the newest tapped block was TX audio
};

}  // namespace data2g::app
