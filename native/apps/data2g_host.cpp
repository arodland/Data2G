// data2g-host: data2g/host.py's server in C++. One radio, two
// personalities, each on or off: VARA (ARQ sessions on a command port and
// the next one, as VARA HF) and KISS (frames on --kiss-port, modes shifted
// per station). Same command line as host.py's main(), plus --audio-io,
// --decode-worker / --no-decode-worker.
//
//   data2g-host --mycall W1AW --input-device USB --output-device USB --rigctld-port 4532
//
// Threads:
// - main: the Qt event loop, the TCP servers (command, data, KISS);
// - engine: reads the capture FIFO a block (0.1 s) at a time, steps the
//   Engine, keys PTT and queues TX audio (rig::Keyer);
// - decode worker (the default; --no-decode-worker: none): the engine's
//   session stage, so a burst's decode and DD don't delay search and BUSY;
// - capture and playback (Qt Multimedia QThreads, or PipeIo's two);
// - the rig controller's worker (Hamlib).
// The Host (VARA semantics) lives on the session stage: the main thread
// reaches it with Engine::post(); what it says goes back as queued calls.

#include <QCoreApplication>
#include <QHostAddress>
#include <QHostInfo>
#include <QPointer>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTimer>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "arq/engine.hpp"
#include "arq/modes.hpp"
#include "arq/policy.hpp"
#include "audio/audio.hpp"
#include "audio/fifo.hpp"
#include "audio/filters.hpp"
#include "audio/pipe.hpp"
#include "generated/config.hpp"
#include "host/host.hpp"
#include "kisslink/kisslink.hpp"
#include "rig/controller.hpp"
#include "rig/ptt.hpp"
#include "tnc/tnc.hpp"

#ifdef DATA2G_HAVE_QTAUDIO
#include "audio/qt/qtaudio.hpp"
#endif
#ifdef DATA2G_HAVE_RIG
#include "rig/hamlib/hamlib.hpp"
#endif

using namespace data2g;

namespace {

// --- logging: Python's basicConfig(format="%(asctime)s %(levelname)s %(message)s") ---

enum Level { DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40, CRITICAL = 50 };
std::atomic<int> g_level{INFO};
std::mutex g_log_mu;

const char* level_name(int level) {
    switch (level) {
        case DEBUG: return "DEBUG";
        case INFO: return "INFO";
        case WARNING: return "WARNING";
        case ERROR: return "ERROR";
        case CRITICAL: return "CRITICAL";
        default: return "Level";
    }
}

void log_line(int level, const std::string& msg) {
    if (level < g_level) return;
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm{};
    localtime_r(&t, &tm);
    char stamp[32];
    std::strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", &tm);
    std::lock_guard lock(g_log_mu);
    std::fprintf(stderr, "%s,%03d %s %s\n", stamp, static_cast<int>(ms), level_name(level), msg.c_str());
}

[[gnu::format(printf, 2, 3)]] void logf(int level, const char* fmt, ...) {
    if (level < g_level) return;
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    log_line(level, buf);
}

// --- the command line: host.py main()'s argparse ---------------------------------------

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
    int ptt_on_delay_ms = 100, ptt_off_delay_ms = 50, tx_lead_ms = 100;
    double min_header_score = 0.0;
    int buffer_credit = -1;
    std::string record_dir;
    std::string log_level = "INFO";
    double stats_interval = 60.0;
    // additions
    bool decode_worker = true;
    std::string audio_io;  // "pipe:IN,OUT"; empty: the sound card
};

const char* USAGE =
    "usage: data2g-host [-h] [--vara | --no-vara] [--kiss | --no-kiss] [--kiss-port KISS_PORT]\n"
    "                   [--kiss-address KISS_ADDRESS] [--kiss-busy-limit S] [--kiss-bw {2400,500}]\n"
    "                   [--broadcast-mode MODE] [--mycall MYCALL] [--host HOST] [--command-port COMMAND_PORT]\n"
    "                   [--list-audio-devices] [--input-device INPUT_DEVICE] [--output-device OUTPUT_DEVICE]\n"
    "                   [--sample-rate SAMPLE_RATE] [--output-volume OUTPUT_VOLUME] [--rigctld-host RIGCTLD_HOST]\n"
    "                   [--rigctld-port RIGCTLD_PORT] [--ptt-on-delay-ms PTT_ON_DELAY_MS]\n"
    "                   [--ptt-off-delay-ms PTT_OFF_DELAY_MS] [--tx-lead-ms TX_LEAD_MS]\n"
    "                   [--min-header-score MIN_HEADER_SCORE] [--buffer-credit BUFFER_CREDIT]\n"
    "                   [--record-dir RECORD_DIR] [--log-level LOG_LEVEL] [--stats-interval S] [--list-modes]\n"
    "                   [--decode-worker | --no-decode-worker] [--audio-io pipe:IN,OUT]\n";

const char* HELP =
    "\nData2G server: a VARA-style TNC and a KISS TNC on one radio\n\n"
    "options:\n"
    "  -h, --help            show this help message and exit\n"
    "  --vara, --no-vara     the VARA personality: ARQ sessions on --command-port and the next (default: on)\n"
    "  --kiss, --no-kiss     the KISS personality: frames on --kiss-port, modes shifted per station (default: on)\n"
    "  --kiss-port KISS_PORT as VARA HF's (default 8100)\n"
    "  --kiss-address KISS_ADDRESS (default 127.0.0.1)\n"
    "  --kiss-busy-limit S   a KISS burst held this long by BUSY is sent anyway (default 60)\n"
    "  --kiss-bw {2400,500}  KISS bandwidth cap, Hz (default 2400)\n"
    "  --broadcast-mode MODE KISS mode for UI frames, non-AX.25 and unreported stations\n"
    "                        (default: qpsk-r1/5, n10-qpsk-r1/5 with --kiss-bw 500)\n"
    "  --mycall MYCALL\n"
    "  --host HOST           (default 127.0.0.1)\n"
    "  --command-port COMMAND_PORT  data is on the next port (default 8300)\n"
    "  --list-audio-devices\n"
    "  --input-device INPUT_DEVICE   index or name substring (see --list-audio-devices)\n"
    "  --output-device OUTPUT_DEVICE index or name substring\n"
    "  --sample-rate SAMPLE_RATE     a multiple of 8000 (default 48000)\n"
    "  --output-volume OUTPUT_VOLUME dB; 0 puts a burst's peak at full scale (default 0)\n"
    "  --rigctld-host RIGCTLD_HOST   (default localhost)\n"
    "  --rigctld-port RIGCTLD_PORT   0: no PTT (default 4532)\n"
    "  --ptt-on-delay-ms PTT_ON_DELAY_MS   (default 100)\n"
    "  --ptt-off-delay-ms PTT_OFF_DELAY_MS (default 50)\n"
    "  --tx-lead-ms TX_LEAD_MS       TX audio queued ahead of the sound card: slack for a late audio step\n"
    "                                (default 100)\n"
    "  --min-header-score MIN_HEADER_SCORE (default 0.0)\n"
    "  --buffer-credit BUFFER_CREDIT bytes queued for the next burst that BUFFER leaves out, at most, so VARA\n"
    "                                clients that throttle on it (Pat) keep a whole burst queued; -1: the next\n"
    "                                burst's full capacity, 0: report every unacked byte (plain VARA) (default -1)\n"
    "  --record-dir RECORD_DIR       where every burst heard and sent is logged ('' turns it off)\n"
    "                                (default recordings/YYYYmmdd-HHMMSS)\n"
    "  --log-level LOG_LEVEL         (default INFO)\n"
    "  --stats-interval S            log a connection's throughput this often (0: only at disconnect)\n"
    "                                (default 60)\n"
    "  --list-modes                  modes within --kiss-bw, narrowest first\n"
    "\nadded in the C++ host:\n"
    "  --decode-worker, --no-decode-worker\n"
    "                        burst decode, DD and the session on their own thread, so preamble search and\n"
    "                        BUSY keep running during a decode (BUSY lag with a 1 s DD in flight: 0.02 s\n"
    "                        instead of 0.23 s). --no-decode-worker is host.py's one thread, deterministic.\n"
    "                        (default: on)\n"
    "  --audio-io pipe:IN,OUT  no sound card: raw float32 8 kHz mono read from IN and written to OUT (files\n"
    "                        or named pipes), both at real time, silence while not keyed. Two hosts cross-\n"
    "                        connect through two mkfifo pipes. --sample-rate and the devices are then unused.\n";

[[noreturn]] void usage_error(const std::string& msg) {
    std::fprintf(stderr, "%sdata2g-host: error: %s\n", USAGE, msg.c_str());
    std::exit(2);
}

template <typename T>
T number(const std::string& opt, const std::string& v) {
    try {
        std::size_t used = 0;
        T out;
        if constexpr (std::is_same_v<T, int>) out = std::stoi(v, &used);
        else out = std::stod(v, &used);
        if (used == v.size()) return out;
    } catch (const std::exception&) {
    }
    usage_error("argument " + opt + ": invalid " + (std::is_same_v<T, int> ? "int" : "float") + " value: '" + v + "'");
}

std::string default_record_dir() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", &tm);
    return std::string("recordings/") + buf;
}

Args parse(int argc, char** argv) {
    Args a;
    a.record_dir = default_record_dir();
    std::map<std::string, bool*> flags = {
        {"--list-audio-devices", &a.list_audio_devices}, {"--list-modes", &a.list_modes}};
    std::map<std::string, std::pair<bool*, bool>> switches = {
        {"--vara", {&a.vara, true}},   {"--no-vara", {&a.vara, false}},
        {"--kiss", {&a.kiss, true}},   {"--no-kiss", {&a.kiss, false}},
        {"--decode-worker", {&a.decode_worker, true}}, {"--no-decode-worker", {&a.decode_worker, false}}};
    std::map<std::string, std::function<void(const std::string&, const std::string&)>> valued = {
        {"--kiss-port", [&](auto& o, auto& v) { a.kiss_port = number<int>(o, v); }},
        {"--kiss-address", [&](auto&, auto& v) { a.kiss_address = v; }},
        {"--kiss-busy-limit", [&](auto& o, auto& v) { a.kiss_busy_limit = number<double>(o, v); }},
        {"--kiss-bw", [&](auto& o, auto& v) {
             a.kiss_bw = number<int>(o, v);
             if (a.kiss_bw != 2400 && a.kiss_bw != 500)
                 usage_error("argument --kiss-bw: invalid choice: " + v + " (choose from 2400, 500)");
         }},
        {"--broadcast-mode", [&](auto&, auto& v) { a.broadcast_mode = v; }},
        {"--mycall", [&](auto&, auto& v) { a.mycall = v; }},
        {"--host", [&](auto&, auto& v) { a.host = v; }},
        {"--command-port", [&](auto& o, auto& v) { a.command_port = number<int>(o, v); }},
        {"--input-device", [&](auto&, auto& v) { a.input_device = v; }},
        {"--output-device", [&](auto&, auto& v) { a.output_device = v; }},
        {"--sample-rate", [&](auto& o, auto& v) { a.sample_rate = number<int>(o, v); }},
        {"--output-volume", [&](auto& o, auto& v) { a.output_volume = number<double>(o, v); }},
        {"--rigctld-host", [&](auto&, auto& v) { a.rigctld_host = v; }},
        {"--rigctld-port", [&](auto& o, auto& v) { a.rigctld_port = number<int>(o, v); }},
        {"--ptt-on-delay-ms", [&](auto& o, auto& v) { a.ptt_on_delay_ms = number<int>(o, v); }},
        {"--ptt-off-delay-ms", [&](auto& o, auto& v) { a.ptt_off_delay_ms = number<int>(o, v); }},
        {"--tx-lead-ms", [&](auto& o, auto& v) { a.tx_lead_ms = number<int>(o, v); }},
        {"--min-header-score", [&](auto& o, auto& v) { a.min_header_score = number<double>(o, v); }},
        {"--buffer-credit", [&](auto& o, auto& v) { a.buffer_credit = number<int>(o, v); }},
        {"--record-dir", [&](auto&, auto& v) { a.record_dir = v; }},
        {"--log-level", [&](auto&, auto& v) { a.log_level = v; }},
        {"--stats-interval", [&](auto& o, auto& v) { a.stats_interval = number<double>(o, v); }},
        {"--audio-io", [&](auto&, auto& v) { a.audio_io = v; }},
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::printf("%s%s", USAGE, HELP);
            std::exit(0);
        }
        std::optional<std::string> inline_value;
        if (const auto eq = arg.find('='); arg.starts_with("--") && eq != std::string::npos) {
            inline_value = arg.substr(eq + 1);
            arg.resize(eq);
        }
        if (auto f = flags.find(arg); f != flags.end() && !inline_value) {
            *f->second = true;
        } else if (auto s = switches.find(arg); s != switches.end() && !inline_value) {
            *s->second.first = s->second.second;
        } else if (auto v = valued.find(arg); v != valued.end()) {
            if (!inline_value && i + 1 >= argc) usage_error("argument " + arg + ": expected one argument");
            v->second(arg, inline_value ? *inline_value : std::string(argv[++i]));
        } else {
            usage_error("unrecognized arguments: " + std::string(argv[i]));
        }
    }
    return a;
}

int parse_level(const std::string& s) {
    static const std::map<std::string, int> names = {
        {"DEBUG", DEBUG}, {"INFO", INFO}, {"WARNING", WARNING}, {"WARN", WARNING}, {"ERROR", ERROR}, {"CRITICAL", CRITICAL}, {"FATAL", CRITICAL}};
    std::string u = s;
    for (auto& c : u) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (auto it = names.find(u); it != names.end()) return it->second;
    try {
        std::size_t used = 0;
        const int v = std::stoi(s, &used);
        if (used == s.size()) return v;
    } catch (const std::exception&) {
    }
    std::fprintf(stderr, "data2g-host: Unknown level: '%s'\n", s.c_str());
    std::exit(1);
}

int kiss_cap(int hz) { return hz == 500 ? 0 : 2; }

std::atomic<bool> g_quit{false};
extern "C" void on_signal(int) { g_quit = true; }

// --- the server ------------------------------------------------------------------------

// One TCP listener holding at most one client (host.py's _Port): a new
// client replaces the old one, whose close then isn't a loss.
class Port : public QObject {
public:
    Port(const QHostAddress& addr, quint16 port, bool lines, std::function<void(const QByteArray&)> on_input,
         std::function<void()> on_close = {})
        : lines_(lines), on_input_(std::move(on_input)), on_close_(std::move(on_close)) {
        if (!srv_.listen(addr, port))
            throw std::runtime_error("can't listen on " + addr.toString().toStdString() + ":" + std::to_string(port) + ": " +
                                     srv_.errorString().toStdString());
        connect(&srv_, &QTcpServer::newConnection, this, [this] { accept(); });
    }

    void send(const QByteArray& data) {
        if (client_ && !data.isEmpty()) client_->write(data);
    }
    quint16 port() const { return srv_.serverPort(); }

private:
    void accept() {
        while (QTcpSocket* c = srv_.nextPendingConnection()) {
            logf(INFO, "client %s:%d on port %d", c->peerAddress().toString().toStdString().c_str(), c->peerPort(), srv_.serverPort());
            QPointer<QTcpSocket> old = client_;
            client_ = c;  // the new client first: the old one's close isn't a loss
            buf_.clear();
            if (old) {
                old->disconnect(this);
                old->close();
                old->deleteLater();
            }
            connect(c, &QTcpSocket::readyRead, this, [this, c] { read(c); });
            connect(c, &QTcpSocket::disconnected, this, [this, c] {
                c->deleteLater();
                if (client_ == c) {
                    client_ = nullptr;
                    if (on_close_) on_close_();
                }
            });
        }
    }

    void read(QTcpSocket* c) {
        const QByteArray d = c->readAll();
        if (!lines_) {
            on_input_(d);
            return;
        }
        buf_ += d;
        while (true) {
            const qsizetype r = buf_.indexOf('\r'), n = buf_.indexOf('\n');
            const qsizetype i = r < 0 ? n : (n < 0 ? r : std::min(r, n));
            if (i < 0) break;
            const QByteArray line = buf_.left(i);
            buf_.remove(0, i + 1);
            if (!line.trimmed().isEmpty()) on_input_(line);
        }
    }

    QTcpServer srv_;
    QPointer<QTcpSocket> client_;
    QByteArray buf_;
    bool lines_;
    std::function<void(const QByteArray&)> on_input_;
    std::function<void()> on_close_;
};

// tnc.py's KissServer: any number of clients; data frames to on_packet,
// other KISS commands to on_command; frames heard go to every client.
class KissServer : public QObject {
public:
    KissServer(const QHostAddress& addr, quint16 port, std::function<void(tnc::Bytes)> on_packet,
               std::function<void(int, tnc::Bytes)> on_command)
        : on_packet_(std::move(on_packet)), on_command_(std::move(on_command)) {
        if (!srv_.listen(addr, port))
            throw std::runtime_error("can't listen on " + addr.toString().toStdString() + ":" + std::to_string(port) + ": " +
                                     srv_.errorString().toStdString());
        connect(&srv_, &QTcpServer::newConnection, this, [this] {
            while (QTcpSocket* c = srv_.nextPendingConnection()) {
                const std::string peer = c->peerAddress().toString().toStdString() + ":" + std::to_string(c->peerPort());
                clients_[c] = {};
                logf(INFO, "KISS client %s connected", peer.c_str());
                connect(c, &QTcpSocket::readyRead, this, [this, c] {
                    const QByteArray d = c->readAll();
                    auto it = clients_.find(c);
                    if (it == clients_.end()) return;
                    const auto* p = reinterpret_cast<const std::uint8_t*>(d.constData());
                    for (auto& [cmd, payload] : it->second.feed({p, static_cast<std::size_t>(d.size())})) {
                        if ((cmd & 0x0F) == 0) on_packet_(std::move(payload));
                        else on_command_(cmd & 0x0F, std::move(payload));
                    }
                });
                connect(c, &QTcpSocket::disconnected, this, [this, c, peer] {
                    clients_.erase(c);
                    c->deleteLater();
                    logf(INFO, "KISS client %s disconnected", peer.c_str());
                });
            }
        });
    }

    void broadcast(const tnc::Bytes& data) {
        const auto f = tnc::kiss_encode(data);
        const QByteArray b(reinterpret_cast<const char*>(f.data()), static_cast<qsizetype>(f.size()));
        for (auto& [c, _] : clients_) c->write(b);
    }
    void close_clients() {
        for (auto& [c, _] : clients_) c->disconnectFromHost();
    }

private:
    QTcpServer srv_;
    std::map<QTcpSocket*, tnc::KissDecoder> clients_;
    std::function<void(tnc::Bytes)> on_packet_;
    std::function<void(int, tnc::Bytes)> on_command_;
};

QHostAddress resolve(const std::string& name) {
    QHostAddress a;
    if (a.setAddress(QString::fromStdString(name))) return a;
    const QHostInfo info = QHostInfo::fromName(QString::fromStdString(name));
    for (const QHostAddress& x : info.addresses())
        if (x.protocol() == QAbstractSocket::IPv4Protocol) return x;
    if (!info.addresses().isEmpty()) return info.addresses().first();
    throw std::runtime_error("can't resolve " + name);
}

// What the session stage hands the main thread after a command or a block.
struct Outbox {
    std::vector<std::string> cmd;
    arq::Bytes data;
    std::vector<arq::Bytes> kiss;
};

class Server {
public:
    explicit Server(const Args& a) : a_(a) {}

    int run();

private:
    // session stage (the worker, or the engine thread with --no-decode-worker)
    void flush();
    // main thread
    void deliver(const Outbox& o);
    void engine_loop();
    void watch_counters();

    const Args& a_;
    QObject ctx_;  // queued calls from other threads land here, on the main thread
    std::unique_ptr<kisslink::KissLink> link_;
    std::unique_ptr<arq::Engine> engine_;
    std::unique_ptr<host::Host> host_;
    std::unique_ptr<Port> cmd_, data_;
    std::unique_ptr<KissServer> kiss_;
    int rate_ = config::FS;
    std::unique_ptr<audio::CaptureFifo> cap_;
    std::unique_ptr<audio::PlaybackFifo> play_;
    std::unique_ptr<rig::RigController> rig_;
    std::unique_ptr<rig::Keyer> keyer_;
    std::atomic<bool> stop_{false};
    std::uint64_t overflows_ = 0, underruns_ = 0, late_ = 0, dropped_ = 0;  // engine thread
};

void Server::flush() {
    Outbox o;
    o.cmd.swap(host_->out_cmd);
    o.data.swap(host_->out_data);
    o.kiss.swap(engine_->kiss_rx());
    if (o.cmd.empty() && o.data.empty() && o.kiss.empty()) return;
    QMetaObject::invokeMethod(&ctx_, [this, o = std::move(o)] { deliver(o); }, Qt::QueuedConnection);
}

void Server::deliver(const Outbox& o) {
    if (cmd_)
        for (const auto& line : o.cmd) cmd_->send(QByteArray::fromStdString(line + "\r"));
    if (data_ && !o.data.empty())
        data_->send(QByteArray(reinterpret_cast<const char*>(o.data.data()), static_cast<qsizetype>(o.data.size())));
    if (kiss_)
        for (const auto& f : o.kiss) kiss_->broadcast(f);
}

void Server::watch_counters() {
    if (const auto n = cap_->overflows(); n != overflows_) {
        overflows_ = n;
        logf(WARNING, "RX audio overflow (%llu)", static_cast<unsigned long long>(n));
    }
    if (const auto n = play_->underruns(); n != underruns_) {
        underruns_ = n;
        logf(WARNING, "TX audio underrun (%llu)", static_cast<unsigned long long>(n));
    }
    if (const auto n = cap_->late_events(); n != late_) {
        late_ = n;
        logf(WARNING, "RX audio %.1f s behind the card", cap_->backlog_s());
    }
    if (const auto n = cap_->dropped(); n != dropped_) {
        logf(WARNING, "RX audio: %llu samples dropped, the capture FIFO full", static_cast<unsigned long long>(n - dropped_));
        dropped_ = n;
    }
}

void Server::engine_loop() {
    const std::size_t block = config::FS / 10;
    audio::Interpolator interp(rate_);
    const double gain = std::pow(10.0, a_.output_volume / 20);
    int slow = 0;
    try {
        while (!stop_) {
            auto x = cap_->read(block);
            if (!x) break;
            if (keyer_->keyed()) std::fill(x->begin(), x->end(), 0.0);
            const auto t0 = std::chrono::steady_clock::now();
            auto out = engine_->step(*x);
            const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (took > static_cast<double>(block) / config::FS) {
                ++slow;
                logf(DEBUG, "step took %.2f s (%d slow)", took, slow);
            }
            if (out.ptt && !keyer_->keyed()) keyer_->key();
            if (keyer_->keyed()) {
                auto y = interp(out.audio);
                for (auto& v : y) v = std::clamp(v * gain, -1.0, 1.0);
                play_->write(y);
            }
            if (keyer_->keyed() && !out.ptt) keyer_->unkey();
            watch_counters();
        }
    } catch (const std::exception& e) {
        logf(CRITICAL, "engine thread: %s", e.what());
        g_quit = true;
    }
}

int Server::run() {
    if (a_.kiss) {
        link_ = std::make_unique<kisslink::KissLink>(kiss_cap(a_.kiss_bw), a_.broadcast_mode.value_or(""));
        link_->busy_limit_s = a_.kiss_busy_limit;
    }
    arq::EngineConfig cfg;
    cfg.ptt_delay_s = a_.ptt_on_delay_ms / 1000.0;
    cfg.record_dir = a_.record_dir;
    cfg.min_header_score = a_.min_header_score;
    cfg.kiss = link_.get();
    cfg.stats_interval_s = a_.stats_interval;
    cfg.worker = a_.decode_worker;
    engine_ = std::make_unique<arq::Engine>(a_.mycall.value_or("NOCALL"), cfg);
    host_ = std::make_unique<host::Host>(*engine_, a_.buffer_credit < 0 ? std::nullopt : std::optional<int>(a_.buffer_credit));
    engine_->set_after_block([this](bool ptt) {
        host_->after_step(ptt);
        flush();
    });
    auto post = [this](std::function<void()> f) {
        engine_->post([this, f = std::move(f)] {
            f();
            flush();  // replies at once, not after the next block (clients time out at ~2 s)
        });
    };

    if (a_.vara) {
        const QHostAddress addr = resolve(a_.host);
        cmd_ = std::make_unique<Port>(
            addr, a_.command_port, true,
            [this, post](const QByteArray& line) {
                std::string text;  // ascii, as host.py decodes it (others replaced)
                for (unsigned char c : line) text += c < 128 ? static_cast<char>(c) : '?';
                logf(INFO, "command: %s", text.c_str());
                post([this, text] { host_->command(text); });
            },
            [this, post] { post([this] { host_->client_gone(); }); });
        data_ = std::make_unique<Port>(addr, a_.command_port + 1, false, [this, post](const QByteArray& d) {
            arq::Bytes b(d.begin(), d.end());
            post([this, b = std::move(b)] { host_->data_in(b); });
        });
    }
    if (a_.kiss) {
        kiss_ = std::make_unique<KissServer>(
            resolve(a_.kiss_address), a_.kiss_port,
            [this, post](tnc::Bytes f) { post([this, f = std::move(f)]() mutable { link_->enqueue(std::move(f)); }); },
            [this, post](int cmd, tnc::Bytes p) { post([this, cmd, p = std::move(p)] { link_->command(cmd, p); }); });
    }

    // PTT
    rig::Keyer::Ptt ptt;
    if (a_.rigctld_port) {
#ifdef DATA2G_HAVE_RIG
        rig_ = std::make_unique<rig::RigController>(nullptr, [](const std::string& text, bool error) {
            log_line(error ? ERROR : INFO, "rig: " + text);
        });
        rig::HamlibConfig hc;
        hc.model = rig::MODEL_NET_RIGCTL;
        hc.device = a_.rigctld_host + ":" + std::to_string(a_.rigctld_port);
        rig::RigConfig rc;
        rc.poll_interval_s = 0;  // key only, as tnc.Rigctld
        rig_->start(rig::make_hamlib_backend(hc), rc);
        ptt = rig_->ptt_function();
#else
        log_line(WARNING, "built without Hamlib: no PTT (--rigctld-port ignored)");
#endif
    }

    // audio
    const double lead_s = a_.tx_lead_ms / 1000.0;
    cap_ = std::make_unique<audio::CaptureFifo>(config::FS);
    std::unique_ptr<audio::PipeIo> pipe;
#ifdef DATA2G_HAVE_QTAUDIO
    std::unique_ptr<audio::qt::Capture> qcap;
    std::unique_ptr<audio::qt::Playback> qplay;
#endif
    const auto report = [](const std::string& s) { log_line(ERROR, s); };
    if (!a_.audio_io.empty()) {
        const auto comma = a_.audio_io.find(',');
        if (!a_.audio_io.starts_with("pipe:") || comma == std::string::npos) usage_error("--audio-io: want pipe:IN,OUT");
        play_ = std::make_unique<audio::PlaybackFifo>(config::FS, lead_s);
        const std::string in = a_.audio_io.substr(5, comma - 5), out = a_.audio_io.substr(comma + 1);
        pipe = std::make_unique<audio::PipeIo>(in, out, *cap_, *play_, report);
        logf(INFO, "audio: in %s, out %s (float32 at %d Hz)", in.c_str(), out.c_str(), config::FS);
    } else {
#ifdef DATA2G_HAVE_QTAUDIO
        rate_ = a_.sample_rate;
        play_ = std::make_unique<audio::PlaybackFifo>(rate_, lead_s);
        try {
            const auto in = audio::select_device(audio::qt::input_devices(), a_.input_device.value_or(""), "input");
            const auto out = audio::select_device(audio::qt::output_devices(), a_.output_device.value_or(""), "output");
            qcap = std::make_unique<audio::qt::Capture>(in, rate_, *cap_, report);
            qplay = std::make_unique<audio::qt::Playback>(out, rate_, *play_, report);
            logf(INFO, "audio: in %s (%d ch), out %s (%d ch) at %d Hz", qcap->device_name().c_str(), qcap->channels(),
                 qplay->device_name().c_str(), qplay->channels(), rate_);
        } catch (const std::invalid_argument& e) {
            std::fprintf(stderr, "data2g-host: %s (see --list-audio-devices)\n", e.what());
            return 1;
        }
#else
        usage_error("built without Qt Multimedia: only --audio-io pipe:IN,OUT");
#endif
    }
    keyer_ = std::make_unique<rig::Keyer>(ptt, *play_, a_.ptt_off_delay_ms / 1000.0,
                                          [](const std::string& s) { log_line(ERROR, s); });

    if (a_.vara) logf(INFO, "VARA: commands on %s:%d, data on %d", a_.host.c_str(), a_.command_port, a_.command_port + 1);
    if (a_.kiss)
        logf(INFO, "KISS on %s:%d: %d Hz cap, broadcasts in %s", a_.kiss_address.c_str(), a_.kiss_port, a_.kiss_bw,
             link_->broadcast.c_str());
    logf(INFO, "recording to %s", a_.record_dir.empty() ? "(off)" : a_.record_dir.c_str());
    logf(INFO, "decode worker %s", a_.decode_worker ? "on" : "off");

    std::thread engine_thread(&Server::engine_loop, this);
    QTimer quit_check;
    QObject::connect(&quit_check, &QTimer::timeout, [] {
        if (g_quit) QCoreApplication::quit();
    });
    quit_check.start(100);
    QCoreApplication::exec();

    log_line(INFO, "shutting down");
    stop_ = true;
    cap_->close();
    engine_thread.join();
    engine_->stop();
    keyer_.reset();  // PTT off
    if (rig_) {
        rig_->stop();
        rig_->wait_for_shutdown(2.0);
    }
    if (kiss_) kiss_->close_clients();
    if (pipe) pipe->stop();
#ifdef DATA2G_HAVE_QTAUDIO
    if (qcap) qcap->stop();
    if (qplay) qplay->stop();
#endif
    return 0;
}

void list_modes(int kiss_bw) {
    auto ms = arq::allowed(kiss_cap(kiss_bw));
    std::sort(ms.begin(), ms.end(), [](const arq::Mode* x, const arq::Mode* y) {
        const double wx = arq::width_hz(*x), wy = arq::width_hz(*y);
        return wx != wy ? wx < wy : x->name < y->name;
    });
    for (const auto* m : ms)
        std::printf("%-18s %5.0f Hz  %4d bytes/codeword\n", std::string(m->name).c_str(), arq::width_hz(*m), arq::payload_bytes(*m));
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    if (a.list_modes) {
        list_modes(a.kiss_bw);
        return 0;
    }
    if (!a.vara && !a.kiss) usage_error("nothing to serve: --no-vara and --no-kiss");
    if (a.kiss && a.broadcast_mode) {
        const auto ok = arq::allowed(kiss_cap(a.kiss_bw));
        const auto* m = arq::mode(*a.broadcast_mode);
        if (!m || std::find(ok.begin(), ok.end(), m) == ok.end())
            usage_error("--broadcast-mode " + *a.broadcast_mode + ": not a mode within " + std::to_string(a.kiss_bw) + " Hz");
    }
    g_level = parse_level(a.log_level);
    arq::set_log_sink({[](const char*, int level) { return level >= g_level; },
                       [](const char*, int level, const std::string& msg) { log_line(level, msg); }});

    QCoreApplication app(argc, argv);
    if (a.list_audio_devices) {
#ifdef DATA2G_HAVE_QTAUDIO
        for (const auto& [what, devs] : {std::pair{"input", audio::qt::input_devices()}, {"output", audio::qt::output_devices()}}) {
            std::printf("%s:\n", what);
            for (std::size_t i = 0; i < devs.size(); ++i) std::printf("%3zu  %2d ch  %s\n", i, devs[i].channels, devs[i].name.c_str());
        }
#else
        std::printf("built without Qt Multimedia: no sound cards (use --audio-io)\n");
#endif
        return 0;
    }
    if (a.audio_io.empty() && a.sample_rate % config::FS)
        usage_error("--sample-rate must be a multiple of " + std::to_string(config::FS));
#ifndef _WIN32
    std::signal(SIGPIPE, SIG_IGN);  // a pipe or socket whose reader went: an error, not death
#endif
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    try {
        Server s(a);
        return s.run();
    } catch (const std::exception& e) {
        logf(CRITICAL, "%s", e.what());
        return 1;
    }
}
