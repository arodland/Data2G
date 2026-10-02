#include "station.hpp"

#include <QHostAddress>
#include <QHostInfo>
#include <QPointer>
#include <QTcpServer>
#include <QTcpSocket>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <functional>
#include <map>
#include <cstdlib>

#include "arq/engine.hpp"
#include "arq/modes.hpp"
#include "arq/policy.hpp"
#include "audio/audio.hpp"
#include "audio/fifo.hpp"
#include "audio/filters.hpp"
#include "audio/pipe.hpp"
#include "util/pool.hpp"
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

namespace data2g::app {

// --- logging -----------------------------------------------------------------------------

std::atomic<int> g_level{INFO};

namespace {

std::mutex g_log_mu;

// localtime_r is POSIX; MSVC has localtime_s with the arguments swapped.
std::tm local_tm(std::time_t t) {
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    return tm;
}

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

}  // namespace

void log_line(int level, const std::string& msg) {
    if (level < g_level) return;
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    const std::tm tm = local_tm(t);
    char stamp[32];
    std::strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", &tm);
    std::lock_guard lock(g_log_mu);
    std::fprintf(stderr, "%s,%03d %s %s\n", stamp, static_cast<int>(ms), level_name(level), msg.c_str());
}

void logf(int level, const char* fmt, ...) {
    if (level < g_level) return;
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    log_line(level, buf);
}

std::optional<int> parse_level(const std::string& s) {
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
    return std::nullopt;
}

void install_arq_log() {
    arq::set_log_sink({[](const char*, int level) { return level >= g_level; },
                       [](const char*, int level, const std::string& msg) { log_line(level, msg); }});
}

// --- the options -------------------------------------------------------------------------

namespace {

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
    "                   [--decode-worker | --no-decode-worker] [--audio-io pipe:IN,OUT] [--threads N]\n";

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
    "                        connect through two mkfifo pipes. --sample-rate and the devices are then unused.\n"
    "  --threads N           threads for decode and sync, the calling one included (default: min(4, cores / 2))\n";

template <typename T>
T number(const std::string& opt, const std::string& v, const char* prog) {
    try {
        std::size_t used = 0;
        T out;
        if constexpr (std::is_same_v<T, int>) out = std::stoi(v, &used);
        else out = std::stod(v, &used);
        if (used == v.size()) return out;
    } catch (const std::exception&) {
    }
    usage_error("argument " + opt + ": invalid " + (std::is_same_v<T, int> ? "int" : "float") + " value: '" + v + "'", prog);
}

}  // namespace

void usage_error(const std::string& msg, const char* prog) {
    std::fprintf(stderr, "%s%s: error: %s\n", USAGE, prog, msg.c_str());
    std::exit(2);
}

std::string default_record_dir() {
    const std::time_t t = std::time(nullptr);
    const std::tm tm = local_tm(t);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", &tm);
    return std::string("recordings/") + buf;
}

Args parse(int argc, char** argv, Args a, const char* prog) {
    std::map<std::string, bool*> flags = {
        {"--list-audio-devices", &a.list_audio_devices}, {"--list-modes", &a.list_modes}};
    std::map<std::string, std::pair<bool*, bool>> switches = {
        {"--vara", {&a.vara, true}},   {"--no-vara", {&a.vara, false}},
        {"--kiss", {&a.kiss, true}},   {"--no-kiss", {&a.kiss, false}},
        {"--decode-worker", {&a.decode_worker, true}}, {"--no-decode-worker", {&a.decode_worker, false}}};
    const auto i_ = [prog](auto& o, auto& v) { return number<int>(o, v, prog); };
    const auto d_ = [prog](auto& o, auto& v) { return number<double>(o, v, prog); };
    std::map<std::string, std::function<void(const std::string&, const std::string&)>> valued = {
        {"--kiss-port", [&](auto& o, auto& v) { a.kiss_port = i_(o, v); }},
        {"--kiss-address", [&](auto&, auto& v) { a.kiss_address = v; }},
        {"--kiss-busy-limit", [&](auto& o, auto& v) { a.kiss_busy_limit = d_(o, v); }},
        {"--kiss-bw", [&](auto& o, auto& v) {
             a.kiss_bw = i_(o, v);
             if (a.kiss_bw != 2400 && a.kiss_bw != 500)
                 usage_error("argument --kiss-bw: invalid choice: " + v + " (choose from 2400, 500)", prog);
         }},
        {"--broadcast-mode", [&](auto&, auto& v) { a.broadcast_mode = v; }},
        {"--mycall", [&](auto&, auto& v) { a.mycall = v; }},
        {"--host", [&](auto&, auto& v) { a.host = v; }},
        {"--command-port", [&](auto& o, auto& v) { a.command_port = i_(o, v); }},
        {"--input-device", [&](auto&, auto& v) { a.input_device = v; }},
        {"--output-device", [&](auto&, auto& v) { a.output_device = v; }},
        {"--sample-rate", [&](auto& o, auto& v) { a.sample_rate = i_(o, v); }},
        {"--output-volume", [&](auto& o, auto& v) { a.output_volume = d_(o, v); }},
        {"--rigctld-host", [&](auto&, auto& v) { a.rigctld_host = v; }},
        {"--rigctld-port", [&](auto& o, auto& v) { a.rigctld_port = i_(o, v); }},
        {"--ptt-on-delay-ms", [&](auto& o, auto& v) { a.ptt_on_delay_ms = i_(o, v); }},
        {"--ptt-off-delay-ms", [&](auto& o, auto& v) { a.ptt_off_delay_ms = i_(o, v); }},
        {"--tx-lead-ms", [&](auto& o, auto& v) { a.tx_lead_ms = i_(o, v); }},
        {"--min-header-score", [&](auto& o, auto& v) { a.min_header_score = d_(o, v); }},
        {"--buffer-credit", [&](auto& o, auto& v) { a.buffer_credit = i_(o, v); }},
        {"--record-dir", [&](auto&, auto& v) { a.record_dir = v; }},
        {"--log-level", [&](auto&, auto& v) { a.log_level = v; }},
        {"--stats-interval", [&](auto& o, auto& v) { a.stats_interval = d_(o, v); }},
        {"--audio-io", [&](auto&, auto& v) { a.audio_io = v; }},
        {"--threads", [&](auto& o, auto& v) { pool::set_threads(i_(o, v)); }},
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
            if (!inline_value && i + 1 >= argc) usage_error("argument " + arg + ": expected one argument", prog);
            v->second(arg, inline_value ? *inline_value : std::string(argv[++i]));
        } else {
            usage_error("unrecognized arguments: " + std::string(argv[i]), prog);
        }
    }
    return a;
}

int kiss_cap(int hz) { return hz == 500 ? 0 : 2; }

std::optional<std::string> check(const Args& a) {
    if (!a.vara && !a.kiss) return "nothing to serve: --no-vara and --no-kiss";
    if (a.kiss && a.broadcast_mode) {
        const auto ok = arq::allowed(kiss_cap(a.kiss_bw));
        const auto* m = arq::mode(*a.broadcast_mode);
        if (!m || std::find(ok.begin(), ok.end(), m) == ok.end())
            return "--broadcast-mode " + *a.broadcast_mode + ": not a mode within " + std::to_string(a.kiss_bw) + " Hz";
    }
    return std::nullopt;
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

// --- the servers -------------------------------------------------------------------------

// One TCP listener holding at most one client (host.py's _Port): a new
// client replaces the old one, whose close then isn't a loss.
struct Station::Port : QObject {
    Port(const QHostAddress& addr, quint16 port, bool lines, std::function<void(const QByteArray&)> on_input,
         std::function<void()> on_close = {})
        : lines_(lines), on_input_(std::move(on_input)), on_close_(std::move(on_close)) {
        if (!srv_.listen(addr, port))
            throw std::runtime_error("can't listen on " + addr.toString().toStdString() + ":" + std::to_string(port) + ": " +
                                     srv_.errorString().toStdString());
        connect(&srv_, &QTcpServer::newConnection, this, [this] { accept(); });
    }
    // Its sockets are srv_'s children, closed after the members their
    // handlers use are gone: unhook them first.
    ~Port() override {
        for (auto* c : srv_.findChildren<QTcpSocket*>()) c->disconnect(this);
    }

    void send(const QByteArray& data) {
        if (client_ && !data.isEmpty()) client_->write(data);
    }

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
struct Station::KissServer : QObject {
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

    ~KissServer() override {  // as ~Port
        for (auto* c : srv_.findChildren<QTcpSocket*>()) c->disconnect(this);
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

#ifdef DATA2G_HAVE_QTAUDIO
struct Station::SoundCard {
    std::unique_ptr<audio::qt::Capture> cap;
    std::unique_ptr<audio::qt::Playback> play;
};
#else
struct Station::SoundCard {};
#endif

namespace {

QHostAddress resolve(const std::string& name) {
    QHostAddress a;
    if (a.setAddress(QString::fromStdString(name))) return a;
    const QHostInfo info = QHostInfo::fromName(QString::fromStdString(name));
    for (const QHostAddress& x : info.addresses())
        if (x.protocol() == QAbstractSocket::IPv4Protocol) return x;
    if (!info.addresses().isEmpty()) return info.addresses().first();
    throw std::runtime_error("can't resolve " + name);
}

constexpr std::size_t MAX_BURSTS = 256;  // kept for a front end that polls slowly (or never: data2g-host)
constexpr std::size_t TAP_SAMPLES = 4096;

}  // namespace

// What the session stage hands the owner after a command or a block.
struct Station::Outbox {
    std::vector<std::string> cmd;
    arq::Bytes data;
    std::vector<arq::Bytes> kiss;
};

// --- the station -------------------------------------------------------------------------

Station::Station(Args a) : a_(std::move(a)), rate_(config::FS), tap_(TAP_SAMPLES, 0.0) {}

Station::~Station() { stop(); }

void Station::flush() {
    Outbox o;
    o.cmd.swap(host_->out_cmd);
    o.data.swap(host_->out_data);
    o.kiss.swap(engine_->kiss_rx());
    if (o.cmd.empty() && o.data.empty() && o.kiss.empty()) return;
    QMetaObject::invokeMethod(&ctx_, [this, o = std::move(o)] { deliver(o); }, Qt::QueuedConnection);
}

void Station::note_link() {
    auto& s = engine_->session();
    LinkStatus l;
    l.state = arq::state_name(s.state);
    l.peer = s.peer;
    l.cap = s.cap;
    if (s.station) {
        const auto st = s.stats();
        l.tx_bytes = st.at("tx_bytes");
        l.rx_bytes = st.at("rx_bytes");
    }
    if (const auto& tx = engine_->tx()) mode_ = tx->burst->submode;
    l.mode = mode_;
    std::lock_guard lock(status_mu_);
    link_status_ = std::move(l);
}

void Station::deliver(const Outbox& o) {
    if (cmd_)
        for (const auto& line : o.cmd) cmd_->send(QByteArray::fromStdString(line + "\r"));
    if (data_ && !o.data.empty())
        data_->send(QByteArray(reinterpret_cast<const char*>(o.data.data()), static_cast<qsizetype>(o.data.size())));
    if (kiss_)
        for (const auto& f : o.kiss) kiss_->broadcast(f);
}

void Station::watch_counters() {
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

void Station::tap(const std::vector<double>& x) {
    std::lock_guard lock(tap_mu_);
    for (double v : x) tap_[tapped_++ % tap_.size()] = v;
}

std::vector<double> Station::input_tail(std::size_t n, std::uint64_t* total) const {
    std::lock_guard lock(tap_mu_);
    n = std::min({n, tap_.size(), static_cast<std::size_t>(tapped_)});
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = tap_[(tapped_ - n + i) % tap_.size()];
    if (total) *total = tapped_;
    return out;
}

LinkStatus Station::link() const {
    std::lock_guard lock(status_mu_);
    return link_status_;
}

std::vector<BurstLogEntry> Station::take_bursts() {
    std::lock_guard lock(status_mu_);
    std::vector<BurstLogEntry> out(std::make_move_iterator(bursts_.begin()), std::make_move_iterator(bursts_.end()));
    bursts_.clear();
    return out;
}

bool Station::busy() const { return engine_ && engine_->channel_busy(); }

AudioCounters Station::counters() const {
    if (!cap_ || !play_) return {};
    return {cap_->overflows(), play_->underruns(), cap_->late_events(), cap_->dropped(), cap_->backlog_s(),
            engine_ ? engine_->decode_dropped() : 0};
}

void Station::engine_loop() {
    const std::size_t block = config::FS / 10;
    audio::Interpolator interp(rate_);
    const double gain = std::pow(10.0, a_.output_volume / 20);
    int slow = 0;
    try {
        while (!stop_) {
            auto x = cap_->read(block);
            if (!x) break;
            tap(*x);
            if (keyer_->keyed()) std::fill(x->begin(), x->end(), 0.0);
            const auto t0 = std::chrono::steady_clock::now();
            auto out = engine_->step(*x);
            const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (took > static_cast<double>(block) / config::FS) {
                ++slow;
                logf(DEBUG, "step took %.2f s (%d slow)", took, slow);
            }
            if (out.ptt && !keyer_->keyed()) keyer_->key();
            ptt_ = keyer_->keyed();
            if (keyer_->keyed()) {
                auto y = interp(out.audio);
                for (auto& v : y) v = std::clamp(v * gain, -1.0, 1.0);
                play_->write(y);
            }
            if (keyer_->keyed() && !out.ptt) keyer_->unkey();
            ptt_ = keyer_->keyed();
            watch_counters();
        }
    } catch (const std::exception& e) {
        logf(CRITICAL, "engine thread: %s", e.what());
        failed_ = true;
    }
}

void Station::start() {
    if (running_) return;
    try {
        stop_ = false;
        failed_ = false;
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
            note_link();
            flush();
        });
        engine_->set_on_burst([this](const arq::BurstHeard& b) {
            mode_ = b.submode;
            std::lock_guard lock(status_mu_);
            bursts_.push_back({std::chrono::system_clock::now(), b.submode, b.n_cw, b.lost, b.snr_db});
            if (bursts_.size() > MAX_BURSTS) bursts_.pop_front();
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
                [post, this] { post([this] { host_->client_gone(); }); });
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
        const auto report = [](const std::string& s) { log_line(ERROR, s); };
        if (!a_.audio_io.empty()) {
            const auto comma = a_.audio_io.find(',');
            if (!a_.audio_io.starts_with("pipe:") || comma == std::string::npos) throw UsageError("--audio-io: want pipe:IN,OUT");
            play_ = std::make_unique<audio::PlaybackFifo>(config::FS, lead_s);
            const std::string in = a_.audio_io.substr(5, comma - 5), out = a_.audio_io.substr(comma + 1);
            pipe_ = std::make_unique<audio::PipeIo>(in, out, *cap_, *play_, report);
            logf(INFO, "audio: in %s, out %s (float32 at %d Hz)", in.c_str(), out.c_str(), config::FS);
        } else {
#ifdef DATA2G_HAVE_QTAUDIO
            rate_ = a_.sample_rate;
            if (rate_ <= 0 || rate_ % config::FS)
                throw UsageError("--sample-rate must be a multiple of " + std::to_string(config::FS));
            play_ = std::make_unique<audio::PlaybackFifo>(rate_, lead_s);
            const auto in = audio::select_device(audio::qt::input_devices(), a_.input_device.value_or(""), "input");
            const auto out = audio::select_device(audio::qt::output_devices(), a_.output_device.value_or(""), "output");
            card_ = std::make_unique<SoundCard>();
            card_->cap = std::make_unique<audio::qt::Capture>(in, rate_, *cap_, report);
            card_->play = std::make_unique<audio::qt::Playback>(out, rate_, *play_, report);
            logf(INFO, "audio: in %s (%d ch), out %s (%d ch) at %d Hz", card_->cap->device_name().c_str(), card_->cap->channels(),
                 card_->play->device_name().c_str(), card_->play->channels(), rate_);
#else
            throw UsageError("built without Qt Multimedia: only --audio-io pipe:IN,OUT");
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

        running_ = true;
        engine_thread_ = std::thread(&Station::engine_loop, this);
    } catch (...) {
        running_ = true;  // so stop() undoes whatever got started
        stop();
        throw;
    }
}

void Station::stop() {
    if (!running_) return;
    running_ = false;
    stop_ = true;
    if (cap_) cap_->close();
    if (engine_thread_.joinable()) engine_thread_.join();
    if (engine_) engine_->stop();
    keyer_.reset();  // PTT off
    ptt_ = false;
    if (rig_) {
        rig_->stop();
        rig_->wait_for_shutdown(2.0);
    }
    if (kiss_) kiss_->close_clients();
    if (pipe_) pipe_->stop();
#ifdef DATA2G_HAVE_QTAUDIO
    if (card_) {
        if (card_->cap) card_->cap->stop();
        if (card_->play) card_->play->stop();
    }
#endif
    // released in reverse order of use; the ports close here, so a restart can listen again
    cmd_.reset();
    data_.reset();
    kiss_.reset();
    card_.reset();
    pipe_.reset();
    rig_.reset();
    host_.reset();
    engine_.reset();
    link_.reset();
    play_.reset();
    cap_.reset();
    rate_ = config::FS;
    overflows_ = underruns_ = late_ = dropped_ = 0;
}

}  // namespace data2g::app
