#include "modem.hpp"

#include <QCoreApplication>
#include <QDateTime>
#include <QSettings>
#include <QtGlobal>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <future>
#include <memory>

#include "app/log.hpp"
#include "app/settings.hpp"
#include "arq/modes.hpp"
#include "arq/session.hpp"
#ifdef DATA2G_HAVE_RIG
#include "rig/android/androidrig.hpp"
#include "rig/bridged.hpp"
#include "rig/hamlib/hamlib.hpp"
#endif

namespace data2g::androidapp {

namespace {

constexpr std::size_t MAX_LOG_LINES = 1000;

std::string clock_text(std::chrono::system_clock::time_point t) {
    const std::time_t tt = std::chrono::system_clock::to_time_t(t);
    const std::tm tm = app::local_tm(tt);
    char buf[16];
    std::strftime(buf, sizeof buf, "%H:%M:%S", &tm);
    return buf;
}

std::string burst_line(const app::BurstLogEntry& b) {
    char snr[24] = "";
    if (b.snr_db) std::snprintf(snr, sizeof snr, ", SNR %.1f dB", *b.snr_db);
    const std::string result = b.lost ? "lost"
                               : b.decoded == 0 ? "undecoded"
                               : b.failed ? "incomplete (" + std::to_string(b.n_cw - b.failed) + "/" + std::to_string(b.n_cw) + ")"
                                          : "ok";
    return clock_text(b.when) + " heard " + b.mode + " x" + std::to_string(b.n_cw) + snr + ": " + result;
}

std::string link_text(const app::LinkStatus& l) {
    using arq::SessionState;
    switch (l.state) {
        case SessionState::CONNECTING: return "Connecting to " + l.peer;
        case SessionState::CONNECTED: return "Connected to " + l.peer + " (BW" + arq::bw_name(std::clamp(l.cap, 0, 2)) + ")";
        case SessionState::DISCONNECTING: return "Disconnecting from " + l.peer;
        default: return "Listening";
    }
}

}  // namespace

app::Args load_args() {
    app::Args base;
    base.rig = false;       // a phone acoustically coupled, or VOX, has no rig to open
    base.realtime = false;  // RealtimeKit is Linux desktop; the Java audio threads ask Android instead
    QSettings s;
    return app::load_settings(s, base);
}

std::string rig_connection() {
    return QSettings().value(QLatin1String(RIG_CONNECTION_KEY), QStringLiteral("usb")).toString().toStdString();
}

Modem& Modem::instance() {
    // Never destroyed: main() ends with _Exit (see there), and a static
    // destructor racing the engine threads is what that avoids.
    static Modem* m = new Modem;
    return *m;
}

Modem::Modem() {
    app::g_level = app::INFO;
    app::install_arq_log();
    app::set_log_sink([this](int level, const std::string& msg) {
        qInfo("%s", msg.c_str());  // logcat
        const char* tag = level >= app::ERROR ? " ERROR: " : level >= app::WARNING ? " WARNING: " : " ";
        add_log(clock_text(std::chrono::system_clock::now()) + tag + msg);
    });
    QObject::connect(&poll_, &QTimer::timeout, [this] { poll(); });
    poll_.start(500);
}

void Modem::post_start() {
    if (state_.exchange(State::Starting) == State::Running) state_ = State::Running;
    set_status("Starting…");
    QMetaObject::invokeMethod(QCoreApplication::instance(), [this] { start(); }, Qt::QueuedConnection);
}

void Modem::post_stop() {
    QMetaObject::invokeMethod(QCoreApplication::instance(), [this] { stop(); }, Qt::QueuedConnection);
}

void Modem::stop_now() {
    // Bounded, not BlockingQueuedConnection: the caller is Android's main thread,
    // and an app that hangs there is killed as not responding.
    auto done = std::make_shared<std::promise<void>>();
    auto finished = done->get_future();
    QMetaObject::invokeMethod(
        QCoreApplication::instance(),
        [this, done] {
            stop();
            done->set_value();
        },
        Qt::QueuedConnection);
    if (finished.wait_for(std::chrono::seconds(3)) != std::future_status::ready) qWarning("modem stop timed out");
}

void Modem::start() {
    if (station_ && station_->running()) {
        state_ = State::Running;
        return;
    }
    station_.reset();
    const auto fail = [this](const std::string& why) {
        {
            std::lock_guard lock(mu_);
            error_ = why;
        }
        app::log_line(app::ERROR, "not started: " + why);
        station_.reset();
        state_ = State::Stopped;
        poll();
    };
    {
        std::lock_guard lock(mu_);
        error_.clear();
    }
    const app::Args a = load_args();
    if (!a.mycall) return fail("set your callsign in Settings first");
    if (const auto bad = app::check(a)) return fail(*bad);
    station_ = std::make_unique<app::Station>(a);
#ifdef DATA2G_HAVE_RIG
    // USB and Bluetooth radios through the loopback bridge (rig/bridged.hpp);
    // a network one is Hamlib's own host:port. Throws from start() for a
    // configuration that can't work (resolve_serial_params' line conflicts).
    station_->set_rig_factory([conn = rig_connection()](const rig::HamlibConfig& hc) -> std::unique_ptr<rig::RigBackend> {
        if (conn == "network") return rig::make_hamlib_backend(hc);
        const rig::SerialParams params = rig::resolve_serial_params(hc, rig::serial_defaults(hc.model));
        return rig::make_bridged_backend(hc, rig::android::make_transport(params), rig::make_hamlib_backend);
    });
#endif
    try {
        station_->start();
    } catch (const std::exception& e) {
        return fail(e.what());
    }
    state_ = State::Running;
    poll();
}

void Modem::stop() {
    if (station_) {
        station_->stop();  // PTT down, ports closed, audio released
        station_.reset();
        app::log_line(app::INFO, "modem stopped");
    }
    state_ = State::Stopped;
    poll();
}

void Modem::poll() {
    if (station_ && station_->running() && station_->failed()) {
        {
            std::lock_guard lock(mu_);
            error_ = "the modem stopped on an error (see the log)";
        }
        stop();
        return;
    }
    if (!station_ || !station_->running()) {
        if (state_ != State::Starting) set_status(error().empty() ? "Stopped" : "Stopped: " + error());
        return;
    }
    for (const auto& b : station_->take_bursts()) add_log(burst_line(b));
    const app::Args& a = station_->args();
    // ponytail: PTT sampled at the poll (0.5 s), so a shorter key-down goes unlogged; take
    // the Transmitter's announce_ptt edges if that matters.
    const bool ptt = station_->ptt();
    if (ptt && !was_ptt_) add_log(clock_text(std::chrono::system_clock::now()) + " transmitting");
    was_ptt_ = ptt;
    std::string s = ptt ? "Transmitting" : link_text(station_->link());
    s += " · KISS " + std::to_string(a.kiss_port) + ", VARA " + std::to_string(a.command_port);
    set_status(std::move(s));
}

std::string Modem::status_line() const {
    std::lock_guard lock(mu_);
    return status_;
}

std::string Modem::error() const {
    std::lock_guard lock(mu_);
    return error_;
}

void Modem::set_status(std::string s) {
    std::lock_guard lock(mu_);
    status_ = std::move(s);
}

void Modem::add_log(std::string line) {
    std::lock_guard lock(mu_);
    log_.push_back(std::move(line));
    ++log_seq_;
    while (log_.size() > MAX_LOG_LINES) log_.pop_front();
}

std::vector<std::string> Modem::log_after(std::uint64_t after, std::uint64_t* newest) const {
    std::lock_guard lock(mu_);
    *newest = log_seq_;
    const std::uint64_t first = log_seq_ - log_.size() + 1;  // sequence number of log_.front()
    std::vector<std::string> out;
    for (std::uint64_t n = std::max(after + 1, first); n <= log_seq_; ++n) out.push_back(log_[n - first]);
    return out;
}

void Modem::clear_log() {
    std::lock_guard lock(mu_);
    log_.clear();
}

}  // namespace data2g::androidapp
