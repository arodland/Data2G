#include "tnc.hpp"

#include <QCoreApplication>
#include <QJniEnvironment>
#include <QJniObject>
#include <QPermissions>
#include <QSettings>
#include <QtCore/qcoreapplication_platform.h>

#include <algorithm>
#include <cmath>

#include "app/settings.hpp"
#include "audio/card.hpp"
#include "modem.hpp"

using data2g::androidapp::Modem;
using data2g::androidapp::State;

namespace {

constexpr int MAX_LOG_LINES = 500;
const QString SYSTEM_DEFAULT = QStringLiteral("System default");

QString qs(const std::string& s) { return QString::fromStdString(s); }

void call_static(const char* cls, const char* method) {
    QJniObject ctx = QNativeInterface::QAndroidApplication::context();
    if (!ctx.isValid()) return;
    QJniObject::callStaticMethod<void>(cls, method, "(Landroid/content/Context;)V", ctx.object());
    QJniEnvironment().checkAndClearExceptions();
}

// Edits the saved settings in one place, so a field shown and a field saved
// can't be spelled two ways.
template <typename F>
void edit(F f) {
    data2g::app::Args a = data2g::androidapp::load_args();
    f(a);
    QSettings s;
    data2g::app::save_settings(s, a);
}

QStringList with_default(const std::vector<data2g::audio::DeviceInfo>& devices) {
    QStringList out{SYSTEM_DEFAULT};
    for (const auto& d : devices) out << qs(d.name);
    return out;
}

}  // namespace

Tnc::Tnc(QObject* parent) : QObject(parent) {
    refreshDevices();
    poll_.setInterval(250);
    connect(&poll_, &QTimer::timeout, this, [this] {
        std::uint64_t newest = 0;
        const auto lines = Modem::instance().log_after(log_seen_, &newest);
        log_seen_ = newest;
        if (!lines.empty()) {
            for (const auto& l : lines) log_ << qs(l);
            if (log_.size() > MAX_LOG_LINES) log_.remove(0, log_.size() - MAX_LOG_LINES);
            Q_EMIT logChanged();
        }
        Q_EMIT changed();
    });
    poll_.start();
}

bool Tnc::running() const { return Modem::instance().state() != State::Stopped; }
bool Tnc::starting() const { return Modem::instance().state() == State::Starting; }
QString Tnc::status() const { return qs(Modem::instance().status_line()); }
bool Tnc::ptt() const {
    auto* s = Modem::instance().station();
    return s && s->ptt();
}
bool Tnc::busy() const {
    auto* s = Modem::instance().station();
    return s && s->busy();
}
QString Tnc::frequency() const {
    auto* s = Modem::instance().station();
    const auto hz = s ? s->rig_frequency() : std::nullopt;
    return hz ? QStringLiteral("%1 MHz").arg(*hz / 1e6, 0, 'f', 4) : QString();
}
QString Tnc::counters() const {
    auto* s = Modem::instance().station();
    if (!s) return {};
    const auto c = s->counters();
    if (!(c.overflows || c.dropped || c.late || c.underruns || c.decode_dropped)) return {};
    return tr("Audio trouble: %1 overflows, %2 dropped, %3 late, %4 underruns")
        .arg(c.overflows)
        .arg(c.dropped + c.decode_dropped)
        .arg(c.late)
        .arg(c.underruns);
}

// --- settings

QString Tnc::callsign() const { return qs(data2g::androidapp::load_args().mycall.value_or("")); }
void Tnc::setCallsign(const QString& c) {
    const QString call = c.trimmed().toUpper();
    edit([&](auto& a) { a.mycall = call.isEmpty() ? std::nullopt : std::optional(call.toStdString()); });
    Q_EMIT settingsChanged();
}

QString Tnc::inputDevice() const { return qs(data2g::androidapp::load_args().input_device.value_or(SYSTEM_DEFAULT.toStdString())); }
void Tnc::setInputDevice(const QString& d) {
    edit([&](auto& a) { a.input_device = d == SYSTEM_DEFAULT ? std::nullopt : std::optional(d.toStdString()); });
    Q_EMIT settingsChanged();
}
QString Tnc::outputDevice() const { return qs(data2g::androidapp::load_args().output_device.value_or(SYSTEM_DEFAULT.toStdString())); }
void Tnc::setOutputDevice(const QString& d) {
    edit([&](auto& a) { a.output_device = d == SYSTEM_DEFAULT ? std::nullopt : std::optional(d.toStdString()); });
    Q_EMIT settingsChanged();
}

double Tnc::txLevelDb() const { return data2g::androidapp::load_args().output_volume; }
void Tnc::setTxLevelDb(double db) {
    edit([&](auto& a) { a.output_volume = std::clamp(std::round(db), -30.0, 0.0); });
    Q_EMIT settingsChanged();
}

int Tnc::kissPort() const { return data2g::androidapp::load_args().kiss_port; }
int Tnc::varaPort() const { return data2g::androidapp::load_args().command_port; }

void Tnc::refreshDevices() {
    try {
        inputs_ = with_default(data2g::audio::card::input_devices());
        outputs_ = with_default(data2g::audio::card::output_devices());
    } catch (const std::exception& e) {
        Q_EMIT error(qs(e.what()));
    }
    Q_EMIT devicesChanged();
}

void Tnc::clearLog() {
    Modem::instance().clear_log();
    log_.clear();
    Q_EMIT logChanged();
}

// --- start and stop

void Tnc::start() {
    const QMicrophonePermission mic;
    switch (qApp->checkPermission(mic)) {
        case Qt::PermissionStatus::Granted:
            start_service();
            return;
        case Qt::PermissionStatus::Undetermined:
            qApp->requestPermission(mic, this, [this](const QPermission& p) {
                if (p.status() == Qt::PermissionStatus::Granted) start_service();
                else Q_EMIT error(tr("The modem needs microphone access to hear the radio."));
            });
            return;
        case Qt::PermissionStatus::Denied:
            // Android stops asking after a denial: say where to fix it.
            Q_EMIT error(tr("Microphone access was denied. Allow it in Android's settings for this app."));
            return;
    }
}

void Tnc::start_service() {
    // Not awaited: without it the modem runs, only the notification is missing.
    call_static("org/cleverdomain/data2g/Permissions", "requestNotifications");
    call_static("org/cleverdomain/data2g/TncService", "startModem");
}

void Tnc::stop() { call_static("org/cleverdomain/data2g/TncService", "stopModem"); }

void Tnc::moveToBackground() { call_static("org/cleverdomain/data2g/Background", "moveToBack"); }
