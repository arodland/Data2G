#include "rigsettings.hpp"

#include <QCoreApplication>
#include <QPointer>
#include <QSettings>
#include <QVariantMap>

#include <string>
#include <vector>

#include "app/settings.hpp"
#include "modem.hpp"
#include "rig/android/androidrig.hpp"
#include "rig/config.hpp"
#ifdef DATA2G_HAVE_RIG
#include "rig/hamlib/hamlib.hpp"
#endif

namespace rig = data2g::rig;
using data2g::androidapp::load_args;

namespace {

constexpr int MAX_MODEL_ROWS = 60;

template <typename F>
void edit(F f) {
    data2g::app::Args a = load_args();
    f(a);
    QSettings s;
    data2g::app::save_settings(s, a);
}

// rig::list_models() walks every backend: once.
const std::vector<rig::RigModel>& all_models() {
#ifdef DATA2G_HAVE_RIG
    static const std::vector<rig::RigModel> models = rig::list_models();
#else
    static const std::vector<rig::RigModel> models;
#endif
    return models;
}

QVariantMap model_row(const rig::RigModel& m) {
    return {{QStringLiteral("number"), m.model}, {QStringLiteral("label"), QString::fromStdString(m.label())}};
}

}  // namespace

RigSettings::RigSettings(QObject* parent) : QObject(parent) {
    // A grant arrives on Android's broadcast thread; re-list on ours. Guarded:
    // a rotation can destroy this view with the call in flight.
    QPointer<RigSettings> self(this);
    rig::android::set_permission_callback([self](const std::string&, bool) {
        QMetaObject::invokeMethod(
            qApp, [self] {
                if (self) self->refreshDevices();
            },
            Qt::QueuedConnection);
    });
    refreshDevices();
}

RigSettings::~RigSettings() { rig::android::set_permission_callback({}); }

bool RigSettings::enabled() const { return load_args().rig; }
void RigSettings::setEnabled(bool on) {
    edit([&](auto& a) { a.rig = on; });
    Q_EMIT changed();
}

QString RigSettings::connection() const { return QString::fromStdString(data2g::androidapp::rig_connection()); }
void RigSettings::setConnection(const QString& kind) {
    if (kind == connection()) return;
    QSettings().setValue(QLatin1String(data2g::androidapp::RIG_CONNECTION_KEY), kind);
    // A device belongs to its kind: a Bluetooth address means nothing to the USB opener.
    edit([&](auto& a) {
        a.rig_device.clear();
        if (kind == QLatin1String("bluetooth") && (a.ptt_method == "dtr" || a.ptt_method == "rts")) a.ptt_method = "cat";
    });
    error_.clear();
    refreshDevices();
    Q_EMIT changed();
}

int RigSettings::model() const { return load_args().rig_model; }
void RigSettings::setModel(int m) {
    edit([&](auto& a) { a.rig_model = m; });
    Q_EMIT changed();
}
QString RigSettings::modelLabel() const {
    const int m = model();
    for (const auto& r : all_models())
        if (r.model == m) return QString::fromStdString(r.label());
    return tr("model %1").arg(m);
}

QString RigSettings::device() const { return connection() == QLatin1String("network") ? QString() : QString::fromStdString(load_args().rig_device); }
void RigSettings::setDevice(const QString& id) {
    edit([&](auto& a) { a.rig_device = id.toStdString(); });
    Q_EMIT changed();
    Q_EMIT devicesChanged();  // devicePermitted follows the selection
}

bool RigSettings::devicePermitted() const {
    if (connection() == QLatin1String("network")) return true;
    const QString id = device();
    for (const QVariant& row : devices_) {
        const QVariantMap m = row.toMap();
        if (m.value(QStringLiteral("id")).toString() == id) return m.value(QStringLiteral("permitted")).toBool();
    }
    return false;
}

// Read by a QML binding: nothing may throw out of here.
bool RigSettings::bluetoothReady() const {
    try {
        return rig::android::has_permission("bt:");
    } catch (const std::exception&) {
        return false;
    }
}

QString RigSettings::host() const { return connection() == QLatin1String("network") ? QString::fromStdString(load_args().rig_device) : QString(); }
void RigSettings::setHost(const QString& h) {
    edit([&](auto& a) { a.rig_device = h.trimmed().toStdString(); });
    Q_EMIT changed();
}

int RigSettings::baud() const { return load_args().rig_baud; }
void RigSettings::setBaud(int b) {
    edit([&](auto& a) { a.rig_baud = b; });
    Q_EMIT changed();
}

QString RigSettings::pttMethod() const { return QString::fromStdString(load_args().ptt_method); }
void RigSettings::setPttMethod(const QString& m) {
    edit([&](auto& a) { a.ptt_method = m.toStdString(); });
    Q_EMIT changed();
}
// RFCOMM has no control lines: DTR and RTS aren't offered over Bluetooth.
QStringList RigSettings::pttMethods() const {
    QStringList out{QStringLiteral("vox"), QStringLiteral("cat")};
    if (connection() != QLatin1String("bluetooth")) out << QStringLiteral("dtr") << QStringLiteral("rts");
    return out;
}

QString RigSettings::pttAudio() const { return QString::fromStdString(load_args().ptt_audio); }
void RigSettings::setPttAudio(const QString& a) {
    edit([&](auto& args) { args.ptt_audio = a.toStdString(); });
    Q_EMIT changed();
}
bool RigSettings::pttAudioSupported() const {
#ifdef DATA2G_HAVE_RIG
    return rig::supports_ptt_audio_source(model());
#else
    return false;
#endif
}

QVariantList RigSettings::findModels(const QString& query) const {
    const QString needle = query.trimmed();
    QVariantList out;
    // Empty: the dummy and NET rigctl first (no radio needed; share the station PC's), then the rest.
    if (needle.isEmpty()) {
        for (const auto& m : all_models())
            if (m.model == rig::MODEL_DUMMY || m.model == rig::MODEL_NET_RIGCTL) out.append(model_row(m));
    }
    for (const auto& m : all_models()) {
        if (out.size() >= MAX_MODEL_ROWS) break;
        if (needle.isEmpty() ? (m.model == rig::MODEL_DUMMY || m.model == rig::MODEL_NET_RIGCTL)
                             : !QString::fromStdString(m.label()).contains(needle, Qt::CaseInsensitive))
            continue;
        out.append(model_row(m));
    }
    return out;
}

void RigSettings::refreshDevices() {
    devices_.clear();
    const QString kind = connection();
    if (kind != QLatin1String("network")) {
        try {
            const auto found = kind == QLatin1String("bluetooth") ? rig::android::bluetooth_devices() : rig::android::usb_devices();
            for (const auto& d : found)
                devices_.append(QVariantMap{{QStringLiteral("id"), QString::fromStdString(d.id)},
                                            {QStringLiteral("label"), QString::fromStdString(d.label)},
                                            {QStringLiteral("permitted"), d.permitted}});
        } catch (const std::exception& e) {
            // Not the same as an empty list: say so rather than "nothing plugged in".
            error_ = QString::fromStdString(e.what());
        }
    }
    Q_EMIT devicesChanged();
    Q_EMIT changed();
}

void RigSettings::requestPermission() {
    const QString kind = connection();
    if (kind == QLatin1String("network")) return;
    const std::string id = kind == QLatin1String("bluetooth") && device().isEmpty() ? std::string("bt:") : device().toStdString();
    if (id.empty()) return;
    try {
        rig::android::request_permission(id);
    } catch (const std::exception& e) {
        error_ = QString::fromStdString(e.what());
        Q_EMIT changed();
    }
}
