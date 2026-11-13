#include "app/settings.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSettings>
#include <QStandardPaths>

#include <string_view>

#include <utility>

#include "rig/config.hpp"

namespace data2g::app {

namespace {

std::optional<std::string> opt(const QString& s) {
    return s.trimmed().isEmpty() ? std::nullopt : std::optional(s.trimmed().toStdString());
}
QString qs(const std::optional<std::string>& s) { return QString::fromStdString(s.value_or("")); }
QString qs(const std::string& s) { return QString::fromStdString(s); }

// The string-valued rig options; the key is the Args field's name.
// (A named member-pointer type: MSVC can't parse one spelled inside pair<>.)
using StringField = std::string Args::*;
struct RigString {
    const char* key;
    StringField field;
};
const RigString RIG_STRINGS[] = {
    {"rig_device", &Args::rig_device},       {"rig_data_bits", &Args::rig_data_bits},
    {"rig_stop_bits", &Args::rig_stop_bits}, {"rig_parity", &Args::rig_parity},
    {"rig_handshake", &Args::rig_handshake}, {"rig_dtr", &Args::rig_dtr},
    {"rig_rts", &Args::rig_rts},             {"ptt_method", &Args::ptt_method},
    {"ptt_device", &Args::ptt_device},       {"ptt_audio", &Args::ptt_audio},
    {"rig_mode", &Args::rig_mode},
};

// Keys older versions saved: read once, dropped on save.
const char* const OLD_KEYS[] = {"host", "kiss_address", "rigctld_host", "rigctld_port"};

QString default_config() {
    return QStandardPaths::writableLocation(QStandardPaths::GenericConfigLocation) + QStringLiteral("/Data2G/data2g.conf");
}
QString g_config;  // open_config's choice

// The old store (QSettings' native one for Data2G/data2g-gui) to `to`: on
// Linux an ini file of the same format, renamed; elsewhere (the Windows
// registry, a macOS plist) copied key by key, the old one left in place.
void migrate_old(const QString& to) {
    QSettings old(QStringLiteral("Data2G"), QStringLiteral("data2g-gui"));
    old.setFallbacksEnabled(false);  // not /etc/xdg or Data2G.conf: only its own file
    if (old.allKeys().isEmpty()) return;
    QDir().mkpath(QFileInfo(to).absolutePath());
    if (old.fileName().endsWith(QStringLiteral(".conf")) && QFile::rename(old.fileName(), to)) return;
    QSettings neu(to, QSettings::IniFormat);
    for (const QString& k : old.allKeys()) neu.setValue(k, old.value(k));
}

}  // namespace

std::unique_ptr<QSettings> open_config(int argc, char** argv) {
    QString path;
    for (int i = 1; i < argc; ++i) {  // parse() takes the flag too; a missing value is its error
        const std::string_view arg = argv[i];
        if (arg == "--config" && i + 1 < argc) path = QString::fromLocal8Bit(argv[++i]);
        else if (arg.starts_with("--config=")) path = QString::fromLocal8Bit(argv[i] + 9);
    }
    if (path.isEmpty()) {
        path = default_config();
        if (!QFile::exists(path)) migrate_old(path);
    }
    g_config = path;
    return settings_store();
}

std::unique_ptr<QSettings> settings_store() {
    return std::make_unique<QSettings>(g_config.isEmpty() ? default_config() : g_config, QSettings::IniFormat);
}

QVariantMap settings_map(const Args& a) {
    QVariantMap m;
    m["mycall"] = qs(a.mycall);
    m["input_device"] = qs(a.input_device);
    m["output_device"] = qs(a.output_device);
    m["broadcast_mode"] = qs(a.broadcast_mode);
    m["tci"] = a.tci;
    m["tci_host"] = qs(a.tci_host);
    m["sample_rate"] = a.sample_rate;
    m["output_volume"] = a.output_volume;
    m["ptt_on_delay_ms"] = a.ptt_on_delay_ms;
    m["ptt_off_delay_ms"] = a.ptt_off_delay_ms;
    m["rig"] = a.rig;
    m["rig_model"] = a.rig_model;
    m["rig_baud"] = a.rig_baud;
    for (const auto& [key, field] : RIG_STRINGS) m[key] = qs(a.*field);
    m["rig_timeout_ms"] = a.rig_timeout_ms;
    m["rig_retries"] = a.rig_retries;
    m["rig_poll_interval"] = a.rig_poll_interval;
    m["rig_debug"] = a.rig_debug;
    m["listen"] = qs(a.listen);
    m["command_port"] = a.command_port;
    m["kiss_port"] = a.kiss_port;
    m["record"] = a.record;
    m["record_dir"] = qs(a.record_dir);
    m["decode_worker"] = a.decode_worker;
    m["live_constellation"] = a.live_constellation;
    m["noise_rule"] = a.noise_rule;
    m["decoder_effort"] = a.decoder_effort;
    m["cw_id"] = qs(a.cw_id);
    m["cw_wpm"] = a.cw_wpm;
    m["nods"] = a.nods;
    return m;
}

Args from_settings(const QVariantMap& m, Args a) {
    const auto str = [&](const char* k, const std::string& d) { return m.value(k, qs(d)).toString().toStdString(); };
    if (m.contains("mycall")) a.mycall = opt(m.value("mycall").toString());
    if (m.contains("input_device")) a.input_device = opt(m.value("input_device").toString());
    if (m.contains("output_device")) a.output_device = opt(m.value("output_device").toString());
    if (m.contains("broadcast_mode")) a.broadcast_mode = opt(m.value("broadcast_mode").toString());
    a.tci = m.value("tci", a.tci).toBool();
    a.tci_host = str("tci_host", a.tci_host);
    a.sample_rate = m.value("sample_rate", a.sample_rate).toInt();
    a.output_volume = m.value("output_volume", a.output_volume).toDouble();
    a.ptt_on_delay_ms = m.value("ptt_on_delay_ms", a.ptt_on_delay_ms).toInt();
    a.ptt_off_delay_ms = m.value("ptt_off_delay_ms", a.ptt_off_delay_ms).toInt();
    a.rig = m.value("rig", a.rig).toBool();
    a.rig_model = m.value("rig_model", a.rig_model).toInt();
    a.rig_baud = m.value("rig_baud", a.rig_baud).toInt();
    for (const auto& [key, field] : RIG_STRINGS) a.*field = str(key, a.*field);
    a.rig_timeout_ms = m.value("rig_timeout_ms", a.rig_timeout_ms).toInt();
    a.rig_retries = m.value("rig_retries", a.rig_retries).toInt();
    a.rig_poll_interval = m.value("rig_poll_interval", a.rig_poll_interval).toDouble();
    a.rig_debug = m.value("rig_debug", a.rig_debug).toBool();
    // Before --listen and model 2's empty device: "host", and rigctld's
    // address apart (port 0: no rig). Read once; save_settings drops them.
    if (a.rig_model == rig::MODEL_NET_RIGCTL && a.rig_device.empty() && m.contains("rigctld_port")) {
        const int port = m.value("rigctld_port").toInt();
        const std::string host = str("rigctld_host", "localhost");
        if (port == 0) a.rig = false;
        else if (host != "localhost" || port != 4532) a.rig_device = host + ":" + std::to_string(port);
    }
    a.listen = str("listen", str("host", a.listen));
    a.command_port = m.value("command_port", a.command_port).toInt();
    a.kiss_port = m.value("kiss_port", a.kiss_port).toInt();
    a.record = m.value("record", a.record).toBool();
    a.record_dir = str("record_dir", a.record_dir);
    a.decode_worker = m.value("decode_worker", a.decode_worker).toBool();
    a.live_constellation = m.value("live_constellation", a.live_constellation).toBool();
    a.noise_rule = m.value("noise_rule", a.noise_rule).toDouble();
    a.decoder_effort = m.value("decoder_effort", a.decoder_effort).toInt();
    a.cw_id = str("cw_id", a.cw_id);
    a.cw_wpm = m.value("cw_wpm", a.cw_wpm).toDouble();
    a.nods = m.value("nods", a.nods).toBool();
    return a;
}

Args load_settings(QSettings& s, Args base) {
    QVariantMap m;
    for (const QString& k : s.allKeys()) m[k] = s.value(k);
    return from_settings(m, std::move(base));
}

const QStringList LIVE_SETTINGS = {"mycall",          "output_volume",  "ptt_on_delay_ms", "ptt_off_delay_ms", "live_constellation",
                                   "noise_rule",      "decoder_effort", "cw_id",           "cw_wpm",           "record",
                                   "record_dir",      "nods"};

bool needs_restart(const Args& from, const Args& to) {
    QVariantMap live = settings_map(from);
    for (auto it = live.begin(); it != live.end();) it = LIVE_SETTINGS.contains(it.key()) ? std::next(it) : live.erase(it);
    return from_settings(live, to) != from;  // `to` with from's live values: anything left differs
}

void save_settings(QSettings& s, const Args& a) {
    const QVariantMap m = settings_map(a);
    for (auto it = m.begin(); it != m.end(); ++it) s.setValue(it.key(), it.value());
    for (const char* old : OLD_KEYS) s.remove(old);
}

}  // namespace data2g::app
