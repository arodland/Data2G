#include "settings.hpp"

#include <QSettings>
#include <QString>

#include "rig/config.hpp"

namespace data2g::app {

namespace {

std::optional<std::string> opt(const QString& s) {
    return s.trimmed().isEmpty() ? std::nullopt : std::optional(s.trimmed().toStdString());
}
QString qs(const std::optional<std::string>& s) { return QString::fromStdString(s.value_or("")); }
QString qs(const std::string& s) { return QString::fromStdString(s); }

// The string-valued rig options; the QSettings key is the Args field's name.
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

}  // namespace

Args load_settings(QSettings& s, Args a) {
    const auto str = [&](const char* k, const std::string& d) { return s.value(k, qs(d)).toString().toStdString(); };
    if (s.contains("mycall")) a.mycall = opt(s.value("mycall").toString());
    if (s.contains("input_device")) a.input_device = opt(s.value("input_device").toString());
    if (s.contains("output_device")) a.output_device = opt(s.value("output_device").toString());
    a.sample_rate = s.value("sample_rate", a.sample_rate).toInt();
    a.output_volume = s.value("output_volume", a.output_volume).toDouble();
    a.ptt_on_delay_ms = s.value("ptt_on_delay_ms", a.ptt_on_delay_ms).toInt();
    a.ptt_off_delay_ms = s.value("ptt_off_delay_ms", a.ptt_off_delay_ms).toInt();
    a.rig = s.value("rig", a.rig).toBool();
    a.rig_model = s.value("rig_model", a.rig_model).toInt();
    a.rig_baud = s.value("rig_baud", a.rig_baud).toInt();
    for (const auto& [key, field] : RIG_STRINGS) a.*field = str(key, a.*field);
    a.rig_timeout_ms = s.value("rig_timeout_ms", a.rig_timeout_ms).toInt();
    a.rig_retries = s.value("rig_retries", a.rig_retries).toInt();
    a.rig_poll_interval = s.value("rig_poll_interval", a.rig_poll_interval).toDouble();
    a.rig_debug = s.value("rig_debug", a.rig_debug).toBool();
    // Before --listen and model 2's empty device: "host", and rigctld's
    // address apart (port 0: no rig). Read once; save_settings drops them.
    if (a.rig_model == rig::MODEL_NET_RIGCTL && a.rig_device.empty() && s.contains("rigctld_port")) {
        const int port = s.value("rigctld_port").toInt();
        const std::string host = str("rigctld_host", "localhost");
        if (port == 0) a.rig = false;
        else if (host != "localhost" || port != 4532) a.rig_device = host + ":" + std::to_string(port);
    }
    a.listen = str("listen", str("host", a.listen));
    a.command_port = s.value("command_port", a.command_port).toInt();
    a.kiss_port = s.value("kiss_port", a.kiss_port).toInt();
    a.record = s.value("record", a.record).toBool();
    a.record_dir = str("record_dir", a.record_dir);
    a.decode_worker = s.value("decode_worker", a.decode_worker).toBool();
    a.live_constellation = s.value("live_constellation", a.live_constellation).toBool();
    a.noise_rule = s.value("noise_rule", a.noise_rule).toDouble();
    a.decoder_effort = s.value("decoder_effort", a.decoder_effort).toInt();
    a.cw_id = str("cw_id", a.cw_id);
    a.cw_wpm = s.value("cw_wpm", a.cw_wpm).toDouble();
    return a;
}

void save_settings(QSettings& s, const Args& a) {
    s.setValue("mycall", qs(a.mycall));
    s.setValue("input_device", qs(a.input_device));
    s.setValue("output_device", qs(a.output_device));
    s.setValue("sample_rate", a.sample_rate);
    s.setValue("output_volume", a.output_volume);
    s.setValue("ptt_on_delay_ms", a.ptt_on_delay_ms);
    s.setValue("ptt_off_delay_ms", a.ptt_off_delay_ms);
    s.setValue("rig", a.rig);
    s.setValue("rig_model", a.rig_model);
    s.setValue("rig_baud", a.rig_baud);
    for (const auto& [key, field] : RIG_STRINGS) s.setValue(key, qs(a.*field));
    s.setValue("rig_timeout_ms", a.rig_timeout_ms);
    s.setValue("rig_retries", a.rig_retries);
    s.setValue("rig_poll_interval", a.rig_poll_interval);
    s.setValue("rig_debug", a.rig_debug);
    s.setValue("listen", qs(a.listen));
    s.setValue("command_port", a.command_port);
    s.setValue("kiss_port", a.kiss_port);
    s.setValue("record", a.record);
    s.setValue("record_dir", qs(a.record_dir));
    for (const char* old : {"host", "kiss_address", "rigctld_host", "rigctld_port"}) s.remove(old);
    s.setValue("decode_worker", a.decode_worker);
    s.setValue("live_constellation", a.live_constellation);
    s.setValue("noise_rule", a.noise_rule);
    s.setValue("decoder_effort", a.decoder_effort);
    s.setValue("cw_id", qs(a.cw_id));
    s.setValue("cw_wpm", a.cw_wpm);
}

}  // namespace data2g::app
