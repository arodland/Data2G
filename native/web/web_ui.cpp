#include "web_ui.hpp"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSettings>
#include <QTcpSocket>
#include <QTimer>
#include <QUrlQuery>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "app/settings.hpp"
#include "arq/modes.hpp"
#include "arq/session.hpp"
#include "generated/config.hpp"
#ifdef DATA2G_HAVE_AUDIO
#include "audio/card.hpp"
#endif
#ifdef DATA2G_HAVE_RIG
#include "rig/hamlib/hamlib.hpp"
#endif

namespace data2g::web {

namespace {

constexpr qsizetype MAX_REQUEST = 1 << 20;
// The waterfall's dB ranges (gui/waterfall.cpp's): TX audio is full scale, so its own.
constexpr double DB_FLOOR = -95.0, DB_CEIL = -20.0, TX_DB_FLOOR = -80.0, TX_DB_CEIL = -15.0;

QString qs(const std::string& s) { return QString::fromStdString(s); }

QByteArray json(const QJsonObject& o) { return QJsonDocument(o).toJson(QJsonDocument::Compact); }

WebUi::Reply error(int code, const QString& why) { return {code, "application/json", json({{"error", why}})}; }

const char* reason(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 415: return "Unsupported Media Type";
        case 500: return "Internal Server Error";
        default: return "Error";
    }
}

QJsonArray points(const std::vector<std::complex<float>>& zs) {
    QJsonArray out;  // flat re, im, ...; 3 decimals is finer than a pixel
    for (const auto& z : zs) {
        out.append(std::round(z.real() * 1000.0) / 1000.0);
        out.append(std::round(z.imag() * 1000.0) / 1000.0);
    }
    return out;
}

}  // namespace

WebUi::WebUi(app::Args args, QSettings& store) : store_(store) {
    monitoring_ = store_.value(QStringLiteral("web/monitoring"), false).toBool();
    connect(&server_, &QTcpServer::newConnection, this, [this] {
        while (QTcpSocket* s = server_.nextPendingConnection()) {
            connect(s, &QTcpSocket::readyRead, this, [this, s] { read(s); });
            connect(s, &QTcpSocket::disconnected, s, &QObject::deleteLater);
        }
    });
    auto* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &WebUi::poll);
    timer->start(100);
    restart(args);
}

WebUi::~WebUi() = default;

quint16 WebUi::listen(const QHostAddress& address, quint16 port) {
    if (!server_.listen(address, port))
        throw std::runtime_error("web UI: can't listen on " + address.toString().toStdString() + ":" + std::to_string(port) + ": " +
                                 server_.errorString().toStdString());
    return server_.serverPort();
}

void WebUi::restart(const app::Args& args) {
    station_.reset();  // stops it: PTT down, ports closed
    samples_.clear();
    error_.clear();
    station_ = std::make_unique<app::Station>(args);
    if (const auto bad = app::check(args)) {
        error_ = qs(*bad);
    } else {
        try {
            station_->start();
        } catch (const std::exception& e) {
            error_ = qs(e.what());
            app::log_line(app::ERROR, std::string("station: ") + e.what());
        }
    }
    poll();
}

void WebUi::poll() {
    auto& s = *station_;
    if (s.failed() && s.running()) {
        s.stop();
        error_ = QStringLiteral("the engine stopped (see the log)");
    }
    s.set_monitor(monitoring_);
    for (auto& d : s.take_dumps()) dumps_.push(std::move(d));
    if (auto c = s.take_constellation())
        constellation_.push(QJsonObject{{"mode", qs(c->mode)}, {"tones", c->tones}, {"live", c->live}, {"lost", c->lost},
                                        {"symbols", points(c->symbols)}, {"points", points(c->points)}});
    for (const auto& b : s.take_bursts()) {
        const QString result = b.lost           ? QStringLiteral("lost")
                               : b.decoded == 0 ? QStringLiteral("undecoded")  // none of it ours, or none of it decoded
                               : b.failed       ? QStringLiteral("incomplete data (%1/%2)").arg(b.n_cw - b.failed).arg(b.n_cw)
                                                : QStringLiteral("ok");
        bursts_.push({{"t", double(std::chrono::duration_cast<std::chrono::milliseconds>(b.when.time_since_epoch()).count())},
                      {"mode", QStringLiteral("%1 x%2").arg(qs(b.mode)).arg(b.n_cw)},
                      {"snr", b.snr_db ? QJsonValue(std::round(*b.snr_db * 10.0) / 10.0) : QJsonValue()},
                      {"result", result}});
    }

    // a waterfall row when the engine has read a new block: its spectrum as
    // bytes over the display range (RX or TX), and the input's peak (RX only)
    std::uint64_t total = 0;
    bool tx = false;
    const auto block = s.input_tail(app::SPECTRUM_N, &total, &tx);
    if (total != seen_ && block.size() >= app::SPECTRUM_N) {
        seen_ = total;
        const auto db = app::spectrum_db(block);
        const double floor = tx ? TX_DB_FLOOR : DB_FLOOR, ceil = tx ? TX_DB_CEIL : DB_CEIL;
        QByteArray bins(qsizetype(db.size()), '\0');
        for (std::size_t i = 0; i < db.size(); ++i)
            bins[qsizetype(i)] = char(std::lround(std::clamp((db[i] - floor) / (ceil - floor), 0.0, 1.0) * 255.0));
        QJsonObject row{{"tx", tx}, {"bins", QString::fromLatin1(bins.toBase64())}};
        if (!tx) {
            double peak = 0.0;
            for (auto it = block.end() - config::FS / 10; it != block.end(); ++it) peak = std::max(peak, std::abs(*it));
            row["peak"] = peak;
        }
        rows_.push(row);
    }

    const app::LinkStatus l = s.link();
    const auto now = std::chrono::steady_clock::now();
    if (!samples_.empty() && (l.rx_bytes < samples_.back().rx || l.tx_bytes < samples_.back().tx)) samples_.clear();  // a new connection
    samples_.push_back({now, l.rx_bytes, l.tx_bytes});
    const double window = s.args().stats_interval > 0 ? s.args().stats_interval : 60.0;
    while (samples_.size() > 2 && std::chrono::duration<double>(now - samples_[1].t).count() >= window) samples_.pop_front();
}

QJsonObject WebUi::status(const QByteArray& query) const {
    const QUrlQuery q(QString::fromLatin1(query));
    const auto after = [&](const char* key) { return q.queryItemValue(QLatin1String(key)).toULongLong(); };
    // Everything after `seq`; all of it when `seq` is from before a restart of ours.
    const auto since = [](const auto& log, std::uint64_t seq, auto&& fn) {
        if (seq > log.last) seq = 0;
        QJsonArray out;
        for (const auto& [n, item] : log.items)
            if (n > seq) out.append(fn(item));
        return out;
    };
    const auto& s = *station_;
    const app::Args& a = s.args();
    QJsonObject o;

    const QString call = qs(a.mycall.value_or("NOCALL"));
    if (!s.running()) {
        o["station"] = QStringLiteral("%1: stopped: %2").arg(call, error_);
    } else {
        QStringList on;
        on << QStringLiteral("VARA %1:%2").arg(qs(a.listen)).arg(a.command_port);
        on << QStringLiteral("KISS %1:%2").arg(qs(a.listen)).arg(a.kiss_port);
        on << (a.audio_io.empty() ? QStringLiteral("audio %1 / %2").arg(qs(a.input_device.value_or("default")), qs(a.output_device.value_or("default")))
                                  : QStringLiteral("audio %1").arg(qs(a.audio_io)));
        o["station"] = QStringLiteral("%1 on %2").arg(call, on.join(QStringLiteral(", ")));
    }
    o["running"] = s.running();

    using arq::SessionState;
    const app::LinkStatus l = s.link();
    QJsonObject link{{"state", QString::fromLatin1(arq::state_name(l.state))},
                     {"peer", qs(l.peer)},
                     {"bw", qs(arq::bw_name(std::clamp(l.cap, 0, 2)))},
                     {"mode", qs(l.mode)},
                     {"rx_bytes", double(l.rx_bytes)},
                     {"tx_bytes", double(l.tx_bytes)}};
    if (const auto* m = l.mode.empty() ? nullptr : arq::mode(l.mode)) link["mode_hz"] = std::round(arq::width_hz(*m));
    if (!samples_.empty()) {
        const double dt = std::chrono::duration<double>(samples_.back().t - samples_.front().t).count();
        const auto bps = [&](std::int64_t d) { return dt > 0 ? std::round(8.0 * double(d) / dt) : 0.0; };
        link["rx_bps"] = bps(l.rx_bytes - samples_.front().rx);
        link["tx_bps"] = bps(l.tx_bytes - samples_.front().tx);
    }
    o["link"] = link;
    o["abortable"] = s.running() && (l.state == SessionState::CONNECTING || l.state == SessionState::CONNECTED ||
                                     l.state == SessionState::DISCONNECTING);
    if (s.running() && (a.rig_poll_interval > 0 || a.tci)) {  // TCI reports the dial unasked
        const auto hz = s.rig_frequency();
        o["dial"] = hz ? QJsonValue(*hz) : QJsonValue();
    }
    o["busy"] = s.busy();
    o["ptt"] = s.ptt();
    const app::AudioCounters c = s.counters();
    o["counters"] = QJsonObject{{"overflows", double(c.overflows)}, {"dropped", double(c.dropped)}, {"backlog_s", c.backlog_s},
                                {"late", double(c.late)},           {"underruns", double(c.underruns)},
                                {"decode_dropped", double(c.decode_dropped)}};
    o["monitoring"] = monitoring_;

    const auto same = [](const QJsonObject& x) { return x; };
    o["bursts"] = since(bursts_, after("b"), same);
    o["b"] = double(bursts_.last);
    const auto fmt = q.queryItemValue(QStringLiteral("fmt")) == QLatin1String("hex") ? monitor::Format::HEX : monitor::Format::TEXT;
    o["dumps"] = since(dumps_, after("d"), [&](const monitor::Dump& d) {
        return QJsonObject{{"tx", d.tx}, {"text", qs(monitor::render(d, fmt))}};
    });
    o["d"] = double(dumps_.last);
    if (!constellation_.items.empty() && after("c") != constellation_.last) o["constellation"] = constellation_.items.back().second;
    o["c"] = double(constellation_.last);
    QJsonArray rows = since(rows_, after("w"), same);
    while (rows.size() > 100) rows.removeFirst();  // a tab back from the background: the newest 10 s
    o["rows"] = rows;
    o["w"] = double(rows_.last);
    return o;
}

QJsonObject WebUi::settings() const {
    QJsonObject o{{"values", QJsonObject::fromVariantMap(app::settings_map(station_->args()))},
                  {"defaults", QJsonObject::fromVariantMap(app::settings_map(app::Args{}))},
                  {"audio_io", qs(station_->args().audio_io)}};
    QJsonObject choices;
    for (const char* f : {"rig_data_bits", "rig_stop_bits", "rig_parity", "rig_handshake", "rig_dtr", "rig_rts", "ptt_method",
                          "ptt_audio", "rig_mode"}) {
        QJsonArray values;
        for (const auto& v : app::rig_choices(f)) values.append(qs(v));
        choices[QLatin1String(f)] = values;
    }
    o["choices"] = choices;
    QJsonArray inputs, outputs;
#ifdef DATA2G_HAVE_AUDIO
    for (const auto& d : audio::card::input_devices()) inputs.append(qs(d.name));
    for (const auto& d : audio::card::output_devices()) outputs.append(qs(d.name));
#endif
    o["inputs"] = inputs;
    o["outputs"] = outputs;
    QJsonArray models;  // [number, label, port type, has mic/data audio]
#ifdef DATA2G_HAVE_RIG
    for (const rig::RigModel& m : rig::list_models())
        models.append(QJsonArray{m.model, qs(m.label()), int(m.port), rig::supports_ptt_audio_source(m.model)});
#endif
    o["rig_models"] = models;
    return o;
}

WebUi::Reply WebUi::handle(const QByteArray& method, const QByteArray& target, const QByteArray& content_type, const QByteArray& body) {
    const qsizetype qmark = target.indexOf('?');
    const QByteArray path = qmark < 0 ? target : target.left(qmark), query = qmark < 0 ? QByteArray() : target.mid(qmark + 1);
    if (method == "GET") {
        if (path == "/" || path == "/index.html") {
            QFile page(QStringLiteral(":/web/index.html"));
            if (!page.open(QIODevice::ReadOnly)) return error(500, QStringLiteral("the page is missing from the build"));
            return {200, "text/html; charset=utf-8", page.readAll()};
        }
        if (path == "/api/status") return {200, "application/json", json(status(query))};
        if (path == "/api/settings") return {200, "application/json", json(settings())};
        return error(404, QStringLiteral("no such page"));
    }
    if (method != "POST") return error(405, QStringLiteral("GET or POST"));
    if (!content_type.trimmed().startsWith("application/json")) return error(415, QStringLiteral("POST application/json"));
    QJsonParseError bad;
    const QJsonDocument doc = QJsonDocument::fromJson(body, &bad);
    if (!body.isEmpty() && !doc.isObject()) return error(400, bad.error ? bad.errorString() : QStringLiteral("expected a JSON object"));
    const QJsonObject in = doc.object();
    if (path == "/api/abort") {
        station_->abort_session();
        return {200, "application/json", json({{"ok", true}})};
    }
    if (path == "/api/monitor") {
        monitoring_ = in.value(QStringLiteral("on")).toBool();
        store_.setValue(QStringLiteral("web/monitoring"), monitoring_);
        station_->set_monitor(monitoring_);
        return {200, "application/json", json({{"ok", true}})};
    }
    if (path == "/api/settings") {
        // Only the keys sent change: a command line's value stays unless edited.
        // Each must be a setting, of its type: from_settings would read "x" as port 0.
        const QJsonObject known = QJsonObject::fromVariantMap(app::settings_map(station_->args()));
        for (auto it = in.begin(); it != in.end(); ++it) {
            const QJsonValue want = known.value(it.key());
            if (want.isUndefined()) return error(400, QStringLiteral("no such setting: %1").arg(it.key()));
            if (it.value().type() != want.type()) return error(400, QStringLiteral("%1: wrong type").arg(it.key()));
            if (it.value().isDouble() && !std::isfinite(it.value().toDouble())) return error(400, QStringLiteral("%1: not a number").arg(it.key()));
        }
        const app::Args a = app::from_settings(in.toVariantMap(), station_->args());
        if (const auto why = app::check(a)) return error(400, qs(*why));
        app::save_settings(store_, a);
        if (a != station_->args() || !station_->running()) restart(a);
        return {200, "application/json", json({{"ok", true}, {"running", station_->running()}, {"error", error_}})};
    }
    return error(404, QStringLiteral("no such endpoint"));
}

// ponytail: one request per connection (Connection: close) and no
// keep-alive; at 5 polls a second on a LAN that is fine.
void WebUi::read(QTcpSocket* s) {
    QByteArray buf = s->property("req").toByteArray() + s->readAll();
    const auto respond = [s](const Reply& r) {
        s->write(QByteArray("HTTP/1.1 ") + QByteArray::number(r.code) + ' ' + reason(r.code) + "\r\nContent-Type: " + r.type +
                 "\r\nContent-Length: " + QByteArray::number(r.body.size()) +
                 "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n" + r.body);
        s->disconnectFromHost();
    };
    if (buf.size() > MAX_REQUEST) return respond(error(413, QStringLiteral("too large")));
    const qsizetype end = buf.indexOf("\r\n\r\n");
    if (end < 0) {
        s->setProperty("req", buf);
        return;
    }
    const QList<QByteArray> lines = buf.left(end).split('\n');
    const QList<QByteArray> request = lines.first().trimmed().split(' ');
    if (request.size() != 3) return respond(error(400, QStringLiteral("bad request line")));
    qsizetype length = 0;
    QByteArray type;
    for (qsizetype i = 1; i < lines.size(); ++i) {
        const qsizetype colon = lines[i].indexOf(':');
        if (colon < 0) continue;
        const QByteArray name = lines[i].left(colon).trimmed().toLower(), value = lines[i].mid(colon + 1).trimmed();
        if (name == "content-length") length = value.toLongLong();
        if (name == "content-type") type = value;
    }
    if (length < 0 || length > MAX_REQUEST) return respond(error(413, QStringLiteral("too large")));
    if (buf.size() < end + 4 + length) {
        s->setProperty("req", buf);
        return;
    }
    s->setProperty("req", QVariant());
    respond(handle(request[0], request[1], type, buf.mid(end + 4, length)));
}

}  // namespace data2g::web
