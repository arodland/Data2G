#include "web_ui.hpp"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRandomGenerator>
#include <QSettings>
#include <QTcpSocket>
#include <QTimer>
#include <QUrl>
#include <QWebSocket>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "app/settings.hpp"
#include "app/web_password.hpp"
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
        case 401: return "Unauthorized";
        case 404: return "Not Found";
        case 403: return "Forbidden";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 415: return "Unsupported Media Type";
        case 429: return "Too Many Requests";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default: return "Error";
    }
}

constexpr const char* COOKIE = "data2g_session";
constexpr std::size_t MAX_SESSIONS = 64;  // ponytail: past this, all are dropped; an LRU if anyone ever has that many browsers

WebUi::Reply page(int code, const char* name) {
    QFile f(QStringLiteral(":/web/") + QLatin1String(name));
    if (!f.open(QIODevice::ReadOnly)) return error(500, QStringLiteral("the page is missing from the build"));
    return {code, "text/html; charset=utf-8", f.readAll()};
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
    connect(&ws_, &QWebSocketServer::newConnection, this, [this] {
        while (QWebSocket* s = ws_.nextPendingConnection()) {
            s->setParent(this);
            Client& c = clients_.emplace_back(Client{s, {}});
            connect(s, &QWebSocket::textMessageReceived, this, [this, &c](const QString& m) { command(c, m); });
            connect(s, &QWebSocket::disconnected, this, [this, s] {
                clients_.remove_if([s](const Client& c) { return c.socket == s; });
                s->deleteLater();
            });
        }
    });
    auto* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &WebUi::poll);
    timer->start(100);
    refresh_password();
    restart(args);
}

WebUi::~WebUi() {
    // before clients_ goes: a socket deleted later would report its disconnect into it
    for (Client& c : std::exchange(clients_, {})) {
        QObject::disconnect(c.socket, nullptr, this, nullptr);
        delete c.socket;
    }
}

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

bool WebUi::refresh_password() {
    store_.sync();  // set by data2g-gui or --set-password: in the file, not yet in our QSettings
    const QString hash = app::web_password_hash(store_);
    if (hash != password_hash_) {
        if (!password_hash_.isEmpty()) app::log_line(app::INFO, "web UI: the password changed; every session ended");
        password_hash_ = hash;
        sessions_.clear();
        for (Client& c : clients_) c.socket->close(QWebSocketProtocol::CloseCodePolicyViolated, QStringLiteral("the password changed"));
    }
    return !password_hash_.isEmpty();
}

bool WebUi::logged_in(const QByteArray& cookie) const {
    for (const QByteArray& part : cookie.split(';')) {
        const QByteArray kv = part.trimmed();
        if (kv.startsWith(QByteArray(COOKIE) + '=') && sessions_.count(kv.mid(qsizetype(std::strlen(COOKIE)) + 1))) return true;
    }
    return false;
}

void WebUi::poll() {
    if (++ticks_ % 20 == 0) refresh_password();  // every 2 s: a WebSocket makes no requests to notice it by
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

    // ponytail: every client every tick, however slow its link; drop one
    // whose bytesToWrite() keeps growing if that ever backs up
    for (Client& c : clients_) {
        if (!c.hello) continue;
        if (c.fps < 10 && c.seq.w < rows_.last) {  // as the GUI's frame timer: the newest row, once a period
            if (now < c.next_row) {
                c.seq.w = rows_.last;  // skipped
            } else {
                c.seq.w = rows_.last - 1;
                c.next_row = now + std::chrono::milliseconds(1000 / c.fps);
            }
        }
        c.socket->sendTextMessage(QString::fromUtf8(json(status(c.seq, c.hex))));
    }
}

void WebUi::command(Client& c, const QString& message) {
    const QJsonObject in = QJsonDocument::fromJson(message.toUtf8()).object();
    const QString cmd = in.value(QStringLiteral("cmd")).toString();
    const auto seq = [&](const char* k) { return std::uint64_t(std::max(0.0, in.value(QLatin1String(k)).toDouble())); };
    const bool hex = in.value(QStringLiteral("fmt")).toString() == QLatin1String("hex");
    // the GUI's choices (gui/main_window.cpp); anything else: every row
    const auto fps = [&] {
        const int f = in.value(QStringLiteral("fps")).toInt(10);
        return f == 1 || f == 2 || f == 5 ? f : 10;
    };
    if (cmd == QLatin1String("hello")) {
        c.seq = {seq("b"), seq("d"), seq("c"), seq("w")};
        c.hex = hex;
        c.fps = fps();
        c.hello = true;
    } else if (cmd == QLatin1String("fps")) {
        c.fps = fps();
        c.next_row = {};
    } else if (cmd == QLatin1String("fmt")) {
        c.hex = hex;
        c.seq.d = 0;  // every dump kept, again
    } else if (cmd == QLatin1String("abort")) {
        station_->abort_session();
    } else if (cmd == QLatin1String("monitor")) {
        monitoring_ = in.value(QStringLiteral("on")).toBool();
        store_.setValue(QStringLiteral("web/monitoring"), monitoring_);
        station_->set_monitor(monitoring_);
    }
}

QJsonObject WebUi::status(Seqs& seq, bool hex) const {
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
    o["bursts"] = since(bursts_, seq.b, same);
    const auto fmt = hex ? monitor::Format::HEX : monitor::Format::TEXT;
    // every dump kept (a new client, a format change, a restart of ours): the client starts its view over
    if (seq.d == 0 || seq.d > dumps_.last) o["reset_dumps"] = true;
    o["dumps"] = since(dumps_, seq.d, [&](const monitor::Dump& d) {
        return QJsonObject{{"tx", d.tx}, {"text", qs(monitor::render(d, fmt))}};
    });
    if (!constellation_.items.empty() && seq.c != constellation_.last) o["constellation"] = constellation_.items.back().second;
    QJsonArray rows = since(rows_, seq.w, same);
    while (rows.size() > 100) rows.removeFirst();  // a reconnect: the newest 10 s
    o["rows"] = rows;
    seq = {bursts_.last, dumps_.last, constellation_.last, rows_.last};
    o["seq"] = QJsonObject{{"b", double(seq.b)}, {"d", double(seq.d)}, {"c", double(seq.c)}, {"w", double(seq.w)}};  // for a reconnect's hello
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

WebUi::Reply WebUi::handle(const QByteArray& method, const QByteArray& target, const QByteArray& content_type, const QByteArray& cookie,
                           const QByteArray& body) {
    const qsizetype qmark = target.indexOf('?');
    const QByteArray path = qmark < 0 ? target : target.left(qmark);
    const bool api = path.startsWith("/api/");
    if (!refresh_password())
        return api ? error(503, QStringLiteral("no web password set: data2g-web --set-password")) : page(503, "setup.html");
    const bool authed = logged_in(cookie);
    if (method == "GET") {
        if (path == "/" || path == "/index.html") return authed ? page(200, "index.html") : page(200, "login.html");
        if (!authed) return error(401, QStringLiteral("log in"));
        if (path == "/api/session") return {200, "application/json", json({{"ok", true}})};
        if (path == "/api/settings") return {200, "application/json", json(settings())};
        return error(404, QStringLiteral("no such page"));
    }
    if (method != "POST") return error(405, QStringLiteral("GET or POST"));
    if (!content_type.trimmed().startsWith("application/json")) return error(415, QStringLiteral("POST application/json"));
    QJsonParseError bad;
    const QJsonDocument doc = QJsonDocument::fromJson(body, &bad);
    if (!body.isEmpty() && !doc.isObject()) return error(400, bad.error ? bad.errorString() : QStringLiteral("expected a JSON object"));
    const QJsonObject in = doc.object();
    if (path == "/api/login") {
        const auto now = std::chrono::steady_clock::now();
        if (now < next_login_) return error(429, QStringLiteral("wait a second"));
        if (!app::check_web_password(store_, in.value(QStringLiteral("password")).toString())) {
            next_login_ = now + std::chrono::seconds(1);
            app::log_line(app::WARNING, "web UI: a wrong password");
            return error(401, QStringLiteral("wrong password"));
        }
        QByteArray token(32, '\0');
        QRandomGenerator::system()->fillRange(reinterpret_cast<quint32*>(token.data()), token.size() / 4);
        token = token.toHex();
        if (sessions_.size() >= MAX_SESSIONS) sessions_.clear();
        sessions_.insert(token);
        return {200, "application/json", json({{"ok", true}}),
                QByteArray("Set-Cookie: ") + COOKIE + '=' + token + "; Path=/; HttpOnly; SameSite=Strict\r\n"};
    }
    if (!authed) return error(401, QStringLiteral("log in"));
    if (path == "/api/logout") {
        for (const QByteArray& part : cookie.split(';'))
            if (part.trimmed().startsWith(QByteArray(COOKIE) + '=')) sessions_.erase(part.trimmed().mid(qsizetype(std::strlen(COOKIE)) + 1));
        return {200, "application/json", json({{"ok", true}}),
                QByteArray("Set-Cookie: ") + COOKIE + "=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0\r\n"};
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

// ponytail: one request per connection (Connection: close), no keep-alive:
// the page loads once and then lives on its WebSocket.
//
// The request is peeked at, not read, until it is whole: a WebSocket upgrade
// goes to QWebSocketServer with its handshake still unread.
void WebUi::read(QTcpSocket* s) {
    const auto respond = [s](const Reply& r) {
        s->write(QByteArray("HTTP/1.1 ") + QByteArray::number(r.code) + ' ' + reason(r.code) + "\r\nContent-Type: " + r.type +
                 "\r\nContent-Length: " + QByteArray::number(r.body.size()) +
                 "\r\nCache-Control: no-store\r\nConnection: close\r\n" + r.headers + "\r\n" + r.body);
        s->disconnectFromHost();
    };
    if (s->property("answered").toBool()) return;  // anything after a request: ignored until the close
    const QByteArray buf = s->peek(s->bytesAvailable());
    const auto answer = [&](const Reply& r) {
        s->setProperty("answered", true);
        respond(r);
    };
    if (buf.size() > MAX_REQUEST) return answer(error(413, QStringLiteral("too large")));
    const qsizetype end = buf.indexOf("\r\n\r\n");
    if (end < 0) return;
    const QList<QByteArray> lines = buf.left(end).split('\n');
    const QList<QByteArray> request = lines.first().trimmed().split(' ');
    if (request.size() != 3) return answer(error(400, QStringLiteral("bad request line")));
    qsizetype length = 0;
    QByteArray type, host, origin, upgrade, cookie;
    for (qsizetype i = 1; i < lines.size(); ++i) {
        const qsizetype colon = lines[i].indexOf(':');
        if (colon < 0) continue;
        const QByteArray name = lines[i].left(colon).trimmed().toLower(), value = lines[i].mid(colon + 1).trimmed();
        if (name == "content-length") length = value.toLongLong();
        if (name == "content-type") type = value;
        if (name == "host") host = value;
        if (name == "origin") origin = value;
        if (name == "upgrade") upgrade = value.toLower();
        if (name == "cookie") cookie += (cookie.isEmpty() ? "" : "; ") + value;
    }
    if (upgrade == "websocket") {
        if (request[1] != "/ws") return answer(error(404, QStringLiteral("no such WebSocket")));
        // No CORS for WebSockets: any page the browser has open could connect, unless it must be ours.
        if (!origin.isEmpty() && QUrl(QString::fromLatin1(origin)).authority() != QString::fromLatin1(host))
            return answer(error(403, QStringLiteral("not from this page")));
        if (!refresh_password()) return answer(error(503, QStringLiteral("no web password set")));
        if (!logged_in(cookie)) return answer(error(401, QStringLiteral("log in")));
        QObject::disconnect(s, nullptr, this, nullptr);
        QObject::disconnect(s, &QTcpSocket::disconnected, s, &QObject::deleteLater);
        ws_.handleConnection(s);  // the QWebSocket it makes owns s from here
        return;
    }
    if (length < 0 || length > MAX_REQUEST) return answer(error(413, QStringLiteral("too large")));
    if (buf.size() < end + 4 + length) return;
    s->read(end + 4 + length);
    answer(handle(request[0], request[1], type, cookie, buf.mid(end + 4, length)));
}

}  // namespace data2g::web
