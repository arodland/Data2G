// data2g-web's server: the station of data2g-host, and in a browser what
// data2g-gui's windows show (gui/main_window.hpp), on two tabs: Monitor
// (waterfall, constellation, link, lamps, burst log, packet dumps) and
// Settings (the settings dialog's fields, app/settings.hpp's map).
//
// A plain HTTP/1.1 server on QTcpServer, one request per connection, for the
// page (web/index.html, a Qt resource) and the settings; the Monitor tab is a
// WebSocket on the same port, handed to QWebSocketServer at its upgrade.
//
//   GET  /              the page
//   GET  /api/settings  the saved settings, their choices, devices, rig models
//   POST /api/settings  {key: value, ...} over the running settings; saved, station restarted if changed
//   GET  /ws            WebSocket. Every 100 ms the server sends the state, and the
//                       bursts, dumps, constellation and waterfall rows that client
//                       has not had yet (each kept here with a sequence number, so
//                       a reconnect resumes). The client sends
//                       {"cmd": "hello", "b", "d", "c", "w": its last seqs, "fmt": "text"|"hex", "fps"},
//                       {"cmd": "fmt", "fmt"} (every dump kept, again, in that format),
//                       {"cmd": "fps", "fps": 1|2|5|10} (the waterfall's speed, as the GUI's
//                       right-click menu: below 10, the newest row once a period, the rest skipped),
//                       {"cmd": "abort"}, {"cmd": "monitor", "on": bool}.
//
// One password (app/web_password.hpp), no user name. With none set, every
// page is setup.html (how to set one) and every API call a 503; once one is,
// login.html until it is given: POST /api/login {"password"} sets an
// HttpOnly, SameSite=Strict session cookie, which every other request and the
// WebSocket upgrade must carry (POST /api/logout ends it; GET /api/session
// says whether it still holds). Setting a new password, from anywhere, ends
// every session and closes every WebSocket.
//
// POSTs must be application/json: a page on another site can't send that
// without a CORS preflight, which is never answered. WebSockets have no CORS,
// so the upgrade must come from our own page: an Origin, if sent, naming the
// Host it was sent to.
#pragma once

#include <QHostAddress>
#include <QJsonObject>
#include <QObject>
#include <QTcpServer>
#include <QWebSocketServer>

#include <chrono>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <set>

#include "app/station.hpp"
#include "monitor/monitor.hpp"

class QSettings;
class QTcpSocket;
class QWebSocket;

namespace data2g::web {

class WebUi : public QObject {
public:
    // Starts a station with `args`; `store` is where settings are saved.
    WebUi(app::Args args, QSettings& store);
    ~WebUi() override;

    // Throws std::runtime_error when the address can't be bound. Port 0: any free one.
    quint16 listen(const QHostAddress& address, quint16 port);
    // Stops the station and starts one with `a`; a failure is shown, not thrown.
    void restart(const app::Args& a);
    const app::Station& station() const { return *station_; }
    void poll();  // every 100 ms; public so a test can drive it

    struct Reply {
        int code = 200;
        QByteArray type = "application/json", body;
        QByteArray headers = {};  // more header lines, each ending "\r\n"
    };
    // One request, as the socket handler sees it; public for the test.
    Reply handle(const QByteArray& method, const QByteArray& target, const QByteArray& content_type, const QByteArray& cookie,
                 const QByteArray& body);

private:
    struct Seqs {  // the last burst, dump, constellation and waterfall row a client has
        std::uint64_t b = 0, d = 0, c = 0, w = 0;
    };
    struct Client {
        QWebSocket* socket;
        Seqs seq;
        bool hex = false;    // dumps as hex dumps
        bool hello = false;  // nothing is sent before its hello
        int fps = 10;        // waterfall rows a second, at most (10: every row; the engine reads 0.1 s blocks)
        std::chrono::steady_clock::time_point next_row{};
    };
    void read(QTcpSocket* s);
    // The stored password, re-read (another process may have set it); a new
    // one ends every session. False when none is set.
    bool refresh_password();
    bool logged_in(const QByteArray& cookie) const;
    void command(Client& c, const QString& message);
    // The state, and what came after `seq` (advanced to the newest).
    QJsonObject status(Seqs& seq, bool hex) const;
    QJsonObject settings() const;

    template <typename T>
    struct Log {  // the newest `cap` items, numbered from 1
        std::deque<std::pair<std::uint64_t, T>> items;
        std::uint64_t last = 0;
        std::size_t cap;
        void push(T t) {
            items.emplace_back(++last, std::move(t));
            if (items.size() > cap) items.pop_front();
        }
    };

    QSettings& store_;
    QTcpServer server_;
    QWebSocketServer ws_{QStringLiteral("data2g-web"), QWebSocketServer::NonSecureMode};
    std::list<Client> clients_;
    QString password_hash_;          // as last read; empty: none set
    std::set<QByteArray> sessions_;  // session cookies' tokens
    std::chrono::steady_clock::time_point next_login_{};  // a failed login holds off the next for a second
    int ticks_ = 0;
    std::unique_ptr<app::Station> station_;
    QString error_;
    bool monitoring_;
    Log<QJsonObject> bursts_{{}, 0, 500};
    Log<monitor::Dump> dumps_{{}, 0, 1000};
    Log<QJsonObject> rows_{{}, 0, 300};  // waterfall rows
    Log<QJsonObject> constellation_{{}, 0, 1};
    std::uint64_t seen_ = 0;  // input samples read, at the last waterfall row
    struct Sample {
        std::chrono::steady_clock::time_point t;
        std::int64_t rx, tx;
    };
    std::deque<Sample> samples_;  // throughput over the stats interval
};

}  // namespace data2g::web
