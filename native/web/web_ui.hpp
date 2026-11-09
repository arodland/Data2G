// data2g-web's server: the station of data2g-host, and in a browser what
// data2g-gui's windows show (gui/main_window.hpp), on two tabs: Monitor
// (waterfall, constellation, link, lamps, burst log, packet dumps) and
// Settings (the settings dialog's fields, app/settings.hpp's map).
//
// A plain HTTP/1.1 server on QTcpServer, one request per connection. The
// page (web/index.html, a Qt resource) polls GET /api/status; everything it
// shows is kept here with sequence numbers, so any number of browsers can
// watch and each asks only for what it has not seen.
//
//   GET  /                  the page
//   GET  /api/status?b=&d=&c=&w=&fmt=   state, plus bursts, dumps, constellation, waterfall rows after those seqs
//   GET  /api/settings      the saved settings, their choices, devices, rig models
//   POST /api/settings      {key: value, ...} over the running settings; saved, station restarted if changed
//   POST /api/abort         the Abort button
//   POST /api/monitor       {"on": bool}
//
// POSTs must be application/json: a page on another site can't send that
// without a CORS preflight, which is never answered.
#pragma once

#include <QHostAddress>
#include <QJsonObject>
#include <QObject>
#include <QTcpServer>

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>

#include "app/station.hpp"
#include "monitor/monitor.hpp"

class QSettings;
class QTcpSocket;

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
    };
    // One request, as the socket handler sees it; public for the test.
    Reply handle(const QByteArray& method, const QByteArray& target, const QByteArray& content_type, const QByteArray& body);

private:
    void read(QTcpSocket* s);
    QJsonObject status(const QByteArray& query) const;
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
