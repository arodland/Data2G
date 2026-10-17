// The station's TCP listeners, on the owner's (Qt event loop's) thread:
// Port is host.py's _Port (the VARA command and data ports), KissServer is
// tnc.py's KissServer. Each hands what its clients send to a callback and
// knows nothing of the engine; Station wires them to it.
#pragma once

#include <QByteArray>
#include <QHostAddress>
#include <QObject>
#include <QTcpServer>
class QTcpSocket;

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include "tnc/tnc.hpp"

namespace data2g::app {

// One TCP listener (host.py's _Port). With `multi` any number of clients may be connected: output goes to all of
// them and on_close fires when the last one leaves. Otherwise a new client replaces the old, whose close then
// isn't a loss. `lines`: input is split into lines (CR or LF, blank ones dropped), else passed on as it comes.
class Port : public QObject {
public:
    // Throws std::runtime_error if it can't listen.
    Port(const QHostAddress& addr, quint16 port, bool lines, std::function<void(const QByteArray&, std::uint64_t)> on_input,
         std::function<void()> on_close = {}, bool multi = false);
    ~Port() override;

    // Client `id`'s address (127.0.0.1 if it has gone), IPv4 without its IPv6 mapping.
    std::string peer(std::uint64_t id) const;
    // To every client, or to one by id (nowhere if it has gone).
    void send(const QByteArray& data, std::uint64_t to = 0);

private:
    void accept();
    void read(QTcpSocket* c);

    QTcpServer srv_;
    struct Client {
        std::uint64_t id = 0;
        QByteArray buf;  // its partial line
    };
    std::map<QTcpSocket*, Client> clients_;
    std::uint64_t last_id_ = 0;
    bool lines_, multi_;
    std::function<void(const QByteArray&, std::uint64_t)> on_input_;
    std::function<void()> on_close_;
};

// tnc.py's KissServer: any number of clients; data and ACKMODE frames to
// on_packet (port, frame, ack id), other KISS commands to on_command; frames
// heard go to every client, an ack to the client that asked for it.
class KissServer : public QObject {
public:
    // Throws std::runtime_error if it can't listen.
    KissServer(const QHostAddress& addr, quint16 port, std::function<void(int, tnc::Bytes, std::optional<std::int64_t>)> on_packet,
               std::function<void(int, tnc::Bytes)> on_command);
    ~KissServer() override;

    void broadcast(const tnc::Bytes& data, int port);
    // An ACKMODE frame went out: its tag back to the client that sent it.
    void send_ack(std::int64_t id, int port);
    void close_clients();

private:
    QTcpServer srv_;
    std::map<QTcpSocket*, tnc::KissDecoder> clients_;
    std::map<std::int64_t, std::pair<QTcpSocket*, tnc::Bytes>> acks_;  // ACKMODE id -> (client, tag)
    std::int64_t n_acks_ = 0;
    std::function<void(int, tnc::Bytes, std::optional<std::int64_t>)> on_packet_;
    std::function<void(int, tnc::Bytes)> on_command_;
};

}  // namespace data2g::app
