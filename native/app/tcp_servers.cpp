#include "tcp_servers.hpp"

#include <QTcpSocket>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "app/log.hpp"

namespace data2g::app {

Port::Port(const QHostAddress& addr, quint16 port, bool lines, std::function<void(const QByteArray&, std::uint64_t)> on_input,
           std::function<void()> on_close, bool multi)
    : lines_(lines), multi_(multi), on_input_(std::move(on_input)), on_close_(std::move(on_close)) {
    if (!srv_.listen(addr, port))
        throw std::runtime_error("can't listen on " + addr.toString().toStdString() + ":" + std::to_string(port) + ": " +
                                 srv_.errorString().toStdString());
    connect(&srv_, &QTcpServer::newConnection, this, [this] { accept(); });
}

// Its sockets are srv_'s children, closed after the members their
// handlers use are gone: unhook them first.
Port::~Port() {
    for (auto* c : srv_.findChildren<QTcpSocket*>()) c->disconnect(this);
}

std::string Port::peer(std::uint64_t id) const {
    for (const auto& [c, cl] : clients_) {
        if (cl.id != id) continue;
        bool v4 = false;
        const QHostAddress a = c->peerAddress();
        const quint32 ip = a.toIPv4Address(&v4);
        return (v4 ? QHostAddress(ip) : a).toString().toStdString();
    }
    return "127.0.0.1";
}

void Port::send(const QByteArray& data, std::uint64_t to) {
    if (data.isEmpty()) return;
    for (auto& [c, cl] : clients_)
        if (to == 0 || cl.id == to) c->write(data);
}

void Port::accept() {
    while (QTcpSocket* c = srv_.nextPendingConnection()) {
        logf(INFO, "client %s:%d on port %d", c->peerAddress().toString().toStdString().c_str(), c->peerPort(), srv_.serverPort());
        if (!multi_) {
            // the new client first: the old one's close isn't a loss
            auto old = std::move(clients_);
            clients_.clear();
            for (auto& [o, _] : old) {
                o->disconnect(this);
                o->close();
                o->deleteLater();
            }
        }
        clients_[c].id = ++last_id_;
        connect(c, &QTcpSocket::readyRead, this, [this, c] { read(c); });
        connect(c, &QTcpSocket::disconnected, this, [this, c] {
            c->deleteLater();
            if (clients_.erase(c) && clients_.empty() && on_close_) on_close_();
        });
    }
}

void Port::read(QTcpSocket* c) {
    const QByteArray d = c->readAll();
    Client& cl = clients_[c];
    if (!lines_) {
        on_input_(d, cl.id);
        return;
    }
    QByteArray& buf = cl.buf;
    buf += d;
    while (true) {
        const qsizetype r = buf.indexOf('\r'), n = buf.indexOf('\n');
        const qsizetype i = r < 0 ? n : (n < 0 ? r : std::min(r, n));
        if (i < 0) break;
        const QByteArray line = buf.left(i);
        buf.remove(0, i + 1);
        if (!line.trimmed().isEmpty()) on_input_(line, cl.id);
    }
}

KissServer::KissServer(const QHostAddress& addr, quint16 port,
                       std::function<void(int, tnc::Bytes, std::optional<std::int64_t>)> on_packet,
                       std::function<void(int, tnc::Bytes)> on_command)
    : on_packet_(std::move(on_packet)), on_command_(std::move(on_command)) {
    if (!srv_.listen(addr, port))
        throw std::runtime_error("can't listen on " + addr.toString().toStdString() + ":" + std::to_string(port) + ": " +
                                 srv_.errorString().toStdString());
    connect(&srv_, &QTcpServer::newConnection, this, [this] {
        while (QTcpSocket* c = srv_.nextPendingConnection()) {
            const std::string peer = c->peerAddress().toString().toStdString() + ":" + std::to_string(c->peerPort());
            clients_[c] = {};
            logf(INFO, "KISS client %s connected", peer.c_str());
            connect(c, &QTcpSocket::readyRead, this, [this, c] {
                const QByteArray d = c->readAll();
                auto it = clients_.find(c);
                if (it == clients_.end()) return;
                const auto* p = reinterpret_cast<const std::uint8_t*>(d.constData());
                for (auto& [cmd, payload] : it->second.feed({p, static_cast<std::size_t>(d.size())})) {
                    if ((cmd & 0x0F) == tnc::KISS_DATA) {
                        on_packet_(cmd >> 4, std::move(payload), std::nullopt);
                    } else if ((cmd & 0x0F) == tnc::KISS_ACKMODE && payload.size() >= 2) {  // [tag, 2][frame]
                        const std::int64_t id = ++n_acks_;
                        acks_[id] = {c, tnc::Bytes(payload.begin(), payload.begin() + 2)};
                        on_packet_(cmd >> 4, tnc::Bytes(payload.begin() + 2, payload.end()), id);
                    } else {
                        on_command_(cmd & 0x0F, std::move(payload));
                    }
                }
            });
            connect(c, &QTcpSocket::disconnected, this, [this, c, peer] {
                clients_.erase(c);
                std::erase_if(acks_, [c](const auto& kv) { return kv.second.first == c; });
                c->deleteLater();
                logf(INFO, "KISS client %s disconnected", peer.c_str());
            });
        }
    });
}

KissServer::~KissServer() {  // as ~Port
    for (auto* c : srv_.findChildren<QTcpSocket*>()) c->disconnect(this);
}

void KissServer::broadcast(const tnc::Bytes& data, int port) {
    const auto f = tnc::kiss_encode(data, port);
    const QByteArray b(reinterpret_cast<const char*>(f.data()), static_cast<qsizetype>(f.size()));
    for (auto& [c, _] : clients_) c->write(b);
}

void KissServer::send_ack(std::int64_t id, int port) {
    const auto it = acks_.find(id);
    if (it == acks_.end()) return;  // its client has gone
    const auto f = tnc::kiss_encode(it->second.second, port, tnc::KISS_ACKMODE);
    if (clients_.count(it->second.first))
        it->second.first->write(QByteArray(reinterpret_cast<const char*>(f.data()), static_cast<qsizetype>(f.size())));
    acks_.erase(it);
}

void KissServer::close_clients() {
    // disconnectFromHost() can emit disconnected synchronously, which erases from clients_
    std::vector<QTcpSocket*> cs;
    for (auto& [c, _] : clients_) cs.push_back(c);
    for (auto* c : cs) c->disconnectFromHost();
}

}  // namespace data2g::app
