// The station over TCI (--tci) against a fake server in this process: the setup it sends after READY, the
// server's RX audio reaching the engine, the dial from VFO, and a CQ keyed with TRX:0,true,tci whose audio goes
// out in answer to TX_CHRONO without a hole, then TRX:0,false. Run again with a server that sends no TX_CHRONO
// (AetherSDR): the audio is pushed on the station's own clock.

#include <QCoreApplication>
#include <QElapsedTimer>
#include <QHostAddress>
#include <QTcpServer>
#include <QTcpSocket>
#include <QThread>
#include <QTimer>
#include <QWebSocket>
#include <QWebSocketServer>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

#include "app/station.hpp"
#include "check.hpp"

using namespace data2g;

namespace {

bool wait_for(const std::function<bool()>& cond, int ms) {
    QElapsedTimer t;
    t.start();
    while (!cond()) {
        if (t.elapsed() > ms) return false;
        QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
        QThread::msleep(2);
    }
    return true;
}

int free_port_triple() {
    for (int i = 0; i < 100; ++i) {
        QTcpServer a, b, c;
        if (!a.listen(QHostAddress::LocalHost, 0)) continue;
        const int p = a.serverPort();
        if (p < 65534 && b.listen(QHostAddress::LocalHost, p + 1) && c.listen(QHostAddress::LocalHost, p + 2)) return p;
    }
    return 0;
}

std::array<std::uint32_t, 16> header(const QByteArray& f) {
    std::array<std::uint32_t, 16> h;
    std::memcpy(h.data(), f.constData(), sizeof h);
    return h;
}

// A Stream frame: receiver 0, float32, mono.
QByteArray frame(std::uint32_t type, int rate, const std::vector<float>& x, std::uint32_t length) {
    QByteArray f(64 + static_cast<qsizetype>(x.size() * sizeof(float)), '\0');
    const std::uint32_t h[8] = {0, static_cast<std::uint32_t>(rate), 3, 0, 0, length, type, 1};
    std::memcpy(f.data(), h, sizeof h);
    if (!x.empty()) std::memcpy(f.data() + 64, x.data(), x.size() * sizeof(float));
    return f;
}

// ExpertSDR's side, as much as the station uses: a 1 kHz tone as RX audio every 40 ms and, while keyed by TCI,
// a TX_CHRONO for as many samples (unless `chrono` is off); TX audio collected.
struct FakeServer {
    QWebSocketServer srv{QStringLiteral("fake"), QWebSocketServer::NonSecureMode};
    QWebSocket* client = nullptr;
    QStringList got;
    std::vector<float> tx;
    bool chrono = true, keyed = false, bad_tx = false;
    int rate = 48000;
    double phase = 0;
    QTimer timer;

    FakeServer() {
        srv.listen(QHostAddress::LocalHost, 0);
        QObject::connect(&srv, &QWebSocketServer::newConnection, [this] {
            client = srv.nextPendingConnection();
            QObject::connect(client, &QWebSocket::textMessageReceived, [this](const QString& m) { text(m); });
            QObject::connect(client, &QWebSocket::binaryMessageReceived, [this](const QByteArray& f) { binary(f); });
            client->sendTextMessage(QStringLiteral("protocol:ExpertSDR3,2.1;device:SunSDR2PRO;vfo:0,0,14105000;"));
            client->sendTextMessage(QStringLiteral("ready;"));
        });
        QObject::connect(&timer, &QTimer::timeout, [this] { period(); });
        timer.setTimerType(Qt::PreciseTimer);
        timer.start(40);
    }
    int port() const { return srv.serverPort(); }

    void text(const QString& m) {
        got << m;
        if (m.startsWith(QLatin1String("trx:0,true,tci"))) keyed = true;
        if (m.startsWith(QLatin1String("trx:0,false"))) keyed = false;
        if (m.startsWith(QLatin1String("audio_samplerate:"))) rate = m.mid(17).chopped(1).toInt();
    }
    void binary(const QByteArray& f) {
        if (f.size() < 64) return;
        const auto h = header(f);
        if (h[6] != 2) return;
        const std::uint32_t length = h[5], ch = h[7];
        if (ch != 1 || h[2] != 3 || static_cast<int>(h[1]) != rate ||
            f.size() != 64 + static_cast<qsizetype>(length * sizeof(float)))
            bad_tx = true;
        const auto* d = reinterpret_cast<const float*>(f.constData() + 64);
        tx.insert(tx.end(), d, d + (f.size() - 64) / static_cast<qsizetype>(sizeof(float)));
    }
    void period() {
        if (!client) return;
        const std::size_t n = static_cast<std::size_t>(rate / 25);
        std::vector<float> x(n);
        for (auto& v : x) {
            v = static_cast<float>(0.1 * std::sin(phase));
            phase += 2 * M_PI * 1000 / rate;
        }
        client->sendBinaryMessage(frame(1, rate, x, static_cast<std::uint32_t>(n)));
        if (keyed && chrono) client->sendBinaryMessage(frame(3, rate, {}, static_cast<std::uint32_t>(n)));
    }
};

// The first and last sample of sound in `x`, and the longest run of exact zeros between them.
struct Burst {
    std::size_t first = 0, last = 0, longest_hole = 0;
    bool any = false;
};

Burst analyse(const std::vector<float>& x) {
    Burst b;
    for (std::size_t i = 0; i < x.size(); ++i)
        if (std::abs(x[i]) > 1e-4f) {
            if (!b.any) b.first = i;
            b.last = i;
            b.any = true;
        }
    std::size_t run = 0;
    for (std::size_t i = b.first; b.any && i <= b.last; ++i) {
        run = x[i] == 0.0f ? run + 1 : 0;
        b.longest_hole = std::max(b.longest_hole, run);
    }
    return b;
}

void run(bool chrono) {
    check::current_step = chrono ? "TCI, server-paced (TX_CHRONO)" : "TCI, no TX_CHRONO (pushed)";
    FakeServer server;
    server.chrono = chrono;
    app::Args a;
    a.mycall = "N0TX";
    a.tci = true;
    a.tci_host = "127.0.0.1:" + std::to_string(server.port());
    a.rig_mode = "pkt_usb";
    a.command_port = free_port_triple();
    a.kiss_port = a.command_port + 2;
    app::Station st(a);
    st.start();

    check::is_true(wait_for([&] { return server.got.contains(QStringLiteral("audio_start:0;")); }, 5000), "audio started");
    for (const char* c : {"audio_samplerate:48000;", "audio_stream_sample_type:float32;", "audio_stream_channels:1;",
                          "modulation:0,digu;"})
        check::is_true(server.got.contains(QLatin1String(c)), c);
    check::is_true(wait_for([&] { return st.rig_frequency() == 14105000.0; }, 2000), "dial from VFO");
    check::is_true(wait_for(
                       [&] {
                           const auto x = st.input_tail(800);
                           return !x.empty() && *std::max_element(x.begin(), x.end()) > 0.05;
                       },
                       5000),
                   "RX audio reached the engine");

    QTcpSocket client;
    client.connectToHost(QHostAddress::LocalHost, static_cast<quint16>(a.command_port));
    check::is_true(client.waitForConnected(5000), "client connected");
    client.write("LISTEN ON\r");
    client.write("CQFRAME N0TX 500\r");
    check::is_true(wait_for([&] { return st.ptt(); }, 20000), "PTT up");
    check::is_true(wait_for([&] { return server.keyed; }, 2000), "keyed by trx:0,true,tci");
    check::is_true(wait_for([&] { return !st.ptt(); }, 60000), "PTT down at the end");
    check::is_true(wait_for([&] { return !server.keyed; }, 2000), "released by trx:0,false");
    st.stop();

    check::is_true(!server.bad_tx, "TX frames: float32, mono, at our rate, length matching");
    const Burst b = analyse(server.tx);
    const double seconds = static_cast<double>(b.last - b.first) / server.rate;
    std::printf("  %s: %.2f s of sound, longest zero run inside %.1f ms\n", chrono ? "chrono" : "pushed", seconds,
                1000.0 * static_cast<double>(b.longest_hole) / server.rate);
    check::is_true(b.any && seconds > 1.5, "a whole burst was sent");
    check::is_true(b.longest_hole < static_cast<std::size_t>(0.01 * server.rate), "no hole in the transmission");
}

}  // namespace

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    check::Watchdog dog(240, "test_tci");
    run(true);
    run(false);
    return check::report("test_tci");
}
