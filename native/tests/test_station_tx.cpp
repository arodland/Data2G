// The Station's transmit path on its own thread: with the receive loop stalled (Args::rx_stall_ms, a test hook:
// a busy machine, a slow decode) a transmission still reaches the sound card without a hole, ABORT stops one
// at once, and PTT comes up and goes down around it. Over pipe audio (no device, no rig), like the GUI test.

#include <QCoreApplication>
#include <QElapsedTimer>
#include <QHostAddress>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QThread>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <vector>

#include "app/station.hpp"
#include "check.hpp"
#include "generated/config.hpp"

using namespace data2g;

namespace {

bool wait_for(const std::function<bool()>& cond, int ms) {
    QElapsedTimer t;
    t.start();
    while (!cond()) {
        if (t.elapsed() > ms) return false;
        QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
        QThread::msleep(5);
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

std::vector<float> read_floats(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    std::vector<float> v(static_cast<std::size_t>(f.tellg()) / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    return v;
}

struct Burst {
    std::size_t first = 0, last = 0, longest_hole = 0;  // sample indices of the sound; the longest run of exact zeros inside it
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

double full_seconds = 0;  // how long the whole burst plays, from the first run

// One station, one command: CQ (a burst of a couple of seconds), with the receive loop stalled 250 ms a block.
void run(bool abort_midway) {
    check::current_step = abort_midway ? "abort a transmission" : "a transmission with a stalled receive loop";
    QTemporaryDir dir;
    const std::string in = dir.filePath(QStringLiteral("in.f32")).toStdString();
    const std::string out = dir.filePath(QStringLiteral("out.f32")).toStdString();
    std::ofstream(in, std::ios::binary).close();  // no input: silence at real time

    app::Args a;
    a.mycall = "N0TX";
    a.audio_io = "pipe:" + in + "," + out;
    a.rigctld_port = 0;
    a.record_dir = "";
    a.command_port = free_port_triple();
    a.kiss_port = a.command_port + 2;
    a.rx_stall_ms = 250;  // 2.5x a block: the old block-by-block feeding would have starved the card at once
    app::Station st(a);
    st.start();

    QTcpSocket client;
    client.connectToHost(QHostAddress::LocalHost, static_cast<quint16>(a.command_port));
    check::is_true(client.waitForConnected(5000), "client connected");
    client.write("LISTEN ON\r");
    client.write("CQFRAME N0TX 500\r");

    check::is_true(wait_for([&] { return st.ptt(); }, 20000), "PTT up");
    QElapsedTimer keyed;
    keyed.start();
    if (abort_midway) {
        // Early: the TX thread keeps up to half a second queued ahead of the card and the abort waits out a stalled
        // receive block, so a late ABORT (slow runner) finds the whole ~2 s burst already handed over.
        wait_for([] { return false; }, 400);
        QElapsedTimer t;
        t.start();
        client.write("ABORT\r");
        check::is_true(wait_for([&] { return !st.ptt(); }, 5000), "PTT down after ABORT");
        check::is_true(t.elapsed() < 2500, "ABORT released PTT within 2.5 s");
    } else {
        check::is_true(wait_for([&] { return !st.ptt(); }, 60000), "PTT down at the end");
    }
    const double keyed_s = keyed.elapsed() / 1000.0;
    wait_for([] { return false; }, 300);  // the pipe thread writes its last period
    st.stop();

    const Burst b = analyse(read_floats(out));
    check::is_true(b.any, "sound reached the output");
    const double seconds = static_cast<double>(b.last - b.first) / config::FS;
    std::printf("  %s: %.2f s of sound, PTT up %.2f s, longest zero run inside %.1f ms\n", abort_midway ? "abort" : "stalled rx", seconds,
                keyed_s, 1000.0 * static_cast<double>(b.longest_hole) / config::FS);
    if (!abort_midway) {
        full_seconds = seconds;
        check::is_true(seconds > 1.5, "a whole burst was played");
        // exact zeros inside a burst are a starved card; a modulated burst has none this long
        check::is_true(b.longest_hole < static_cast<std::size_t>(0.01 * config::FS), "no hole in the transmission");
    } else {
        check::is_true(full_seconds > 0 && seconds < 0.8 * full_seconds, "the transmission was cut short");
    }
}

}  // namespace

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    check::Watchdog dog(300, "test_station_tx");
    run(false);
    run(true);
    return check::report("test_station_tx");
}
