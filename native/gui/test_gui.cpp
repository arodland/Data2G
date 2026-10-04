// data2g-gui's window, offscreen, over a real Station with pipe audio: a CQ
// burst synthesized by a second Engine is fed in as the input file, and the
// burst log, mode, BUSY and link-state labels must follow. Also: settings
// round-trip through QSettings (a temp ini) and through the dialog.
// Saves a screenshot to DATA2G_GUI_SHOT.

#include <QApplication>
#include <QSlider>
#include <QElapsedTimer>
#include <QLabel>
#include <QSettings>
#include <QTableWidget>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QThread>

#include <fstream>
#include <functional>
#include <random>

#include "arq/engine.hpp"
#include "check.hpp"
#include "generated/config.hpp"
#include "main_window.hpp"
#include "waterfall.hpp"
#include "settings_dialog.hpp"

using namespace data2g;
namespace check = data2g::check;

namespace {

bool wait_for(const std::function<bool()>& cond, int ms) {
    QElapsedTimer t;
    t.start();
    while (!cond()) {
        if (t.elapsed() > ms) return false;
        QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
        QThread::msleep(10);
    }
    return true;
}

// A port with the next one free too (VARA's command and data ports).
int free_port_pair() {
    for (int i = 0; i < 100; ++i) {
        QTcpServer a, b;
        if (!a.listen(QHostAddress::LocalHost, 0)) continue;
        if (a.serverPort() < 65535 && b.listen(QHostAddress::LocalHost, a.serverPort() + 1)) return a.serverPort();
    }
    return 0;
}

// A CQ frame as another station sends it, with a second of silence either side.
std::vector<float> cq_burst(std::string* mode) {
    arq::EngineConfig cfg;
    cfg.seed = 1;
    arq::Engine tx("N0CQ", cfg);
    tx.send_cq("N0CQ", 2);
    std::vector<float> out(config::FS, 0.0f);
    bool keyed = false;
    for (int i = 0; i < 400; ++i) {
        const auto o = tx.step(std::vector<double>(config::FS / 10, 0.0));
        if (o.ptt && tx.tx()) *mode = tx.tx()->burst->submode;
        if (o.ptt) out.insert(out.end(), o.audio.begin(), o.audio.end());
        if (keyed && !o.ptt) break;
        keyed = o.ptt;
    }
    out.insert(out.end(), config::FS, 0.0f);
    return out;
}

app::Args persisted() {
    app::Args a;
    a.mycall = "W1AW";
    a.input_device = "USB Audio";
    a.output_device = "Gone Card";  // not in the dialog's list: kept, marked
    a.sample_rate = 96000;
    a.output_volume = -6.5;
    a.rigctld_host = "radio.lan";
    a.rigctld_port = 4533;
    a.ptt_on_delay_ms = 150;
    a.ptt_off_delay_ms = 70;
    a.rig_model = 3073;  // IC-7300: serial, so the serial fields show
    a.rig_device = "/dev/ttyUSB1";
    a.rig_baud = 19200;
    a.rig_data_bits = "8";
    a.rig_stop_bits = "2";
    a.rig_parity = "even";
    a.rig_handshake = "hardware";
    a.rig_dtr = "high";
    a.rig_rts = "low";
    a.ptt_method = "rts";
    a.ptt_device = "/dev/ttyUSB2";
    a.ptt_audio = "data";
    a.rig_mode = "pkt_usb";
    a.rig_timeout_ms = 750;
    a.rig_retries = 3;
    a.rig_poll_interval = 2.5;
    a.rig_debug = true;
    a.vara = false;
    a.host = "0.0.0.0";
    a.command_port = 8400;
    a.kiss = false;
    a.kiss_address = "0.0.0.0";
    a.kiss_port = 8101;
    a.decode_worker = false;
    a.noise_rule = 0.5;
    return a;
}

// The noise rule's slider: 0.00-1.50, a detent at 1.00 while dragging, 0 off;
// a weight past its top (a command line's) is kept when it isn't touched.
void test_noise_rule_slider() {
    check::current_step = "noise rule slider";
    app::Args a;
    gui::SettingsDialog d(a, {}, {});
    auto* s = d.findChild<QSlider*>(QStringLiteral("noise_rule"));
    auto* label = d.findChild<QLabel*>(QStringLiteral("noise_rule_value"));
    check::is_true(s && label, "slider and its value shown");
    check::is_true(s->minimum() == 0 && s->maximum() == 150 && s->value() == 100, "0-150 hundredths, at the default");
    check::equal(label->text().toStdString(), std::string("1.00 (default)"), "the default named");
    Q_EMIT s->sliderMoved(97);
    check::is_true(s->value() == 100, "dragged near 1.00: the detent holds it");
    s->setValue(90);
    Q_EMIT s->sliderMoved(90);
    check::is_true(s->value() == 90, "past the detent: free");
    s->setValue(37);
    app::Args e;
    d.apply_to(e);
    check::is_true(e.noise_rule == 0.37, "0.37 applied");
    s->setValue(0);
    check::equal(label->text().toStdString(), std::string("off"), "0 is off");
    d.apply_to(e);
    check::is_true(e.noise_rule == 0.0, "0 applied");
    a.noise_rule = 2.0;
    gui::SettingsDialog high(a, {}, {});
    high.apply_to(e);
    check::is_true(e.noise_rule == 2.0, "a weight past the slider's top kept when untouched");
}

void test_settings(const QTemporaryDir& dir) {
    check::current_step = "settings";
    const QString ini = dir.filePath(QStringLiteral("settings.ini"));
    const app::Args a = persisted();
    {
        QSettings s(ini, QSettings::IniFormat);
        gui::save_settings(s, a);
    }
    QSettings s(ini, QSettings::IniFormat);
    app::Args b = gui::load_settings(s);
    b.record_dir = a.record_dir;  // not persisted; a timestamp
    check::is_true(b == a, "settings round trip through QSettings");
    // unset optionals come back unset
    app::Args none;
    none.record_dir = a.record_dir;
    gui::save_settings(s, none);
    app::Args c = gui::load_settings(s, a);
    c.record_dir = a.record_dir;
    check::is_true(c == none, "cleared callsign and devices round trip");

    gui::SettingsDialog d(a, {QStringLiteral("Built-in"), QStringLiteral("USB Audio")}, {QStringLiteral("Built-in")});
    app::Args e;
    e.record_dir = a.record_dir;
    d.apply_to(e);
    check::is_true(e == a, "settings round trip through the dialog");

    // model 2 at rigctld's address stays that (no device), and a saved
    // 'no rig' stays off; neither opens anything
    app::Args r;
    r.record_dir = a.record_dir;
    r.rigctld_host = "radio.lan";
    r.rigctld_port = 4533;
    for (bool on : {true, false}) {
        r.rig = on;
        gui::SettingsDialog rd(r, {}, {});
        app::Args f;
        f.record_dir = a.record_dir;
        rd.apply_to(f);
        check::is_true(f == r, on ? "rigctld settings round trip through the dialog" : "rig off round trips");
    }
    // legacy settings (no rig keys): rigctld at their host and port
    QSettings legacy(dir.filePath(QStringLiteral("legacy.ini")), QSettings::IniFormat);
    legacy.setValue("rigctld_host", QStringLiteral("old.lan"));
    legacy.setValue("rigctld_port", 4534);
    const app::Args l = gui::load_settings(legacy);
    check::is_true(app::rig_enabled(l) && app::rig_device(l) == "old.lan:4534", "legacy rigctld settings: model 2 there");
}

void test_window(const QTemporaryDir& dir) {
    check::current_step = "window: synthesize a CQ";
    std::string cq_mode;
    const auto burst = cq_burst(&cq_mode);
    check::is_true(!cq_mode.empty() && burst.size() > 2u * config::FS, "a CQ burst synthesized");
    const std::string in = dir.filePath(QStringLiteral("in.f32")).toStdString();
    const std::string out = dir.filePath(QStringLiteral("out.f32")).toStdString();
    std::ofstream(in, std::ios::binary).write(reinterpret_cast<const char*>(burst.data()),
                                               static_cast<std::streamsize>(burst.size() * sizeof(float)));

    app::Args a;
    a.mycall = "N0GUI";
    a.audio_io = "pipe:" + in + "," + out;
    a.rigctld_port = 0;  // no rig
    a.record_dir = "";
    a.kiss = false;
    a.command_port = free_port_pair();
    QSettings store(dir.filePath(QStringLiteral("window.ini")), QSettings::IniFormat);
    check::current_step = "window: start the station";
    gui::MainWindow w(a, store);
    w.show();
    check::is_true(w.station().running(), "station running");
    auto* link = w.findChild<QLabel*>(QStringLiteral("link_state"));
    auto* mode = w.findChild<QLabel*>(QStringLiteral("mode"));
    auto* busy = w.findChild<QLabel*>(QStringLiteral("busy"));
    auto* log = w.findChild<QTableWidget*>(QStringLiteral("burst_log"));
    check::is_true(link && mode && busy && log, "labels found");
    if (!(link && mode && busy && log)) return;
    w.poll();
    check::equal(link->text().toStdString(), std::string("idle"), "link idle at start");
    auto* dial = w.findChild<QLabel*>(QStringLiteral("dial"));
    check::is_true(dial && dial->isHidden(), "no dial frequency unless the rig is polled");

    check::current_step = "window: VARA client";
    QTcpSocket client;
    client.connectToHost(QHostAddress::LocalHost, static_cast<quint16>(a.command_port));
    check::is_true(client.waitForConnected(5000), "VARA client connected");
    client.write("LISTEN ON\r");
    check::is_true(wait_for([&] { return link->text() == QStringLiteral("listening"); }, 5000), "link listening");

    check::current_step = "window: hear the CQ";
    bool busy_seen = false;
    const bool heard = wait_for(
        [&] {
            busy_seen |= busy->property("lit").toBool();
            return log->rowCount() > 0;
        },
        30000);
    check::is_true(heard, "a burst in the log");
    check::is_true(busy_seen, "BUSY lit while it arrived");
    if (heard) {
        check::is_true(log->item(0, 1)->text().startsWith(QString::fromStdString(cq_mode)), "logged in the CQ's mode");
        check::equal(log->item(0, 3)->text().toStdString(), std::string("ok"), "logged ok");
        check::is_true(log->item(0, 2)->text() != QStringLiteral("-"), "an SNR logged");
        check::is_true(mode->text().startsWith(QString::fromStdString(cq_mode)), "mode label shows it");
    }
    QByteArray said;
    check::is_true(wait_for([&] {
                       said += client.readAll();
                       return said.contains("CQFRAME N0CQ 2300");
                   }, 5000),
                   "the client heard CQFRAME");
    const app::AudioCounters c = w.station().counters();
    check::is_true(c.overflows == 0 && c.dropped == 0 && c.underruns == 0 && c.decode_dropped == 0, "no audio faults");

    // our own CQ: its audio goes on the waterfall (TX colours) while PTT is up
    check::current_step = "window: transmit";
    auto* ptt = w.findChild<QLabel*>(QStringLiteral("ptt"));
    client.write("CQFRAME N0GUI 500\r");
    bool keyed = false, tapped_tx = false;
    const bool sent = wait_for(
        [&] {
            if (ptt && ptt->property("lit").toBool()) {
                keyed = true;
                bool tx = false;
                w.station().input_tail(1, nullptr, &tx);
                tapped_tx |= tx;
            }
            return keyed && ptt && !ptt->property("lit").toBool();
        },
        30000);
    check::is_true(sent, "PTT up for our CQ, then down");
    check::is_true(tapped_tx, "the waterfall was fed our TX audio while keyed");

    check::current_step = "window: screenshot";
    check::is_true(w.grab().save(QStringLiteral(DATA2G_GUI_SHOT)), "screenshot saved");

    // a restart on new settings listens again on the same ports
    check::current_step = "window: restart";
    a.mycall = "N1GUI";
    w.restart(a);
    check::is_true(w.station().running(), "station restarted");
    QTcpSocket again;
    again.connectToHost(QHostAddress::LocalHost, static_cast<quint16>(a.command_port));
    check::is_true(again.waitForConnected(5000), "command port open after the restart");
    check::current_step = "window: close";
}

// TX audio goes on the waterfall in its own colours and stays off the meter:
// full-scale noise as TX, then as RX.
void test_waterfall_tx() {
    check::current_step = "waterfall TX";
    gui::Waterfall wf(nullptr, 1);
    wf.resize(200, 100);
    std::uint64_t total = 0;
    bool as_tx = true;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    wf.set_source([&](std::size_t n, std::uint64_t* t, bool* tx) {
        std::vector<double> x(n);
        for (auto& v : x) v = u(rng);
        total += config::FS / 10;
        *t = total;
        *tx = as_tx;
        return x;
    });
    const auto top_pixel = [&] { return wf.grab().toImage().pixelColor(50, 0); };

    wf.tick();
    const QColor tx = top_pixel();
    check::is_true(tx.red() > tx.green() + 50 && tx.blue() > tx.green(), "TX row in TX colours (magenta, not RX's ramp)");
    check::equal(wf.peak(), 0.0, "TX audio stays off the input meter");
    check::is_true(!wf.clip_latched(), "full-scale TX audio doesn't latch CLIP");

    as_tx = false;
    wf.tick();
    const QColor rx = top_pixel();
    check::is_true(rx.green() > rx.blue() + 100, "RX row back on the RX ramp");
    check::is_true(wf.peak() > 0.9 && wf.clip_latched(), "full-scale input does latch CLIP");
}

}  // namespace

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");  // never on the user's display
    qputenv("QT_FORCE_STDERR_LOGGING", "1");   // Windows: a qFatal to stderr, not OutputDebugString
    check::report_crashes_instead_of_prompting();
    check::Watchdog dog(100, "gui");  // under ctest's TIMEOUT 120, so it names the step
    check::current_step = "QApplication";
    QApplication app(argc, argv);
    QTemporaryDir dir;
    test_settings(dir);
    test_noise_rule_slider();
    test_waterfall_tx();
    test_window(dir);
    return check::report("gui");
}
