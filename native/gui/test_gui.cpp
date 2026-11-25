// data2g-gui's window, offscreen, over a real Station with pipe audio: a CQ
// burst synthesized by a second Engine is fed in as the input file, and the
// burst log, mode, BUSY and link-state labels must follow. Also: settings
// round-trip through QSettings (a temp ini) and through the dialog.
// Saves a screenshot to DATA2G_GUI_SHOT.

#include <QApplication>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QSlider>
#include <QComboBox>
#include <QElapsedTimer>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenuBar>
#include <QPlainTextEdit>
#include <QScrollBar>
#include <QSettings>
#include <QTableWidget>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTextBlock>
#include <QTextDocument>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>
#include <QThread>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <random>

#include "arq/engine.hpp"
#include "arq/modes.hpp"
#include "check.hpp"
#include "generated/config.hpp"
#include "main_window.hpp"
#include "constellation.hpp"
#include "waterfall.hpp"
#include "settings_dialog.hpp"
#include "app/settings.hpp"
#include "app/station.hpp"

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

// A port with the next two free too: VARA's command and data ports, and
// KISS's. Picked at once: macOS and Windows hand out ephemeral ports in
// sequence, so a second pick after the first was released could be its
// data port.
int free_port_triple() {
    for (int i = 0; i < 100; ++i) {
        QTcpServer a, b, c;
        if (!a.listen(QHostAddress::LocalHost, 0)) continue;
        const int p = a.serverPort();
        if (p < 65534 && b.listen(QHostAddress::LocalHost, p + 1) && c.listen(QHostAddress::LocalHost, p + 2)) return p;
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
    a.tci_host = "sdr.lan:50002";
    a.sample_rate = 96000;
    a.output_volume = -6.5;
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
    a.listen = "0.0.0.0";
    a.command_port = 8400;
    a.kiss_port = 8101;
    a.record = true;
    a.record_dir = "/srv/rec";
    a.decode_worker = false;
    a.live_constellation = false;
    a.nods = false;
    a.noise_rule = 0.5;
    a.decoder_effort = 35;
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

// The pages scroll, so the dialog shrinks to fit a 768-pixel screen (#170).
void test_settings_fit_small_screen() {
    check::current_step = "settings fit a small screen";
    gui::SettingsDialog d(app::Args{}, {}, {});
    check::is_true(d.minimumSizeHint().height() < 600, "minimum height under 600 px");
}

// Decoder effort's slider: 15-40 in steps of 5, 25 by default; a command line's cap off a step (or past the
// ends) is kept when it isn't touched.
void test_decoder_effort() {
    check::current_step = "decoder effort";
    app::Args a;
    gui::SettingsDialog d(a, {}, {});
    auto* s = d.findChild<QSlider*>(QStringLiteral("decoder_effort"));
    auto* label = d.findChild<QLabel*>(QStringLiteral("decoder_effort_value"));
    check::is_true(s && label, "slider and its value shown");
    check::is_true(s->minimum() == 3 && s->maximum() == 8 && s->value() == 5, "15-40 in steps of 5, at 25");
    check::equal(label->text().toStdString(), std::string("25 (default)"), "the default named");
    s->setValue(6);
    check::equal(label->text().toStdString(), std::string("30"), "30 shown");
    app::Args e;
    d.apply_to(e);
    check::is_true(e.decoder_effort == 30, "30 applied");
    for (int off : {33, 60}) {
        a.decoder_effort = off;
        gui::SettingsDialog keep(a, {}, {});
        keep.apply_to(e);
        check::is_true(e.decoder_effort == off, "a cap off the slider's steps kept when untouched");
    }
}

// LIVE_SETTINGS change without a restart; anything else restarts, and the dialog says so while the station runs.
void test_restart_warning() {
    check::current_step = "restart warning";
    const app::Args a;
    app::Args b = a;
    b.mycall = "N0LIVE";
    b.output_volume = -3.0;
    b.noise_rule = 0.5;
    b.decoder_effort = 30;
    b.cw_id = "add";
    b.nods = false;
    b.ptt_on_delay_ms = 200;
    check::is_true(!app::needs_restart(a, b), "live settings: no restart");
    b.record = true;
    b.record_dir = "elsewhere";
    check::is_true(!app::needs_restart(a, b), "recording: no restart");
    for (auto set : {+[](app::Args& x) { x.kiss_port = 9000; }, +[](app::Args& x) { x.decode_worker = false; },
                     +[](app::Args& x) { x.tci = true; }, +[](app::Args& x) { x.kiss_bw = 500; }}) {
        app::Args c = b;
        set(c);
        check::is_true(app::needs_restart(a, c), "anything else: a restart");
    }
    gui::SettingsDialog d(a, {}, {}, true);
    auto* note = d.findChild<QLabel*>(QStringLiteral("restart_warning"));
    check::is_true(note && note->isHidden(), "no warning at first");
    d.findChild<QDoubleSpinBox*>(QStringLiteral("output_volume"))->setValue(-6.0);
    check::is_true(note->isHidden(), "volume: no warning");
    auto* kiss = d.findChild<QSpinBox*>(QStringLiteral("kiss_port"));
    kiss->setValue(9000);
    check::is_true(!note->isHidden(), "KISS port: the warning");
    kiss->setValue(a.kiss_port);
    check::is_true(note->isHidden(), "changed back: no warning");
    gui::SettingsDialog stopped(a, {}, {}, false);
    stopped.findChild<QSpinBox*>(QStringLiteral("kiss_port"))->setValue(9000);
    check::is_true(stopped.findChild<QLabel*>(QStringLiteral("restart_warning"))->isHidden(), "a stopped station: no warning");
}

void test_settings(const QTemporaryDir& dir) {
    check::current_step = "settings";
    const QString ini = dir.filePath(QStringLiteral("settings.ini"));
    const app::Args a = persisted();
    {
        QSettings s(ini, QSettings::IniFormat);
        app::save_settings(s, a);
    }
    QSettings s(ini, QSettings::IniFormat);
    const app::Args b = app::load_settings(s);
    check::is_true(b == a, "settings round trip through QSettings");
    // unset optionals come back unset
    const app::Args none;
    app::save_settings(s, none);
    const app::Args c = app::load_settings(s, a);
    check::is_true(c == none, "cleared callsign and devices round trip");

    gui::SettingsDialog d(a, {QStringLiteral("Built-in"), QStringLiteral("USB Audio")}, {QStringLiteral("Built-in")});
    app::Args e;
    d.apply_to(e);
    check::is_true(e == a, "settings round trip through the dialog");

    // the interface picks what shows: TCI's server, or the devices and Hamlib
    {
        app::Args t = a;
        t.tci = true;
        gui::SettingsDialog td(t, {QStringLiteral("USB Audio")}, {});
        app::Args f;
        td.apply_to(f);
        check::is_true(f == t, "TCI round trips through the dialog");
        auto* iface = td.findChild<QComboBox*>(QStringLiteral("interface"));
        auto* host = td.findChild<QLineEdit*>(QStringLiteral("tci_host"));
        auto* hamlib = td.findChild<QWidget*>(QStringLiteral("hamlib"));
        auto* tci_box = td.findChild<QWidget*>(QStringLiteral("tci"));
        check::is_true(iface && host && hamlib && tci_box, "interface, TCI server and the two boxes found");
        if (iface && host && hamlib && tci_box) {
            check::is_true(!tci_box->isHidden() && hamlib->isHidden(), "TCI: the server shows, Hamlib does not");
            iface->setCurrentIndex(0);
            check::is_true(tci_box->isHidden() && !hamlib->isHidden(), "sound card: Hamlib shows, the server does not");
            host->clear();
            iface->setCurrentIndex(1);
            td.apply_to(f);
            check::is_true(f.tci && f.tci_host == "localhost:50001", "an empty server is the default");
        }
    }

    // model 2 with no device (rigctld on localhost) stays that, and a saved
    // 'no rig' stays off; neither opens anything
    app::Args r;
    for (bool on : {true, false}) {
        r.rig = on;
        gui::SettingsDialog rd(r, {}, {});
        app::Args f;
        rd.apply_to(f);
        check::is_true(f == r, on ? "model 2 default round trips through the dialog" : "rig off round trips");
    }
    // older settings: rigctld's address apart, "host" for --listen
    QSettings legacy(dir.filePath(QStringLiteral("legacy.ini")), QSettings::IniFormat);
    legacy.setValue("rigctld_host", QStringLiteral("old.lan"));
    legacy.setValue("rigctld_port", 4534);
    legacy.setValue("host", QStringLiteral("0.0.0.0"));
    const app::Args l = app::load_settings(legacy);
    check::is_true(l.rig && l.rig_device == "old.lan:4534" && l.listen == "0.0.0.0", "older settings: rigctld's address, host");
    legacy.setValue("rigctld_port", 0);
    check::is_true(!app::load_settings(legacy).rig, "older settings: rigctld port 0 is no rig");
    app::save_settings(legacy, l);
    check::is_true(!legacy.contains("rigctld_port") && !legacy.contains("host"), "...and saving drops the old keys");
}

// The profile's settings file name (app/settings.cpp): data2g.conf, FM's data2g-fm.conf.
QString conf_name() {
    return config::PROFILE == "hf" ? QStringLiteral("data2g.conf") : QStringLiteral("data2g-fm.conf");
}

// open_config: a data2g.conf beside the program before the config dir's, and
// --config's before both.
void test_config_beside_program(const QTemporaryDir& dir) {
    check::current_step = "config file beside the program";
    const QString beside = QCoreApplication::applicationDirPath() + QStringLiteral("/") + conf_name();
    if (QFile::exists(beside)) {
        check::is_true(false, "no data2g.conf beside the test before it runs");
        return;
    }
    struct Remove {
        QString path;
        ~Remove() { QFile::remove(path); }
    } cleanup{beside};
    QFile made(beside);
    check::is_true(made.open(QIODevice::WriteOnly), "the test can write beside itself");
    made.close();
    char prog[] = "data2g-gui";
    char* bare[] = {prog};
    const QString other = dir.filePath(QStringLiteral("beside-other.conf"));
    QByteArray other_arg = ("--config=" + other).toLocal8Bit();
    char* with[] = {prog, other_arg.data()};
    check::is_true(QFileInfo(app::open_config(1, bare)->fileName()) == QFileInfo(beside), "a data2g.conf beside the program is the one used");
    check::is_true(QFileInfo(app::settings_store()->fileName()) == QFileInfo(beside), "...by settings_store() too");
    check::is_true(QFileInfo(app::open_config(2, with)->fileName()) == QFileInfo(other), "--config over it");
}

// open_config: data2g.conf in the config dir, the old store moved to it once;
// --config's file instead, with nothing moved.
void test_config_file(const QTemporaryDir& dir) {
#ifdef Q_OS_LINUX  // the old store an ini file renamed; XDG_CONFIG_HOME the config dir
    check::current_step = "config file";
    QSettings::setPath(QSettings::NativeFormat, QSettings::UserScope, dir.filePath(QStringLiteral("old")));
    qputenv("XDG_CONFIG_HOME", dir.filePath(QStringLiteral("new")).toUtf8());
    QString old_file;
    {
        QSettings old(QStringLiteral("Data2G"), QStringLiteral("data2g-gui"));
        old.setValue("mycall", QStringLiteral("OLD1"));
        old_file = old.fileName();
    }
    const QString conf = dir.filePath(QStringLiteral("new/Data2G/") + conf_name());
    char prog[] = "data2g-gui";
    char* bare[] = {prog};
    const QString other = dir.filePath(QStringLiteral("other.conf"));
    QByteArray other_arg = ("--config=" + other).toLocal8Bit();
    char* with[] = {prog, other_arg.data()};
    check::is_true(app::open_config(2, with)->fileName() == other && QFile::exists(old_file) && !QFile::exists(conf),
                   "--config: its file, nothing moved");
    const auto s = app::open_config(1, bare);
    if (config::PROFILE == "hf")
        check::is_true(s->fileName() == conf && s->value("mycall").toString() == "OLD1" && !QFile::exists(old_file),
                       "data2g-gui.conf moved to data2g.conf");
    else  // the old store was HF's: FM starts its own file and leaves it be
        check::is_true(s->fileName() == conf && !s->contains("mycall") && QFile::exists(old_file),
                       "FM: its own file, the old HF store left alone");
    check::is_true(app::settings_store()->fileName() == conf, "settings_store() opens the same file");
#else
    (void)dir;
#endif
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
    a.rig = false;
    a.command_port = free_port_triple();
    a.kiss_port = a.command_port + 2;  // KISS always runs: a port of its own
    QSettings store(dir.filePath(QStringLiteral("window.ini")), QSettings::IniFormat);
    // Heights: the offscreen screen is narrower than the window can be.
    const auto saved_height = [&store] {
        QWidget probe;
        probe.restoreGeometry(store.value(QStringLiteral("gui/geometry")).toByteArray());
        return probe.height();
    };
    {
        QWidget last;  // where the window was left by the run before
        last.resize(760, 600);
        store.setValue(QStringLiteral("gui/geometry"), last.saveGeometry());
    }
    check::current_step = "window: start the station";
    gui::MainWindow w(a, store);
    w.show();
    check::is_true(w.station().running(), "station running");
    check::equal(w.height(), 600, "the window opens at the size it was left at");
    {
        // macOS: any other role and Qt moves the entry to the application menu, emptying File
        QStringList seen;
        for (const QAction* m : w.menuBar()->actions())
            for (const QAction* e : m->menu()->actions())
                if (!e->isSeparator()) {
                    seen << m->text() + QLatin1Char('>') + e->text();
                    check::is_true(e->menuRole() == QAction::NoRole, "menu entries keep NoRole");
                }
        check::equal(seen.join(QLatin1Char(' ')).toStdString(),
                     std::string("&File>&Settings... &File>E&xit &Help>&About Data2G"), "File and Help menus");
    }
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
    // monitoring on before the CQ arrives (after a second of silence), its window still closed
    auto* monitoring = w.findChild<QCheckBox*>(QStringLiteral("monitoring"));
    auto* monitor = w.findChild<QPlainTextEdit*>(QStringLiteral("monitor_text"));
    auto* monitor_format = w.findChild<QComboBox*>(QStringLiteral("monitor_format"));
    check::is_true(monitoring && !monitoring->isChecked(), "monitoring off at first");
    check::is_true(monitor && monitor_format && !monitor->isVisible(), "monitor window closed");
    if (!(monitoring && monitor && monitor_format)) return;
    monitoring->setChecked(true);
    check::is_true(store.value(QStringLiteral("gui/monitoring")).toBool(), "monitoring saved");

    check::current_step = "window: VARA client";
    QTcpSocket client;
    client.connectToHost(QHostAddress::LocalHost, static_cast<quint16>(a.command_port));
    check::is_true(client.waitForConnected(5000), "VARA client connected");
    client.write("LISTEN ON\r");
    check::is_true(wait_for([&] { return link->text() == QStringLiteral("listening"); }, 5000), "link listening");

    // a second client on the command port joins the first; neither is dropped, a reply goes to the asker only
    check::current_step = "window: second client";
    QTcpSocket second;
    second.connectToHost(QHostAddress::LocalHost, static_cast<quint16>(a.command_port));
    check::is_true(second.waitForConnected(5000), "second client connected");
    second.write("VERSION\r");
    QByteArray first_heard, second_heard;
    check::is_true(wait_for([&] {
                       first_heard += client.readAll();
                       second_heard += second.readAll();
                       return second_heard.contains("VERSION");
                   }, 5000),
                   "the asker got the reply");
    wait_for([] { return false; }, 300);
    first_heard += client.readAll();
    check::is_true(!first_heard.contains("VERSION"), "the other client did not");
    check::is_true(client.state() == QAbstractSocket::ConnectedState, "the first client is still connected");
    second.disconnectFromHost();
    wait_for([&] { return second.state() == QAbstractSocket::UnconnectedState; }, 2000);
    wait_for([] { return false; }, 200);
    check::is_true(link->text() == QStringLiteral("listening"), "one client gone, the session stays");

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
        const auto& plot = w.findChild<gui::ConstellationPlot*>(QStringLiteral("constellation"))->shown();
        check::equal(plot.mode, cq_mode, "constellation plot shows the CQ");
        check::is_true(!plot.live && !plot.lost, "the decoded burst's plot, not the live view");
        check::is_true(!plot.symbols.empty() && !plot.points.empty(), "with symbols and the mode's points");
        // a clean loopback, but our TX clipping spreads each point: 0.5 is half QPSK's spacing
        std::size_t near = 0;
        for (const auto& z : plot.symbols) {
            float d = 1e9f;
            for (const auto& q : plot.points) d = std::min(d, std::abs(z - q));
            near += d < 0.5f;
        }
        check::is_true(near > plot.symbols.size() * 9 / 10, "symbols land on the constellation");
    }
    check::is_true(wait_for([&] { return monitor->toPlainText().contains(QStringLiteral("CQ N0CQ bw 2")); }, 5000),
                   "the monitor dumped the CQ, its window closed");
    w.open_monitor();
    check::is_true(monitor->isVisible() && monitor->toPlainText().contains(QStringLiteral("CQ N0CQ bw 2")),
                   "opened, the CQ in its scrollback");
    monitor_format->setCurrentIndex(1);
    check::is_true(monitor->toPlainText().contains(QStringLiteral("CQ N0CQ bw 2")), "re-rendered as hex, the CQ kept");
    QByteArray said;
    check::is_true(wait_for([&] {
                       said += client.readAll();
                       return said.contains(QByteArray::fromStdString("CQFRAME N0CQ " + arq::bw_name(2)));
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
    monitor_format->setCurrentIndex(0);
    check::is_true(wait_for([&] { return monitor->toPlainText().contains(QStringLiteral("CQ N0GUI bw 0")); }, 5000),
                   "the monitor dumped our CQ");
    {
        // ours in a colour of its own, what was heard in the view's
        const auto colour = [&](const QString& line) {
            for (QTextBlock b = monitor->document()->begin(); b.isValid(); b = b.next())
                if (b.text().contains(line)) return b.charFormat().foreground().color();
            return QColor();
        };
        const QColor ours = colour(QStringLiteral("CQ N0GUI bw 0")), theirs = colour(QStringLiteral("CQ N0CQ bw 2"));
        check::is_true(ours.isValid() && ours != theirs, "our CQ in a colour of its own");
        check::is_true(monitor->toPlainText().contains(QStringLiteral(" TX ")), "marked TX");
    }

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

    // live settings: the client stays; anything else is the caller's restart
    check::current_step = "window: live update";
    app::Args live = w.station().args();
    live.output_volume -= 3.0;
    live.mycall = "N2GUI";
    w.apply_settings(live);
    check::is_true(w.station().args() == live, "the station's args are the new ones");
    wait_for([] { return false; }, 300);  // a dropped client would show by now
    check::is_true(again.state() == QAbstractSocket::ConnectedState, "the command client kept");
    // recording on: a run folder of its own, started where the engine is (its start event)
    live.record = true;
    live.record_dir = dir.filePath(QStringLiteral("live-rec")).toStdString();
    w.apply_settings(live);
    const auto runs = [&] {
        const QDir d(QString::fromStdString(live.record_dir));
        const auto l = d.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        return l.isEmpty() ? QString() : d.filePath(l.first());
    };
    check::is_true(wait_for([&] { return QFile::exists(runs() + QStringLiteral("/events.jsonl")); }, 5000),
                   "recording started live");
    check::is_true(again.state() == QAbstractSocket::ConnectedState, "the command client kept through it");
    live.record = false;
    w.apply_settings(live);
    live.kiss_port += 1;
    w.apply_settings(live);
    check::is_true(wait_for([&] { return again.state() != QAbstractSocket::ConnectedState; }, 5000),
                   "a port change restarts: the client dropped");
    check::is_true(w.station().running() && w.station().args() == live, "running on the new port");
    check::current_step = "window: resize";
    w.resize(w.width(), 620);
    check::is_true(wait_for([&] { return saved_height() == 620; }, 3000), "a resize is saved without closing the window");
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

// The plot's width as its window resizes. (Its points: tests/test_rx_plot.cpp.)
void test_constellation_plot() {
    check::current_step = "constellation: square";
    QWidget host;
    auto* row = new QHBoxLayout(&host);
    row->addWidget(new QWidget, 1);
    auto* c = new gui::ConstellationPlot;
    row->addWidget(c);
    host.show();
    // wide windows: as wide as the row is tall; narrow ones: 30% of the window wide, the height spare
    for (const QSize s : {QSize(900, 200), QSize(700, 320), QSize(400, 150), QSize(400, 300), QSize(300, 400)}) {
        host.resize(s);
        wait_for([] { return false; }, 50);
        const std::string at = " at " + std::to_string(s.width()) + "x" + std::to_string(s.height());
        check::equal(c->width(), std::min(c->height(), s.width() * 3 / 10), "as wide as tall, up to 30%" + at);
    }
}

// The burst log stays pinned to the top (newest) by default even when a
// previously selected row ages past MAX_LOG_ROWS and is trimmed (#294), and
// keeps the visible row stationary when the user has scrolled down.
void test_burst_log_scroll(const QTemporaryDir& dir) {
    for (const auto mode : {QAbstractItemView::ScrollPerItem, QAbstractItemView::ScrollPerPixel}) {
        const std::string tag = mode == QAbstractItemView::ScrollPerPixel ? "pixel" : "item";
        check::current_step =
            mode == QAbstractItemView::ScrollPerPixel ? "burst log scroll (pixel)" : "burst log scroll (item)";
        app::Args a;
        a.sample_rate = 1;  // rejected by app::check: no audio device or ports opened
        QSettings store(dir.filePath(QStringLiteral("scroll-%1.ini").arg(QString::fromStdString(tag))), QSettings::IniFormat);
        gui::MainWindow w(a, store);
        w.resize(720, 560);
        w.show();
        QCoreApplication::processEvents();
        auto* log = w.findChild<QTableWidget*>(QStringLiteral("burst_log"));
        check::is_true(log != nullptr, "burst_log found");
        if (!log) return;
        log->setVerticalScrollMode(mode);
        QScrollBar* bar = log->verticalScrollBar();

        int next_id = 0;
        const auto make_bursts = [&](int n) {
            std::vector<app::BurstLogEntry> out;
            out.reserve(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                app::BurstLogEntry e;
                e.when = std::chrono::system_clock::now();
                e.mode = "b" + std::to_string(next_id++);
                e.n_cw = 1;
                e.decoded = 1;
                e.snr_db = 5.0;
                out.push_back(std::move(e));
            }
            return out;
        };
        const auto top_text = [&] {
            const int r = log->rowAt(0);
            return r >= 0 && log->item(r, 1) ? log->item(r, 1)->text().toStdString() : std::string();
        };

        // 1. Growing (< MAX_LOG_ROWS): default stays pinned at the top (newest).
        w.append_bursts(make_bursts(30));
        QCoreApplication::processEvents();
        check::equal(bar->value(), bar->minimum(), tag + ": starts pinned at top");
        check::equal(top_text(), std::string("b29 x1"), tag + ": newest at top");

        // 2. Scrolled down while still below MAX_LOG_ROWS: stays anchored on the same burst.
        bar->setValue(bar->maximum());
        QCoreApplication::processEvents();
        const std::string anchored_grow = top_text();
        check::is_true(!anchored_grow.empty() && bar->value() > bar->minimum(), tag + ": scrolled to bottom");
        w.append_bursts(make_bursts(3));
        QCoreApplication::processEvents();
        check::equal(top_text(), anchored_grow, tag + ": scrolled view stays anchored while growing");

        // 3. Regression #294: click row 0, let it drift to the oldest row and be trimmed at MAX_LOG_ROWS.
        bar->setValue(bar->minimum());
        log->setCurrentCell(0, 0);
        QCoreApplication::processEvents();
        w.append_bursts(make_bursts(gui::MainWindow::MAX_LOG_ROWS - log->rowCount()));
        QCoreApplication::processEvents();
        // Now the table is full; put currentIndex on the very last row, scroll to top, and append.
        log->setCurrentCell(log->rowCount() - 1, 0);
        bar->setValue(bar->minimum());
        QCoreApplication::processEvents();
        w.append_bursts(make_bursts(1));
        QCoreApplication::processEvents();
        check::equal(log->rowCount(), gui::MainWindow::MAX_LOG_ROWS, tag + ": capped at MAX_LOG_ROWS");
        check::equal(bar->value(), bar->minimum(), tag + ": trimming current row at bottom does not yank to bottom");
        check::equal(log->currentRow(), -1, tag + ": trimmed currentIndex cleared");
        check::equal(top_text(), "b" + std::to_string(next_id - 1) + " x1", tag + ": newest still visible at top");

        // 4. Mid-scroll when full + currentRow trimmed: stays anchored on the same burst.
        log->setCurrentCell(log->rowCount() - 1, 0);
        bar->setValue(bar->maximum() / 2);
        QCoreApplication::processEvents();
        const std::string anchored_full = top_text();
        w.append_bursts(make_bursts(3));
        QCoreApplication::processEvents();
        check::equal(top_text(), anchored_full, tag + ": mid-scroll stays anchored across trim");
    }
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
    test_config_beside_program(dir);
    test_config_file(dir);
    test_noise_rule_slider();
    test_decoder_effort();
    test_settings_fit_small_screen();
    test_restart_warning();
    test_waterfall_tx();
    test_constellation_plot();
    test_burst_log_scroll(dir);
    test_window(dir);
    return check::report("gui");
}
