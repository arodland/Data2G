#include "main_window.hpp"

#include <QDateTime>
#include <QGridLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPushButton>
#include <QSettings>
#include <QStatusBar>
#include <QTableWidget>
#include <QTimer>
#include <QVBoxLayout>

#include "arq/modes.hpp"
#include "arq/policy.hpp"
#include "audio/qt/qtaudio.hpp"
#include "settings_dialog.hpp"
#include "waterfall.hpp"

namespace data2g::gui {

namespace {

constexpr int MAX_LOG_ROWS = 500;

QString qs(const std::string& s) { return QString::fromStdString(s); }

QString link_text(const app::LinkStatus& l) {
    static const char* caps[] = {"500", "1200", "2300"};
    const QString peer = qs(l.peer);
    if (l.state == "listen") return QObject::tr("listening");
    if (l.state == "connecting") return QObject::tr("connecting to %1").arg(peer);
    if (l.state == "connected") return QObject::tr("connected to %1 (BW%2)").arg(peer, caps[std::clamp(l.cap, 0, 2)]);
    if (l.state == "disconnecting") return QObject::tr("disconnecting from %1").arg(peer);
    return qs(l.state);  // idle, closed
}

QString bytes_text(std::int64_t n) {
    return n < 10000 ? QObject::tr("%1 B").arg(n) : QObject::tr("%1 kB").arg(static_cast<double>(n) / 1000.0, 0, 'f', 1);
}

QStringList names(const std::vector<audio::DeviceInfo>& devs) {
    QStringList out;
    for (const auto& d : devs) out << qs(d.name);
    return out;
}

}  // namespace

MainWindow::MainWindow(app::Args args, QSettings& store, QWidget* parent) : QMainWindow(parent), store_(store) {
    setWindowTitle(tr("Data2G"));
    auto* central = new QWidget;
    auto* top = new QVBoxLayout(central);

    waterfall_ = new Waterfall;
    waterfall_->set_source([this](std::size_t n, std::uint64_t* total, bool* tx) {
        return station_ ? station_->input_tail(n, total, tx) : std::vector<double>{};
    });
    top->addWidget(waterfall_, 2);

    auto* grid = new QGridLayout;
    const auto value = [&](const char* name, int row, int col, const QString& label) {
        grid->addWidget(new QLabel(label), row, col);
        auto* v = new QLabel(QStringLiteral("-"));
        v->setObjectName(QLatin1String(name));
        v->setTextInteractionFlags(Qt::TextSelectableByMouse);
        grid->addWidget(v, row, col + 1);
        return v;
    };
    link_ = value("link_state", 0, 0, tr("Link:"));
    mode_ = value("mode", 1, 0, tr("Mode:"));
    rx_ = value("throughput_rx", 0, 2, tr("Received:"));
    tx_ = value("throughput_tx", 1, 2, tr("Sent:"));
    counters_ = value("audio_counters", 2, 0, tr("Audio:"));
    grid->addWidget(counters_, 2, 1, 1, 3);
    dial_ = value("dial", 3, 0, tr("Dial:"));
    dial_label_ = qobject_cast<QLabel*>(grid->itemAtPosition(3, 0)->widget());
    const auto lamp = [&](const char* name, const QString& text) {
        auto* l = new QLabel(text);
        l->setObjectName(QLatin1String(name));
        l->setAlignment(Qt::AlignCenter);
        l->setMinimumWidth(60);
        set_lamp(l, false, "");
        return l;
    };
    busy_ = lamp("busy", tr("BUSY"));
    ptt_ = lamp("ptt", tr("PTT"));
    auto* lamps = new QHBoxLayout;
    lamps->addWidget(busy_);
    lamps->addWidget(ptt_);
    lamps->addStretch();
    auto* settings = new QPushButton(tr("Settings..."));
    connect(settings, &QPushButton::clicked, this, &MainWindow::open_settings);
    lamps->addWidget(settings);
    grid->setColumnStretch(1, 1);
    grid->setColumnStretch(3, 1);
    top->addLayout(grid);
    top->addLayout(lamps);

    log_ = new QTableWidget(0, 4);
    log_->setObjectName(QStringLiteral("burst_log"));
    log_->setHorizontalHeaderLabels({tr("Time"), tr("Mode"), tr("SNR (dB)"), tr("Result")});
    log_->horizontalHeader()->setStretchLastSection(true);
    log_->verticalHeader()->setVisible(false);
    log_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    log_->setSelectionBehavior(QAbstractItemView::SelectRows);
    top->addWidget(log_, 3);

    setCentralWidget(central);
    status_ = new QLabel;
    status_->setObjectName(QStringLiteral("station_state"));
    statusBar()->addWidget(status_, 1);
    resize(720, 560);

    auto* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::poll);
    timer->start(200);
    restart(args);
}

MainWindow::~MainWindow() = default;

void MainWindow::restart(const app::Args& args) {
    station_.reset();  // stops it: PTT down, ports closed
    samples_.clear();
    error_.clear();
    // each run records to its own directory: a second Recorder in one would
    // number its rx_NNNNN.npz from 0 again, over the first one's
    app::Args a = args;
    if (starts_++ == 0) record_base_ = a.record_dir;
    else if (!record_base_.empty()) a.record_dir = record_base_ + "." + std::to_string(starts_);
    station_ = std::make_unique<app::Station>(a);
    if (const auto bad = app::check(a)) {
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

void MainWindow::set_lamp(QLabel* lamp, bool lit, const char* color) {
    lamp->setProperty("lit", lit);
    lamp->setStyleSheet(lit ? QStringLiteral("QLabel { background: %1; color: black; border-radius: 3px; padding: 2px; }").arg(color)
                            : QStringLiteral("QLabel { color: gray; border: 1px solid gray; border-radius: 3px; padding: 2px; }"));
}

void MainWindow::poll() {
    auto& s = *station_;
    if (s.failed() && s.running()) {
        s.stop();
        error_ = tr("the engine stopped (see the log)");
    }
    const app::Args& a = s.args();
    const QString call = qs(a.mycall.value_or("NOCALL"));
    if (!s.running()) {
        status_->setText(tr("%1: stopped: %2").arg(call, error_));
    } else {
        QStringList on;
        on << tr("VARA %1:%2").arg(qs(a.host)).arg(a.command_port);
        on << tr("KISS %1:%2").arg(qs(a.kiss_address)).arg(a.kiss_port);
        on << (a.audio_io.empty() ? tr("audio %1 / %2").arg(qs(a.input_device.value_or("default")), qs(a.output_device.value_or("default")))
                                  : tr("audio %1").arg(qs(a.audio_io)));
        status_->setText(tr("%1 on %2").arg(call, on.join(QStringLiteral(", "))));
    }

    const app::LinkStatus l = s.link();
    link_->setText(link_text(l));
    if (l.mode.empty()) {
        mode_->setText(QStringLiteral("-"));
    } else if (const auto* m = arq::mode(l.mode)) {
        mode_->setText(tr("%1, %2 Hz").arg(qs(l.mode)).arg(arq::width_hz(*m), 0, 'f', 0));
    } else {
        mode_->setText(qs(l.mode));
    }
    const bool polled = s.running() && a.rig_poll_interval > 0;
    dial_label_->setVisible(polled);
    dial_->setVisible(polled);
    if (polled) {
        const auto hz = s.rig_frequency();
        dial_->setText(hz ? tr("%1 MHz").arg(*hz / 1e6, 0, 'f', 4) : QStringLiteral("-"));
    }
    set_lamp(busy_, s.busy(), "#e6a01e");
    set_lamp(ptt_, s.ptt(), "#f03c3c");

    // throughput: bytes this connection, and their rate over the stats interval
    const auto now = std::chrono::steady_clock::now();
    if (!samples_.empty() && (l.rx_bytes < samples_.back().rx || l.tx_bytes < samples_.back().tx)) samples_.clear();  // a new connection
    samples_.push_back({now, l.rx_bytes, l.tx_bytes});
    const double window = a.stats_interval > 0 ? a.stats_interval : 60.0;
    while (samples_.size() > 2 && std::chrono::duration<double>(now - samples_[1].t).count() >= window) samples_.pop_front();
    const double dt = std::chrono::duration<double>(now - samples_.front().t).count();
    const auto bps = [&](std::int64_t d) { return dt > 0 ? 8.0 * static_cast<double>(d) / dt : 0.0; };
    rx_->setText(tr("%1, %2 bps").arg(bytes_text(l.rx_bytes)).arg(bps(l.rx_bytes - samples_.front().rx), 0, 'f', 0));
    tx_->setText(tr("%1, %2 bps").arg(bytes_text(l.tx_bytes)).arg(bps(l.tx_bytes - samples_.front().tx), 0, 'f', 0));

    const app::AudioCounters c = s.counters();
    counters_->setText(tr("overflows %1, dropped %2, backlog %3 s (late %4), underruns %5, decode dropped %6")
                           .arg(c.overflows)
                           .arg(c.dropped)
                           .arg(c.backlog_s, 0, 'f', 1)
                           .arg(c.late)
                           .arg(c.underruns)
                           .arg(c.decode_dropped));
    counters_->setStyleSheet(c.overflows || c.dropped || c.late || c.underruns || c.decode_dropped
                                 ? QStringLiteral("color: #d03030")
                                 : QString());

    for (const auto& b : s.take_bursts()) {
        log_->insertRow(0);  // newest first
        const auto when = QDateTime::fromMSecsSinceEpoch(
            std::chrono::duration_cast<std::chrono::milliseconds>(b.when.time_since_epoch()).count());
        const QStringList cells = {when.toString(QStringLiteral("HH:mm:ss")), tr("%1 x%2").arg(qs(b.mode)).arg(b.n_cw),
                                   b.snr_db ? QString::number(*b.snr_db, 'f', 1) : QStringLiteral("-"),
                                   b.lost ? tr("lost") : tr("ok")};
        for (int i = 0; i < cells.size(); ++i) log_->setItem(0, i, new QTableWidgetItem(cells[i]));
    }
    while (log_->rowCount() > MAX_LOG_ROWS) log_->removeRow(log_->rowCount() - 1);
}

void MainWindow::open_settings() {
    SettingsDialog d(station_->args(), names(audio::qt::input_devices()), names(audio::qt::output_devices()), this);
    if (d.exec() != QDialog::Accepted) return;
    app::Args a = station_->args();
    d.apply_to(a);
    save_settings(store_, a);
    if (a != station_->args() || !station_->running()) restart(a);
}

}  // namespace data2g::gui
