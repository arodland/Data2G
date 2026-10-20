// data2g-gui's one window: the station's status over a Station runtime
// (app/station.hpp, the same one data2g-host runs), polled by a QTimer from
// its snapshots and atomics, as SSTVAE's rx_panel polls its decoder. Nothing
// here waits on the engine.
//
// Shown: the RX waterfall and input level, the last OFDM burst's constellation, link state, mode and bandwidth,
// BUSY and PTT, throughput, the received-burst log, the audio counters, and
// the dial frequency when the rig is polled.
// Monitor (a check box) decodes every burst heard and sent as a packet dump,
// into the monitor window's log whether it is open or not; Monitor... opens
// that window (and turns monitoring on).
// Settings... opens the dialog; OK saves to QSettings and, when anything
// changed, restarts the station.
#pragma once

#include <QMainWindow>
#include <QStringList>

#include <chrono>
#include <deque>
#include <memory>

#include "app/station.hpp"

class QCheckBox;
class QLabel;
class QPushButton;
class QSettings;
class QTableWidget;

namespace data2g::gui {

class ConstellationPlot;
class MonitorWindow;
class Waterfall;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    // Starts a station with `args`; `store` is where settings are saved.
    MainWindow(app::Args args, QSettings& store, QWidget* parent = nullptr);
    ~MainWindow() override;

    // Stops the station and starts one with `a`; a failure is shown, not thrown.
    void restart(const app::Args& a);
    const app::Station& station() const { return *station_; }

public Q_SLOTS:
    void poll();  // a slot so a test can drive it
    void open_settings();
    void open_monitor();  // the Monitor window: every burst heard and sent, as a packet dump

private:
    void set_lamp(QLabel* lamp, bool lit, const char* color);

    QSettings& store_;
    std::unique_ptr<app::Station> station_;
    QString error_;
    Waterfall* waterfall_;
    ConstellationPlot* constellation_;
    QLabel *link_, *mode_, *busy_, *ptt_, *rx_, *tx_, *counters_, *status_;
    QPushButton* abort_;  // enabled while a session is under way
    QLabel *dial_label_, *dial_;  // shown only with --rig-poll-interval
    QTableWidget* log_;
    QCheckBox* monitoring_;
    MonitorWindow* monitor_;  // made hidden with this window, fed while monitoring_ is checked
    struct Sample {
        std::chrono::steady_clock::time_point t;
        std::int64_t rx, tx;
    };
    std::deque<Sample> samples_;  // throughput over the stats interval
};

}  // namespace data2g::gui
