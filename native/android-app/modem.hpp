// The running modem, and the only thing that owns one: an app::Station (the
// same runtime data2g-host and data2g-gui run) built from the saved settings.
//
// SSTVAE's ownership inversion (its android-app/session.hpp): the modem is
// process-wide, TncService keeps the process alive around it, and every
// screen is a view that reads it. Only the service starts and stops it
// (service_jni.cpp), so it belongs to something Android won't destroy with
// the activity.
//
// Threads: the Station is made, started, polled and stopped on Qt's thread,
// whose event loop runs its TCP servers. The service asks from Android's
// main thread through post_start()/post_stop(), and reads state(), and
// status_line() which poll() caches: nothing it calls touches the Station.
#pragma once

#include <QObject>
#include <QTimer>

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "app/station.hpp"

namespace data2g::androidapp {

enum class State { Stopped, Starting, Running };

// The saved settings (app/settings.hpp) over the app's defaults: no rig, no
// real-time request (the Java audio threads ask Android for their priority).
app::Args load_args();
// How the rig is reached: "usb", "bluetooth" or "network" (Hamlib dials
// rig_device itself). Picked in Settings, not guessed from the device string.
inline constexpr const char* RIG_CONNECTION_KEY = "android/rig_connection";
std::string rig_connection();

class Modem {
public:
    static Modem& instance();

    // Any thread: start or stop on Qt's thread. post_start() is Starting at once.
    void post_start();
    void post_stop();
    // post_stop(), waiting up to 3 s for it: for a destroy that ends the process next.
    void stop_now();

    State state() const { return state_.load(); }
    // One line for the ongoing notification.
    std::string status_line() const;
    // Why the last start failed or the engine stopped; empty when it didn't.
    std::string error() const;

    // Qt's thread only. Null when stopped.
    app::Station* station() { return station_ && station_->running() ? station_.get() : nullptr; }

    // The activity log: the station's log lines and every burst, newest last.
    // Lines with sequence numbers above `after`, and the newest number.
    std::vector<std::string> log_after(std::uint64_t after, std::uint64_t* newest) const;
    void clear_log();

private:
    Modem();
    void start();
    void stop();
    void poll();
    void add_log(std::string line);
    void set_status(std::string s);

    std::unique_ptr<app::Station> station_;
    std::atomic<State> state_{State::Stopped};
    bool was_ptt_ = false;  // Qt's thread
    QTimer poll_;

    mutable std::mutex mu_;  // the rest
    std::string status_, error_;
    std::deque<std::string> log_;
    std::uint64_t log_seq_ = 0;  // of log_.back()
};

}  // namespace data2g::androidapp
