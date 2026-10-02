// host.py serve()'s key-up / key-down sequence around the TX FIFO, on the
// engine thread. The PTT on delay is not here: the engine puts that much
// silence ahead of a burst (Engine ptt_delay_s, --ptt-on-delay-ms).
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "audio/fifo.hpp"

namespace data2g::rig {

class Keyer {
public:
    // RigController::ptt_function(), or empty for no PTT (--rigctld-port 0).
    using Ptt = std::function<void(bool)>;
    using Report = std::function<void(const std::string&)>;

    // `off_delay_s`: --ptt-off-delay-ms, between the audio draining and PTT off.
    Keyer(Ptt ptt, audio::PlaybackFifo& out, double off_delay_s, Report on_error = {});
    ~Keyer();  // PTT always comes back down

    Keyer(const Keyer&) = delete;
    Keyer& operator=(const Keyer&) = delete;

    void key();    // PTT on, then the TX lead queued
    void unkey();  // drain the FIFO and the device, wait off_delay_s, PTT off
    bool keyed() const { return keyed_; }
    // PTT calls that failed (logged and carried on from, as tnc.Rigctld does).
    std::uint64_t failures() const { return failures_; }

private:
    void ptt(bool on);

    Ptt ptt_;
    audio::PlaybackFifo& out_;
    double off_delay_s_;
    Report report_;
    bool keyed_ = false;
    std::uint64_t failures_ = 0;
};

}  // namespace data2g::rig
