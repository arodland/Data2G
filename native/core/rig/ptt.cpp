#include "rig/ptt.hpp"

#include <chrono>
#include <exception>
#include <thread>
#include <utility>

namespace data2g::rig {

Keyer::Keyer(Ptt ptt, audio::PlaybackFifo& out, double off_delay_s, Report on_error)
    : ptt_(std::move(ptt)), out_(out), off_delay_s_(off_delay_s), report_(std::move(on_error)) {}

Keyer::~Keyer() { ptt(false); }

void Keyer::key() {
    if (keyed_) return;
    ptt(true);
    out_.start();
    keyed_ = true;
}

void Keyer::unkey() {
    if (!keyed_) return;
    out_.drain();
    if (off_delay_s_ > 0) std::this_thread::sleep_for(std::chrono::duration<double>(off_delay_s_));
    ptt(false);
    keyed_ = false;
}

void Keyer::ptt(bool on) {
    if (!ptt_) return;
    try {
        ptt_(on);
    } catch (const std::exception& e) {
        ++failures_;
        if (report_) report_(std::string("PTT ") + (on ? "on" : "off") + " failed: " + e.what());
    }
}

}  // namespace data2g::rig
