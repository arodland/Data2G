#include "transmitter.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <span>
#include <string>
#include <utility>

#ifdef __linux__
#include <pthread.h>
#endif

#include "app/log.hpp"
#include "audio/fifo.hpp"
#include "audio/filters.hpp"
#include "audio/thread.hpp"
#include "generated/config.hpp"

namespace data2g::app {

Transmitter::Transmitter(audio::PlaybackFifo& play, int rate, double gain, rig::Keyer::Ptt ptt,
                         rig::Keyer::MustRelease must_release, double off_delay_s, std::function<void(bool)> on_ptt)
    : play_(play), rate_(rate), gain_(gain), on_ptt_(std::move(on_ptt)),
      keyer_(std::move(ptt), play, off_delay_s, [](const std::string& s) { log_line(ERROR, s); }, std::move(must_release)),
      thread_(&Transmitter::run, this) {}

Transmitter::~Transmitter() { stop(); }

void Transmitter::stop() {
    {
        std::lock_guard lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void Transmitter::submit(std::shared_ptr<const std::vector<double>> burst) {
    {
        std::lock_guard lock(mu_);
        q_.push_back({std::move(burst), generation_.load()});
    }
    cv_.notify_one();
}

void Transmitter::abort() {
    std::lock_guard lock(mu_);
    ++generation_;
    q_.clear();
    cv_.notify_all();
}

// For each burst: key, feed the playback FIFO a block at a time (never more than half a second ahead of the
// card, so an abort is quick), wait for the card to finish, and unkey.
void Transmitter::run() {
#ifdef __linux__
    pthread_setname_np(pthread_self(), "d2g-tx");
#endif
    audio::thread_init("tx");
    const std::size_t block = config::FS / 10;
    const double gain = gain_;
    const std::size_t ahead = static_cast<std::size_t>(rate_) / 2;
    for (;;) {
        Job job;
        {
            std::unique_lock lock(mu_);
            cv_.wait(lock, [&] { return stop_ || !q_.empty(); });
            if (stop_) break;
            job = std::move(q_.front());
            q_.pop_front();
        }
        if (job.generation != generation_) continue;  // aborted before it began
        const auto stopped = [&] { return stop_ || job.generation != generation_; };
        try {
            active_ = true;
            keyer_.key();  // PTT on, the lead queued
            ptt_ = true;
            on_ptt_(true);
            audio::Interpolator interp(rate_);
            const auto& a = *job.audio;
            for (std::size_t pos = 0; pos < a.size() && !stopped(); pos += block) {
                while (play_.queued() > ahead && !stopped()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
                if (stopped()) break;
                auto y = interp(std::span<const double>(a).subspan(pos, std::min(block, a.size() - pos)));
                for (auto& v : y) v = std::clamp(v * gain, -1.0, 1.0);
                play_.write(y);
            }
            if (stopped()) play_.flush();
            keyer_.unkey([this] { active_ = false; });  // drain, off delay, unmute, PTT off
            // a PTT off the rig didn't take stays pending in the keyer, and is tried again (it gives up after a few)
            for (int i = 0; keyer_.keyed() && i < 10; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                keyer_.unkey();
            }
        } catch (const std::exception& e) {
            logf(ERROR, "transmit: %s", e.what());
            try {
                keyer_.unkey();  // never leave the radio transmitting
            } catch (...) {
            }
        }
        if (ptt_) on_ptt_(false);  // clients that key the rig themselves release it now
        ptt_ = false;
        active_ = false;
    }
}

}  // namespace data2g::app
