#include "audio/fifo.hpp"

#include <chrono>
#include <cstddef>
#include <thread>

namespace data2g::audio {

namespace {

std::size_t samples(double seconds, int rate) { return static_cast<std::size_t>(seconds * rate); }

void sleep_s(double s) {
    if (s > 0) std::this_thread::sleep_for(std::chrono::duration<double>(s));
}

}  // namespace

CaptureFifo::CaptureFifo(int rate, double capacity_s) : rate_(rate), ring_(samples(capacity_s, rate)) {}

void CaptureFifo::write(std::span<const double> x) {
    const std::size_t n = ring_.write(x);
    if (n < x.size()) dropped_.fetch_add(x.size() - n, std::memory_order_relaxed);
    { std::lock_guard<std::mutex> lock(m_); }  // a reader between its check and its wait sees this notify
    cv_.notify_one();
}

std::optional<std::vector<double>> CaptureFifo::read(std::size_t n) {
    {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&] { return ring_.size() >= n || closed_.load(); });
    }
    if (ring_.size() < n) return std::nullopt;
    std::vector<double> out(n);
    ring_.read(out);  // outside the lock
    const bool late = backlog_s() > LATE_S;
    if (late && !late_) late_events_.fetch_add(1, std::memory_order_relaxed);
    late_ = late;
    return out;
}

void CaptureFifo::close() {
    {
        std::lock_guard<std::mutex> lock(m_);
        closed_ = true;
    }
    cv_.notify_all();
}

PlaybackFifo::PlaybackFifo(int rate, double lead_s, double capacity_s)
    : rate_(rate), lead_(samples(lead_s, rate)), ring_(samples(capacity_s, rate)) {}

void PlaybackFifo::start() {
    t0_ = std::chrono::steady_clock::now();
    pulled0_ = pulled_.load(std::memory_order_relaxed);
    underruns0_ = underruns_.load(std::memory_order_relaxed);
    written_ = 0;
    write(std::vector<double>(lead_, 0.0));
    low_.store(ring_.size(), std::memory_order_relaxed);
    active_.store(true, std::memory_order_release);
}

void PlaybackFifo::write(std::span<const double> y) {
    written_ += y.size();
    scratch_.assign(y.begin(), y.end());
    std::span<const float> left(scratch_);
    // ponytail: polls for room; it only waits if the engine runs capacity_s ahead of the card
    while (!left.empty()) {
        left = left.subspan(ring_.write(left));
        if (!left.empty()) sleep_s(0.002);
    }
    std::lock_guard lock(hook_mu_);
    if (on_write_) on_write_();
}

void PlaybackFifo::drain() {
    active_.store(false, std::memory_order_release);  // the last pull coming up short is the end, not an underrun
    while (const std::size_t left = ring_.size()) sleep_s(static_cast<double>(left) / rate_ + 0.002);
    const double r = rate_;
    last_ = {static_cast<double>(written_) / r,
             static_cast<double>(pulled_.load(std::memory_order_relaxed) - pulled0_) / r,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count(),
             static_cast<double>(low_.load(std::memory_order_relaxed)) / r,
             underruns_.load(std::memory_order_relaxed) - underruns0_};
    sleep_s(output_latency());
}

std::size_t PlaybackFifo::pull(std::span<float> out) {
    if (active_.load(std::memory_order_acquire))
        low_.store(std::min(low_.load(std::memory_order_relaxed), ring_.size()), std::memory_order_relaxed);
    pulled_.fetch_add(out.size(), std::memory_order_relaxed);
    const std::size_t n = ring_.read(out);
    std::fill(out.begin() + static_cast<std::ptrdiff_t>(n), out.end(), 0.0f);
    if (n < out.size() && active_.load(std::memory_order_acquire)) underruns_.fetch_add(1, std::memory_order_relaxed);
    return n;
}

std::size_t PlaybackFifo::pull_some(std::span<float> out) {
    if (!active_.load(std::memory_order_acquire)) {
        pull(out);
        return out.size();
    }
    low_.store(std::min(low_.load(std::memory_order_relaxed), ring_.size()), std::memory_order_relaxed);
    const std::size_t n = ring_.read(out);
    pulled_.fetch_add(n, std::memory_order_relaxed);
    if (n == 0 && !out.empty()) underruns_.fetch_add(1, std::memory_order_relaxed);
    return n;
}

}  // namespace data2g::audio
