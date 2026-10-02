// The audio layer against a fake sound card: no device, no Qt. Parity of
// Decimator / Interpolator / Blanker with the Python is in
// tests/test_native_audio.py; this checks behaviour (host.py's Capture and
// Player semantics) and thread safety (run it under TSan).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "audio/audio.hpp"
#include "audio/fifo.hpp"
#include "audio/pipe.hpp"
#include "audio/filters.hpp"
#include "check.hpp"
#include "generated/config.hpp"
#include "rig/ptt.hpp"

using namespace data2g;
using namespace std::chrono_literals;

namespace {

std::vector<double> noise(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> g;
    std::vector<double> x(n);
    for (double& v : x) v = g(rng);
    return x;
}

std::vector<double> chunked(auto& f, const std::vector<double>& x, std::size_t chunk) {
    std::vector<double> out;
    for (std::size_t i = 0; i < x.size(); i += chunk) {
        const auto y = f(std::span<const double>(x).subspan(i, std::min(chunk, x.size() - i)));
        out.insert(out.end(), y.begin(), y.end());
    }
    return out;
}

// Interleaved stereo float32 bytes: `x` on the left, junk on the right.
std::vector<std::byte> stereo(std::span<const double> x) {
    std::vector<std::byte> out(x.size() * 8);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float l = static_cast<float>(x[i]), r = -7.0f;
        std::memcpy(out.data() + 8 * i, &l, 4);
        std::memcpy(out.data() + 8 * i + 4, &r, 4);
    }
    return out;
}

// A sound card: a thread that delivers capture periods and pulls playback
// periods, `period` frames at a time, at (roughly) the real rate or as fast
// as it can.
class FakeCard {
public:
    FakeCard(int rate, std::size_t period, bool realtime) : rate_(rate), period_(period), realtime_(realtime) {}
    ~FakeCard() { stop(); }

    void capture_into(audio::CapturePipeline& p, std::vector<double> signal) {
        thread_ = std::thread([this, &p, signal = std::move(signal)] {
            for (std::size_t i = 0; i < signal.size() && !stop_; i += period_) {
                const auto chunk = std::span<const double>(signal).subspan(i, std::min(period_, signal.size() - i));
                p(stereo(chunk));
                tick();
            }
            done_ = true;
        });
    }

    void play_from(audio::PlaybackFifo& f) {
        thread_ = std::thread([this, &f] {
            std::vector<float> buf(period_);
            while (!stop_) {
                const std::size_t n = f.pull(buf);
                played_.insert(played_.end(), buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(n));
                tick();
            }
        });
    }

    void stop() {
        stop_ = true;
        if (thread_.joinable()) thread_.join();
    }
    bool done() const { return done_; }
    const std::vector<float>& played() const { return played_; }  // after stop()

private:
    void tick() const {
        if (realtime_) std::this_thread::sleep_for(std::chrono::duration<double>(double(period_) / rate_));
    }

    int rate_;
    std::size_t period_;
    bool realtime_;
    std::atomic<bool> stop_{false}, done_{false};
    std::vector<float> played_;
    std::thread thread_;
};

void test_filters() {
    check::current_step = "filters";
    std::vector<double> zi(2, 0.0);
    const std::vector<double> b{1.0, 2.0, 3.0};
    check::close(audio::lfilter_fir(b, std::vector<double>{1, 0, 0, 1}, zi), {1, 2, 3, 1}, 0, "lfilter impulse");
    check::close(zi, {2, 3}, 0, "lfilter state");

    const auto x = noise(48000, 1);
    audio::Decimator whole(48000), parts(48000);
    check::equal(whole.factor(), 6, "decimator factor");
    check::equal(whole.taps().size(), std::size_t{193}, "decimator taps");
    check::close(chunked(parts, x, 1234), whole(x), 0, "decimator seamless across chunks");
    audio::Decimator dc(48000);
    check::is_true(std::abs(dc(std::vector<double>(4800, 1.0)).back() - 1.0) < 1e-12, "decimator unit DC gain");
    audio::Decimator same(config::FS);
    check::close(same(x), x, 0, "decimator at FS is the identity");

    const auto y = noise(8000, 2);
    audio::Interpolator iw(48000), ip(48000);
    const auto up = iw(y);
    check::equal(up.size(), std::size_t{48000}, "interpolator length");
    check::close(chunked(ip, y, 777), up, 0, "interpolator seamless across chunks");
    audio::Interpolator idc(48000);
    const auto dcout = idc(std::vector<double>(800, 1.0));
    double period = 0;  // one output per polyphase branch: their mean is the DC gain
    for (std::size_t i = dcout.size() - 6; i < dcout.size(); ++i) period += dcout[i] / 6;
    check::is_true(std::abs(period - 1.0) < 1e-12, "interpolator unit DC gain");
    bool threw = false;
    try {
        audio::Decimator bad(44100);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "44.1 kHz refused");

    audio::Blanker bl;
    const auto n = noise(2 * config::FS, 3);
    check::close(bl(n), n, 0, "blanker leaves noise alone");
    check::equal(bl.n_blanked, std::int64_t{0}, "blanker counted nothing");
    auto clicky = n;
    clicky[5000] = 200.0;
    const auto z = bl(clicky);
    check::equal(z[5000], 0.0, "blanker zeroes a click");
    check::is_true(z[5000 - 8] == 0.0 && z[5000 + 8] == 0.0 && z[5000 + 9] == clicky[5000 + 9], "blanker guard is 8");
    audio::Blanker silent;
    check::close(silent(std::vector<double>(800, 0.0)), std::vector<double>(800, 0.0), 0, "blanker on silence");
}

void test_devices_and_formats() {
    check::current_step = "devices";
    const std::vector<audio::DeviceInfo> devs{{"HDA Intel PCH", 2}, {"USB Audio CODEC", 1}, {"usb monitor", 0}};
    check::is_true(!audio::select_device(devs, "", "input"), "empty: the default");
    check::equal(*audio::select_device(devs, "1", "input"), std::size_t{1}, "digits: an index");
    check::equal(*audio::select_device(devs, "usb", "input"), std::size_t{1}, "substring, any case");
    for (const char* bad : {"9", "monitor", "nope"}) {
        bool threw = false;
        try {
            audio::select_device(devs, bad, "input");
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check::is_true(threw, std::string("no match: ") + bad);
    }
    check::equal(audio::open_channels(devs[0]), 2, "stereo device opened stereo");
    check::equal(audio::open_channels(devs[1]), 1, "mono device opened mono");

    std::vector<std::byte> b;
    audio::to_device(std::vector<float>{1.0f, -1.0f, 0.5f}, audio::SampleFormat::Int16, 2, b);
    std::int16_t s[6];
    std::memcpy(s, b.data(), sizeof s);
    check::is_true(s[0] == 32767 && s[1] == 32767 && s[2] == -32767 && s[4] == 16384, "int16 out: no wrap, both channels");
    const auto back = audio::channel0(b, audio::SampleFormat::Int16, 2);
    check::close(back, {32767 / 32768.0, -32767 / 32768.0, 16384 / 32768.0}, 0, "int16 in: channel 0");
}

void test_capture_keeps_every_sample() {
    check::current_step = "capture";
    // test_host.py's scenario: the reader stalls while 10k frames arrive in
    // odd-sized pieces; they come out whole, in order, left channel.
    audio::CaptureFifo fifo(config::FS);
    audio::CapturePipeline pipe(audio::SampleFormat::Float, 2, config::FS, fifo);
    std::vector<double> ramp(10000);
    for (std::size_t i = 0; i < ramp.size(); ++i) ramp[i] = static_cast<double>(i);
    const auto raw = stereo(ramp);
    for (std::size_t i = 0; i < raw.size(); i += 1001)  // splits frames
        pipe(std::span<const std::byte>(raw).subspan(i, std::min<std::size_t>(1001, raw.size() - i)));
    fifo.overflow();
    std::vector<double> got = *fifo.read(4800), more = *fifo.read(4800);
    got.insert(got.end(), more.begin(), more.end());
    check::close(got, std::vector<double>(ramp.begin(), ramp.begin() + 9600), 0, "every sample, in order");
    check::equal(fifo.overflows(), std::uint64_t{1}, "overflow counted");
    check::equal(pipe.frames_in(), std::uint64_t{10000}, "frames in");
    fifo.close();
    check::is_true(!fifo.read(4800), "400 left and closed: nothing");
    check::equal(fifo.read(400)->size(), std::size_t{400}, "...but what is there is still readable");
}

void test_capture_through_a_card() {
    check::current_step = "capture card";
    // 48 kHz stereo from a card thread, the reader sleeping now and then:
    // the decimated stream equals one-shot decimation, nothing dropped.
    const auto x = noise(3 * 48000, 4);
    audio::CaptureFifo fifo(config::FS);
    audio::CapturePipeline pipe(audio::SampleFormat::Float, 2, 48000, fifo);
    FakeCard card(48000, 1024, false);
    std::vector<double> xf(x.size());  // what float32 delivers
    for (std::size_t i = 0; i < x.size(); ++i) xf[i] = static_cast<float>(x[i]);
    audio::Decimator ref32(48000);
    const auto want32 = ref32(xf);
    card.capture_into(pipe, xf);
    std::vector<double> got;
    while (got.size() + 800 <= want32.size()) {
        if (got.size() % 8000 == 0) std::this_thread::sleep_for(20ms);  // a slow step
        const auto b = fifo.read(800);
        got.insert(got.end(), b->begin(), b->end());
    }
    card.stop();
    check::close(got, std::vector<double>(want32.begin(), want32.begin() + static_cast<std::ptrdiff_t>(got.size())), 0,
                 "card -> fifo seamless");
    check::equal(fifo.dropped(), std::uint64_t{0}, "nothing dropped");

    // Backlog: over LATE_S counts once per excursion; past capacity, drops.
    audio::CaptureFifo small(config::FS, 2.0);
    small.write(std::vector<double>(12000, 0.0));
    small.read(800);
    small.read(800);
    check::equal(small.late_events(), std::uint64_t{1}, "late once, not per read");
    small.read(4000);
    small.write(std::vector<double>(12000, 0.0));
    check::equal(small.dropped(), std::uint64_t{2400}, "past capacity: dropped and counted");
}

void test_close_unblocks_a_reader() {
    check::current_step = "close";
    audio::CaptureFifo fifo(config::FS);
    std::atomic<bool> got_nothing{false};
    std::thread reader([&] { got_nothing = !fifo.read(800).has_value(); });
    std::this_thread::sleep_for(20ms);
    fifo.close();
    reader.join();
    check::is_true(got_nothing.load(), "close() returns a blocked read empty");
}

void test_playback() {
    check::current_step = "playback";
    const int rate = 48000;
    audio::PlaybackFifo fifo(rate, 0.1);
    fifo.set_output_latency(0.05);
    FakeCard card(rate, 2048, true);
    card.play_from(fifo);
    std::this_thread::sleep_for(50ms);  // idle: silence, not underruns
    check::equal(fifo.underruns(), std::uint64_t{0}, "idle is not an underrun");

    fifo.start();
    std::vector<double> burst;
    for (int k = 0; k < 5; ++k) {  // the engine: 0.1 s blocks
        std::vector<double> y(4800);
        for (std::size_t i = 0; i < y.size(); ++i) y[i] = 0.5 * std::sin(0.01 * double(burst.size() + i));
        fifo.write(y);
        burst.insert(burst.end(), y.begin(), y.end());
        std::this_thread::sleep_for(80ms);
    }
    const auto t0 = std::chrono::steady_clock::now();
    fifo.drain();
    const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    check::equal(fifo.queued(), std::size_t{0}, "drain empties the fifo");
    check::is_true(waited >= 0.05, "drain waits out the output latency");
    std::this_thread::sleep_for(50ms);
    card.stop();
    check::equal(fifo.underruns(), std::uint64_t{0}, "lead covers late steps; the end is not an underrun");
    const auto& p = card.played();
    check::equal(p.size(), std::size_t{4800 + burst.size()}, "lead then the burst, nothing lost");
    bool same = true;
    for (std::size_t i = 0; i < burst.size(); ++i) same &= p[4800 + i] == static_cast<float>(burst[i]);
    check::is_true(same && p[0] == 0.0f, "played in order after the lead");

    // Starved while keyed: counted.
    audio::PlaybackFifo starved(rate, 0.0);
    starved.start();
    std::vector<float> buf(256);
    check::equal(starved.pull(buf), std::size_t{0}, "nothing to play");
    check::equal(starved.underruns(), std::uint64_t{1}, "underrun counted while active");
}

void test_keyer() {
    check::current_step = "keyer";
    std::vector<std::string> calls;
    audio::PlaybackFifo fifo(48000, 0.1);
    {
        rig::Keyer k([&](bool on) { calls.push_back(on ? "on" : "off"); }, fifo, 0.0);
        k.key();
        check::is_true(k.keyed() && calls == std::vector<std::string>{"on"}, "key: PTT on");
        check::equal(fifo.queued(), std::size_t{4800}, "key: TX lead queued");
        std::vector<float> buf(4800);
        fifo.pull(buf);  // the card plays it
        k.unkey();
        check::is_true(!k.keyed() && calls.back() == "off", "unkey: PTT off after drain");
    }
    check::equal(calls.size(), std::size_t{3}, "destructor: PTT off again, always");

    std::string reported;
    rig::Keyer failing([](bool) { throw std::runtime_error("rigctld gone"); }, fifo, 0.0,
                       [&](const std::string& s) { reported = s; });
    failing.key();
    check::is_true(failing.keyed() && failing.failures() == 1, "a PTT failure is counted, not fatal");
    check::is_true(reported.find("rigctld gone") != std::string::npos, "...and reported");
    std::vector<float> buf(4800);
    fifo.pull(buf);
    failing.unkey();

    rig::Keyer none({}, fifo, 0.0);  // --rigctld-port 0
    none.key();
    check::is_true(none.keyed() && none.failures() == 0, "no PTT: keys the audio only");
    fifo.pull(buf);
    none.unkey();
}

}  // namespace

void test_pipe_io() {
    check::current_step = "pipe io";
    // Regular files: the input read at real time into the capture FIFO, the
    // output written at real time, silence around what was played.
    const auto dir = std::filesystem::temp_directory_path() /
                     ("data2g_test_pipe_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    const auto in = (dir / "in.f32").string(), out = (dir / "out.f32").string();
    const auto x = noise(4000, 7);
    {
        std::ofstream f(in, std::ios::binary);
        for (double v : x) {
            const float s = static_cast<float>(v);
            f.write(reinterpret_cast<const char*>(&s), sizeof s);
        }
    }
    audio::CaptureFifo cap(config::FS);
    audio::PlaybackFifo play(config::FS, 0.1);
    std::vector<double> ramp(800);
    for (std::size_t i = 0; i < ramp.size(); ++i) ramp[i] = static_cast<float>((i + 1) / 1000.0);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<double> got;
    {
        audio::PipeIo io(in, out, cap, play);
        play.start();
        play.write(ramp);
        const auto first = cap.read(4000);
        const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        check::is_true(took > 0.4, "input paced at real time (" + std::to_string(took) + " s for 0.5 s)");
        if (first) got = *first;
        play.drain();
        const auto silence = cap.read(800);  // after the file: silence at real time
        check::is_true(silence && std::all_of(silence->begin(), silence->end(), [](double v) { return v == 0.0; }),
                       "silence after the end");
    }
    std::vector<double> want(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) want[i] = static_cast<float>(x[i]);
    check::close(got, want, 0, "input: every sample");
    std::vector<float> o;
    {
        std::ifstream f(out, std::ios::binary);  // closed before remove_all: Windows won't delete an open file
        for (float s; f.read(reinterpret_cast<char*>(&s), sizeof s);) o.push_back(s);
    }
    check::equal(o.size() % audio::PipeIo::PERIOD, std::size_t{0}, "output: whole periods");
    const auto start = std::find_if(o.begin(), o.end(), [](float v) { return v != 0.0f; });
    check::is_true(start - o.begin() >= 800, "the lead's silence first");
    check::is_true(o.end() - start >= 800 && std::equal(ramp.begin(), ramp.end(), start, [](double a, float b) { return float(a) == b; }),
                   "output: the samples played, whole");
    check::is_true(std::all_of(start + 800, o.end(), [](float v) { return v == 0.0f; }), "then silence");
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);  // a leftover temp dir is not a failure
}

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(100, "test_audio");
    test_filters();
    test_devices_and_formats();
    test_capture_keeps_every_sample();
    test_capture_through_a_card();
    test_close_unblocks_a_reader();
    test_playback();
    test_keyer();
    test_pipe_io();
    return check::report("audio");
}
