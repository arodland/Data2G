// Exercise the sound card path for real (not run in CI): list devices, or
// play a tone out one device and measure it coming back in another, through
// the same FIFOs, Decimator and Interpolator the host uses.
//
//   data2g-audio-check                         # list devices
//   data2g-audio-check --loop --in USB --out USB [--seconds 5] [--rate 48000] [--lead-ms 100] [--buffer-ms 0]
//
// No loopback hardware on Linux: a null sink and a remapped monitor (Qt
// does not list monitor sources):
//   pactl load-module module-null-sink sink_name=d2g-null
//   pactl load-module module-remap-source source_name=d2g_loop master=d2g-null.monitor
//       source_properties=device.description=D2G-Loopback   (one line)
//   data2g-audio-check --loop --out d2g-null --in D2G-Loopback

#include <QCoreApplication>

#include <cmath>
#include <complex>
#include <cstdio>
#include <exception>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include "audio/audio.hpp"
#include "audio/fifo.hpp"
#include "audio/filters.hpp"
#include "audio/qt/qtaudio.hpp"
#include "generated/config.hpp"

using namespace data2g;

namespace {

void list(const char* what, const std::vector<audio::DeviceInfo>& devs) {
    std::printf("%s:\n", what);
    for (std::size_t i = 0; i < devs.size(); ++i) std::printf("%3zu  %2d ch  %s\n", i, devs[i].channels, devs[i].name.c_str());
}

// Power of the tone at `f` against the rest, dB (a single-bin DFT).
double tone_snr_db(const std::vector<double>& x, double f) {
    std::complex<double> acc;
    double total = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        acc += x[i] * std::polar(1.0, -2 * std::numbers::pi * f * double(i) / config::FS);
        total += x[i] * x[i];
    }
    const double tone = 2 * std::norm(acc) / double(x.size());
    return 10 * std::log10(tone / std::max(total - tone, 1e-30));
}

}  // namespace

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    std::string in, out;
    int rate = 48000;
    double seconds = 5.0, f = 1000.0, lead_ms = 100, buffer_ms = 0;  // the host's --tx-lead-ms, --output-buffer-ms
    bool loop = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--loop") loop = true;
        else if (a == "--in") in = next();
        else if (a == "--out") out = next();
        else if (a == "--rate") rate = std::stoi(next());
        else if (a == "--seconds") seconds = std::stod(next());
        else if (a == "--lead-ms") lead_ms = std::stod(next());
        else if (a == "--buffer-ms") buffer_ms = std::stod(next());
        else {
            std::fprintf(stderr, "usage: data2g-audio-check [--loop --in DEV --out DEV --rate HZ --seconds S --lead-ms MS --buffer-ms MS]\n");
            return 2;
        }
    }
    try {
        const auto ins = audio::qt::input_devices(), outs = audio::qt::output_devices();
        if (!loop) {
            list("input", ins);
            list("output", outs);
            return 0;
        }
        const auto report = [](const std::string& s) { std::fprintf(stderr, "%s\n", s.c_str()); };
        audio::CaptureFifo cap(config::FS);
        audio::PlaybackFifo play(rate, lead_ms / 1000);
        audio::qt::Capture capture(audio::select_device(ins, in, "input"), rate, cap, report);
        audio::qt::Playback playback(audio::select_device(outs, out, "output"), rate, play, report, buffer_ms / 1000);
        std::printf("in:  %s (%d ch)\nout: %s (%d ch)\n", capture.device_name().c_str(), capture.channels(),
                    playback.device_name().c_str(), playback.channels());

        audio::Interpolator interp(rate);
        const auto block = static_cast<std::size_t>(config::FS / 10);
        const auto blocks = static_cast<std::size_t>(seconds * 10);
        std::vector<double> heard;
        play.start();
        for (std::size_t k = 0; k < blocks; ++k) {  // as the host: one block in, one block out
            const auto x = cap.read(block);
            if (!x) break;
            heard.insert(heard.end(), x->begin(), x->end());
            std::vector<double> y(block);
            for (std::size_t i = 0; i < block; ++i)
                y[i] = 0.5 * std::sin(2 * std::numbers::pi * f * double(k * block + i) / config::FS);
            play.write(interp(y));
        }
        play.drain();
        const auto& b = play.last_burst();
        std::printf("burst: %.2f s written (lead included), the card took %.2f s in %.2f s, FIFO low %.3f s, %llu underruns\n",
                    b.written_s, b.pulled_s, b.wall_s, b.low_s, static_cast<unsigned long long>(b.underruns));
        // Where "now" is on the capture timeline when drain() returns (the
        // host drops PTT here, after its off delay): read so far plus the
        // backlog. Input latency shifts both this and the tone alike.
        const std::size_t drained_at = heard.size() + cap.backlog();
        const std::size_t sent = heard.size();
        while (heard.size() < drained_at + static_cast<std::size_t>(1.5 * config::FS)) {
            const auto x = cap.read(block);
            if (!x) break;
            heard.insert(heard.end(), x->begin(), x->end());
        }
        // The last 20 ms window holding the tone.
        const std::size_t win = config::FS / 50;
        std::optional<std::size_t> last;
        for (std::size_t i = 0; i + win <= heard.size(); i += win)
            if (tone_snr_db({heard.begin() + static_cast<std::ptrdiff_t>(i), heard.begin() + static_cast<std::ptrdiff_t>(i + win)}, f) > 10)
                last = i + win;
        // The last half of what was sent, past the loop's latency.
        const std::vector<double> tail(heard.begin() + static_cast<std::ptrdiff_t>(sent / 2), heard.begin() + static_cast<std::ptrdiff_t>(sent));
        std::printf("tone %.0f Hz: %.1f dB over the rest\n", f, tone_snr_db(tail, f));
        if (last)
            std::printf("tone still playing %.0f ms after drain() returned (PTT must stay up that long)\n",
                        1000.0 * (static_cast<double>(*last) - static_cast<double>(drained_at)) / config::FS);
        else std::printf("tone not heard: no drain timing\n");
        std::printf("capture: %llu device frames, overflows %llu, dropped %llu, late %llu, backlog %.2f s\n",
                    static_cast<unsigned long long>(capture.frames_in()),
                    static_cast<unsigned long long>(cap.overflows()), static_cast<unsigned long long>(cap.dropped()),
                    static_cast<unsigned long long>(cap.late_events()), cap.backlog_s());
        std::printf("playback: underruns %llu, output latency %.0f ms\n",
                    static_cast<unsigned long long>(play.underruns()), 1000 * play.output_latency());
        return cap.dropped() || play.underruns() ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "data2g-audio-check: %s\n", e.what());
        return 1;
    }
}
