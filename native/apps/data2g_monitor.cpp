// data2g-monitor: a promiscuous receiver. Every burst heard goes to stdout
// as a packet dump (core/monitor): its header, then its payloads as text or
// as a hex dump. Receive only: no transmitter, no output device, no rig.
//
//   data2g-monitor --input-device USB            # a sound card
//   data2g-monitor --input-file rx.f32 --hex     # raw float32 mono at 8 kHz ("-": stdin, or a named pipe)

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "audio/fifo.hpp"
#include "monitor/monitor.hpp"

#ifdef DATA2G_HAVE_AUDIO
#include <QCoreApplication>

#include "audio/card.hpp"
#endif

using namespace data2g;

namespace {

constexpr std::size_t BLOCK = config::FS / 10;

[[noreturn]] void usage(int code) {
    std::fprintf(code ? stderr : stdout,
                 "usage: data2g-monitor [--input-device NAME|INDEX] [--sample-rate HZ] [--input-file PATH|-]\n"
                 "                      [--hex] [--list-audio-devices]\n"
                 "\n"
                 "Decodes every burst it hears, whoever it is for, and prints each as a packet dump.\n"
                 "\n"
                 "  --input-device       sound card to listen on (a name part or an index; default: the system's)\n"
                 "  --sample-rate        its rate, a multiple of %d (default 48000)\n"
                 "  --input-file         raw float32 mono at %d Hz instead of a sound card (\"-\": stdin)\n"
                 "  --hex                payloads as a hex dump (default: text, non-printables as <AB>)\n"
                 "  --list-audio-devices list input devices and exit\n",
                 config::FS, config::FS);
    std::exit(code);
}

std::string wall_clock() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    char s[40];
    std::strftime(s, sizeof s, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return s + std::string(".") + std::to_string(ms / 100);
}

}  // namespace

int main(int argc, char** argv) {
    std::optional<std::string> device, file;
    int rate = 48000;
    bool hex = false, list = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) usage(2);
            return argv[++i];
        };
        if (a == "--input-device") device = value();
        else if (a == "--sample-rate") rate = std::atoi(value().c_str());
        else if (a == "--input-file") file = value();
        else if (a == "--hex") hex = true;
        else if (a == "--list-audio-devices") list = true;
        else if (a == "-h" || a == "--help") usage(0);
        else usage(2);
    }
    const auto fmt = hex ? monitor::Format::HEX : monitor::Format::TEXT;
    monitor::Monitor mon;
    auto show = [&](std::span<const double> x, const std::function<std::string(double)>& when) {
        for (const auto& d : mon.feed(x, when)) std::fputs((monitor::render(d, fmt) + "\n").c_str(), stdout);
        std::fflush(stdout);
    };

    if (file) {  // as fast as it can be read (a named pipe: as its writer goes)
        FILE* f = *file == "-" ? stdin : std::fopen(file->c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "data2g-monitor: %s: %s\n", file->c_str(), std::strerror(errno));
            return 1;
        }
        auto when = [](double t) {
            char s[24];
            std::snprintf(s, sizeof s, "%.1f s", t);
            return std::string(s);
        };
        std::vector<float> in(BLOCK);
        std::vector<double> x;
        for (std::size_t n; (n = std::fread(in.data(), sizeof(float), BLOCK, f)) > 0;) {
            x.assign(in.begin(), in.begin() + static_cast<std::ptrdiff_t>(n));
            show(x, when);
        }
        show(std::vector<double>(2 * config::FS, 0.0), when);  // the last burst's tail
        return 0;
    }

#ifdef DATA2G_HAVE_AUDIO
    QCoreApplication app(argc, argv);
    const auto devs = audio::card::input_devices();
    if (list) {
        for (std::size_t i = 0; i < devs.size(); ++i) std::printf("%3zu  %2d ch  %s\n", i, devs[i].channels, devs[i].name.c_str());
        return 0;
    }
    if (rate <= 0 || rate % config::FS) {
        std::fprintf(stderr, "data2g-monitor: --sample-rate must be a multiple of %d\n", config::FS);
        return 2;
    }
    audio::CaptureFifo fifo(config::FS);
    std::unique_ptr<audio::card::Capture> cap;
    try {
        cap = std::make_unique<audio::card::Capture>(audio::select_device(devs, device.value_or(""), "input"), rate, fifo,
                                                     [](const std::string& s) { std::fprintf(stderr, "data2g-monitor: %s\n", s.c_str()); });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "data2g-monitor: %s (see --list-audio-devices)\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "data2g-monitor: listening on %s at %d Hz\n", cap->device_name().c_str(), rate);
    // ponytail: Ctrl-C ends it the default way; the dumps are flushed as they go
    while (const auto x = fifo.read(BLOCK)) show(*x, [](double) { return wall_clock(); });
    return 0;
#else
    (void)device, (void)rate, (void)list;
    std::fprintf(stderr, "data2g-monitor: built without sound card audio: only --input-file\n");
    return 2;
#endif
}
