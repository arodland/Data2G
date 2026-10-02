// data2g-host: data2g/host.py's server in C++, headless. Same command line
// as host.py's main(), plus --audio-io, --decode-worker / --no-decode-worker.
//
//   data2g-host --mycall W1AW --input-device USB --output-device USB --rigctld-port 4532
//
// The station itself (servers, engine, audio, PTT, and their threads) is
// app/station.{hpp,cpp}, shared with data2g-gui.

#include <QCoreApplication>
#include <QTimer>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <utility>

#include "app/station.hpp"
#include "generated/config.hpp"

#ifdef DATA2G_HAVE_QTAUDIO
#include "audio/qt/qtaudio.hpp"
#endif

using namespace data2g;
using namespace data2g::app;

namespace {

std::atomic<bool> g_quit{false};
extern "C" void on_signal(int) { g_quit = true; }

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    if (a.list_modes) {
        list_modes(a.kiss_bw);
        return 0;
    }
    if (a.list_rigs) {
        if (list_rigs()) return 0;
        std::fprintf(stderr, "data2g-host: built without Hamlib: no rig models\n");
        return 1;
    }
    if (const auto bad = check(a)) usage_error(*bad);
    const auto level = parse_level(a.log_level);
    if (!level) {
        std::fprintf(stderr, "data2g-host: Unknown level: '%s'\n", a.log_level.c_str());
        return 1;
    }
    g_level = *level;
    install_arq_log();

    QCoreApplication app(argc, argv);
    if (a.list_audio_devices) {
#ifdef DATA2G_HAVE_QTAUDIO
        for (const auto& [what, devs] : {std::pair{"input", audio::qt::input_devices()}, {"output", audio::qt::output_devices()}}) {
            std::printf("%s:\n", what);
            for (std::size_t i = 0; i < devs.size(); ++i) std::printf("%3zu  %2d ch  %s\n", i, devs[i].channels, devs[i].name.c_str());
        }
#else
        std::printf("built without Qt Multimedia: no sound cards (use --audio-io)\n");
#endif
        return 0;
    }
    if (a.audio_io.empty() && a.sample_rate % config::FS)
        usage_error("--sample-rate must be a multiple of " + std::to_string(config::FS));
#ifndef _WIN32
    std::signal(SIGPIPE, SIG_IGN);  // a pipe or socket whose reader went: an error, not death
#endif
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    try {
        Station s(a);
        try {
            s.start();
        } catch (const UsageError& e) {
            usage_error(e.what());
        } catch (const std::invalid_argument& e) {
            std::fprintf(stderr, "data2g-host: %s (see --list-audio-devices)\n", e.what());
            return 1;
        }
        QTimer quit_check;
        QObject::connect(&quit_check, &QTimer::timeout, [&s] {
            if (g_quit || s.failed()) QCoreApplication::quit();
        });
        quit_check.start(100);
        QCoreApplication::exec();
        log_line(INFO, "shutting down");
        s.stop();
        return 0;
    } catch (const std::exception& e) {
        logf(CRITICAL, "%s", e.what());
        return 1;
    }
}
