// data2g-gui: the station of data2g-host with a status window and a
// settings dialog. Settings come from QSettings, then data2g-host's flags on
// the command line override them for this run.

#include <QApplication>
#include <QSettings>

#include <csignal>
#include <cstdio>

#include "app/station.hpp"
#include "main_window.hpp"
#include "settings_dialog.hpp"

using namespace data2g;

int main(int argc, char** argv) {
    QApplication qapp(argc, argv);  // takes Qt's own flags out of argv first
    QCoreApplication::setOrganizationName(QStringLiteral("Data2G"));
    QCoreApplication::setApplicationName(QStringLiteral("data2g-gui"));
    QSettings store;
    const app::Args a = app::parse(argc, argv, gui::load_settings(store), "data2g-gui");
    if (a.list_modes || a.list_audio_devices || a.list_rigs)
        app::usage_error("--list-modes, --list-audio-devices and --list-rigs are data2g-host's", "data2g-gui");
    const auto level = app::parse_level(a.log_level);
    if (!level) {
        std::fprintf(stderr, "data2g-gui: Unknown level: '%s'\n", a.log_level.c_str());
        return 1;
    }
    app::g_level = *level;
    app::install_arq_log();
#ifndef _WIN32
    std::signal(SIGPIPE, SIG_IGN);  // --audio-io pipes, sockets: an error, not death
#endif
    gui::MainWindow w(a, store);
    w.show();
    return QApplication::exec();
}
