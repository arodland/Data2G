// data2g-web: the station of data2g-host, headless, with data2g-gui's
// windows as a web page (web/web_ui.hpp).
//
//   data2g-web [--web-listen ADDR] [--web-port N] [data2g-host's options]
//
// --web-listen defaults to 127.0.0.1: the page can change the rig and the
// ports and has no login, so 0.0.0.0 is for a network you trust.
// Settings come from QSettings (data2g-web's own, as data2g-gui's), then
// data2g-host's flags on the command line override them for this run.

#include <QCoreApplication>
#include <QHostAddress>
#include <QSettings>
#include <QTimer>

#include <cstdio>
#include <string>
#include <vector>

#include "app/settings.hpp"
#include "app/station.hpp"
#include "web_ui.hpp"

using namespace data2g;

int main(int argc, char** argv) {
    QCoreApplication qapp(argc, argv);
    QCoreApplication::setOrganizationName(QStringLiteral("Data2G"));
    QCoreApplication::setApplicationName(QStringLiteral("data2g-web"));
    QCoreApplication::setApplicationVersion(QStringLiteral(DATA2G_VERSION));

    // Our two flags out of argv; the rest is data2g-host's.
    QString web_listen = QStringLiteral("127.0.0.1");
    int web_port = 8400;
    std::vector<char*> rest{argv[0]};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        bool ours = false;
        for (const char* flag : {"--web-listen", "--web-port"}) {
            const std::string f = flag;
            std::string v;
            if (arg == f) {
                if (i + 1 == argc) app::usage_error("argument " + f + ": expected one argument", "data2g-web");
                v = argv[++i];
            } else if (arg.rfind(f + "=", 0) == 0) {
                v = arg.substr(f.size() + 1);
            } else {
                continue;
            }
            if (f == "--web-listen") {
                web_listen = QString::fromStdString(v);
            } else {
                try {
                    web_port = std::stoi(v);
                } catch (const std::exception&) {
                    web_port = -1;
                }
                if (web_port < 0 || web_port > 65535) app::usage_error("argument --web-port: invalid port: '" + v + "'", "data2g-web");
            }
            ours = true;
            break;
        }
        if (!ours) rest.push_back(argv[i]);
    }
    QSettings store;
    const app::Args a = app::parse(int(rest.size()), rest.data(), app::load_settings(store), "data2g-web");
    if (a.list_modes || a.list_audio_devices || a.list_rigs)
        app::usage_error("--list-modes, --list-audio-devices and --list-rigs are data2g-host's", "data2g-web");
    const auto level = app::parse_level(a.log_level);
    if (!level) {
        std::fprintf(stderr, "data2g-web: Unknown level: '%s'\n", a.log_level.c_str());
        return 1;
    }
    app::g_level = *level;
    app::install_arq_log();
    app::quit_on_signals();

    QHostAddress address;
    if (!address.setAddress(web_listen)) app::usage_error("argument --web-listen: not an address: '" + web_listen.toStdString() + "'", "data2g-web");
    web::WebUi ui(a, store);  // a station that fails to start is reported on the page, to be fixed there
    try {
        const quint16 port = ui.listen(address, quint16(web_port));
        app::logf(app::INFO, "web UI on http://%s:%u/", web_listen.toStdString().c_str(), unsigned(port));
    } catch (const std::exception& e) {
        app::logf(app::CRITICAL, "%s", e.what());
        return 1;
    }
    QTimer quit_check;
    QObject::connect(&quit_check, &QTimer::timeout, [] {
        if (app::quit_requested()) QCoreApplication::quit();
    });
    quit_check.start(100);
    QCoreApplication::exec();
    app::log_line(app::INFO, "shutting down");
    return 0;
}
