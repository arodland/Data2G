// data2g-web: the station of data2g-host, headless, with data2g-gui's
// windows as a web page (web/web_ui.hpp).
//
//   data2g-web [--web-listen ADDR] [--web-port N] [data2g-host's options]
//   data2g-web --set-password      the web UI's password (app/web_password.hpp), then exit
//
// --web-listen defaults to 127.0.0.1. The page has a password, but no TLS:
// on 0.0.0.0 it crosses the network in the clear, so use a network you trust.
// Started with no password set: on a terminal, it asks for one first;
// otherwise (launched from a desktop, a service) the page says how to set one.
// Settings come from the config file data2g-gui saves to (app::open_config()),
// then data2g-host's flags on the command line override them for this run.
// Either front end's settings changes are the other's next start: running
// both at once, they want the same ports and sound card, so one fails.

#include <QCoreApplication>
#include <QHostAddress>
#include <QSettings>
#include <QTimer>

#include <cstdio>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
// The three console calls, declared by hand rather than by including
// <windows.h> (as tests/check.hpp's SetErrorMode): its ERROR macro breaks
// app/log.hpp's level of that name, and its others <complex>. The
// signatures are consoleapi.h's and processenv.h's exactly -- HANDLE is
// void*, DWORD unsigned long, BOOL int -- so a translation unit that sees
// both still sees one declaration each.
extern "C" {
__declspec(dllimport) void* __stdcall GetStdHandle(unsigned long std_handle);
__declspec(dllimport) int __stdcall GetConsoleMode(void* console, unsigned long* mode);
__declspec(dllimport) int __stdcall SetConsoleMode(void* console, unsigned long mode);
}
#else
#include <termios.h>
#include <unistd.h>

#include <csignal>
#endif

#include "app/settings.hpp"
#include "app/web_password.hpp"
#include "web_ui.hpp"

using namespace data2g;

namespace {

bool on_terminal() {
#ifdef _WIN32
    return _isatty(_fileno(stdin));
#else
    return isatty(fileno(stdin));
#endif
}

#ifndef _WIN32
termios g_saved;  // the terminal as it was, for a ^C while echo is off
extern "C" void restore_and_exit(int) {
    tcsetattr(STDIN_FILENO, TCSANOW, &g_saved);
    _exit(130);
}
#endif

// A line from stdin, typed without echo when it is a terminal; nullopt at EOF.
std::optional<std::string> read_secret(const char* prompt) {
    std::fprintf(stderr, "%s", prompt);
    std::fflush(stderr);
    const bool tty = on_terminal();
#ifdef _WIN32
    // STD_INPUT_HANDLE and ENABLE_ECHO_INPUT, spelled out: <windows.h>'s macros, and ABI.
    constexpr unsigned long std_input_handle = static_cast<unsigned long>(-10), enable_echo_input = 0x0004;
    void* const in = GetStdHandle(std_input_handle);
    unsigned long mode = 0;
    if (tty) {
        GetConsoleMode(in, &mode);
        SetConsoleMode(in, mode & ~enable_echo_input);
    }
#else
    if (tty) {
        tcgetattr(STDIN_FILENO, &g_saved);
        termios quiet = g_saved;
        quiet.c_lflag &= ~tcflag_t(ECHO);
        std::signal(SIGINT, restore_and_exit);
        tcsetattr(STDIN_FILENO, TCSANOW, &quiet);
    }
#endif
    std::string line;
    const bool got = bool(std::getline(std::cin, line));
#ifdef _WIN32
    if (tty) SetConsoleMode(in, mode);
#else
    if (tty) {
        tcsetattr(STDIN_FILENO, TCSANOW, &g_saved);
        std::signal(SIGINT, SIG_DFL);
    }
#endif
    if (tty) std::fprintf(stderr, "\n");  // the Enter that wasn't echoed
    if (!got) return std::nullopt;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return line;
}

// Asks for a new password and saves it. On a terminal: typed twice, until
// it is not empty and the two match. Piped: one line, as it is. False at EOF.
bool ask_password(QSettings& store) {
    for (;;) {
        const auto pw = read_secret("New web UI password: ");
        if (!pw) return false;
        if (pw->empty()) {
            std::fprintf(stderr, "data2g-web: the password can't be empty\n");
            if (!on_terminal()) return false;
            continue;
        }
        if (on_terminal()) {
            const auto again = read_secret("Again: ");
            if (!again) return false;
            if (*again != *pw) {
                std::fprintf(stderr, "data2g-web: they don't match; once more\n");
                continue;
            }
        }
        app::set_web_password(store, QString::fromStdString(*pw));
        std::fprintf(stderr, "data2g-web: password saved\n");
        return true;
    }
}

}  // namespace

int main(int argc, char** argv) {
    QCoreApplication qapp(argc, argv);
    QCoreApplication::setOrganizationName(QStringLiteral("Data2G"));
    QCoreApplication::setApplicationName(QStringLiteral("data2g-web"));
    QCoreApplication::setApplicationVersion(QStringLiteral(DATA2G_VERSION));

    // Our two flags out of argv; the rest is data2g-host's.
    QString web_listen = QStringLiteral("127.0.0.1");
    int web_port = 8400;
    bool set_password = false;
    std::vector<char*> rest{argv[0]};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--set-password") {
            set_password = true;
            continue;
        }
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
    const auto shared = app::open_config(argc, argv);  // data2g-gui's too
    QSettings& store = *shared;
    if (set_password) return ask_password(store) ? 0 : 1;
    const app::Args a = app::parse(int(rest.size()), rest.data(), app::load_settings(store), "data2g-web",
                                   "\ndata2g-web's own options:\n"
                                   "  --web-listen ADDR     the web UI's address (default 127.0.0.1; no TLS, so 0.0.0.0\n"
                                   "                        only on a network you trust)\n"
                                   "  --web-port N          the web UI's port (default 8400; 0 picks a free one)\n"
                                   "  --set-password        set the web UI's password, then exit\n");
    if (a.list_modes || a.list_audio_devices || a.list_rigs)
        app::usage_error("--list-modes, --list-audio-devices and --list-rigs are data2g-host's", "data2g-web");
    const auto level = app::parse_level(a.log_level);
    if (!level) {
        std::fprintf(stderr, "data2g-web: Unknown level: '%s'\n", a.log_level.c_str());
        return 1;
    }
    app::g_level = *level;
    app::install_arq_log();

    if (!app::has_web_password(store)) {
        if (on_terminal()) {
            std::fprintf(stderr, "data2g-web: the web UI has no password yet. Choose one now; it is saved\n"
                                 "(change it later with data2g-web --set-password, or in data2g-gui's settings).\n");
            if (!ask_password(store)) std::fprintf(stderr, "\n");
        }
        if (!app::has_web_password(store))
            app::log_line(app::WARNING, "web UI: no password set: the page is locked until one is (data2g-web --set-password)");
    }
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
