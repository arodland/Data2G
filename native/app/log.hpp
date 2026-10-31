// The apps' log: Python's basicConfig(format="%(asctime)s %(levelname)s %(message)s"),
// as host.py's main() sets it up, on stderr; and the quit signals.
//
// Plain C++ (data2g_applog, no Qt): data2g-host, data2g-gui (through
// data2g_station) and data2g-replay log alike.
#pragma once

#include <atomic>
#include <ctime>
#include <functional>
#include <optional>
#include <string>

namespace data2g::app {

enum Level { DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40, CRITICAL = 50 };
extern std::atomic<int> g_level;
void log_line(int level, const std::string& msg);
[[gnu::format(printf, 2, 3)]] void logf(int level, const char* fmt, ...);
// Also hands every message logged (without the stderr line's time and level) to `sink`,
// under the log's lock: it must not block or log. Empty: stderr only. A front
// end with no terminal (Android) shows the log this way.
void set_log_sink(std::function<void(int level, const std::string& msg)> sink);
// --log-level's value -> a level; nullopt: not one.
std::optional<int> parse_level(const std::string& s);
// Routes the arq log (sessions, engine) through log_line at g_level.
void install_arq_log();
// SIGINT, SIGTERM and SIGHUP (a closed terminal) set quit_requested() instead of ending the process, so the exit
// path runs and the rig is unkeyed. SIGPIPE is ignored: a pipe or socket whose reader went is an error, not death.
void quit_on_signals();
bool quit_requested();

// localtime_r / localtime_s, whichever this platform has.
std::tm local_tm(std::time_t t);

}  // namespace data2g::app
