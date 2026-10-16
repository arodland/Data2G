#include "log.hpp"

#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <map>
#include <mutex>

#include "arq/link.hpp"  // set_log_sink

namespace data2g::app {

std::atomic<int> g_level{INFO};

namespace {

std::mutex g_log_mu;

const char* level_name(int level) {
    switch (level) {
        case DEBUG: return "DEBUG";
        case INFO: return "INFO";
        case WARNING: return "WARNING";
        case ERROR: return "ERROR";
        case CRITICAL: return "CRITICAL";
        default: return "Level";
    }
}

}  // namespace

// localtime_r is POSIX; MSVC has localtime_s with the arguments swapped.
std::tm local_tm(std::time_t t) {
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    return tm;
}

void log_line(int level, const std::string& msg) {
    if (level < g_level) return;
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    const std::tm tm = local_tm(t);
    char stamp[32];
    std::strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", &tm);
    std::lock_guard lock(g_log_mu);
    std::fprintf(stderr, "%s,%03d %s %s\n", stamp, static_cast<int>(ms), level_name(level), msg.c_str());
}

void logf(int level, const char* fmt, ...) {
    if (level < g_level) return;
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    log_line(level, buf);
}

std::optional<int> parse_level(const std::string& s) {
    static const std::map<std::string, int> names = {
        {"DEBUG", DEBUG}, {"INFO", INFO}, {"WARNING", WARNING}, {"WARN", WARNING}, {"ERROR", ERROR}, {"CRITICAL", CRITICAL}, {"FATAL", CRITICAL}};
    std::string u = s;
    for (auto& c : u) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (auto it = names.find(u); it != names.end()) return it->second;
    try {
        std::size_t used = 0;
        const int v = std::stoi(s, &used);
        if (used == s.size()) return v;
    } catch (const std::exception&) {
    }
    return std::nullopt;
}

void install_arq_log() {
    arq::set_log_sink({[](const char*, int level) { return level >= g_level; },
                       [](const char*, int level, const std::string& msg) { log_line(level, msg); }});
}

namespace {
std::atomic<bool> g_quit{false};
extern "C" void on_quit_signal(int) { g_quit = true; }
}  // namespace

void quit_on_signals() {
    std::signal(SIGINT, on_quit_signal);
    std::signal(SIGTERM, on_quit_signal);
#ifndef _WIN32
    std::signal(SIGHUP, on_quit_signal);
    std::signal(SIGPIPE, SIG_IGN);
#endif
}

bool quit_requested() { return g_quit; }

}  // namespace data2g::app
