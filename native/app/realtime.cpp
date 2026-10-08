#include "app/realtime.hpp"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>

#ifdef DATA2G_HAVE_QTDBUS
#include <QDBusConnection>
#include <QDBusError>
#include <QDBusInterface>
#include <QDBusReply>
#include <QVariant>
#endif
#endif

namespace data2g::app {

#if defined(__linux__)
namespace {

// A real-time thread may use this much CPU without blocking before the kernel kills the process (RLIMIT_RTTIME);
// RealtimeKit refuses to promote a process without one at or under its own cap (200 ms by default).
constexpr rlim_t RTTIME_US = 200000;

void limit_rttime() {
    rlimit rl{};
    if (getrlimit(RLIMIT_RTTIME, &rl) != 0) return;
    if (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur <= RTTIME_US && rl.rlim_max != RLIM_INFINITY && rl.rlim_max <= RTTIME_US) return;
    rl.rlim_cur = rl.rlim_max = std::min<rlim_t>(RTTIME_US, rl.rlim_max == RLIM_INFINITY ? RTTIME_US : rl.rlim_max);
    setrlimit(RLIMIT_RTTIME, &rl);  // lowering a hard limit is allowed
}

}  // namespace

RealtimeResult request_realtime(int priority) {
    limit_rttime();
    sched_param sp{};
    sp.sched_priority = priority;
    const int direct_error = pthread_setschedparam(pthread_self(), SCHED_FIFO | SCHED_RESET_ON_FORK, &sp);  // an errno value
    if (direct_error == 0) return {true, "SCHED_FIFO " + std::to_string(priority)};
#ifdef DATA2G_HAVE_QTDBUS
    QDBusConnection bus = QDBusConnection::systemBus();
    if (!bus.isConnected()) return {false, "no system bus for RealtimeKit (direct: " + std::string(std::strerror(direct_error)) + ")"};
    QDBusInterface rtkit("org.freedesktop.RealtimeKit1", "/org/freedesktop/RealtimeKit1", "org.freedesktop.RealtimeKit1", bus);
    if (!rtkit.isValid()) return {false, "RealtimeKit is not running (direct: " + std::string(std::strerror(direct_error)) + ")"};
    const int cap = rtkit.property("MaxRealtimePriority").toInt();
    if (cap > 0) priority = std::min(priority, cap);
    const QDBusReply<void> reply = rtkit.call("MakeThreadRealtimeWithPID", static_cast<qulonglong>(getpid()),
                                              static_cast<qulonglong>(syscall(SYS_gettid)), static_cast<uint>(priority));
    if (!reply.isValid()) return {false, "RealtimeKit refused: " + reply.error().message().toStdString()};
    return {true, "SCHED_RR " + std::to_string(priority) + " via RealtimeKit"};  // it grants round-robin real-time
#else
    return {false, std::string("not permitted (") + std::strerror(direct_error) + "); built without D-Bus for RealtimeKit"};
#endif
}

#else

RealtimeResult request_realtime(int) { return {false, "real-time scheduling is not supported on this platform yet"}; }

#endif

}  // namespace data2g::app
