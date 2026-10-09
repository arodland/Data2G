// app::request_realtime: a thread that asks ends up real-time (SCHED_FIFO, or SCHED_RR through RealtimeKit) when the machine allows it (direct permission or
// RealtimeKit); when it doesn't, the answer says why and the thread is unchanged. Either is a pass: a CI
// container has no RealtimeKit. What must hold is that the claim matches the kernel's view, and that the
// process doesn't die (RLIMIT_RTTIME is set by the request, and a real-time thread must block regularly).

#include <QCoreApplication>
#include <chrono>
#include <cstdio>
#include <thread>

#include "app/realtime.hpp"
#include "check.hpp"

#if defined(__linux__)
#include <sched.h>
#endif

using namespace data2g;

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);  // QtDBus wants one
    app::RealtimeResult r;
    int policy = -1;
    std::thread t([&] {
        r = app::request_realtime(10);
#if defined(__linux__)
        policy = sched_getscheduler(0);
        // a real-time thread that blocks now and then: the limit is on running without blocking
        for (int i = 0; i < 20; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
#endif
    });
    t.join();
    std::printf("request_realtime: %s: %s\n", r.ok ? "granted" : "not granted", r.how.c_str());
    check::is_true(!r.how.empty(), "an answer either way");
#if defined(__linux__)
    const int base = policy & ~SCHED_RESET_ON_FORK;
    check::is_true(r.ok == (base == SCHED_FIFO || base == SCHED_RR), "the claim matches the kernel's policy for the thread");
    check::is_true(sched_getscheduler(0) != SCHED_FIFO, "the calling (main) thread was not changed");
#else
    check::is_true(!r.ok, "unsupported platforms say so");
#endif
    return check::report("test_realtime");
}
