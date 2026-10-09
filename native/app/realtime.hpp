// Real-time scheduling for the threads that move audio, so a busy machine can't starve the transmitter
// (a late block is a hole in the burst). Linux only for now: other platforms return "unsupported".
//
// What may be real-time: threads that do a little work and then block (the sound card's capture and playback
// threads, the transmit thread). Not the engine's acquisition and decode threads: they compute for hundreds of
// milliseconds at a time, which a real-time thread is not allowed to (the kernel kills a process whose
// real-time thread runs past RLIMIT_RTTIME without blocking, and rtkit insists on that limit) and which would
// starve everything else.
#pragma once

#include <string>

namespace data2g::app {

struct RealtimeResult {
    bool ok = false;
    std::string how;  // "SCHED_FIFO 20" / "SCHED_RR 20 via RealtimeKit", or why not
};

// Asks for real-time scheduling at `priority` (capped to what RealtimeKit allows) for the calling thread: SCHED_FIFO
// directly when the user's limits allow it (RLIMIT_RTPRIO, the audio group), else SCHED_RR through RealtimeKit on
// the system bus.
RealtimeResult request_realtime(int priority = 20);

}  // namespace data2g::app
