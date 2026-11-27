# Embedding Data2G

The HF modem as a static library for an application that owns the sound card
and the PTT, such as a mail client on a phone. The application passes received
audio in and gets back transmit audio, PTT requests, and the lines and bytes of
a VARA host session. There is no Qt in it, no socket, no file, and nothing runs
between the application's calls.

It is `native/embed/`: the same `arq::Engine` and `host::Host` that
`data2g-host` runs, stepped by the caller instead of by the station's threads.

| File | What |
|---|---|
| [`native/embed/include/CData2G/data2g.h`](../native/embed/include/CData2G/data2g.h) | The C API (`d2g_*`). Plain C99, the whole public surface. |
| `native/embed/modem.{hpp,cpp}` | `embed::Modem`: resampling, stepping, keying, reports. C++ callers may use it directly. |
| `native/embed/data2g_c.cpp` | The C boundary: calls from inside callbacks, exceptions, `d2g_free`. |
| `native/packaging/apple/` | The XCFramework build and the Swift package `CData2G`. |

## In short

```c
D2gConfig cfg;
d2g_config_default(&cfg);          /* 48 kHz, 100 ms PTT lead, 50 ms tail */

D2gCallbacks cbs = {
    .tx_audio  = on_tx_audio,      /* one transmission, whole: mono float */
    .tx_flush  = on_tx_flush,      /* drop what is queued; it was not played */
    .ptt       = on_ptt,           /* may block */
    .host_line = on_host_line,     /* CONNECTED ..., BUFFER 120, PTT ON, ... */
    .host_data = on_host_data,     /* the session's bytes */
    .log       = on_log,           /* optional */
    .tx_report = on_tx_report,     /* optional */
    .rx_report = on_rx_report,     /* optional */
};

D2gModem *m = d2g_new(&cfg, &cbs, ctx);   /* NULL: cfg or cbs refused */
d2g_command(m, "MYCALL N0CALL");
d2g_command(m, "BW2300");
d2g_command(m, "LISTEN ON");

/* On the modem's thread, for each chunk of captured audio: */
d2g_tx_played(m, total_played);    /* transmit samples played so far */
d2g_rx_audio(m, samples, n);       /* decodes, and is the clock */

/* While there is no captured audio (keyed on a half-duplex path, a stall): */
d2g_tx_played(m, total_played);
d2g_advance(m, n);                 /* the clock alone */

d2g_free(m);
```

`native/embed/tests/smoke.c` is a complete C program;
`native/tests/test_embed.cpp` and
`native/packaging/apple/Tests/CData2GTests/LoopbackTests.swift` run two modems
back to back through a noisy channel, in C++ and in Swift.

## Threads

Make every `d2g_*` call from one thread or one serial queue. Do not make them
from a real-time audio callback: one `d2g_rx_audio` call can decode a whole
burst, which allocates, takes hundreds of milliseconds, and may then wait in
`ptt`.

The audio callbacks only move samples:

- The capture callback copies its samples into a ring buffer. The modem's
  thread takes them out and gives them to `d2g_rx_audio`, 10 to 100 ms at a time.
- `tx_audio` copies the transmission into a second ring buffer. The render
  callback plays from it and adds what it played to an atomic counter.
- The modem's thread reads that counter and passes it to `d2g_tx_played`
  before each `d2g_rx_audio` or `d2g_advance`.
- `tx_flush` runs on the modem's thread and empties the second ring buffer.

Decoding uses a pool of worker threads (`D2gConfig.decode_threads`; 1 keeps it
on the calling thread). They work only while a call is in progress.

## The contract

- **One call at a time.** Every callback runs on the calling thread, inside
  the call that caused it, before that call returns.
- **The sample clock.** Modem time is the count of samples given to
  `d2g_rx_audio` plus those given to `d2g_advance`. Every timer runs on it:
  the session's retries and timeouts, the PTT release, the watchdog. Keep it
  running at the real rate. If the capture stops (many radio links are
  half-duplex and deliver nothing while keyed), call `d2g_advance` with the
  samples that would have arrived. A clock that stops while the radio is keyed
  leaves it keyed.
- **Receive audio.** Mono float at `D2gConfig.sample_rate`, which is 8000 Hz
  times a whole number (48000 by default; 8000 skips the resampler). Set the
  level as for any sound card modem: band noise well above digital silence,
  peaks clear of ±1.0. Chunks may be any size. While the modem is transmitting it
  hears silence in place of what it is given, so the application need not gate
  its capture.
- **Transmit audio.** `ptt(true)` comes first, then `tx_audio` with the whole
  transmission, which starts with `ptt_on_delay_s` of silence. Play it at
  once. A second `tx_audio` can follow while the first is still playing and no
  new `ptt(true)` comes with it: play it straight after. A burst peaks at
  1.0 times `tx_gain` (and the host's `TXPOWER`), clipped to ±1.0. The buffer
  is valid during the callback only.
- **`d2g_tx_played`.** A running total, since `d2g_new`, of transmit samples
  that have really gone out. The modem needs it for two things. It releases
  the radio `ptt_off_delay_s` after the last sample was played, on the next
  `d2g_rx_audio` or `d2g_advance`. And it measures how far playback runs
  behind its own clock, so that it listens for the reply at the right time.
  Report often and honestly: a total that runs late lengthens every
  turnaround.
- **`tx_flush`.** An abort (the `ABORT` command, or the session's own), the
  watchdog, a fault and `d2g_free` ask the application to drop its queued
  transmit audio. Never add dropped samples to the played
  total: the modem accounts for them. It takes the total it holds when
  `tx_flush` returns as exact, so if the render callback may have played more
  than was last reported, call `d2g_tx_played` from inside `tx_flush`, after
  emptying the queue.
- **`ptt` may block**, to wait for the radio to confirm. The sample clock
  just waits with it.
- **The watchdog.** Transmit audio not reported played within its length plus
  3 s (of the sample clock) is flushed and the radio released, with a
  `warning:` log line. It is what stops a stuck audio path from holding the
  transmitter on.
- **From inside a callback** `d2g_command`, `d2g_send`, `d2g_tx_played` and
  `d2g_free` are allowed. Commands and data are queued and run, in order,
  before the outer call returns. `d2g_rx_audio` and `d2g_advance` from inside
  a callback are ignored and logged.
- **`d2g_free`** calls `tx_flush` and `ptt(false)`, then frees the modem.
  These are always the last two callbacks. From inside a callback it only
  marks the modem: no other callback fires after that, and the flush, the
  un-key and the free happen when the outer call returns. `ctx` must stay
  valid until then.
- **Faults.** No C++ exception leaves the library. One caught at the boundary
  (out of memory, a bug) ends the modem: `tx_flush`, `ptt(false)`, an
  `error:` log line, and every later call except `d2g_free` does nothing.
  An exception thrown by one of the application's own callbacks is treated the
  same way.

## Host commands

`d2g_command` takes the lines a VARA client writes to the command port,
without the CR, and `host_line` gets the port's lines back. Each command is
answered (`OK`, `WRONG`, or its own reply) before `d2g_command` returns.
`native/core/host/host.hpp` has the full list; these are what a mail client
needs:

| Command | |
|---|---|
| `MYCALL call` | This station's call sign, answered `OK` and `REGISTERED call`. Send it before `LISTEN ON`. |
| `BW500`, `BW1200`, `BW2300` | The widest signal to use. |
| `LISTEN ON`, `LISTEN OFF` | Answer incoming calls, or not. |
| `CONNECT me peer` | Call a station. `CONNECTED me peer BW2300` when it answers, `DISCONNECTED` when it does not. |
| `RETRIES n` | How many times `CONNECT` calls again (4 unless set, 20 at most). |
| `DISCONNECT` | End the session gracefully: the far station is told, and what is queued still goes. |
| `ABORT` | End it now. A transmission in progress is flushed and the radio released before `d2g_command` returns. |
| `TXPOWER db` | After `DATA2G ON`: the output level in dB, from -60 to 0, on top of `tx_gain`. |
| `VERSION` | `VERSION Data2G 0.5.3`. |

Lines that arrive on their own: `CONNECTED`, `DISCONNECTED`, `BUFFER n`,
`PTT ON` and `PTT OFF` (beside the `ptt` callback), `MODE name` (the mode of a
transmission, when it changes), `BUSY ON` and `BUSY OFF` (the channel is
occupied), and `IAMALIVE` once a minute.

`BUFFER n` is what a VARA client expects, not a plain byte count. It is 0 once
the far station has acknowledged everything given to `d2g_send`. Above 0 it
leaves out what the next burst will carry (`native/core/host/host.cpp` has
the reasons).

`ACCEPT` is answered `WRONG`: it belongs to the TCP host. A line is cut at the
first CR or LF, and a byte outside ASCII becomes `?`.

`d2g_send` is the data port: its bytes go to the far station once connected
and theirs arrive through `host_data`.

## Reports

`tx_report` fires as each burst is keyed and `rx_report` once the session has
read a burst it heard. Both are optional and meant for a display: mode, width,
codewords, time on air, bytes and bit rate.

- `D2gTxReport.bytes` counts the bytes of the session's stream first sent in
  that burst. A burst that only repeats earlier codewords reports 0. Summed
  over a session it is what was sent, as compressed on the wire.
- `D2gRxReport.bytes` counts the payload of the codewords new to the stream.
  A codeword counts whole, so the last one of a message adds its padding and
  the sum comes out a little above the sender's.
- `net_bps` is `bytes` over the burst's time on air. It leaves out the turns
  between bursts, so a session's throughput is lower.
- `d2g_max_net_bps(bandwidth)` is the most any report's `net_bps` can reach
  within a bandwidth: the fastest mode there, its longest burst, every
  codeword full. It is a gauge's full scale, not a forecast. Today it is 1965
  bit/s for both `D2G_BW_500` and `D2G_BW_1200` (the fastest mode that fits
  1200 Hz is a 500 Hz one) and 7899 bit/s for `D2G_BW_2300`.
- A nod (the short keyed reply) is reported with the mode `"nod"`. A burst
  whose header was heard and whose body was not has `lost` set.

## Building

The library is the CMake target `data2g_embed`, a static library on top of
`data2g_core`. It is part of every HF build (`DATA2G_BUILD_EMBED`, default ON),
with two ctests: `embed` (sessions through the C API) and `embed_smoke` (a C99
program). To build it alone, with no Qt installed:

```sh
cmake -S native -B native/build-embed -DCMAKE_BUILD_TYPE=Release \
  -DDATA2G_BUILD_TOOLS=OFF -DDATA2G_BUILD_PYMODULE=OFF -DDATA2G_BUILD_HOST=OFF \
  -DDATA2G_BUILD_GUI=OFF -DDATA2G_BUILD_QTAUDIO=OFF -DDATA2G_BUILD_RIG=OFF
cmake --build native/build-embed --target test_embed test_embed_smoke --parallel
ctest --test-dir native/build-embed -R '^embed'
```

Link `libdata2g_embed.a`, `libdata2g_core.a`, zlib and the C++ runtime, and
put `native/embed/include` on the include path (`#include "CData2G/data2g.h"`).

### Apple

```sh
native/packaging/apple/build-xcframework.sh          # all three slices
native/packaging/apple/build-xcframework.sh --test   # ... then swift test
SLICES="macos" native/packaging/apple/build-xcframework.sh
```

It makes `native/packaging/apple/build/Data2G.xcframework` with the slices
`ios-arm64`, `ios-arm64_x86_64-simulator` and `macos-arm64_x86_64`, each one
`libdata2g.a` (the embed library and the core merged) with the header and a
module map, plus a zip and its SwiftPM checksum. It needs Xcode and CMake.

`native/packaging/apple/Package.swift` is the Swift package `Data2G` with the
binary target `CData2G`. Add it to an app as a local package after building
the XCFramework, and `import CData2G`. The module map links libc++ and libz,
so the app needs no linker flags.

The deployment targets are iOS 16.3 and macOS 13.3, the first whose libc++ has
the floating-point `std::to_chars` the core uses.

CI (`native-build.yml`, job `embed`) builds the library without Qt, runs the
two ctests, builds the XCFramework and runs the Swift tests. It uploads the zip
as a workflow artifact and attaches nothing to a release.

## Limits

- **One modem per process.** The log, the decode pool's size and the decoder's
  iteration cap are process-wide. A second modem works (the tests run two) but
  shares them: the first one created gets both modems' log lines.
- **HF only.** The FM profile has another sample rate and other bandwidths,
  and `data2g_embed` is not built with `-DDATA2G_PROFILE=fm`.
- **Decoding runs in the caller's call**, so the modem's thread must keep up
  with real time on average. A capture ring buffer of a second or two absorbs
  one slow decode. `decoder_effort` and `dd_budget_s` trade sensitivity for
  time on a slow device.
- **No rig control and no KISS port.** The application owns the radio, and
  broadcast over KISS (`docs/broadcast.md`) has no entry point here.

## Licence

Data2G is under the [Artistic License 2.0](../LICENSE) with the
[Data2G Interface Exception](../LICENSE_exception). Section 8 covers embedding
the modem in a larger work, and the exception covers an application that also
offers the modem's interfaces to other programs, provided it is a significant
work of its own. Sections 4 to 6 apply to distributing a modified version, and
`native/embed/` in a fork is one. Read both files before shipping an app.
