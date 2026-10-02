# Plan: native C++ / Qt 6 port

Approved 2026-10-02 with all four recommendations, plus two tweaks (more
threads where they cut latency; Android kept possible). Based on master
157ca38.

TLDR: port the live path (`data2g/` minus the torch and channel modules,
about 7.9k lines) to C++20 / Qt 6 under `native/`, using SSTVAE's stack and
lifting its code where the waveform is shared. Python stays the normative
reference and the home of every study. Parity is checked the SSTVAE way:
golden vectors plus `pytest --native`, which runs the existing suite with C++
functions substituted in.

Decisions (approved 2026-10-02):
1. Same repo, `native/` beside `data2g/`.
2. Lift SSTVAE `native/` code by copying, not a shared library.
3. Rig control: bundled Hamlib as in SSTVAE (model 2 still reaches an
   external rigctld), from Phase 3. Until then, the rigctld TCP client.
4. GUI scope for Phase 4: a status window (waterfall, link state, mode,
   throughput, settings dialog). Android is not planned, but nothing may
   foreclose it (see "Android stays possible").

## Why

Same reasons as SSTVAE (`~/code/SSTVAE/docs/native-app.md`, "Audio: use
QtMultimedia"). A Python audio callback needs the GIL on the host's realtime
thread. Data2G already shows it:
- c2cdb75: TX moved to a callback FIFO because a main thread holding the GIL
  made 1024-frame buffers late.
- 157ca38: a blocking capture read dropped 35-51 s of input per session, with
  450-2300-sample holes inside bursts. The FIFO turns that into latency, but
  `paInputOverflow` still fires when the callback itself is late.
- host.py:395: stereo is forced because PortAudio's ALSA backend corrupted
  the heap on mono over PipeWire.

Everything (demod, decode, ARQ, gear shift) runs inside `engine.step` on the
one main thread, so a slow decode delays both capture draining and replies.
In C++ audio lives on its own threads and never waits for decode.

## Scope

Ported (line counts from 157ca38):

| Group | Files | Lines |
|---|---|---|
| Modem core | modem, cpm, equalizer, constellation, codes, ldpc, polar, config, waveform/{ofdm,sync,dsp} | 3778 |
| ARQ | arq/{engine,session,link,frames,phy,policy,predictor,modes} | 2627 |
| Host | host, tnc, kisslink | 1482 |

Stays Python only: channel_torch, decoders_torch, hfchannel, scripts/,
tools/freeze_format.py, training. Studies can import the native module for
speed later. That is a bonus, not a goal.

Goal for studies: every study script runs unchanged on the C++ through
`tools/with_native.py` (no `--skip`), so a result (speed test, loss study)
can be re-run on the port and shown to hold. Status 2026-10-02: gap closed
for every script that drives sessions or the engine. The wrappers now
honour what scripts reach into: an instance override of
`Session._on_timeout` (linksim counts timeouts), session.py's patched
constants (idle_study's variants, read when a Session is made), a repeated
burst returned as the same object (linksim keys on identity), and
`phy.ModemRx` is a class whose methods a study can wrap (crc_exposure).
Checked 2026-10-02, Python vs C++, same seeds, DD budget inf: identical
rows for loss_study, phy_session, session_data, outcome_data,
crc_exposure, sync_loss_study; idle_study 23 of 24 rows (one session
diverges at a single borderline codeword decode, slot 13 of a 19-cw
w48-16qam-r1/2 burst, C++ fails where Python decodes, DD on or off).
What remains:
- `crc_exposure_engine.py` fails on Python too (its ModemRx.__init__ patch
  predates the dd_budget argument). The C++ Engine also has no
  `_start_tx` and never calls Python's ModemRx, so its counts need engine
  hooks once the script is fixed.
- `linksim.py run` fails on Python too (stale `run()` call). Its sweeps
  hard-code 8 workers. Its session loop is covered through phy_session's
  `sim_score`.

Python remains the definition of the on-air format. When the two disagree,
Python is right until shown otherwise. New DSP lands in both.

## Stack

Same as SSTVAE unless noted.

| Need | Choice | Source |
|---|---|---|
| Build | CMake >= 3.24, C++20, `-fno-fast-math` / `/fp:precise` | SSTVAE top-level CMakeLists |
| FFT | pocketfft (scipy.fft is pocketfft) | lift `core/dsp/fft.hpp` |
| firwin, hilbert, fftconvolve | hand-written | lift `core/dsp/dsp.cpp` |
| Audio | Qt Multimedia, capture and playback each on a QThread | lift `core/audio/` |
| Ring buffer | single-writer wait-free | lift `core/rx/ringbuffer.hpp` |
| Rig | Hamlib (decision 3) | lift `core/rig/`, `cmake/hamlib.cmake` |
| Deflate with dictionary | zlib, vendored | new. Qt's qCompress can't do raw deflate with a preset dictionary; miniz can't either |
| TCP (VARA 8300/8301, KISS 8100) | QtNetwork on the main event loop | new |
| JSON (clip_constants, settings) | QJsonDocument | |
| Data files | generated C++ tables compiled into `core/` (see "Android stays possible"); `npy.hpp` only in tests | `tools/gen_data_tables.py` |
| Tests | hand-rolled `check.hpp` + ctest | lift |
| Parity | pybind11 module | lift the pattern from `bindings/module/` |
| GUI | Qt Widgets, optional (`DATA2G_BUILD_GUI=AUTO`) | pattern from `gui/` |

No Eigen, no BLAS. The matrices (header ML correlation, LMMSE, MLP 28x64x64x96)
are small enough for plain loops.

Headless: `data2g-host` is a QCoreApplication with the same CLI as host.py
`main`. It links no Widgets. `data2g-gui` is a second executable over the same
core.

## Threads

- Capture QThread: device rate -> ring buffer at 8 kHz (stateful resampler).
- Playback QThread: pulls from a TX FIFO.
- Engine thread (std::thread): reads the ring, runs `Engine::step`, writes TX
  audio, posts events. Sample-clocked like the Python engine, so its timing
  logic ports unchanged.
- Main thread: Qt event loop, TCP sockets, rig. Talks to the engine through
  two queues (commands in, events out).

More threads are welcome where they cut latency (approved 2026-10-02), as
long as the count stays moderate. Candidates, each measured against reply
latency before it is kept:
- Decode worker: burst decode (equalize, LDPC/polar, DD) off the engine
  thread, so preamble search and BUSY keep running during a 1 s DD pass. The
  engine stays sample-clocked; a decode result arrives as an event stamped
  with the sample position it belongs to.
- Codeword-parallel LDPC/polar decode within a burst.
- CFO-grid parallel sync correlation.

One shared pool, sized min(4, cores/2) and settable, because the machine is
shared and a modem must not saturate a laptop either.

Status 2026-10-02: the pool landed (`core/util/pool`). Fixed threads, one
job at a time (a second caller, or a nested call, runs inline),
`parallel_for` over indices whose arithmetic never depends on the thread.
Size: `pool::set_threads`, `data2g-host --threads N`, `DATA2G_THREADS`
(`tools/with_native.py` sets 1, so forked study workers don't
oversubscribe), `data2g_native.set_threads`. Used by LDPC (codewords within
each iteration; the batch still stops together), polar (rows),
`equalizer::refine` (data rows, then frames) and the sync CFO grid (only
from 6000 samples: a StreamDetector hop went 1.2 -> 0.55 ms at +15% CPU,
every 250 ms per band, so hops stay serial). `test_pool` checks every
part bit for bit at sizes 1, 2, 4 and 8 (LDPC posteriors, a whole receive
and DD decodes included), clean under TSan (`-DDATA2G_TSAN=ON`).

## What must be frozen, not ported

SSTVAE's rule: generate format constants, don't port the algorithms that make
them.

| Item | Where | Freeze as |
|---|---|---|
| CPM header tone sequences, numpy PCG64 on air | cpm.py:220 | table per (m, rate, word) in codes_data |
| Interleavers | codes.py:149 | already frozen in format/*.npz |
| ACE directions | constellation.py:70 (ConvexHull) | per-constellation table |
| `gamma.ppf(0.99, n) / n` | equalizer.py:176 | table over the n values used |
| Decimator / interpolator / TX bandpass taps | tnc.py:504, host.py:210, dsp.py:102 | either the lifted firwin (SSTVAE already matches scipy) or tables |
| config.py constants | config.py | generated `native/core/config.hpp`, `--check` in CI |

Packaging bug found on the way: `format/*` is missing from
`[tool.setuptools.package-data]`, so an installed wheel silently recomputes
interleavers. Fix it in Phase 0.

Not bit-exact, by design: `random.Random` keepalive jitter and session keys;
deflate output (only the peer decodes it, so interop is the test, not bytes).

Bit-exact where IEEE allows (integers, `+ - * /`, seeded draws). Tolerance plus
fingerprint for anything through exp, FFT or reductions: equalizer, sync
statistics, LLRs. Float32 must stay float32 in LDPC, polar and header
correlation (CH_CLAMP, `_phi` floor, BIG), or decode decisions drift.

`DD_BUDGET_S` is wall-clock, so decode results depend on CPU speed. C++ will
be faster and finish more DD passes. Compare with the budget disabled; keep it
on air.

## Phases

Sized like SSTVAE: volume, what verifies it, what needs you.

### Phase 0: scaffolding and parity harness

- `native/` CMake, lifted check.hpp, npy.hpp, pocketfft, pybind11 module
  skeleton, `tools/gen_config_header.py`, `tools/gen_golden_vectors.py`,
  `tools/check_layering.py`, `tests/conftest.py --native`.
- Freeze the tables above. Fix the package-data bug.
- Port codes.py's CRC, PN9 scrambler and interleaver as the end-to-end proof.
- Volume: about 500 lines Python displaced; harness mostly lifted.
- Verified by: `pytest --native` green with those substituted.
- Needs you for: decisions 1-4.

Status 2026-10-02: met on Linux.
- `native/`: `data2g_core` (codes: CRC-16/24/32, `with_crc`, PN9
  scrambler, scramble seeds, frozen interleavers and polar info sets),
  `test_codes` (known answers, no Python), the `data2g_native` module.
- `tools/gen_native_tables.py` writes `native/core/generated/config.hpp`
  (constants, 4 bands, 42 submodes with derived sizes) and `format.cpp`
  (interleavers, info sets); `--check` runs in `build_native.sh` and in
  `test_native_parity.py`.
- `pytest --native`: 5 substitutions in `data2g.codes`; 438 fast tests pass,
  as without it. `test_native_parity.py`: 47 direct comparisons.
- ASan/UBSan build clean. `check_layering.py` in place.
- Package-data fix landed.

Moved out of Phase 0, each to the phase that first consumes it:
- Golden-vector corpus: Phase 1, with the first ctest that needs bulk
  reference data (a modem round trip). Phase 0's ctest uses known answers.
- CPM tones, ACE directions, `gamma.ppf` and filter-tap tables: Phase 1,
  generated alongside the modules that read them.
- pocketfft: Phase 1, with ofdm and sync.
- CI workflows: Phase 5 as planned; until then `tools/build_native.sh --test`.

### Phase 1: modem core

ofdm, sync (StreamDetector), dsp, constellation, ldpc, polar, equalizer, cpm,
modem (burst modulate/receive, header ML, find_burst/find_copy).
- Volume: 3.3k lines.
- Verified by: the whole fast suite under `--native` (test_modem, test_ldpc,
  test_polar, test_cpm, test_equalizer, test_dd, test_preamble,
  test_sstvae_golden), plus a paired ladder: same seeds, Python vs C++
  receiver, decode rates within noise at the 1% and 10% points.
- Needs you for: nothing.

Status 2026-10-02: met.
- Ported: constellation, ldpc, polar, cpm, waveform (ofdm, dsp, sync),
  equalizer, codes, modem. 1145 fast tests pass under `--native` with
  every module substituted, as without.
- Decisions match Python exactly on every parity vector; floats to
  1e-9..1e-13 of scale (each module's test states its tolerance and why).
- Frozen as generated tables: constellations and ACE directions, LDPC
  shifts, polar GA design for the CPM control code (the one live non-frozen
  polar code), CPM header tones and interleavers (numpy PCG64 on air),
  `gamma.ppf`, header codes, modem constants, clip constants.
- Speed, single thread, vs numpy: polar 7-9x, equalizer.estimate 6x,
  live receive 1.6x, DD pass 2.2x, LDPC 1.2-1.3x, sync and CPM about
  equal (FFT- and libm-bound). See "Performance follow-ups".
- Paired check: `scripts/loss_study.py` (real-modem sessions) instead of
  the ladder, whose simulator decodes with torch, not the live receiver.
  Default 5 cells, 12 seeds x 600 s, `shift+cpm`, `DATA2G_PEP_REF_DB=5`,
  DD budget infinite on both sides (`tools/with_native.py --dd-budget inf
  --skip arq`; the study reaches into Session internals). All 5643 burst
  rows identical; wall time 199 s Python, 78 s C++ (6 workers).
  Re-run without `--skip` (C++ sessions too), default cells, `shift`,
  4 seeds x 300 s: all 967 burst rows identical; 37 s Python, 14 s C++
  (6 workers, peak RSS 263 MB per process Python, 163 MB C++).

### Phase 2: ARQ and engine

frames (zlib + zdict + history), session, link, phy, policy, predictor,
modes, engine, recorder.
- Volume: 2.6k lines.
- Verified by: test_arq*, test_engine under `--native`; two C++ engines and
  mixed C++/Python engines through the simulated channel; a paired loss study
  with `DATA2G_PEP_REF_DB` set. The ARQ state-agreement fuzz (loss plus
  duplication) runs on the mixed pair.
- Needs you for: nothing.

Status 2026-10-02: frames, link, session, phy, policy, predictor, modes,
kisslink and the tnc Receiver are ported and substituted. Python and C++
stations exchange identical bursts in both mixed pairings, with loss and
duplication.
Engine and recorder landed (`core/arq/engine.*`).
- `--native` substitutes the C++ Engine (sync mode) for engine.Engine;
  `test_native_engine.py` runs a C++ engine against a Python one both
  ways (VARA session, KISS) and reads both recordings alike.
- Decode worker (`EngineConfig::worker`): the receiver stage (search,
  BUSY) stays in step(); burst receive, DD and the session run on one
  worker, blocks in order, so session times equal sync mode's and only
  the output's lag behind input varies (as Python's backlog does). Not
  deterministic, so sync stays the default and the parity mode. With a
  1 s decode in flight, the next header's BUSY was 0.23 s late in sync
  mode and 0.02 s with the worker (`test_engine`, real time).
- State-agreement fuzz (`tests/test_native_fuzz.py`): seeded runs in all
  four pairings at link, session and engine level, with asymmetric loss,
  duplication, late (reordered) bursts, delayed decode, long fades, a
  restarting peer, and CRC-valid corrupted control. Every link and session
  pairing sends Py-Py's bursts; no stall past the watchdog; streams exact
  except the reference behaviours under Findings. Default 386 runs x 4
  pairings in ~18 s; `-m slow` about 3800 more.
- Loss study: paired without `--skip`, rows identical (Phase 1, "Paired check").

### Phase 3: headless host

Audio, VARA command and data ports, KISS server and kisslink, PTT,
Decimator/Interpolator, Blanker, `--record-dir` in the same format, all CLI
flags.
- Volume: 1.5k lines plus the lifted audio and rig layers.
- Verified by: test_host / test_tnc behaviour against a fake device; then Pat
  through the C++ host to a Python host over the VARA Wine / PipeWire
  loopback, both directions. Overflow and backlog counters stay at zero for a
  full session.
- Needs you for: the loopback session check and the first on-air contact.

Status 2026-10-02: device layer landed (no host yet).
- `core/audio/`: CaptureFifo / PlaybackFifo (host.py Capture / Player
  semantics, SPSC rings, no lock across a copy), CapturePipeline (channel
  0, Decimator on the capture thread), device selection, Decimator /
  Interpolator / Blanker. Tested against a fake card (`test_audio`), clean
  under TSan; parity in `test_native_audio.py` and `--native` substitutions.
- `core/audio/qt/` -> `data2g_audio_qt` (`DATA2G_BUILD_QTAUDIO`), and
  `data2g-audio-check` (list devices, tone loop; not in CI).
- `core/rig/`: SSTVAE's RigController (plus poll interval 0 = key only) and
  a Keyer (key, lead, drain, off delay, unkey). `core/rig/hamlib/` ->
  `data2g_rig` (`DATA2G_BUILD_RIG`, Hamlib 4.7.2 pinned via
  `cmake/hamlib.cmake`); `test_rig_hamlib` keys a spawned `rigctld -m 1`
  through model 2.

Status 2026-10-02: headless host landed (`data2g-host`, not yet on air).
- `core/host/`: host.py's Host (commands, notifications, BUFFER credit).
  `test_native_host.py` runs a scripted pair with the Python and the C++
  Host over the same engines: every line and byte identical. `--native`
  substitutes it (test_host, test_engine).
- `apps/data2g_host.cpp`: host.py main's flags, plus `--decode-worker`
  (default on) and `--audio-io pipe:IN,OUT`. Threads: main (Qt loop, TCP
  ports), engine (capture -> step -> Keyer), decode worker (session stage
  and the Host, reached by `post()`), audio, rig.
- `core/audio/pipe.*`: raw float32 8 kHz files or named pipes at real
  time. `test_native_host_e2e.py`: two hosts over two mkfifo pipes, VARA
  session (2 kB each way) and KISS both ways, worker on and off, no
  overflow/underrun/backlog lines.
- `--threads N`: the shared pool's size (see "Threads").
- Not yet checked: sound cards through the host, Hamlib PTT, Pat.

### Phase 4: GUI

Status window and settings dialog over the same core. Exit: you use it for a
session.

Status 2026-10-02: `data2g-gui` landed (not yet used for a session).
- `app/station.*` (`data2g_station`, Qt Core + Network): the options, the
  servers, the engine thread, audio and PTT, moved out of
  `apps/data2g_host.cpp`. Both apps run it; data2g-host's CLI and output
  are unchanged. It also keeps what a front end polls: link snapshot
  (after every block), burst log (`Engine::set_on_burst`), PTT, BUSY,
  audio counters, a 4096-sample tap of the 8 kHz input.
- Found on the way: a `Port`/`KissServer` destroyed with a client still
  connected ran its disconnect handler on destroyed members (heap
  corruption on a GUI restart; at exit in data2g-host). Fixed.
- `gui/` (`DATA2G_BUILD_GUI` AUTO/ON/OFF; needs Widgets, Network,
  Multimedia): waterfall and level meter lifted from SSTVAE (0-4 kHz,
  a row per new engine block), link state, mode and width, BUSY/PTT,
  throughput over `--stats-interval`, burst log, audio counters. The
  settings dialog persists to QSettings; data2g-host's flags override
  it per run; OK with changes restarts the station (each run records to
  its own directory).
- `test_gui` (ctest, offscreen, pipe audio): a CQ burst from a second
  Engine lands in the burst log, mode and BUSY follow, LISTEN ON shows
  listening, a restart re-listens; settings round-trip through QSettings
  and the dialog. Writes `native/build/data2g_gui_shot.png`.
- `check_layering.py`: Qt Widgets only under `gui/`.

### Phase 5: packaging and CI

Lift SSTVAE's ci.yml / native-build.yml matrix (Linux x86_64 and aarch64,
macOS, Windows MSVC), package_app.sh, make_installer.sh, signing. ASan/UBSan
and TSan jobs over the engine, ring buffer and queues.
- Needs you for: signing keys, a Windows or macOS on-air check.

Status 2026-10-02: written, not yet run on GitHub (CI minutes not approved).
- `.github/workflows/`: `ci.yml` (pull_request, push to master,
  workflow_dispatch only) runs generated/layering/includes checks, the
  Python suite with CPU torch, `native-build.yml` on five targets (build,
  ctest, `pytest --native`, stage, installer, sign), ASan/UBSan and TSan
  (engine, audio, rig, host). `release.yml` on `v*` tags and by hand.
- `tools/check_includes.py` (SSTVAE's; fixed the 27 includes it found),
  `tools/package_app.sh`, `make_installer.sh` (AppImage, .dmg, NSIS),
  `sign.sh` (inert without secrets), `gen_icons.py` and a placeholder icon.
- zlib: FetchContent fallback (1.3.2, sha256) when there is no system
  zlib (`native/cmake/zlib.cmake`).
- Signing: add the SSTVAE-named secrets (AZURE_*, BUILD_CERTIFICATE_BASE64,
  P12_PASSWORD, KEYCHAIN_PASSWORD, APPLE_ID, APPLE_PASSWORD), then set the
  repository variable `DATA2G_REQUIRE_SIGNING=1`. A release requires them.
- Publisher strings in `data2g.rc.in` / `installer.nsi` are placeholders
  until the certificate subject is known; no LICENSE file yet, so packages
  ship none and the metainfo names no project_license.

## Findings during the port (for review)

Reference behaviour, unchanged in Python, ported as is:
- Fixed 2026-10 (both implementations): a CRC-valid but malformed control
  frame raised out of `Station.handle` / `Session.on_rx` (short T_RV,
  T_ABANDON, empty T_NEW, 3-byte CONNECT_ACK, callsign codes >= 39; 16% of
  fuzzed link runs and 9% of session runs with corrupted control). Now
  dropped as a failed control, with a warning (docs/arq.md §4).
- Fixed 2026-10: a flipped T_COMP bit in a CRC-valid control delivered a
  deflated codeword raw (4% of corrupted-control link runs with text). The
  compression flag is now in the data codeword's CRC identity (§2).
- Fixed 2026-10: a reordered burst (heard after a later one from the same
  sender) could corrupt the stream: the abandon epoch wasn't in the CRC
  mask, and a fresh build answering a stale burst took its stale ACK as
  current. Now the epoch is in the data CRC identity, an answer to a
  repeat may not abandon, and a pre-abandon ACK past the abandon point
  fails the link (docs/arq.md §2, §4). Reproducers: `test_late_burst_*`.
- `kisslink.on_burst` builds `ModemRx(r, {})` with no DD budget: a failed
  KISS burst runs DD unbounded.
- `modem.modulate` drops codewords silently when `rvs` is shorter than
  `payloads` (zip). C++ raises.
- numpy sums the header score in float32 through OpenBLAS, whose kernel
  varies by CPU, so Python near-ties can break differently per machine.
  C++ sums in double.
- CPM pending locks' `header_end` stays a buffer index (chunking-dependent);
  CPM rx positions count from the receiver buffer, OFDM from the segment.
  Nothing reads them in streaming use.
- `sync._repeat_corr(s)` don't reduce phase before exp (up to 78 rad).

SSTVAE (not changed from here):
- TSan found a race in `RigController::wait_for_shutdown` (polls
  `weak_ptr::expired()`, a relaxed load); fixed in the Data2G copy with a
  release/acquire flag.
- Its C++ firwin uses Hamming 0.46 where scipy uses `1 - 0.54` (1 ulp).

Hamlib 4.7.2 rigctld: a client that connects microseconds after another
disconnects can fail `rig_open` (short read in `dump_state`). Seen as the
`rig_hamlib` test flake (its readiness probe), fixed there with a 300 ms
settle. The host connects once at startup, so live use should not hit it;
a reconnect loop would need the same pause.

Portability notes:
- Bitwise parity with numpy relies on glibc libm and on mirroring numpy's
  AVX-512 FMA complex multiply (`std::fma`). On other libms/CPUs expect
  ulp-level differences, inside the stated tolerances.
- `std::fma` without `-mfma` (the default x86-64 build) is a call into
  libm. glibc 2.44 dispatches it to the FMA instruction where the CPU has
  one: numpy's complex multiply then costs 2.0 ns against 0.66 ns plain
  (0.69 ns inline with `-mfma`), and 7.7 ns on glibc's software path (no
  FMA hardware, `GLIBC_TUNABLES=glibc.cpu.hwcaps=-FMA,-FMA4,-AVX2`). Its
  only caller is `constellation`'s LLRs, 0.05% of a live receive, so
  neither matters; `-mfma` would also drop pre-Haswell CPUs. MSVC (from
  its documentation, not measured): `std::fma` is the UCRT's `fma`, which
  picks the FMA3 instruction at run time where present and a software
  routine otherwise; `/arch:AVX2` lets the compiler inline it. Either way
  correctly rounded, so the same bits, only slower without hardware.
- No FP contraction anywhere (`-ffp-contract=off`; Clang contracts by
  default, GCC in GNU mode): `a * b + c` rounds twice as numpy's does,
  which is also what lets the SIMD clones below match bit for bit.
- `polar.cpp`'s `#pragma GCC optimize("O3")` is now a per-file `-O3` for
  GCC in CMake (`set_source_files_properties`); Clang and MSVC vectorize
  those loops at their defaults.
- `util/simd.hpp`: `target_clones("avx512f", "avx2", "default")` on hot
  loops, x86-64 Linux with GCC or Clang only (ifunc). Elsewhere the plain
  build runs, same bits, slower.

## Performance follow-ups

Status 2026-10-02 (performance pass). `tools/bench_native.py` (run under
`tools/with_native.py`), best of 5, Ryzen 9 9900X, shared machine; before
= the port as it was (no pool), so its 4-thread column equals 1-thread.
Every change bit-identical to before (LDPC posteriors, receives, DD
results and stored soft bits hashed; `phi` over every float32).

| Case | before 1t | after 1t | after 4t (CPU) |
|---|---|---|---|
| decode_many, 64 cw w48-16qam-r1/2, converging | 125 ms | 56 | 15 (57) |
| the same, all failing (40 iterations) | 385 ms | 168 | 43 (170) |
| failed DD pass, w48-qpsk-r1/2 16 cw (64 frames) | 63 ms | 38 | 15 (40) |
| live receive, w48 16 cw, head-limited | 12.5 ms | 8.8 | 6.9 (9.3) |
| receive, same burst, whole buffer | 143 ms | 129 | 62 (138) |
| StreamDetector hop, per band | 1.2-1.3 ms | 1.05-1.15 | same (serial) |
| tnc Receiver, 60 s mixed audio, CPU per s | 13.5 ms | 12.3 | 12.3 |

Kept:
- LDPC: AVX-512 clone of `phi_all` (1.5x on a decode); the posterior's
  sign applied by a bit flip (the `?:` was a branch mispredicted on half
  the edges: 1.5x); the variable-to-check gather in its own cloned loop
  (1.15x). Then codeword-parallel per iteration (3.9x at 4 threads).
- `demod_window`'s carrier sums and the sync repeat statistic in cloned
  loops: live receive 1.4x, a hop 10%.
- Pool in refine and on the CFO grid above 6000 samples (see "Threads").

Measured and dropped:
- CFO-grid parallel StreamDetector hops: half the latency, +15% CPU, no
  reply-latency value (hops are 250 ms apart).
- `lu_solve` for doubles with back substitution across the right-hand
  sides, fixed at 5, cloned: no measurable change. `-O3` on ldpc.cpp: none.

Not done:
- `equalizer::refine`'s low-rank form: `lu_solve` is now ~24% of a
  single-thread DD pass (~9 ms), parallel at 4 threads; a low-rank
  solve changes bits, for ~6 ms single-thread at most.
- Python-side changes that would go further (proposals, parity first):
  a per-codeword stop (each converged codeword frozen, so batch-mates
  stop costing iterations; needs a batch-independent check sum, now
  pairwise only at B = 1); a cheaper `phi` (table or offset min-sum,
  as in most LDPC decoders) since `phi` is still ~40% of a decode;
  layered scheduling (about half the iterations to converge).

## Android stays possible

Not planned, low priority, but no choice may rule it out. In practice:
- `core/` has no Qt Widgets, no desktop-only APIs and no filesystem paths
  for data. Everything Data2G loads (ldpc_shifts, header codes, format
  perms, constellations, predictor, capacity tables, clip constants, zdict)
  is compiled into the binary as generated tables, so there is nothing to
  locate at runtime.
- Audio and rig are split at the device boundary as in SSTVAE, so an
  Android backend (JNI AudioRecord, as SSTVAE did) is a new leaf, not a
  refactor.
- The host CLI and TCP servers sit outside `core/`, so a mobile front end
  can drive the engine directly.
- `check_layering.py` enforces the first two.

## Lessons carried from SSTVAE

- Substituting into the real suite finds bugs that purpose-written parity
  tests agree with themselves about.
- A skip is not a pass: `--native` fails on an unbuilt module.
- Mark committed generated files `-text` in `.gitattributes` (CRLF broke
  Windows).
- Pin the MSVC toolchain on Windows CI (MinGW built an unloadable .pyd).
- Nothing holds the ring-buffer lock across a bulk copy.
- Split audio at the device boundary and test the logic against a fake device.
- Above the modem, identical behaviour is not required. The wire format is.

## Not doing

- Porting studies, channel simulators or training.
- Porting numpy PCG64. Tables instead.
- An Android build. SSTVAE shows it is a fourth build of the same core.
- Deleting the Python host. It goes only after Phase 3 passes on air.

## Working conventions (for each module port)

- Files: `native/core/<module>/<module>.{hpp,cpp}`, namespace `data2g::<module>`.
  CMake globs `core/*.cpp`, `bindings/*.cpp` and `tests/test_*.cpp`.
- Frozen data: a generator function in `tools/gen_native_tables.py`
  registered with `FILES["<name>.cpp"] = fn` under its definition;
  declarations in `native/core/tables/tables.hpp`. Nothing is loaded from
  files at run time.
- Precision follows the Python: float64 -> double, float32 -> float. No
  fast-math. Integers and decisions exact; floats to a stated tolerance.
- Const tables only; any cache thread-safe. Hot functions reentrant.
- Bindings: `native/bindings/bind_<module>.cpp` defining
  `void bind_<module>(py::module_&)`, registered in `module.cpp`; helpers in
  `bindings/convert.hpp`. Submodes, bands, grids cross by name.
- Tests: a Python-free `native/tests/test_<module>.cpp` (check.hpp); direct
  comparisons in `tests/test_native_<module>.py` (fixtures `native`,
  `reference`); `--native` substitutions as one `@provider` function per
  module, appended at the end of `tests/conftest.py`.
- Python in `data2g/` is not changed by a port. Reference bugs are reported.
- Build and test: `JOBS=4 tools/build_native.sh --test`.
