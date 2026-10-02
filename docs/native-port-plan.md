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

### Phase 2: ARQ and engine

frames (zlib + zdict + history), session, link, phy, policy, predictor,
modes, engine, recorder.
- Volume: 2.6k lines.
- Verified by: test_arq*, test_engine under `--native`; two C++ engines and
  mixed C++/Python engines through the simulated channel; a paired loss study
  with `DATA2G_PEP_REF_DB` set. The ARQ state-agreement fuzz (loss plus
  duplication) runs on the mixed pair.
- Needs you for: nothing.

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

### Phase 4: GUI

Status window and settings dialog over the same core. Exit: you use it for a
session.

### Phase 5: packaging and CI

Lift SSTVAE's ci.yml / native-build.yml matrix (Linux x86_64 and aarch64,
macOS, Windows MSVC), package_app.sh, make_installer.sh, signing. ASan/UBSan
and TSan jobs over the engine, ring buffer and queues.
- Needs you for: signing keys, a Windows or macOS on-air check.

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
