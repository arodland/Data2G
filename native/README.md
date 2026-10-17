# native/

The C++20 / Qt 6 port of the live path: the modem, the receiver, the ARQ
and the station, as `data2g-host` (headless) and `data2g-gui`, plus
`data2g-monitor`, `data2g-replay` and `data2g-audio-check`. `data2g/`
(Python) stays the normative reference: everything here is checked against
it (`docs/ARCHITECTURE.md`, "The parity contract").

    tools/build_native.sh            # configure + build into native/build
    tools/build_native.sh --test     # then ctest and pytest --native

## Directories

| Directory | What |
|---|---|
| `core/` | The library, `data2g_core`: no Qt, no files read at run time. One directory per Python module, `core/<module>/<module>.{hpp,cpp}`; every header's first line names its Python counterpart. |
| `core/generated/` | Constants and frozen tables written by `tools/gen_native_tables.py` from `data2g/`. Never edited. |
| `core/tables/` | The declarations for `generated/*.cpp`, one header per area. |
| `core/{dsp,util}/` | Native only: numpy/scipy primitives (`dsp/`, *not* `data2g/waveform/dsp.py`, which is `core/waveform/dsp`), `Mat`, the shared worker pool, cancellation, linear algebra, SIMD clones. |
| `core/{rig,monitor}/` | Native only: rig control and PTT (`rig/hamlib/` is the libhamlib backend, `data2g_rig`); the promiscuous receiver behind `data2g-monitor`. |
| `core/audio/{qt,miniaudio}/` | Sound card backends (`data2g_audio`), outside `data2g_core`. |
| `app/` | `data2g_station`: the runtime both front ends share (`host.py`'s `serve()`), Qt Core + Network only. `app/station.hpp` lists its threads. |
| `apps/` | The command-line programs' `main()`s. |
| `gui/` | `data2g-gui`: the only place Qt Widgets may appear. |
| `bindings/` | `data2g_native`, the pybind11 module that `pytest --native` substitutes into the Python suite (`tests/native_subs/`). |
| `tests/` | One ctest per area (`test_<area>.cpp`), Python-free. |
| `cmake/`, `packaging/`, `third_party/` | Pinned dependencies (Hamlib, zlib), icons and installers, vendored pocketfft and miniaudio. |

## Targets

```
data2g_core ─┬─ data2g_audio (sound cards) ─┐
             ├─ data2g_rig (libhamlib) ─────┤
             ├─ data2g_applog (app/log) ────┼─ data2g_station (app/) ─┬─ data2g-host
             │                              │                         └─ data2g_gui ─ data2g-gui
             ├─ data2g-monitor, data2g-replay, data2g-audio-check
             ├─ test_* (ctest)
             └─ data2g_native (bindings/, pybind11)
```

`tools/check_layering.py` enforces the walls: no Qt in `core/` outside the
device backends, Widgets only in `gui/`, `core/ldpc` and `core/polar`
below `codes/`, and `core/generated/` only as generated.
`tools/check_includes.py` catches `std::` names used without their header
(MSVC is the strict one).

## Python ↔ C++ map

Binding: `bindings/bind_<x>.cpp`, registering `data2g_native.<x>`. Parity:
`tests/test_native_<x>.py` (side by side), on top of `pytest --native`
(the whole suite, substituted). ctest: `tests/test_<x>.cpp`.

| Python | core/ | Binding (submodule) | Parity test | ctest |
|---|---|---|---|---|
| `config.py` | `config/`, `generated/config.hpp` | `bind_config` (`config`) | `test_native_parity` | |
| `constellation.py` | `constellation/` | `bind_constellation` | `test_native_constellation` | `constellation` |
| `waveform/{dsp,ofdm,sync}.py` | `waveform/` | `bind_waveform` (`waveform`) | `test_native_waveform` | `waveform` |
| (numpy, scipy) | `dsp/` | `bind_dsp` (`dsp`) | `test_native_waveform` | |
| `equalizer.py` | `equalizer/` | `bind_equalizer` | `test_native_equalizer` | `equalizer` |
| `ldpc.py` | `ldpc/` | `bind_ldpc` | `test_native_ldpc` | `ldpc` |
| `polar.py` | `polar/` | `bind_polar` | `test_native_polar` | `polar` |
| `codes.py` | `codes/` | `bind_codes` | `test_native_codes`, `test_native_parity` | `codes` |
| `modem.py` | `modem/modem`, `modem/timing` | `bind_modem`, `bind_timing` (`timing`) | `test_native_modem`, `test_native_arq` (timing) | `modem` |
| `cpm.py` | `cpm/` | `bind_cpm` | `test_native_cpm` | `cpm` |
| `audio.py` | `audio/` (`filters`, `fifo`, `audio`) | `bind_audio` | `test_native_audio` | `audio` |
| `receiver.py` | `receiver/` | `bind_receiver` | `test_native_receiver` | `receiver` |
| `tnc.py` | `tnc/` (KISS codec; its server and PTT are `app/tcp_servers`, `rig/`) | `bind_tnc` | `test_native_tnc` | `tnc` |
| `kisslink.py` | `kisslink/` | `bind_kisslink` | `test_native_arq_phy` | `arq_phy`, `monitor` |
| `arq/{frames,burst,link,session}.py` | `arq/` | `bind_arq_link` (`arq`) | `test_native_arq_link`, `test_native_fuzz` | `arq_link` |
| `arq/{modes,predictor,policy}.py` | `arq/` | `bind_arq` (`arq`) | `test_native_arq` | `arq` |
| `arq/phy.py` | `arq/phy` | `bind_arq_phy` (`phy`) | `test_native_arq_phy` | `arq_phy` |
| `arq/engine.py` | `arq/engine` | `bind_engine` | `test_native_engine` | `engine`, `pool` |
| `host.py` `Host` | `host/` | `bind_host` | `test_native_host` | `host` |
| `host.py` `serve()`, `main()` | `app/`, `apps/data2g_host.cpp` | | `test_native_host_e2e` | `args`, `station_tx`, `realtime`, `rx_plot` |
| | `monitor/`, `rig/`, `util/` | | | `monitor`, `rig`, `rig_hamlib`, `pool` |
| | `gui/` | | | `gui` |

`tests/native_subs/<area>.py` holds the substitutions for each area: which
Python names the binding stands in for, and when it falls back to Python.

## Where to start

- **A DSP, FEC or modem change** lands in `data2g/` first, then in
  `core/<module>/` with the same names. Run `tests/test_native_<module>.py`
  and `pytest --native`; if a constant or table changed, run
  `tools/gen_native_tables.py` (CI fails on `--check` until you do).
- **An ARQ or gear-shifter change**: `core/arq/` mirrors `data2g/arq/`
  module for module; `docs/arq.md` has the section-to-file map.
- **The station, threads, sound cards, PTT**: `app/station.hpp`,
  `app/transmitter.hpp`, `core/audio/`, `core/rig/`. These have no Python
  twin beyond `host.py`'s `serve()`; their ctests run on pipe audio and
  Hamlib's dummy rig.
- **The GUI**: `gui/`; `test_gui` renders it offscreen.

Performance notes live where the code is (`core/util/simd.hpp`,
`core/util/pool.hpp`, the GCC `-O3` note for `polar.cpp` in
`CMakeLists.txt`); `tools/bench_native.py` times the hot paths.
`docs/native-port-plan.md` is the port's history: its rules
(threads, what is frozen, working conventions) still hold.
