# Data2G architecture

Where things are and how they fit, for someone about to change the code.
Module docstrings (Python) and header comments (C++) carry the detail; this
page is the map. `docs/README.md` lists the other docs.

## The repository

| Path | What |
|---|---|
| `data2g/` | The Python package: the **normative reference** for the on-air format, the receiver and the protocols, and a working station (`data2g-host`). |
| `native/` | The C++20 / Qt 6 port of the live path: `data2g-host`, `data2g-gui`, `data2g-monitor`. Checked against `data2g/`. See `native/README.md`. |
| `tests/` | The Python suite. Under `pytest --native` the same suite runs on the C++ core. |
| `tools/` | Generators (`gen_native_tables.py`, `freeze_format.py`), layering checks, build and packaging scripts, `with_native.py`. See `tools/README.md`. |
| `scripts/` | Studies and the data generators behind every tuned number and committed table. See `scripts/README.md`. |
| `docs/` | Protocol specs (`arq.md`, `broadcast.md`), this page, and the history of past plans. |
| `runs/` | Study output; ignored by git except a few measured CSVs and the `*.sh` drivers that reproduce studies. |

## A burst's life

A **burst** is one transmission: a preamble, a 16-bit header naming the
submode and codeword count, then codewords. Each codeword carries one
payload with its own CRC, and is the unit of acknowledgement and resend.
Two waveform families share everything above the modem: OFDM
(`modem.py`, four bands from 200 to 2400 Hz) and constant-envelope CPM
(`cpm.py`, the slowest rungs). `arq/modes.py` lists every mode of both.

### Transmit

```
host.Host / kisslink.KissLink          what to send (ARQ stream, broadcast frames)
  -> arq.session.Session.poll           a TxBurst: slots of (mask_id, rv, payload)
     (arq.link.Station.build)
  -> arq.phy.tx_audio                   one burst to audio
       codes.encode                     CRC (XOR the slot's mask), PN9 scramble, LDPC or polar,
                                        rate matching at the RV, BICM interleave
       codes.spread                     codewords across the burst's symbols
       modem.modulate_bits              constellation.modulate -> ofdm.Band.modulate_symbols
                                        (preamble, header, frames of 1 pilot + 5 data symbols)
                                        -> waveform.dsp.tx_condition (clip and filter)
       or cpm.modulate                  tones, Costas sync blocks, header; cpm.bandpass
  -> arq.engine.Engine.step             plays it out, PTT, half duplex
```

### Receive

```
sound card -> audio.Decimator           device rate -> FS (8 kHz)
  -> receiver.Receiver.feed             audio.Blanker; waveform.sync.StreamDetector (preamble
                                        statistics); modem.find_burst / cpm.find /
                                        modem.find_copy; BUSY while the burst arrives
  -> modem.receive or cpm.receive       acquisition, header, equalizer.estimate (2-D LMMSE from the
                                        pilots), raw data symbols
  -> arq.phy.ModemRx                    the burst's soft bits (modem.burst_soft, constellation.llr),
                                        decodes per slot and CRC mask: codes.decode_raw /
                                        decode_buffer (HARQ combining) / decode_posterior;
                                        decision-directed re-estimation (DD, equalizer.refine)
  -> arq.session.Session.on_rx          the ARQ (link.Station.handle), or
     kisslink.KissLink.on_burst         a broadcast group's frames, or a CQ
  -> host.Host                          VARA command/data ports, KISS ports
```

`arq.engine.Engine` is the whole station between the sound card and the
host: `step(audio_in) -> (audio_out, ptt)`, clocked by samples, so the same
code runs on air and back to back in tests (`tests/test_engine.py`) and
studies. Alongside, `arq.phy.measure` turns a received burst into the gear
shifter's measurements, and `arq.policy.GearShifter` recommends the peer's
next mode (`docs/arq.md` §8).

## Layers

Python, bottom up; a module imports only those above it (`data2g/__init__.py`
has the one-line map, `data2g/arq/__init__.py` the ARQ's):

```
config                                   on-air constants, bands, SUBMODES
constellation, waveform/{dsp,ofdm,sync}  PHY primitives
equalizer                                channel estimation
ldpc, polar                              codes, knowing nothing of submodes
codes                                    framing, interleaving, HARQ, decode, keyed by spec
modem, cpm                               burst modems
audio, receiver                          the sound card side, the streaming receiver
arq/ (frames, burst, link, session, modes, predictor, policy, phy, engine)
kisslink, tnc                            broadcast, the KISS byte codec, PTT
host                                     data2g-host
bulk                                     data2g-bulk (one-way text, no ARQ)
```

Off to the side, never on the live path and never ported: `hfchannel` and
`interference` (channel simulators), `channel_torch` and `decoders_torch`
(torch, for training and GPU studies).

The seams worth knowing:

- `arq.burst.RxBurst` and `arq.link.Policy`: `frames`, `burst`, `link` and
  `session` know no modem. `phy.ModemRx` is the RxBurst on air (fakes in
  tests and `scripts/linksim.py`), `policy.GearShifter` the Policy.
- `codes.CodeSpec`: anything with a submode's code fields. `cpm.CpmSpec` and
  `config.SubmodeSpec` both are, so the ARQ's codec path does not care
  which family a mode is.
- A received burst crosses from the modem to the ARQ as a dict
  (`modem.Received`, or cpm's with `family: "cpm"`).

Native keeps the same layers (`core/<module>/`), with `core/dsp` and
`core/util` underneath, `core/rig`, `core/monitor` and the audio backends
beside, and `app/`, `apps/`, `gui/` on top (`native/README.md`).

## Python and native

Most modules map one to one: `data2g/X.py` is `native/core/X/X.{hpp,cpp}`,
namespace `data2g::X`, and each header's first line names its Python file.
The exceptions:

| Python | Native |
|---|---|
| `modem.py` | `core/modem/modem` and `core/modem/timing` (burst timing arithmetic) |
| `waveform/dsp.py` | `core/waveform/dsp`. `core/dsp/` is numpy/scipy, with no Python twin. |
| `audio.py` | `core/audio/filters` (Decimator, Interpolator, Blanker), `fifo` (Capture, Player), `audio` (device selection), and the backends |
| `tnc.py` | `core/tnc` (KISS codec); its TCP server is `app/tcp_servers`, PTT `core/rig` |
| `host.py` | `core/host` (Host); `serve()` is `app/station`; `main()` is `apps/data2g_host.cpp` |
| `config.py` | `core/generated/config.hpp` (generated) and `core/config` (lookups) |
| `arq/policy.py` | `GearShifter` (the model) and `GearPolicy` (its link.Policy) in `core/arq/policy` |
| `bulk.py` | none (Python only) |
| none | `core/rig`, `core/monitor`, `core/util`, `gui/` |

`native/README.md` has the full table, with each area's binding, parity
test and ctest.

## The parity contract

1. **Python is normative.** A change to the format or the receiver lands in
   `data2g/` first, then in `native/` with the same names. Studies run on
   Python, which the port must match.
2. **Frozen data is generated, never edited.** `tools/gen_native_tables.py`
   writes `native/core/generated/` from `data2g/` (constants, submodes,
   constellations, LDPC shifts, polar designs, formats, the outcome model,
   the deflate dictionary). CI runs `--check`, which fails until you
   regenerate after changing a constant or table.
3. **The whole suite is the port's acceptance test.** `pytest --native`
   imports `data2g_native` (built by `tools/build_native.sh` into
   `native/build/python`) and, before collecting, replaces about 200
   `(module, attribute)` pairs in `data2g` with C++-backed versions:
   `tests/native_subs/`, one file per area, each `substitutions(native) ->
   {(module, attr): replacement}`. Every replacement keeps the Python
   signature and falls back to Python for what C++ doesn't have (a spec
   that isn't the frozen one, a test's patched constant, torch). Then
   every test in `tests/` runs against C++.
4. **Side-by-side tests.** `tests/test_native_<area>.py` call both and
   compare, with or without `--native` (they skip if the module isn't
   built; under `--native` a missing module is an error). The `reference`
   fixture gives the Python original of a substituted name.
5. **Names are part of the contract**, private ones included:
   `sync._raw_stat`, `codes._decoder`, `modem._demod_frames` and others are
   substitution keys. Renaming one in `data2g/` means renaming it in
   `tests/native_subs/` and usually in `native/bindings/`. Plain `pytest`
   won't notice; `pytest --native` will. The ported modules' docstrings say
   which of their names are substituted.
6. **Studies on the port**: `tools/with_native.py scripts/<study>.py ...`
   applies the same substitutions (`--skip arq,gear` leaves areas in
   Python; the names are `tests/native_subs/`'s modules).
7. **Layering**: `tools/check_layering.py` and `tools/check_includes.py`
   run in CI.

## Committed data

Every tuned number and table comes from a script; redo the script to change
it.

| Data | Written by |
|---|---|
| `data2g/codes_data/*` | see `data2g/codes_data/README.md` (LDPC shifts, header codes, clip constants, the outcome model, capacity tables, mode thresholds, link abstraction) |
| `data2g/format/*.npz` | `tools/freeze_format.py`: each submode's interleaver and polar info set, the on-air format proper |
| `data2g/constellations/*.npy` | `scripts/train_constellation.py --out`, kept after `scripts/retest_constellations.py` |
| `data2g/arq/zdict.bin` | `scripts/build_zdict.py` |
| `config.SUBMODES` | the ladder studies: `scripts/ladder_study.py`, `scripts/prune.py` (`CANDIDATES.md`, written by prune.py, lists the kept candidates) |
| `scripts/data/measurement_errors.npz` | `scripts/linksim.py export-errors` |
| `native/core/generated/*` | `tools/gen_native_tables.py` |

## Where to start

- **DSP, sync, equalizer** (`waveform/`, `equalizer.py`, `modem.py`'s
  receive half): read `data2g/waveform/__init__.py`, then the module.
  `tests/test_modem.py`, `test_preamble.py`, `test_equalizer.py` and
  `test_receiver.py` pin behaviour; `scripts/sync_floor.py`,
  `scripts/ladder_study.py` and `scripts/loss_study.py` measure it. Port the
  change to `native/core/<module>/` and run `tests/test_native_<module>.py`.
- **FEC** (`ldpc.py`, `polar.py`, `codes.py`): codes.py's docstring lists
  its sections. A change to what goes on air needs `tools/freeze_format.py`
  and `tools/gen_native_tables.py`.
- **Adding or retuning a submode**: `config.SUBMODES`, then
  `tools/freeze_format.py`, the clip constants (`scripts/clip_constants.py`)
  and `tools/gen_native_tables.py`. The outcome model knows each mode by
  name: retrain it (`docs/outcome-training-data.md`).
- **The ARQ protocol**: `docs/arq.md` (the spec, with a section-to-file
  map), `data2g/arq/__init__.py`. `tests/test_arq.py` and
  `test_arq_session.py` run stations in lockstep; `scripts/linksim.py` and
  `scripts/phy_session.py` run whole sessions.
- **The gear shifter and its model**: `arq/policy.py`, `arq/predictor.py`;
  data and training in `docs/outcome-training-data.md`.
- **Broadcast / KISS**: `docs/broadcast.md`, `kisslink.py`.
- **One-way text** (`data2g-bulk`): `docs/bulk.md`, `bulk.py`,
  `tests/test_bulk.py`; `scripts/bulk_study.py` measures it. Python only.
- **The host's commands**: `host.py`'s docstring is the command reference;
  `native/core/host` mirrors it, `tests/test_host.py` and
  `test_native_host.py` check both.
- **Native only** (sound cards, PTT, threads, GUI, packaging):
  `native/README.md`.

## Running things

```
uv sync --extra dev [--extra tnc] [--extra train]
uv run pytest tests -q                    # the reference
tools/build_native.sh --test              # build native/, ctest, pytest --native
uv run pytest tests -q --native           # the suite on the C++ core
uv run data2g-host --help                 # the Python station
native/build/data2g-host --help           # the C++ one
python tools/gen_native_tables.py --check && python tools/check_layering.py
```

`--extra tnc` is pyaudio (the Python host's sound card); `--extra train` is
torch and friends (constellation training, GPU studies). Slow Monte Carlo
tests are marked `slow` and skipped unless asked for (`-m slow`).

## Glossary

- **Submode**: one OFDM mode, `config.SUBMODES[name]`: band, constellation,
  code and rate. **Mode** (ARQ): a submode or a CPM mode, `arq.modes.MODES`.
- **Band**: a run of carriers on the 50 Hz grid: `w` (1200 Hz), `w48`
  (2400 Hz), `n10` (500 Hz), `n4` (200 Hz). Its **sync band** is the band
  whose preamble and header it uses (`n4` syncs on `n10`). A CPM **grid**
  (tones, spacing) is CPM's band.
- **cu**: channel use, one carrier in one data symbol.
- **Frame**: 1 pilot + 5 data symbols on every carrier of the band.
- **RV / IR**: redundancy version; incremental redundancy. A resend at
  another RV carries other parity bits, combined with the first copy's soft
  bits (HARQ).
- **Mask, mask_id**: a codeword's identity, hashed into the CRC (XORed into
  it), so a codeword taken for the wrong session, direction, seq or group
  fails its CRC (`docs/arq.md` §2, `arq.burst.ctl_mask` / `data_mask`).
- **DD**: decision-directed re-estimation: decoded codewords become pilots
  for a better channel estimate (`arq/phy.py`).
- **Soft bits / LLR**: log P(0)/P(1); positive means 0, everywhere.
- **PEP-fair**: compared at equal peak envelope power (SNR plus PAPR), the
  limit a ham transmitter actually has.
- **SNR**: always referred to 2500 Hz of noise (`config.SNR_REF_BW_HZ`).
- **10% point**: the SNR where a mode's smallest ARQ burst fails 10% of the
  time; what the ladder is ranked by.
- **ISS / IRS**: information sending / receiving station (VARA's terms).
- **Gear shifter**: the rate shifter (`arq/policy.py`). Older docstrings
  and scripts label its development steps: phase A, measurement and decode
  latency (`scripts/rx_audit.py`, `decode_latency.py`); C1, link
  abstraction (`link_abstraction.py`); C2, a channel-prediction dataset
  (`predictor_data.py`); D, sessions on the abstraction
  (`linksim.py`); F, incremental redundancy (`ir_study.py`); G, sessions
  through the real modem (`phy_session.py`); H, the live stack
  (`arq/engine.py`, `host.py`).
- **SSTVAE**: the sister project whose OFDM waveform and native stack
  Data2G started from; "lifted from SSTVAE" marks code taken from it.
