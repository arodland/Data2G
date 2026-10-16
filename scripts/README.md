# scripts/

Studies, and the generators behind every committed table and tuned number.
Each script's docstring says what it measures, how to run it and what it
found or decided. They run on the Python reference; `tools/with_native.py`
runs one on the C++ core. Output goes to `runs/` (ignored by git). CI runs
none of them, so a refactor in `data2g/` can break one silently: run the
ones you touched.

## Generators of committed data

Rerun to change the file; `data2g/codes_data/README.md` has the readers.

| Script | Writes |
|---|---|
| `own_shifts.py` | `codes_data/ldpc_shifts.npz` (LDPC circulant shifts: generate, screen, export) |
| `design_header.py` | `codes_data/header_code*.npy` (the burst header's linear codes) |
| `design_pilot.py` | low-PAPR pilot phases (`config.PILOT_*`) |
| `clip_constants.py` | `codes_data/clip_constants.json` (what the clipper does, as the receiver sees it) |
| `clip_study.py`, `pick_headroom.py` | each submode's clip headroom |
| `ace_study.py`, `pick_ace.py` | which submodes use ACE (`pick_ace.py --json` updates the clip constants) |
| `train_constellation.py`, `retest_constellations.py` | `data2g/constellations/*.npy`, and whether a retrained set is kept |
| `build_zdict.py` | `data2g/arq/zdict.bin` (the ARQ's deflate dictionary) |
| `rx_audit.py` | `codes_data/capacity_tables.npz`, and the receiver's measurement errors |
| `link_abstraction.py` | `codes_data/link_abstraction.json` (for `linksim.py`) |
| `outcome_data.py`, `session_data.py` | the outcome model's training rows (`docs/outcome-training-data.md`) |
| `train_outcome.py` | `codes_data/outcome_predictor*.npz` |
| `ladder_study.py` | every mode's thresholds (`codes_data/mode_thresholds.json`) |
| `ladder.py`, `combine_ladders.py`, `prune.py` | the submode ladder: thresholds per candidate, merged, pruned to `CANDIDATES.md` |
| `linksim.py export-errors` | `scripts/data/measurement_errors.npz` |

## Shared harness

Imported by other scripts (via `sys.path`) and, for the first three, by
tests: change them with care.

| Module | |
|---|---|
| `outcome_data.py` | seeded test bursts through the real modem (`burst`), the dataset generator |
| `phy_session.py` | whole ARQ sessions through the real modem and a continuous fading channel (`RealPhy`) |
| `train_outcome.py` | the outcome model's inputs and training (tests check its features) |
| `linksim.py` | the session event loop; sessions on a link-abstraction model (`SimPhy`) |
| `loss_study.py` | real-modem sessions with every burst's fate classified |
| `thresholds.py` | threshold search on the GPU path (`channel_torch`) |
| `clip_constants.py`, `clip_study.py`, `prune.py`, `rx_audit.py`, `sync_floor.py` | pieces other studies reuse |

Some scripts import test helpers in turn: `arq_stress.py` (tests/test_arq.py,
test_arq_session.py) and `crc_exposure_engine.py` (test_engine.py,
test_kiss.py).

## The live stack

| | |
|---|---|
| `replay.py` | Replay a session recorded at both ends (`data2g-host --record-dir`) with exact knowledge of what was on air. |
| `cpu_profile/` | End-to-end runs of two hosts: `run.sh` (profiling over a PipeWire loopback, Pat or raw transfers), `speedtest.py` and `speedtrials.py` (one-way bulk speed, off the clock; `tests/test_speedtest.py` runs it), the VARA comparison (`vara_ref.sh`, `varatrials.py`, `vara_sweep.sh`), `tnc_proxy.py` (log an app talking to a TNC), and their helpers. |

## Studies

One question each, answered in the docstring. By area:

- **Ladder and thresholds**: `ladder_study.py`, `ladder_order_check.py`,
  `compare_ladders.py`, `ladder_page.py`, `paired_loss.py`,
  `spread_screen.py`, `eq_floor.py`, `llr_scale_sweep.py`,
  `llr_clamp_study.py`.
- **Sync and the header**: `sync_floor.py`, `sync_sweep.py`, `sync_diag.py`,
  `w48_sync_diag.py`, `sync_loss_study.py`, `mf_detect_study.py`,
  `header_study.py`, `header_diversity.py`, `copy_acq_study.py`,
  `copy_cpu.py`, `accept_false_locks.py`.
- **Receiver variants**: `rx_ab_study.py`, `blanker_study.py`,
  `erasure_retry_study.py`, `polar_list_study.py`, `polar_gate_study.py`.
- **Peak power**: `clip_study.py`, `ace_study.py`, `slm_study.py`,
  `slm_blind_study.py`, `cpm_papr_study.py`, `plot_constellations.py`.
- **CRC and false accepts**: `crc_study.py`, `crc_cost.py`,
  `crc_exposure.py`, `crc_exposure_engine.py`.
- **ARQ and the gear shifter**: `arq_stress.py`, `idle_study.py`,
  `ir_study.py`, `decode_latency.py`, `compress_study.py`,
  `predictor_data.py`, `calibration.py`, `ood_probe.py`.

"Gear-shifter phase A..H" in older docstrings: see the glossary in
`docs/ARCHITECTURE.md`.
