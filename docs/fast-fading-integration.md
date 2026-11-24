# fast-fading-integration

TLDR: three study branches merged for outside study, now on master `95c94b1`. Every feature keeps its switch.
The defaults here are each branch's best measured setting, not today's on-air behaviour. Python only: native is
not ported, so native builds and `--native` tests do not match this branch with its defaults.

- Base: origin/master `95c94b1` (through each branch's own merge of it). First built on `9894b63`; the old-base
  numbers are kept below beside the new ones.
- Merged heads: `fsk-wide-bins` 495797a, `c32r62-r23` 9b34c66, `scattered-pilots` bf7df2f.
- Only here: the three-way tone likelihood (per-tone floors, sub-window sums, leak model), the acquisition
  switch, the defaults, this file.
- Not a release candidate. Nothing here has been on air.

## Switches

| Switch | Default here | Off value | What it does |
|---|---|---|---|
| `DATA2G_PILOTS` | `mix` | `off` or empty | OFDM pilots scattered through the frame (same 1/6 of the cells) instead of a pilot row. `k3d2`, `gold`, `mix`. Wire change, both ends. |
| `DATA2G_CPM_LEAK` | `ctl,data,dd,ctlsq` | `off` or empty | CPM soft values treat the energy beside a tone as that tone's Doppler leak, not as noise (`cpm.leak_metric`), on master's per-tone floors. `ctlsq`: a control codeword whose rows read spill >= 0.15 stays on the square law. Receiver only. |
| `DATA2G_CPM_LEAK_K` | `0.5` | | The share of unmodelled leak counted as noise. |
| `DATA2G_CPM_CTL_SQ` | `0.15` | | `ctlsq`'s threshold. |
| `DATA2G_CPM_SPILL` | all 1 | | Per-consumer weight of the spill in the square law's noise (studies). |
| `DATA2G_CPM_R23` | `1` | `0`, `off` or empty | `fsk32r62-r2/3` is a mode: header index 4 on c32r62, recommendation index 6. On its own branch the default is 0. Wire addition. |
| `DATA2G_OUTCOME_R23` | on | `0` or `off` | The outcome model is `outcome_predictor_r23.npz` (the installed weights with a row for r2/3). Off: the installed model, which never recommends r2/3. Follows `DATA2G_CPM_R23`. |
| `DATA2G_CPM_ONAIR` | `c8w50` | `off` or empty | The wide-bin grid c8w50 is an on-air grid: every receiver searches it. Its modes are control-only and have no recommendation code. |
| `DATA2G_ROBUST_RULE` | `connect` (while c8w50 is on air) | `off` or empty | CONNECT retries go in `fsk8w50-r1/2`, not `fsk16r25-r1/2`. In-session robust modes stay as today. Others: `replace`, `alt`. |
| `DATA2G_ROBUST_FAST` | off | `off` or empty | `wide`, `always`: c8w50 as the in-session robust mode too (measured worse). |
| `DATA2G_CPM_ACQ_K` | `1` (off) | `1` or `off` | The coarse stage's best K placements go to the fine stage. Small gain, large CPU: not recommended. With `DATA2G_CPM_ACQ_SCREEN`, `DATA2G_CPM_ACQ_FLOOR`. |
| `DATA2G_CPM_SHARE` | `1` | `0` | c8w50's search rows are sums of c8r50's (same result either way). |
| `DATA2G_CPM_WIDE` | off | | Studies: every wide grid's modes (c16w125 too) in `arq.modes.MODES` for fixed bursts. c16w125 is never on air. |

Today's behaviour exactly:

    DATA2G_PILOTS=off DATA2G_CPM_LEAK=off DATA2G_CPM_R23=off DATA2G_CPM_ONAIR=off DATA2G_CPM_ACQ_K=off

The defaults yield where a feature has no path:
- FM profile (`DATA2G_PROFILE=fm`): pilots are rows, c8w50 is not on air, r2/3 is not a mode. An explicit
  `DATA2G_PILOTS` value still raises there, as on its branch.
- Another frame length (`DATA2G_SYMS_PER_FRAME`) or a hand-set FM-path study switch (`DATA2G_SPREAD_WINDOW`,
  `DATA2G_EST_WINDOW`, `DATA2G_DECODE_EARLY`, `DATA2G_SYMBOL_NOISE`): pilots default to rows.

## The tone likelihood

`cpm.tone_metric` is where the three meet: master's per-tone floors (a carrier), the wide grids' sub-window sums,
and the leak model. One function, `_spill_level`, gives N0, the floors, Es and the spill, on the energies whitened
by the floors and at a sub-window's level. The square law (`_tone_gain`) and the leak model (`leak_metric`) both
start from it. Per tone, the noise is its floor plus the leak share; every log term is counted once per
sub-window. With the other features absent it reduces to each branch's own result (checked by hash, below).

## Equality checks

- All switches off against origin/master: the CPM grids' bursts, energies, locks, soft values and DD soft values
  hash equal, with and without an in-grid carrier (`runs/ident.py`, `runs/ident_carrier.py`, `runs/softhash.py`
  style). OFDM transmit audio, soft values, channel estimates and decodes equal master's
  (`scripts/sc_equal.py`: 168 arrays, 0 differ).
- Wide grids, leak off, with and without a carrier: equal to `fsk-wide-bins`.
- On-air grids, leak on, with and without a carrier: equal to `c32r62-r23`.

## Tests

Run a suite alone (`tests/test_host.py` shares a TCP port).

All switches off: 877 passed, 0 failed.

Defaults: 858 passed, 19 failed. The feature tests of all three branches pass (`tests/test_cpm_wide.py`,
`tests/test_scattered.py`, the r2/3 and leak tests in `tests/test_cpm.py`). The failures assert today's wire or
today's receiver on the old pilot layout, and were left as they are. Last build: 17. The two more are tests
master added since.

- Scattered pilots (15; 13 as before):
  - `test_arq_phy.py::test_long_control_is_pinned_where_its_padding_starts`
  - `test_channel_torch.py`: `test_transmit_matches_numpy_modulator`, `test_ace_transmit_matches_numpy_and_keeps_points_in_their_regions`, `test_bmi_agrees_with_the_full_modem`
  - `test_dd.py`: `test_dd_rescues_a_codeword`, `test_dd_budget_spent`
  - `test_equalizer.py::test_a_tone_on_one_carrier_does_not_sink_the_burst`
  - `test_modem.py`: `test_timing_slip_recovered[-64]`, `[85]`, `test_a_steady_tone_does_not_read_as_pilots[n4]`
  - `test_receive_head.py::test_weak_header_at_the_burst_end_is_received`
  - `test_receiver.py::test_a_false_lock_on_data_does_not_cost_the_next_burst`
  - `test_sounding.py::test_fading`
  - New: `test_receiver.py::test_a_burst_decodes_while_it_arrives` (a flagged burst waits for its end),
    `test_fmtorch.py::test_bmi_agrees_with_the_numpy_receiver`.
  - The slip, steady-tone, weak-header and false-lock ones are receiver behaviour, not layout: the unjudged
    `find_copy` gates and slip search under Caveats.
- c8w50 (3): `test_arq_session.py::test_connect_retries_in_the_robust_mode_and_is_answered_in_it`,
  `test_compact_connect_over_the_cpm_modem` (they expect fsk16r25 retries);
  `test_arq_policy.py::test_recommend_roundtrip_every_submode` (the c8w50 modes have no code).
- Native (1): `test_native_parity.py::test_generated_tables_current`.

FM profile, defaults: 924 passed, 30 failed (master: 920 passed). The 30 are the three branches' HF-only feature
tests (`test_scattered` 21, `test_cpm_wide` 7, the r2/3 test), which fail the same way on their own branches,
and the native tables check (it also checks the HF tables, which differ with the defaults).

## Smoke loss study

12 seeds x 600 s, `shift+cpm`, PEP 5, Python, all switches off against the defaults, paired. Delivered bit/s,
mean of the seeds (seeds that delivered nothing in brackets), change ± SE.

| Cap | Cell | Old base: off | defaults | change | New base: off | defaults | change |
|---|---|---|---|---|---|---|---|
| 2400 | hilat-d 10 | 40 (6) | 107 (0) | +165% ± 31 | 41 (6) | 109 (0) | +164% ± 30 |
| 2400 | hilat-m 10 | 80 | 278 | +249% ± 49 | 81 | 315 | +288% ± 45 |
| 2400 | hilat-m 20 | 88 | 330 | +275% ± 59 | 82 | 306 | +272% ± 46 |
| 2400 | lowlat-d 10 | 85 | 131 | +53% ± 9 | 85 | 125 | +47% ± 6 |
| 2400 | nvis-d 10 | 631 | 952 | +51% ± 5 | 631 | 952 | +51% ± 5 |
| 2400 | mpd 0 | 420 | 479 | +14.2% ± 1.9 | 420 | 479 | +14.2% ± 1.9 |
| 2400 | mpp 0 | 483 | 458 | -5.3% ± 2.0 | 483 | 458 | -5.3% ± 2.0 |
| 2400 | mpp -8 | 107 | 102 | -4.4% ± 2.9 | 107 | 102 | -4.4% ± 2.9 |
| 2400 | awgn -12 | 105 | 124 | +17.7% ± 4.6 | 105 | 124 | +17.7% ± 4.6 |
| 2400 | awgn 6 | 1730 | 1714 | -1.0% ± 0.0 | 1730 | 1714 | -1.0% ± 0.0 |
| 2400 | awgn -12, in-grid carrier | | | | 98 | 94 | -3.5% ± 6.5 |
| 500 | mpd 0 | 179 | 251 | +40.5% ± 5.3 | 179 | 251 | +40.5% ± 5.3 |
| 500 | mpp 0 | 270 | 290 | +7.3% ± 1.5 | 270 | 290 | +7.3% ± 1.5 |

The carrier is 18 dB over the floor in 50 Hz at 1507 Hz, inside every CPM grid (`scripts/wide_carrier_loss.py`,
cell `awgn:-12:incar`).

mpp -8 by switch (the defaults with one switch off, same seeds):

| Arm | bit/s | against all off | against the defaults |
|---|---|---|---|
| All off | 107 | | |
| Defaults | 102 | -4.4% ± 2.9 | |
| Pilots off | 110 | +3.0% ± 2.6 | +7.8% ± 2.9 |
| r2/3 off | 98 | -8.4% ± 1.9 | -4.2% ± 2.0 |
| Leak off | 103 | -3.4% ± 2.5 | +1.1% ± 2.0 |
| c8w50 off | 102 | -4.2% ± 4.0 | +0.2% ± 2.6 |

The scattered pilots cost that cell about 8%. r2/3 gives back about 4%. The leak model and c8w50 do nothing there.

Against what the parts predicted:
- hilat-d 10: the leak model alone and c8w50 retries alone each reached about 84 bit/s with no dead sessions.
  With r2/3 it is 107-109: r2/3 carries the data.
- hilat-m: the pilots and r2/3 lift both cells threefold, so c8w50's own effect cannot be read from this table.
- mpp 0 is inside the pilots' known mid-SNR loss. mpp -8 is the pilots too (the split above).
- awgn 6 is down 1.0% on every seed in the same mode: a fixed cost of the pilot layout.

To rerun (no native):

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_PEP_REF_DB=5
    python -m scripts.wide_carrier_loss --policy shift+cpm --seeds 12 --horizon 600 --jobs 6 --cap 2 \
        --cells hilat-d:10,hilat-m:10,mpp:0,awgn:-12:incar --out runs/loss_defaults.csv
    # the same with the off line above in the environment, --out runs/loss_off.csv; --cap 0 is 500 Hz

`DATA2G_PEP_REF_DB=5` sets the noise against each burst's envelope peak. Displayed SNR is the cell + 5.
`scripts.wide_carrier_loss` is `scripts.loss_study` with the carrier preset added. Do not use
`tools/with_native.py` on this branch.

## Caveats

Scattered pilots (`DATA2G_PILOTS`):
- Mid-SNR mpp and mpg lose 2-7%; mpp -8 about 8% here.
- The w48 channel estimate costs more CPU.
- `find_copy`'s gates and the slip search were not judged with the new layout.
- A flagged burst is decoded at its end, not while it arrives.
- HF profile and 6-symbol frames only.
- The policy is untrained on it.

r2/3:
- Its predictor row is zero-shot: no data of the mode went into the model.

Leak model:
- It now sits on master's per-tone floors; cells with a carrier read the same with it on.
- The 0.15 control threshold is tight (fsk16r25 control rows read 0.16-0.24 on hilat-d, everything else under 0.145).
- Nods are untouched: on hilat-d they are missed 46-61% of the time.
- On c8w50 it changes nothing at 30 Hz and helps at 60 Hz.

c8w50:
- hilat-m 20 dB loses 7-12% in the paired loss study of the connect rule alone (-12.0% ± 3.8 on the old base,
  -7.0% ± 2.3 on the new). It is the rule's own cost: rerun with the `ROBUST_MODES` fault fixed, it read the same.
- In-grid carrier: an 8-tone grid loses about 2.7 dB more to one carrier than 16 tones do. At awgn -16 dB with
  the carrier, fsk16r25 retries connect 0.82 of the time and c8w50 retries 0.00. At -12 dB both connect.
- A connect at the first retry takes 3.3 s longer (23.2 s against 19.9 s).
- A fourth grid in every receiver's search: 5.6% of a core against 3.8% in Python, the fine stage, not the transforms.
- False-lock runs were 1-2 h a source, not the 8-16 h the on-air grids passed. `COH_LOCK` is set from noise only.
- Connects at the mild floors are 6-7 points behind at -18 dB on mpp and mpd.
- A session version bump would be needed before any release: an older station does not hear c8w50 retries.

Acquisition switch:
- Five placements decode about 2 points more bursts at the floors for +57% listening CPU, and gave one
  own-grid false lock an hour on c8r50 and c8w50. Off by default.

General:
- `test_generated_tables_current` fails with the defaults: the native tables are today's.
- Wide-grid specs have no recommendation code, so `modes.encode` raises for them. Nothing in the shifter asks.
- `scripts/cpm_false_locks.py` now draws "other grid" traffic from `arq.modes.MODES`, so c8w50 bursts are in it.
