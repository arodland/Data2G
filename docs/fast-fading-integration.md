# fast-fading-integration

TLDR: three study branches merged for outside study, on master `b278852` (which now includes PR #317). Every feature keeps its
switch, and the defaults are each branch's best measured setting, not today's on-air behaviour. Python only: native
is not ported, so native builds and `--native` tests do not match this branch with its defaults.

- Base: origin/master `b278852`. Built before on `9894b63` and `95c94b1`. The measurements below were taken on
  master `1f12161` + PR #317 (merged from its head, 36492a6, before it landed; it landed unchanged), so they
  describe this tree: what master added since is AGENTS.md and a GUI fix (#319, #320).
- Merged heads, each as its own merge commit:
  - `fsk-wide-bins` 1e7f391 (#306)
  - `c32r62-r23` e94ab39 (#307)
  - `scattered-pilots` bbc0b40 (#308)
- Only here: the three-way tone likelihood (per-tone floors, sub-window sums, leak model), the leak gate on sub-window
  sums, the defaults, one `native_subs.study_skip()` for all toggles, this file.
- Not a release candidate. Nothing here has been on air.

## Switches

| Switch | Default here | Off value | What it does |
|---|---|---|---|
| `DATA2G_PILOTS` | `mix` | `off` or empty | OFDM pilots scattered through the frame (same 1/6 of the cells) instead of a pilot row: k3d2's stride on `w` and `w48`, the golden-ratio stride on n10 and n4. `k3d2`, `gold`, `mix`. The header copy frame opens and closes on conj(pilot). Wire change, both ends. |
| `DATA2G_SC_SPREAD_LOO`, `DATA2G_SC_COPY_COHERENT`, `DATA2G_SC_COPY_ROWS` | `1` each (with the pilots) | `0` | Short-burst receiver remedies: the spread by the pilots' leave-one-out fit on `w` and `w48`; find_copy adds the combs coherently; copies gated on the copy frame's two rows. Off one at a time, hilat-m 10 dB loses 24%, 37% and 37%. |
| `DATA2G_CPM_LEAK` | `ctl,data,dd,ctlsq` | `off` or empty | CPM soft values treat the energy beside a tone as that tone's Doppler leak, not as noise (`cpm.leak_metric`), on master's per-tone floors, only where a codeword's spill is significant. `ctlsq`: a control codeword whose rows read a significant spill >= 0.15 stays on the square law. Receiver only. |
| `DATA2G_CPM_LEAK_Z` | `2.0` | `0` (no gate) | The gate: the leak model applies to a codeword only where its spill stands this many standard errors over 0; elsewhere the metric is master's exactly. |
| `DATA2G_CPM_LEAK_K` | `0.5` | | The share of unmodelled leak counted as noise. |
| `DATA2G_CPM_CTL_SQ` | `0.15` | | `ctlsq`'s threshold. |
| `DATA2G_CPM_SPILL` | all 1 | | Per-consumer weight of the spill in the square law's noise (studies). |
| `DATA2G_CPM_R23` | `1` | `0`, `off` or empty | `fsk32r62-r2/3` is a mode: header index 4 on c32r62, recommendation index 6. Its `mode_thresholds.json` row (regenerated: -7.97, -1.17, -1.82, -1.65) loads only while it is offered. Wire addition. |
| `DATA2G_OUTCOME_R23` | on | `0` or `off` | The outcome model is `outcome_predictor_r23.npz` (the installed weights with a zero-shot row for r2/3). Off: the installed model. Follows `DATA2G_CPM_R23`. |
| `DATA2G_CPM_ONAIR` | `c8w50` | `off` or empty | The wide-bin grid c8w50 is an on-air grid: every receiver searches it. Its modes are control-only and have no recommendation code. |
| `DATA2G_CONNECT_SEQ` | `RRWWW` while c8w50 is on air | `RRRRF` (master's) | The CONNECT retries' modes, one letter a try: R `fsk16r25-r1/2`, F `fsk8r50-r1/3`, W `fsk8w50-r1/2`. CONNECT_TRIES is one more than its length; a host's last try is in its last letter. With c8w50 off the air the default is master's `RRRRF`. |
| `DATA2G_ROBUST_RULE` | off | `off` or empty | Older names: `connect` (WWWWW), `alt` (RWRWR); `replace` makes the robust mode itself c8w50 in sessions too. |
| `DATA2G_ROBUST_FAST` | off | `off` or empty | `wide`, `always`: c8w50 as the in-session robust mode too (measured worse). |
| `DATA2G_CPM_ACQ_K` | `1` (off) | `1` or `off` | The coarse stage's best K placements go to the fine stage. Small gain, large CPU: not recommended. With `DATA2G_CPM_ACQ_SCREEN`, `DATA2G_CPM_ACQ_FLOOR`. |
| `DATA2G_CPM_SHARE` | `1` | `0` | c8w50's search rows are sums of c8r50's (same result either way). |
| `DATA2G_CPM_WIDE` | off | | Studies: every wide grid's modes (c16w125 too) in `arq.modes.MODES` for fixed bursts. c16w125 is never on air. |

PR #317 (now master's) has no switch. It bounds a robust start's first data request to modes whose
steady-channel threshold (`MODE_THRESHOLDS`' first column) is within `snr_est` + 2 dB, with that burst's control
sent twice, until a data burst of the peer's is usable.

All switches off, which is master:

    DATA2G_PILOTS=off DATA2G_CPM_LEAK=off DATA2G_CPM_R23=off DATA2G_CPM_ONAIR=off DATA2G_CPM_ACQ_K=off

The defaults yield where a feature has no path:
- FM profile (`DATA2G_PROFILE=fm`): pilots are rows, c8w50 is not on air, r2/3 is not a mode. An explicit
  `DATA2G_PILOTS` value still raises there, as on its branch.
- Another frame length (`DATA2G_SYMS_PER_FRAME`) or a hand-set FM-path study switch (`DATA2G_SPREAD_WINDOW`,
  `DATA2G_EST_WINDOW`, `DATA2G_DECODE_EARLY`, `DATA2G_SYMBOL_NOISE`): pilots default to rows.

`tests/native_subs.study_skip()` keeps in Python every native provider a toggle in effect changes. It reads the
modules' settings, not the environment, since most toggles are on here by default. So `--native` and
`tools/with_native.py` with the defaults run nearly everything in Python; use the off line above for a native
comparison.

## The tone likelihood

`cpm.tone_metric` is where the three meet: master's per-tone floors (a carrier), the wide grids' sub-window sums,
and the leak model. One function, `_spill_level`, gives N0, the floors, Es and the spill, on the energies whitened
by the floors and at a sub-window's level. The square law (`_tone_gain`) and the leak model (`leak_metric`) both
start from it. Per tone, the noise is its floor plus the leak share; every log term is counted once per
sub-window. The leak gate (`spill_significant`) takes the sub-window count too. With the other features absent it
reduces to each branch's own result (checked by hash, below).

## Equality checks

All run from each tree's root, with the import path checked to be that tree's.

- All switches off against master `1f12161` + PR #317 (a scratch merge; master `b278852` adds no code these touch): the CPM grids' bursts, energies, locks, soft
  values and DD soft values hash equal, with and without an in-grid carrier (`runs/ident.py`,
  `runs/ident_carrier.py`, `runs/softhash.py`). OFDM transmit audio, soft values, channel estimates and decodes
  equal (`scripts/sc_equal.py`: 168 arrays, 0 differ).
- Wide grids, leak off, with and without a carrier (`runs/wide_hash.py`): equal to `fsk-wide-bins` 00335b3.
- On-air grids, leak on, with the gate (`runs/softhash.py`): equal to `c32r62-r23` 29279eb.

## Tests

With the native module built (`tools/build_native.sh`, with the off line in the environment: the generator checks
the tables against the settings in effect). Run a suite alone: `tests/test_host.py` shares a TCP port.

| Switches | Plain | `--native` |
|---|---|---|
| All off | 2473 passed, 0 failed | 2446 passed, 0 failed |
| Defaults | 888 passed, 19 failed | 865 passed, 18 failed |

With the defaults most parity tests skip: they compare C++ with a Python reference the toggles change by design.
The feature tests of all three branches pass. The 19 failures assert today's wire or today's receiver and were left
as they are:

- Scattered pilots (15):
  - Defects, see Caveats:
    - `test_modem.py::test_timing_slip_recovered[-64]`, `[85]`
    - `test_modem.py::test_a_steady_tone_does_not_read_as_pilots[n4]`
    - `test_equalizer.py::test_a_tone_on_one_carrier_does_not_sink_the_burst`
    - `test_receiver.py::test_busy_ends_with_the_signal_not_a_false_headers_claim` (one seed holds 4.000000003 s
      against a 4.0 s bound; the false-lock test that failed before now passes)
    - `test_sounding.py::test_fading`
  - Not ported: `test_channel_torch.py` x3, `test_fmtorch.py::test_bmi_agrees_with_the_numpy_receiver`.
  - As designed: `test_receiver.py::test_a_burst_decodes_while_it_arrives` (a flagged burst waits for its end;
    python only), `test_receive_head.py::test_weak_header_at_the_burst_end_is_received` (an on-air recording of
    the old wire), `test_dd.py` x2 and `test_arq_phy.py::test_long_control_is_pinned_where_its_padding_starts`
    (seeds picked for today's wire).
- c8w50 (2): `test_arq_session.py::test_connect_retries_end_in_the_fast_fading_robust_mode` (master's RRRRF
  ladder), `test_arq_policy.py::test_recommend_roundtrip_every_submode` (the c8w50 modes have no code).
- r2/3 (1): `test_arq_policy.py::test_a_robust_start_bounds_the_first_data_request` (#317's test: its unbounded
  pick at -13.8 dB is r2/3 here, not r1/2; the bounded picks hold).
- Native (1): `test_native_parity.py::test_generated_tables_current` (the tables are master's).

The FM profile was not rerun on this build.

## Smoke loss study

`scripts.loss_study_carrier`, 12 seeds x 600 s, `shift+cpm`, PEP 5, Python, all switches off (master) against the
defaults, paired. Delivered bit/s, mean of the seeds, change ± SE, seeds up / down. The all-off arm is
`runs/smoke3/` (the all-off code is identical to master by hash); the defaults arm with the short-burst remedies is
`runs/smoke4/`.

| Cap | Cell | All off | Defaults | Change | Seeds | Before the remedies |
|---|---|---|---|---|---|---|
| 2400 | hilat-d 10 | 82 | 110 | +35% ± 3 | 12/0 | +36% |
| 2400 | hilat-m 10 | 89 | 646 | +623% ± 60 | 12/0 | +220% |
| 2400 | hilat-m 20 | 89 | 555 | +527% ± 78 | 12/0 | +242% |
| 2400 | lowlat-d 10 | 92 | 131 | +42% ± 7 | 12/0 | +51% |
| 2400 | nvis-d 10 | 631 | 1052 | +67% ± 3 | 12/0 | +51% |
| 2400 | mpd 0 | 420 | 455 | +8.4% ± 2.6 | 10/1 | +14.2% |
| 2400 | mpp 0 | 483 | 485 | +0.3% ± 2.0 | 6/4 | -5.3% |
| 2400 | mpp -8 | 107 | 111 | +3.6% ± 2.2 | 6/2 | -3.6% |
| 2400 | awgn -12 | 106 | 107 | +1.3% ± 0.5 | 5/0 | +1.3% |
| 2400 | awgn -12, in-grid carrier | 106 | 134 | +27% ± 1 | 12/0 | +28% |
| 2400 | awgn -15 | 82 | 76 | -7.6% ± 2.7 | 1/8 | -7.6% |
| 2400 | awgn -19 | 21 | 20 | -1.9% ± 2.7 | 1/2 | -1.9% |
| 2400 | awgn -20 | 11 | 10 | -5.4% ± 27 | 3/5 | -5.4% |
| 2400 | awgn 6 | 1730 | 1714 | -1.0% ± 0.0 | 0/12 | -1.0% |
| 500 | mpd 0 | 179 | 251 | +40.5% ± 5.3 | 12/0 | +40.5% |
| 500 | mpp 0 | 270 | 290 | +7.3% ± 1.5 | 11/0 | +7.3% |

- The remedies against the defaults before them, paired: hilat-m 10 / 20 +126% / +83%, mpp -8 +7.4% (9/2), mpp 0
  +5.9% (9/3), nvis-d 10 +10.5% (10/2); mpd 0 -5.1% ± 1.8 (2/10), lowlat-d 10 -5.7% ± 7.1; every other cell 0.
- mpd 0's loss has no single cause found. Most of it is one seed (479 to 367); the leave-one-out spread accounts
  for about 2 points of it; the coherent copy search and the row gate for none. Accepted for now.
- The carrier is `carrier12`: a steady 1573.4 Hz tone 12 dB over the floor in 50 Hz (#295's cell).
- hilat-d's gain is smaller than on the first builds because master's #313 and #316 lifted the all-off arm from 41
  to 82 bit/s.
- awgn -12's earlier +17.7% came from r2/3 on its zero-shot row. With r2/3's regenerated threshold row, robust
  start's bound and the changed replies, the shifter picks it less there.
- 2 sessions of 12 deliver nothing at awgn -20 in every arm, plain master included.

PR #317 alone in this harness (plain `1f12161` against `1f12161` + #317, every switch off): awgn -20 -6.7% ± 9.5
(1/2), -19 -1.3% ± 3.6 (2/3), -15 +4.3% ± 3.1 (2/0). In this loss study #317 is neutral. Its gain is in live
sessions' completion: the c32r62-r23 agent's session study, our scale's -20 dB, 3 to 13 of 16 completed.

awgn -15 by switch (the defaults with one switch off, same seeds):

| Arm | bit/s | Against the defaults |
|---|---|---|
| Defaults | 76 | |
| Pilots off | 81 | +6.0% ± 2.7 (7/1) |
| r2/3 off | 79 | +3.3% ± 3.3 (1/0) |
| Leak off | 76 | 0 (0/0) |
| c8w50 off | 76 | 0 (0/0) |

The scattered pilots cost that cell about 6% and r2/3 about 3%; the leak (now gated) and c8w50 nothing. The
previous build's split of mpp -8 put about 8% on the pilots as well.

To rerun (no native):

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_PEP_REF_DB=5
    python -m scripts.loss_study_carrier --policy shift+cpm --seeds 12 --horizon 600 --jobs 6 --cap 2 \
        --cells hilat-d:10,mpp:-8,awgn:-15,awgn:-12:carrier12 --out runs/loss_defaults.csv
    # the same with the off line above in the environment; --cap 0 is 500 Hz
    python scripts/paired_loss.py runs/loss_off.csv runs/loss_defaults.csv

`DATA2G_PEP_REF_DB=5` sets the noise against each burst's envelope peak. Displayed SNR is the cell + 5. Do not use
`tools/with_native.py` with the defaults (see Switches).

## Caveats

Scattered pilots (`DATA2G_PILOTS`):
- The short-burst remedies took mpp 0 and mpp -8 from -5% and -4% to level; awgn -15 still loses about 6% (the
  split above) and mpd 0 at 2400 Hz lost 5% to them (see the smoke notes). Header recall with the preamble cut is
  under half of pilot rows' (0.09 against 0.18 on slow cells).
- The false lock is fixed: the copy frame (pilot, header, pilot) had the shape of the real header (last preamble
  repeat, header, header pilot), so a preamble hypothesis two frames early read the header as its copy. The copy
  frame now opens and closes on conj(pilot). Own-traffic runs, 3 h: busy off air 13.4 s/h against 11.6 with
  pilot rows (16.8 before the fix); false headers and received bursts unchanged.
- `test_busy_ends_with_the_signal_not_a_false_headers_claim` fails on one seed by a hold of 4.000000003 s
  against a 4.0 s bound (over 4 seeds the hold is 10.4 s against 21.6 s before).
- No slip search; no per-carrier noise; the n4 pilot-coherence gate is not recalibrated for gold.
- The w48 channel estimate costs more CPU (250-290 ms once per new statistics, about 120 ms a burst on a fast
  channel, Python).
- A flagged burst is decoded at its end, not while it arrives. HF profile and 6-symbol frames only.
- The policy is untrained on it.

r2/3:
- Its predictor row is zero-shot: no data of the mode went into the model.
- Its threshold row was regenerated from its own measurements (the first one rated it more robust than r1/2 on
  AWGN, which let #317's bound pass it where r1/2 was barred).

Leak model:
- Gated: a codeword's metric is master's unless its spill is significant (390 of 402 low-SNR AWGN/mpp/mpg codewords
  bit-identical). The gate removed an awgn -15 loss that a spill reading on noise caused.
- On its own against master: hilat-d +9% to +13%, connects there in 22 s against 46-51 s; 0 elsewhere.
- The 0.15 control threshold is tight (fsk16r25 control rows read 0.16-0.24 on hilat-d, everything else under 0.145).
- Nods are untouched: on hilat-d they are missed 46-61% of the time.

c8w50 (`RRWWW` retries):
- Connect study on master `1f12161`, leak on, 40-120 attempts a cell: hilat-d -16 dB 0.20 -> 0.93 connected,
  -14 0.62 -> 1.00; 30 Hz / 12 ms -16 0.07 -> 0.93; 60 Hz / 7 ms -12 0 -> 0.97, -14 0 -> 0.68; hilat-d from -8 dB
  up in 20 s against master's 54-67 s (the leak's doing).
- Costs: awgn -16 with an in-grid carrier 0.82 -> 0.48 (an 8-tone grid loses about 2.7 dB more to one carrier);
  mpp -18 0.93 -> 0.89, mpd -18 0.88 -> 0.86; in-session about 4% on hilat-m 10/20 against the leak alone (sessions
  connected through c8w50). `RRRWW` halves both the gain and the cost; `RRRRF` is master's.
- A fourth grid in every receiver's search: 5.6% of a core against 3.8% in Python.
- False-lock runs were 1-2 h a source, not the 8-16 h the on-air grids passed. `COH_LOCK` is set from noise only.
- A session version bump would be needed before any release: an older station does not hear c8w50 retries.

PR #317 (landed on master after this was measured, unchanged):
- In this loss study it is neutral; in live sessions it rescues the
  weakest AWGN cell. Its 2 dB margin is the one value its author tried (0 dB slower at -16 to -14, 4 dB loses the
  rescue).

Acquisition switch:
- Five placements decode about 2 points more bursts at the floors for +57% listening CPU, and gave one own-grid
  false lock an hour on c8r50 and c8w50. Off by default.

General:
- `test_generated_tables_current` fails with the defaults: the native tables are master's.
- Wide-grid specs have no recommendation code, so `modes.encode` raises for them. Nothing in the shifter asks.
- `scripts/cpm_false_locks.py` draws "other grid" traffic from `arq.modes.MODES`, so c8w50 bursts are in it.
