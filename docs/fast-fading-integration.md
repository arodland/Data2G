# fast-fading-integration

TLDR: four study branches merged for outside study, on master `6dafe77`. Every feature keeps its switch, and the
defaults are each branch's best measured setting, not today's on-air behaviour. Native is ported and follows this
branch's defaults; with every switch off the tree is master's (by hash). In sessions the full combination gains on
every fast-fading cell (hilat-m 10 +469%, nvis-d 10 +40%, hilat-d 10 +36%, lowlat-d 10 +65%) and costs 2-8% on
awgn -15, awgn 0 and mpg 0.

- Base: origin/master `6dafe77`. Built before on `b278852`, `9894b63` and `95c94b1`.
- Merged heads, each as its own merge commit:
  - `fsk-wide-bins` 0b21a45 (#306, native ported)
  - `c32r62-r23` 97b2adf (#307, native ported)
  - `sc-slepian` b44a888, then cc4e543 (#337, stacked on #308 `scattered-pilots` 030bd82; native ported)
- Only here:
  - the three-way tone likelihood (per-tone floors, sub-window sums, leak model), in Python and C++;
  - the leak gate on sub-window sums;
  - the defaults, in Python and C++;
  - one `native_subs.study_skip()` for all toggles;
  - this file.
- What changed since the last build (d00b642):
  - the native ports of all three branches, and the combination ported on top of them;
  - pilots default `fk3d2` (was `mix`);
  - master's mid-burst presence (#331) merged beside the wide grids and the scattered pilots;
  - c8w50 has no recommendation code in native either.
- Not a release candidate. Nothing here has been on air.

## Switches

| Switch | Default here | Off value | What it does |
|---|---|---|---|
| `DATA2G_PILOTS` | `fk3d2` | `off` or empty | OFDM pilots scattered through the frame (the same 1/6 of the cells) instead of a pilot row. `fk3d2` (#337): on `w` and `w48`, `nc/3` pilots on symbols 0, 2 and 4 of each frame, on comb `fa % 3`; `gold` on n10 and n4. Also `mix` (#308: k3d2's stride on `w`/`w48`, gold's on n10/n4), `k3d2`, `gold`, and the study layouts `fk2d3`, `k2d3`, `mixs`. The header copy frame opens and closes on conj(pilot). Wire change, both ends. |
| `DATA2G_SC_SPREAD_LOO`, `DATA2G_SC_COPY_COHERENT`, `DATA2G_SC_COPY_ROWS` | `1` each (with the pilots) | `0` | #308's short-burst receiver remedies. |
| `DATA2G_CPM_LEAK` | `ctl,data,dd,ctlsq` | `off` or empty | CPM soft values treat the energy beside a tone as that tone's Doppler leak, not as noise (`cpm.leak_metric`), on master's per-tone floors, only where a codeword's spill is significant. `ctlsq`: a control codeword whose rows read a significant spill >= 0.15 stays on the square law. Receiver only. |
| `DATA2G_CPM_LEAK_Z` | `2.0` | `0` (no gate) | The gate: the leak model applies to a codeword only where its spill stands this many standard errors over 0. Elsewhere the metric is master's exactly. |
| `DATA2G_CPM_LEAK_K` | `0.5` | | The share of unmodelled leak counted as noise. |
| `DATA2G_CPM_CTL_SQ` | `0.15` | | `ctlsq`'s threshold. |
| `DATA2G_CPM_SPILL` | all 1 | | Per-consumer weight of the spill in the square law's noise (studies). |
| `DATA2G_CPM_R23` | `1` | `0`, `off` or empty | `fsk32r62-r2/3` is a mode: header index 4 on c32r62, recommendation code 54 (CPM index 6). Its `mode_thresholds.json` row (-7.97, -1.17, -1.82, -1.65) loads only while it is offered. Wire addition. |
| `DATA2G_OUTCOME_R23` | on | `0` or `off` | The outcome model is `outcome_predictor_r23.npz` (the installed weights plus a zero-shot row for r2/3). Off: the installed model. Follows `DATA2G_CPM_R23`. |
| `DATA2G_CPM_ONAIR` | `c8w50` | `off` or empty | The wide-bin grid c8w50 is an on-air grid: every receiver searches it. Its modes are control-only and have no recommendation code. |
| `DATA2G_CONNECT_SEQ` | `RRWWW` while c8w50 is on air | `RRRRF` (master's) | The CONNECT retries' modes, one letter a try: R `fsk16r25-r1/2`, F `fsk8r50-r1/3`, W `fsk8w50-r1/2`. CONNECT_TRIES is one more than its length. With c8w50 off the air the default is master's. |
| `DATA2G_ROBUST_RULE`, `DATA2G_ROBUST_FAST` | off | empty | Older names and the in-session variants (#306). |
| `DATA2G_CPM_ACQ_K` | `1` (off) | `1` or `off` | The coarse stage's best K placements go to the fine stage. Small gain, large CPU: not recommended. With `DATA2G_CPM_ACQ_SCREEN`, `DATA2G_CPM_ACQ_FLOOR`. Python only. |
| `DATA2G_CPM_SHARE` | `1` | `0` | c8w50's search rows are sums of c8r50's (same result either way). |
| `DATA2G_CPM_WIDE` | off | | Studies: every wide grid's modes (c16w125 too) in `arq.modes.MODES` for fixed bursts. |

All switches off, which is master:

    DATA2G_PILOTS=off DATA2G_CPM_LEAK=off DATA2G_CPM_R23=off DATA2G_CPM_ONAIR=off DATA2G_CPM_ACQ_K=off

The defaults yield where a feature has no path:
- FM profile: pilots are rows, c8w50 is not on air, r2/3 is not a mode.
- Another frame length, or a hand-set FM-path study switch (`DATA2G_SPREAD_WINDOW`, `DATA2G_EST_WINDOW`,
  `DATA2G_DECODE_EARLY`, `DATA2G_SYMBOL_NOISE`): pilots default to rows. Native applies the same rule.

## Native

Native reads every switch from the same variable as Python, with this branch's default when it is unset ("off"
accepted everywhere). It has the wide grids and c8w50 on air, the connect ladder, the leak model on every grid
(the three-way likelihood: `cpm::tone_metric`, `leak_metric` and `spill_significant` take the grid), r2/3 and
every `DATA2G_PILOTS` layout. Recommendation codes follow Python's: the order of `cpm.SPECS` (r2/3 54; the wide
grids' specs have no code).

`tools/gen_native_tables.py` pins `DATA2G_CPM_ONAIR`, `DATA2G_OUTCOME_R23` and `DATA2G_PILOTS` off, so the
generated tables are master's plus the branches' own rows (cpm_wide.cpp, r2/3's rows trimmed at run time) whatever
the environment. `--check` passes either way.

`tests/native_subs.study_skip()` keeps in Python only what native has not:
- r2/3's zero-shot outcome model (native compiles the installed one): gear, arq, phy, engine and host. So
  `--native` with the defaults runs the shifter and sessions in Python.
- The acquisition study: every CPM search.
- #337's scattered-pilot study switches (`SC_STUDY_ENV`): all but base, waveform, cpm, codes and audio.

The ctests hold master's defaults: they pass with the off line in the environment (33 of 33); with the defaults,
arq, cpm and engine fail on their master goldens.

## The tone likelihood

`cpm.tone_metric` is where the three meet: master's per-tone floors (a carrier), the wide grids' sub-window sums,
and the leak model. One function, `_spill_level` (native `spill_level`), gives N0, the floors, Es and the spill, on
the energies whitened by the floors and at a sub-window's level. The square law (`_tone_gain`) and the leak model
(`leak_metric`) both start from it. Per tone, the noise is its floor plus the leak share; every log term is
counted once per sub-window. The leak gate (`spill_significant`) takes the sub-window count too. With the other
features absent it reduces to each branch's own result. C++ matches Python within 1e-12 on every grid and rate,
the wide ones included (`test_native_cpm.py::test_leak_model`, 39 wide cases).

## Equality checks

All run from each tree's root, with the import path checked to be that tree's. Both after the b44a888 merge and
after the cc4e543 merge:
- All switches off against origin/master `6dafe77`: the CPM grids' bursts, energies, locks, soft values and DD
  soft values hash equal, with and without an in-grid carrier (`runs/ident.py`, `runs/ident_carrier.py`,
  `runs/softhash.py`). OFDM transmit audio, soft values, channel estimates and decodes equal
  (`scripts/sc_equal.py`: 168 arrays, 0 differ).
- Wide grids, leak off, with and without a carrier (`runs/wide_hash.py`): equal to `fsk-wide-bins` 0b21a45.
- On-air grids, leak on (`runs/softhash.py`): equal to `c32r62-r23` 97b2adf.

## Tests

With the native module built. `tests/test_host.py` is run alone (it shares a TCP port).

| Switches | Plain | `--native` |
|---|---|---|
| All off | 2720 passed, 0 failed | 2713 passed, 0 failed |
| Defaults | 2704 passed, 35 failed | 2702 passed, 34 failed |

`tools/build_native.sh --test` with the off line: green.

The failures with the defaults:
- CPM switches (7):
  - `test_arq_policy.py::test_recommend_roundtrip_every_submode` and `test_native_arq.py::test_policy_helpers`:
    the c8w50 modes have no recommendation code.
  - `test_arq_session.py::test_connect_retries_end_in_the_fast_fading_robust_mode`: master's RRRRF ladder.
  - `test_arq_policy.py::test_a_robust_start_bounds_the_first_data_request`: #317's unbounded pick at -13.8 dB
    is r2/3 here.
  - `test_native_arq.py::test_outcome_model`, `test_shifter_decisions_match[7]`, `[9]`: they compare C++'s
    installed model with r2/3's zero-shot one.
- Scattered pilots, `fk3d2` (24). Each also fails on `sc-slepian` cc4e543 itself with `DATA2G_PILOTS=fk3d2`:
  - defects: `test_modem.py::test_timing_slip_recovered[-64]`, `[85]` (no slip search);
    `test_a_steady_tone_does_not_read_as_pilots[n4]`; `test_equalizer.py::test_in_band_tone_does_not_bias_ofdm_cfo_or_jump_alias`
    (see Open items); `test_receiver.py::test_busy_ends_with_the_signal_not_a_false_headers_claim` (one seed);
    `test_sounding.py::test_fading`;
  - premises of pilot rows or seeds picked for today's wire: `test_arq_phy.py::test_data_mer_catches_tx_distortion_and_leaves_clean_links_unchanged`,
    `test_dd.py` x2, `test_native_arq_phy.py` x3 (DD rescue seeds), `test_native_equalizer.py::test_live_estimate_and_refine` x5,
    `test_native_modem.py::test_search_parts_match`, `test_receive_head.py::test_weak_header_at_the_burst_end_is_received`
    (an on-air recording of the old wire);
  - not ported: `test_channel_torch.py` x3, `test_fmtorch.py::test_bmi_agrees_with_the_numpy_receiver`;
  - as designed: `test_receiver.py::test_a_burst_decodes_while_it_arrives` (plain only; a flagged burst waits
    for its end).
- Master #331 x scattered pilots (4), only here: `test_modem.py::test_pilot_present_detects_mid_burst_and_rejects_noise_tones_and_impulses`,
  `test_receiver.py::test_a_burst_joined_partway_reads_as_data2g_on_air[ofdm]`,
  `test_engine.py::test_kiss_waits_for_a_data2g_burst_joined_partway`,
  `test_engine.py::test_kiss_waits_after_its_own_burst_for_one_begun_under_it`. See Open items.

## Session study

`scripts.loss_study_carrier`, 12 seeds (0-11) x 600 s, `--policy shift+cpm`, PEP 5 (`DATA2G_PEP_REF_DB=5`;
displayed SNR is the cell + 5), Python, paired (same seeds in every arm), 6 workers, thread caps 1. One job's peak
RSS 0.70 GB. The CSVs and the chain that ran them are in the main checkout's `runs/ffi-study5/` (`runs/` output is
gitignored). Arms:
- A: all switches off (master).
- B: the defaults (the full combination, fk3d2).
- C: the defaults with `DATA2G_PILOTS=mix`.
- D: fk3d2 alone (every other switch off; connect retries master's RRRRF).

Heads: A at 2400 Hz ran on the b44a888 merge; A at 500 Hz and B, C, D on the cc4e543 merge. Flag-off is
identical on both (Equality checks).

Delivered bit/s, mean of the seeds; change against A ± SE, seeds up/down:

| Cap | Cell | A | B | B vs A | C vs A | D vs A |
|---|---|---|---|---|---|---|
| 2400 | awgn -20 | 11 | 10 | -5.4% ± 27 (3/5) | -5.4% ± 27 (3/5) | +9.0% ± 29 (5/4) |
| 2400 | awgn -19 | 21 | 20 | -1.9% ± 2.7 (1/2) | -1.9% ± 2.7 (1/2) | -1.9% ± 2.7 (1/2) |
| 2400 | awgn -18 | 43 | 42 | -1.9% ± 1.3 (1/4) | -1.9% ± 1.3 (1/4) | -1.9% ± 1.3 (1/4) |
| 2400 | awgn -15 | 82 | 76 | -7.6% ± 2.7 (1/8) | -7.6% ± 2.7 (1/8) | -4.5% ± 4.6 (2/8) |
| 2400 | awgn -12 | 106 | 107 | +1.3% ± 0.5 (5/0) | +1.3% ± 0.5 (5/0) | 0 (0/0) |
| 2400 | awgn -12, carrier12 | 106 | 134 | +27.1% ± 1.1 (12/0) | +28.0% ± 0.8 (12/0) | -1.1% ± 0.8 (0/2) |
| 2400 | awgn 0 | 843 | 825 | -2.2% ± 0.4 (0/12) | -2.9% ± 0.5 (0/12) | -2.2% ± 0.4 (0/12) |
| 2400 | awgn 8, carriers | 829 | 1683 | +103% ± 3 (12/0) | +106% ± 3 (12/0) | +103% ± 3 (12/0) |
| 2400 | mpg 0 | 450 | 424 | -5.8% ± 3.1 (3/9) | -4.1% ± 2.8 (3/7) | -5.8% ± 3.1 (3/9) |
| 2400 | mpg 0, carriers | 228 | 188 | -17.7% ± 11.9 (5/6) | -14.1% ± 11.0 (4/8) | -16.5% ± 10.8 (5/6) |
| 2400 | mpp -12 | 58 | 60 | +4.6% ± 5.8 (5/5) | +1.7% ± 6.3 (5/7) | +7.6% ± 5.3 (6/4) |
| 2400 | mpp -8 | 107 | 113 | +5.9% ± 1.7 (9/1) | +4.5% ± 3.1 (8/2) | -3.2% ± 2.7 (3/7) |
| 2400 | mpp 0 | 486 | 481 | -0.9% ± 1.9 (5/5) | -2.1% ± 1.8 (4/8) | -1.0% ± 1.9 (4/6) |
| 2400 | mpp 0, carriers | 250 | 213 | -14.8% ± 9.0 (4/8) | -19.1% ± 7.6 (2/10) | -15.8% ± 8.5 (4/8) |
| 2400 | mpp 8 | 1706 | 1693 | -0.8% ± 0.4 (3/9) | -1.1% ± 0.4 (2/10) | -0.8% ± 0.4 (3/9) |
| 2400 | mpp 10, carriers | 857 | 1575 | +83.7% ± 5.4 (12/0) | +84.4% ± 4.9 (12/0) | +83.7% ± 5.4 (12/0) |
| 2400 | mpd 0 | 420 | 454 | +8.2% ± 2.8 (9/3) | +10.4% ± 3.0 (11/1) | +8.2% ± 2.8 (9/3) |
| 2400 | mpd 8 | 1568 | 1605 | +2.4% ± 1.8 (8/4) | +4.5% ± 2.2 (10/2) | +2.4% ± 1.8 (8/4) |
| 2400 | hilat-d 0 | 83 | 114 | +36.8% ± 4.0 (12/0) | +36.1% ± 4.2 (12/0) | -1.0% ± 4.3 (5/5) |
| 2400 | hilat-d 10 | 82 | 111 | +36.1% ± 2.9 (12/0) | +35.4% ± 3.0 (12/0) | -2.5% ± 4.1 (4/6) |
| 2400 | hilat-m 10 | 90 | 513 | +469% ± 81 (12/0) | +559% ± 68 (12/0) | +592% ± 84 (12/0) |
| 2400 | hilat-m 20 | 91 | 553 | +510% ± 89 (12/0) | +504% ± 62 (12/0) | +474% ± 91 (11/0) |
| 2400 | lowlat-d 10 | 89 | 147 | +64.7% ± 8.6 (12/0) | +54.4% ± 8.3 (12/0) | +80.0% ± 23 (9/3) |
| 2400 | nvis-d 10 | 664 | 929 | +39.9% ± 6.2 (12/0) | +53.7% ± 6.4 (12/0) | +40.0% ± 6.1 (12/0) |
| 2400 | nvis-d 20 | 874 | 1380 | +57.9% ± 6.3 (12/0) | +65.4% ± 8.0 (12/0) | +56.8% ± 7.3 (12/0) |
| 500 | mpp 0 | 272 | 289 | +6.2% ± 1.6 (11/0) | same as B | same as B |
| 500 | mpd 0 | 174 | 255 | +47.1% ± 5.1 (12/0) | same | same |
| 500 | nvis-d 10 | 145 | 185 | +27.9% ± 8.3 (11/1) | same | same |
| 500 | hilat-m 10 | 42 | 39 | -6.3% ± 3.6 (1/8) | same | -6.3% ± 2.2 (1/8) |

The combination against its parts:
- B vs D (the CPM changes on top of fk3d2): hilat-d 0 / 10 +38% / +40% (12/0), awgn -12 with carrier12 +28.6%
  (12/0), mpp -8 +9.5% ± 3.3 (9/2), awgn -12 +1.3% (5/0); hilat-m 10 -17.8% ± 15.7 (5/6), lowlat-d 10 -8.5% ± 10.5
  (6/5), mpp -12 -2.8% ± 2.0 (0/2), awgn -15 -3.2% ± 3.2 (0/1), awgn -20 -13% ± 16; every other cell within 1.5%.
- B vs C (fk3d2 against mix, both with the CPM changes): nvis-d 10 / 20 -9.0% ± 5.6 / -4.5% ± 2.2, mpd 8
  -2.0% ± 1.0, hilat-m 10 -13.6% ± 14.8; lowlat-d 10 +6.6% ± 6.1, mpp -12 +2.8% ± 1.9, awgn 0 +0.7% ± 0.3. Level at
  500 Hz.

Interactions (B loses to a single part by more than its SE):
- hilat-m 10 at 2400 Hz: B 513 against D's 625 (-17.8% ± 15.7). The CPM-only change gained 27% there on master
  (#307's table, OFDM on pilot rows failing). With the pilots, OFDM carries the cell; c8w50's RRWWW connects
  (#306 measured about 4-5% in-session on hilat-m 10/20) and r2/3's zero-shot row moving shifter time onto CPM
  are the likely costs. The seed spread is wide (5/6); hilat-m 20 goes the other way (+6.4% ± 12.3).
- mpp -12: B 60 against D's 62 (-2.8% ± 2.0, 0/2 seeds): two seeds, each a few bit/s.
- lowlat-d 10 (-8.5% ± 10.5) and awgn -15 (-3.2% ± 3.2) are within their SE. awgn -15's -7.6% against master is
  about 4.5 points from the pilots (D) and 3 from the CPM changes (r2/3, as in the last build's split).
- No cell where B loses to the CPM-only change implied by B vs D: wherever the CPM changes gain on master (hilat-d,
  carrier12, mpp -8, awgn -12) they gain the same on top of fk3d2.

Against #337's own table (fk3d2 vs off, master 43b2968 + #308): D reproduces it exactly on all 21 shared cells
(bit/s, %, SE and seed counts). So neither cc4e543's decimated copy search nor master's commits since 43b2968 move
these sessions. B differs from #337's fk3d2 only where the CPM changes act: hilat-m 10 (513 against 625, the
interaction above), lowlat-d 10 (147 against 160), mpp -8 (113 against 103), mpp -12 (60 against 62), awgn -15
(76 against 79), and the cells #337 did not run (hilat-d, carrier12). C against #337's mix column: hilat-m 10
595 against 640, the same interaction.

## Open items

- Master #331's mid-burst OFDM presence (`modem.pilot_present`, `fold_verify`) folds frame-apart pilot rows: with
  scattered pilots a burst joined partway does not read as `data2g_on_air`, so BUSY misses it (4 tests above). The
  loss study does not exercise it. Needs a presence rule for scattered pilots before any on-air use.
- fk3d2 reads #330's in-band tone CFO 0.126 Hz off (tone at 825 Hz) against the test's 0.075 Hz bound
  (`test_in_band_tone_does_not_bias_ofdm_cfo_or_jump_alias`; #337 raised the bound to 0.075 for mix).
- hilat-m 10: the CPM changes cost the pilots about 18% there (Interactions).

## Caveats

Scattered pilots (`DATA2G_PILOTS`):
- `fk3d2` keeps mix's hilat-m gain and beats it on lowlat-d; mix stays about 9 points ahead on nvis-d 10.
- awgn -15 loses about 4.5% to the pilots, mpg 0 about 6%, awgn 0 2%.
- No slip search; the n4 pilot-coherence gate is not recalibrated for gold. HF profile and 6-symbol frames only.
- A flagged burst is decoded at its end, not while it arrives.
- The policy is untrained on it.
- CPU: #337's CM4 numbers (w48 receive 150-180 ms with fk3d2) are on its own branch; no Pi bench on this build.

r2/3:
- Its predictor row is zero-shot. Native compiles the installed model, so the shifter stays Python under it.

Leak model:
- Gated: a codeword's metric is master's unless its spill is significant.
- The 0.15 control threshold is tight (fsk16r25 control rows read 0.16-0.24 on hilat-d).
- Nods are untouched: on hilat-d they are missed 46-61% of the time.

c8w50 (`RRWWW` retries):
- Connect gains and costs as #306 measured: hilat-d -16 dB 0.20 -> 0.93 connected; awgn -16 with an in-grid
  carrier 0.82 -> 0.48.
- A fourth grid in every receiver's search. No presence rule of its own (`cpm.PRESENCE` has none; c8r50's hears its
  tones).
- A session version bump would be needed before any release: an older station does not hear c8w50 retries.

Acquisition switch: off by default, Python only.
