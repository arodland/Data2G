# Outcome-model training data

Every dataset in the current training mix: how it was made, what steered it,
and why it is there, so a future retrain can rebuild or extend the mix.

**Archive.** `runs/` is gitignored (but for a few inputs and the `*.sh`
drivers) and worktrees get deleted (v10's 2900
sessions and 83,000 offline samples went with `outcome-capacity`). Copies of
every file below, with `MANIFEST.sha256` (sha256, rows), are in the main
checkout's `runs/outcome-data/`. Keep it outside any worktree.

**Common to all.** `DATA2G_PEP_REF_DB=5` (noise against each burst's peak).
`session_data.py` and `outcome_data.py` with `--first` seeds as listed, so a
re-run with the same code, model and seeds reproduces the rows. Sessions that
record the link history or energy inputs need the shifter in Python: set
`DATA2G_OUTCOME_MODEL` (then `tools/with_native.py` leaves the shifter
unsubstituted) and pass `DATA2G_LOGIT_OFFSETS` explicitly.

## Session sets

Rows: one per burst a receiver had a measurement for (`session_data.py`
docstring). "Steering" is the model whose recommendations the sessions
followed; exploration is `session_data.EXPLORE` (0.2) unless noted.

| file | generator (sessions, `--first`) | kind | steering | code | why |
|---|---|---|---|---|---|
| `session_data_v11.csv` | `session_data --sessions 2900 --first 700000` (`runs/v11_round.sh`) | regular: -8..22 dB drifting, every channel kind, 300 s, cap 500 Hz 1/4 | v10, no logit offsets, `DATA2G_BIAS_FIX=1` | f8d8e0c | v12's own sessions |
| `session_data_slow.csv` | `--slow --sessions 800 --first 900000` (`runs/v12_round.sh`) | sustained -8..0 dB, MPG 70% else 0.05-0.3 Hz, 600 s | v11b, offset w48-16qam-r1/2 -1.0, bias fix | 05c4b40 | MPG -4 (v12's supplement) |
| `session_data_ir_clean.csv` | `with_native session_data --sessions 2900 --first 3000000` (`runs/interference_round.sh`, interference-sim) | regular | v12 as installed (C++ shifter), offset w48-16qam-r1/2 -1.0 | 1b6f2d1 | clean arm A of the interference rounds |
| `session_data_ir_clean_slow.csv` | `--slow --sessions 800 --first 3100000` (same) | slow | same | 1b6f2d1 | |
| `session_data_ir_clean_high.csv` | `--high --sessions 1200 --first 3200000` (same) | +10..+30 dB drifting, fading only | same | 1b6f2d1 | high-SNR fading (cpmc lost 13-22% there) |
| `session_data_n10p.csv` | `--sessions 2900 --first 3400000` (`runs/n10p_round.sh`) | regular; records link history + energy | v12 + n10 extension (`runs/outcome_n10ext.npz` = 2def7a1's model), offsets w48-16qam-r1/2 -1.0, n10-256l-r3/4 -0.5, Python shifter | 8ab5379 | the n10 modes in sessions; energy inputs |
| `session_data_n10p_slow.csv` | `--slow --sessions 800 --first 3500000` (same) | slow | same | 8ab5379 | |
| `session_data_n10p_high.csv` | `--high --sessions 1200 --first 3600000` (same) | high | same | 8ab5379 | |
| `session_data_n10p_awgnhigh.csv` | `--high --kinds awgn --sessions 400 --first 3700000` (`runs/n10p2_round.sh`) | +10..+30 dB AWGN | same | bea359f | `--high` has no AWGN: with energy, AWGN +15 lost 10% |
| `session_data_n10p_fastlow.csv` | `--fastlow --sessions 1200 --first 3800000` (`runs/n10f_round.sh`) | sustained -8..+6 dB on MPP/MPD/random 1-3 Hz, 600 s; exploration 0.35, 70% of it at modes up to 2.5x faster than the pick | same | 55162be | the faster modes' failure edges at MPD 0 and MPP -8 |
| `session_data_e2.csv` | `--sessions 2900 --first 5000000` (`runs/e2_round.sh`) | regular; energy as the engine measures it (NoiseProfile.span_snr_db against PHY.peak_db) | v12 + n10 extension gated with n10qf (`codes_data/outcome_predictor{,_gate}.npz`, 0.5 Hz / 12 dB), offsets as the n10p sets, Python shifter | 570cf8a | the engine-measured energy inputs |
| `session_data_e2_slow.csv` | `--slow --sessions 800 --first 5100000` (same) | slow | same | 570cf8a | |
| `session_data_e2_high.csv` | `--high --sessions 1200 --first 5200000` (same) | high | same | 570cf8a | |
| `session_data_e2_awgnhigh.csv` | `--high --kinds awgn --sessions 400 --first 5300000` (same) | high AWGN | same | 570cf8a | |
| `session_data_e2_fastlow.csv` | `--fastlow --sessions 1200 --first 5400000` (same) | fastlow | same | 570cf8a | |
| `session_data_cp.csv` | `--sessions 2900 --first 6000000` (`runs/cp_round.sh`) | regular; control padding pinned (`phy.PIN_CUTS`) | n10pe gated with n10qf, CPM floor, offsets w48-16qam-r1/2 -1.0, n10-256l-r3/4 -0.5, Python shifter | 7e54f2d | the pinned receiver in sessions |
| `session_data_cp_slow.csv` | `--slow --sessions 800 --first 6100000` (same) | slow, pinned | same | 7e54f2d | |
| `session_data_cp_high.csv` | `--high --sessions 1200 --first 6200000` (same) | high, pinned | same | 7e54f2d | |
| `session_data_cp_awgnhigh.csv` | `--high --kinds awgn --sessions 400 --first 6300000` (same) | high AWGN, pinned | same | 7e54f2d | |
| `session_data_cp_fastlow.csv` | `--fastlow --sessions 1200 --first 6400000` (same) | fastlow, pinned | same | 7e54f2d | |

| `session_data_gt.csv` | `--explore-faster --sessions 2900 --first 7000000` (`runs/gt_round.sh`) | regular; FASTLOW's exploration on every kind (`--explore-faster`) | n10pe gated with n10qf, CPM floor, offsets w48-16qam-r1/2 -1.0, n10-256l-r3/4 -0.5, Python shifter | 8340bbf (PR #91's noise-gated receiver) | the fast region's model (gtp) |
| `session_data_gt_slow.csv` | `--slow --explore-faster --sessions 800 --first 7100000` (same) | slow | same | 8340bbf | |
| `session_data_gt_high.csv` | `--high --explore-faster --sessions 1200 --first 7200000` (same) | high | same | 8340bbf | |
| `session_data_gt_awgnhigh.csv` | `--high --kinds awgn --explore-faster --sessions 400 --first 7300000` (same) | high AWGN | same | 8340bbf | |
| `session_data_gt_fastlow.csv` | `--fastlow --explore-faster --sessions 1200 --first 7400000` (same) | fastlow | same | 8340bbf | |
| `session_data_gs.csv` | `--sessions 2900 --first 10000000` (`runs/gs_round.sh`), `DATA2G_SPREAD_CARRY=1` | regular; a defaulted spread_est carries the peer's last measured one; no `--explore-faster` | the 512x3 student (PR #93), offsets as shipped, Python shifter | 3f50fe9 | the installed CNP's data |
| `session_data_gs_slow.csv` | `--slow --sessions 800 --first 10100000` (same) | slow | same | 3f50fe9 | |
| `session_data_gs_high.csv` | `--high --sessions 1200 --first 10200000` (same) | high | same | 3f50fe9 | |
| `session_data_gs_awgnhigh.csv` | `--high --kinds awgn --sessions 400 --first 10300000` (same) | high AWGN | same | 3f50fe9 | |
| `session_data_gs_fastlow.csv` | `--fastlow --sessions 1200 --first 10400000` (same) | fastlow | same | 3f50fe9 | |
| `session_data_cor.csv` | `--sessions 2900 --first 13000000` (`runs/cor_round.sh`), `DATA2G_SPREAD_CARRY=1` | regular | the installed CNP (cnpd), offsets as shipped, Python shifter | 12d1235 (#126: CPM acquisition gates + GF(M) code, protocol 14) | CPM outcomes on the new CPM receiver |
| `session_data_cor_slow.csv` | `--slow --sessions 800 --first 13100000` (same) | slow | same | 12d1235 | |
| `session_data_cor_high.csv` | `--high --sessions 1200 --first 13200000` (same) | high | same | 12d1235 | |
| `session_data_cor_awgnhigh.csv` | `--high --kinds awgn --sessions 400 --first 13300000` (same) | high AWGN | same | 12d1235 | |
| `session_data_cor_fastlow.csv` | `--fastlow --sessions 1200 --first 13400000` (same) | fastlow | same | 12d1235 | |
| `session_data_cor_floor.csv` | `--floor --sessions 1500 --first 13500000` (same) | sustained -19..-4 dB, every kind, 600 s; while the CPM floor holds, exploration picks CPM modes only | same | 8bc4f5d (12d1235 + `--floor`) | the CPM floor cells (-18..-9 dB), which the other kinds reach 0.5% of the time |
Older sets carry no link-history or energy columns: their rows read as "no
history" (the inputs' flag 0).

## Offline sets

`outcome_data.py`: one sample is a measured history plus 4 candidate bursts
sent at the same moment; one row per candidate.

| file | generator | steering | code | why |
|---|---|---|---|---|
| `outcome_data_n10top.csv` | `with_native outcome_data --cands n10-64l-r3/4,n10-256l-r3/4 --samples 8000 --first 3000000` (`runs/n10_top_data.sh`) | none (candidates fixed) | 2613567 | the two new 500 Hz modes, 0..36 dB |
| `outcome_data_general.csv` | `with_native outcome_data --samples 40000 --first 4000000` (`runs/n10p3_round.sh`) | none | 046f8be | stands in for v10's lost offline set: retrains without one lost 4-10% on MPD |
| `outcome_data_general_cp.csv` | `with_native outcome_data --samples 40000 --first 4000000` (`runs/cp_round.sh`) | none | 7e54f2d | the general set's seeds again, control padding pinned: only the decoder differs |
| `outcome_data_general_gt.csv` | `with_native outcome_data --samples 40000 --first 7500000` (`runs/gt_round.sh`) | none | 8340bbf | the general set on the noise-gated receiver, new seeds |
| `outcome_data_n10top_gt.csv` | `with_native outcome_data --cands n10-64l-r3/4,n10-256l-r3/4 --samples 8000 --first 7600000` (same) | none | 8340bbf | the two top n10 modes, likewise |
| `outcome_data_general_cor.csv` | `with_native outcome_data --samples 40000 --first 13600000` (`runs/cor_round.sh`) | none | 12d1235 | the general set on the #126 CPM receiver, new seeds |

## Mixes

- **Shipped (v12 + n10 extension):** v12 as trained (v10's sessions and offline set, v11, slow;
  lost) plus output units for the n10 modes from `outcome_data_n10top.csv`
  (`train_outcome --extend`).
- **n10pf / n10qf** (`runs/n10f_round.sh`): the n10p sets, the older sets and both offline sets; n10pf with
  `--energy-inputs` (the sim's oracle energy, superseded). n10qf ships as the gate's model.
- **n10pe / n10qe** (`runs/e2_round.sh`): the e2 sets instead of the n10p ones, plus the general offline set, the
  older sets and `outcome_data_n10top.csv`; n10pe with `--energy-inputs` (engine-measured).
  Five bootstrap members, seeds 1-5, averaged.
- **cpe / cpo / cpn** (`runs/cp_round.sh`, `runs/cp_round2.sh`; not shipped): the cp sets in place of the e2
  ones. cpe and cpo with the older sets as n10pe (cpe with `outcome_data_general_cp.csv`, cpo with the
  original general set); cpn the cp sets, `outcome_data_general_cp.csv` and `outcome_data_n10top.csv` only.
  `--energy-inputs`, five members each. Against n10pe on the pinned receiver (`runs/cp_judge.sh`) all three
  traded cells: +4 to +9% at MPD 0 and MPP -4/0, losses at MPP -6, MPG +8 and high-SNR fading.
- **gtp / gtq** (`runs/gt_round.sh`): the gt sets only (no set from an older receiver), five members each;
  gtp `--energy-inputs`. As a pair against the shipped pair (`runs/gt_judge.sh`): MPD 0 +16%, MPP 0 +7%, but
  AWGN 0 -20%, AWGN +4 -12% (gtq's codeword head reads 0.83-0.95 where 1.00 decoded) and MPG -2..-4%. A second
  gtp on the same data (seeds 6-10, `runs/gt_var.sh`) moved mid/high-SNR cells by under 1%: the trades are the
  data's, not training noise. gtp ships as the fast region's model only (`predictor.FAST`, `runs/fast_judge.sh`).
- **h512d3 (installed; distilled):** one 512x3 tanh MLP trained on the router's (n10pe, n10qf gate, gtp fast
  region) dense outputs over every gt row (`scripts/distill_outcome.py`; no outcome labels). Smaller students
  fall behind on flat channels (64x2: AWGN +8 -31%; 256x3 close), so capacity, not data, limits a single
  model there. Level with the router on seeds 24-35 and 36-47 (`runs/distill_judge.sh`). Fine-tuning a
  student on the outcomes (`train_outcome --init`) found gains (256x3: MPG 0 +13%, AWGN 0 +6%) but brought
  back the labels' trades (AWGN +4 -7%, MPG +8 -4%): next, a fine-tune anchored to the teacher.
- **cnpd (installed; CNP):** `scripts/cnp_outcome.py`, an encoder of the conditions and a query head per mode
  descriptor (ladder thresholds, constellation, code rate, band, payload, frames; `descriptor()`) and burst
  length. Five CNPs trained on the outcomes (`--labels`, seeds 1-5) of the gs sets plus
  `outcome_data_general_gt.csv` and `outcome_data_n10top_gt.csv`; ensembled with LCB 0.5; qpsk-r1/5 -1.0 on
  fast fading under -2 dB (MPP -8, the CPM floor's edge); distilled into one CNP (`runs/cnpd_round.sh`).
  Against the 512x3 student on fresh seeds 132-143: mean +1.8% over 28 cells, AWGN 0 +8.2%, MPG 0 +13.9%; the
  MPP bottom -4..-6% (about 1 SE each). Why it held where label-trained MLPs traded cells: the spread carry
  (the 2.0 Hz default read as MPD), no `--explore-faster` (it cost AWGN 0/+4), and the shared query head
  (gsx, a 512x3 MLP on the same data, lost MPP -8 -16% and MPD +8 -5%).
- **corle8 / cordf (installed; CNP):** the CPM receiver of #125/#126 (`runs/cor_round.sh`). corle8: eight CNPs
  on the outcomes (`--labels`, seeds 1-8) of the cor sets, `outcome_data_general_cor.csv` and
  `outcome_data_n10top_gt.csv`. cordf: the installed cnpd's weights (`--init`), distilled 20 epochs at lr 2e-4
  on targets that are cnpd's own for the OFDM modes and, for the CPM modes, corle8 with LCB 0.5 where the
  row's snr_est is under -6 dB, cnpd's above -4 dB, ramped between (`--teacher-cpm`, `--cpm-below -4`).
  OFDM outputs stay within 0.001 of cnpd. The descriptors carry the GF code's re-measured ladder thresholds;
  the code type needs no field (every CPM mode is GF, is_cpm says it). Against cnpd on this tree
  (`runs/cor_judge.sh`, seeds 200-215 all cells, 300-331 and 400-431 edge and floor cells): AWGN -18/-16
  +69%/+187%, MPD -15/-13 +36%/+28%, MPP -14 +20%, AWGN -14 +15%; OFDM cells within 1%; MPP -6 -2.3%
  (+-0.9, 80 seeds). cord, the same without the -4 dB edge (ensemble CPM targets everywhere), lost MPP -6
  -3.6% on two seed sets.
