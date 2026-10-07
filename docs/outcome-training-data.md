# Outcome-model training data

Every dataset in the current training mix: how it was made, what steered it,
and why it is there, so a future retrain can rebuild or extend the mix.

**Archive.** `runs/` is gitignored and worktrees get deleted (v10's 2900
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

Older sets carry no link-history or energy columns: their rows read as "no
history" (the inputs' flag 0).

## Offline sets

`outcome_data.py`: one sample is a measured history plus 4 candidate bursts
sent at the same moment; one row per candidate.

| file | generator | steering | code | why |
|---|---|---|---|---|
| `outcome_data_n10top.csv` | `with_native outcome_data --cands n10-64l-r3/4,n10-256l-r3/4 --samples 8000 --first 3000000` (`runs/n10_top_data.sh`) | none (candidates fixed) | 2613567 | the two new 500 Hz modes, 0..36 dB |
| `outcome_data_general.csv` | `with_native outcome_data --samples 40000 --first 4000000` (`runs/n10p3_round.sh`) | none | 046f8be | stands in for v10's lost offline set: retrains without one lost 4-10% on MPD |

## Mixes

- **Shipped (v12 + n10 extension):** v12 as trained (v10's sessions and offline set, v11, slow;
  lost) plus output units for the n10 modes from `outcome_data_n10top.csv`
  (`train_outcome --extend`).
- **n10pf / n10qf** (`runs/n10f_round.sh`): every file above; n10pf with `--energy-inputs`.
  Five bootstrap members, seeds 1-5, averaged.
