#!/bin/bash
# Idea 1 for MPG -4: 800 sessions of sustained low SNR on slow fading
# (session_data --slow) under the candidate policy (v11b + DATA2G_BIAS_FIX +
# one offset), v12 trained on v10's and v11's sessions + these + v10's
# offline set, then loss studies in the candidate's configuration on both
# cell sets, paired against v7 (production) and the candidate.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
SRC=/home/andrew/code/Data2G/.claude/worktrees/outcome-capacity/runs
export DATA2G_PEP_REF_DB=5 DATA2G_BIAS_FIX=1 DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4

date
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_OUTCOME_MODEL=runs/outcome_v11b.npz \
  $PY -m scripts.session_data --slow --sessions 800 --first 900000 --jobs 16 --out runs/session_data_slow.csv \
  > runs/session_data_slow.log 2>&1
date

pids=(); members=()
for k in 1 2 3 4 5; do
  env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m scripts.train_outcome \
    $SRC/session_data_v10.csv,runs/session_data_v11.csv,runs/session_data_slow.csv,$SRC/outcome_data_v10.csv \
    --seed $k --out runs/outcome_v12_m$k.npz > runs/train_v12_m$k.log 2>&1 &
  pids+=($!); members+=(runs/outcome_v12_m$k.npz)
done
for p in "${pids[@]}"; do wait "$p"; done
$PY -m scripts.train_outcome x --ensemble "$(IFS=,; echo "${members[*]}")" --out runs/outcome_v12.npz > runs/train_v12.log 2>&1
date

for c in 9 12; do
  cells=C$c
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_OUTCOME_MODEL=runs/outcome_v12.npz \
    $PY -m scripts.loss_study --out runs/loss_v12_c$c.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm \
    --cells ${!cells} > runs/loss_v12_c$c.log 2>&1
  date
done
echo "== v7 vs v12, 9 cells"; $PY scripts/paired_loss.py runs/loss_v7_c9.csv runs/loss_v12_c9.csv
echo "== v7 vs v12, PR #21's 12 cells"; $PY scripts/paired_loss.py runs/loss_v7_c12.csv runs/loss_v12_c12.csv
echo "== candidate (v11b) vs v12, 9 cells"; $PY scripts/paired_loss.py runs/loss_v11b_off1.csv runs/loss_v12_c9.csv
echo "== candidate (v11b) vs v12, 12 cells"; $PY scripts/paired_loss.py runs/loss_cand_c12.csv runs/loss_v12_c12.csv
