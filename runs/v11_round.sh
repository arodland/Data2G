#!/bin/bash
# One on-policy round without v7's logit offsets: sessions driven by v10 with
# no offsets + DATA2G_BIAS_FIX (the N1 policy), then v11 (new sessions +
# v10's offline set) and v11b (v10's sessions too), 5 bootstrap members each
# (seeds 1-5), and a paired loss study of each against N1
# (runs/loss_off_N1.csv: v10, no offsets, the fix).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
SRC=/home/andrew/code/Data2G/.claude/worktrees/outcome-capacity/runs
V10=data2g/codes_data/outcome_predictor.npz
export DATA2G_PEP_REF_DB=5 DATA2G_BIAS_FIX=1

date
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_OUTCOME_MODEL=$V10 \
  $PY -m scripts.session_data --sessions 2900 --first 700000 --jobs 16 --out runs/session_data_v11.csv \
  > runs/session_data_v11.log 2>&1
date

train() {  # name, data
  local pids=() members=()
  for k in 1 2 3 4 5; do
    env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      $PY -m scripts.train_outcome "$2" --seed $k --out runs/outcome_$1_m$k.npz > runs/train_$1_m$k.log 2>&1 &
    pids+=($!); members+=(runs/outcome_$1_m$k.npz)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  $PY -m scripts.train_outcome x --ensemble "$(IFS=,; echo "${members[*]}")" --out runs/outcome_$1.npz > runs/train_$1.log 2>&1
}
train v11 runs/session_data_v11.csv,$SRC/outcome_data_v10.csv
train v11b $SRC/session_data_v10.csv,runs/session_data_v11.csv,$SRC/outcome_data_v10.csv
date

for m in v11 v11b; do
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_OUTCOME_MODEL=runs/outcome_$m.npz \
    $PY -m scripts.loss_study --out runs/loss_$m.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm \
    --cells awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8 > runs/loss_$m.log 2>&1
  date
done
for m in v11 v11b; do
  echo "== N1 (v10, no offsets, fix) vs $m"; $PY scripts/paired_loss.py runs/loss_off_N1.csv runs/loss_$m.csv
  echo "== v10 as installed vs $m"; $PY scripts/paired_loss.py runs/loss_prune_v10.csv runs/loss_$m.csv
done
