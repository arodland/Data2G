#!/bin/bash
# Training noise: gtp's recipe on identical data with member seeds 6-10
# (gtp2), judged with the shipped gate model as gtp_shipq was. gtp2 against
# gtp is what a retrain moves by chance alone.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
D=runs/session_data_gt.csv,runs/session_data_gt_slow.csv,runs/session_data_gt_high.csv
D=$D,runs/session_data_gt_awgnhigh.csv,runs/session_data_gt_fastlow.csv
D=$D,runs/outcome_data_general_gt.csv,runs/outcome_data_n10top_gt.csv
if [ ! -s runs/outcome_gtp2.npz ]; then
  pids=() members=()
  for k in 6 7 8 9 10; do
    env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m scripts.train_outcome "$D" --energy-inputs \
      --seed $k --out runs/outcome_gtp2_m$k.npz > runs/train_gtp2_m$k.log 2>&1 &
    pids+=($!); members+=(runs/outcome_gtp2_m$k.npz)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  $PY -m scripts.train_outcome x --ensemble "$(IFS=,; echo "${members[*]}")" --out runs/outcome_gtp2.npz > runs/train_gtp2.log 2>&1
fi
TAG=gtp2_shipq M=gtp2 G=shipq runs/gt_judge.sh
for c in 9 12 x l; do
  echo "== c$c: gtp vs gtp2 (training noise)"; $PY scripts/paired_loss.py runs/loss_gt_gtp_shipq_c$c.csv runs/loss_gt_gtp2_shipq_c$c.csv
done
