#!/bin/bash
# Distillation round 2: (a) h256d3 fine-tuned on the gt outcomes at a low
# learning rate (train_outcome --init); (b) h512d3 distilled 100 epochs. Both
# judged alone (gate and fast off) against the router on seeds 24-35 by
# runs/distill_judge.sh (selection; the pick is confirmed on 36-47).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
D=runs/session_data_gt.csv,runs/session_data_gt_slow.csv,runs/session_data_gt_high.csv
D=$D,runs/session_data_gt_awgnhigh.csv,runs/session_data_gt_fastlow.csv
D=$D,runs/outcome_data_general_gt.csv,runs/outcome_data_n10top_gt.csv
[ -s runs/distill_h512d3.npz ] || OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m scripts.distill_outcome "$D" \
  --hidden 512 --depth 3 --epochs 100 --out runs/distill > runs/distill_h512.log 2>&1 &
[ -s runs/distill_h256d3ft.npz ] || OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 $PY -m scripts.train_outcome "$D" \
  --energy-inputs --init runs/distill_h256d3.npz --lr 1e-4 --epochs 40 --out runs/distill_h256d3ft.npz > runs/distill_ft.log 2>&1 &
wait
STUDENTS="h256d3ft h512d3" runs/distill_judge.sh
