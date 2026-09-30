#!/bin/bash
# v11b + DATA2G_BIAS_FIX with a single logit offset (w48-16qam-r1/2 -1.0,
# v7's value) instead of v7's six; paired against v11b and v10 as installed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_PEP_REF_DB=5 DATA2G_BIAS_FIX=1
export DATA2G_OUTCOME_MODEL=runs/outcome_v11b.npz DATA2G_LOGIT_OFFSETS="w48-16qam-r1/2:-1.0"
date
$PY -m scripts.loss_study --out runs/loss_v11b_off1.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm \
  --cells awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8 > runs/loss_v11b_off1.log 2>&1
date
echo "== v11b vs v11b + offset"; $PY scripts/paired_loss.py runs/loss_v11b.csv runs/loss_v11b_off1.csv
echo "== v10 as installed vs v11b + offset"; $PY scripts/paired_loss.py runs/loss_prune_v10.csv runs/loss_v11b_off1.csv
