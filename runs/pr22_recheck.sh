#!/bin/bash
# PR #22 after merging master (DD on, #19's usable flag): v7 as master ships
# it (v7 model, v7's six offsets, bound 3) against v12 as this branch ships
# it, both on this code, paired, on the 17 cells of the old README table.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_PEP_REF_DB=5
C=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8,mpd:-6,mpp:-8,awgn:15,mpp:-6,mpd:20,mpg:15,mpp:15,mpg:-4
V7OFF="16qam-r1/3:-1.0,w48-16qam-r1/2:-1.0,n10-16qam-r3/4:-1.0,n10-qpsk-r3/4:-1.0,w48-qpsk-r1/3:-0.7,w48-qpsk-r2/3:-0.7"
git show origin/master:data2g/codes_data/outcome_predictor.npz > runs/outcome_v7.npz
date
env DATA2G_OUTCOME_MODEL=runs/outcome_v7.npz DATA2G_LOGIT_OFFSETS=$V7OFF DATA2G_BIAS_FIX=0 \
  $PY -m scripts.loss_study --out runs/loss_merged_v7.csv --seeds 12 --horizon 600 --jobs 12 --policy shift+cpm \
  --cells $C > runs/loss_merged_v7.log 2>&1
date
$PY -m scripts.loss_study --out runs/loss_merged_v12.csv --seeds 12 --horizon 600 --jobs 12 --policy shift+cpm \
  --cells $C > runs/loss_merged_v12.log 2>&1
date
$PY scripts/paired_loss.py runs/loss_merged_v7.csv runs/loss_merged_v12.csv
