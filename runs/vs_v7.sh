#!/bin/bash
# The candidate (v11b + DATA2G_BIAS_FIX + one offset) against production (v7
# with its six offsets, no fix), paired on this code: v7 on the 9 cells used
# here, then both on PR #21's 12 cells.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_PEP_REF_DB=5
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
V7OFF="16qam-r1/3:-1.0,w48-16qam-r1/2:-1.0,n10-16qam-r3/4:-1.0,n10-qpsk-r3/4:-1.0,w48-qpsk-r1/3:-0.7,w48-qpsk-r2/3:-0.7"
study() {  # out, cells, then env assignments
  local out=$1 cells=$2; shift 2
  env "$@" $PY -m scripts.loss_study --out runs/$out.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm \
    --cells $cells > runs/$out.log 2>&1
  date
}
date
study loss_v7_c9 $C9 DATA2G_OUTCOME_MODEL=runs/outcome_v7.npz DATA2G_LOGIT_OFFSETS=$V7OFF
study loss_v7_c12 $C12 DATA2G_OUTCOME_MODEL=runs/outcome_v7.npz DATA2G_LOGIT_OFFSETS=$V7OFF
study loss_cand_c12 $C12 DATA2G_OUTCOME_MODEL=runs/outcome_v11b.npz DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0 DATA2G_BIAS_FIX=1
echo "== v7 vs candidate, 9 cells"; $PY scripts/paired_loss.py runs/loss_v7_c9.csv runs/loss_v11b_off1.csv
echo "== v7 vs candidate, PR #21's 12 cells"; $PY scripts/paired_loss.py runs/loss_v7_c12.csv runs/loss_cand_c12.csv
echo "== v7 vs v10 as installed, 9 cells (check)"; $PY scripts/paired_loss.py runs/loss_v7_c9.csv runs/loss_prune_v10.csv
