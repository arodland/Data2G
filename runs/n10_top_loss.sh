#!/bin/bash
# Wide-band regression check for the n10 extension: master's v12 against v12 +
# the two n10 modes (offset as shipped), same code, paired (12 seeds x 600 s, PEP5).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_PEP_REF_DB=5
cp data2g/codes_data/outcome_predictor.npz runs/outcome_n10ext.npz
CELLS=awgn:8,awgn:15,mpg:15,mpp:15,mpd:20,mpp:8
DATA2G_OUTCOME_MODEL=runs/outcome_v12.npz DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0 \
  $PY -m scripts.loss_study --out runs/loss_n10top_v12.csv --seeds 12 --horizon 600 --jobs 8 --policy shift+cpm \
  --cells $CELLS > runs/loss_n10top_v12.log 2>&1
DATA2G_OUTCOME_MODEL=runs/outcome_n10ext.npz DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5 \
  $PY -m scripts.loss_study --out runs/loss_n10top_ext.csv --seeds 12 --horizon 600 --jobs 8 --policy shift+cpm \
  --cells $CELLS > runs/loss_n10top_ext.log 2>&1
$PY scripts/paired_loss.py runs/loss_n10top_v12.csv runs/loss_n10top_ext.csv
