#!/bin/bash
# Wide-band regression check for the n10 64l/256l extension: v12 against v12
# + the two new modes' outputs, same code, paired (12 seeds x 600 s, PEP5).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DATA2G_PEP_REF_DB=5 DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0
for m in v12 n10ext; do
  DATA2G_OUTCOME_MODEL=runs/outcome_$m.npz $PY -m scripts.loss_study --out runs/loss_n10_$m.csv --seeds 12 \
    --horizon 600 --jobs 8 --policy shift+cpm --cells awgn:8,awgn:15,mpg:15,mpp:15,mpd:20,mpp:8 \
    > runs/loss_n10_$m.log 2>&1
done
$PY scripts/paired_loss.py runs/loss_n10_v12.csv runs/loss_n10_n10ext.csv
