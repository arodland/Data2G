#!/bin/bash
# v12 on data-ladder, before (runs/ladder_compare.sh) and after the receive(head) fix.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5 DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DATA2G_OUTCOME_MODEL=data2g/codes_data/outcome_predictor.npz
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
for c in 9 12; do
  cells=C$c
  $PY tools/with_native.py scripts/loss_study.py --out runs/loss_headfix_v12_c$c.csv --seeds 12 --horizon 600 \
    --jobs 16 --policy shift+cpm --cells ${!cells} > runs/loss_headfix_v12_c$c.log 2>&1
done
for c in 9 12; do
  echo "== c$c: v12, before vs after the receive(head) fix"
  $PY scripts/paired_loss.py runs/loss_ladder_v12_c$c.csv runs/loss_headfix_v12_c$c.csv
done
