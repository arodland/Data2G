#!/bin/bash
# The ARQ_DUP pair pinned once combined: this branch against itself before
# that change (runs/cp_base.sh's pin arm), shipped model, same cells and seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
CX=awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12
CL=mpp:-10,mpd:-8
for c in 9 12 x l; do
  cells=C$c; [ $c = x ] && cells=CX; [ $c = l ] && cells=CL
  [ -s runs/loss_cp_dup_c$c.csv ] && continue
  $T1 $PY tools/with_native.py scripts/loss_study.py --out runs/loss_cp_dup_c$c.csv \
    --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm --cells ${!cells} > runs/loss_cp_dup_c$c.log 2>&1
done
for c in 9 12 x l; do
  echo "== c$c: pin vs pin + dup pair"; $PY scripts/paired_loss.py runs/loss_cp_pin_c$c.csv runs/loss_cp_dup_c$c.csv
done
