#!/bin/bash
# As shipped: the installed model (the 512x3 student, float32, the C++
# shifter; no gate or fast region) on seeds 36-47, paired with the router
# (loss_gt_fast0_s36_*) and the Python float64 student (loss_gt_st_h512d3_s36_*).
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
  out=runs/loss_gt_ship512_s36_c$c.csv
  [ -s $out ] || $T1 $PY tools/with_native.py scripts/loss_study.py --out $out --seed0 36 --seeds 12 --horizon 600 \
    --jobs ${JOBS:-16} --policy shift+cpm --cells ${!cells} > ${out%.csv}.log 2>&1
done
for c in 9 12 x l; do
  echo "== c$c: router vs shipped student"; $PY scripts/paired_loss.py runs/loss_gt_fast0_s36_c$c.csv runs/loss_gt_ship512_s36_c$c.csv
  echo "== c$c: float64 student vs shipped (float32, C++)"; $PY scripts/paired_loss.py runs/loss_gt_st_h512d3_s36_c$c.csv runs/loss_gt_ship512_s36_c$c.csv
done
