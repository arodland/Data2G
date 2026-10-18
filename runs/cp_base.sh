#!/bin/bash
# What pinning a failed control's padding (phy.PIN_CUTS) does in sessions,
# before any retrain: the shipped model (n10pe, gated with n10qf, CPM floor,
# its offsets) on master and on this branch, same cells and seeds as
# runs/e2_judge.sh (12 seeds x 600 s, PEP5, shift+cpm, 16 jobs).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
MASTER=/home/andrew/code/Data2G  # its own native build
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
CX=awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12
CL=mpp:-10,mpd:-8

study() {  # checkout, out-tag, cells
  local root=$1 tag=$2 cells=$3
  [ -s runs/loss_cp_$tag.csv ] && return
  (cd "$root" && $T1 $PY tools/with_native.py scripts/loss_study.py --out "$OLDPWD/runs/loss_cp_$tag.csv" \
    --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm --cells $cells) > runs/loss_cp_$tag.log 2>&1
  date
}
date
for arm in master pin; do
  root=$PWD; [ $arm = master ] && root=$MASTER
  for c in 9 12 x l; do
    cells=C$c; [ $c = x ] && cells=CX; [ $c = l ] && cells=CL
    study "$root" ${arm}_c$c ${!cells}
  done
done
for c in 9 12 x l; do
  echo "== c$c: master vs pin (shipped model)"; $PY scripts/paired_loss.py runs/loss_cp_master_c$c.csv runs/loss_cp_pin_c$c.csv
done
