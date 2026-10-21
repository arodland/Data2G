#!/bin/bash
# The CNP ensemble (cnple5) with a lower confidence bound (DATA2G_OUTCOME_LCB=k:
# each logit less k x its members' SD) against the winner's curse it shows on
# fast fading (MPP 0: w48-qpsk-r1/3 picked 11% at 41% decoded, while
# calibrated on unsteered rows). k 0 (as judged on 108-119), 0.5, 1; fresh
# seeds 120-131, all cells, against the installed student.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
ALL=$C9,mpd:-6,mpp:-8,awgn:15,mpp:-6,mpd:20,mpg:15,mpp:15,mpg:-4,awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12,mpp:-10,mpd:-8
ALL=$ALL,awgn:4:slow,awgn:15:slow,mpg:8:slow,mpp:0:slow,mpd:0:slow
run() {  # tag, env...
  local out=runs/loss_conf_$1_s120.csv; shift
  [ -s $out ] || env "$@" $T1 $PY tools/with_native.py scripts/loss_study.py --out $out --seed0 120 --seeds 12 \
    --horizon 600 --jobs 16 --policy shift+cpm --cells $ALL > ${out%.csv}.log 2>&1
  date
}
run base X=1
for k in 0 0.5 1; do
  run cnple5_lcb$k DATA2G_OUTCOME_LCB=$k DATA2G_SPREAD_CARRY=1 DATA2G_OUTCOME_MODEL=runs/cnple5.npz DATA2G_OUTCOME_GATE= \
    DATA2G_OUTCOME_FAST= DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
done
for k in 0 0.5 1; do
  echo "== seeds 120-131: installed student vs cnple5, LCB $k"
  $PY scripts/paired_loss.py runs/loss_conf_base_s120.csv runs/loss_conf_cnple5_lcb${k}_s120.csv
done
