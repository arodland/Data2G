#!/bin/bash
# Polar payload reversal + known padding (codes.polar_order, phy.POLAR_PIN_CUTS):
# this branch against PR #84's head (runs/cp_dup.sh's arm in the ctl-pin
# worktree), shipped model, same cells and seeds; then speed trials at the
# low-SNR points (CPM and the polar modes) on both.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
BASE=/home/andrew/code/Data2G/.claude/worktrees/ctl-pin
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
CX=awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12
CL=mpp:-10,mpd:-8
for c in 9 12 x l; do
  cells=C$c; [ $c = x ] && cells=CX; [ $c = l ] && cells=CL
  [ -s runs/loss_pp_c$c.csv ] && continue
  $T1 $PY tools/with_native.py scripts/loss_study.py --out runs/loss_pp_c$c.csv \
    --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm --cells ${!cells} > runs/loss_pp_c$c.log 2>&1
done
for c in 9 12 x l; do
  echo "== c$c: PR #84 vs polar"; $PY scripts/paired_loss.py $BASE/runs/loss_cp_dup_c$c.csv runs/loss_pp_c$c.csv
done
date
LOW='^(2300 \w+ (0|-5)|500 \w+ (5|0|-5)) '
PY=$PY JOBS=8 scripts/cpu_profile/speed_sweep.sh "$LOW" $PWD/runs/speedtrials_pp > runs/speed_pp.log 2>&1
PY=$PY JOBS=8 $BASE/scripts/cpu_profile/speed_sweep.sh "$LOW" $PWD/runs/speedtrials_base > runs/speed_base.log 2>&1
date
