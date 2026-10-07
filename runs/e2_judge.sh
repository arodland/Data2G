#!/bin/bash
# Judge runs/e2_round.sh's models against what ships (v12 + n10 extension,
# gated with n10qf: runs/n10g_round.sh's runs, same seeds and settings):
# n10pe (engine-measured energy inputs) and n10qe (same data, none), on C9,
# C12, the gate's edge cells (CX) and deeper fading (CL). 12 seeds x 600 s,
# PEP5, default DD budget, 16 jobs (as the shipped runs).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
OFFS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
CX=awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12
CL=mpp:-10,mpd:-8
SHIP=n10g_0.5_12

study() {  # out-tag, cells, env...
  local tag=$1 cells=$2; shift 2
  [ -s runs/loss_n10r_$tag.csv ] && return
  $T1 DATA2G_LOGIT_OFFSETS=$OFFS "$@" $PY tools/with_native.py scripts/loss_study.py --out runs/loss_n10r_$tag.csv \
    --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm --cells $cells > runs/loss_n10r_$tag.log 2>&1
  date
}
study ${SHIP}_cl $CL DATA2G_OUTCOME_MODEL=runs/outcome_n10ext.npz DATA2G_OUTCOME_GATE=runs/outcome_n10qf.npz:0.5:12
for m in n10pe n10qe; do
  for c in 9 12 x l; do
    cells=C$c; [ $c = x ] && cells=CX; [ $c = l ] && cells=CL
    study ${m}_c$c ${!cells} DATA2G_OUTCOME_MODEL=runs/outcome_$m.npz DATA2G_OUTCOME_GATE=
  done
done
for m in n10pe n10qe; do
  for c in 9 12 x l; do
    echo "== c$c: shipped (gated) vs $m"; $PY scripts/paired_loss.py runs/loss_n10r_${SHIP}_c$c.csv runs/loss_n10r_${m}_c$c.csv
  done
done
