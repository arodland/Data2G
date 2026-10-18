#!/bin/bash
# Judge runs/cp_round.sh's models on this branch against the shipped model on
# this branch (runs/cp_base.sh's pin arm, same cells and seeds): each as the
# installed model, gated with the shipped n10qf, CPM floor and offsets as
# shipped. 12 seeds x 600 s, PEP5, 16 jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5
# a DATA2G_OUTCOME_MODEL turns the gate and offsets off unless given (predictor.py)
export DATA2G_OUTCOME_GATE=data2g/codes_data/outcome_predictor_gate.npz:0.5:12
export DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
CX=awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12
CL=mpp:-10,mpd:-8

[ -n "${1:-}" ] && while kill -0 "$1" 2>/dev/null; do sleep 60; done
for m in ${MODELS:-cpe cpo}; do
  for c in 9 12 x l; do
    cells=C$c; [ $c = x ] && cells=CX; [ $c = l ] && cells=CL
    [ -s runs/loss_cp_${m}_c$c.csv ] && continue
    $T1 DATA2G_OUTCOME_MODEL=runs/outcome_$m.npz $PY tools/with_native.py scripts/loss_study.py \
      --out runs/loss_cp_${m}_c$c.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm \
      --cells ${!cells} > runs/loss_cp_${m}_c$c.log 2>&1
    date
  done
done
for m in ${MODELS:-cpe cpo}; do
  for c in 9 12 x l; do
    echo "== c$c: shipped vs $m (both pinned)"; $PY scripts/paired_loss.py runs/loss_cp_pin_c$c.csv runs/loss_cp_${m}_c$c.csv
  done
done
