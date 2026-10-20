#!/bin/bash
# Judge runs/gt_round.sh's pair (gtp installed, gtq as the gate's model)
# against the shipped pair, both on this branch's receiver: the CPM floor and
# offsets as shipped. 12 seeds x 600 s, PEP5. cp_judge.sh's cells.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
J=${JOBS:-16}
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
CX=awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12
CL=mpp:-10,mpd:-8
study() {  # tag, cells, env...
  local tag=$1 cells=$2; shift 2
  [ -s runs/loss_gt_$tag.csv ] && return
  env "$@" $T1 $PY tools/with_native.py scripts/loss_study.py --out runs/loss_gt_$tag.csv \
    --seeds 12 --horizon 600 --jobs $J --policy shift+cpm --cells $cells > runs/loss_gt_$tag.log 2>&1
  date
}
NEW=(DATA2G_OUTCOME_MODEL=runs/outcome_${M:-gtp}.npz DATA2G_OUTCOME_GATE=runs/outcome_${G:-gtq}.npz:0.5:12
     DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5)
for c in 9 12 x l; do
  cells=C$c; [ $c = x ] && cells=CX; [ $c = l ] && cells=CL
  study ship_c$c "${!cells}" X=1
  study ${M:-gtp}_c$c "${!cells}" "${NEW[@]}"
done
for c in 9 12 x l; do
  echo "== c$c: shipped vs ${M:-gtp}"; $PY scripts/paired_loss.py runs/loss_gt_ship_c$c.csv runs/loss_gt_${M:-gtp}_c$c.csv
done
