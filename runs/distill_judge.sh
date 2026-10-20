#!/bin/bash
# Distilled single models (scripts/distill_outcome.py, runs/distill_h*.npz)
# alone, gate and fast region off, against the router (runs/fast_judge.sh's
# fast0 arm), on seeds S0..S0+11 (SEED0, default 24: fast_judge's confirmation
# seeds; 36: the student pick's own confirmation, the router re-run there too).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
J=${JOBS:-16}
S0=${SEED0:-24}
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
CX=awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12
CL=mpp:-10,mpd:-8
for s in ${STUDENTS:-h64d2 h256d3}; do
  for c in 9 12 x l; do
    cells=C$c; [ $c = x ] && cells=CX; [ $c = l ] && cells=CL
    out=runs/loss_gt_st_${s}_s${S0}_c$c.csv
    [ -s $out ] && continue
    env DATA2G_OUTCOME_MODEL=runs/distill_$s.npz DATA2G_OUTCOME_GATE= DATA2G_OUTCOME_FAST= \
      DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5 $T1 $PY tools/with_native.py scripts/loss_study.py \
      --out $out --seed0 $S0 --seeds 12 --horizon 600 --jobs $J --policy shift+cpm --cells ${!cells} > ${out%.csv}.log 2>&1
    date
  done
done
ROUTED=(DATA2G_OUTCOME_MODEL=data2g/codes_data/outcome_predictor.npz
        DATA2G_OUTCOME_GATE=data2g/codes_data/outcome_predictor_gate.npz:0.5:12
        DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5 DATA2G_OUTCOME_FAST=runs/outcome_gtp.npz:0.7:0:8)
for c in 9 12 x l; do  # the router on these seeds, if not run yet
  cells=C$c; [ $c = x ] && cells=CX; [ $c = l ] && cells=CL
  out=runs/loss_gt_fast0_s${S0}_c$c.csv
  [ -s $out ] && continue
  env "${ROUTED[@]}" $T1 $PY tools/with_native.py scripts/loss_study.py --out $out --seed0 $S0 --seeds 12 \
    --horizon 600 --jobs $J --policy shift+cpm --cells ${!cells} > ${out%.csv}.log 2>&1
done
for s in ${STUDENTS:-h64d2 h256d3}; do
  for c in 9 12 x l; do
    echo "== c$c seeds $S0-$((S0 + 11)): router vs student $s"
    $PY scripts/paired_loss.py runs/loss_gt_fast0_s${S0}_c$c.csv runs/loss_gt_st_${s}_s${S0}_c$c.csv
  done
done
