#!/bin/bash
# Anchored fine-tunes of the installed 512x3 student (runs/distill_h512d3.npz):
# train_outcome --init, lr 1e-4, up to 30 epochs, outcome loss + ANCHOR x KL
# from the student's own outputs, every mode. Judged alone against the
# installed student on seeds 48-59 (selection); the pick confirmed on 60-71.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
D=runs/session_data_gt.csv,runs/session_data_gt_slow.csv,runs/session_data_gt_high.csv
D=$D,runs/session_data_gt_awgnhigh.csv,runs/session_data_gt_fastlow.csv
D=$D,runs/outcome_data_general_gt.csv,runs/outcome_data_n10top_gt.csv
for k in ${ANCHORS:-0.3 1 3}; do
  [ -s runs/outcome_aft$k.npz ] || OMP_NUM_THREADS=4 $PY -m scripts.train_outcome "$D" --energy-inputs \
    --init runs/distill_h512d3.npz --lr 1e-4 --anchor $k --epochs 30 --device cuda \
    --out runs/outcome_aft$k.npz > runs/train_aft$k.log 2>&1
  date
done
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
S0=${SEED0:-48}
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
CX=awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12
CL=mpp:-10,mpd:-8
study() {  # tag, cells, env...
  local tag=$1 cells=$2; shift 2
  local out=runs/loss_aft_${tag}_s${S0}.csv
  [ -s $out ] && return
  env "$@" $T1 $PY tools/with_native.py scripts/loss_study.py --out $out --seed0 $S0 --seeds 12 --horizon 600 \
    --jobs ${JOBS:-16} --policy shift+cpm --cells $cells > ${out%.csv}.log 2>&1
}
ALL=$C9,mpd:-6,mpp:-8,awgn:15,mpp:-6,mpd:20,mpg:15,mpp:15,mpg:-4,$CX,$CL
study base "$ALL" X=1
for k in ${ANCHORS:-0.3 1 3}; do
  study aft$k "$ALL" DATA2G_OUTCOME_MODEL=runs/outcome_aft$k.npz DATA2G_OUTCOME_GATE= DATA2G_OUTCOME_FAST= \
    DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
  date
done
for k in ${ANCHORS:-0.3 1 3}; do
  echo "== seeds $S0-$((S0 + 11)): installed student vs anchored fine-tune $k"
  $PY scripts/paired_loss.py runs/loss_aft_base_s${S0}.csv runs/loss_aft_aft${k}_s${S0}.csv
done
