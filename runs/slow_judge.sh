#!/bin/bash
# Slow-fade cells (loss_study 'slow': the SNR wanders both ways, 3 dB SD, 60 s
# time constant) for the installed student and the fine-tunes that traded
# fixed-SNR cells; seeds 48-59. Does drift-trained caution pay where the SNR moves?
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
CELLS=awgn:4:slow,awgn:15:slow,mpg:8:slow,mpp:0:slow,mpd:0:slow
OFF=DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
run() {  # tag, env...
  local out=runs/loss_slow_$1_s48.csv; shift
  [ -s $out ] || env "$@" $T1 $PY tools/with_native.py scripts/loss_study.py --out $out --seed0 48 --seeds 12 \
    --horizon 600 --jobs 16 --policy shift+cpm --cells $CELLS > ${out%.csv}.log 2>&1
}
run base X=1
for m in cpft3 gc3 gh3 aft3 aft0.3; do
  run $m DATA2G_OUTCOME_MODEL=runs/outcome_$m.npz DATA2G_OUTCOME_GATE= DATA2G_OUTCOME_FAST= $OFF
  date
done
for m in cpft3 gc3 gh3 aft3 aft0.3; do
  echo "== slow cells, seeds 48-59: installed student vs $m"; $PY scripts/paired_loss.py runs/loss_slow_base_s48.csv runs/loss_slow_${m}_s48.csv
done
