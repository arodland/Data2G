#!/bin/bash
# DATA2G_SPREAD_CARRY=1 with the installed student (no retrain): a defaulted
# spread_est (short or weak burst) replaced by the peer's last measured one.
# Reduced fixed cells and the slow cells, seeds 48-59, against the installed
# shifter (loss_aft_base_s48, loss_slow_base_s48).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
CARRY=(DATA2G_SPREAD_CARRY=1 DATA2G_OUTCOME_MODEL=data2g/codes_data/outcome_predictor.npz
       DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5)
FIX=awgn:0,awgn:4,awgn:8,awgn:15,mpg:0,mpg:4,mpg:8,mpg:12,mpp:-6,mpp:-4,mpp:0,mpd:-6,mpd:0,mpd:20
SLOW=awgn:4:slow,awgn:15:slow,mpg:8:slow,mpp:0:slow,mpd:0:slow
for t in "aft:$FIX" "slow:$SLOW"; do
  out=runs/loss_${t%%:*}_carry_s48.csv
  [ -s $out ] || env "${CARRY[@]}" $T1 $PY tools/with_native.py scripts/loss_study.py --out $out --seed0 48 --seeds 12 \
    --horizon 600 --jobs 16 --policy shift+cpm --cells ${t#*:} > ${out%.csv}.log 2>&1
done
echo "== fixed cells: installed vs spread carry"; $PY scripts/paired_loss.py runs/loss_aft_base_s48.csv runs/loss_aft_carry_s48.csv 2>/dev/null | grep -v nan
echo "== slow cells: installed vs spread carry"; $PY scripts/paired_loss.py runs/loss_slow_base_s48.csv runs/loss_slow_carry_s48.csv
