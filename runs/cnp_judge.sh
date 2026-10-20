#!/bin/bash
# CNP outcome models (scripts/cnp_outcome.py, distilled from the installed
# student) alone, against the installed student: cnp_all (every mode trained)
# and cnp_hold4 (n10-16qam-r2/3, w48-64l-r2/3, fsk16r25-r1/2, qpsk-r1/3 never
# seen: zero-shot from their descriptors). Reduced fixed cells and slow
# cells, seeds 48-59; the spread carry off, as the installed baseline.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
FIX=awgn:0,awgn:4,awgn:8,awgn:15,mpg:0,mpg:4,mpg:8,mpg:12,mpp:-6,mpp:-4,mpp:0,mpd:-6,mpd:0,mpd:20
SLOW=awgn:4:slow,awgn:15:slow,mpg:8:slow,mpp:0:slow,mpd:0:slow
for m in cnp_all cnp_hold4; do
  for t in "aft:$FIX" "slow:$SLOW"; do
    out=runs/loss_${t%%:*}_${m}_s48.csv
    [ -s $out ] || env DATA2G_OUTCOME_MODEL=runs/$m.npz DATA2G_OUTCOME_GATE= DATA2G_OUTCOME_FAST= \
      DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5 $T1 $PY tools/with_native.py scripts/loss_study.py \
      --out $out --seed0 48 --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm --cells ${t#*:} > ${out%.csv}.log 2>&1
  done
  date
done
for m in cnp_all cnp_hold4; do
  echo "== fixed: installed student vs $m"; $PY scripts/paired_loss.py runs/loss_aft_base_s48.csv runs/loss_aft_${m}_s48.csv 2>/dev/null | grep -v nan
  echo "== slow: installed student vs $m"; $PY scripts/paired_loss.py runs/loss_slow_base_s48.csv runs/loss_slow_${m}_s48.csv
done
