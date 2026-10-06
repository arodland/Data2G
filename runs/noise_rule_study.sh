#!/bin/bash
# Option 1 of docs/interference-plan.md: the clean-trained model (A) plus an
# explicit rule on the noise profile (DATA2G_NOISE_RULE: each candidate
# predicted as if its SNR were lower by predictor.noise_shift_db), against A
# alone, rerun here so every arm runs the same code. Tail weights 0.5 and 1.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
REC=/home/andrew/code/Data2G/recordings
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
CW=mpg:0:wander,mpp:0:wander,mpd:0:wander,mpp:-4:wander,awgn:0:wander
CI=""
for ch in mpg:0 mpg:8 mpp:0 mpp:-4 mpd:0 awgn:0; do
  for i in imp edge mid carriers imp_long qrm_wide rec; do CI="$CI${CI:+,}$ch:$i"; done
done

study() {  # arm name, rule weight ("" off), set name, cells
  DATA2G_NOISE_RULE=$2 DATA2G_LOGIT_OFFSETS= DATA2G_OUTCOME_MODEL=runs/outcome_irA.npz $T1 $PY tools/with_native.py \
    scripts/loss_study.py --out runs/loss_rule$1_$3.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm \
    --cells $4 --recordings $REC > runs/loss_rule$1_$3.log 2>&1
  date
}
date
for arm in "A0:" "R05:0.5" "R10:1.0"; do
  name=${arm%%:*}; w=${arm#*:}
  study $name "$w" c9 $C9; study $name "$w" c12 $C12; study $name "$w" ci $CI; study $name "$w" cw $CW
done
for set in c9 c12 ci cw; do
  for r in R05 R10; do
    echo "== $set: A vs A + rule ($r)"; $PY scripts/paired_loss.py runs/loss_ruleA0_$set.csv runs/loss_rule${r}_$set.csv
  done
done
