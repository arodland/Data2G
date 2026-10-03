#!/bin/bash
# Round 3: D and E again, on the clean data plus interfered data from the
# milder draw (interference.DRAWS["mild"], by judgement: the recordings are too
# few to fit), every station's floor wandering 1 dB (clean receivers too).
#   D3  + noise inputs        E3  without (D3's control)
# Judged on the same 63 cells as rounds 1-2 (A's results reused), and on
# clean cells with a wandering floor (A evaluated there too).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
REC=/home/andrew/code/Data2G/recordings
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
SD="$PY tools/with_native.py scripts/session_data.py --jobs 16 --interference mild --wander 1.0"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
CW=mpg:0:wander,mpp:0:wander,mpd:0:wander,mpp:-4:wander,awgn:0:wander
CI=""
for ch in mpg:0 mpg:8 mpp:0 mpp:-4 mpd:0 awgn:0; do
  for i in imp edge mid carriers imp_long qrm_wide rec; do CI="$CI${CI:+,}$ch:$i"; done
done

date
export DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0  # the installed model drives the sessions, as before
$T1 $SD --sessions 2900 --first 3000000 --out runs/session_data_ir_mild.csv > runs/session_data_ir_mild.log 2>&1
$T1 $SD --slow --sessions 800 --first 3100000 --out runs/session_data_ir_mild_slow.csv > runs/session_data_ir_mild_slow.log 2>&1
$T1 $SD --high --sessions 1200 --first 3200000 --out runs/session_data_ir_mild_high.csv > runs/session_data_ir_mild_high.log 2>&1
unset DATA2G_LOGIT_OFFSETS
date

DATA=runs/session_data_ir_clean.csv,runs/session_data_ir_clean_slow.csv,runs/session_data_ir_clean_high.csv
DATA=$DATA,runs/session_data_ir_mild.csv,runs/session_data_ir_mild_slow.csv,runs/session_data_ir_mild_high.csv
train() {  # name, extra flags
  local pids=() members=()
  for k in 1 2 3 4 5; do
    env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m scripts.train_outcome "$DATA" $2 \
      --seed $k --out runs/outcome_ir$1_m$k.npz > runs/train_ir$1_m$k.log 2>&1 &
    pids+=($!); members+=(runs/outcome_ir$1_m$k.npz)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  $PY -m scripts.train_outcome x --ensemble "$(IFS=,; echo "${members[*]}")" --out runs/outcome_ir$1.npz > runs/train_ir$1.log 2>&1
}
train D3 "--noise-inputs"
train E3 ""
date

study() {  # model, set name, cells
  DATA2G_LOGIT_OFFSETS= DATA2G_OUTCOME_MODEL=runs/outcome_ir$1.npz $T1 $PY tools/with_native.py scripts/loss_study.py \
    --out runs/loss_ir$1_$2.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm --cells $3 \
    --recordings $REC > runs/loss_ir$1_$2.log 2>&1
  date
}
study A cw $CW
for m in D3 E3; do
  study $m c9 $C9; study $m c12 $C12; study $m ci $CI; study $m cw $CW
done

for set in c9 c12 ci cw; do
  echo "== $set: E3 vs D3 (does measuring the noise help?)"; $PY scripts/paired_loss.py runs/loss_irE3_$set.csv runs/loss_irD3_$set.csv
  echo "== $set: A vs D3"; $PY scripts/paired_loss.py runs/loss_irA_$set.csv runs/loss_irD3_$set.csv
  echo "== $set: A vs E3"; $PY scripts/paired_loss.py runs/loss_irA_$set.csv runs/loss_irE3_$set.csv
  echo "== $set: D vs D3 (round 2 -> 3)"; [ -f runs/loss_irD_$set.csv ] && $PY scripts/paired_loss.py runs/loss_irD_$set.csv runs/loss_irD3_$set.csv || true
done
