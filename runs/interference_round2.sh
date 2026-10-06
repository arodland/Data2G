#!/bin/bash
# After interference_round.sh: C (interfered data + noise inputs) lost to A
# (clean data) on most cells, interfered ones too; B (interfered data, no
# noise inputs) lost to both. Is it the data mix (~65% of rows from an
# interfered receiver)? Two more arms on A's and B's data together:
#   D  + noise inputs        E  without (D's control)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
REC=/home/andrew/code/Data2G/recordings
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
CI=""
for ch in mpg:0 mpg:8 mpp:0 mpp:-4 mpd:0 awgn:0; do
  for i in imp edge mid carriers imp_long qrm_wide rec; do CI="$CI${CI:+,}$ch:$i"; done
done
AB=runs/session_data_ir_clean.csv,runs/session_data_ir_clean_slow.csv,runs/session_data_ir_clean_high.csv
AB=$AB,runs/session_data_ir_intf.csv,runs/session_data_ir_intf_slow.csv,runs/session_data_ir_intf_high.csv

train() {  # name, extra flags
  local pids=() members=()
  for k in 1 2 3 4 5; do
    env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m scripts.train_outcome "$AB" $2 \
      --seed $k --out runs/outcome_ir$1_m$k.npz > runs/train_ir$1_m$k.log 2>&1 &
    pids+=($!); members+=(runs/outcome_ir$1_m$k.npz)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  $PY -m scripts.train_outcome x --ensemble "$(IFS=,; echo "${members[*]}")" --out runs/outcome_ir$1.npz > runs/train_ir$1.log 2>&1
}
date
train D "--noise-inputs"
train E ""
date
for m in D E; do
  for set in c9 c12 ci; do
    cells=$C9; [ $set = c12 ] && cells=$C12; [ $set = ci ] && cells=$CI
    DATA2G_LOGIT_OFFSETS= DATA2G_OUTCOME_MODEL=runs/outcome_ir$m.npz $T1 $PY tools/with_native.py scripts/loss_study.py \
      --out runs/loss_ir${m}_$set.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm --cells $cells \
      --recordings $REC > runs/loss_ir${m}_$set.log 2>&1
    date
  done
done
for set in c9 c12 ci; do
  echo "== $set: E vs D (does measuring the noise help, on mixed data?)"; $PY scripts/paired_loss.py runs/loss_irE_$set.csv runs/loss_irD_$set.csv
  echo "== $set: A vs D (mixed data + noise inputs against clean-trained)"; $PY scripts/paired_loss.py runs/loss_irA_$set.csv runs/loss_irD_$set.csv
  echo "== $set: A vs E"; $PY scripts/paired_loss.py runs/loss_irA_$set.csv runs/loss_irE_$set.csv
done
