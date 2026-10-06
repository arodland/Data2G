#!/bin/bash
# The three arms of docs/interference-plan.md, on data from this code:
#   A  sessions clean                      today's inputs
#   B  the same sessions with interference today's inputs
#   C  B's sessions                        + the noise profile (--noise-inputs)
# The two datasets share their channel seeds; only the interference differs.
# Judged on the clean cell sets and on interfered cells (in-distribution
# presets, held-out presets, noise recorded on air), 12 paired seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
REC=/home/andrew/code/Data2G/recordings
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
SD="$PY tools/with_native.py scripts/session_data.py --jobs 16"

C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
CI=""
for ch in mpg:0 mpg:8 mpp:0 mpp:-4 mpd:0 awgn:0; do
  for i in imp edge mid carriers imp_long qrm_wide rec; do CI="$CI${CI:+,}$ch:$i"; done
done

date
for set in clean intf; do
  flag=""; [ $set = intf ] && flag="--interference"
  # the installed model drives the sessions, as in the earlier rounds (with its one offset)
  DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0 $T1 $SD $flag --sessions 2900 --first 3000000 \
    --out runs/session_data_ir_$set.csv > runs/session_data_ir_$set.log 2>&1
  DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0 $T1 $SD $flag --slow --sessions 800 --first 3100000 \
    --out runs/session_data_ir_${set}_slow.csv > runs/session_data_ir_${set}_slow.log 2>&1
  DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0 $T1 $SD $flag --high --sessions 1200 --first 3200000 \
    --out runs/session_data_ir_${set}_high.csv > runs/session_data_ir_${set}_high.log 2>&1
  date
done

train() {  # name, csv list, extra flags
  local pids=() members=()
  for k in 1 2 3 4 5; do
    env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m scripts.train_outcome "$2" $3 \
      --seed $k --out runs/outcome_ir$1_m$k.npz > runs/train_ir$1_m$k.log 2>&1 &
    pids+=($!); members+=(runs/outcome_ir$1_m$k.npz)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  $PY -m scripts.train_outcome x --ensemble "$(IFS=,; echo "${members[*]}")" --out runs/outcome_ir$1.npz > runs/train_ir$1.log 2>&1
}
A=runs/session_data_ir_clean.csv,runs/session_data_ir_clean_slow.csv,runs/session_data_ir_clean_high.csv
B=runs/session_data_ir_intf.csv,runs/session_data_ir_intf_slow.csv,runs/session_data_ir_intf_high.csv
train A "$A" ""
train B "$B" ""
train C "$B" "--noise-inputs"
date

# no logit offsets for the new models (v12's patch); the same for every arm
for m in A B C; do
  for set in c9 c12 ci; do
    cells=$C9; [ $set = c12 ] && cells=$C12; [ $set = ci ] && cells=$CI
    DATA2G_LOGIT_OFFSETS= DATA2G_OUTCOME_MODEL=runs/outcome_ir$m.npz $T1 $PY tools/with_native.py scripts/loss_study.py \
      --out runs/loss_ir${m}_$set.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm --cells $cells \
      --recordings $REC > runs/loss_ir${m}_$set.log 2>&1
    date
  done
done

for set in c9 c12 ci; do
  echo "== $set: B vs C (does measuring the noise help?)"; $PY scripts/paired_loss.py runs/loss_irB_$set.csv runs/loss_irC_$set.csv
  echo "== $set: A vs B (training on interference alone)"; $PY scripts/paired_loss.py runs/loss_irA_$set.csv runs/loss_irB_$set.csv
  echo "== $set: A vs C"; $PY scripts/paired_loss.py runs/loss_irA_$set.csv runs/loss_irC_$set.csv
done
