#!/bin/bash
# Round 4: D3 collapsed on interference past the mild draw's range (carriers
# 25 dB: 359 -> 3 bps). Coverage without the bias, and bounded inputs:
#   data: clean + mild (+ wander) + a third of round 1's v1 sessions (seed % 3 == 0)
#   D4  + noise inputs, clipped to their training range (train_outcome)
#   E4  without (D4's control)
# Same cells as round 3 (A, D, D3 reused).
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

date
for s in "" _slow _high; do
  $PY - runs/session_data_ir_intf$s.csv runs/session_data_ir_v1third$s.csv <<'EOF'
import csv, sys
src, dst = sys.argv[1:]
with open(src) as f, open(dst, "w", newline="") as g:
    r = csv.DictReader(f)
    w = csv.DictWriter(g, r.fieldnames)
    w.writeheader()
    w.writerows(row for row in r if int(row["seed"]) % 3 == 0)
EOF
done

DATA=runs/session_data_ir_clean.csv,runs/session_data_ir_clean_slow.csv,runs/session_data_ir_clean_high.csv
DATA=$DATA,runs/session_data_ir_mild.csv,runs/session_data_ir_mild_slow.csv,runs/session_data_ir_mild_high.csv
DATA=$DATA,runs/session_data_ir_v1third.csv,runs/session_data_ir_v1third_slow.csv,runs/session_data_ir_v1third_high.csv
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
train D4 "--noise-inputs"
train E4 ""
date

study() {  # model, set name, cells
  DATA2G_LOGIT_OFFSETS= DATA2G_OUTCOME_MODEL=runs/outcome_ir$1.npz $T1 $PY tools/with_native.py scripts/loss_study.py \
    --out runs/loss_ir$1_$2.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm --cells $3 \
    --recordings $REC > runs/loss_ir$1_$2.log 2>&1
  date
}
for m in D4 E4; do
  study $m c9 $C9; study $m c12 $C12; study $m ci $CI; study $m cw $CW
done

for set in c9 c12 ci cw; do
  echo "== $set: E4 vs D4 (does measuring the noise help?)"; $PY scripts/paired_loss.py runs/loss_irE4_$set.csv runs/loss_irD4_$set.csv
  echo "== $set: A vs D4"; $PY scripts/paired_loss.py runs/loss_irA_$set.csv runs/loss_irD4_$set.csv
  echo "== $set: A vs E4"; $PY scripts/paired_loss.py runs/loss_irA_$set.csv runs/loss_irE4_$set.csv
  for prev in D D3; do
    [ -f runs/loss_ir${prev}_$set.csv ] && { echo "== $set: $prev vs D4"; $PY scripts/paired_loss.py runs/loss_ir${prev}_$set.csv runs/loss_irD4_$set.csv; } || true
  done
done
