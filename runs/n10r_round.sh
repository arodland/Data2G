#!/bin/bash
# A full outcome-model retrain with the n10 modes in the data, against what
# ships on this branch (v12 + the n10 extension, offsets as shipped):
# sessions steered by it (regular, --slow, --high; C++ core), then n10r trained
# on them + the surviving previous-version sessions (interference-sim's clean
# arm A, v12's v11 + slow) + the offline n10 supplement; 5 members averaged.
# Judged by paired loss studies (C9, C12) and BW500 speedtrials.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
W=/home/andrew/code/Data2G/.claude/worktrees
IR=$W/interference-sim/runs
OLD=$W/eq-floor-genie/.claude/worktrees/prune-v10/runs
OFFS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
cp data2g/codes_data/outcome_predictor.npz runs/outcome_n10ext.npz

date
for arm in "" "--slow" "--high"; do
  case $arm in "") n=2900 f=2300000 s="" ;; --slow) n=800 f=2400000 s=_slow ;; --high) n=1200 f=2500000 s=_high ;; esac
  [ -s runs/session_data_n10$s.csv ] && continue
  $T1 $PY tools/with_native.py scripts/session_data.py $arm --sessions $n --first $f --jobs 16 \
    --out runs/session_data_n10$s.csv > runs/session_data_n10$s.log 2>&1
  date
done

DATA=runs/session_data_n10.csv,runs/session_data_n10_slow.csv,runs/session_data_n10_high.csv
DATA=$DATA,$IR/session_data_ir_clean.csv,$IR/session_data_ir_clean_slow.csv,$IR/session_data_ir_clean_high.csv
DATA=$DATA,$OLD/session_data_v11.csv,$OLD/session_data_slow.csv,runs/outcome_data_n10top.csv
pids=(); members=()
for k in 1 2 3 4 5; do
  env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m scripts.train_outcome "$DATA" \
    --seed $k --out runs/outcome_n10r_m$k.npz > runs/train_n10r_m$k.log 2>&1 &
  pids+=($!); members+=(runs/outcome_n10r_m$k.npz)
done
for p in "${pids[@]}"; do wait "$p"; done
$PY -m scripts.train_outcome x --ensemble "$(IFS=,; echo "${members[*]}")" --out runs/outcome_n10r.npz > runs/train_n10r.log 2>&1
date

for m in n10ext n10r; do
  for c in 9 12; do
    cells=C$c
    [ -s runs/loss_n10r_${m}_c$c.csv ] && continue
    $T1 DATA2G_OUTCOME_MODEL=runs/outcome_$m.npz DATA2G_LOGIT_OFFSETS=$OFFS $PY tools/with_native.py \
      scripts/loss_study.py --out runs/loss_n10r_${m}_c$c.csv --seeds 12 --horizon 600 --jobs 16 \
      --policy shift+cpm --cells ${!cells} > runs/loss_n10r_${m}_c$c.log 2>&1
    date
  done
done
for c in 9 12; do
  echo "== c$c: shipped (v12 + n10 extension) vs n10r"
  $PY scripts/paired_loss.py runs/loss_n10r_n10ext_c$c.csv runs/loss_n10r_n10r_c$c.csv
done

echo "== BW500 speedtrials, n10r (20 kB, B/min)"
for c in "awgn 15" "awgn 20" "awgn 22" "awgn 25" "awgn 30" "mpg 25" "mpp 25"; do
  DATA2G_OUTCOME_MODEL=runs/outcome_n10r.npz DATA2G_LOGIT_OFFSETS=$OFFS \
    $PY scripts/cpu_profile/speedtrials.py $c --bw 500 --bytes 20000 2>/dev/null | tail -2
done
date
