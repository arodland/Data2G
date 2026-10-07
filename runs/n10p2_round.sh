#!/bin/bash
# n10p's data + 400 high-SNR AWGN sessions (session_data --high --kinds awgn):
# --high has no AWGN, so with the energy inputs a 20 dB-PEP reading on AWGN
# looked like fading rows at that energy, where w48-64l-r7/12 fails (n10p
# lost 10% at AWGN +15 sending r1/2). Both arms retrained: n10p2 (energy) and
# n10q2 (none); paired loss studies as runs/n10p_round.sh.
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

date
[ -s runs/session_data_n10p_awgnhigh.csv ] || $T1 DATA2G_OUTCOME_MODEL=runs/outcome_n10ext.npz DATA2G_LOGIT_OFFSETS=$OFFS \
  $PY tools/with_native.py scripts/session_data.py --high --kinds awgn --sessions 400 --first 3700000 --jobs 16 \
  --out runs/session_data_n10p_awgnhigh.csv > runs/session_data_n10p_awgnhigh.log 2>&1
date

DATA=runs/session_data_n10p.csv,runs/session_data_n10p_slow.csv,runs/session_data_n10p_high.csv
DATA=$DATA,runs/session_data_n10p_awgnhigh.csv
DATA=$DATA,$IR/session_data_ir_clean.csv,$IR/session_data_ir_clean_slow.csv,$IR/session_data_ir_clean_high.csv
DATA=$DATA,$OLD/session_data_v11.csv,$OLD/session_data_slow.csv,runs/outcome_data_n10top.csv
train() {  # name, extra flags
  local pids=() members=()
  for k in 1 2 3 4 5; do
    env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m scripts.train_outcome "$DATA" $2 \
      --seed $k --out runs/outcome_$1_m$k.npz > runs/train_$1_m$k.log 2>&1 &
    pids+=($!); members+=(runs/outcome_$1_m$k.npz)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  $PY -m scripts.train_outcome x --ensemble "$(IFS=,; echo "${members[*]}")" --out runs/outcome_$1.npz > runs/train_$1.log 2>&1
}
[ -s runs/outcome_n10p2.npz ] || train n10p2 --energy-inputs &
[ -s runs/outcome_n10q2.npz ] || train n10q2 "" &
wait
date

for m in n10p2 n10q2; do
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
  echo "== c$c: n10q2 (same data, no energy inputs) vs n10p2"; $PY scripts/paired_loss.py runs/loss_n10r_n10q2_c$c.csv runs/loss_n10r_n10p2_c$c.csv
  echo "== c$c: shipped vs n10p2"; $PY scripts/paired_loss.py runs/loss_n10r_n10ext_c$c.csv runs/loss_n10r_n10p2_c$c.csv
done
date
