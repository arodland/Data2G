#!/bin/bash
# The link history as outcome-model inputs (predictor.N_LINK): sessions that
# record it (steered by what ships: v12 + the n10 extension; the shifter in
# Python so it keeps the history), then two arms on identical data, n10l with
# --link-inputs and n10r2 without; paired loss studies against the shipped
# model (runs/n10r_round.sh's baseline runs, same seeds) and each other.
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
for arm in "" "--slow" "--high"; do
  case $arm in "") n=2900 f=2600000 s="" ;; --slow) n=800 f=2700000 s=_slow ;; --high) n=1200 f=2800000 s=_high ;; esac
  [ -s runs/session_data_n10l$s.csv ] && continue
  $T1 DATA2G_OUTCOME_MODEL=runs/outcome_n10ext.npz DATA2G_LOGIT_OFFSETS=$OFFS $PY tools/with_native.py \
    scripts/session_data.py $arm --sessions $n --first $f --jobs 16 --out runs/session_data_n10l$s.csv \
    > runs/session_data_n10l$s.log 2>&1
  date
done

DATA=runs/session_data_n10l.csv,runs/session_data_n10l_slow.csv,runs/session_data_n10l_high.csv
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
[ -s runs/outcome_n10l.npz ] || train n10l --link-inputs &
[ -s runs/outcome_n10r2.npz ] || train n10r2 "" &
wait
date

for m in n10l n10r2; do
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
  echo "== c$c: shipped vs n10l (link inputs)"; $PY scripts/paired_loss.py runs/loss_n10r_n10ext_c$c.csv runs/loss_n10r_n10l_c$c.csv
  echo "== c$c: n10r2 (same data, no link inputs) vs n10l"; $PY scripts/paired_loss.py runs/loss_n10r_n10r2_c$c.csv runs/loss_n10r_n10l_c$c.csv
  echo "== c$c: shipped vs n10r2"; $PY scripts/paired_loss.py runs/loss_n10r_n10ext_c$c.csv runs/loss_n10r_n10r2_c$c.csv
done

echo "== BW500 speedtrials, n10l (20 kB, B/min)"
for c in "awgn 15" "awgn 20" "awgn 22" "awgn 25" "awgn 30" "mpg 25" "mpp 25"; do
  DATA2G_OUTCOME_MODEL=runs/outcome_n10l.npz DATA2G_LOGIT_OFFSETS=$OFFS \
    $PY scripts/cpu_profile/speedtrials.py $c --bw 500 --bytes 20000 2>/dev/null | tail -2
done
date
