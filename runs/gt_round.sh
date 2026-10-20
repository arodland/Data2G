#!/bin/bash
# Retrain on the noise-gated receiver (PR #91), fresh data only: every session
# kind of runs/cp_round.sh regenerated with new seeds and --explore-faster
# (the steering model avoids the modes that gained dB since it was trained),
# steered by what ships (n10pe gated with n10qf, CPM floor, its offsets; the
# shifter in Python for the energy inputs), plus both offline sets on new
# seeds. No set from an older receiver. Two models, five bootstrap members
# each: gtp (--energy-inputs, the installed model) and gtq (none, the gate's).
# Usage: runs/gt_round.sh. Dataset record: docs/outcome-training-data.md.
# Judged by runs/gt_judge.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5 DATA2G_OUTCOME_MODEL=data2g/codes_data/outcome_predictor.npz
export DATA2G_OUTCOME_GATE=data2g/codes_data/outcome_predictor_gate.npz:0.5:12
export DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
J=${JOBS:-16}

date
for arm in "" "--slow" "--high" "--high --kinds awgn" "--fastlow"; do
  case $arm in
    "") n=2900 f=7000000 s="" ;;
    --slow) n=800 f=7100000 s=_slow ;;
    --high) n=1200 f=7200000 s=_high ;;
    "--high --kinds awgn") n=400 f=7300000 s=_awgnhigh ;;
    --fastlow) n=1200 f=7400000 s=_fastlow ;;
  esac
  [ -s runs/session_data_gt$s.csv ] && [ "$(($(wc -l < runs/session_data_gt$s.csv)))" -gt 1 ] && \
    grep -q "^done" runs/session_data_gt$s.log 2>/dev/null && continue
  $T1 $PY tools/with_native.py scripts/session_data.py $arm --explore-faster --sessions $n --first $f --jobs $J \
    --out runs/session_data_gt$s.csv > runs/session_data_gt$s.log 2>&1
  echo done >> runs/session_data_gt$s.log
  date
done
[ -s runs/outcome_data_general_gt.csv ] || {
  $T1 $PY tools/with_native.py scripts/outcome_data.py --samples 40000 --first 7500000 --jobs $J \
    --out runs/outcome_data_general_gt.csv > runs/outcome_data_general_gt.log 2>&1
  date
}
[ -s runs/outcome_data_n10top_gt.csv ] || {
  $T1 $PY tools/with_native.py scripts/outcome_data.py --cands n10-64l-r3/4,n10-256l-r3/4 --samples 8000 \
    --first 7600000 --jobs $J --out runs/outcome_data_n10top_gt.csv > runs/outcome_data_n10top_gt.log 2>&1
  date
}

D=runs/session_data_gt.csv,runs/session_data_gt_slow.csv,runs/session_data_gt_high.csv
D=$D,runs/session_data_gt_awgnhigh.csv,runs/session_data_gt_fastlow.csv
D=$D,runs/outcome_data_general_gt.csv,runs/outcome_data_n10top_gt.csv
train() {  # name, extra flags
  local pids=() members=()
  for k in 1 2 3 4 5; do
    env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m scripts.train_outcome "$D" $2 \
      --seed $k --out runs/outcome_$1_m$k.npz > runs/train_$1_m$k.log 2>&1 &
    pids+=($!); members+=(runs/outcome_$1_m$k.npz)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  $PY -m scripts.train_outcome x --ensemble "$(IFS=,; echo "${members[*]}")" --out runs/outcome_$1.npz > runs/train_$1.log 2>&1
}
[ -s runs/outcome_gtp.npz ] || train gtp --energy-inputs &
[ -s runs/outcome_gtq.npz ] || train gtq "" &
wait
date
