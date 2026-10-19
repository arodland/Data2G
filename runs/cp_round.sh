#!/bin/bash
# Retrain for pinned control padding (phy.PIN_CUTS): runs/e2_round.sh's recipe
# on this branch. Every e2 session type regenerated with new seeds, steered by
# what ships (n10pe gated with n10qf, CPM floor, its offsets; the shifter in
# Python for the energy inputs); the general offline set regenerated with its
# old seeds, so its labels differ from outcome_data_general.csv only by the
# decoder. Two arms, five bootstrap members each, --energy-inputs:
#   cpe  the e2 mix with every regenerated set swapped in
#   cpo  the same, with the old offline set (isolates the offline relabel)
# Usage: runs/cp_round.sh [pid to wait for]. Dataset record:
# docs/outcome-training-data.md. Judged by runs/cp_judge.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
A=/home/andrew/code/Data2G/runs/outcome-data
export DATA2G_PEP_REF_DB=5 DATA2G_OUTCOME_MODEL=data2g/codes_data/outcome_predictor.npz
export DATA2G_OUTCOME_GATE=data2g/codes_data/outcome_predictor_gate.npz:0.5:12
export DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"

[ -n "${1:-}" ] && while kill -0 "$1" 2>/dev/null; do sleep 60; done
date
for arm in "" "--slow" "--high" "--high --kinds awgn" "--fastlow"; do
  case $arm in
    "") n=2900 f=6000000 s="" ;;
    --slow) n=800 f=6100000 s=_slow ;;
    --high) n=1200 f=6200000 s=_high ;;
    "--high --kinds awgn") n=400 f=6300000 s=_awgnhigh ;;
    --fastlow) n=1200 f=6400000 s=_fastlow ;;
  esac
  [ -s runs/session_data_cp$s.csv ] && continue
  $T1 $PY tools/with_native.py scripts/session_data.py $arm --sessions $n --first $f --jobs 16 \
    --out runs/session_data_cp$s.csv > runs/session_data_cp$s.log 2>&1
  date
done
[ -s runs/outcome_data_general_cp.csv ] || {
  $T1 $PY tools/with_native.py scripts/outcome_data.py --samples 40000 --first 4000000 --jobs 16 \
    --out runs/outcome_data_general_cp.csv > runs/outcome_data_general_cp.log 2>&1
  date
}

S=runs/session_data_cp.csv,runs/session_data_cp_slow.csv,runs/session_data_cp_high.csv
S=$S,runs/session_data_cp_awgnhigh.csv,runs/session_data_cp_fastlow.csv
OLD=$A/session_data_ir_clean.csv,$A/session_data_ir_clean_slow.csv,$A/session_data_ir_clean_high.csv
OLD=$OLD,$A/session_data_v11.csv,$A/session_data_slow.csv,$A/outcome_data_n10top.csv
train() {  # name, data
  local pids=() members=()
  for k in 1 2 3 4 5; do
    env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m scripts.train_outcome "$2" --energy-inputs \
      --seed $k --out runs/outcome_$1_m$k.npz > runs/train_$1_m$k.log 2>&1 &
    pids+=($!); members+=(runs/outcome_$1_m$k.npz)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  $PY -m scripts.train_outcome x --ensemble "$(IFS=,; echo "${members[*]}")" --out runs/outcome_$1.npz > runs/train_$1.log 2>&1
}
[ -s runs/outcome_cpe.npz ] || train cpe "$S,runs/outcome_data_general_cp.csv,$OLD" &
[ -s runs/outcome_cpo.npz ] || train cpo "$S,$A/outcome_data_general.csv,$OLD" &
wait
date
