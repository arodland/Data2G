#!/bin/bash
# Energy as the engine measures it (NoiseProfile.span_snr_db over a burst's
# blocks, against PHY.peak_db; engine.Engine._energy): every session type
# regenerated with it, steered by what ships (v12 + n10 extension, gated, its
# offsets; the shifter in Python), then two arms on identical data: n10pe
# (--energy-inputs) and n10qe (none). Loss studies and the gate search follow
# in runs/e2_judge.sh. Dataset record: docs/outcome-training-data.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
W=/home/andrew/code/Data2G/.claude/worktrees
IR=$W/interference-sim/runs
OLD=$W/eq-floor-genie/.claude/worktrees/prune-v10/runs
export DATA2G_PEP_REF_DB=5 DATA2G_OUTCOME_MODEL=data2g/codes_data/outcome_predictor.npz
export DATA2G_OUTCOME_GATE=data2g/codes_data/outcome_predictor_gate.npz:0.5:12
export DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"

date
for arm in "" "--slow" "--high" "--high --kinds awgn" "--fastlow"; do
  case $arm in
    "") n=2900 f=5000000 s="" ;;
    --slow) n=800 f=5100000 s=_slow ;;
    --high) n=1200 f=5200000 s=_high ;;
    "--high --kinds awgn") n=400 f=5300000 s=_awgnhigh ;;
    --fastlow) n=1200 f=5400000 s=_fastlow ;;
  esac
  [ -s runs/session_data_e2$s.csv ] && continue
  $T1 $PY tools/with_native.py scripts/session_data.py $arm --sessions $n --first $f --jobs 16 \
    --out runs/session_data_e2$s.csv > runs/session_data_e2$s.log 2>&1
  date
done

DATA=runs/session_data_e2.csv,runs/session_data_e2_slow.csv,runs/session_data_e2_high.csv
DATA=$DATA,runs/session_data_e2_awgnhigh.csv,runs/session_data_e2_fastlow.csv,runs/outcome_data_general.csv
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
[ -s runs/outcome_n10pe.npz ] || train n10pe --energy-inputs &
[ -s runs/outcome_n10qe.npz ] || train n10qe "" &
wait
date
