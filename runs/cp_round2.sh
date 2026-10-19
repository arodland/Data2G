#!/bin/bash
# cpn: runs/cp_round.sh's regenerated sets only (no pre-pin sessions: ir_clean,
# v11, slow), plus the regenerated offline set and outcome_data_n10top.csv
# (high SNR, where pinning changes nothing). Then judged as runs/cp_judge.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
A=/home/andrew/code/Data2G/runs/outcome-data
D=runs/session_data_cp.csv,runs/session_data_cp_slow.csv,runs/session_data_cp_high.csv
D=$D,runs/session_data_cp_awgnhigh.csv,runs/session_data_cp_fastlow.csv
D=$D,runs/outcome_data_general_cp.csv,$A/outcome_data_n10top.csv
pids=() members=()
for k in 1 2 3 4 5; do
  env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m scripts.train_outcome "$D" --energy-inputs \
    --seed $k --out runs/outcome_cpn_m$k.npz > runs/train_cpn_m$k.log 2>&1 &
  pids+=($!); members+=(runs/outcome_cpn_m$k.npz)
done
for p in "${pids[@]}"; do wait "$p"; done
$PY -m scripts.train_outcome x --ensemble "$(IFS=,; echo "${members[*]}")" --out runs/outcome_cpn.npz > runs/train_cpn.log 2>&1
date
MODELS=cpn runs/cp_judge.sh
