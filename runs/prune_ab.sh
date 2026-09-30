#!/bin/bash
# Pruned-set theory, A/B, on retrain-v10: v10's own data
# (session_data_v10.csv + outcome_data_v10.csv, generated on this code), A
# trained on all 48 modes and B on a pruned set (DATA2G_DROP_MODES), 5
# bootstrap members each (seeds 1-5, same settings), then a paired loss
# study; the installed v10 (with its logit offsets) runs alongside as a
# reference.
#
#   runs/prune_ab.sh train A
#   runs/prune_ab.sh train B <drop,list>
#   runs/prune_ab.sh loss <drop,list>
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
SRC=/home/andrew/code/Data2G/.claude/worktrees/outcome-capacity/runs
DATA=$SRC/session_data_v10.csv,$SRC/outcome_data_v10.csv
CELLS=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8

train() {  # arm, drop list ("" for none)
  export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 DATA2G_DROP_MODES="$2"
  local pids=() members=()
  for k in 1 2 3 4 5; do
    $PY -m scripts.train_outcome $DATA --seed $k --out runs/outcome_$1_m$k.npz > runs/train_$1_m$k.log 2>&1 &
    pids+=($!); members+=(runs/outcome_$1_m$k.npz)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  $PY -m scripts.train_outcome x --ensemble "$(IFS=,; echo "${members[*]}")" --out runs/outcome_$1.npz > runs/train_$1.log 2>&1
}

loss() {  # out, model ("" = installed), drop list
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_PEP_REF_DB=5 \
    DATA2G_OUTCOME_MODEL="$2" DATA2G_DROP_MODES="$3" \
    $PY -m scripts.loss_study --out runs/$1.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm --cells $CELLS \
    > runs/$1.log 2>&1
}

case $1 in
  train) train "$2" "${3:-}" ;;
  loss)
    loss loss_prune_A runs/outcome_A.npz ""
    loss loss_prune_B runs/outcome_B.npz "$2"
    loss loss_prune_v10 "" ""
    $PY scripts/paired_loss.py runs/loss_prune_A.csv runs/loss_prune_B.csv
    ;;
esac
