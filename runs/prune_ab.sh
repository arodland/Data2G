#!/bin/bash
# Pruned-set theory, A/B: the same data (runs/session_data_dd.csv +
# runs/outcome_data_dd.csv), A trained on all 48 modes, B on the 1% prune
# (DATA2G_DROP_MODES), 5 bootstrap members each (seeds 1-5), then a paired
# loss study with each arm's model and mode set.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
DATA=runs/session_data_dd.csv,runs/outcome_data_dd.csv
DROP=fsk8r50-r1/3,n4-qpsk-r1/5,polar-k96-f8,n4-qpsk-r1/2,polar-k96-f4,n4-qpsk-r2/3,w48-64l-r1/2
date
pids=()
for k in 1 2 3 4 5; do
  $PY -m scripts.train_outcome $DATA --seed $k --out runs/outcome_A_m$k.npz > runs/train_A_m$k.log 2>&1 & pids+=($!)
  DATA2G_DROP_MODES=$DROP $PY -m scripts.train_outcome $DATA --seed $k --out runs/outcome_B_m$k.npz > runs/train_B_m$k.log 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
date
$PY -m scripts.train_outcome x --ensemble runs/outcome_A_m1.npz,runs/outcome_A_m2.npz,runs/outcome_A_m3.npz,runs/outcome_A_m4.npz,runs/outcome_A_m5.npz --out runs/outcome_A.npz > runs/train_A.log 2>&1
DATA2G_DROP_MODES=$DROP $PY -m scripts.train_outcome x --ensemble runs/outcome_B_m1.npz,runs/outcome_B_m2.npz,runs/outcome_B_m3.npz,runs/outcome_B_m4.npz,runs/outcome_B_m5.npz --out runs/outcome_B.npz > runs/train_B.log 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_PEP_REF_DB=5
CELLS=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
DATA2G_OUTCOME_MODEL=runs/outcome_A.npz $PY -m scripts.loss_study --out runs/loss_prune_A.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm --cells $CELLS > runs/loss_prune_A.log 2>&1
date
DATA2G_OUTCOME_MODEL=runs/outcome_B.npz DATA2G_DROP_MODES=$DROP $PY -m scripts.loss_study --out runs/loss_prune_B.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm --cells $CELLS > runs/loss_prune_B.log 2>&1
date
$PY scripts/paired_loss.py runs/loss_prune_A.csv runs/loss_prune_B.csv
