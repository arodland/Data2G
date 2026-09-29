#!/bin/bash
# v13: v12's data + 1600 --sustained sessions (fixed SNR -8..0 dB, 600 s,
# every channel kind) steered by v12 as installed, so the model sees fast
# modes succeed on flat channels at low SNR next to failing on slow fading.
# Loss studies on both cell sets, paired against v12 and v7.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
SRC=/home/andrew/code/Data2G/.claude/worktrees/outcome-capacity/runs
export DATA2G_PEP_REF_DB=5
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4

date
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  $PY -m scripts.session_data --sustained --sessions 1600 --first 1000000 --jobs 16 --out runs/session_data_sustained.csv \
  > runs/session_data_sustained.log 2>&1
date

pids=(); members=()
for k in 1 2 3 4 5; do
  env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m scripts.train_outcome \
    $SRC/session_data_v10.csv,runs/session_data_v11.csv,runs/session_data_slow.csv,runs/session_data_sustained.csv,$SRC/outcome_data_v10.csv \
    --seed $k --out runs/outcome_v13_m$k.npz > runs/train_v13_m$k.log 2>&1 &
  pids+=($!); members+=(runs/outcome_v13_m$k.npz)
done
for p in "${pids[@]}"; do wait "$p"; done
$PY -m scripts.train_outcome x --ensemble "$(IFS=,; echo "${members[*]}")" --out runs/outcome_v13.npz > runs/train_v13.log 2>&1
date

for c in 9 12; do
  cells=C$c
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_OUTCOME_MODEL=runs/outcome_v13.npz \
    DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0 \
    $PY -m scripts.loss_study --out runs/loss_v13_c$c.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm \
    --cells ${!cells} > runs/loss_v13_c$c.log 2>&1
  date
done
echo "== v12 vs v13, 9 cells"; $PY scripts/paired_loss.py runs/loss_v12_c9.csv runs/loss_v13_c9.csv
echo "== v12 vs v13, 12 cells"; $PY scripts/paired_loss.py runs/loss_v12_c12.csv runs/loss_v13_c12.csv
echo "== v7 vs v13, 9 cells"; $PY scripts/paired_loss.py runs/loss_v7_c9.csv runs/loss_v13_c9.csv
echo "== v7 vs v13, 12 cells"; $PY scripts/paired_loss.py runs/loss_v7_c12.csv runs/loss_v13_c12.csv
