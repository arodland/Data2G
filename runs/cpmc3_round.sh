#!/bin/bash
# cpmc's data + high-SNR fading sessions (session_data --high), standing in
# for v10's lost coverage (cpmc lost 13-17% at MPD +20 and MPP +15 against
# v12; cpmc2's sustained sessions made that worse). Baselines: cpmc_round.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
OLD=/home/andrew/code/Data2G/.claude/worktrees/eq-floor-genie/.claude/worktrees/prune-v10/runs
export DATA2G_PEP_REF_DB=5 DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
DATA=$OLD/session_data_v11.csv,$OLD/session_data_slow.csv,runs/session_data_cpmc.csv,runs/session_data_cpmc_slow.csv,runs/session_data_cpmc_high.csv

date
$T1 $PY tools/with_native.py scripts/session_data.py --high --sessions 1200 --first 2200000 --jobs 16 \
  --out runs/session_data_cpmc_high.csv > runs/session_data_cpmc_high.log 2>&1
date
pids=(); members=()
for k in 1 2 3 4 5; do
  env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m scripts.train_outcome "$DATA" \
    --seed $k --out runs/outcome_cpmc3_m$k.npz > runs/train_cpmc3_m$k.log 2>&1 &
  pids+=($!); members+=(runs/outcome_cpmc3_m$k.npz)
done
for p in "${pids[@]}"; do wait "$p"; done
$PY -m scripts.train_outcome x --ensemble "$(IFS=,; echo "${members[*]}")" --out runs/outcome_cpmc3.npz > runs/train_cpmc3.log 2>&1
date
for c in 9 12; do
  cells=C$c
  $T1 DATA2G_OUTCOME_MODEL=runs/outcome_cpmc3.npz $PY tools/with_native.py scripts/loss_study.py \
    --out runs/loss_cpmc_cpmc3_c$c.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm --cells ${!cells} \
    > runs/loss_cpmc_cpmc3_c$c.log 2>&1
  date
done
for c in 9 12; do
  echo "== c$c: v12 vs cpmc3"; $PY scripts/paired_loss.py runs/loss_cpmc_v12_c$c.csv runs/loss_cpmc_cpmc3_c$c.csv
  echo "== c$c: cpmc vs cpmc3 (high-SNR fading sessions)"; $PY scripts/paired_loss.py runs/loss_cpmc_cpmc_c$c.csv runs/loss_cpmc_cpmc3_c$c.csv
done
