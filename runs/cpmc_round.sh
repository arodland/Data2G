#!/bin/bash
# Retrain on cpm-connect-revive's sessions (CPM robust polls on bad channels,
# the proven-mode floor), then loss studies of this branch with three models:
#   v12   the installed model
#   v12r  control: retrained on the surviving v12 inputs (v11 + slow sessions)
#   cpmc  the same + this branch's sessions (2900 regular + 800 --slow)
# v10's session and offline data, part of v12's training set, are gone.
# Sessions and studies run with the C++ core (tools/with_native.py): 2.5x faster,
# same decisions and outcomes as Python on a 4-session check.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
OLD=/home/andrew/code/Data2G/.claude/worktrees/eq-floor-genie/.claude/worktrees/prune-v10/runs
export DATA2G_PEP_REF_DB=5 DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4

date
$T1 $PY tools/with_native.py scripts/session_data.py --sessions 2900 --first 2000000 --jobs 16 --out runs/session_data_cpmc.csv \
  > runs/session_data_cpmc.log 2>&1
date
$T1 $PY tools/with_native.py scripts/session_data.py --slow --sessions 800 --first 2100000 --jobs 16 --out runs/session_data_cpmc_slow.csv \
  > runs/session_data_cpmc_slow.log 2>&1
date

train() {  # name, csv list
  local pids=() members=()
  for k in 1 2 3 4 5; do
    env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m scripts.train_outcome "$2" \
      --seed $k --out runs/outcome_$1_m$k.npz > runs/train_$1_m$k.log 2>&1 &
    pids+=($!); members+=(runs/outcome_$1_m$k.npz)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  $PY -m scripts.train_outcome x --ensemble "$(IFS=,; echo "${members[*]}")" --out runs/outcome_$1.npz > runs/train_$1.log 2>&1
}
train v12r $OLD/session_data_v11.csv,$OLD/session_data_slow.csv
train cpmc $OLD/session_data_v11.csv,$OLD/session_data_slow.csv,runs/session_data_cpmc.csv,runs/session_data_cpmc_slow.csv
date

for m in v12 v12r cpmc; do
  model=runs/outcome_$m.npz
  if [ $m = v12 ]; then model=data2g/codes_data/outcome_predictor.npz; fi
  for c in 9 12; do
    cells=C$c
    $T1 DATA2G_OUTCOME_MODEL=$model $PY tools/with_native.py scripts/loss_study.py --out runs/loss_cpmc_${m}_c$c.csv --seeds 12 \
      --horizon 600 --jobs 16 --policy shift+cpm --cells ${!cells} > runs/loss_cpmc_${m}_c$c.log 2>&1
    date
  done
done
for c in 9 12; do
  echo "== c$c: v12 vs v12r (lost v10 data)"; $PY scripts/paired_loss.py runs/loss_cpmc_v12_c$c.csv runs/loss_cpmc_v12r_c$c.csv
  echo "== c$c: v12r vs cpmc (this branch's sessions)"; $PY scripts/paired_loss.py runs/loss_cpmc_v12r_c$c.csv runs/loss_cpmc_cpmc_c$c.csv
done
