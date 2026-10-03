#!/bin/bash
# v12 (installed) vs cpmc3 (cpm-connect-revive's retrain) on the data-ladder code.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
CPMC=/home/andrew/code/Data2G/.claude/worktrees/looser-reply-deadline/.claude/worktrees/cpm-connect-revive/runs
export DATA2G_PEP_REF_DB=5 DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4

date
for m in v12 cpmc3; do
  model=$CPMC/outcome_$m.npz
  if [ $m = v12 ]; then model=data2g/codes_data/outcome_predictor.npz; fi
  for c in 9 12; do
    cells=C$c
    $T1 DATA2G_OUTCOME_MODEL=$model $PY tools/with_native.py scripts/loss_study.py --out runs/loss_ladder_${m}_c$c.csv \
      --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm --cells ${!cells} > runs/loss_ladder_${m}_c$c.log 2>&1
    date
  done
done
for c in 9 12; do
  echo "== c$c: v12 vs cpmc3 (data ladder)"; $PY scripts/paired_loss.py runs/loss_ladder_v12_c$c.csv runs/loss_ladder_cpmc3_c$c.csv
done
