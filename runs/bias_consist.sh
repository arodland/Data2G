#!/bin/bash
# BIAS_CONSIST trial (branch bias-consistency): off vs N=4 vs N=8, all native,
# 29 cells x 12 seeds x 600 s, PEP5; seeds 0-11 and fresh 144-155.
set -euo pipefail
cd /home/andrew/code/Data2G/.claude/worktrees/ladder-refresh
PY=/home/andrew/code/Data2G/.venv/bin/python D=/home/andrew/code/Data2G/runs/bias_consist
export DATA2G_PEP_REF_DB=5 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
ALL=$C9,mpd:-6,mpp:-8,awgn:15,mpp:-6,mpd:20,mpg:15,mpp:15,mpg:-4,awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12,mpp:-10,mpd:-8
ALL=$ALL,awgn:4:slow,awgn:15:slow,mpg:8:slow,mpp:0:slow,mpd:0:slow
for s in 0 144; do for n in 0 4 8; do
  out=$D/n${n}_s$s.csv
  [ -s $out ] || DATA2G_BIAS_CONSIST=$n $PY tools/with_native.py scripts/loss_study.py --out $out --seed0 $s --seeds 12 \
    --horizon 600 --jobs 16 --policy shift+cpm --cells $ALL > ${out%.csv}.log 2>&1
  echo "$(date +%T) n$n s$s done"
done; done
for s in 0 144; do for n in 4 8; do
  echo "== seeds $s-$((s + 11)): off vs N=$n"; $PY scripts/paired_loss.py $D/n0_s$s.csv $D/n${n}_s$s.csv
done; done
