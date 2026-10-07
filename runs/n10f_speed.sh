#!/bin/bash
# BW500 speedtrials (20 kB) for a model of runs/n10f_round.sh: M=n10pf or n10qf.
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
for c in "awgn 15" "awgn 20" "awgn 22" "awgn 25" "awgn 30" "mpg 25" "mpp 25"; do
  DATA2G_OUTCOME_MODEL=runs/outcome_$M.npz DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5 \
    $PY scripts/cpu_profile/speedtrials.py $c --bw 500 --bytes 20000 2>/dev/null | tail -2
done
