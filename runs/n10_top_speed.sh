#!/bin/bash
# BW500 speedtrials (20 kB, 5 successes or 10 failures): ARM=v12 (master's
# model, Python), new (the n10 extension, Python), native (the extension, C++ core).
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
for c in "awgn 15" "awgn 20" "awgn 22" "awgn 25" "awgn 30" "mpg 25" "mpp 25"; do
  case $ARM in
    v12) DATA2G_OUTCOME_MODEL=runs/outcome_v12.npz DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0 \
           $PY scripts/cpu_profile/speedtrials.py $c --bw 500 --bytes 20000 ;;
    new) $PY scripts/cpu_profile/speedtrials.py $c --bw 500 --bytes 20000 ;;
    native) $PY tools/with_native.py scripts/cpu_profile/speedtrials.py $c --bw 500 --bytes 20000 ;;
  esac 2>/dev/null | tail -2
done
