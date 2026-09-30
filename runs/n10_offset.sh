#!/bin/bash
# A logit offset on n10-256l-r3/4 (tried at AWGN 20 dB below its threshold,
# 5 bursts lost): speedtrials BW500 at AWGN 20 and 25 per offset.
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
for off in ${OFFS:--1.0 -2.0}; do
  for snr in ${SNRS:-20 25}; do
    echo "== offset $off"
    DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:$off $PY scripts/cpu_profile/speedtrials.py awgn $snr \
      --bw 500 --bytes 20000 2>/dev/null | tail -2
  done
done
