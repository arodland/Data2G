#!/bin/sh
# What channel estimation costs the bottom and middle of the ladder: each
# mode's 10% point on the shipped estimate and on the true channel
# (dfts_study.py /genie), delays drawn per trial. No DD on either.
set -e
cd "$(dirname "$0")/.."  # -m from the root: this checkout's data2g
PY=${PY:-python}
"$PY" -m scripts.dfts_study thresholds --out runs/genie_gap.csv --jobs ${JOBS:-8} --arms ofdm ofdm/genie \
  --channels awgn mpg~ mpp~ mpd~ --modes n10-ack-4f n10-qpsk-r1/5 n10-qpsk-r1/3 n10-qpsk-r1/2 n10-16qam-r1/3 \
  n4-qpsk-r1/3 n4-qpsk-r1/2 ack-4f qpsk-r1/5 qpsk-r1/3 qpsk-r1/2 w48-qpsk-r1/5 w48-qpsk-r1/2 w48-16qam-r1/2
