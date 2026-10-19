#!/bin/sh
# The padded delay support picked per burst (/sauto) against the shipped
# estimator, over the narrow ladder and a few 1200 Hz modes; scripts/dfts_study.py
set -e
O=runs/dfts_round2.csv
uv run --no-sync python scripts/dfts_study.py thresholds --out $O --arms ofdm ofdm/sauto --modes \
  n10-16qam-r1/2 n10-16qam-r2/3 n10-16qam-r3/4 n10-64l-r3/4 n10-256l-r3/4 n10-qpsk-r3/4 n10-16qam-r1/3 \
  n10-qpsk-r1/2 n10-qpsk-r1/3 n10-qpsk-r1/5 n4-qpsk-r1/2 n4-16qam-r1/3 qpsk-r3/4 16qam-r1/2
