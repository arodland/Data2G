#!/bin/sh
# DFT-spread vs OFDM, shipped vs padded (slack 20) support; scripts/dfts_study.py
set -e
O=runs/dfts_round1.csv
run() { uv run --no-sync python scripts/dfts_study.py thresholds --out $O "$@"; }
run --modes n10-16qam-r2/3 --arms ofdm ofdm/s20 dfts/s20 dfts@0/s20 dfts@4/s20
run --modes n10-16qam-r3/4 --arms ofdm ofdm/s20 dfts/s20 dfts@1/s20 dfts@5/s20
run --modes n10-16qam-r1/2 --arms ofdm ofdm/s20 dfts/s20 dfts@2/s20
run --modes n10-qpsk-r1/2 n10-qpsk-r1/5 --arms ofdm ofdm/s20 dfts/s20
run --modes n10-64l-r3/4 --arms ofdm ofdm/s20 dfts/s20 dfts@4/s20
run --modes w48-16qam-r3/4 --arms ofdm dfts dfts@1 dfts@5
run --modes w48-64l-r2/3 --arms ofdm dfts dfts@4 dfts@8
