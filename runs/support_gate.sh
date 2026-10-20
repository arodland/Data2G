#!/bin/sh
# delay_support's noise gate (and TIME_TAPS 8 on top), through the ARQ's
# receiver with DD, paired with genie_gap.sh's ofdm/dd (the ungated
# receiver, same seeds and modes). Delays drawn per trial.
set -e
cd "$(dirname "$0")/.."  # -m from the root: this checkout's data2g
PY=${PY:-python}
"$PY" -m scripts.dfts_study thresholds --out runs/support_gate.csv --jobs ${JOBS:-8} --arms ofdm/dd ofdm/t8/dd \
  --channels awgn mpg~ mpp~ mpd~ --modes n10-ack-4f n10-qpsk-r1/5 n10-qpsk-r1/3 n10-qpsk-r1/2 n10-16qam-r1/3 \
  n4-qpsk-r1/3 n4-qpsk-r1/2 ack-4f qpsk-r1/5 qpsk-r1/3 qpsk-r1/2 w48-qpsk-r1/5 w48-qpsk-r1/2 w48-16qam-r1/2
