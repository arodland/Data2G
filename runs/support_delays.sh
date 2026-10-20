#!/bin/sh
# support_slack on channels whose second path's delay is drawn per trial
# (dfts_study.py's "<preset>~", 0.5-1.5x the preset's): tight-only (/s4,
# the receiver before it) against the shipped receiver. No DD.
set -e
cd "$(dirname "$0")/.."  # -m from the root: this checkout's data2g, not the editable install's
PY=${PY:-python}
"$PY" -m scripts.dfts_study thresholds --out runs/support_delays.csv --jobs ${JOBS:-8} --arms ofdm/s4 ofdm \
  --channels mpg~ mpp~ mpd~ --modes n10-qpsk-r1/5 n10-qpsk-r1/2 n10-16qam-r1/2 n10-16qam-r2/3 n10-64l-r3/4
