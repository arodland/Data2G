#!/bin/bash
# The paper sweep (speed_sweep.sh, paper_points.txt: the IONOS SIM paper's
# channels, SNRs and message sizes) with the outcome gate as shipped: BW2300,
# then BW500, 12 points at a time, into runs/speedtrials_gate/.
cd "$(dirname "$0")/.."
export PY=/home/andrew/code/Data2G/.venv/bin/python JOBS=12
for bw in 2300 500; do
  scripts/cpu_profile/speed_sweep.sh "^$bw " runs/speedtrials_gate
  echo "== BW$bw done $(date)"
done
