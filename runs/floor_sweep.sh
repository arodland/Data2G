#!/bin/bash
# The paper sweep (speed_sweep.sh, paper_points.txt) with the CPM floor gated on
# spread (policy.CPM_FLOOR_DEFAULT snr:-4:1): BW2300, then BW500, 12 points at a
# time, into runs/sweep-floor/.
cd "$(dirname "$0")/.."
export PY=/home/andrew/code/Data2G/.venv/bin/python JOBS=${JOBS:-12}
for bw in 2300 500; do
  scripts/cpu_profile/speed_sweep.sh "^$bw " runs/sweep-floor
  echo "== BW$bw done $(date)"
done
