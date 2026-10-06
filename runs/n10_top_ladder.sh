#!/bin/bash
# The two n10 modes on the full receiver (DD on, own shift tables), as e5519aa
# measured the rest: 10% then 1%, seeded --seed 0. Rows go into
# runs/ladder_dd_{10,1}pct.csv; the 10% ones into codes_data/mode_thresholds.json.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for f in 0.1 0.01; do
  $PY -m scripts.ladder_study --fail $f --seed 0 --jobs 8 --modes n10-64l-r3/4 n10-256l-r3/4 \
    --out runs/ladder_n10top_$f.csv
done
