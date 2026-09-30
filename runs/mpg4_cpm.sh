#!/bin/bash
# MPG -4 dB: the ceiling of CPM alone (fixed fsk32r62-r1/2, 2-8 codewords
# a burst), same 12 seeds as the candidate/v7 loss studies.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_PEP_REF_DB=5
for n in 2 4 6 8; do
  $PY -m scripts.loss_study --out runs/loss_mpg4_cpm$n.csv --seeds 12 --horizon 600 --jobs 12 \
    --policy fixed:fsk32r62-r1/2:$n --cells mpg:-4 > runs/loss_mpg4_cpm$n.log 2>&1
  grep "^== " runs/loss_mpg4_cpm$n.log
done
