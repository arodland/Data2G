#!/bin/bash
# The shipping path: the C++ gear shifter's own noise rule (on by default,
# no study variables: the native shifter and the installed model) against
# the Python shifter with the rule on and off.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
CELLS=mpg:0,mpp:0,mpg:0:carriers,mpp:0:carriers,mpg:0:edge,mpp:0:edge,mpg:8:carriers
run() {  # name, env...
  local name=$1; shift
  env "$@" $T1 $PY tools/with_native.py scripts/loss_study.py --out runs/loss_nrc_$name.csv --seeds 8 --horizon 600 \
    --jobs 16 --policy shift+cpm --cells $CELLS > runs/loss_nrc_$name.log 2>&1
}
run native PATH="$PATH"
run pyon DATA2G_NOISE_RULE=1.0
run pyoff DATA2G_NOISE_RULE=off
echo "== Python, rule off vs C++ (default)"; $PY scripts/paired_loss.py runs/loss_nrc_pyoff.csv runs/loss_nrc_native.csv
echo "== Python, rule on vs C++ (default)"; $PY scripts/paired_loss.py runs/loss_nrc_pyon.csv runs/loss_nrc_native.csv
