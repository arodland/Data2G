#!/bin/bash
# Offline outcome data for the two n10 modes only (outcome_data --cands), on
# the C++ core as the host runs it (DD budget as shipped), PEP-referenced.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
DATA2G_PEP_REF_DB=5 $PY tools/with_native.py scripts/outcome_data.py --cands n10-64l-r3/4,n10-256l-r3/4 \
  --samples 8000 --first 3000000 --jobs 8 --out runs/outcome_data_n10top.csv > runs/outcome_data_n10top.log 2>&1
