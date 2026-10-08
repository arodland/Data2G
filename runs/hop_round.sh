#!/bin/bash
# Hopped fsk8r50 (docs/hopping-fsk.md), as run 2026-10-07: the TX sweeps,
# then paired per-draw thresholds (scripts/hop_study.py) on AWGN, the random
# ensemble and the delay x Doppler grid, 4 workers (~4 h). The rows are
# runs/hop_{awgn,ens,grid}.csv.gz; tables: scripts/hop_report.py.
set -euo pipefail
cd "$(dirname "$0")/.."
P="uv run python scripts/hop_study.py --jobs 4"

# 1. TX shape (no decoding): peaks and band edges
uv run python scripts/hop_tx_shape.py k1,k4,g2-1000 > runs/hop_tx_shape.tsv

# 2. AWGN, 200 draws: the untuned TX against tuned variants
V="k1"
for gl in 0.1 0.2 0.3; do V="$V,k1/glide=$gl/passes=6"; done
for m in k4 k4b g2-1000; do
  V="$V,$m"
  for bp in 150 350; do for gl in 0 0.1 0.2 0.3; do for dw in 1 4; do
    V="$V,$m/bp=$bp/glide=$gl/passes=6/dwell=$dw"
  done; done; done
done
V="$V,k1/bp=150/glide=0.1/passes=6,k1/bp=250/glide=0.1/passes=6,k1/bp=350/glide=0.1/passes=6"
V="$V,k1/bp=350/glide=0.2/passes=6,k1/bp=350/passes=6"
$P --out runs/hop_awgn.csv --variants "$V" --channels awgn --draws 200

# 3. the tuned variants on fading: the ensemble, then the grid
K1W=k1/bp=350/glide=0.1/passes=6
K4=k4/bp=350/glide=0.1/passes=6
G2=g2-1000/bp=350/glide=0.1/passes=6/dwell=4
$P --out runs/hop_ens.csv --variants "k1,$K1W,$K4,$G2" --channels ens --draws 4000
G=""
for t in 0.1 0.25 0.5 0.75 1 1.25 1.5 2 2.5 3 4; do for d in 0.05 0.1 0.2 0.5 1 2; do G="$G,grid:$t:$d"; done; done
$P --out runs/hop_grid.csv --variants "$K1W,$K4,$G2" --channels "${G#,}" --draws 300
gzip -f runs/hop_awgn.csv runs/hop_ens.csv runs/hop_grid.csv
