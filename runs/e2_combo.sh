#!/bin/bash
# n10pe as the installed model, gated with n10qf (0.5 Hz / 12 dB, as shipped):
# against the shipped gated pair (v12 + n10 extension, n10qf), same cells.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
OFFS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
CX=awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12
CL=mpp:-10,mpd:-8
G=${G:-n10pe_qf}
for c in 9 12 x l; do
  cells=C$c; [ $c = x ] && cells=CX; [ $c = l ] && cells=CL
  [ -s runs/loss_n10r_${G}_c$c.csv ] && continue
  $T1 DATA2G_LOGIT_OFFSETS=$OFFS DATA2G_OUTCOME_MODEL=runs/outcome_n10pe.npz \
    DATA2G_OUTCOME_GATE=runs/outcome_n10qf.npz:0.5:12 ${FLOOR:+DATA2G_CPM_FLOOR=$FLOOR} $PY tools/with_native.py \
    scripts/loss_study.py --out runs/loss_n10r_${G}_c$c.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm \
    --cells ${!cells} > runs/loss_n10r_${G}_c$c.log 2>&1
done
for c in 9 12 x l; do
  echo "== c$c: shipped (gated) vs $G"; $PY scripts/paired_loss.py runs/loss_n10r_n10g_0.5_12_c$c.csv runs/loss_n10r_${G}_c$c.csv
done
