#!/bin/bash
# A gate between two models on existing inputs (predictor.GATE): n10qf
# (runs/n10f_round.sh, no energy inputs) when the median spread_est of the
# peer's last 5 bursts is under SPREAD Hz and snr_est under SNR dB, the
# shipped model (v12 + n10 extension) otherwise. Paired loss studies against
# the shipped model on C9, C12 and cells around the gate's edges.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
OFFS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
SPREAD=${SPREAD:-0.5} SNR=${SNR:-12}
G=n10g_${SPREAD}_${SNR}
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
CX=awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12

study() {  # out-tag, cells, extra env
  [ -s runs/loss_n10r_$1.csv ] && return
  $T1 DATA2G_OUTCOME_MODEL=runs/outcome_n10ext.npz DATA2G_LOGIT_OFFSETS=$OFFS $3 $PY tools/with_native.py \
    scripts/loss_study.py --out runs/loss_n10r_$1.csv --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm \
    --cells $2 > runs/loss_n10r_$1.log 2>&1
  date
}
date
study n10ext_cx $CX ""
for c in 9 12 x; do
  cells=C$c; [ $c = x ] && cells=CX
  study ${G}_c$c ${!cells} "DATA2G_OUTCOME_GATE=runs/outcome_n10qf.npz:$SPREAD:$SNR"
done
for c in 9 12 x; do
  echo "== c$c: shipped vs gate ($SPREAD Hz, $SNR dB)"
  $PY scripts/paired_loss.py runs/loss_n10r_n10ext_c$c.csv runs/loss_n10r_${G}_c$c.csv
done
