#!/bin/bash
# The CPM floor (policy.CPM_FLOOR, a stopgap) on n10pe gated with n10qf: data in CPM modes only when the
# median snr_est of the peer's last 3 bursts is under X. Arms off / snr:-3 /
# snr:-1 on the shipped model (gated), low and mid SNR cells, 12 seeds x 600 s,
# PEP5, DD budget unbounded so arms run at different loads compare (paired).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5 DATA2G_OUTCOME_MODEL=runs/outcome_n10pe.npz
export DATA2G_OUTCOME_GATE=runs/outcome_n10qf.npz:0.5:12
export DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
CF=mpp:-10,mpp:-8,mpp:-6,mpp:-4,mpp:0,mpd:-8,mpd:-6,mpd:-4,mpd:0,mpg:-4,mpg:0,awgn:-4,awgn:0
for arm in ${ARMS:-off snr:-3 snr:-1}; do
  tag=${arm//:/}
  [ -s runs/loss_cfpe_$tag.csv ] && continue
  floor=""; [ $arm != off ] && floor=$arm
  $T1 DATA2G_CPM_FLOOR=$floor $PY tools/with_native.py --dd-budget inf scripts/loss_study.py --out runs/loss_cfpe_$tag.csv \
    --seeds 12 --horizon 600 --jobs 8 --policy shift+cpm --cells $CF > runs/loss_cfpe_$tag.log 2>&1
  date
done
for arm in $(for a in ${ARMS:-off snr:-3 snr:-1}; do [ $a != off ] && echo ${a//:/}; done); do
  echo "== CPM floor off vs $arm"; $PY scripts/paired_loss.py runs/loss_cfpe_off.csv runs/loss_cfpe_$arm.csv
done
