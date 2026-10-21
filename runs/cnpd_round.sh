#!/bin/bash
# One CNP distilled from the CNP ensemble (cnple5) with LCB 0.5 and the
# conditional offset qpsk-r1/5 -1.0 on fast fading under -2 dB (MPP -8: QPSK
# picked at the CPM floor's edge, 23% decoded), on the carried-spread gs rows.
# Judged alone (carry on) against the installed student on fresh seeds 132-143.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
D=runs/session_data_gs.csv,runs/session_data_gs_slow.csv,runs/session_data_gs_high.csv
D=$D,runs/session_data_gs_awgnhigh.csv,runs/session_data_gs_fastlow.csv
D=$D,runs/outcome_data_general_gt.csv,runs/outcome_data_n10top_gt.csv
COND="0.7:-2:qpsk-r1/5:-1.0"
[ -s runs/cnpd.npz ] || DATA2G_OUTCOME_LCB=0.5 OMP_NUM_THREADS=4 $PY -m scripts.cnp_outcome "$D" --teacher runs/cnple5.npz \
  --cond "$COND" --epochs 40 --out runs/cnpd.npz > runs/cnpd.log 2>&1
grep -v "No such" runs/cnpd.log | tail -3
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
ALL=$C9,mpd:-6,mpp:-8,awgn:15,mpp:-6,mpd:20,mpg:15,mpp:15,mpg:-4,awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12,mpp:-10,mpd:-8
ALL=$ALL,awgn:4:slow,awgn:15:slow,mpg:8:slow,mpp:0:slow,mpd:0:slow
run() {  # tag, env...
  local out=runs/loss_conf_$1_s132.csv; shift
  [ -s $out ] || env "$@" $T1 $PY tools/with_native.py scripts/loss_study.py --out $out --seed0 132 --seeds 12 \
    --horizon 600 --jobs 16 --policy shift+cpm --cells $ALL > ${out%.csv}.log 2>&1
  date
}
run base X=1
run cnpd DATA2G_SPREAD_CARRY=1 DATA2G_OUTCOME_MODEL=runs/cnpd.npz DATA2G_OUTCOME_GATE= DATA2G_OUTCOME_FAST= \
  DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
echo "== seeds 132-143: installed student vs the distilled CNP"
$PY scripts/paired_loss.py runs/loss_conf_base_s132.csv runs/loss_conf_cnpd_s132.csv
