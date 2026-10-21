#!/bin/bash
# A 5-seed ensemble of the outcome-trained CNP (cnpl1-5, probabilities
# averaged), judged alone against the installed student on fresh seeds
# 108-119, all cells, spread carry on.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
D=runs/session_data_gs.csv,runs/session_data_gs_slow.csv,runs/session_data_gs_high.csv
D=$D,runs/session_data_gs_awgnhigh.csv,runs/session_data_gs_fastlow.csv
D=$D,runs/outcome_data_general_gt.csv,runs/outcome_data_n10top_gt.csv
for k in 3 4 5; do
  [ -s runs/cnpl$k.npz ] || OMP_NUM_THREADS=4 $PY -m scripts.cnp_outcome "$D" --labels --seed $k --epochs 60 \
    --out runs/cnpl$k.npz > runs/cnpl$k.log 2>&1
done
$PY -m scripts.train_outcome x --ensemble runs/cnpl1.npz,runs/cnpl2.npz,runs/cnpl3.npz,runs/cnpl4.npz,runs/cnpl5.npz \
  --out runs/cnple5.npz
date
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
ALL=$C9,mpd:-6,mpp:-8,awgn:15,mpp:-6,mpd:20,mpg:15,mpp:15,mpg:-4,awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12,mpp:-10,mpd:-8
ALL=$ALL,awgn:4:slow,awgn:15:slow,mpg:8:slow,mpp:0:slow,mpd:0:slow
run() {  # tag, env...
  local out=runs/loss_conf_$1_s108.csv; shift
  [ -s $out ] || env "$@" $T1 $PY tools/with_native.py scripts/loss_study.py --out $out --seed0 108 --seeds 12 \
    --horizon 600 --jobs 16 --policy shift+cpm --cells $ALL > ${out%.csv}.log 2>&1
  date
}
run base X=1
run cnple5 DATA2G_SPREAD_CARRY=1 DATA2G_OUTCOME_MODEL=runs/cnple5.npz DATA2G_OUTCOME_GATE= DATA2G_OUTCOME_FAST= \
  DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
echo "== seeds 108-119: installed student vs cnple5"; $PY scripts/paired_loss.py runs/loss_conf_base_s108.csv runs/loss_conf_cnple5_s108.csv
