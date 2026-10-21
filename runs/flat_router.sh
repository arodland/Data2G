#!/bin/bash
# The flat region: gsx+gsx2 (ensemble, carried-spread data) where the peer's
# median spread_est < 0.08 Hz and snr_est < 12 dB (the gate mechanism), the
# installed student elsewhere; DATA2G_SPREAD_CARRY on. The edge was chosen on
# seeds 60-71 (gsx won AWGN -4..+8, lost slow fading and the MPP bottom):
# judged on fresh seeds 72-83, all cells, against the installed student.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
ALL=$C9,mpd:-6,mpp:-8,awgn:15,mpp:-6,mpd:20,mpg:15,mpp:15,mpg:-4,awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12,mpp:-10,mpd:-8
ALL=$ALL,awgn:4:slow,awgn:15:slow,mpg:8:slow,mpp:0:slow,mpd:0:slow
S0=${SEED0:-72}
GATE=${GATE:-runs/outcome_gsxe.npz:0.08:12}
TAG=${TAG:-flat}
run() {  # tag, env...
  local out=runs/loss_conf_$1_s$S0.csv; shift
  [ -s $out ] || env "$@" $T1 $PY tools/with_native.py scripts/loss_study.py --out $out --seed0 $S0 --seeds 12 \
    --horizon 600 --jobs 16 --policy shift+cpm --cells $ALL > ${out%.csv}.log 2>&1
  date
}
run base X=1
run $TAG DATA2G_SPREAD_CARRY=1 DATA2G_OUTCOME_MODEL=data2g/codes_data/outcome_predictor.npz \
  DATA2G_OUTCOME_GATE=$GATE DATA2G_OUTCOME_FAST= DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
echo "== seeds $S0-$((S0 + 11)): installed student vs the flat router"
$PY scripts/paired_loss.py runs/loss_conf_base_s$S0.csv runs/loss_conf_${TAG}_s$S0.csv
