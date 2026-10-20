#!/bin/bash
# The fast region (predictor.FAST): gtp where the peer's median spread_est >=
# 0.7 Hz and snr_est in [0, 8) dB (FAST; -2 lost MPD -4 on seeds 0-11 and 12-23), the shipped pair (n10pe, n10qf gate)
# everywhere else; edges from runs/gt_judge*.log (gtp won MPD/MPP 0, median
# snr_est 2-3 dB; lost MPD -6, MPP -8 at -3..-4.5; neutral at -4 and +8).
# Selection: seeds 0-11 against loss_gt_ship_*; confirmation: seeds 12-23,
# both arms, all cells. 600 s, PEP5, cp_judge.sh's cells.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
J=${JOBS:-16}
FAST=${FAST:-runs/outcome_gtp.npz:0.7:-2:8}
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
C12=mpd:-6,mpp:-8,awgn:15,mpp:-6,mpg:8,mpg:0,mpp:0,mpd:20,mpg:15,mpp:15,mpp:-4,mpg:-4
CX=awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12
CL=mpp:-10,mpd:-8
study() {  # tag, seed0, cells, env...
  local tag=$1 s0=$2 cells=$3; shift 3
  [ -s runs/loss_gt_$tag.csv ] && return
  env "$@" $T1 $PY tools/with_native.py scripts/loss_study.py --out runs/loss_gt_$tag.csv --seed0 $s0 \
    --seeds 12 --horizon 600 --jobs $J --policy shift+cpm --cells $cells > runs/loss_gt_$tag.log 2>&1
  date
}
ROUTED=(DATA2G_OUTCOME_MODEL=data2g/codes_data/outcome_predictor.npz
        DATA2G_OUTCOME_GATE=data2g/codes_data/outcome_predictor_gate.npz:0.5:12
        DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5 DATA2G_OUTCOME_FAST=$FAST)
T=${TAG:-fast}
for c in 9 12 x l; do
  cells=C$c; [ $c = x ] && cells=CX; [ $c = l ] && cells=CL
  study ${T}_c$c 0 "${!cells}" "${ROUTED[@]}"
done
for c in 9 12 x l; do
  echo "== c$c seeds 0-11: shipped vs $T"; $PY scripts/paired_loss.py runs/loss_gt_ship_c$c.csv runs/loss_gt_${T}_c$c.csv
done
[ -n "${SELECT_ONLY:-}" ] && exit 0
S=${CONF_SEED0:-12}  # confirmation seeds S..S+11 (24: after an edge moved on 12-23)
for c in 9 12 x l; do
  cells=C$c; [ $c = x ] && cells=CX; [ $c = l ] && cells=CL
  study ship_s${S}_c$c $S "${!cells}" X=1
  study ${T}_s${S}_c$c $S "${!cells}" "${ROUTED[@]}"
done
for c in 9 12 x l; do
  echo "== c$c seeds $S-$((S + 11)) (confirmation): shipped vs $T"
  $PY scripts/paired_loss.py runs/loss_gt_ship_s${S}_c$c.csv runs/loss_gt_${T}_s${S}_c$c.csv
done
