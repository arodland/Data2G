#!/bin/bash
# Clean spread inputs: every session kind regenerated with DATA2G_SPREAD_CARRY=1
# (a defaulted spread_est carries the peer's last measured one), steered by the
# installed student, no --explore-faster (it cost AWGN 0/+4). Two models:
# gsf, the student fine-tuned (anchor 1); gsx, 512x3 from scratch on the labels.
# Both judged with the carry on: reduced fixed cells and slow cells, seeds 48-59.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5 DATA2G_SPREAD_CARRY=1 DATA2G_OUTCOME_MODEL=data2g/codes_data/outcome_predictor.npz
export DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
for arm in "" "--slow" "--high" "--high --kinds awgn" "--fastlow"; do
  case $arm in
    "") n=2900 f=10000000 s="" ;;
    --slow) n=800 f=10100000 s=_slow ;;
    --high) n=1200 f=10200000 s=_high ;;
    "--high --kinds awgn") n=400 f=10300000 s=_awgnhigh ;;
    --fastlow) n=1200 f=10400000 s=_fastlow ;;
  esac
  grep -q "^done" runs/session_data_gs$s.log 2>/dev/null && continue
  $T1 $PY tools/with_native.py scripts/session_data.py $arm --sessions $n --first $f --jobs 16 \
    --out runs/session_data_gs$s.csv > runs/session_data_gs$s.log 2>&1
  echo done >> runs/session_data_gs$s.log
  date
done
D=runs/session_data_gs.csv,runs/session_data_gs_slow.csv,runs/session_data_gs_high.csv
D=$D,runs/session_data_gs_awgnhigh.csv,runs/session_data_gs_fastlow.csv
D=$D,runs/outcome_data_general_gt.csv,runs/outcome_data_n10top_gt.csv
[ -s runs/outcome_gsf.npz ] || OMP_NUM_THREADS=4 $PY -m scripts.train_outcome "$D" --energy-inputs \
  --init runs/distill_h512d3.npz --lr 1e-4 --anchor 1 --epochs 30 --device cuda --out runs/outcome_gsf.npz > runs/train_gsf.log 2>&1
[ -s runs/outcome_gsx.npz ] || OMP_NUM_THREADS=4 $PY -m scripts.train_outcome "$D" --energy-inputs \
  --hidden 512 --depth 3 --lr 1e-3 --epochs 60 --device cuda --out runs/outcome_gsx.npz > runs/train_gsx.log 2>&1
date
FIX=awgn:0,awgn:4,awgn:8,awgn:15,mpg:0,mpg:4,mpg:8,mpg:12,mpp:-6,mpp:-4,mpp:0,mpd:-6,mpd:0,mpd:20
SLOW=awgn:4:slow,awgn:15:slow,mpg:8:slow,mpp:0:slow,mpd:0:slow
for m in gsf gsx; do
  for t in "aft:$FIX" "slow:$SLOW"; do
    out=runs/loss_${t%%:*}_${m}_s48.csv
    [ -s $out ] || env DATA2G_OUTCOME_MODEL=runs/outcome_$m.npz DATA2G_OUTCOME_GATE= DATA2G_OUTCOME_FAST= \
      $T1 $PY tools/with_native.py scripts/loss_study.py --out $out --seed0 48 --seeds 12 \
      --horizon 600 --jobs 16 --policy shift+cpm --cells ${t#*:} > ${out%.csv}.log 2>&1
  done
  date
done
for m in gsf gsx; do
  echo "== fixed: installed vs $m (carry on)"; $PY scripts/paired_loss.py runs/loss_aft_base_s48.csv runs/loss_aft_${m}_s48.csv 2>/dev/null | grep -v nan
  echo "== slow: installed vs $m (carry on)"; $PY scripts/paired_loss.py runs/loss_slow_base_s48.csv runs/loss_slow_${m}_s48.csv
done
