#!/bin/bash
# History inputs (predictor.N_HIST: the 5-burst median spread_est the gate
# reads, the 3-burst median snr_est): every gt session kind regenerated with
# its columns (new seeds, --explore-faster), steered by the installed 512x3
# student (Python shifter). Fine-tunes of the student (train_outcome --init,
# history weights starting at 0, lr 1e-4, 30 epochs): gh1/gh3 with
# --hist-inputs at anchor 1/3; gc3 the same data without them (control).
# Reduced judge (14 cells) against the installed student on seeds 48-59.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export DATA2G_PEP_REF_DB=5 DATA2G_OUTCOME_MODEL=data2g/codes_data/outcome_predictor.npz
export DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
T1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
for arm in "" "--slow" "--high" "--high --kinds awgn" "--fastlow"; do
  case $arm in
    "") n=2900 f=9000000 s="" ;;
    --slow) n=800 f=9100000 s=_slow ;;
    --high) n=1200 f=9200000 s=_high ;;
    "--high --kinds awgn") n=400 f=9300000 s=_awgnhigh ;;
    --fastlow) n=1200 f=9400000 s=_fastlow ;;
  esac
  grep -q "^done" runs/session_data_gh$s.log 2>/dev/null && continue
  $T1 $PY tools/with_native.py scripts/session_data.py $arm --explore-faster --sessions $n --first $f --jobs 16 \
    --out runs/session_data_gh$s.csv > runs/session_data_gh$s.log 2>&1
  echo done >> runs/session_data_gh$s.log
  date
done
unset DATA2G_OUTCOME_MODEL DATA2G_LOGIT_OFFSETS
D=runs/session_data_gh.csv,runs/session_data_gh_slow.csv,runs/session_data_gh_high.csv
D=$D,runs/session_data_gh_awgnhigh.csv,runs/session_data_gh_fastlow.csv
D=$D,runs/outcome_data_general_gt.csv,runs/outcome_data_n10top_gt.csv
ft() {  # name, anchor, extra flags
  [ -s runs/outcome_$1.npz ] || OMP_NUM_THREADS=4 $PY -m scripts.train_outcome "$D" --energy-inputs $3 \
    --init runs/distill_h512d3.npz --lr 1e-4 --anchor $2 --epochs 30 --device cuda --out runs/outcome_$1.npz \
    > runs/train_$1.log 2>&1
}
ft gh1 1 --hist-inputs
ft gh3 3 --hist-inputs
ft gc3 3 ""
date
CELLS=awgn:0,awgn:4,awgn:8,awgn:15,mpg:0,mpg:4,mpg:8,mpg:12,mpp:-6,mpp:-4,mpp:0,mpd:-6,mpd:0,mpd:20
for m in gh1 gh3 gc3; do
  out=runs/loss_aft_${m}_s48.csv
  [ -s $out ] || env DATA2G_OUTCOME_MODEL=runs/outcome_$m.npz DATA2G_OUTCOME_GATE= DATA2G_OUTCOME_FAST= \
    DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5 $T1 $PY tools/with_native.py scripts/loss_study.py \
    --out $out --seed0 48 --seeds 12 --horizon 600 --jobs 16 --policy shift+cpm --cells $CELLS > ${out%.csv}.log 2>&1
  date
done
for m in gh1 gh3 gc3; do
  echo "== seeds 48-59: installed student vs $m"; $PY scripts/paired_loss.py runs/loss_aft_base_s48.csv runs/loss_aft_${m}_s48.csv
done
