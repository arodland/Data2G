#!/bin/bash
# Is --explore-faster what turns outcome fine-tunes against AWGN? The student
# fine-tuned as gc3 (lr 1e-4, anchor 3, 30 epochs) on the archived cp session
# sets (no --explore-faster; the pinned pre-gate receiver; AWGN +8/+15 are past
# the noise gate's reach) plus the gt offline sets. Reduced judge, seeds 48-59.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
A=/home/andrew/code/Data2G/runs/outcome-data
D=$A/session_data_cp.csv,$A/session_data_cp_slow.csv,$A/session_data_cp_high.csv
D=$D,$A/session_data_cp_awgnhigh.csv,$A/session_data_cp_fastlow.csv
D=$D,runs/outcome_data_general_gt.csv,runs/outcome_data_n10top_gt.csv
[ -s runs/outcome_cpft3.npz ] || OMP_NUM_THREADS=4 $PY -m scripts.train_outcome "$D" --energy-inputs \
  --init runs/distill_h512d3.npz --lr 1e-4 --anchor 3 --epochs 30 --device cuda --out runs/outcome_cpft3.npz \
  > runs/train_cpft3.log 2>&1
export DATA2G_PEP_REF_DB=5
CELLS=awgn:0,awgn:4,awgn:8,awgn:15,mpg:0,mpg:4,mpg:8,mpg:12,mpp:-6,mpp:-4,mpp:0,mpd:-6,mpd:0,mpd:20
out=runs/loss_aft_cpft3_s48.csv
[ -s $out ] || env DATA2G_OUTCOME_MODEL=runs/outcome_cpft3.npz DATA2G_OUTCOME_GATE= DATA2G_OUTCOME_FAST= \
  DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 $PY tools/with_native.py scripts/loss_study.py --out $out --seed0 48 --seeds 12 --horizon 600 \
  --jobs 16 --policy shift+cpm --cells $CELLS > ${out%.csv}.log 2>&1
echo "== seeds 48-59: installed student vs cpft3"; $PY scripts/paired_loss.py runs/loss_aft_base_s48.csv $out 2>/dev/null | grep -v nan
