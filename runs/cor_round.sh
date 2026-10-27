#!/bin/bash
# The outcome model on the CPM receiver of #125 + #126 (acquisition gates, GF(M)
# code, protocol 14). Data: the gs session kinds again on this tree plus --floor
# (sustained -19..-4 dB, CPM exploration under the floor) and the general
# offline set, steered by the installed CNP. Eight outcome-trained CNPs
# (--labels), ensembled; one student started from the installed CNP and
# distilled on the installed model's OFDM outputs and the ensemble's CPM ones
# (LCB 0.5), so OFDM predictions stay where they are. Phases: data, train.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
SHIPPED=data2g/codes_data/outcome_predictor.npz
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DATA2G_PEP_REF_DB=5 DATA2G_SPREAD_CARRY=1
export DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
J=12
phase=${1:-all}
if [[ $phase == data || $phase == all ]]; then
  for arm in "" "--slow" "--high" "--high --kinds awgn" "--fastlow" "--floor"; do
    case $arm in
      "") n=2900 f=13000000 s="" ;;
      --slow) n=800 f=13100000 s=_slow ;;
      --high) n=1200 f=13200000 s=_high ;;
      "--high --kinds awgn") n=400 f=13300000 s=_awgnhigh ;;
      --fastlow) n=1200 f=13400000 s=_fastlow ;;
      --floor) n=1500 f=13500000 s=_floor ;;
    esac
    grep -q "^done" runs/session_data_cor$s.log 2>/dev/null && continue
    DATA2G_OUTCOME_MODEL=$SHIPPED $PY tools/with_native.py scripts/session_data.py $arm --sessions $n --first $f \
      --jobs $J --out runs/session_data_cor$s.csv > runs/session_data_cor$s.log 2>&1
    echo done >> runs/session_data_cor$s.log
    date
  done
  if ! grep -q "^done" runs/outcome_data_general_cor.log 2>/dev/null; then
    $PY tools/with_native.py scripts/outcome_data.py --samples 40000 --first 13600000 --jobs $J \
      --out runs/outcome_data_general_cor.csv > runs/outcome_data_general_cor.log 2>&1
    echo done >> runs/outcome_data_general_cor.log
  fi
  date
fi
D=runs/session_data_cor.csv,runs/session_data_cor_slow.csv,runs/session_data_cor_high.csv
D=$D,runs/session_data_cor_awgnhigh.csv,runs/session_data_cor_fastlow.csv,runs/session_data_cor_floor.csv
D=$D,runs/outcome_data_general_cor.csv,runs/outcome_data_n10top_gt.csv
if [[ $phase == train || $phase == all ]]; then
  M=""
  for k in 1 2 3 4 5 6 7 8; do
    [ -s runs/corl$k.npz ] || OMP_NUM_THREADS=4 $PY -m scripts.cnp_outcome "$D" --labels --seed $k --epochs 60 \
      --out runs/corl$k.npz > runs/corl$k.log 2>&1
    M=$M${M:+,}runs/corl$k.npz
  done
  [ -s runs/corle8.npz ] || $PY -m scripts.train_outcome x --ensemble $M --out runs/corle8.npz
  [ -s runs/cord.npz ] || DATA2G_OUTCOME_LCB=0.5 OMP_NUM_THREADS=4 $PY -m scripts.cnp_outcome "$D" --teacher $SHIPPED \
    --teacher-cpm runs/corle8.npz --init $SHIPPED --lr 2e-4 --epochs 20 --out runs/cord.npz > runs/cord.log 2>&1
  date
fi
