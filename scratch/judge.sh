#!/bin/bash
# Paired loss study of the coherent CPM receiver, same tree, Python only
# (python -m, so data2g is this tree's), shift+cpm, PEP 5. Floor cells plus
# controls above. Arms (ARMS, default "cohctl coh off"):
#   cohctl  all on; coh  DD + either-gate lock, square-law control; off  all off (== master)
#   scratch/judge.sh SEED0 SEEDS
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DATA2G_PEP_REF_DB=5
S0=$1 NS=$2
FLOOR=awgn:-18,awgn:-16,awgn:-14,awgn:-12,mpp:-16,mpp:-14,mpp:-12,mpd:-15,mpd:-13,mpd:-11,mpd:-9
FLOOR=$FLOOR,mpg:-14,mpg:-12,mpg:-10,mpg:-8,mpg:-6
CTRL=mpp:-8,mpd:-6,mpg:-4,awgn:-4,mpp:0
mkdir -p runs/judge
for arm in ${ARMS:-cohctl coh off}; do
  out=runs/judge/loss_${arm}_s$S0.csv
  case $arm in
    cohctl) export DATA2G_CPM_DD=1 DATA2G_CPM_COH_LOCK=1 DATA2G_CPM_COH_CTL=1 ;;
    coh) export DATA2G_CPM_DD=1 DATA2G_CPM_COH_LOCK=1 DATA2G_CPM_COH_CTL=0 ;;
    off) export DATA2G_CPM_DD=0 DATA2G_CPM_COH_LOCK=0 DATA2G_CPM_COH_CTL=0 ;;
  esac
  [ -s $out ] || $PY -m scripts.loss_study --out $out --seed0 $S0 --seeds $NS --horizon 600 --jobs 8 \
    --policy shift+cpm --cells $FLOOR,$CTRL > ${out%.csv}.log 2>&1
  date
done
