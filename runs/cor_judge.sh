#!/bin/bash
# The installed CNP against cord (runs/cor_round.sh) on this tree, paired loss
# study, shift+cpm, PEP 5, both arms with the Python shifter and the shipped
# offsets. The usual 29 cells plus the CPM floor cells.
#   runs/cor_judge.sh SEED0 SEEDS TAG=MODEL ...
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DATA2G_PEP_REF_DB=5 DATA2G_SPREAD_CARRY=1 DATA2G_OUTCOME_GATE= DATA2G_OUTCOME_FAST=
export DATA2G_LOGIT_OFFSETS=w48-16qam-r1/2:-1.0,n10-256l-r3/4:-0.5
S0=$1 NS=$2; shift 2
C9=awgn:0,mpg:0,mpg:8,mpp:-4,mpp:0,mpp:8,mpd:-4,mpd:0,mpd:8
ALL=$C9,mpd:-6,mpp:-8,awgn:15,mpp:-6,mpd:20,mpg:15,mpp:15,mpg:-4,awgn:-4,awgn:4,awgn:8,mpg:4,mpg:12,mpp:-10,mpd:-8
ALL=$ALL,awgn:4:slow,awgn:15:slow,mpg:8:slow,mpp:0:slow,mpd:0:slow
FLOOR=awgn:-18,awgn:-16,awgn:-14,awgn:-12,mpp:-16,mpp:-14,mpp:-12,mpd:-15,mpd:-13,mpd:-11,mpd:-9
FLOOR=$FLOOR,mpg:-14,mpg:-12,mpg:-10,mpg:-8,mpg:-6
for arm in "$@"; do
  tag=${arm%%=*} model=${arm#*=}
  out=runs/loss_cor_${tag}_s$S0.csv
  [ -s $out ] || DATA2G_OUTCOME_MODEL=$model $PY tools/with_native.py scripts/loss_study.py --out $out --seed0 $S0 \
    --seeds $NS --horizon 600 --jobs 12 --policy shift+cpm --cells $ALL,$FLOOR > ${out%.csv}.log 2>&1
  date
done
