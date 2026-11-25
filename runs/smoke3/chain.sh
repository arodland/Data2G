#!/bin/bash
# fast-fading-integration smoke loss study on master 1f12161 + PR #317 (2026-11-25)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_PEP_REF_DB=5
PY=/home/andrew/code/Data2G/.venv/bin/python
OFF="DATA2G_PILOTS=off DATA2G_CPM_LEAK=off DATA2G_CPM_R23=off DATA2G_CPM_ONAIR=off DATA2G_CPM_ACQ_K=off"
C2400=hilat-d:10,hilat-m:10,hilat-m:20,lowlat-d:10,nvis-d:10,mpd:0,mpp:0,mpp:-8,awgn:-12,awgn:6,awgn:-15,awgn:-19,awgn:-20,awgn:-12:carrier12
C500=mpd:0,mpp:0
D=runs/smoke3
run() { tag=$1; cap=$2; cells=$3; shift 3; date; env "$@" $PY -m scripts.loss_study_carrier --out $D/$tag.csv --seeds 12 --horizon 600 --jobs 6 --policy shift+cpm --cap $cap --cells $cells > $D/$tag.log 2>&1; }
run off2400 2 $C2400 $OFF
run def2400 2 $C2400
run off500 0 $C500 $OFF
run def500 0 $C500
# plain master 1f12161 (no #317) on #317's cells, from its own tree
(cd ../ref-1f12161 && date && env DATA2G_PEP_REF_DB=5 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m scripts.loss_study_carrier --out ../fast-fading-integration/$D/master2400.csv --seeds 12 --horizon 600 --jobs 6 --policy shift+cpm --cap 2 --cells awgn:-15,awgn:-19,awgn:-20 > ../fast-fading-integration/$D/master2400.log 2>&1)
date; echo DONE
