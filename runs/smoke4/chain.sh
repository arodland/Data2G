#!/bin/bash
# defaults arm after merging scattered-pilots bbc0b40 (short-burst remedies); the all-off arm is runs/smoke3's
# (all-off code is identical to master b278852 and to smoke3's tree by hash)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_PEP_REF_DB=5
PY=/home/andrew/code/Data2G/.venv/bin/python
C2400=hilat-d:10,hilat-m:10,hilat-m:20,lowlat-d:10,nvis-d:10,mpd:0,mpp:0,mpp:-8,awgn:-12,awgn:6,awgn:-15,awgn:-19,awgn:-20,awgn:-12:carrier12
C500=mpd:0,mpp:0
D=runs/smoke4
date; $PY -m scripts.loss_study_carrier --out $D/def2400.csv --seeds 12 --horizon 600 --jobs 6 --policy shift+cpm --cap 2 --cells $C2400 > $D/def2400.log 2>&1
date; $PY -m scripts.loss_study_carrier --out $D/def500.csv --seeds 12 --horizon 600 --jobs 6 --policy shift+cpm --cap 0 --cells $C500 > $D/def500.log 2>&1
date; echo DONE
