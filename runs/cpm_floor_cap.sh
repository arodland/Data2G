#!/bin/bash
# Is the CPM floor (policy.CPM_FLOOR, snr:-4: data in CPM only while the peer's
# median snr_est is under -4 dB) still worth it with the CNP shifter? Arms: the
# shipped snr:-4, off, snr:-6, snr:-8; every arm sets DATA2G_CPM_FLOOR, so all
# run the Python shifter. Low-SNR cells on all four channels at the 500 Hz and
# 2400 Hz caps, 12 seeds x 600 s, PEP5, DD unbudgeted (paired).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/andrew/code/Data2G/.venv/bin/python}
JOBS=${JOBS:-16}
export DATA2G_PEP_REF_DB=5 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CF=awgn:-12,awgn:-10,awgn:-8,awgn:-6,awgn:-4,awgn:0,mpg:-12,mpg:-10,mpg:-8,mpg:-6,mpg:-4,mpg:0,mpp:-10,mpp:-8,mpp:-6,mpp:-4,mpp:0,mpd:-10,mpd:-8,mpd:-6,mpd:-4,mpd:0
ARMS=${ARMS:-snr:-4 off snr:-6 snr:-8}
SEED0=${SEED0:-0}  # a second, independent set: 12
sfx=$([ $SEED0 = 0 ] || echo _s$SEED0)
for cap in 0 2; do
  for arm in $ARMS; do
    out=runs/loss_cfcap${cap}_${arm//:/}$sfx.csv
    [ -s $out ] && continue
    floor=""; [ $arm != off ] && floor=$arm
    DATA2G_CPM_FLOOR=$floor $PY tools/with_native.py --dd-budget inf scripts/loss_study.py --out $out --cap $cap \
      --seeds 12 --seed0 $SEED0 --horizon 600 --jobs $JOBS --policy shift+cpm --cells $CF > ${out%.csv}.log 2>&1
    date
  done
done
for cap in 0 2; do
  for arm in $ARMS; do
    [ $arm = snr:-4 ] && continue
    echo "== cap $cap: snr:-4 (shipped) vs $arm"; $PY scripts/paired_loss.py runs/loss_cfcap${cap}_snr-4$sfx.csv runs/loss_cfcap${cap}_${arm//:/}$sfx.csv
  done
done
