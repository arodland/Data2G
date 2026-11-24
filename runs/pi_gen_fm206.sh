#!/bin/sh
# runs/pi_gen_fm206.sh: bursts for the FM plan's M2b Pi bench (HF and FM profiles, one tree).
# Today's top w48 modes on mpp at a converging, a middling and a marginal SNR.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
root=$(cd "$(dirname "$0")/.." && pwd)
for p in hf fm; do
  out="$root/runs/pi-fm206/bursts-$p"
  mkdir -p "$out"
  {
    for s in 12 16 20; do echo "w48-16qam-r2/3:19:mpp:$s:1"; done
    for s in 18 24 30; do echo "w48-64l-r3/4:30:mpp:$s:1"; done
    for s in 20 24 30; do echo "w48-256l-r5/8:39:mpp:$s:1"; done
  } | DATA2G_PROFILE=$p PYTHONPATH="$root:$root/scripts" xargs -P 8 -n 3 \
      /home/andrew/code/Data2G/.venv/bin/python "$root/scripts/pi_genburst.py" "$out"
done
