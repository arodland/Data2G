#!/bin/sh
# runs/pi_gen_all.sh TREE OUT: the CPM sweep and the OFDM sanity bursts with TREE's data2g.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
tree=$1; out=$2
gen=$(dirname "$0")/../scripts/pi_genburst.py
{
for m in fsk8r50 fsk16r25 fsk32r62; do
  case $m in fsk8r50) snrs="-16 -13 -11 -8";; fsk16r25) snrs="-18 -15 -13 -10";; fsk32r62) snrs="-14 -11 -9 -6";; esac
  for r in r1/3 r1/2; do for n in 2 5 9; do for s in $snrs; do echo "$m-$r:$n:awgn:$s:1"; done; done; done
done
for s in 20 22 24 30; do echo "w48-256l-r5/8:39:mpp:$s:1"; done
for s in 12 14 16 20; do echo "w48-16qam-r2/3:19:mpp:$s:1"; done
} | PYTHONPATH=$tree:$tree/scripts xargs -P 8 -n 4 /home/andrew/code/Data2G/.venv/bin/python "$gen" "$out"
