#!/bin/sh
# runs/pi_gen_fmladder.sh OUT: FM Pi bench bursts for the frozen ladders (f55, f108): top rungs
# clean / just above their 10% static knee / failing, plus mobile cases and a low rung.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATA2G_PROFILE=fm
root=$(cd "$(dirname "$0")/.." && pwd)
out=$1
mkdir -p "$out"
{
  for c in 30 13 9; do echo "f55-64l-r83:13:static:mic:$c:1"; done
  for c in 30 17 13; do echo "f55-256l-r72:26:static:mic:$c:1"; done
  for c in 35 29 25; do echo "f55-256l-r88:26:static:mic:$c:1"; done
  for c in 30 17 13; do echo "f55-256l-r88:26:static:flat:$c:1"; done
  for c in 30 17 13; do echo "f108-64l-r80:26:static:flat:$c:1"; done
  for c in 30 23 19; do echo "f108-256l-r80:26:static:flat:$c:1"; done
  for c in 35 25 21; do echo "f108-256l-r87:26:static:flat:$c:1"; done
  echo "f108-256l-r87:26:mobile-rural:flat:35:1"
  echo "f55-64l-r67:13:mobile-rural:mic:25:1"
  echo "f55-qpsk-r21:6:static:mic:10:1"
} | PYTHONPATH="$root:$root/scripts" xargs -P 8 -n 3 /home/andrew/code/Data2G/.venv/bin/python "$root/scripts/pi_genburst_fm.py" "$out"
ls "$out" | wc -l
