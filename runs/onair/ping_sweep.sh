#!/bin/sh
# runs/onair/ping_sweep.sh OUT FROM_PORT TO_PORT TO_CALL BW SINK LABEL V1 [V2 ...]: one ping at each PipeWire
# volume V of SINK (the sender's audio), appending the PINGACK/PINGVERDICT lines, labelled, to OUT; leaves
# SINK at the last V.
out=$1 fp=$2 tp=$3 call=$4 bw=$5 sink=$6 label=$7
shift 7
here=$(cd "$(dirname "$0")" && pwd)
py=/home/andrew/code/Data2G/.venv/bin/python
for v in "$@"; do
  pactl set-sink-volume "$sink" "$v%"
  "$py" "$here/ping_pair.py" "$fp" "$tp" "$call" "$bw" 25 | grep -E "PINGACK|VERDICT" | sed "s/^/$label sink$v /" >> "$out"
done
