#!/bin/bash
# Speedtrials (20 kB, 5 successes or 10 failures) at BW500 and BW2300, the
# outcome gate on (the default) and off (DATA2G_OUTCOME_GATE=''), 12 at a time;
# one file per run in runs/gate_speed/, then a table.
cd "$(dirname "$0")/.."
PY=/home/andrew/code/Data2G/.venv/bin/python
mkdir -p runs/gate_speed
one() {  # bw arm chan snr
  local out=runs/gate_speed/$1_$2_$3_$4.txt
  [ -s "$out" ] && return
  if [ "$2" = off ]; then export DATA2G_OUTCOME_GATE=""; fi
  $PY scripts/cpu_profile/speedtrials.py "$3" "$4" --bw "$1" --bytes 20000 2>/dev/null | tail -2 > "$out.tmp" && mv "$out.tmp" "$out"
}
export -f one
export PY
for bw in 500 2300; do for arm in on off; do
  for c in "awgn 5" "awgn 10" "awgn 15" "awgn 20" "awgn 25" "awgn 30" "mpg 15" "mpg 25" "mpp 15" "mpp 25" "mpd 15"; do
    echo "$bw $arm $c"
  done
done; done | xargs -P 12 -L 1 bash -c 'one $0 $1 $2 $3'
for bw in 500 2300; do
  echo "== BW$bw: B/min, gate off -> on (top mode, on)"
  for c in "awgn 5" "awgn 10" "awgn 15" "awgn 20" "awgn 25" "awgn 30" "mpg 15" "mpg 25" "mpp 15" "mpp 25" "mpd 15"; do
    set -- $c
    off=$(grep -o '[0-9]* B/min' runs/gate_speed/${bw}_off_$1_$2.txt | tail -1)
    on=$(grep -o '[0-9]* B/min' runs/gate_speed/${bw}_on_$1_$2.txt | tail -1)
    ok=$(grep -o '[0-9]*/[0-9]* succeeded' runs/gate_speed/${bw}_on_$1_$2.txt)
    top=$(grep 'mean of' runs/gate_speed/${bw}_on_$1_$2.txt | sed 's/.*; //')
    printf "%-5s %3s dB  %10s -> %10s  (%s; %s)\n" "$1" "$2" "${off:-fail}" "${on:-fail}" "$ok" "$top"
  done
done
