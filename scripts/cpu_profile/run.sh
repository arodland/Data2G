#!/bin/bash
# Profile both data2g-host processes (py-spy, sampling) over a noisy PipeWire
# loopback: two null sinks cross-connect the hosts, noise.py (not profiled)
# adds noise to both paths. Phase "idle": listening only. Phase "pat": a Pat
# P2P exchange (W1AW -> K2XYZ: text and an 8 kB attachment; 4 kB back).
#   scripts/cpu_profile/run.sh <idle|pat> <seconds | timeout> <out dir>
# then: python scripts/cpu_profile/analyze.py <out>/prof_a.txt <out>/prof_b.txt
# Needs pactl, pacat, pat, uvx (py-spy). PY: the python with Data2G's
# dependencies (default: the repo's .venv). Uses ports 8300/8400 (+1, +20)
# and Pat HTTP 5061/5062.
set -u
PHASE=$1; T=$2; OUT=$(realpath -m "$3")
W=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$W/../.." && pwd)
PY=${PY:-$WT/.venv/bin/python}
rm -rf $OUT; mkdir -p $OUT
cd $WT

mods=()
for s in d2g_ab d2g_ba; do
  mods+=($(pactl load-module module-null-sink sink_name=$s sink_properties=device.description=$s rate=48000 channels=1 format=float32le))
done
$PY $W/noise.py 7 $OUT/snr.log & NOISE=$!
sleep 2

start_host() {  # name call port sink source
  PULSE_SINK=$4 PULSE_SOURCE=$5 PYTHONPATH=$WT OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    uvx py-spy record --rate 100 --format raw --output $OUT/prof_$1.txt --nonblocking -- \
    $PY -m data2g.host --mycall $2 --input-device pulse --output-device pulse --rigctld-port 0 \
    --command-port $3 --kiss-port $(( $3 + 20 )) --record-dir $OUT/rec_$1 > $OUT/host_$1.log 2>&1 &
}
start_host a W1AW 8300 d2g_ab d2g_ba.monitor
start_host b K2XYZ 8400 d2g_ba d2g_ab.monitor
sleep 6
PIDS=$(pgrep -f "^$PY -m data2g.host" | tr '\n' ' ')
# CPU timeline, 1 s, from /proc (utime+stime ticks per host)
( while true; do echo "$(date +%s) $(for p in $PIDS; do awk '{print $14+$15}' /proc/$p/stat 2>/dev/null || echo -; done)"; sleep 1; done ) > $OUT/cpu.log &
MON=$!

if [ "$PHASE" = idle ]; then
  sleep $T
else
  P=$OUT/pat; mkdir -p $P/forms $P/prehooks
  for s in a:W1AW:8300:5061 b:K2XYZ:8400:5062; do IFS=: read k call port http <<< "$s"; mkdir -p $P/$k
    printf '{"mycall": "%s", "locator": "FN31pr", "http_addr": "127.0.0.1:%s", "listen": [], "version_reporting_disabled": true,\n "varahf": {"addr": "localhost:%s", "bandwidth": 2300, "rig": "", "ptt_ctrl": false}}\n' $call $http $port > $P/$k/config.json
  done
  pa() { k=$1; shift; pat --config $P/$k/config.json --mbox $P/$k/mbox --event-log $P/$k/events.json --log $P/$k/pat.log --forms $P/forms --prehooks $P/prehooks "$@"; }
  head -c 8000 /dev/urandom > $P/a8k.bin; head -c 4000 /dev/urandom > $P/b4k.bin
  printf 'Status report from W1AW.\nAll well here; band noisy.\n%.0s' {1..20} | pa a compose --p2p-only -s "Status" K2XYZ
  echo "Photo attached." | pa a compose --p2p-only -s "Photo" -a $P/a8k.bin K2XYZ
  echo "Log attached." | pa b compose --p2p-only -s "Log" -a $P/b4k.bin W1AW
  pa b --listen varahf http > $P/b_stdout.log 2>&1 & PATB=$!
  sleep 5
  START=$(date +%s)
  timeout $T bash -c "$(declare -f pa); P=$P; pa a connect varahf:///K2XYZ" > $P/a_stdout.log 2>&1
  echo "exit $? after $(( $(date +%s) - START )) s" > $OUT/pat_result.txt
  sleep 10
  kill $PATB 2>/dev/null
fi

kill $MON
pkill -INT -f "^$PY -m data2g.host" ; sleep 8   # py-spy writes its profile when the child exits
kill $NOISE; pkill -f "[p]acat --playback --device=d2g_" ; sleep 1
for m in "${mods[@]}"; do pactl unload-module $m; done
echo done
