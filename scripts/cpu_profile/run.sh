#!/bin/bash
# Profile both data2g-host processes (py-spy, sampling) over a noisy PipeWire
# loopback: two null sinks cross-connect the hosts, noise.py (not profiled)
# adds noise to both paths. Phase "idle": listening only. Phase "pat": a Pat
# P2P exchange (W1AW -> K2XYZ: text and an 8 kB attachment; 4 kB back), the
# attachments random bytes. "pat-text": the same with text attachments (the
# repo's docs); "pat-mixed": each attachment half text, then half random.
# "raw", "raw-text", "raw-mixed": the same files sent both ways at once
# straight over the VARA data ports (raw.py), no Pat: B2F LZHUF-compresses
# every message, so under Pat data2g's own compression never engages.
# Result in <out>/pat_result.txt (with A's payload-only throughput, from
# pat_throughput.py) or raw_result.txt.
#   scripts/cpu_profile/run.sh <idle|pat|pat-text|pat-mixed|raw|raw-text|raw-mixed> <seconds | timeout> <out dir>
# then: python scripts/cpu_profile/analyze.py <out>/prof_a.txt <out>/prof_b.txt
# Needs pactl, pacat, pat (pat phases), uvx (py-spy). PY: the python with Data2G's
# dependencies (default: the repo's .venv). Uses ports 8300/8400 (+1, +20)
# and Pat HTTP 5061/5062. HOST_ARGS: extra data2g.host flags for both hosts;
# A_BYTES, B_BYTES: W1AW's and K2XYZ's attachment sizes (default 8000, 4000;
# B_BYTES=0: K2XYZ sends nothing). DATA2G_COMPRESS=0: both
# hosts send raw (no T_COMP), the baseline for the raw-text/raw-mixed phases.
# Channel and noise as ARSFI's HFSimulator makes them. CHANNEL=<awgn|mpg|mpp|
# mpd|mps | doppler_hz:delay_ms>: Watterson fading on both paths (channel.py,
# not profiled): each host plays into its own TX sink and channel.py carries
# it to the other's RX sink. Unset or awgn: noise only (its WGN mode). SNR
# (noise.py): the input's PEP over the noise in 3000 Hz; NOISE_SNR=<dB> holds
# it constant, else it wanders (snr.log).
set -u
PHASE=$1; T=$2; OUT=$(realpath -m "$3")
case $PHASE in idle|pat|pat-text|pat-mixed|raw|raw-text|raw-mixed) ;; *) echo "unknown phase $PHASE" >&2; exit 2;; esac
W=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$W/../.." && pwd)
PY=${PY:-$WT/.venv/bin/python}
rm -rf $OUT; mkdir -p $OUT
cd $WT

mods=()
for s in d2g_ab d2g_ba; do
  mods+=($(pactl load-module module-null-sink sink_name=$s sink_properties=device.description=$s rate=48000 channels=1 format=float32le))
done
TX_A=d2g_ab TX_B=d2g_ba
if [ "${CHANNEL:-awgn}" != awgn ]; then
  TX_A=d2g_a_tx TX_B=d2g_b_tx
  for s in $TX_A $TX_B; do
    mods+=($(pactl load-module module-null-sink sink_name=$s sink_properties=device.description=$s rate=48000 channels=1 format=float32le))
  done
  PYTHONPATH=$WT $PY $W/channel.py $CHANNEL 11 $TX_A d2g_ab > $OUT/channel_a.log 2>&1 & CHA=$!
  PYTHONPATH=$WT $PY $W/channel.py $CHANNEL 12 $TX_B d2g_ba > $OUT/channel_b.log 2>&1 & CHB=$!
fi
$PY $W/noise.py 7 $OUT/snr.log & NOISE=$!
sleep 2

start_host() {  # name call port sink source
  PULSE_SINK=$4 PULSE_SOURCE=$5 PYTHONPATH=$WT OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    uvx py-spy record --rate 100 --format raw --output $OUT/prof_$1.txt --nonblocking -- \
    $PY -m data2g.host --mycall $2 --input-device pulse --output-device pulse --rigctld-port 0 \
    --command-port $3 --kiss-port $(( $3 + 20 )) --record-dir $OUT/rec_$1 ${HOST_ARGS:-} > $OUT/host_$1.log 2>&1 &
}
start_host a W1AW 8300 $TX_A d2g_ba.monitor
start_host b K2XYZ 8400 $TX_B d2g_ab.monitor
sleep 6
PIDS=$(pgrep -f "^$PY -m data2g.host" | tr '\n' ' ')
# CPU timeline, 1 s, from /proc (utime+stime ticks per host)
( while true; do echo "$(date +%s) $(for p in $PIDS; do awk '{print $14+$15}' /proc/$p/stat 2>/dev/null || echo -; done)"; sleep 1; done ) > $OUT/cpu.log &
MON=$!

text() { while cat $WT/docs/*.md $WT/README.md; do :; done 2>/dev/null | head -c $1; }  # ends on SIGPIPE
attachment() {  # bytes, per phase
  case $PHASE in
    *-text) text $1;;
    *-mixed) text $(( $1 / 2 )); head -c $(( $1 - $1 / 2 )) /dev/urandom;;
    *) head -c $1 /dev/urandom;;
  esac
}

if [ "$PHASE" = idle ]; then
  sleep $T
elif [[ $PHASE == raw* ]]; then
  attachment ${A_BYTES:-8000} > $OUT/a8k.bin; attachment ${B_BYTES:-4000} > $OUT/b4k.bin
  $PY $W/raw.py 8300 8400 $OUT/a8k.bin $OUT/b4k.bin $T > $OUT/raw_result.txt 2>&1
  echo "exit $?" >> $OUT/raw_result.txt
  sleep 10
else
  P=$OUT/pat; mkdir -p $P/forms $P/prehooks
  for s in a:W1AW:8300:5061 b:K2XYZ:8400:5062; do IFS=: read k call port http <<< "$s"; mkdir -p $P/$k
    printf '{"mycall": "%s", "locator": "FN31pr", "http_addr": "127.0.0.1:%s", "listen": [], "version_reporting_disabled": true,\n "varahf": {"addr": "localhost:%s", "bandwidth": 2300, "rig": "", "ptt_ctrl": false}}\n' $call $http $port > $P/$k/config.json
  done
  pa() { k=$1; shift; pat --config $P/$k/config.json --mbox $P/$k/mbox --event-log $P/$k/events.json --log $P/$k/pat.log --forms $P/forms --prehooks $P/prehooks "$@"; }
  attachment ${A_BYTES:-8000} > $P/a8k.bin; attachment ${B_BYTES:-4000} > $P/b4k.bin
  printf 'Status report from W1AW.\nAll well here; band noisy.\n%.0s' {1..20} | pa a compose --p2p-only -s "Status" K2XYZ
  echo "Photo attached." | pa a compose --p2p-only -s "Photo" -a $P/a8k.bin K2XYZ
  [ "${B_BYTES:-4000}" != 0 ] && echo "Log attached." | pa b compose --p2p-only -s "Log" -a $P/b4k.bin W1AW
  pa b --listen varahf http > $P/b_stdout.log 2>&1 & PATB=$!
  sleep 5
  START=$(date +%s)
  timeout $T bash -c "$(declare -f pa); P=$P; pa a connect varahf:///K2XYZ" > $P/a_stdout.log 2>&1
  echo "exit $? after $(( $(date +%s) - START )) s" > $OUT/pat_result.txt
  sleep 10
  kill $PATB 2>/dev/null
  $PY $W/pat_throughput.py $OUT >> $OUT/pat_result.txt 2>&1
fi

kill $MON
pkill -INT -f "^$PY -m data2g.host" ; sleep 8   # py-spy writes its profile when the child exits
kill $NOISE ${CHA:-} ${CHB:-}; pkill -f "[p]acat --(playback|record) --device=d2g_" ; sleep 1
for m in "${mods[@]}"; do pactl unload-module $m; done
echo done
