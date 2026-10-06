#!/bin/bash
# VARA HF through run.sh's path, to place our numbers against published
# IONOS SIM results: two VARA HF instances (Wine, headless under Xvfb) on a
# PipeWire loopback of their own (var_* null sinks) with noise.py's noise
# and, with CHANNEL, channel.py's fading. raw.py connects KC2G -> KC2G-2
# (VARA registration goes by call) and writes <bytes> of random data at once:
# the payload rate, as the IONOS SIM paper's and our speed test's. (Not Pat:
# Pat writes 7 x 127 bytes per BUFFER report, and VARA reports after each
# burst: 889 bytes an over.)
#   scripts/cpu_profile/vara_ref.sh <500|2300> <bytes> <timeout s> <out dir>
# Needs a registered VARA HF in WINEPREFIX (default ~/.wine-vara) copied to
# C:\d2g_vara\a and \b, their VARA.ini set to TCP Command Port 8510 and 8610,
# KISS off (their sound devices are set here, per run). Only processes
# started here are stopped (marked by D2G_VARA_RUN in their environment);
# other Wine programs are left alone.
# NOISE_SNR, NOISE_PEAK, CHANNEL as run.sh (CHANNEL: each VARA plays into its
# own TX sink, channel.py fades it into the other's). REC=1: record both RX sinks.
# HEADLESS=0: VARA on the real display instead of Xvfb.
# WINEDEBUG: -all unless set (VARA floods Wine's debug log: disk, and slow).
# SEED: the noise's seed, and the fading's (2 SEED + 11, + 12); unset: 7, 11, 12.
# Ports 8510/8610 (+1).
set -u
BW=$1; BYTES=$2; T=$3; OUT=$(realpath -m "$4")
W=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$W/../.." && pwd)
PY=${PY:-$WT/.venv/bin/python}
export WINEPREFIX=${WINEPREFIX:-$HOME/.wine-vara} WINEDEBUG=${WINEDEBUG:--all}
rm -rf $OUT; mkdir -p $OUT
cd $WT

mods=() pids=()
sink() { mods+=($(pactl load-module module-null-sink sink_name=$1 sink_properties=device.description=$1 rate=48000 channels=1 format=float32le)); }
marked() {  # the VARA instances, their Xvfb and the Wine services they started
  for p in /proc/[0-9]*; do grep -qzxF "D2G_VARA_RUN=$OUT" $p/environ 2>/dev/null && echo ${p#/proc/}; done
}
VLOG="$WINEPREFIX/drive_c/d2g_vara"  # each instance's VARAHF.log: this run's lines kept in $OUT
declare -A vlines=([a]=$(cat "$VLOG/a/VARAHF.log" 2>/dev/null | wc -l) [b]=$(cat "$VLOG/b/VARAHF.log" 2>/dev/null | wc -l))
cleanup() {
  for k in a b; do tail -n +$(( vlines[$k] + 1 )) "$VLOG/$k/VARAHF.log" > "$OUT/VARAHF_$k.log" 2>/dev/null; done
  kill "${pids[@]}" $(marked) 2>/dev/null
  sleep 3
  kill -9 $(marked) 2>/dev/null  # winedevice.exe outlives SIGTERM, holding the audio driver
  timeout 30 wineserver -w  # a next run joining this dying server gets no audio
  sleep 1
  # by name, not the IDs load-module printed: PipeWire reuses them
  for m in $(pactl list short modules | grep -E "sink_name=var_" | cut -f1); do pactl unload-module $m; done
}
trap cleanup EXIT
sink var_ab; sink var_ba
TX_A=var_ab TX_B=var_ba
if [ "${CHANNEL:-awgn}" != awgn ]; then
  TX_A=var_a_tx TX_B=var_b_tx
  sink $TX_A; sink $TX_B
  PYTHONPATH=$WT $PY $W/channel.py $CHANNEL $(( ${SEED:-0} * 2 + 11 )) $TX_A var_ab > $OUT/channel_a.log 2>&1 & pids+=($!)
  PYTHONPATH=$WT $PY $W/channel.py $CHANNEL $(( ${SEED:-0} * 2 + 12 )) $TX_B var_ba > $OUT/channel_b.log 2>&1 & pids+=($!)
fi
NOISE_SINKS=var_ab,var_ba $PY $W/noise.py ${SEED:-7} $OUT/snr.log & pids+=($!)
if [ "${REC:-0}" = 1 ]; then
  for s in var_ab var_ba; do
    pacat --record --device=$s.monitor --format=float32le --rate=48000 --channels=1 > $OUT/$s.f32 & pids+=($!)
  done
fi

vara() {  # k tx_sink rx_sink
  # devices by Wine's names for the sinks (its default devices follow
  # PipeWire's defaults, not PULSE_SINK/PULSE_SOURCE: the station's own audio)
  sed -i "s/^Output Device Name=.*/Output Device Name=Speakers ($2)\r/; s/^Input Device Name=.*/Input Device Name=Microphone (Monitor of $3)\r/" \
    "$WINEPREFIX/drive_c/d2g_vara/$1/VARA.ini"
  local x=()
  [ "${HEADLESS:-1}" = 1 ] && x=(xvfb-run -a -s "-screen 0 1280x1024x24")
  D2G_VARA_RUN=$OUT "${x[@]}" wine "C:\\d2g_vara\\$1\\VARA.exe" > $OUT/vara_$1.log 2>&1 &
}
# Wine lists the devices once, at start: the sinks and monitors must be up
until [ "$(pactl list short sources | grep -cE "($TX_A|$TX_B|var_ab|var_ba)\.monitor")" -ge $( [ $TX_A = var_ab ] && echo 2 || echo 4) ]; do sleep 0.2; done
sleep 1
wine_streams() { { pactl list sink-inputs; pactl list source-outputs; } 2>/dev/null | grep -c 'binary = "wine-preloader"'; }
wait_audio() {  # $1 streams open and staying open for 1 s (VARA may open and close them starting up), 60 s at most
  local ok=0 i
  for i in $(seq 300); do
    if [ "$(wine_streams)" -ge "$1" ]; then ok=$(( ok + 1 )); [ $ok -ge 5 ] && return 0; else ok=0; fi
    sleep 0.2
  done
  return 1
}
start_varas() {  # one at a time: started together, one VARA may get no audio device
  local base
  base=$(wine_streams)
  vara a $TX_A var_ba && wait_audio $(( base + 2 )) || return 1  # its playback and capture
  vara b $TX_B var_ab && wait_audio $(( base + 4 ))
}
for try in 1 2 3; do  # a slow start (a loaded machine) is retried, not the end of a sweep
  start_varas && break
  { pactl list sink-inputs; pactl list source-outputs; } > $OUT/streams_fail$try.txt 2>&1
  kill -9 $(marked) 2>/dev/null; timeout 30 wineserver -w
  [ $try = 3 ] && { echo "VARA opened no audio in 60 s, 3 tries" | tee $OUT/result.txt; exit 1; }
done
sleep 3

head -c $BYTES /dev/urandom > $OUT/a.bin; : > $OUT/b.bin
RAW_CALLS="KC2G KC2G-2" RAW_BW=BW$BW $PY $W/raw.py 8510 8610 $OUT/a.bin $OUT/b.bin $T | tee $OUT/result.txt
