#!/bin/sh
# runs/onair/tx_rx.sh DIR MAX_S PTT SINK SOURCE CHANNELS: play DIR/seq.wav into SINK while keyed by PTT, and
# record SOURCE (CHANNELS, 48 kHz) to DIR/rec.wav from 1 s before to 2 s after.
#   PTT: serial:PORT:rts | serial:PORT:dtr-rts (runs/onair/serial_ptt.py) | cat:PORT (IC-9700, Hamlib 3081, 19200)
# Never leaves the transmitter keyed: the serial holder unkeys on exit, at MAX_S and when it dies; CAT keying
# has an exit trap and a MAX_S watchdog.
here=$(cd "$(dirname "$0")" && pwd)
dir=$1 max=$2 ptt=$3 sink=$4 src=$5 ch=$6
parecord --device="$src" --file-format=wav --rate=48000 --channels="$ch" --format=s16le "$dir/rec.wav" &
rec=$!
sleep 1
case $ptt in
serial:*)
  port=$(echo "$ptt" | cut -d: -f2) kind=$(echo "$ptt" | cut -d: -f3)
  python3 -I "$here/serial_ptt.py" "$port" "$kind" "$max" &
  holder=$!
  trap 'kill $holder 2>/dev/null' EXIT INT TERM
  key_off() { kill $holder; wait $holder 2>/dev/null; } ;;
cat:*)
  port=$(echo "$ptt" | cut -d: -f2)
  cat_ptt() { rigctl -m 3081 -r "$port" -s 19200 T "$1"; }
  trap 'cat_ptt 0' EXIT INT TERM
  (sleep "$max"; cat_ptt 0) &
  dog=$!
  cat_ptt 1
  key_off() { cat_ptt 0; kill $dog 2>/dev/null; } ;;
*) echo "PTT: serial:PORT:rts, serial:PORT:dtr-rts or cat:PORT" >&2; kill $rec; exit 2 ;;
esac
t_key=$(date +%s.%N)
sleep 0.5
paplay --device="$sink" "$dir/seq.wav"
key_off
t_unkey=$(date +%s.%N)
sleep 2
kill $rec
wait $rec 2>/dev/null
echo "key $t_key unkey $t_unkey" > "$dir/times.txt"
cat "$dir/times.txt"
