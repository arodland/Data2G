#!/bin/sh
# runs/onair/tx_9700_rx_d75.sh DIR [MAX_S]: the IC-9700 (CAT PTT on /dev/ttyUSB1, 19200 baud, Hamlib 3081; audio
# into its USB input, DATA MOD USB) plays DIR/seq.wav while the TH-D75's AF through the AIOC is recorded to
# DIR/rec.wav. Never leaves the transmitter keyed: an exit trap and a MAX_S watchdog both unkey.
dir=$1
max=${2:-45}
ic9700=alsa_output.usb-Burr-Brown_from_TI_USB_Audio_CODEC-00.analog-stereo
aioc=alsa_input.usb-AIOC_All-In-One-Cable_8ea50682-00.mono-fallback
ptt() { rigctl -m 3081 -r /dev/ttyUSB1 -s 19200 T "$1"; }
trap 'ptt 0' EXIT INT TERM
(sleep "$max"; ptt 0) &
dog=$!
parecord --device=$aioc --file-format=wav --rate=48000 --channels=1 --format=s16le "$dir/rec.wav" &
rec=$!
sleep 1
t_key=$(date +%s.%N)
ptt 1
sleep 0.5
t_audio=$(date +%s.%N)
paplay --device=$ic9700 "$dir/seq.wav"
ptt 0
t_unkey=$(date +%s.%N)
sleep 2
kill $rec
wait $rec 2>/dev/null
kill $dog 2>/dev/null
echo "key $t_key audio $t_audio unkey $t_unkey" > "$dir/times.txt"
cat "$dir/times.txt"
