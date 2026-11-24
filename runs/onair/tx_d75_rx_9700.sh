#!/bin/sh
# runs/onair/tx_d75_rx_9700.sh DIR [MAX_S]: the TH-D75 (AIOC; PTT on /dev/ttyACM0, DTR high and RTS low (RTS high inhibits):
# runs/onair/aioc_ptt.py) plays DIR/seq.wav while the IC-9700's USB audio (IF or AF, as its menu is set) is
# recorded to DIR/rec.wav. Never leaves the transmitter keyed: the PTT holder unkeys on SIGTERM, on its own MAX_S
# limit, and (the port closing) if it dies.
here=$(cd "$(dirname "$0")" && pwd)
dir=$1
max=${2:-45}
aioc=alsa_output.usb-AIOC_All-In-One-Cable_8ea50682-00.mono-fallback
ic9700=alsa_input.usb-Burr-Brown_from_TI_USB_Audio_CODEC-00.analog-stereo
parecord --device=$ic9700 --file-format=wav --rate=48000 --channels=2 --format=s16le "$dir/rec.wav" &
rec=$!
sleep 1  # noise with no carrier: the analyser reads the radio's IF filter and the noise from it
t_key=$(date +%s.%N)
python3 -I "$here/aioc_ptt.py" /dev/ttyACM0 "$max" &
ptt=$!
trap 'kill $ptt 2>/dev/null' EXIT INT TERM
sleep 0.5  # TX delay before the audio
t_audio=$(date +%s.%N)
paplay --device=$aioc "$dir/seq.wav"
kill $ptt
wait $ptt 2>/dev/null
t_unkey=$(date +%s.%N)
sleep 2
kill $rec
wait $rec 2>/dev/null
echo "key $t_key audio $t_audio unkey $t_unkey" > "$dir/times.txt"
cat "$dir/times.txt"
