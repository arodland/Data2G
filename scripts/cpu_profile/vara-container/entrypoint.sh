#!/bin/bash
# The container's own audio stack (D-Bus, PipeWire, wireplumber,
# pipewire-pulse), then varatrials.py with the container's arguments, in
# /out (mount the host's output directory there).
set -e
export XDG_RUNTIME_DIR=/tmp/xdg
mkdir -p -m 700 "$XDG_RUNTIME_DIR"
DBUS_SESSION_BUS_ADDRESS=$(dbus-daemon --session --fork --print-address)
export DBUS_SESSION_BUS_ADDRESS
pipewire > /tmp/pipewire.log 2>&1 &
wireplumber > /tmp/wireplumber.log 2>&1 &
pipewire-pulse > /tmp/pipewire-pulse.log 2>&1 &
for _ in $(seq 100); do pactl info > /dev/null 2>&1 && break; sleep 0.1; done
pactl info > /dev/null || { echo "no PulseAudio server in the container" >&2; exit 2; }
exec python3 /d2g/scripts/cpu_profile/varatrials.py "$@"
