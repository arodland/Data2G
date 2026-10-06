#!/bin/bash
# Build the d2g-vara image: this checkout's VARA harness and the Wine prefix
# (default ~/.wine-vara, with C:\d2g_vara\a and \b set up as vara_ref.sh
# needs). Private: the prefix's VARA.ini carries the registration.
#   scripts/cpu_profile/vara-container/build.sh
# Then, e.g. (the entrypoint takes varatrials.py's arguments; reports and runs
# land in the directory mounted at /out):
#   $PODMAN run --rm --network none --userns keep-id -v "$PWD/runs:/out" d2g-vara awgn 10 --bytes 20000
# (--userns keep-id: rootless podman, so the container's vara can write /out;
# docker needs no such flag).
# PODMAN: default podman with its own vfs store (rootless overlay does not
# work on ZFS); on other machines plain `podman` or `docker` will do.
set -eu
W=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$W/../../.." && pwd)
PREFIX=${WINEPREFIX:-$HOME/.wine-vara}
PODMAN=${PODMAN:-podman --root $HOME/.local/share/containers/vfs --runroot ${XDG_RUNTIME_DIR:-/tmp}/containers-vfs --storage-driver vfs}
# not /tmp (may be RAM) and not ZFS (buildah overlays the context): /var/tmp
CTX=$(mktemp -d "${BUILD_TMP:-/var/tmp}/d2g-vara-build.XXXXXX")
trap 'rm -rf "$CTX"' EXIT
cp -a "$PREFIX" "$CTX/wineprefix"
mkdir -p "$CTX/d2g/scripts/cpu_profile" "$CTX/d2g/data2g/waveform" "$CTX/d2g/data2g/codes_data"
cp "$REPO"/scripts/cpu_profile/{varatrials.py,trials.py,vara_ref.sh,raw.py,noise.py,channel.py} "$CTX/d2g/scripts/cpu_profile/"
cp "$REPO"/data2g/{__init__,config,hfchannel}.py "$CTX/d2g/data2g/"  # channel.py's imports
cp "$REPO"/data2g/waveform/{__init__,dsp}.py "$CTX/d2g/data2g/waveform/"
cp "$REPO"/data2g/codes_data/clip_constants.json "$CTX/d2g/data2g/codes_data/"
cp "$W"/{Containerfile,entrypoint.sh,setup.sh} "$CTX/"
if grep -qw overlay /proc/filesystems || modinfo overlay > /dev/null 2>&1; then
  $PODMAN build --build-arg UID="$(id -u)" -t d2g-vara "$CTX"
else
  # podman build overlays the context; without overlay in the kernel (e.g. its
  # modules upgraded under it until a reboot), run setup.sh and commit, with
  # the Containerfile's settings
  $PODMAN rm -f d2g-vara-build > /dev/null 2>&1 || true
  $PODMAN run --name d2g-vara-build -v "$CTX:/ctx:ro" docker.io/library/archlinux:latest /ctx/setup.sh "$(id -u)"
  $PODMAN commit --change 'USER vara' --change 'ENV WINEPREFIX=/home/vara/.wine-vara WINEDEBUG=-all' \
    --change 'WORKDIR /out' --change 'ENTRYPOINT ["/entrypoint.sh"]' d2g-vara-build d2g-vara
  $PODMAN rm d2g-vara-build > /dev/null
fi
