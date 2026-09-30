#!/bin/bash
# The d2g-vara image's contents, run as root in an Arch container with
# build.sh's context at /ctx: packages, the vara user (UID $1, the prefix's
# owner), the Wine prefix and scripts, and the prefix brought up to this Wine
# (not on every run's first launch).
set -eu
printf '[multilib]\nInclude = /etc/pacman.d/mirrorlist\n' >> /etc/pacman.conf
pacman -Syu --noconfirm --needed wine xorg-server-xvfb xorg-xauth pipewire pipewire-pulse \
  wireplumber libpulse dbus python python-numpy python-scipy
pacman -Scc --noconfirm
useradd -m -u "$1" vara
install -d -o vara /out
cp -a /ctx/wineprefix /home/vara/.wine-vara
chown -R vara: /home/vara/.wine-vara
cp -a /ctx/d2g /d2g
install -m 755 /ctx/entrypoint.sh /entrypoint.sh
# Mono and Gecko off: Wine would ask to install them, in a dialog nobody
# sees on Xvfb (the update waited on it forever); VARA uses neither. A
# dialog like it again fails the build instead of hanging it.
su vara -c 'export WINEPREFIX=/home/vara/.wine-vara WINEDEBUG=-all WINEDLLOVERRIDES="mscoree=;mshtml="
  timeout 900 xvfb-run -a wineboot -u && timeout 300 wineserver -w'
