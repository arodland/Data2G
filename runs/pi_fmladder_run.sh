#!/bin/sh
# On minstrel (~/Data2G-bench/fmladder): the frozen FM ladders (f55, f108; fm-stream). Build the FM
# data2g-bench-rx, then: listening on noise with the FM engine's bands (120 s), and every burst
# streamed (pool 2, DD budget 1 s, best of 3), with and without decoding while it arrives.
# Bursts: runs/pi_gen_fmladder.sh (FM simulator, a station's fixed level, tx_path by radio).
cd "$HOME/Data2G-bench/fmladder"
cmake -S src/native -B build-fm -G Ninja -DDATA2G_PROFILE=fm -DDATA2G_BUILD_HOST=OFF -DDATA2G_BUILD_GUI=OFF \
  -DDATA2G_BUILD_RIG=OFF -DDATA2G_BUILD_TESTS=OFF -DDATA2G_BUILD_PYMODULE=OFF > cmake-fm.log 2>&1 &&
  ninja -C build-fm -j3 data2g-bench-rx > ninja-fm.log 2>&1
echo "built fm $?"
echo "== listen, FM engine bands, 120 s"
build-fm/data2g-bench-rx --threads 2 --listen 120 --engine
for mode in "" --no-early; do
  echo "== stream $mode"
  build-fm/data2g-bench-rx --threads 2 --reps 3 --stream $mode bursts/*.d2gb
done
echo finished
