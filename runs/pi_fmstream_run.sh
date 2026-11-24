#!/bin/sh
# On minstrel (~/Data2G-bench/fmstream): decoding while the burst arrives (fm-stream). Build the FM
# data2g-bench-rx, then stream each burst through the native Receiver and engine-like loop (pool 2,
# DD budget 1 s, best of 3), with and without early decoding. Bursts: FM profile, protocol 2
# (runs/pi_gen_fm206.sh's set, plus seeds 1-4 of marginal 256l/64l on mpp).
cd "$HOME/Data2G-bench/fmstream"
cmake -S src/native -B build-fm -G Ninja -DDATA2G_PROFILE=fm -DDATA2G_BUILD_HOST=OFF -DDATA2G_BUILD_GUI=OFF \
  -DDATA2G_BUILD_RIG=OFF -DDATA2G_BUILD_TESTS=OFF -DDATA2G_BUILD_PYMODULE=OFF > cmake-fm.log 2>&1 &&
  ninja -C build-fm -j3 data2g-bench-rx > ninja-fm.log 2>&1
echo "built fm $?"
for mode in "" --no-early; do
  echo "== stream $mode"
  build-fm/data2g-bench-rx --threads 2 --reps 3 --stream $mode bursts/*.d2gb
done
echo finished
