#!/bin/sh
# On minstrel (~/Data2G-bench/fm206): FM plan M2b. Build data2g-bench-rx for the HF and FM profiles of one
# tree (fm-rx, #206), then per profile: listening on noise (all bands, 120 s) and burst end to payload
# (pool 2, DD budget 1 s, best of 3), each profile's own bursts (runs/pi_gen_fm206.sh).
cd "$HOME/Data2G-bench/fm206"
for p in hf fm; do
  cmake -S src/native -B build-$p -G Ninja -DDATA2G_PROFILE=$p -DDATA2G_BUILD_HOST=OFF -DDATA2G_BUILD_GUI=OFF \
    -DDATA2G_BUILD_RIG=OFF -DDATA2G_BUILD_TESTS=OFF -DDATA2G_BUILD_PYMODULE=OFF > cmake-$p.log 2>&1 &&
    ninja -C build-$p -j3 data2g-bench-rx > ninja-$p.log 2>&1
  echo "built $p $?"
done
for p in hf fm; do
  echo "== $p listen all bands 120 s"
  build-$p/data2g-bench-rx --threads 2 --listen 120
  echo "== $p bursts"
  build-$p/data2g-bench-rx --threads 2 --reps 3 bursts-$p/*.d2gb
done
echo finished
