#!/bin/sh
# FM ladder pass: a constellation per 64/256-point rung (and two 16-QAM rungs),
# trained under the FM_TX transmitter at the rung's measured 10% knees, static
# and flutter at 440 (runs/fm_ladder/ladder_tx.csv). Writes
# runs/fm_const/<name>.npy and .log; scripts/fm_const_retest.py measures them.
# Rows: name m band radio init static_cnr flutter_cnr
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/fm_const
export FM_TX=1 DATA2G_PROFILE=fm PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_THREADS=2
STEPS=${STEPS:-600}
cat <<'EOF' | xargs -P 4 -L 1 sh -c '
  exec /home/andrew/code/Data2G/.venv/bin/python scripts/train_constellation.py --m $1 --band $2 --headroom 6 \
    --at fm-$3:$5 fm-$3-flutter@440:$6 --init $4 --steps '"$STEPS"' --batch 12 --frames 4 --lr 0.005 \
    --device cpu --out runs/fm_const/$0.npy > runs/fm_const/$0.log 2>&1'
c16-f55m-r47 4 f55 mic gray-qam16 5.25 6.0
c64-f55m-r50 6 f55 mic c64-w48-r12 7.0 8.0
c64-f55m-r67 6 f55 mic c64-w48-r12 8.5 9.5
c64-f55m-r86 6 f55 mic c64-w48-r34 12.5 13.75
c256-f55m-r67 8 f55 mic c256-w48-r12 14.25 15.25
c256-f55m-r72 8 f55 mic c256-w48-r12 15.75 17.0
c16-f55f-r67 4 f55 flat gray-qam16 5.75 6.5
c256-f55f-r47 8 f55 flat c256-w48-r12 7.5 8.5
c256-f55f-r60 8 f55 flat c256-w48-r12 9.0 10.0
c256-f55f-r67 8 f55 flat c256-w48-r12 10.0 11.0
c256-f55f-r72 8 f55 flat c256-w48-r12 11.0 11.75
c64-f108f-r40 6 f108 flat c64-w48-r12 8.5 9.25
c64-f108f-r50 6 f108 flat c64-w48-r12 10.0 10.5
c64-f108f-r59 6 f108 flat c64-w48-r12 11.5 12.25
c64-f108f-r67 6 f108 flat c64-w48-r23 13.0 13.75
c256-f108f-r59 8 f108 flat c256-w48-r12 15.5 16.0
c256-f108f-r67 8 f108 flat c256-w48-r12 17.5 18.0
c256-f108f-r75 8 f108 flat c256-w48-r58 19.75 20.5
c256-f108f-r80 8 f108 flat c256-w48-r58 21.25 22.0
EOF
