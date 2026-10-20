#!/bin/sh
# The OFDM ladder (10% points, DD, the ARQ's receiver) before and after
# support_slack, paired: the same seeds, DD unbudgeted, on the C++ core.
# BASE: a worktree at the commit before it (8393c09) with its own native build.
#   git worktree add ../Data2G-base 8393c09 && (cd ../Data2G-base && tools/build_native.sh)
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
BASE=${BASE:-$HERE/../Data2G-base}
PY=${PY:-python}  # the project's venv python (a worktree has no .venv of its own)
JOBS=${JOBS:-2}  # pool workers per tree; both trees run at once
MODES=$(cd "$HERE" && "$PY" -c "from data2g.arq.modes import MODES; print(' '.join(m for m in MODES if not m.startswith('fsk')))")
run() {  # tree out
    (cd "$1" && "$PY" tools/with_native.py --dd-budget inf scripts/ladder_study.py \
        --fail 0.1 --jobs "$JOBS" --seed 0 --start "$HERE/runs/ladder_dd_10pct.csv" --modes $MODES --out "$2")
}
run "$BASE" "$HERE/runs/support_ladder_base.csv" > "$HERE/runs/support_ladder_base.log" 2>&1 &
run "$HERE" "$HERE/runs/support_ladder_new.csv" > "$HERE/runs/support_ladder_new.log" 2>&1
wait
