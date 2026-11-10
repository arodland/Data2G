#!/bin/bash
# speedtrials.py (Data2G, simulated) at the IONOS SIM paper's VARA points
# (paper_points.txt), same channels, SNRs and message sizes, JOBS points at
# once. Default: the VARA 2300 points, as Data2G at its 2400 Hz cap (BW2300).
#   scripts/cpu_profile/speed_sweep.sh [regex on the line "bw channel snr bytes rate"] [out dir]
# Each point's report: <out>/2400_<channel>_<snr>dB.txt (500 Hz points:
# 500_...); a point with one is skipped, so a stopped sweep resumes. JOBS
# (default 4): each takes a full core and ~1.1 GB; VARA runs in real time
# alongside, so keep its neighbours few. REVERSE=1: the callee sends the bulk
# data (speedtrials.py --reverse); reports: <out>/2400_<channel>_<snr>dB_rev.txt.
# BOTH=1: both send half at once (--both); reports: ..._<snr>dB_both.txt.
# NATIVE (default 1): the C++ core substituted (tools/with_native.py; build it
# first with tools/build_native.sh); NATIVE=0 runs the Python reference. PY:
# the interpreter, default this checkout's .venv, else the main checkout's.
# The release recipe: docs/speedtests/README.md.
set -u
FILTER=${1:-^2300 }
OUT=${2:-runs/speedtrials}
W=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$W/../.." && pwd)
MAIN=$(cd "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)/.." && pwd)
[ -x "$ROOT/.venv/bin/python" ] && DEF_PY=$ROOT/.venv/bin/python || DEF_PY=$MAIN/.venv/bin/python
export PY=${PY:-$DEF_PY} W ROOT NATIVE=${NATIVE:-1}
JOBS=${JOBS:-4}
export REVERSE=${REVERSE:-} BOTH=${BOTH:-}
on() { [ -n "$1" ] && [ "$1" != 0 ]; }
if on "$REVERSE" && on "$BOTH"; then echo "REVERSE and BOTH: pick one" >&2; exit 2; fi
export -f on
mkdir -p "$OUT"
cd "$OUT"
point() {  # bw chan snr bytes (rate unused)
  local bw=$1 chan=$2 snr=$3 bytes=$4
  local name=$([ "$bw" = 2300 ] && echo 2400 || echo "$bw")
  local dir=() suffix= tag= wrap=()
  on "$NATIVE" && wrap=("$ROOT/tools/with_native.py")
  on "$REVERSE" && dir=(--reverse) suffix=_rev tag=" reverse"
  on "$BOTH" && dir=(--both) suffix=_both tag=" both"
  local report="${name}_${chan}_${snr}dB${suffix}.txt"
  grep -q succeeded "$report" 2>/dev/null && return 0
  "$PY" "${wrap[@]}" "$W/speedtrials.py" "$chan" "$snr" --bytes "$bytes" --bw "$bw" "${dir[@]}" < /dev/null > "$report.partial" 2> "$report.err"
  mv "$report.partial" "$report"
  echo "== Data2G $name $chan $snr dB, $bytes B$tag: $(tail -1 "$report")"
}
export -f point
grep -v '^#' "$W/paper_points.txt" | grep -E "$FILTER" | xargs -P "$JOBS" -L 1 bash -c 'point "$@"' _
