#!/bin/bash
# speedtrials.py (Data2G, simulated) at the IONOS SIM paper's VARA points
# (paper_points.txt), same channels, SNRs and message sizes, JOBS points at
# once. Default: the VARA 2300 points, as Data2G at its 2400 Hz cap (BW2300).
#   scripts/cpu_profile/speed_sweep.sh [regex on the line "bw channel snr bytes rate"] [out dir]
# Each point's report: <out>/2400_<channel>_<snr>dB.txt (500 Hz points:
# 500_...); a point with one is skipped, so a stopped sweep resumes. JOBS
# (default 4): each takes a full core and ~1.1 GB; VARA runs in real time
# alongside, so keep its neighbours few. HALF_BURST=1: speedtrials.py
# --half-burst (data bursts capped at 6 s OFDM, 12 s FSK), reports by default
# in runs/speedtrials_half so they never mix with (or resume from) full ones.
set -u
FILTER=${1:-^2300 }
export HALF=$([ "${HALF_BURST:-0}" = 1 ] && echo --half-burst)
OUT=${2:-runs/speedtrials${HALF:+_half}}
W=$(cd "$(dirname "$0")" && pwd)
export PY=${PY:-$(cd "$W/../.." && pwd)/.venv/bin/python} W
JOBS=${JOBS:-4}
mkdir -p "$OUT"
cd "$OUT"
point() {  # bw chan snr bytes (rate unused)
  local bw=$1 chan=$2 snr=$3 bytes=$4
  local name=$([ "$bw" = 2300 ] && echo 2400 || echo "$bw")
  local report="${name}_${chan}_${snr}dB.txt"
  grep -q succeeded "$report" 2>/dev/null && return 0
  "$PY" "$W/speedtrials.py" "$chan" "$snr" --bytes "$bytes" --bw "$bw" $HALF < /dev/null > "$report.partial" 2> "$report.err"
  mv "$report.partial" "$report"
  echo "== Data2G $name $chan $snr dB, $bytes B${HALF:+, half bursts}: $(tail -1 "$report")"
}
export -f point
grep -v '^#' "$W/paper_points.txt" | grep -E "$FILTER" | xargs -P "$JOBS" -L 1 bash -c 'point "$@"' _
