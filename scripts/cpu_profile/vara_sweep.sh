#!/bin/bash
# varatrials.py at every point of the IONOS SIM paper's VARA tables
# (sim-mar-2021.xlsx: "VARA 4.0 2300Hz" on HF Wide, "VARA 500" on HF 500),
# SNR 40 .. -5 on WGN, MPG 2 and MPP 2 (awgn, mpg, mpp here), each with the
# paper's message size. One point at a time (vara_ref.sh's sinks and ports
# are fixed); ~113 h of real time at the paper's rates, failures extra.
#   scripts/cpu_profile/vara_sweep.sh [regex on the line "bw channel snr bytes rate"] [out dir]
# e.g. vara_sweep.sh '^2300 awgn ' or '^500 mpg (40|-5) '. Each point's report:
# <out>/<bw>_<channel>_<snr>dB.txt;
# a point with one is skipped, so a stopped sweep resumes. A transfer taking
# over 3x the paper's time (at least 15 min) fails. Stops if varatrials
# stops (the harness failed, not VARA).
set -u
FILTER=${1:-.}
OUT=${2:-runs/varatrials}
W=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$(cd "$W/../.." && pwd)/.venv/bin/python}
mkdir -p "$OUT"
# bw channel snr bytes paper_B/min (the paper's)
POINTS="
2300 awgn 40 205451 46227
2300 mpg 40 205451 31158
2300 mpp 40 205451 19711
2300 awgn 35 205451 46055
2300 mpg 35 205451 31284
2300 mpp 35 205451 21031
2300 awgn 30 205451 46208
2300 mpg 30 205451 28972
2300 mpp 30 205451 17911
2300 awgn 25 205451 37490
2300 mpg 25 205451 19282
2300 mpp 25 205451 17694
2300 awgn 20 205451 23511
2300 mpg 20 205451 11745
2300 mpp 20 205451 10945
2300 awgn 15 205451 11446
2300 mpg 15 137017 6829
2300 mpp 15 137017 7215
2300 awgn 10 205451 5456
2300 mpg 10 68731 3060
2300 mpp 10 137017 3672
2300 awgn 5 68925 2331
2300 mpg 5 19392 1482
2300 mpp 5 19392 1421
2300 awgn 0 19392 1158
2300 mpg 0 19392 839
2300 mpp 0 19392 988
2300 awgn -5 19392 514
2300 mpg -5 9732 214
2300 mpp -5 19392 326
500 awgn 40 137007 10377
500 mpg 40 137007 7640
500 mpp 40 137007 6269
500 awgn 35 137007 10388
500 mpg 35 137007 7344
500 mpp 35 69311 5883
500 awgn 30 137007 10402
500 mpg 30 137007 6955
500 mpp 30 69311 5072
500 awgn 25 137007 10322
500 mpg 25 137007 5443
500 mpp 25 69311 4897
500 awgn 20 137007 7853
500 mpg 20 69311 4749
500 mpp 20 69311 4336
500 awgn 15 69311 5897
500 mpg 15 69311 2389
500 mpp 15 69311 2590
500 awgn 10 69311 3199
500 mpg 10 19380 1382
500 mpp 10 69311 1601
500 awgn 5 19451 1183
500 mpg 5 19380 526
500 mpp 5 19380 839
500 awgn 0 19451 462
500 mpg 0 8640 193
500 mpp 0 8640 312
500 awgn -5 8654 211
500 mpg -5 8640 84
500 mpp -5 8640 104
"
echo "$POINTS" | grep -E "$FILTER" | while read -r bw chan snr bytes rate; do
  [ -z "$bw" ] && continue
  report="$OUT/${bw}_${chan}_${snr}dB.txt"
  grep -q succeeded "$report" 2>/dev/null && continue
  limit=$(( bytes * 60 * 3 / rate )); [ $limit -lt 900 ] && limit=900
  echo "== VARA $bw $chan $snr dB, $bytes B (paper $rate B/min), limit ${limit} s: $(date '+%F %T')"
  $PY "$W/varatrials.py" "$chan" "$snr" --bytes "$bytes" --bw "$bw" --timeout "$limit" --out "$OUT" < /dev/null \
    | tee "$report.partial"
  status=${PIPESTATUS[0]}
  mv "$report.partial" "$report"
  [ "$status" = 2 ] && { echo "varatrials stopped: the harness failed; fix it and rerun to resume"; exit 2; }
done
