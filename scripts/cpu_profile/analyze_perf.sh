#!/bin/bash
# perf profiles of the native host (run.sh with NATIVE=1) -> text reports, one set per host.
#   scripts/cpu_profile/analyze_perf.sh <out dir> [top N (default 40)]
# Writes <out>/perf_<a|b>_{threads,self,incl}.txt and prints the thread split and the top lists.
#   threads: samples per thread name (pool workers, the engine thread, audio, ...)
#   self:    own time per symbol, callers folded away
#   incl:    time under each symbol, its callees included (a call-graph build, DWARF unwinding: slow)
set -u
OUT=$(realpath "$1"); N=${2:-40}
for k in a b; do
  D=$OUT/perf_$k.data
  [ -f "$D" ] || continue
  perf report -i $D --no-children --sort tid --stdio -g none 2>/dev/null | grep -v '^#' | grep -v '^$' > $OUT/perf_${k}_threads.txt
  perf report -i $D --no-children --sort sym --stdio -g none 2>/dev/null | grep -v '^#' | grep -v '^$' > $OUT/perf_${k}_self.txt
  perf report -i $D --children --sort sym --stdio -g none 2>/dev/null | grep -v '^#' | grep -v '^$' > $OUT/perf_${k}_incl.txt
  echo "=== host $k: threads"; head -12 $OUT/perf_${k}_threads.txt
  echo "=== host $k: self (top $N)"; head -$N $OUT/perf_${k}_self.txt
  echo "=== host $k: inclusive (top $N)"; head -$N $OUT/perf_${k}_incl.txt
done
