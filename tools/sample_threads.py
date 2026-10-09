#!/usr/bin/env python3
"""Sample every thread of a running process: state, kernel wait channel, CPU share.

    tools/sample_threads.py PID [seconds=10] [interval_ms=100]

Prints, per thread, the share of samples in each state/wchan and the CPU used. A thread that is "S" in
futex_wait or poll all the time while the capture backlog grows is waiting on something; one at 100% CPU is
computing. Needs no perf and no root (own processes).
"""
import collections
import os
import sys
import time

pid = int(sys.argv[1])
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 10
step = (float(sys.argv[3]) if len(sys.argv) > 3 else 100) / 1000
tck = os.sysconf("SC_CLK_TCK")
base = f"/proc/{pid}/task"


def rd(p):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return ""


def stat(tid):
    s = rd(f"{base}/{tid}/stat")
    if not s:
        return None
    name, rest = s[s.index("(") + 1 : s.rindex(")")], s[s.rindex(")") + 2 :].split()
    return name, rest[0], int(rest[11]) + int(rest[12])  # state, utime + stime ticks


seen = collections.defaultdict(collections.Counter)
cpu0, cpu1, names = {}, {}, {}
n = 0
end = time.monotonic() + secs
while time.monotonic() < end:
    for tid in os.listdir(base):
        st = stat(tid)
        if not st:
            continue
        name, state, ticks = st
        names[tid] = name
        cpu0.setdefault(tid, ticks)
        cpu1[tid] = ticks
        seen[tid][f"{state} {rd(f'{base}/{tid}/wchan').strip() or '-'}"] += 1
    n += 1
    time.sleep(step)

print(f"{n} samples over {secs:.0f} s")
for tid in sorted(seen, key=lambda t: -(cpu1[t] - cpu0[t])):
    print(f"{tid:>7} {names[tid]:<16} cpu {100 * (cpu1[tid] - cpu0[tid]) / tck / secs:5.1f}%")
    for k, c in seen[tid].most_common(3):
        print(f"          {100 * c / n:5.1f}%  {k}")
