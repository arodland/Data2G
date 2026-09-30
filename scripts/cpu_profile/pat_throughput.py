"""Throughput of W1AW's message payload in a run.sh pat phase, from the logs.

    python pat_throughput.py <out dir>

B2F is half duplex, so host_a.log alternates turns of new data: K2XYZ's
banner, W1AW's proposals, K2XYZ's FS, then W1AW's messages. The window is
that 4th turn: from its first data burst's TX to the RX that acks its last
codeword. Excludes connect, handshake, proposals, B's turn and disconnect.
Bytes: "wire" is the new codeword bytes data2g carried in the window (B2F
blocks, framing included); "B2F" and "orig" are the compressed and
uncompressed sizes from W1AW's FD proposals in pat/a/pat.log."""

import re
import sys
from datetime import datetime

LINE = re.compile(r"^(\S+ \S+) INFO (TX|RX) b\d (.*)$")


def window(lines):
    """(start, end, wire bytes) of A's second turn of new data."""
    turns, side = 0, None  # A's turns seen so far; who sent new data last
    start = last_seq = end = None
    wire = 0
    for line in lines:
        m = LINE.match(line)
        if not m:
            continue
        t = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S,%f")
        d, rest = m[2], m[3]
        if d == "TX" and (n := re.search(r"\| new (\d+)(?:-(\d+))? (\d+) B", rest)):
            if side != "A":
                side, turns = "A", turns + 1
            if turns == 2:
                start, end = start or t, None  # more outstanding
                last_seq = int(n[2] or n[1])
                wire += int(n[3])
            elif turns > 2:
                break
        elif d == "RX":
            if (c := re.search(r"cum (\d+)->(\d+)", rest)) and int(c[2]) > int(c[1]):
                side = "B"
            a = re.search(r"acked \d+->(\d+)", rest)
            if turns == 2 and end is None and a and int(a[1]) > last_seq:
                end = t
    if end is None:
        sys.exit("no complete payload turn in host_a.log")
    return start, end, wire


def main():
    out = sys.argv[1]
    start, end, wire = window(open(f"{out}/host_a.log", errors="replace"))
    fd = [l.split() for l in open(f"{out}/pat/a/pat.log") if l.startswith(">FD ")]
    orig, b2f = sum(int(f[3]) for f in fd), sum(int(f[4]) for f in fd)
    s = (end - start).total_seconds()
    print(f"A payload {start:%H:%M:%S}-{end:%H:%M:%S} {s:.1f} s: "
          + ", ".join(f"{k} {b} B = {b * 60 / s:.0f} B/min" for k, b in (("wire", wire), ("B2F", b2f), ("orig", orig))))


if __name__ == "__main__":
    main()
