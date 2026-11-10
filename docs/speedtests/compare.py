"""A speed sweep's reports (scripts/cpu_profile/speed_sweep.sh) against the
Data2G values in make_charts.py: per point the new B/min, the old, the
change, and transfers completed. `--arrays` also prints the new values as
make_charts.py's "data2g" lists, to paste in.

    python docs/speedtests/compare.py runs/speedtrials [--flag 5] [--arrays]
"""

import argparse
import re
from pathlib import Path

from make_charts import BW2300, BW500, SNRS


def read(report: Path):
    """-> (B/min, transfers completed, transfers run), or None if unfinished."""
    text = report.read_text()
    ok = re.search(r": (\d+)/(\d+) succeeded", text)
    bpm = re.search(r"^mean of .* = (\d+) B/min", text, re.M)
    if not ok:
        return None
    return int(bpm.group(1)) if bpm else 0, int(ok.group(1)), int(ok.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--flag", type=float, default=5.0, help="mark changes past this many percent")
    ap.add_argument("--arrays", action="store_true")
    a = ap.parse_args()
    for bw, name, table in (("2300", "2400", BW2300), ("500", "500", BW500)):
        print(f"BW{bw}: B/min new | old | change | completed")
        new_arrays = {}
        for chan, series in table.items():
            old = {p[0]: p[1] for p in series["data2g"]}
            new_arrays[chan] = []
            for snr in SNRS:
                f = a.runs / f"{name}_{chan.lower()}_{snr}dB.txt"
                got = read(f) if f.exists() else None
                if got is None:
                    print(f"  {chan:4s} {snr:3d} dB: missing")
                    continue
                bpm, ok, n = got
                pct = 100.0 * (bpm - old[snr]) / old[snr] if snr in old else float("nan")
                mark = "  <--" if abs(pct) >= a.flag or ok < 5 else ""
                print(f"  {chan:4s} {snr:3d} dB: {bpm:6d} | {old.get(snr, 0):6d} | {pct:+6.1f}% | {ok}/{n}{mark}")
                new_arrays[chan].append([snr, bpm] if ok >= 5 else [snr, bpm, ok, f"{ok} of {n} transfers completed"])
        if a.arrays:
            for chan, pts in new_arrays.items():
                print(f'    "{chan}": "data2g": {pts},')
        print()


if __name__ == "__main__":
    main()
