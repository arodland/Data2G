"""Merge the band ladders into one CSV for prune.py: the 1200 Hz band's
fading rows from runs/ladder.csv with its AWGN rows replaced by the PER
rerun (runs/ladder_awgn_per_all.csv; the original AWGN column was BER
1e-6), plus the narrow and 2400 Hz ladders, which used PER throughout.

    uv run python scripts/combine_ladders.py && uv run python scripts/prune.py runs/ladder_all.csv
"""

import csv

rows = [r for r in csv.DictReader(open("runs/ladder.csv")) if r["channel"] != "awgn"]
rows += list(csv.DictReader(open("runs/ladder_awgn_per_all.csv")))
for extra in ("runs/ladder_narrow.csv", "runs/ladder_w48.csv"):
    rows += list(csv.DictReader(open(extra)))
with open("runs/ladder_all.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> runs/ladder_all.csv")
