"""Pick each submode's clip headroom from a clip_study CSV.

Rule (decided 2026-09-23): the LOWEST headroom whose PEP-fair score
(threshold + peak-to-average, dB) is within TOLERANCE of that submode's
best on every channel measured. Amplifiers are not linear, so on a
near-flat curve the lower-PAPR end is worth up to half a dB of PEP-fair
SNR, as SSTVAE chose for its own clipper.

    uv run python scripts/pick_headroom.py runs/clip_w48.csv
"""

import csv
import sys
from collections import defaultdict

TOLERANCE = 0.5
INF = float("inf")


def load(path) -> dict:
    """submode -> {(headroom, channel): PEP-fair score}"""
    score = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        score[r["submode"]][(float(r["headroom"]), r["channel"])] = float(r["threshold_db"]) + float(r["peak_db"])
    return score


def pick(sc: dict) -> float | None:
    """The rule, for one submode's scores."""
    chans = sorted({c for _, c in sc})
    heads = sorted({h for h, _ in sc})
    best = {c: min(sc.get((h, c), INF) for h in heads) for c in chans}
    ok = [h for h in heads if all((h, c) in sc and sc[(h, c)] <= best[c] + TOLERANCE for c in chans)]
    return min(ok) if ok else None


def picks(path) -> dict:
    return {sub: pick(sc) for sub, sc in load(path).items()}


def main():
    score = load(sys.argv[1])
    for sub, sc in score.items():
        chans = sorted({c for _, c in sc})
        heads = sorted({h for h, _ in sc})
        best = {c: min(sc.get((h, c), INF) for h in heads) for c in chans}
        pick_ = pick(sc)
        cells = "  ".join(
            f"{h:g}dB:" + "/".join(f"{sc.get((h, c), INF) - best[c]:+.2f}" for c in chans) for h in heads
        )
        print(f"{sub:12s} pick {pick_ if pick_ is not None else '-':>4}   excess over best ({'/'.join(chans)}): {cells}")


if __name__ == "__main__":
    main()
