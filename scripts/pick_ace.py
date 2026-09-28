"""Per submode, whether ACE beats today's plain clipping, PEP-fair, from
scripts/ace_study.py's CSV.

PEP-fair score per channel: 1% threshold + mean burst peak (dB), lower
better. Rule: among the ACE settings (closing, headroom) no worse than
today's plain setting on any channel, the one with the best mean gain;
adopted if that is at least MIN_GAIN. Prints the table and the config
lines; --json adds the chosen settings' clip constants to
codes_data/clip_constants.json (measured in the same study).

    uv run --no-sync python scripts/pick_ace.py runs/ace_study.csv [--json data2g/codes_data/clip_constants.json]
"""

import argparse
import csv
import json
from collections import defaultdict

from data2g.config import SUBMODES, clip_key

MIN_GAIN = 0.2  # dB, mean over channels
CLOSINGS = {"": (), "ace1": (1.0,), "ace1-1.5-2": (1.0, 1.5, 2.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--json")
    a = ap.parse_args()
    score = defaultdict(dict)  # (mode, closing, headroom) -> {channel: score}
    consts = {}
    for r in csv.DictReader(open(a.csv)):
        mode, _, var = r["submode"].partition("+")
        key = (mode, var, float(r["headroom"]))
        score[key][r["channel"]] = float(r["threshold_db"]) + float(r["peak_db"])
        consts[key] = dict(gain_1f=float(r["gain_1f"]), gain=float(r["gain"]), sdr_db=float(r["sdr_db"]),
                           papr_db=float(r["papr_db"]), peak_db=float(r["peak_db"]))
    chans = sorted({c for v in score.values() for c in v})
    picks = {}
    print("mode | today (headroom: PEP-fair per channel) | best ACE (closing, headroom: gain per channel, dB)")
    for mode in sorted({k[0] for k in score}):
        h0 = SUBMODES[mode].headroom
        base = score.get((mode, "", float(h0)))
        if not base or len(base) < len(chans):
            continue
        best = None
        for (m, var, hr), sc in score.items():
            if m != mode or not var or len(sc) < len(chans):
                continue
            gains = {c: base[c] - sc[c] for c in chans}
            if any(g < 0 or g != g for g in gains.values()):
                continue
            mean = sum(gains.values()) / len(gains)
            if best is None or mean > best[0]:
                best = (mean, var, hr, gains)
        base_s = " ".join(f"{c} {base[c]:.2f}" for c in chans)
        if best is None:
            print(f"{mode} | {h0:g} dB: {base_s} | none no worse on every channel")
            continue
        mean, var, hr, gains = best
        tag = "ADOPT" if mean >= MIN_GAIN else "keep plain"
        print(f"{mode} | {h0:g} dB: {base_s} | {var} {hr:g} dB: "
              + " ".join(f"{c} {g:+.2f}" for c, g in gains.items()) + f" (mean {mean:+.2f}) {tag}")
        if mean >= MIN_GAIN:
            picks[mode] = (var, hr)
    print("\nconfig:")
    for mode, (var, hr) in picks.items():
        print(f"  {mode}: headroom={hr:g}, ace={CLOSINGS[var]}")
    if a.json and picks:
        d = json.load(open(a.json))
        for mode, (var, hr) in picks.items():
            s = SUBMODES[mode]
            d["entries"][clip_key(s.band, hr, CLOSINGS[var], s.constellation)] = {
                k: round(v, 4 if k.startswith("gain") else 2) for k, v in consts[(mode, var, hr)].items()}
        with open(a.json, "w") as f:
            json.dump(d, f, indent=1)
        print(f"wrote {len(picks)} entries to {a.json}")


if __name__ == "__main__":
    main()
