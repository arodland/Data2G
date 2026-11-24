"""FM ladder pass: BMI cross-evaluation of the trained rung constellations
(scripts/fm_const_train.sh), static, FM_TX transmitter, through fmtorch
(fm_ladder_screen._bmi: drive re-picked per set):
- per family and band/radio, every trained set at every rung's static knee:
  does one set per family capture what the per-rate ones gain?
- gray QAM at each rung's knee (the learned sets' gain over it).
In bits per cell; at these knees about 0.3-0.7 bit per dB.

    DATA2G_PROFILE=fm FM_TX=1 python scripts/fm_const_cross.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_const_retest as R
import fm_ladder_screen as S

KNEE = {"c16-f55m-r47": 5.25, "c64-f55m-r50": 7.0, "c64-f55m-r67": 8.5, "c64-f55m-r86": 12.5, "c256-f55m-r67": 14.25,
        "c256-f55m-r72": 15.75, "c16-f55f-r67": 5.75, "c256-f55f-r47": 7.5, "c256-f55f-r60": 9.0,
        "c256-f55f-r67": 10.0, "c256-f55f-r72": 11.0, "c64-f108f-r40": 8.5, "c64-f108f-r50": 10.0,
        "c64-f108f-r59": 11.5, "c64-f108f-r67": 13.0, "c256-f108f-r59": 15.5, "c256-f108f-r67": 17.5,
        "c256-f108f-r75": 19.75, "c256-f108f-r80": 21.25}
BURSTS = 6


def bmi(const: str, rung: str) -> float:
    radio, mode = R.RUNGS[rung]
    band, _, frames, k, hr = mode.split(":")
    spec = S.register(f"{band}:{const}:{frames}:{k}:{hr}")
    return S._bmi(spec, radio, "static", 146.0, KNEE[rung], None, BURSTS)[0]


def main():
    groups = {}
    for n in R.RUNGS:  # family, band, radio
        groups.setdefault((n.split("-")[0], n.split("-")[1]), []).append(n)
    for (fam, br), rungs in groups.items():
        cur = {r: R.RUNGS[r][1].split(":")[1] for r in rungs}
        print(f"\n{fam} {br}: rows = set, columns = rung knee (bits/cell, static)")
        print("set".ljust(22) + "".join(r.split("-")[-1].rjust(9) for r in rungs))
        rows = [("gray-qam" + fam[1:], None)] + [(c, None) for c in sorted(set(cur.values()))] + [(r, r) for r in rungs]
        for const, _ in rows:
            print(const.ljust(22) + "".join(f"{bmi(const, r):9.4f}" for r in rungs), flush=True)


if __name__ == "__main__":
    main()
