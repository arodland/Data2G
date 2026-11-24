"""FM ladder pass: the rung constellations trained by scripts/fm_const_train.sh,
each through the streamed ladder study (scripts/fm_ladder_study.py, FM_TX=1)
as its rung's mode with only the constellation swapped, against that mode's
row in runs/fm_ladder/ladder_tx.csv (same trials, seeds and conditions:
paired). Kept: at least 0.2 dB better at 10% on static or flutter at 440,
and no more than 0.2 dB worse on any condition (retest_constellations' rule).

    DATA2G_PROFILE=fm FM_TX=1 python scripts/fm_const_retest.py run [--workers 4]
    python scripts/fm_const_retest.py report
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
CONST = ROOT / "data2g" / "constellations"
# trained name -> (radio, the ladder's mode it replaces the constellation of)
RUNGS = {
    "c16-f55m-r47": ("mic", "f55:gray-qam16:2:1040:3"),
    "c64-f55m-r50": ("mic", "f55:c64-w48-r12:2:1648:6"),
    "c64-f55m-r67": ("mic", "f55:c64-w48-r12:2:2216:6"),
    "c64-f55m-r86": ("mic", "f55:c64-w48-r34:2:2824:6"),
    "c256-f55m-r67": ("mic", "f55:c256-w48-r12:1:1480:6"),
    "c256-f55m-r72": ("mic", "f55:c256-w48-r12:1:1584:6"),
    "c16-f55f-r67": ("flat", "f55:gray-qam16:2:1480:6"),
    "c256-f55f-r47": ("flat", "f55:c256-w48-r12:1:1040:6"),
    "c256-f55f-r60": ("flat", "f55:c256-w48-r12:1:1320:6"),
    "c256-f55f-r67": ("flat", "f55:c256-w48-r12:1:1480:6"),
    "c256-f55f-r72": ("flat", "f55:c256-w48-r12:1:1584:6"),
    "c64-f108f-r40": ("flat", "f108:c64-w48-r12:1:1296:6"),
    "c64-f108f-r50": ("flat", "f108:c64-w48-r12:1:1616:6"),
    "c64-f108f-r59": ("flat", "f108:c64-w48-r12:1:1920:6"),
    "c64-f108f-r67": ("flat", "f108:c64-w48-r23:1:2176:6"),
    "c256-f108f-r59": ("flat", "f108:c256-w48-r12:1:2560:6"),
    "c256-f108f-r67": ("flat", "f108:c256-w48-r12:1:2896:6"),
    "c256-f108f-r75": ("flat", "f108:c256-w48-r58:1:3240:6"),
    "c256-f108f-r80": ("flat", "f108:c256-w48-r58:1:3456:6"),
}
CONDS = ("static", "flutter@440", "mobile-rural")


def swapped(name: str) -> str:
    band, _, frames, k, hr = RUNGS[name][1].split(":")
    return f"{band}:{name}:{frames}:{k}:{hr}"


def run(a):
    for name in RUNGS:  # the sets load by name (constellation.load): new names only, never an existing one
        dst = CONST / f"{name}.npy"
        assert not dst.exists() or dst.read_bytes() == (RUNS / "fm_const" / f"{name}.npy").read_bytes(), dst
        shutil.copy(RUNS / "fm_const" / f"{name}.npy", dst)
    for radio in ("mic", "flat"):
        modes = [swapped(n) for n, (r, _) in RUNGS.items() if r == radio]
        subprocess.run([sys.executable, str(ROOT / "scripts" / "fm_ladder_study.py"), "run", "--modes", *modes,
                        "--radios", radio, "--conditions", *CONDS, "--trials", "30", "--workers", str(a.workers),
                        "--out", f"const_{radio}"], check=True, env={**os.environ, "FM_TX": "1", "DATA2G_PROFILE": "fm"})


def knees(path: Path) -> dict:
    with path.open() as f:
        return {(r["mode"], r["radio"], r["condition"]): float(r["k0.1"]) for r in csv.DictReader(f)}  # a mode runs on both radios


def report(_a):
    before = knees(RUNS / "fm_ladder" / "ladder_tx.csv")
    after = {}
    for radio in ("mic", "flat"):
        p = RUNS / "fm_ladder" / f"const_{radio}.csv"
        if p.exists():
            after.update(knees(p))
    print("rung | radio | 10% knee before -> after (static / flutter@440 / mobile-rural), dB | kept")
    for name, (radio, mode) in RUNGS.items():
        b = [before.get((mode, radio, c), float("nan")) for c in CONDS]
        n = [after.get((swapped(name), radio, c), float("nan")) for c in CONDS]
        gain = [x - y for x, y in zip(b, n)]  # positive: the trained set is better
        keep = max(gain[0], gain[1]) >= 0.2 and min(gain) >= -0.2
        cells = " / ".join(f"{x:.2f}->{y:.2f}" for x, y in zip(b, n))
        print(f"{name} | {radio} | {cells} | {'kept' if keep else 'no'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "report"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    (run if a.what == "run" else report)(a)
    if a.what == "run":
        report(a)


if __name__ == "__main__":
    main()
