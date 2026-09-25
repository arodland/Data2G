"""Freeze the computed parts of the on-air format as committed data.

Two things are computed rather than tabulated, and either could change
with a numpy version or a platform: each submode's bit interleaver (a
seeded permutation plus a Monte Carlo label-reliability ranking) and a
polar submode's info set (floating-point density evolution). As in
SSTVAE, what goes on the air must be written down, so this writes them
to data2g/format/<band>_<index>.npz, each stamped with the submode
parameters it was built for (codes.frozen ignores a stale one).

    uv run python tools/freeze_format.py            # (re)write every submode's file
    uv run python tools/freeze_format.py --verify   # recompute and compare, change nothing

--verify is information, not a gate: a mismatch means this numpy would
now produce a different format, and the committed file is still right.
"""

import argparse
import sys

import numpy as np

from data2g import codes
from data2g.config import SUBMODES


def compute(spec) -> dict:
    out = {"fingerprint": codes._fingerprint(spec), "perm": codes.compute_interleaver(spec)}
    if spec.code == "polar":
        from data2g import polar

        out["info_pos"] = polar.PolarCode(spec.k, spec.coded_bits,
                                          design_snr_db=codes.POLAR_DESIGN_SNR_DB).info_pos
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    codes.FORMAT_DIR.mkdir(exist_ok=True)
    bad = 0
    for spec in SUBMODES.values():
        path = codes.FORMAT_DIR / f"{spec.band}_{spec.index:02d}.npz"
        new = compute(spec)
        if a.verify:
            old = codes.frozen(spec)
            same = old is not None and all(np.array_equal(old[k], new[k]) for k in new if k != "fingerprint")
            bad += not same
            print(f"{'ok  ' if same else 'DIFF'} {path.name} {spec.name}")
        else:
            np.savez(path, **new)
            print(f"wrote {path.name} {spec.name}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
