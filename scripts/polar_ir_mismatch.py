"""Where can polar IR (codes.polar_ir_code) lose to Chase? Code level,
BPSK on AWGN, CA-SCL with the real CRC, native decoders (tools/build_native.sh).

  curve    RV 0 and the resend at the same Es/N0, down to low FER
  unequal  RV 0 at a fixed Es/N0, the resend `delta` dB away (fading:
           one of the two transmissions in a fade)

Chase adds a second RV 0; IR decodes RV 0 + RV 1 as codes.decode_buffer
does. "one" is RV 0 alone (what the receiver had before the resend).

    uv run python scripts/polar_ir_mismatch.py curve --out runs/polar_ir_curve.csv
    uv run python scripts/polar_ir_mismatch.py unequal --out runs/polar_ir_unequal.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "native" / "build" / "python"))
import data2g_native as N  # noqa: E402

from data2g import codes, cpm  # noqa: E402
from data2g.config import SUBMODES  # noqa: E402


def specs():
    out = {s.name: s for s in SUBMODES.values() if s.code == "polar"}
    out["cpm-ctl"] = next(iter(cpm.CTL.values()))
    return out


class Codes:
    def __init__(self, spec):
        self.spec = spec
        if spec.name in SUBMODES:
            self.base = N.polar.polar_code(spec.name)
        else:
            self.base = N.polar.PolarCode(spec.k, spec.coded_bits)
        self.ir = N.polar.PolarCode.ir(self.base, N.polar.ir_copies(spec.k, spec.coded_bits))
        self.dec_base = N.polar.SCLDecoder(self.base, codes.POLAR_LIST)
        self.dec_ir = N.polar.SCLDecoder(self.ir, codes.POLAR_LIST)

    def ok(self, dec, llr, bits):
        paths, _ = dec.decode(np.asarray(llr, np.float32))
        B, L, k = paths.shape
        crc = np.asarray(N.codes.crc_ok(self.spec.name, paths.reshape(-1, k), np.zeros(B * L, np.uint32),
                                        np.zeros(B * L, np.int32))).reshape(B, L).astype(bool)
        pick = np.where(crc.any(1), crc.argmax(1), 0)
        return crc.any(1) & (paths[np.arange(B), pick] == bits).all(1)

    def trial(self, esn0_0, esn0_1, n, rng):
        spec = self.spec
        bits = np.stack([codes.info_bits(spec, rng.bytes(codes.payload_bytes(spec))) for _ in range(n)])
        x = 1.0 - 2.0 * np.asarray(self.ir.encode(bits))
        E = spec.coded_bits
        s0, s1 = (np.sqrt(1 / (2 * 10 ** (e / 10))) for e in (esn0_0, esn0_1))
        l0 = 2 * (x[:, :E] + rng.normal(scale=s0, size=(n, E))) / s0**2
        l1 = 2 * (x[:, E:] + rng.normal(scale=s1, size=(n, E))) / s1**2
        l0b = 2 * (x[:, :E] + rng.normal(scale=s1, size=(n, E))) / s1**2  # Chase: RV 0 again, same SNR as RV 1
        return dict(one=self.ok(self.dec_base, l0, bits).mean(),
                    chase=self.ok(self.dec_base, l0 + l0b, bits).mean(),
                    ir=self.ok(self.dec_ir, np.concatenate([l0, l1], 1), bits).mean())


def chase_point(c, target, rng, lo=-20.0, hi=10.0):
    """Es/N0 where Chase's FER crosses `target` (bisection)."""
    for _ in range(8):
        mid = (lo + hi) / 2
        if 1 - c.trial(mid, mid, 300, rng)["chase"] > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("curve", "unequal"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default=",".join(specs()))
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    rows = []
    for name in a.modes.split(","):
        c = Codes(specs()[name])
        rng = np.random.default_rng(1)
        ref = chase_point(c, 0.5, rng)  # Chase's 50% point
        if a.what == "curve":
            grid = [(ref + d, ref + d) for d in np.arange(-1.5, 4.01, 0.5)]
        else:
            grid = [(ref + d0, ref + d0 + d) for d0 in (0.0, 2.0) for d in (-12, -9, -6, -4, -2, 0, 2, 4)]
        for e0, e1 in grid:
            r = c.trial(e0, e1, a.n, rng)
            row = dict(mode=name, esn0_rv0=round(e0, 2), esn0_resend=round(e1, 2), n=a.n,
                       **{f"fer_{k}": round(1 - v, 5) for k, v in r.items()})
            print(row, flush=True)
            rows.append(row)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
