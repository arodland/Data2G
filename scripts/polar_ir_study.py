"""Polar IR-HARQ (polar.IRPolarCode, arXiv:1708.09679) against Chase, at
the code level: BPSK on AWGN (gray-qam4 is two of them), CA-SCL with the
codeword's real CRC. Per polar code and transmission scheme, the per-bit
Es/N0 where the frame error rate crosses 50% and 10%, by bisection.

  one    RV 0 alone
  chase  RV 0 twice, LLRs added (what polar resends did before)
  ir@d   RV 0 + RV 1, the extension designed at d dB

    uv run python scripts/polar_ir_study.py --out runs/polar_ir_study.csv
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np

from data2g import codes, cpm, polar
from data2g.config import SUBMODES

TRIALS = 400


def specs():
    out = {s.name: s for s in SUBMODES.values() if s.code == "polar"}
    out["cpm-ctl"] = next(iter(cpm.CTL.values()))
    return out


def fer(spec, scheme: str, esn0_db: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    base = codes.polar_code(spec)
    payloads = [rng.bytes(codes.payload_bytes(spec)) for _ in range(TRIALS)]
    bits = np.stack([codes.info_bits(spec, p) for p in payloads])
    sig = np.sqrt(1 / (2 * 10 ** (esn0_db / 10)))
    if scheme.startswith("ir@"):
        code = polar.IRPolarCode(base, float(scheme[3:]))
    else:
        code = base
    x = 1.0 - 2.0 * code.encode(bits)
    llr = 2 * (x + rng.normal(scale=sig, size=x.shape)) / sig**2
    if scheme == "chase":
        llr = llr + 2 * (x + rng.normal(scale=sig, size=x.shape)) / sig**2
    paths, _ = polar.SCLDecoder(code, codes.POLAR_LIST).decode(llr)
    ok = codes.crc_ok(spec, paths.reshape(-1, spec.k), 0, 0).reshape(paths.shape[:2])
    pick = np.where(ok.any(1), ok.argmax(1), 0)
    good = ok.any(1) & (paths[np.arange(TRIALS), pick] == bits).all(1)
    return 1.0 - good.mean()


def threshold(spec, scheme: str, target: float, lo=-16.0, hi=8.0, steps=9) -> float:
    for i in range(steps):
        mid = (lo + hi) / 2
        if fer(spec, scheme, mid, 1000 * i + int(mid * 100) + 5000) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def job(args):
    name, scheme = args
    spec = specs()[name]
    row = dict(mode=name, k=spec.k, e=spec.coded_bits, scheme=scheme)
    if scheme.startswith("ir@"):
        row["copies"] = len(polar.IRPolarCode(codes.polar_code(spec), float(scheme[3:])).copies)
    for t in (0.5, 0.1):
        row[f"esn0_{int(t * 100)}"] = round(threshold(spec, scheme, t), 2)
    print(row, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/polar_ir_study.csv")
    ap.add_argument("--design", default="-12,-9,-6,-3,0")
    ap.add_argument("--modes", default=",".join(specs()))
    a = ap.parse_args()
    schemes = ["one", "chase"] + [f"ir@{d}" for d in a.design.split(",")]
    jobs = [(m, s) for m in a.modes.split(",") for s in schemes]
    with Pool(os.cpu_count()) as p:
        rows = p.map(job, jobs)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, ["mode", "k", "e", "scheme", "copies", "esn0_50", "esn0_10"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
