"""Source-aware decoding of control codewords: what pinning a control's
zero padding (codes.zero_tail, codes.pin) buys at code level.

A control is a few bytes (median 12, at most 28 in linksim sessions over
bulk, winlink and chat) zero-padded to its codeword's payload, 46-396 B in
the LDPC modes. The receiver decodes plain first; only a failed slot is
decoded again with the payload past `cut` bytes pinned to zero, for each
cut in turn. Reported per mode: the Es/N0 (dB, per constellation symbol,
AWGN through the mode's constellation, exact LLRs) where 10% of control
codewords fail, for
  data   a full random payload (what the data slots face)
  plain  a control, decoded as today
  <cut>  plain, then pinned past each cut up to that one, CRC-checked
  genie  pinned past the content's exact length (the bound)

    uv run python scripts/ctl_prior_study.py --out runs/ctl_prior_study.csv

Found (2026-10-18, 200 codewords a point, 0.5 dB steps): plain controls fail
where data does (within 0.5 dB). Pinned, the 10% point moves 1.0-6.0 dB
lower, median 4.0 over the 36 LDPC modes: 1.0 in n4/n10 qpsk-r1/5 (22-23 B
payloads, little padding), 3.5-4.5 in most qpsk and 16qam modes, 5-6 at
high rates (qpsk-r3/4, w48-16qam-r3/4 and r5/6, the 64l and 256l modes).
Cut 16 gets all of it for a 12 B control (32 is for longer ones); the
genie is at most 1 dB better, and only in the smallest payloads.
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
from multiprocessing import Pool

import numpy as np

from data2g import codes, constellation
from data2g.config import SUBMODES

CONTENT = 12  # bytes: the median control
CUTS = (16, 32)
TRIALS = 200
STEP = 0.5


def soft_at(spec, cw: np.ndarray, esn0: float, rng) -> np.ndarray:
    pts = constellation.load(spec.constellation)
    x = constellation.modulate(cw.reshape(-1), pts).reshape(len(cw), -1)
    var = np.full(x.shape, 10 ** (-esn0 / 10))
    y = x + np.sqrt(var / 2) * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape))
    return constellation.llr(y, np.ones_like(y), var, pts).reshape(len(cw), -1)


def failures(spec, esn0: float, seed: int, content: int) -> dict:
    """Failed codewords out of TRIALS per scheme at one Es/N0."""
    rng = np.random.default_rng(seed)
    pb, mask = codes.payload_bytes(spec), 0x5A5A
    n = min(content, pb) if content else pb
    pays = [bytes(rng.integers(0, 256, n, dtype=np.uint8)) + bytes(pb - n) for _ in range(TRIALS)]
    idx = rng.integers(0, 4, TRIALS)  # control slots 0-3
    cw = np.stack([codes.encode(spec, p, 0, mask, i) for p, i in zip(pays, idx)])
    soft = soft_at(spec, cw, esn0, rng)

    def ok(s, rows):
        cands, usable = codes.decode_raw(spec, s, index=idx[rows])
        return np.array([codes.check(spec, c, u, mask) == pays[r] for c, u, r in zip(cands, usable, rows)])

    rows = np.arange(TRIALS)
    good = ok(soft, rows)
    out = {"plain": int((~good).sum())}
    if not content:
        return {"data": out["plain"]}
    for cut in CUTS:
        left = rows[~good]
        if cut < pb and len(left):
            pinned = np.stack([codes.pin(spec, soft[r], codes.zero_tail(spec, cut), idx[r]) for r in left])
            good[left] |= ok(pinned, left)
        out[str(cut)] = int((~good).sum())
    pinned = np.stack([codes.pin(spec, soft[r], codes.zero_tail(spec, n), idx[r]) for r in rows])
    out["genie"] = int((~ok(pinned, rows)).sum())
    return out


def sweep(spec, content: int, lo: float, seed: int, target=0.1) -> dict:
    """Each scheme's lowest Es/N0 on the STEP grid from `lo` where it fails
    <= target (the grid walks up until every scheme has passed)."""
    out, e = {}, lo
    while e < 40:
        for scheme, fails in failures(spec, e, seed, content).items():
            if scheme not in out and fails <= target * TRIALS:
                out[scheme] = e
        if len(out) == (1 if not content else 2 + len(CUTS)):
            return out
        e += STEP
    return out


def study(name: str) -> dict:
    spec = SUBMODES[name]
    seed = 1000 + spec.index
    data = sweep(spec, 0, -8.0, seed)["data"]
    row = {"mode": name, "payload_bytes": codes.payload_bytes(spec), "k": spec.k,
           "pinned_bits_32": int(((codes.zero_tail(spec, 32) >= 0) & (codes.systematic_positions(spec) >= 0)).sum()),
           "data": data}
    th = sweep(spec, CONTENT, data - 8.0, seed)
    return row | {s: th.get(s, float("nan")) for s in ("plain", *map(str, CUTS), "genie")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ctl_prior_study.csv")
    ap.add_argument("--modes", nargs="*", default=[n for n, s in SUBMODES.items() if s.code == "ldpc"])
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    with Pool(a.jobs) as pool:
        rows = pool.map(study, a.modes)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(" ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
