"""Source-aware decoding of control codewords: what knowing a control's
zero padding (codes.zero_tail, codes.decode_raw_pinned) buys at code level.

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
    uv run python scripts/ctl_prior_study.py --polar --out runs/ctl_prior_polar.csv
    uv run python scripts/ctl_prior_study.py --content 40 --out runs/ctl_inline_40.csv   # a control carrying T_INLINE

--polar: the polar control codewords (CPM's, and the polar and ACK modes'),
BPSK LLRs per coded bit (Es/N0 per bit), cuts phy.POLAR_PIN_CUTS. A control
longer than one codeword is studied at its last codeword (the one with
padding), its cuts counted along the control stream as the receiver does.

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

from data2g import codes, constellation, cpm
from data2g.arq.modes import MODES, ctl_spec
from data2g.config import SUBMODES

CONTENT = 12  # bytes: the median control (LDPC in linksim, CPM's in real-modem sessions)
CUTS = {"ldpc": (16, 32), "polar": (12, 16)}  # phy.PIN_CUTS, POLAR_PIN_CUTS
POLAR = {s.name: s for s in [*cpm.CTL.values(), *(ctl_spec(m) for m in MODES.values())] if s.code == "polar"}
TRIALS = 200
STEP = 0.5


def soft_at(spec, cw: np.ndarray, esn0: float, rng) -> np.ndarray:
    if spec.code == "polar":  # BPSK per coded bit (CPM's soft bits are not a QAM's)
        sigma = np.sqrt(1 / (2 * 10 ** (esn0 / 10)))
        return 2 * ((1 - 2.0 * cw) + sigma * rng.normal(size=cw.shape)) / sigma**2
    pts = constellation.load(spec.constellation)
    x = constellation.modulate(cw.reshape(-1), pts).reshape(len(cw), -1)
    var = np.full(x.shape, 10 ** (-esn0 / 10))
    y = x + np.sqrt(var / 2) * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape))
    return constellation.llr(y, np.ones_like(y), var, pts).reshape(len(cw), -1)


def failures(spec, esn0: float, seed: int, content: int) -> dict:
    """Failed codewords out of TRIALS per scheme at one Es/N0."""
    rng = np.random.default_rng(seed)
    pb, mask = codes.payload_bytes(spec), 0x5A5A
    i = (content - 1) // pb if content else 0  # the control's last codeword (its padding)
    n = content - i * pb if content else pb
    offs = [c - i * pb for c in CUTS[spec.code] if 0 < c - i * pb < pb]
    pays = [bytes(rng.integers(0, 256, n, dtype=np.uint8)) + bytes(pb - n) for _ in range(TRIALS)]
    idx = rng.integers(0, 4, TRIALS)  # control slots 0-3
    cw = np.stack([codes.encode(spec, p, 0, mask, i) for p, i in zip(pays, idx)])
    soft = soft_at(spec, cw, esn0, rng)

    def ok(s, rows):
        cands, usable = codes.decode_raw(spec, s, index=idx[rows])
        return np.array([codes.check(spec, c, u, mask) == pays[r] for c, u, r in zip(cands, usable, rows)])

    def ok_known(rows, nbytes):
        out = []
        for r in rows:
            c, u = codes.decode_raw_pinned(spec, soft[r][None], codes.zero_tail(spec, nbytes), idx[r])
            out.append(codes.check(spec, c[0], u[0], mask) == pays[r])
        return np.array(out, bool)

    rows = np.arange(TRIALS)
    good = ok(soft, rows)
    out = {"plain": int((~good).sum())}
    if not content:
        return {"data": out["plain"]}
    for cut in CUTS[spec.code]:
        left = rows[~good]
        if cut - i * pb in offs and len(left):
            good[left] |= ok_known(left, cut - i * pb)
        out[str(cut)] = int((~good).sum())
    out["genie"] = int((~ok_known(rows, n)).sum())
    return out


def sweep(spec, content: int, lo: float, seed: int, target=0.1) -> dict:
    """Each scheme's lowest Es/N0 on the STEP grid from `lo` where it fails
    <= target (the grid walks up until every scheme has passed)."""
    out, e = {}, lo
    while e < 40:
        for scheme, fails in failures(spec, e, seed, content).items():
            if scheme not in out and fails <= target * TRIALS:
                out[scheme] = e
        if len(out) == (1 if not content else 2 + len(CUTS[spec.code])):
            return out
        e += STEP
    return out


def study(name: str, content: int = CONTENT) -> dict:
    spec = SUBMODES.get(name) or POLAR[name]
    seed = 1000 + spec.index
    data = sweep(spec, 0, -12.0, seed)["data"]
    row = {"mode": name, "payload_bytes": codes.payload_bytes(spec), "k": spec.k, "data": data}
    th = sweep(spec, content, data - 8.0, seed)
    cuts = map(str, CUTS[spec.code])
    return row | {s: th.get(s, float("nan")) for s in ("plain", *cuts, "genie")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ctl_prior_study.csv")
    ap.add_argument("--modes", nargs="*", default=None)
    ap.add_argument("--polar", action="store_true", help="the polar control codewords (see the docstring)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--content", type=int, default=CONTENT,
                    help="control bytes (a control with T_INLINE: its data in the padding, 5 B more)")
    a = ap.parse_args()
    modes = a.modes or (sorted(POLAR) if a.polar else [n for n, s in SUBMODES.items() if s.code == "ldpc"])
    with Pool(a.jobs) as pool:
        rows = pool.starmap(study, [(m, a.content) for m in modes])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(" ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
