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

--boundary 12 48 ...: per control length, the methods for finding where the
padding starts (pad_scores) against fixed cuts (A) and the exact length (F),
paired on the same noise, the control's last codeword, 400 trials (--trials)
on a 0.5 dB grid down from data's 10% point; --fading for i.i.d. Rayleigh with
the gains known. Per point: failures, decodes past plain and wrong codewords
passing the CRC per method, and each failed row's estimate error. Methods:
B<d>p/B1c a hard cut d bytes past S's argmax (posterior's, channel's), C1p
then the runner-up, D/Dc soft priors from softmax(S) (shipped: Dc,
codes.pad_prior), D2/Dc2 a second pass from D's posterior.

    python tools/with_native.py scripts/ctl_prior_study.py --boundary 12 16 24 32 48 64 96 128 200 264 --out x.csv

Found (2026-10-27, 36 modes, AWGN and Rayleigh): Dc 0.0 dB median from F at
every length at 10% and 1%, 0.6 at most at 10%, 0.8 at 1%, one decode; A as
good to 32 B, 1.2-3.5 dB median worse from 48 B on; B1 0.1-0.4 median, up to
2.4 at 1%; Dc2 at most 0.2 dB better, 1.13 decodes. No false passes.

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


# --- boundary estimates (--boundary) ------------------------------------------------
# Instead of fixed cuts, estimate where the padding starts from the failed
# decode and pin from there. Per trial, the control's last codeword, paired
# over every method (same noise).

def pad_scores(spec, info_llr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(B, k) LLRs of the info bits, scrambled, 0 where not sent -> (B, pb + 1)
    S[j] = sum over the known bits of bytes j.. of log 2 P(bit = its value
    if padding)."""
    pb = codes.payload_bytes(spec)
    s = np.stack([codes.scrambler(spec.k, codes.scramble_seed(int(i)))[: 8 * pb] for i in idx])
    t = np.log(2) - np.logaddexp(0, -info_llr[:, : 8 * pb] * (1 - 2.0 * s))
    t[:, codes.systematic_positions(spec)[: 8 * pb] < 0] = 0
    per = t.reshape(len(t), pb, 8).sum(-1)
    return np.concatenate([np.cumsum(per[:, ::-1], axis=1)[:, ::-1], np.zeros((len(t), 1))], axis=1)


def first_sent_byte(spec) -> int:
    return int(np.flatnonzero(codes.systematic_positions(spec) >= 0)[0]) // 8


def best_cut(S: np.ndarray, b0: int) -> np.ndarray:
    """argmax of S over j in max(1, b0)..pb, ties to the latest j (pin the least)."""
    pb = S.shape[1] - 1
    lo = max(1, b0)
    return pb - S[:, pb : lo - 1 : -1].argmax(1) if lo > 0 else None


def info_llrs(spec, code_llr: np.ndarray) -> np.ndarray:
    pos = codes.systematic_positions(spec)
    return np.where(pos >= 0, code_llr[:, np.maximum(pos, 0)], 0.0)


def soft_prior(spec, soft: np.ndarray, S: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """D: the boundary posterior softmax(S) over j = 1..pb gives each payload
    bit P(padding) = P(boundary <= its byte); its soft bit gets the prior LLR
    toward the known value (content bits uniform) added."""
    pb = codes.payload_bytes(spec)
    z = S[:, 1:] - S[:, 1:].max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    pad = np.concatenate([np.zeros((len(S), 1)), np.cumsum(p, 1)[:, :-1]], axis=1)  # byte b: P(j <= b), j >= 1
    prior = 2 * np.arctanh(np.clip(pad, 0, 1 - 1e-7))
    pos = codes.systematic_positions(spec)[: 8 * pb]
    sel = pos >= 0
    out = np.array(soft, dtype=np.float32)
    for r, i in enumerate(idx):
        sgn = 1 - 2.0 * codes.scrambler(spec.k, codes.scramble_seed(int(i)))[: 8 * pb]
        pr = np.repeat(prior[r], 8) * sgn
        out[r, pos[sel]] = np.clip(out[r, pos[sel]] + pr[sel], -16, 16)
    tail = codes.zero_tail(spec, pb)  # the fill past the CRC is always known
    return np.stack([codes.pin(spec, o, tail, int(i)) for o, i in zip(out, idx)])


def decode_post(spec, soft):
    """Plain decodes (codes.decode_posterior, native under with_native) ->
    (info bits scrambled, converged, posteriors, the soft bits; mapping order)."""
    got = [codes.decode_posterior(spec, s) for s in np.asarray(soft, np.float32)]
    return (np.stack([g[0] for g in got]), np.array([g[1] for g in got]), np.stack([g[2] for g in got]),
            np.asarray(soft))


def channel(spec, cw, esn0, rng, fading):
    pts = constellation.load(spec.constellation)
    x = constellation.modulate(cw.reshape(-1), pts).reshape(len(cw), -1)
    h = (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)) / np.sqrt(2) if fading else np.ones_like(x)
    var = np.full(x.shape, 10 ** (-esn0 / 10))
    y = h * x + np.sqrt(var / 2) * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape))
    return constellation.llr(y, h, var, pts).reshape(len(cw), -1)


BMETHODS = ["plain", "A", "B0p", "B1p", "B2p", "B1c", "C1p", "D", "D2", "Dc", "Dc2", "F"]


def bfailures(spec, esn0, seed, content, trials, fading):
    """-> ({method: failures}, {method: decodes past plain}, {method: false
    passes}, [(boundary error, posterior's and channel's)] of the failed rows)."""
    rng = np.random.default_rng(seed)
    pb, mask = codes.payload_bytes(spec), 0x5A5A
    i = (content - 1) // pb
    n = content - i * pb
    pays = []
    for _ in range(trials):  # last content byte nonzero: n is where the padding starts
        c = rng.integers(0, 256, n, dtype=np.uint8)
        c[-1] = c[-1] or 1
        pays.append(bytes(c) + bytes(pb - n))
    idx = rng.integers(0, 4, trials)
    cw = np.stack([codes.encode(spec, p, 0, mask, i_) for p, i_ in zip(pays, idx)])
    soft = channel(spec, cw, esn0, rng, fading)
    bits, ok, post, deint = decode_post(spec, soft)

    def chk(r, b):
        return codes.check(spec, codes.descramble(spec, b, idx[r])[None], np.ones(1, bool), mask)

    good0 = np.array([bool(ok[r]) and chk(r, bits[r]) == pays[r] for r in range(trials)])
    fails = np.flatnonzero(~good0)
    fail, dec, false = {m: 0 for m in BMETHODS}, {m: 0 for m in BMETHODS}, {m: 0 for m in BMETHODS}
    fail["plain"] = len(fails)
    if not len(fails):
        return fail, dec, false, []
    cache = {}

    def pinned(r, o):
        if (r, o) not in cache:
            c, u = codes.decode_raw_pinned(spec, soft[r][None], codes.zero_tail(spec, o), int(idx[r]))
            got = codes.check(spec, c[0], u[0], mask)
            cache[(r, o)] = (got == pays[r], got is not None and got != pays[r])
        return cache[(r, o)]

    def run(name, tries):
        """tries: per failed row, the offsets tried in order (CRC-checked each)."""
        for r, offs in zip(fails, tries):
            done = False
            for o in offs:
                dec[name] += 1
                g, fp = pinned(r, o)
                false[name] += fp
                if g or fp:
                    done = g
                    break
            fail[name] += not done

    run("A", [[c - i * pb for c in CUTS["ldpc"] if 0 < c - i * pb < pb]] * len(fails))
    Sp = pad_scores(spec, info_llrs(spec, post[fails]), idx[fails])
    Sc = pad_scores(spec, info_llrs(spec, deint[fails]), idx[fails])
    b0 = first_sent_byte(spec)  # pins before it pin nothing more (punctured)
    jp, jc = best_cut(Sp, b0), best_cut(Sc, b0)
    errs = list(zip((jp - max(n, b0)).tolist(), (jc - max(n, b0)).tolist()))
    some = lambda j: [int(j)] if j < pb else []  # noqa: E731
    for d in (0, 1, 2):
        run(f"B{d}p", [some(j + d) for j in jp])
    run("B1c", [some(j + 1) for j in jc])
    tries = []
    for r, j in enumerate(jp):
        S = Sp[r].copy()
        S[max(0, j - 2): j + 3] = -np.inf  # outside the first try's neighbourhood
        j2 = best_cut(S[None], b0)[0]
        tries.append(some(j + 1) + [o for o in some(j2 + 1) if o != j + 1])
    run("C1p", tries)
    # D: soft priors from S (posterior's, Dc: channel's), one decode; D2: re-estimated from its posterior
    for name, S0 in (("D", Sp), ("Dc", Sc)):
        b2, ok2, post2, _ = decode_post(spec, soft_prior(spec, soft[fails], S0, idx[fails]))
        good = np.zeros(len(fails), bool)
        for t, r in enumerate(fails):
            got = chk(r, b2[t]) if ok2[t] else None
            good[t] = got == pays[r]
            false[name] += got is not None and got != pays[r]
        dec[name], fail[name] = len(fails), int((~good).sum())
        left = np.flatnonzero(~good)
        good2, false[name + "2"], dec[name + "2"] = good.copy(), false[name], len(fails) + len(left)
        if len(left):
            S2 = pad_scores(spec, info_llrs(spec, post2[left]), idx[fails][left])
            b3, ok3, _, _ = decode_post(spec, soft_prior(spec, soft[fails][left], S2, idx[fails][left]))
            for t, q in enumerate(left):
                r = fails[q]
                got = chk(r, b3[t]) if ok3[t] else None
                good2[q] = got == pays[r]
                false[name + "2"] += got is not None and got != pays[r]
        fail[name + "2"] = int((~good2).sum())
    run("F", [some(n)] * len(fails))
    return fail, dec, false, errs


def bstudy(name: str, content: int, fading: bool, trials: int, lo: float, hi: float) -> list:
    spec = SUBMODES[name]
    rows, e = [], hi
    while e >= lo - 1e-9:  # down from data's 10% point until even the genie fails half
        f, d, fp, errs = bfailures(spec, e, 1000 + spec.index + 7 * content, content, trials, fading)
        rows.append({"mode": name, "content": content, "fading": int(fading), "esn0": e, "trials": trials,
                     **{f"fail_{m}": v for m, v in f.items()}, **{f"dec_{m}": v for m, v in d.items()},
                     **{f"false_{m}": v for m, v in fp.items()}, "errs": ";".join(f"{a},{b}" for a, b in errs)})
        if f["F"] > trials / 2:
            break
        e -= STEP
    return rows


def _data_point(name, fading):
    """data's 10% failure point (100 random payloads a step)."""
    spec = SUBMODES[name]
    e = -12.0
    while e < 40:
        rng = np.random.default_rng(1000 + spec.index)
        pb = codes.payload_bytes(spec)
        cw = np.stack([codes.encode(spec, bytes(rng.integers(0, 256, pb, dtype=np.uint8)), 0, 0, 0) for _ in range(100)])
        if (~decode_post(spec, channel(spec, cw, e, rng, fading))[1]).sum() <= 10:
            return e
        e += STEP
    return e


def boundary_main(a):
    modes = a.modes or [n for n, s in SUBMODES.items() if s.code == "ldpc"]
    with Pool(a.jobs) as pool:
        data = dict(zip(modes, pool.starmap(_data_point, [(m, a.fading) for m in modes])))
        jobs = [(m, c, a.fading, a.trials, data[m] + a.span[0], data[m] + a.span[1]) for c in a.boundary for m in modes]
        out = pool.starmap(bstudy, jobs, chunksize=1)
    rows = [r for rs in out for r in rs]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ctl_prior_study.csv")
    ap.add_argument("--modes", nargs="*", default=None)
    ap.add_argument("--polar", action="store_true", help="the polar control codewords (see the docstring)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--content", type=int, default=CONTENT,
                    help="control bytes (a control with T_INLINE: its data in the padding, 5 B more)")
    ap.add_argument("--boundary", nargs="*", type=int, default=None,
                    help="content lengths: pinning from a boundary estimate (pad_scores), per method")
    ap.add_argument("--fading", action="store_true", help="--boundary: i.i.d. Rayleigh per symbol, gains known")
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--span", type=float, nargs=2, default=(-9.0, 1.0),
                    help="--boundary: Es/N0 range relative to data's 10%% point")
    a = ap.parse_args()
    if a.boundary is not None:
        return boundary_main(a)
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
