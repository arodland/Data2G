"""GF(M) LDPC for the CPM modes (data2g.gfldpc) against the shipped binary
LDPC + cpm.llrs, on ideal channels (cpm_floor_study's `ideal`: synthetic
tone energies, AWGN or Rayleigh per tone, the receiver's own metric).

    python -m scripts.gf_study ideal --modes fsk32r62-r1/3 [--n 200] [--profiles 20x5 30x4]

    python -m scripts.gf_study ir --modes fsk8r50-r1/3 --profiles 60x4:1 [--d 2]
    python -m scripts.gf_study write   # PICKS -> data2g/codes_data/gf_codes.npz

Prints 10% / 1% codeword failure Es/N0 (per tone) per arm, and the
undetected errors (converged to a wrong codeword) of the GF arms. `ir`:
two receptions, Chase against the raptor-like extension.

Found (2026-10-27, 200 codewords a point, 50 flooding iterations):
- 10% points, ideal Rayleigh | AWGN, shipped LDPC -> PICKS: fsk8r50-r1/3
  6.80 -> 5.86 | 4.95 -> 4.25; r1/2 9.00 -> 8.24 | 6.42 -> 5.70;
  fsk16r25-r1/3 7.35 -> 6.31 | 5.82 -> 4.71; r1/2 9.74 -> 8.66 | 7.20 ->
  6.14; fsk32r62-r1/3 8.00 -> 6.69 | 6.65 -> 5.14; r1/2 10.39 -> 9.03 |
  7.78 -> 6.47. GF(8) wants many heavy columns (r1/2 with 20-40 had no
  fading gain), GF(32) is flat over 24-60.
- Undetected (converged to a wrong codeword): 0-2 per 4000-5000 codewords
  for PICKS over each whole sweep; sparser profiles 10-80. CRC-16 kept.
- ir, two receptions (10%, Rayleigh): extension rows of 2 symbols beat 3
  by 0.4-0.7 dB; IR over Chase 0.4-0.7 dB on GF(8), +-0.1 on GF(32).
- End to end (cpm_floor_study e2e, 200 trials, seeds 50000+), 10% points
  against the shipped LDPC's: with the shipped receiver fsk32r62-r1/3
  -0.26 (mpp) to -0.96 (awgn), fsk16r25 -0.10 to -0.76, fsk8r50 0.00 to
  -0.63; with genie acquisition (start, CFO, header known) fsk32r62 -1.05
  to -1.45, fsk16r25 -0.76 to -1.06, fsk8r50 -0.28 to -0.60. Acquisition
  (front lock, header) is what the gain runs into.
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
from dataclasses import replace  # noqa: E402
from functools import lru_cache  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, cpm, gfldpc  # noqa: E402
from data2g.arq.modes import MODES  # noqa: E402


def square_law(E):
    """cpm.llrs's per-tone metric (the codeword's own es and N0)."""
    top = E.max(axis=1)
    n0 = max(np.median(np.sort(E, axis=1)[:, :-1]) / np.log(2), 1e-12 * np.mean(top) + 1e-300)
    es = max(np.mean(top) - n0, 1e-3 * n0)
    return E * (es / (n0 + es) / n0)


@lru_cache(maxsize=None)
def gf_code(name, heavy, seed):
    spec = MODES[name]
    g = cpm.GRIDS[spec.grid]
    n = spec.coded_bits // g.bits
    k = -(-spec.k // g.bits)
    return gfldpc.design(g.bits, n, k, heavy, seed)


@lru_cache(maxsize=None)
def gf_dec(name, heavy, seed):
    return gfldpc.Decoder(gf_code(name, heavy, seed))


def parse(arm):
    """"30x4+10x6:1" -> (((30, 4), (10, 6)), 1)"""
    prof, seed = arm.split(":")
    return tuple(tuple(int(v) for v in t.split("x")) for t in prof.split("+") if t), int(seed)


def channel(rng, t, m, esn0, fading):
    es = 10 ** (esn0 / 10)
    h = (rng.normal(size=t.shape) + 1j * rng.normal(size=t.shape)) / np.sqrt(2) if fading else np.ones(t.shape)
    y = (rng.normal(size=t.shape + (m,)) + 1j * rng.normal(size=t.shape + (m,))) / np.sqrt(2)
    np.put_along_axis(y, t[..., None], np.take_along_axis(y, t[..., None], -1) + (np.sqrt(es) * h)[..., None], -1)
    return np.abs(y) ** 2


def point(args):
    name, fading, esn0, n, arm, iters = args
    spec = replace(MODES[name], code="ldpc")  # the shipped binary LDPC (protocol 13) for the ldpc arm
    g = cpm.GRIDS[spec.grid]
    rng = np.random.default_rng(1000 + int(esn0 * 100) + 7 * fading)
    gray = cpm._gray(g.m)
    if arm == "ldpc":
        ok = 0
        for i in range(n):
            payload = bytes(rng.integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8))
            t = cpm.to_tones(g, codes.encode(spec, payload, index=1))
            E = channel(rng, t, g.m, esn0, fading)
            out = codes.decode_many(spec, cpm.llrs(g, E)[None], index=[1])[0]
            ok += out[1] and out[0] == payload
        return name, fading, esn0, arm, ok / n, 0.0
    heavy, seed = parse(arm)
    c, dec = gf_code(name, heavy, seed), gf_dec(name, heavy, seed)
    u = rng.integers(0, g.m, (n, c.k))
    x = c.encode(u)
    E = channel(rng, gray[x], g.m, esn0, fading)
    L = np.stack([square_law(e) for e in E])[:, :, gray]  # (n, sym, value): tone gray[v]
    hard, conv, _ = dec.decode(L, iters)
    good = conv & (hard == x).all(axis=1)
    return name, fading, esn0, arm, good.mean(), (conv & ~good).mean()


def ir_point(args):
    """Two receptions: Chase (RV 0 twice, likelihoods summed), IR (RV 0
    then n extension symbols, the mother decoded); the shipped LDPC: its
    own IR (RV 0 + RV 1 combined) in both columns."""
    name, fading, esn0, n, arm, d, iters = args
    spec = replace(MODES[name], code="ldpc")  # the shipped binary LDPC (protocol 13) for the ldpc arm
    g = cpm.GRIDS[spec.grid]
    rng = np.random.default_rng(2000 + int(esn0 * 100) + 7 * fading)
    gray = cpm._gray(g.m)
    if arm == "ldpc":
        ok = 0
        for i in range(n):
            payload = bytes(rng.integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8))
            buf = None
            for rv in (0, 1):
                t = cpm.to_tones(g, codes.encode(spec, payload, rv=rv, index=codes.PLAIN))
                buf = codes.combine(spec, buf, cpm.llrs(g, channel(rng, t, g.m, esn0, fading))[None], rv)
            out = codes.decode_buffer(spec, buf, 1, index=codes.PLAIN)[0]
            ok += out[1] and out[0] == payload
        return name, fading, esn0, arm, ok / n, ok / n
    heavy, seed = parse(arm)
    c = gf_code(name, heavy, seed)
    mo = mother(name, heavy, seed, d)
    u = rng.integers(0, g.m, (n, c.k))
    x = mo.encode(u)

    def L(t):
        return np.stack([square_law(e) for e in channel(rng, gray[t], g.m, esn0, fading)])[:, :, gray]

    a, b = L(x[:, :c.n]), L(x[:, :c.n])
    h, conv, _ = gf_dec(name, heavy, seed).decode(a + b, iters)
    chase = (conv & (h == x[:, :c.n]).all(1)).mean()
    e = L(x[:, c.n:])
    h, conv, _ = mother_dec(name, heavy, seed, d).decode(np.concatenate([a, e], 1), iters)
    ir = (conv & (h == x).all(1)).mean()
    return name, fading, esn0, arm, chase, ir


@lru_cache(maxsize=None)
def mother(name, heavy, seed, d):
    c = gf_code(name, heavy, seed)
    return gfldpc.extend(c, c.n, d, seed + 100)


@lru_cache(maxsize=None)
def mother_dec(name, heavy, seed, d):
    return gfldpc.Decoder(mother(name, heavy, seed, d))


def ir(a):
    grid = np.arange(a.lo, a.hi + 1e-9, a.step)
    arms = (["ldpc"] if a.ldpc else []) + a.profiles
    jobs = [(m, f, float(e), a.n, arm, a.d, a.iters) for m in a.modes for f in (False, True) for e in grid
            for arm in arms]
    with Pool(a.jobs) as p:
        res = p.map(ir_point, jobs, chunksize=1)
    print("Two receptions, Es/N0 per tone (dB) at 10% failure: chase | IR (ldpc: its own IR)")
    for m in a.modes:
        for f in (False, True):
            for arm in arms:
                r = [x for x in res if x[0] == m and x[1] == f and x[3] == arm]
                c, i = (crossing(list(grid), [1 - x[k] for x in r], 0.1) for k in (4, 5))
                print(f"  {m:15s} {'ray ' if f else 'awgn'} {arm:10s} {c:6.2f} | {i:6.2f}")


def crossing(xs, fail, target):
    for i in range(len(xs) - 1):
        if fail[i] >= target > fail[i + 1]:
            return xs[i] + (fail[i] - target) / (fail[i] - fail[i + 1]) * (xs[i + 1] - xs[i])
    return float("nan")


def ideal(a):
    grid = np.arange(a.lo, a.hi + 1e-9, a.step)
    arms = (["ldpc"] if a.ldpc else []) + a.profiles
    jobs = [(m, f, float(e), a.n, arm, a.iters) for m in a.modes for f in (False, True) for e in grid for arm in arms]
    with Pool(a.jobs) as p:
        res = p.map(point, jobs, chunksize=1)
    print("Es/N0 per tone (dB) at 10% / 1% codeword failure; undetected (converged wrong) per codeword in brackets")
    for m in a.modes:
        for f in (False, True):
            for arm in arms:
                r = [x for x in res if x[0] == m and x[1] == f and x[3] == arm]
                fail = [1 - x[4] for x in r]
                und = sum(x[5] for x in r) * a.n
                print(f"  {m:15s} {'ray ' if f else 'awgn'} {arm:10s} {crossing(list(grid), fail, 0.1):6.2f} / "
                      f"{crossing(list(grid), fail, 0.01):6.2f}  (undetected {und:.0f} of {a.n * len(r)})")


# The frozen codes (`write`): per spec, the profile ((count, degree) of the
# heavy columns; the rest degree 2), the PEG seed, and the extension rows'
# symbols besides their own. Picked from `ideal` / `ir` (2026-10-26).
PICKS = {
    "fsk8r50-r1/3": (((100, 4),), 1, 2),
    "fsk8r50-r1/2": (((120, 4),), 1, 2),
    "fsk16r25-r1/3": (((60, 4),), 2, 2),
    "fsk16r25-r1/2": (((80, 4),), 1, 2),
    "fsk32r62-r1/3": (((50, 4),), 1, 2),
    "fsk32r62-r1/2": (((60, 4),), 1, 2),
}


def write(a):
    out = {}
    for name, (heavy, seed, d) in PICKS.items():
        spec = MODES[name]
        p = cpm.GRIDS[spec.grid].bits
        n, k = spec.coded_bits // p, -(-spec.k // p)
        c = gfldpc.design(p, n, k, heavy, seed)
        assert c.k == k, f"{name}: H has rank {n - c.k}, short of {n - k}: pick another seed"
        mo = gfldpc.extend(c, 6 * spec.k // p - n, d, seed + 100)  # mother rate 1/6, as the binary LDPC's
        assert mo.k == k and (mo.info == c.info).all()
        key = gfldpc.key(name)
        out.update({f"{key}_p": p, f"{key}_n": mo.n, f"{key}_n0": n, f"{key}_var": mo.chk_var.astype(np.int16),
                    f"{key}_h": mo.chk_h.astype(np.uint8)})
        print(f"{name}: GF({1 << p}) n {n} k {k}, heavy {heavy}, seed {seed}; mother n {mo.n}, "
              f"{int((c.chk_var >= 0).sum())} edges in the code")
    np.savez_compressed(gfldpc.TABLES, **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ideal", "ir", "write"])
    ap.add_argument("--d", type=int, default=2, help="ir: an extension row's symbols besides its own")
    ap.add_argument("--modes", nargs="+", default=["fsk32r62-r1/3"])
    ap.add_argument("--profiles", nargs="+", default=["30x4:1"], help="countxdegree+...:seed")
    ap.add_argument("--no-ldpc", dest="ldpc", action="store_false")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--lo", type=float, default=3.0)
    ap.add_argument("--hi", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    {"ideal": ideal, "ir": ir, "write": write}[a.cmd](a)


if __name__ == "__main__":
    main()
