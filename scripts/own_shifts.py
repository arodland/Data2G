"""The LDPC circulant shift tables: generated, screened, exported.

One table per (base graph, Z) a submode uses. Fixed: the block masks,
the dual-diagonal core (weight-3 column shifts 1,0,1; every other core
and extension diagonal 0). Every other shift is generated here.

Generation: greedy, one edge at a time, rows in order (the top rows are
in every code; later rows only in low rates and IR), random order within
a row. Each shift is the value closing the fewest 4-cycles, then (mode
g6) the fewest 6-cycles with every already-placed edge; g8 also weighs
8-cycles, ga weights cycles by ACE, g4 only avoids 4-cycles, rand is
uniform. Cycles are enumerated once in the base graph: a base cycle lifts
to Z cycles when its alternating shift sum is 0 mod Z, else to none of
that length.

What screening found (2026-09-30, runs/own_shifts):
- Uniform random shifts floor; avoiding 4-cycles is necessary.
- Cycle counts do not predict floors (small trapping sets, a few low-
  weight codewords): seeds of one family spread 0-14 errors per 100k at
  one point, and fewer 8-cycles floored worse.
- So: gen, then `screen` (one point just above the waterfall, 100k BPSK
  blocks, early rejection) on every code of that Z and at 2n (IR), then
  a paired ladder (scripts/ladder_study.py) before shipping.

    python -m scripts.own_shifts gen --bg 2 --modes g6 g8 --seeds 32 --tag _new
    python -m scripts.own_shifts screen --cands _new --point 2,96,960,1920,2.07
    python -m scripts.own_shifts export      # after editing PICKS
"""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from data2g import config, cpm, decoders_torch, ldpc

OUT = Path("runs/own_shifts")

# The shipped table per (bg, Z): lift(mode, seed) regenerates it.
PICKS = {
    (1, 36): "g6-27", (1, 72): "g6-29", (1, 104): "g6-23", (1, 144): "g6-22", (1, 160): "g6-7",
    (2, 26): "g6-17", (2, 32): "g6-26", (2, 40): "g6-8", (2, 44): "g8-21", (2, 48): "g6-11",
    (2, 60): "g6-21", (2, 64): "g6-15", (2, 72): "g6-5", (2, 96): "g6-30", (2, 104): "g6-15",
    (2, 128): "g6-2", (2, 144): "g6-0", (2, 176): "g6-4", (2, 192): "g8-30", (2, 240): "g8-9",
    (2, 256): "g6-0",
}


def pick_table(bg: int, z: int) -> np.ndarray:
    mode, seed = PICKS[(bg, z)].split("-")
    return lift(Cycles(ldpc.mask(bg), eight=mode != "g6"), bg, z, mode, int(seed))


def shipped() -> dict[tuple[int, int], list[tuple[str, int, int]]]:
    """(bg, z) -> [(submodes, k, n)] over the LDPC submodes (OFDM and CPM);
    submodes with the same (k, n) are one code here (BPSK: no constellation)."""
    out = {}
    specs = [s for s in config.SUBMODES.values() if s.code == "ldpc"] + list(cpm.SPECS.values())
    for s in specs:
        out.setdefault(ldpc.layout(s.k, s.coded_bits), {}).setdefault((s.k, s.coded_bits), []).append(s.name)
    return {g: [("+".join(v), k, n) for (k, n), v in d.items()] for g, d in sorted(out.items())}


def fixed_shifts(bg: int) -> np.ndarray:
    """Core and extension-diagonal shifts; -1 where free or absent."""
    m = ldpc.mask(bg)
    kb = ldpc.KB[bg]
    f = np.full(m.shape, -1, dtype=np.int64)
    rows = np.flatnonzero(m[:4, kb])  # the weight-3 core column
    f[rows, kb] = [1, 0, 1]
    for c in range(kb + 1, kb + 4):
        f[m[:4, c].nonzero()[0], c] = 0
    for r in range(4, m.shape[0]):
        f[r, kb + r] = 0
    return f


class Cycles:
    """Base-graph 4- and 6-cycles (and 8-, `eight`) as edge ids: the lifted
    cycle exists when the alternating sum of their shifts is 0 mod z."""

    def __init__(self, mask: np.ndarray, eight: bool = False):
        self.mask = mask
        self.eid = np.full(mask.shape, -1, dtype=np.int64)
        self.er, self.ec = np.nonzero(mask)
        self.eid[self.er, self.ec] = np.arange(len(self.er))
        nb = [set(np.flatnonzero(mask[r])) for r in range(mask.shape[0])]
        e = self.eid
        c4, c6 = [], []
        for a, b in itertools.combinations(range(mask.shape[0]), 2):
            for x, y in itertools.combinations(sorted(nb[a] & nb[b]), 2):
                c4.append((e[a, x], e[b, x], e[b, y], e[a, y]))
        for a, b, c in itertools.combinations(range(mask.shape[0]), 3):
            A, B, C = nb[a] & nb[b], nb[b] & nb[c], nb[c] & nb[a]
            if not (A and B and C):
                continue
            for x in A:
                for y in B - {x}:
                    for w in C - {x, y}:
                        c6.append((e[a, x], e[b, x], e[b, y], e[c, y], e[c, w], e[a, w]))
        self.c = {4: np.array(c4), 6: np.array(c6)}
        if eight:  # base graph 2: 256k base 8-cycles; a = min row, neighbours b < d
            c8 = []
            M = mask.shape[0]
            for a in range(M):
                for b, d in itertools.combinations(range(a + 1, M), 2):
                    X, V = nb[a] & nb[b], nb[d] & nb[a]
                    if not (X and V):
                        continue
                    for c in range(a + 1, M):
                        if c in (b, d):
                            continue
                        Y, W = nb[b] & nb[c], nb[c] & nb[d]
                        for x in X:
                            for y in Y - {x}:
                                for w in W - {x, y}:
                                    for v in V - {x, y, w}:
                                        c8.append((e[a, x], e[b, x], e[b, y], e[c, y],
                                                   e[c, w], e[d, w], e[d, v], e[a, v]))
            self.c[8] = np.array(c8)
        # a cycle's max row: it is in a truncated graph iff that row is kept
        self.top = {L: self.er[cy].max(1) for L, cy in self.c.items()}
        # ACE (Tian et al. 2004): sum of (degree - 2) over a cycle's variable
        # nodes, mother-graph degrees; each node is on two of its edges
        deg = mask.sum(0)
        self.ace = {L: (deg[self.ec[cy]] - 2).sum(1) / 2 for L, cy in self.c.items()}

    def sign(self, L):
        return np.tile([1, -1], L // 2)

    def balanced(self, s: np.ndarray, z: int, L: int) -> np.ndarray:
        """(cycles,) bool: lifts to Z cycles of length L."""
        return (s[self.c[L]] * self.sign(L)).sum(1) % z == 0

    def counts(self, s: np.ndarray, z: int, mb: int) -> tuple[int, ...]:
        """Balanced base 4-, 6- (and 8-) cycles among rows < mb."""
        return tuple(int((self.balanced(s, z, L) & (self.top[L] < mb)).sum()) for L in sorted(self.c))


G8_W6 = 4.0  # g8: one 6-cycle counts as this many 8-cycles
GA_ETA = 3.0  # ga: a cycle's weight falls by e per GA_ETA of ACE


def lift(cyc: Cycles, bg: int, z: int, mode: str, seed: int) -> np.ndarray:
    """Full (M, N) shift table, -1 where the mask has no block."""
    rng = np.random.default_rng(seed)
    f = fixed_shifts(bg)
    fixed = f[cyc.er, cyc.ec] >= 0
    s = np.where(fixed, f[cyc.er, cyc.ec], -1) % z
    s[~fixed] = -1
    free = np.flatnonzero(~fixed)
    order = sorted(free, key=lambda e: (cyc.er[e], rng.random()))
    if mode == "rand":
        s[free] = rng.integers(0, z, len(free))
    else:
        # rank: fixed edges first, then placement order; a cycle is scored
        # when its last-placed edge is chosen
        rank = np.full(len(s), -1)
        rank[order] = np.arange(len(order))
        closing = {}
        for L, cy in cyc.c.items():
            last = cy[np.arange(len(cy)), rank[cy].argmax(1)]
            ok = rank[cy].max(1) >= 0
            for e, i in zip(last[ok], np.flatnonzero(ok)):
                closing.setdefault((L, e), []).append(i)
        closing = {k: np.array(v) for k, v in closing.items()}
        for e in order:
            score = rng.random(z) * 0.5
            weights = {"g6": ((4, 1e6), (6, 1.0)), "g4": ((4, 1e6),),
                       "g8": ((4, 1e6), (6, G8_W6), (8, 1.0)),
                       "ga": ((4, 1e6), (6, G8_W6), (8, 1.0))}[mode]
            for L, w in weights:
                idx = closing.get((L, e))
                if idx is None or w == 0:
                    continue
                if mode == "ga" and L > 4:  # low-ACE cycles are the trapping-set ones
                    w = w * np.exp(-cyc.ace[L][idx] / GA_ETA)
                cy = cyc.c[L][idx]
                sg = np.broadcast_to(cyc.sign(L), cy.shape)
                pos = cy == e
                rest = np.where(pos, 0, s[cy] * sg).sum(1)
                se = sg[pos]  # this edge's sign in each cycle
                bad = (-rest * se) % z  # se * v + rest = 0 mod z
                score += np.bincount(bad, weights=np.broadcast_to(w, bad.shape), minlength=z)
            s[e] = int(np.argmin(score))
    out = np.full(cyc.mask.shape, -1, dtype=np.int64)
    out[cyc.er, cyc.ec] = s
    return out


def code_with(table: np.ndarray, bg: int, k: int, n: int) -> ldpc.QCLDPC:
    return ldpc.QCLDPC(base=table.copy(), z=ldpc.layout(k, n, bg)[1], kb=ldpc.KB[bg], k=k, n=n)


def rank_key(r):
    """Fewest 4-cycles anywhere, then 6-cycles in the shipped graphs
    (the smallest truncation first: it is in every code of this z)."""
    mbs = sorted(int(k[5:]) for k in r if k.startswith("c4_mb"))
    c4 = sum(r[f"c4_mb{m}"] for m in mbs)
    if f"c8_mb{mbs[0]}" in r:  # with 8-cycles: g8's own weighting, summed
        return (c4, sum(G8_W6 * r[f"c6_mb{m}"] + r[f"c8_mb{m}"] for m in mbs))
    return (c4,) + tuple(r[f"c6_mb{m}"] for m in mbs)


def cmd_gen(a):
    """Candidates per shipped (bg, z) with their cycle counts, next to the
    shipped table ("cur") where there is one."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    eight = bool({"g8", "ga"} & set(a.modes))
    for bg in a.bg:
        cyc = Cycles(ldpc.mask(bg), eight=eight)
        Ls = sorted(cyc.c)
        for (g, z), subs in shipped().items():
            if g != bg:
                continue
            mbs = sorted({code_with(np.zeros(cyc.mask.shape), bg, k, n).mb for _, k, n in subs})
            mbs.append(cyc.mask.shape[0])  # mother
            tables = {}
            if (bg, z) in PICKS:
                tables["cur"] = ldpc.tables()[f"bg{bg}_z{z}"]
            for mode in a.modes:
                for seed in range(a.seeds if mode != "rand" else 2):
                    tables[f"{mode}-{seed}"] = lift(cyc, bg, z, mode, seed)
            for name, t in tables.items():
                code_with(t, bg, *subs[0][1:])._core_inv  # the encoder needs an invertible core
                r = {"bg": bg, "z": z, "cand": name}
                for mb in mbs:
                    for L, c in zip(Ls, cyc.counts(t[cyc.er, cyc.ec], z, mb)):
                        r[f"c{L}_mb{mb}"] = c
                rows.append(r)
            np.savez_compressed(OUT / f"cands_bg{bg}_z{z}{a.tag}.npz", **tables)
            here = [x for x in rows if x["bg"] == bg and x["z"] == z]
            best = sorted((x for x in here if x["cand"] != "cur"), key=rank_key)[:3]
            fmt = lambda x: " ".join("/".join(str(x[f"c{L}_mb{m}"]) for L in Ls) for m in mbs)
            cur = next((f"cur {fmt(x)}  |  " for x in here if x["cand"] == "cur"), "")
            print(f"bg{bg} z={z:3d} mb {mbs}  {'/'.join(map(str, Ls))}-cycles  {cur}"
                  + "  ".join(f"{x['cand']} {fmt(x)}" for x in best), flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / f"cycles{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


# The GPU is shared with the desktop: the decoder's (batch, checks, dmax)
# intermediates are sized to this, and the process is capped (gpu_cap).
VRAM_BUDGET = 1.0e9
VRAM_CAP = 0.12  # of the device; an overshoot is our OOM, not another app's


def gpu_cap(device):
    import torch

    if str(device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(VRAM_CAP)


def bler_curve(code, ebn0s, blocks, device, iters=40, seed=1, batch=2000):
    """BPSK/AWGN BLER, BP at the modem's 40 iterations; the same bits and
    noise for any code of this (k, n) and seed (paired)."""
    import torch

    dec = decoders_torch.MinSumDecoder(code, device=device)
    k, n = code.k, code.n
    # ~10 live float32 tensors of (batch, checks, dmax) at the peak
    batch = max(100, min(batch, int(VRAM_BUDGET / (40 * dec.chk.numel()))))
    blocks = -(-blocks // batch) * batch
    out = []
    for ebn0 in ebn0s:
        rng = np.random.default_rng([seed, int(round(ebn0 * 100)) + 10000])
        fe = done = 0
        while done < blocks:
            bits = rng.integers(0, 2, (batch, k))
            noise = rng.normal(size=(batch, n))
            sig = np.sqrt(1 / (2 * k / n * 10 ** (ebn0 / 10)))
            y = 1 - 2.0 * code.encode(bits) + sig * noise
            est, _ = dec.decode(torch.tensor(2 * y / sig**2, dtype=torch.float32, device=device), iters=iters)
            fe += int((est.cpu().numpy() != bits).any(1).sum())
            done += batch
        out.append(fe / done)
    del dec
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()  # code sizes vary; don't hold the high-water mark
    return np.array(out)


def cmd_screen(a):
    """At each --point (bg,z,k,n,Eb/N0), the reference (the shipped table,
    or --ref) over --blocks, then the top candidates of each --cands set
    with early rejection: stop once errors pass max(3 x ref's, ref's + 10)."""
    import torch

    torch.set_num_threads(2)
    gpu_cap(a.device)
    fh = open(OUT / f"screen_{a.tag}.csv", "a", newline="")
    w = csv.writer(fh)
    for p in a.point:
        bg, z, k, n = (int(x) for x in p.split(",")[:4])
        ebn0 = float(p.split(",")[4])
        ref = np.load(a.ref)[f"bg{bg}_z{z}"] if a.ref else ldpc.tables()[f"bg{bg}_z{z}"]
        e_ref = int(round(bler_curve(code_with(ref, bg, k, n), [ebn0], a.blocks, a.device, seed=11)[0] * a.blocks))
        limit = max(3 * e_ref, e_ref + 10)
        print(f"bg{bg} z={z:3d} k={k} n={n} @{ebn0} dB: ref {e_ref} errors / {a.blocks}, reject above {limit}",
              flush=True)
        w.writerow([bg, z, k, n, ebn0, "ref", e_ref, a.blocks])
        for tag in a.cands:
            tables = np.load(OUT / f"cands_bg{bg}_z{z}{tag}.npz")
            with open(OUT / f"cycles{tag}.csv") as f:
                rk = {r["cand"]: rank_key({q: (v if q == "cand" else int(v)) for q, v in r.items() if v != ""})
                      for r in csv.DictReader(f) if int(r["bg"]) == bg and int(r["z"]) == z}
            names = sorted((c for c in tables.files if c != "cur" and not c.startswith("rand")), key=rk.get)
            for cand in names[: a.top]:
                code = code_with(tables[cand], bg, k, n)
                errs = done = 0
                step = 10000
                while done < a.blocks and errs <= limit:
                    errs += int(round(bler_curve(code, [ebn0], step, a.device, seed=11 + done)[0] * step))
                    done += step
                verdict = "REJECT" if errs > limit else ("ok" if errs <= e_ref + 2 * np.sqrt(e_ref + 1) else "worse")
                print(f"  {tag or '-':4s} {cand:6s} {errs:4d} errors / {done:6d}  {verdict}", flush=True)
                w.writerow([bg, z, k, n, ebn0, f"{tag}:{cand}", errs, done])
                fh.flush()


def cmd_export(a):
    """PICKS and the masks -> the runtime's table file."""
    missing = set(shipped()) - set(PICKS)
    if missing:
        raise SystemExit(f"shipped (bg, z) without a pick: {sorted(missing)}")
    out = {f"mask_bg{bg}": ldpc.mask(bg) for bg in ldpc.KB}
    out.update({f"bg{bg}_z{z}": pick_table(bg, z).astype(np.int16) for bg, z in PICKS})
    np.savez_compressed(ldpc.SHIFTS, **out)
    print(f"{len(PICKS)} tables -> {ldpc.SHIFTS}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    g = sub.add_parser("gen")
    g.add_argument("--seeds", type=int, default=32)
    g.add_argument("--modes", nargs="+", default=["g6", "g4", "rand"])
    g.add_argument("--bg", type=int, nargs="+", default=[1, 2])
    g.add_argument("--tag", default="", help="output suffix: cands_bg2_z96<tag>.npz, cycles<tag>.csv")
    c = sub.add_parser("screen")
    c.add_argument("--point", nargs="+", required=True, help="bg,z,k,n,ebn0 ...")
    c.add_argument("--cands", nargs="+", default=[""], help="gen tags")
    c.add_argument("--ref", default="", help="an npz of bg<b>_z<Z> tables (default: the shipped ones)")
    c.add_argument("--blocks", type=int, default=100000)
    c.add_argument("--top", type=int, default=8, help="per tag, best by cycle rank")
    c.add_argument("--tag", default="floor")
    c.add_argument("--device", default="cuda")
    a = ap.parse_args()
    {"gen": cmd_gen, "screen": cmd_screen, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
