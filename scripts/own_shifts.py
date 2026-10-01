"""Our own circulant shifts for NR's LDPC base graphs, per shipped (bg, Z).

Why: the TS 38.212 shift tables are the 5G-specific (and likely
patent-declared) part of the LDPC code; the block mask and the
dual-diagonal core are older (802.11n/802.16e) structure. This keeps
NR's mask, the core (weight-3 column shifts 1,0,1; every other core and
extension diagonal 0) and replaces every other shift.

Generation: greedy, one edge at a time, rows in order (the top rows are
in every code; later rows only in low rates and IR), random order within
a row. Each shift is the value closing the fewest 4-cycles, then (mode
g6) the fewest 6-cycles, with every already-placed edge; mode g4 only
avoids 4-cycles, rand is uniform. Cycles are enumerated once in the
base graph: a base cycle lifts to Z cycles when its alternating shift
sum is 0 mod Z, else to none of that length.

Screening, cheapest first:
  gen  exact 4/6-cycle counts of every candidate, for each shipped
       code's truncated graph and the mother (all rows, IR).
  sim  paired BPSK/AWGN BLER against NR on each shipped (k, n): same
       bits and noise, BP 40 iterations (the modem's), the Eb/N0 of
       10% and 1% BLER by log-linear interpolation.
  deep the stage-1 pick per (bg, Z) against NR: 100k blocks just above
       NR's 1% point (floors), and the code at 2n (IR's extension rows).

  screen  one floor point per Z (where a pick floored) with early
       rejection: bad candidates die in 10-40k blocks.

First run (runs/own_shifts): uniform random shifts floor (4-cycles).
Greedy picks are within +0.03 dB of NR at 10% and 1%, but four BG2 picks
floored in stage 2. The floors were small trapping sets plus weight-11
codewords, and cycle counts did not predict them: g8 has fewer 8-cycles
than NR and still floored. Seeds of the same family spread 0-14 errors
per 100k at one point, so the floor screen is the filter. PICKS below.

    python -m scripts.own_shifts gen --seeds 32
    python -m scripts.own_shifts sim --top 3 --blocks 10000
    python -m scripts.own_shifts deep --blocks 100000
"""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from data2g import config, ldpc

OUT = Path("runs/own_shifts")

# The screened table per shipped (bg, Z): lift(mode, seed) regenerates it.
# BG1 and most of BG2: stage-1 g6 picks, clean in stage 2. BG2 Z=44/60/176/
# 192/240: the g6 pick floored; these won the single-point floor screen and
# then passed stage 2. Marginal, under 0.05 dB: Z=26 and 44 run 1.4-2x NR's
# BLER near 5e-4.
PICKS = {
    (1, 36): "g6-27", (1, 72): "g6-29", (1, 104): "g6-23", (1, 144): "g6-22", (1, 160): "g6-7",
    (2, 26): "g6-17", (2, 32): "g6-26", (2, 40): "g6-8", (2, 44): "g8-21", (2, 48): "g6-11",
    (2, 60): "g6-21", (2, 64): "g6-15", (2, 72): "g6-5", (2, 96): "g6-30", (2, 104): "g6-15",
    (2, 128): "g6-2", (2, 144): "g6-0", (2, 176): "g6-4", (2, 192): "g8-30", (2, 240): "g8-9",
    (2, 256): "g6-0",
}


def pick_table(bg: int, z: int) -> np.ndarray:
    mode, seed = PICKS[(bg, z)].split("-")
    return lift(Cycles(mask_of(bg), eight=mode != "g6"), bg, z, mode, int(seed))


def shipped() -> dict[tuple[int, int], list[tuple[str, int, int]]]:
    """(bg, z) -> [(submodes, k, n)] over the LDPC submodes; submodes
    with the same (k, n) are one code here (BPSK: no constellation)."""
    out = {}
    for s in config.SUBMODES.values():
        if s.code != "ldpc":
            continue
        q = ldpc.nr_code(s.k, s.coded_bits, shifts="nr")  # (bg, z) only; no table needed
        out.setdefault((1 if q.kb == 22 else 2, q.z), {}).setdefault((s.k, s.coded_bits), []).append(s.name)
    return {g: [("+".join(v), k, n) for (k, n), v in d.items()] for g, d in sorted(out.items())}


def mask_of(bg: int) -> np.ndarray:
    return ldpc.nr_base_graph(bg, 0) >= 0


def fixed_shifts(bg: int) -> np.ndarray:
    """Core and extension-diagonal shifts; -1 where free or absent."""
    m = mask_of(bg)
    kb = 22 if bg == 1 else 10
    f = np.full(m.shape, -1, dtype=np.int64)
    rows = np.flatnonzero(m[:4, kb])  # the weight-3 core column
    f[rows, kb] = [1, 0, 1]
    for c in range(kb + 1, kb + 4):
        f[m[:4, c].nonzero()[0], c] = 0
    for r in range(4, m.shape[0]):
        f[r, kb + r] = 0
    return f


class Cycles:
    """Base-graph 4- and 6-cycles as (edge ids, signs): the lifted cycle
    exists when sum(sign * shift[edge]) = 0 mod z."""

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
        if eight:  # BG2: 256k base 8-cycles; a = min row, neighbours b < d
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


def nr_table(bg: int, z: int) -> np.ndarray:
    i_ls = next(i for i, zs in enumerate(ldpc.LIFTING_SETS) if z in zs)
    b = ldpc.nr_base_graph(bg, i_ls)
    return np.where(b >= 0, b % z, -1)


def code_with(table: np.ndarray, bg: int, k: int, n: int) -> ldpc.QCLDPC:
    ref = ldpc.nr_code(k, n, bg=bg, shifts="nr")
    return ldpc.QCLDPC(base=table.copy(), z=ref.z, kb=ref.kb, k=k, n=n)


def cmd_gen(a):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    eight = bool({"g8", "ga"} & set(a.modes))
    for bg in a.bg:
        cyc = Cycles(mask_of(bg), eight=eight)
        Ls = sorted(cyc.c)
        for (g, z), subs in shipped().items():
            if g != bg:
                continue
            mbs = sorted({code_with(nr_table(bg, z), bg, k, n).mb for _, k, n in subs})
            mbs.append(cyc.mask.shape[0])  # mother
            tables = {"nr": nr_table(bg, z)}
            for mode in a.modes:
                for seed in range(a.seeds if mode != "rand" else 2):
                    tables[f"{mode}-{seed}"] = lift(cyc, bg, z, mode, seed)
            for name, t in tables.items():
                s = t[cyc.er, cyc.ec]
                if name != "nr":  # the encoder needs an invertible core
                    code_with(t, bg, *subs[0][1:])._core_inv
                r = {"bg": bg, "z": z, "cand": name}
                for mb in mbs:
                    for L, c in zip(Ls, cyc.counts(s, z, mb)):
                        r[f"c{L}_mb{mb}"] = c
                rows.append(r)
            np.savez_compressed(OUT / f"cands_bg{bg}_z{z}{a.tag}.npz", **tables)
            best = sorted((x for x in rows if x["bg"] == bg and x["z"] == z and x["cand"] != "nr"),
                          key=lambda x: rank_key(x))[:3]
            nr = next(x for x in rows if x["bg"] == bg and x["z"] == z and x["cand"] == "nr")
            fmt = lambda x: " ".join("/".join(str(x[f"c{L}_mb{m}"]) for L in Ls) for m in mbs)
            print(f"bg{bg} z={z:3d} mb {mbs}  {'/'.join(map(str, Ls))}-cycles  nr {fmt(nr)}  |  "
                  + "  ".join(f"{x['cand']} {fmt(x)}" for x in best), flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / f"cycles{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def rank_key(r):
    """Fewest 4-cycles anywhere, then 6-cycles in the shipped graphs
    (the smallest truncation first: it is in every code of this z)."""
    mbs = sorted(int(k[5:]) for k in r if k.startswith("c4_mb"))
    c4 = sum(r[f"c4_mb{m}"] for m in mbs)
    if f"c8_mb{mbs[0]}" in r:  # with 8-cycles: g8's own weighting, summed
        return (c4, sum(G8_W6 * r[f"c6_mb{m}"] + r[f"c8_mb{m}"] for m in mbs))
    return (c4,) + tuple(r[f"c6_mb{m}"] for m in mbs)


# The GPU is shared with the desktop: the decoder's (batch, checks, dmax)
# intermediates are sized to this, and the process is capped (gpu_cap).
VRAM_BUDGET = 1.0e9
VRAM_CAP = 0.12  # of the device; an overshoot is our OOM, not another app's


def gpu_cap(device):
    import torch

    if str(device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(VRAM_CAP)


def bler_curve(code, ebn0s, blocks, device, iters=40, seed=1, batch=2000):
    import torch

    dec = ldpc.MinSumDecoder(code, device=device)
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


def crossing(ebn0s, bl, target):
    """Eb/N0 where BLER falls through target, log-linear; nan if not bracketed."""
    lb = np.log10(np.maximum(bl, 1e-9))
    t = np.log10(target)
    for i in range(len(bl) - 1):
        if lb[i] >= t > lb[i + 1]:
            return ebn0s[i] + (t - lb[i]) / (lb[i + 1] - lb[i]) * (ebn0s[i + 1] - ebn0s[i])
    return float("nan")


def cmd_sim(a):
    import torch

    torch.set_num_threads(2)
    gpu_cap(a.device)
    with open(OUT / "cycles.csv") as f:
        cyc_rows = [{k: (v if k == "cand" else int(v)) for k, v in r.items() if v != ""} for r in csv.DictReader(f)]
    out_path = OUT / f"sim_{a.tag}.csv"
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {(int(r["k"]), int(r["n"]), r["cand"]) for r in csv.DictReader(f)}
    fh = open(out_path, "a", newline="")
    w = csv.writer(fh)
    if not done:
        w.writerow(["bg", "z", "submode", "k", "n", "cand", "ebn0_10", "ebn0_1", "d10_db", "d1_db", "bler_top"])
    for (bg, z), subs in shipped().items():
        if a.only and f"bg{bg}z{z}" not in a.only:
            continue
        tables = np.load(OUT / f"cands_bg{bg}_z{z}.npz")
        mine = [r for r in cyc_rows if r["bg"] == bg and r["z"] == z and r["cand"] != "nr"]
        cands = ["nr"] + [r["cand"] for r in sorted(mine, key=rank_key)[: a.top]] + a.extra
        for name, k, n in subs:
            if all((k, n, c) in done for c in cands):
                continue
            nr_code = code_with(tables["nr"], bg, k, n)
            # coarse sweep to place the grid around NR's waterfall
            lo = np.arange(-2.0, 8.01, 0.5)
            b0 = bler_curve(nr_code, lo, 2000, a.device)
            e_hi = lo[np.argmax(b0 < 3e-3)] if (b0 < 3e-3).any() else lo[-1]
            grid = np.round(np.arange(e_hi - 1.5, e_hi + 0.51, 0.25), 2)
            ref = None
            for cand in cands:
                if (k, n, cand) in done:
                    continue
                bl = bler_curve(code_with(tables[cand], bg, k, n), grid, a.blocks, a.device)
                e10, e1 = crossing(grid, bl, 0.1), crossing(grid, bl, 0.01)
                if cand == "nr":
                    ref = (e10, e1)
                elif ref is None:  # resumed past NR's row: rerun it
                    nb = bler_curve(nr_code, grid, a.blocks, a.device)
                    ref = (crossing(grid, nb, 0.1), crossing(grid, nb, 0.01))
                w.writerow([bg, z, name, k, n, cand, f"{e10:.3f}", f"{e1:.3f}",
                            f"{e10 - ref[0]:+.3f}", f"{e1 - ref[1]:+.3f}", f"{bl[-1]:.2e}"])
                fh.flush()
                print(f"bg{bg} z={z:3d} {name:16s} {cand:8s} 10% {e10:6.2f} ({e10 - ref[0]:+.2f})"
                      f"  1% {e1:6.2f} ({e1 - ref[1]:+.2f})  top {grid[-1]:.2f} dB: {bl[-1]:.1e}", flush=True)


def picks(stage1: Path) -> dict[tuple[int, int], str]:
    """(bg, z) -> the candidate with the least worst-case 1% loss to NR."""
    by = {}
    with open(stage1) as f:
        for r in csv.DictReader(f):
            if r["cand"] != "nr":
                by.setdefault((int(r["bg"]), int(r["z"])), {}).setdefault(r["cand"], []).append(float(r["d1_db"]))
    return {g: min(c, key=lambda n: (max(c[n]), np.mean(c[n]))) for g, c in by.items()}


def cmd_deep(a):
    """Floor: NR and the pick at NR's 1% point +0.25/+0.5/+0.75 dB, many
    blocks. IR: the same pair at n2 = min(2n, mother), stage-1 style."""
    import torch

    torch.set_num_threads(2)
    gpu_cap(a.device)
    path = OUT / f"deep_{a.tag}.csv"
    done = set()
    if path.exists():  # resume: a (code, cand)'s rows are written together, at its end
        with open(path) as f:
            done = {(r["submode"], r["cand"]) for r in csv.DictReader(f)}
    fh = open(path, "a", newline="")
    w = csv.writer(fh)
    if not done:
        w.writerow(["bg", "z", "submode", "k", "n", "cand", "test", "ebn0", "bler_nr", "bler_cand"])
    with open(OUT / "sim_stage1.csv") as f:
        e1 = {(int(r["k"]), int(r["n"])): float(r["ebn0_1"]) for r in csv.DictReader(f) if r["cand"] == "nr"}
    if a.cands_tag:  # the top of a gen ranking, untested by sim
        with open(OUT / f"cycles{a.cands_tag}.csv") as f:
            cyc_rows = [{k: (v if k == "cand" else int(v)) for k, v in r.items() if v != ""}
                        for r in csv.DictReader(f)]
        todo = {}
        for r in sorted((r for r in cyc_rows if r["cand"] != "nr"), key=rank_key):
            todo.setdefault((r["bg"], r["z"]), []).append(r["cand"])
        todo = {g: c[: a.top] for g, c in todo.items()}
    else:
        todo = {g: [c] for g, c in picks(OUT / "sim_stage1.csv").items()}
    tags = {}
    for p in a.pick:  # bg2z44=_g8:g8-21 (tag '' for the g6 set)
        key, val = p.split("=")
        bg, z = (int(x) for x in key[2:].split("z"))
        tags[(bg, z)], c = val.split(":")
        todo[(bg, z)] = [c]
    if a.pick:
        todo = {g: todo[g] for g in tags}
    for (bg, z), cands in todo.items():
        if a.only and f"bg{bg}z{z}" not in a.only:
            continue
        tables = np.load(OUT / f"cands_bg{bg}_z{z}{tags.get((bg, z), a.cands_tag)}.npz")
        for name, k, n in shipped()[(bg, z)]:
            left = [c for c in cands if (name, c) not in done]
            if not left:
                continue
            grid = np.round(e1[(k, n)] + np.array([0.25, 0.5, 0.75]), 2)
            bn = bler_curve(code_with(tables["nr"], bg, k, n), grid, a.blocks, a.device, seed=7)
            n2 = min(2 * n, code_with(tables["nr"], bg, k, n).mother().n)
            if n2 > n:
                nr2 = code_with(tables["nr"], bg, k, n2)
                lo = np.arange(-4.0, 6.01, 0.5)
                b0 = bler_curve(nr2, lo, 2000, a.device)
                e_hi = lo[np.argmax(b0 < 3e-3)] if (b0 < 3e-3).any() else lo[-1]
                g2 = np.round(np.arange(e_hi - 1.5, e_hi + 0.51, 0.25), 2)
                bn2 = bler_curve(nr2, g2, a.blocks // 10, a.device)
            for cand in left:
                rows = []
                bc = bler_curve(code_with(tables[cand], bg, k, n), grid, a.blocks, a.device, seed=7)
                for e, x, y in zip(grid, bn, bc):
                    rows.append([bg, z, name, k, n, cand, "floor", e, f"{x:.2e}", f"{y:.2e}"])
                print(f"bg{bg} z={z:3d} {name:16s} {cand:8s} floor " + "  ".join(
                    f"{e:.2f}: {x:.1e}/{y:.1e}" for e, x, y in zip(grid, bn, bc)), flush=True)
                if n2 > n:
                    bc = bler_curve(code_with(tables[cand], bg, k, n2), g2, a.blocks // 10, a.device)
                    d = [crossing(g2, bc, t) - crossing(g2, bn2, t) for t in (0.1, 0.01)]
                    for e, x, y in zip(g2, bn2, bc):
                        rows.append([bg, z, name, k, n2, cand, "ir", e, f"{x:.2e}", f"{y:.2e}"])
                    print(f"bg{bg} z={z:3d} {name:16s} {cand:8s} IR n={n2}  d10 {d[0]:+.2f}  d1 {d[1]:+.2f}"
                          f"  top {g2[-1]:.2f}: {bn2[-1]:.1e}/{bc[-1]:.1e}", flush=True)
                w.writerows(rows)
                fh.flush()


# The code and Eb/N0 where stage 2 saw each g6 pick floor (NR's errors per
# 100k there in the comment). One point per Z: a floor screen, not a curve.
FLOOR_POINTS = {
    (2, 44): (336, 1000, 1.91),  # NR 4
    (2, 60): (480, 1920, 1.50),  # IR; NR 0/10k
    (2, 176): (1680, 2880, 2.03),  # NR 1
    (2, 192): (1920, 2880, 2.45),  # NR 2
    (2, 240): (2400, 7680, 1.00),  # IR; NR 0/10k
    (2, 256): (2560, 3840, 2.37),  # NR 1
}


def cmd_screen(a):
    """NR at each floor point, then every candidate there with early
    rejection: stop once its errors pass max(3 x NR's, NR's + 10)."""
    import torch

    torch.set_num_threads(2)
    gpu_cap(a.device)
    fh = open(OUT / f"screen_{a.tag}.csv", "a", newline="")
    w = csv.writer(fh)
    for (bg, z), (k, n, ebn0) in FLOOR_POINTS.items():
        if a.only and f"bg{bg}z{z}" not in a.only:
            continue
        nr = code_with(nr_table(bg, z), bg, k, n)
        e_nr = int(round(bler_curve(nr, [ebn0], a.blocks, a.device, seed=11)[0] * a.blocks))
        limit = max(3 * e_nr, e_nr + 10)
        print(f"bg{bg} z={z:3d} k={k} n={n} @{ebn0} dB: NR {e_nr} errors / {a.blocks}, reject above {limit}", flush=True)
        w.writerow([bg, z, k, n, ebn0, "nr", e_nr, a.blocks])
        for tag in a.cands_tags:
            tables = np.load(OUT / f"cands_bg{bg}_z{z}{tag}.npz")
            names = [c for c in tables.files if c != "nr" and not c.startswith("rand")]
            with open(OUT / f"cycles{tag}.csv") as f:
                rk = {r["cand"]: rank_key({q: (v if q == "cand" else int(v)) for q, v in r.items() if v != ""})
                      for r in csv.DictReader(f) if int(r["bg"]) == bg and int(r["z"]) == z}
            names.sort(key=lambda c: rk[c])
            for cand in names[: a.top]:
                code = code_with(tables[cand], bg, k, n)
                errs = done = 0
                step = 10000
                while done < a.blocks and errs <= limit:
                    errs += int(round(bler_curve(code, [ebn0], step, a.device, seed=11 + done)[0] * step))
                    done += step
                verdict = "REJECT" if errs > limit else ("ok" if errs <= e_nr + 2 * np.sqrt(e_nr + 1) else "worse")
                print(f"  {tag or 'g6':4s} {cand:6s} {errs:4d} errors / {done:6d}  {verdict}", flush=True)
                w.writerow([bg, z, k, n, ebn0, f"{tag}:{cand}", errs, done])
                fh.flush()


def cmd_export(a):
    """PICKS -> the runtime's table file (data2g/codes_data/ldpc_shifts.npz)."""
    tables = {f"bg{bg}_z{z}": pick_table(bg, z).astype(np.int16) for bg, z in PICKS}
    missing = set(shipped()) - set(PICKS)
    if missing:
        raise SystemExit(f"shipped (bg, z) without a pick: {sorted(missing)}")
    np.savez_compressed(ldpc.OWN_SHIFTS, **tables)
    print(f"{len(tables)} tables -> {ldpc.OWN_SHIFTS}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    g = sub.add_parser("gen")
    g.add_argument("--seeds", type=int, default=32)
    g.add_argument("--modes", nargs="+", default=["g6", "g4", "rand"])
    g.add_argument("--bg", type=int, nargs="+", default=[1, 2])
    g.add_argument("--tag", default="", help="output suffix: cands_bg2_z96<tag>.npz, cycles<tag>.csv")
    s = sub.add_parser("sim")
    s.add_argument("--top", type=int, default=3)
    s.add_argument("--extra", nargs="*", default=[], help="candidates to add, e.g. rand-0")
    s.add_argument("--blocks", type=int, default=20000)
    s.add_argument("--only", nargs="*", default=[], help="bg1z72 ...")
    s.add_argument("--tag", default="stage1")
    s.add_argument("--device", default="cuda")
    d = sub.add_parser("deep")
    d.add_argument("--blocks", type=int, default=100000)
    d.add_argument("--tag", default="stage2")
    d.add_argument("--cands-tag", default="", help="test the top of cycles<tag>.csv, not stage-1 picks")
    d.add_argument("--top", type=int, default=2)
    d.add_argument("--only", nargs="*", default=[], help="bg2z192 ...")
    d.add_argument("--pick", nargs="*", default=[], help="bg2z44=_g8:g8-21 ... (tag '' = the g6 set)")
    d.add_argument("--device", default="cuda")
    c = sub.add_parser("screen")
    c.add_argument("--blocks", type=int, default=100000)
    c.add_argument("--cands-tags", nargs="+", default=["", "_g8", "_ga"], help="'' is the g6/g4 set")
    c.add_argument("--top", type=int, default=8, help="per tag, best by cycle rank")
    c.add_argument("--only", nargs="*", default=[])
    c.add_argument("--tag", default="floor")
    c.add_argument("--device", default="cuda")
    a = ap.parse_args()
    {"gen": cmd_gen, "sim": cmd_sim, "deep": cmd_deep, "screen": cmd_screen, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
