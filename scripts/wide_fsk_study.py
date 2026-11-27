"""Wide-bin FSK grids (cpm.WIDE_GRIDS): fronts, lock floors, frame error rates.

    python -m scripts.wide_fsk_study front --grid c16w125 --n 28
    python -m scripts.wide_fsk_study cal --grids c16w125 c8w50 c8r50
    DATA2G_CPM_WIDE=1 DATA2G_PEP_REF_DB=5 python -m scripts.wide_fsk_study e2e --out runs/wide_e2e.csv

`front`: a non-periodic front of n symbols (cpm.FRONTS): tones used evenly,
the fewest coincidences with itself at any other time shift and whole-bin
shift the detector searches (+-150 Hz). Hill climbing over swaps, restarts.

`cal`: on noise, the median coarse score of a 0.25 s hop (SYNC_THRESHOLD,
MID_THRESHOLD: detect's CPU floor) and the largest header scores at the
best placement (what LOCK_LLR / MID_LLR must clear).

`e2e`: scripts/cpm_floor_study.py's trial (control + one data codeword
through phy_session.ContinuousChannel, the same seed at every SNR and for
every mode), arms base (find / late lock / mid lock, cpm.receive) and genie
(known start, CFO and header). Rows: data and control codeword decoded.
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data2g import cpm  # noqa: E402
from data2g.config import FS  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

# A steady carrier inside every CPM grid (c8w50 / c8r50's fifth tone + 7 Hz), 18 dB over the floor in 50 Hz: a
# channel named "<chan>/incar" has it (master's per-tone floors, #295).
from data2g import interference as INTF  # noqa: E402

INTF.PRESETS["incar"] = INTF.Spec(qrm=(INTF.Qrm("carrier", 1507.0, 50.0, 18.0, 1e6, 1.0),))


def channel(PS, chan: str, snr: float, seed: int, horizon: float):
    """phy_session.ContinuousChannel for a CHANS name, "/<interference preset>" after it."""
    name, _, intf = chan.partition("/")
    kw = dict(interference=(INTF.PRESETS[intf],) * 2) if intf else {}
    return PS.ContinuousChannel("awgn", snr, seed, horizon, *(CHANS[name] or (None, None)), **kw)


# --- front ---------------------------------------------------------------------------

def coincidences(p: np.ndarray, dk_max: int) -> tuple:
    """(worst, shifts at it) of p[r] + dk == p[r + tau] counts over every (tau, dk) but (0, 0)."""
    n = len(p)
    worst, at = 0, 0
    for tau in range(n):
        for dk in range(-dk_max, dk_max + 1):
            if tau == 0 and dk == 0:
                continue
            c = int(np.sum(p[:n - tau] + dk == p[tau:]))
            if c > worst:
                worst, at = c, 1
            elif c == worst:
                at += 1
    return worst, at


def front(a):
    g = cpm.GRIDS[a.grid]
    dk = int(np.ceil(150.0 / g.rate))
    rng = np.random.default_rng(a.seed)
    best, bc = None, (99, 0)
    for _ in range(a.restarts):
        p = rng.permutation(np.arange(a.n) % g.m)
        c = coincidences(p, dk)
        for _ in range(a.steps):
            i, j = rng.integers(a.n, size=2)
            if p[i] == p[j]:
                continue
            q = p.copy()
            q[i], q[j] = q[j], q[i]
            cq = coincidences(q, dk)
            if cq <= c:
                p, c = q, cq
        if c < bc:
            best, bc = p, c
            print(bc, tuple(int(v) for v in best), flush=True)


# --- cal -----------------------------------------------------------------------------

def cal(a):
    for name in a.grids:
        g = cpm.GRIDS[name]
        rng = np.random.default_rng(1)
        lay = cpm.layout(name, cpm.stream_symbols(name, 0, False))
        hop = FS // 4
        front_n = (lay.sync_rows[lay.front - 1] + 2) * g.T
        mid_n = (len(cpm.mid_block(g)) + 1) * g.T
        sf, sm, h1, h2 = [], [], [], []
        for _ in range(a.n):
            x = rng.normal(size=hop + cpm.hdr_span(name, 2) + 2 * g.T)
            sc, s0, cfo = cpm.detect(g, x[:hop + front_n + g.T], front_only=True, fine=False)
            sf.append(sc)
            sm.append(cpm.detect(g, x[:hop + mid_n + g.T], mid=True, fine=False)[0])
            _, s0, cfo = cpm.detect(g, x[:hop + front_n + g.T], front_only=True)
            h1.append(cpm.read_header(g, x, max(s0, 0), cfo, 1)[3])
            h2.append(cpm.read_header(g, x, max(s0, 0), cfo, 2)[3])
        print(f"{name}: front {len(cpm.preamble_pattern(g))} mid {len(cpm.mid_block(g))} hdr {cpm.hdr_len(g)} symbols; "
              f"coarse median front {np.median(sf):.3f} mid {np.median(sm):.3f}; "
              f"header score max 1 copy {max(h1):.1f} 2 copies {max(h2):.1f} (99.9%: {np.percentile(h1, 99.9):.1f} "
              f"{np.percentile(h2, 99.9):.1f}), {a.n} hops")


# --- e2e -----------------------------------------------------------------------------

CHANS = {"awgn": None, "mpg": (0.1, 0.5), "mpp": (1.0, 2.0), "mpd": (2.0, 4.0), "hilat-m": (10.0, 3.0),
         "lowlat-d": (10.0, 6.0), "hilat-d": (30.0, 7.0), "nvis-d": (1.0, 7.0),
         # the search for a place: faster, longer, both, flutter
         "d60t7": (60.0, 7.0), "d100t7": (100.0, 7.0), "d30t12": (30.0, 12.0), "d10t16": (10.0, 16.0),
         "d60t12": (60.0, 12.0), "flutter": (10.0, 0.5), "d30t0.5": (30.0, 0.5), "d60t0.5": (60.0, 0.5),
         "d100t0.5": (100.0, 0.5), "d200t0.5": (200.0, 0.5), "d200t7": (200.0, 7.0), "d2t16": (2.0, 16.0)}


ARMS = ("base", "basec", "genie", "geniec")


def trial(args):
    """-> (mode, chan, snr, seed, {arm: decoded}): base / genie data, basec /
    geniec control, as the receiver ships; a wide mode also n<arm>, its
    coherent lock, control and DD off (last round's receiver), and tms /
    ntms, the genie lock's receive + decodes in ms. `w` (optional fifth
    item): the wide grid's data window, sub-window rows."""
    import time
    import outcome_data as O
    import phy_session as PS
    from dataclasses import replace
    from data2g.arq import phy as PHY
    from data2g.arq.modes import MODES
    from scripts.cpm_floor_study import receive_traced

    mode, chan, snr, seed = args[:4]
    spec = MODES[mode]
    g = cpm.GRIDS[spec.grid]
    rng = np.random.default_rng(seed)
    b = replace(O.burst(mode, 2, rng), cap=2)
    ch = channel(PS, chan, snr, seed, 120.0)
    y = ch.apply(PHY.tx_audio(b), float(rng.uniform(5, 100)), 0)  # (station 0: a channel's interference is per receiver)

    def dec(r, slot):
        if r is None or r["spec"].name != mode or r["n_cw"] != 2:
            return 0
        r = dict(r)
        r.pop("_path", None)
        return int(PHY.ModemRx(r, {}).decode(slot, b.slots[slot].mask_id, 0, None) == b.slots[slot].payload)

    def run(coh: bool, pre: str, out: dict):
        keep = cpm.COH_CTL, cpm.COH_LOCK.get(g.name), PHY.CPM_DD, cpm.WIDE_COH_W.get(g.name)
        try:
            if not coh:
                cpm.COH_CTL = PHY.CPM_DD = False
                cpm.COH_LOCK.pop(g.name, None)
            if len(args) > 4:
                cpm.WIDE_COH_W[g.name] = (args[4], keep[3][1])
            r = receive_traced(y, mode)
            t = time.perf_counter()
            gr = cpm.receive(y, dict(spec=spec, n_data=1, dup=False, start=int(PS.PAD_S * FS), cfo=PS.CFO_HZ, band=g.name))
            out.update({pre + "genie": dec(gr, 1), pre + "geniec": dec(gr, 0)})
            out[pre + "tms"] = round(1000 * (time.perf_counter() - t), 1)
            out.update({pre + "base": dec(r, 1), pre + "basec": dec(r, 0)})
            out[pre + "lock"] = 0 if r is None else (2 if r["spec"].name == mode and r["n_cw"] == 2 else 1)
        finally:
            cpm.COH_CTL, PHY.CPM_DD = keep[0], keep[2]
            if keep[1] is not None:
                cpm.COH_LOCK[g.name] = keep[1]
            if keep[3] is not None:
                cpm.WIDE_COH_W[g.name] = keep[3]

    out = {}
    run(True, "", out)
    if g.L > 1:
        run(False, "n", out)
    return mode, chan, snr, seed, out


def cohw(a):
    """The wide grids' data window: genie-lock decodes at each W (sub-window rows) near the 10% points."""
    pts = {"fsk16w125-r1/3": {"awgn": -16.5, "mpp": -11.5, "hilat-m": -13.0, "hilat-d": -11.0, "nvis-d": -10.5},
           "fsk8w50-r1/3": {"awgn": -19.5, "mpp": -16.5, "hilat-m": -17.0, "hilat-d": -16.0, "nvis-d": -16.0}}
    jobs = [(m, c, s, 50000 + i, w) for m, cs in pts.items() for c, s in cs.items() for w in a.ws for i in range(a.trials)]
    acc = {}
    with Pool(a.jobs) as p:
        for (m, c, s, seed, out), job in zip(p.imap(trial, jobs, chunksize=4), jobs):
            acc.setdefault((m, c, job[4]), []).append(out)
    print("decoded, genie lock: data / control at each W; nc: the square law")
    for m, cs in pts.items():
        for c, s in cs.items():
            nc = acc[m, c, a.ws[0]]
            cells = "  ".join(f"W{w} {np.mean([o['genie'] for o in acc[m, c, w]]):.2f}/"
                              f"{np.mean([o['geniec'] for o in acc[m, c, w]]):.2f}" for w in a.ws)
            print(f"  {m} {c:8s} {s:6.1f} dB  nc {np.mean([o['ngenie'] for o in nc]):.2f}/"
                  f"{np.mean([o['ngeniec'] for o in nc]):.2f}  {cells}")


def cohcal(a):
    """coh_read_header's largest scores on noise at detect's placements (COH_LOCK must clear them)."""
    for name in a.grids:
        g = cpm.GRIDS[name]
        rng = np.random.default_rng(2)
        lay = cpm.layout(name, cpm.stream_symbols(name, 0, False))
        hop, front_n = FS // 4, (lay.sync_rows[lay.front - 1] + 2) * g.T
        h = [[], []]
        for _ in range(a.n):
            x = rng.normal(size=hop + cpm.hdr_span(name, 2) + 2 * g.T)
            _, s0, cfo = cpm.detect(g, x[:hop + front_n + g.T], front_only=True)
            for i in (0, 1):
                h[i].append(cpm.coh_read_header(g, x, max(s0, 0), cfo, i + 1)[3])
        print(f"{name}: coherent header score on noise, max 1 copy {max(h[0]):.1f} 2 copies {max(h[1]):.1f} "
              f"(99.9%: {np.percentile(h[0], 99.9):.1f} {np.percentile(h[1], 99.9):.1f}), {a.n} hops")


# --- connecting ----------------------------------------------------------------------

def connect_trial(args):
    """One CONNECT through the real sessions and modem (linksim.run on
    phy_session.RealPhy), the policy's rule as the environment set it:
    -> (chan, snr, seed, connected, seconds to the caller's CONNECTED, caller's bursts)."""
    import random
    import linksim as L
    import phy_session as G
    from data2g.arq import session as S

    chan, snr, seed, horizon = args
    at = {}
    orig = S.Session._connected

    def connected(self, now):
        at.setdefault(self.call, now)
        return orig(self, now)
    S.Session._connected = connected
    try:
        ch = channel(G, chan, snr, seed, horizon + 60)
        res = L.run(L.make_policy("shift+cpm"), L.make_policy("shift+cpm"), None, [("a", 64, None, 0.0)], seed=seed,
                    horizon=horizon, phy=G.RealPhy(ch), cap=2)
    finally:
        S.Session._connected = orig
    t = at.get("W1AW")
    return chan, snr, seed, int(t is not None), -1.0 if t is None else round(t, 1), res.get("bursts", res.get("stats", {}).get("bursts", -1))


def connect(a):
    import os

    rule = os.environ.get("ARM") or os.environ.get("DATA2G_ROBUST_RULE", "") or ("today" if not cpm.ON_AIR_WIDE else "today+grid")
    cells = [ln.split() for ln in open(a.plan) if ln.strip()]  # chan lo hi step [attempts]
    jobs = [(c, float(s), 70000 + i, a.horizon) for c, lo, hi, step, *n in cells
            for s in np.arange(float(lo), float(hi) + 0.01, float(step)) for i in range(int(n[0]) if n else a.trials)]
    with Pool(a.jobs) as p, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "channel", "snr", "seed", "connected", "t", "bursts"])
        for r in p.imap_unordered(connect_trial, jobs, chunksize=2):
            w.writerow([rule, *r])
            f.flush()


def connect_table(a):
    rows = [r for path in a.csv for r in csv.DictReader(open(path))]
    rules = list(dict.fromkeys(r["rule"] for r in rows))
    print("connected (share of attempts) and the median seconds to connect of those; rules:", ", ".join(rules))
    for c in dict.fromkeys(r["channel"] for r in rows):
        for s in sorted({float(r["snr"]) for r in rows if r["channel"] == c}):
            cell = []
            for rule in rules:
                sel = [r for r in rows if r["channel"] == c and float(r["snr"]) == s and r["rule"] == rule]
                ok = [float(r["t"]) for r in sel if r["connected"] == "1"]
                cell.append(f"{len(ok) / max(len(sel), 1):4.2f} {np.median(ok) if ok else float('nan'):5.1f}s")
            print(f"  {c:9s} {s:6.1f}  " + "   ".join(cell) + f"   n {len(sel)}")


def cpu(a):
    """Listening CPU on noise: the streaming Receiver, CPM grids only."""
    import time
    from data2g import modem, receiver

    y = np.random.default_rng(3).normal(size=int(a.seconds * FS))

    def run(grids, share=True):
        receiver.SHARE_ROWS = share
        rx = receiver.Receiver(modem.Accept((), 0.0), cpm_grids=grids)
        t = time.process_time()
        for i in range(0, len(y), FS // 10):
            rx.feed(y[i:i + FS // 10])
        return 100 * (time.process_time() - t) / a.seconds
    on = ("c16r25", "c8r50", "c32r62")
    for label, grids, share in ([(g, (g,), True) for g in (*on, "c8w50", "c16w125")]
                                + [("three on air", on, True), ("four, c8w50 its own transforms", (*on, "c8w50"), False),
                                   ("four, c8w50 from c8r50's rows", (*on, "c8w50"), True)]):
        print(f"{label:34s} {min(run(grids, share) for _ in range(a.repeat)):6.2f}% of a core", flush=True)


def acq_one(args):
    """Where a burst the genie decoded got no lock: the coarse score at the
    best placement and whether it is the burst's, and the header scores at
    the true start (square law and coherent, one copy and both)."""
    import outcome_data as O
    import phy_session as PS
    from dataclasses import replace
    from data2g.arq import phy as PHY
    from data2g.arq.modes import MODES
    from data2g.receiver import search_span

    mode, chan, snr, seed = args
    spec = MODES[mode]
    g = cpm.GRIDS[spec.grid]
    rng = np.random.default_rng(seed)
    b = replace(O.burst(mode, 2, rng), cap=2)
    ch = channel(PS, chan, snr, seed, 120.0)
    y = ch.apply(PHY.tx_audio(b), float(rng.uniform(5, 100)), 0)  # (station 0: a channel's interference is per receiver)
    s0, cfo = int(PS.PAD_S * FS), PS.CFO_HZ
    head = y[:s0 + FS // 2 + search_span((), (g.name,))]
    sc, sd, cd = cpm.detect(g, head, front_only=True)
    n1, n2 = cpm.read_header(g, y, s0, cfo, 1), cpm.read_header(g, y, s0, cfo, 2)
    c1, c2 = cpm.coh_read_header(g, y, s0, cfo, 1), cpm.coh_read_header(g, y, s0, cfo, 2)
    ok = lambda h: h[0].name == mode and h[1] == 1  # noqa: E731
    return dict(chan=chan, coarse=sc, placed=abs(sd - s0) <= g.T // 4 and abs(cd - cfo) < g.rate / 4,
                n1=n1[3] if ok(n1) else -1.0, n1m=n1[3] - n1[4], n2=n2[3] if ok(n2) else -1.0, n2m=n2[3] - n2[4],
                c1=c1[3] if ok(c1) else -1.0, c1m=c1[3] - c1[4], c2=c2[3] if ok(c2) else -1.0, c2m=c2[3] - c2[4],
                front=n1[5])


def acqdiag(a):
    lost = [(r["mode"], r["channel"], float(r["snr"]), int(r["seed"])) for path in a.csv for r in csv.DictReader(open(path))
            if r["mode"] in a.modes and float(r["genie"]) == 1 and float(r["base"]) == 0 and float(r["lock"]) == 0]
    with Pool(a.jobs) as p:
        out = p.map(acq_one, lost, chunksize=4)
    for mode in a.modes:
        g = cpm.GRIDS[cpm.SPECS[mode].grid]
        o = [x for x, k in zip(out, lost) if k[0] == mode]
        if not o:
            continue
        q = lambda k: np.round(np.percentile([x[k] for x in o], [10, 50, 90]), 2)  # noqa: E731
        lk, ck = cpm.LOCK_LLR[g.name], cpm.COH_LOCK[g.name]
        print(f"{mode}: {len(o)} bursts with no lock that the genie decoded")
        print(f"  coarse score p10/50/90 {q('coarse')} (floor {cpm.SYNC_THRESHOLD[g.name]}); under the floor "
              f"{np.mean([x['coarse'] < cpm.SYNC_THRESHOLD[g.name] for x in o]):.2f}; best placement not the burst's "
              f"{np.mean([not x['placed'] for x in o]):.2f}")
        print(f"  at the true start: front LLR {q('front')}; square law 1 copy {q('n1')} (gate {lk[0]}), both {q('n2')} "
              f"({lk[1]}); coherent {q('c1')} ({ck[0]}), both {q('c2')} ({ck[1]})")
        for d in (0, 2, 4, 6):
            got = np.mean([x["placed"] and x["coarse"] >= cpm.SYNC_THRESHOLD[g.name]
                           and ((x["n2"] >= lk[1] - d and x["n2m"] >= cpm.HDR_MARGIN[1])
                                or (x["c2"] >= ck[1] - d and x["c2m"] >= ck[3])) for x in o])
            alln = np.mean([(x["n2"] >= lk[1] - d and x["n2m"] >= cpm.HDR_MARGIN[1])
                            or (x["c2"] >= ck[1] - d and x["c2m"] >= ck[3]) for x in o])
            print(f"  both-copy gates {d} lower: {got:.2f} would lock as placed; {alln:.2f} at the true start")


def crossing(xs, fail, target):
    """The lowest SNR from which the failure rate stays at or under target (interpolated)."""
    k = len(xs)
    while k > 0 and fail[k - 1] <= target:
        k -= 1
    if k == 0:
        return float("-inf")  # under the grid
    if k == len(xs):
        return float("inf")
    a, b = fail[k - 1], fail[k]
    return xs[k - 1] + (xs[k] - xs[k - 1]) * (a - target) / max(a - b, 1e-9)


def summarize(rows, modes, chans):
    print("cell SNR (dB) at 10% failure: data base/genie, control base/genie; then the failure floor at the top SNR")
    for c in chans:
        for m in modes:
            snrs = sorted({r[2] for r in rows if r[0] == m and r[1] == c})
            if not snrs:
                continue
            cell = []
            sel = [r for r in rows if r[0] == m and r[1] == c]
            for pre in ("", "n") if "nbase" in sel[0][4] else ("",):
                cell = []
                for arm in ("base", "genie", "basec", "geniec"):
                    fail = [1 - np.mean([r[4][pre + arm] for r in sel if r[2] == s]) for s in snrs]
                    cell.append(f"{crossing(snrs, fail, 0.1):6.1f}")
                fb = [1 - np.mean([r[4][pre + "base"] for r in sel if r[2] == s]) for s in snrs]
                tms = np.mean([r[4][pre + "tms"] for r in sel])
                print(f"  {c:9s} {m + (' nc' if pre else ''):18s} data {cell[0]} /{cell[1]}  ctl {cell[2]} /{cell[3]}   "
                      f"{snrs[0]:.0f}..{snrs[-1]:.0f} dB, top: base {fb[-1]:.2f}  n {len(sel) // len(snrs)}  {tms:.0f} ms")


COLS = [p + k for p in ("", "n") for k in ("base", "basec", "genie", "geniec", "lock", "tms")]


def e2e(a):
    if a.plan:  # "mode chan lo hi" lines: each cell's own SNR range
        cells = [ln.split() for ln in open(a.plan) if ln.strip()]
        jobs = [(m, c, float(s), 50000 + i) for m, c, lo, hi in cells
                for s in np.arange(float(lo), float(hi) + 0.01, a.step) for i in range(a.trials)]
        a.modes, a.chans = list(dict.fromkeys(c[0] for c in cells)), list(dict.fromkeys(c[1] for c in cells))
    else:
        jobs = [(m, c, float(s), 50000 + i) for m in a.modes for c in a.chans
                for s in (np.arange(a.alo, a.ahi + 0.01, a.step) if c == "awgn" else np.arange(a.lo, a.hi + 0.01, a.step))
                for i in range(a.trials)]
    rows = []
    with Pool(a.jobs) as p, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "channel", "snr", "seed"] + COLS)
        for m, c, s, seed, out in p.imap_unordered(trial, jobs, chunksize=4):
            w.writerow([m, c, s, seed] + [out.get(k, "") for k in COLS])
            rows.append((m, c, s, seed, out))
            f.flush()
    summarize(rows, a.modes, a.chans)


def table(a):
    rows = []
    for path in a.csv:
        for r in csv.DictReader(open(path)):
            rows.append((r["mode"], r["channel"], float(r["snr"]), int(r["seed"]),
                         {k: float(r[k]) for k in COLS if r.get(k, "") != ""}))
    summarize(rows, list(dict.fromkeys(r[0] for r in rows)), list(dict.fromkeys(r[1] for r in rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["front", "cal", "e2e", "table", "probe", "cohw", "cohcal", "connect", "connect-table", "cpu", "acqdiag"])
    ap.add_argument("--grid", default="c16w125")
    ap.add_argument("--grids", nargs="+", default=list(cpm.WIDE_GRIDS))
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--restarts", type=int, default=30)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--modes", nargs="+", default=["fsk16w125-r1/3", "fsk16w125-r1/2", "fsk32r62-r1/3", "fsk32r62-r1/2",
                                                   "fsk8w50-r1/3", "fsk8w50-r1/2", "fsk16r25-r1/3", "fsk16r25-r1/2",
                                                   "fsk8r50-r1/3", "fsk8r50-r1/2"])
    ap.add_argument("--chans", nargs="+", default=["awgn", "mpg", "mpp", "mpd", "hilat-m", "lowlat-d", "hilat-d", "nvis-d"])
    ap.add_argument("--alo", type=float, default=-24.0, help="awgn: cell SNRs alo..ahi")
    ap.add_argument("--ahi", type=float, default=-9.0)
    ap.add_argument("--lo", type=float, default=-18.0, help="fading: lo..hi")
    ap.add_argument("--hi", type=float, default=9.0)
    ap.add_argument("--step", type=float, default=3.0)
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--plan", default="", help="e2e: a file of 'mode chan lo hi' cells")
    ap.add_argument("--ws", nargs="+", type=int, default=[4, 8, 16, 32])
    ap.add_argument("--horizon", type=float, default=80.0, help="connect: seconds a CONNECT gets")
    ap.add_argument("--seconds", type=float, default=120.0, help="cpu: audio")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out", default="runs/wide_e2e.csv")
    ap.add_argument("--csv", nargs="+", default=["runs/wide_e2e.csv"])
    a = ap.parse_args()
    if a.cmd == "probe":
        import resource
        import time

        for m in a.modes:
            t = time.time()
            print(trial((m, "awgn", 20.0, 5)), trial((m, "hilat-d", 0.0, 5))[4], round(time.time() - t, 1), "s")
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3, "MB peak RSS")
        return
    {"front": front, "cal": cal, "e2e": e2e, "table": table, "cohw": cohw, "cohcal": cohcal, "connect": connect,
     "connect-table": connect_table, "cpu": cpu, "acqdiag": acqdiag}[a.cmd](a)


if __name__ == "__main__":
    main()
