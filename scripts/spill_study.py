"""CPM soft values on fast fading (2026-11-23): candidates for cpm.tone_metric's
Doppler-spill handling, judged on fixed bursts (control + one data codeword;
c32_rate_study's channel: CFO +-50 Hz, 10 ppm, random lead, PEP-referenced
SNR as DATA2G_PEP_REF_DB=5 cells).

Every trial is received once by the shipped receiver (its locks untouched),
and each arm remakes the soft values from that result's energies and bins,
then decodes as arq.phy does (square law, then cpm DD): arms are paired on
the same lock, fades and noise.

    python -m scripts.spill_study profile [snr]
    python -m scripts.spill_study cache --cells fsk32r62-r2/3:hilat-d:18 ... --out runs/spill_dev.pkl
    python -m scripts.spill_study eval runs/spill_dev.pkl --arms w1 w0.5 leak2
    python -m scripts.spill_study run --out runs/spill_fixed.csv --arms w1 leak2
"""

import os

os.environ.setdefault("DATA2G_CPM_R23", "1")  # fsk32r62-r2/3 offered: not a mode by default

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, cpm, hfchannel  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402
from data2g.arq.modes import MODES, ctl_spec  # noqa: E402
from data2g.config import FS  # noqa: E402
from data2g.receiver import receive_any  # noqa: E402
from data2g.waveform import dsp  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import outcome_data as O  # noqa: E402

PRESETS = {**hfchannel.FADING_PRESETS, **hfchannel.ITU_PRESETS}
CHANS = ("awgn", "mpg", "mpp", "mpd", "hilat-m", "lowlat-d", "hilat-d", "nvis-d")
PEP_REF_DB = 5.0
CARRIER_DB = 12.0
WORKERS = 4


def make(name, chan, snr, seed, n=2):
    rng = np.random.default_rng(seed)
    b = O.burst(name, n, rng)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    x = np.concatenate([np.zeros(lead), PHY.tx_audio(b), np.zeros(FS // 2)])
    s_power = np.max(dsp.envelope(x) ** 2) / 2 / 10 ** (PEP_REF_DB / 10)
    y = hfchannel.freq_shift(hfchannel.sample_clock_offset(x.astype(np.float64), 10), float(rng.uniform(-50, 50)))
    chan, carrier = chan.removesuffix("+c"), chan.endswith("+c")
    if chan != "awgn":
        y = hfchannel.fading(y, PRESETS[chan], seed=seed)
    y = hfchannel.awgn(y, snr, seed=seed + 1, s_power=s_power)
    if carrier:  # "+c": a steady carrier in the grid, CARRIER_DB over the noise in 50 Hz (tests/test_cpm.py's, #295's)
        g = cpm.GRIDS[MODES[name].grid]
        sigma2 = s_power * (FS / 2) / 2500 / 10 ** (snr / 10)
        amp = np.sqrt(2 * sigma2 * 50 / (FS / 2) * 10 ** (CARRIER_DB / 10))
        y = y + amp * np.cos(2 * np.pi * (g.f0 + (g.m // 2 + 0.44) * g.rate) * np.arange(len(y)) / FS + rng.uniform(0, 2 * np.pi))
    return b, y


def sent_tones(b) -> np.ndarray:
    """The burst's stream tones: r["E"]'s rows."""
    spec = MODES[b.submode]
    g = cpm.GRIDS[spec.grid]
    return np.concatenate([cpm.to_tones(g, codes.encode(ctl_spec(spec) if s.mask_id[2] >= 128 else spec, s.payload, s.rv,
                                                        PHY.mask_value(s.mask_id), i)) for i, s in enumerate(b.slots)])


def received(name, chan, snr, seed):
    """-> (burst, the shipped receiver's result or None if it missed the burst or read another header)."""
    b, y = make(name, chan, snr, seed)
    try:
        r = receive_any(y, lead=FS)
    except Exception:  # noqa: BLE001 (a header read past the audio: a miss)
        r = None
    if r is None or r["spec"].name != name or r["n_cw"] != len(b.slots) or r["dup"]:  # dup: another word of this length
        return b, None
    return b, r


# --- candidates: tone_metric(g, E, use) replacements ----------------------------------

TONE_METRIC = cpm.tone_metric  # the shipped one
N0 = cpm._n0


def level(E):
    """tone_metric's N0 (before spill) and Es, and each row's strongest tone."""
    top = E.max(axis=1)
    n0 = np.median(np.sort(E, axis=1)[:, :-1]) / np.log(2)
    n0 = max(n0, 1e-12 * np.mean(top) + 1e-300)
    return n0, max(np.mean(top) - n0, 1e-3 * n0), E.argmax(axis=1)


def shares(E, n0, es, a, D):
    """Per offset 1..D, the share of Es a bin that far from each row's strongest holds over N0 (cpm.spill's, D = 1)."""
    P = np.pad(E, ((0, 0), (D, D)), constant_values=n0)
    r = np.arange(len(E))
    return [float(np.clip(np.mean(P[r, a + D - d] + P[r, a + D + d] - 2 * n0) / 2 / es, 0.0, 1.0)) for d in range(1, D + 1)]


def weighted(w_by_use):
    def f(g, E, use="lock"):
        old = dict(cpm.SPILL)
        cpm.SPILL[use] = w_by_use.get(use, 1.0)
        try:
            return TONE_METRIC(g, E, use)
        finally:
            cpm.SPILL.update(old)
    return f


def leak(D=2, isi=False, uses=("data", "dd", "ctl"), k=0.0):
    """Structured: the sent tone's bin has mean N0 + Es, a bin d away N0 + a_d Es (a_d measured on the codeword's
    own rows around each strongest tone), the rest N0, independent exponentials: log P(E | t) up to a row constant
    is sum_d E[t + d] (1 / N0 - 1 / mu_d) - ln(mu_d / N0) over the offsets inside the grid. `isi`: the same for the
    previous and next rows' strongest tones (delay spread), a bin of mean N0 + b Es under every hypothesis but
    those within D of it. `k`: N0 raised by k x the adjacent bins' share of Es on every bin (what the model
    leaves out is the signal's own too: without it the confidence grows with SNR, as #244's)."""
    def f(g, E, use="lock"):
        if use not in uses:
            return TONE_METRIC(g, E, use)
        n0, es, a = level(E)
        sh = [1.0] + shares(E, n0, es, a, D)
        n0 = n0 + k * sh[1] * es
        gd = [1 / n0 - 1 / (n0 + s * es) for s in sh]
        ld = [-np.log1p(s * es / n0) for s in sh]
        n, m = E.shape
        P = np.pad(E, ((0, 0), (D, D)))
        inside = np.pad(np.ones(m), (D, D))
        M = np.zeros((n, m))
        for d in range(-D, D + 1):
            M += gd[abs(d)] * P[:, D + d:D + d + m] + ld[abs(d)] * inside[D + d:D + d + m]
        if isi:
            r = np.arange(n)
            for nb in (np.roll(a, 1), np.roll(a, -1)):
                far = np.abs(nb - a) > D + 1
                far[[0, -1]] = False
                if far.sum() < 8:
                    continue
                b = float(np.clip(np.mean(E[r[far], nb[far]] - n0) / es, 0.0, 1.0))
                bonus = E[r, nb] * (1 / n0 - 1 / (n0 + b * es)) - np.log1p(b * es / n0)
                M += np.where(np.abs(np.arange(m)[None] - nb[:, None]) > D, bonus[:, None], 0.0)
        return M
    return f


def mlscale(uses=("data", "dd", "ctl")):
    """The square law's scale by maximum likelihood on the codeword's own energies: s maximizing
    sum_rows ln mean_t exp(s E_t) + n ln(1 - s N0) (the sent tone's bin exponential of mean N0 / (1 - s N0))."""
    def f(g, E, use="lock"):
        if use not in uses:
            return TONE_METRIC(g, E, use)
        n0, es, _ = level(E)
        best, arg = -np.inf, None
        for u in np.linspace(0.02, 0.998, 60):  # u = s N0 = Es / (N0 + Es)
            s = u / n0
            v = np.sum(np.logaddexp.reduce(s * E, axis=1)) + len(E) * np.log1p(-u)
            if v > best:
                best, arg = v, s
        return E * arg
    return f


def floor(metric, F, uses=("data",)):
    """`metric`, each row's tones at least -F nats under its best: no one row can rule a tone out by more
    (a model the row does not fit, e.g. the previous symbol's tone through a delayed path, costs at most F)."""
    def f(g, E, use="lock"):
        G = metric(g, E, use)
        return np.maximum(G - G.max(axis=1, keepdims=True), -F) if use in uses else G
    return f


DD_SOFT = cpm.dd_soft


def dd_floor(F):
    """cpm.dd_soft, a data slot's symbol log-likelihoods floored at -F (shipped: -60)."""
    def f(r, metric, slot):
        if slot < r["n_ctl_slots"]:
            return DD_SOFT(r, metric, slot)
        L = np.clip(metric - metric.max(axis=1, keepdims=True), -F, 0)
        return L[:, cpm._gray(cpm.GRIDS[r["band"]].m)].reshape(-1)
    return f


def with_n0(metric, k):
    """`metric`, and dd_metric's coherent N0 raised by k x the adjacent bins' share of Es."""
    def n0(E):
        base = N0(E)
        b, es, a = level(E)
        return base + k * shares(E, b, es, a, 1)[0] * es
    return metric, n0


ARMS = {
    "w1": weighted({}),
    "w0.75": weighted(dict(data=0.75)), "w0.5": weighted(dict(data=0.5)), "w0.25": weighted(dict(data=0.25)),
    "w0": weighted(dict(data=0.0)),
    "w0.5dd": weighted(dict(data=0.5, dd=0.5)), "w0.25dd": weighted(dict(data=0.25, dd=0.25)),
    "c0.5": weighted(dict(ctl=0.5, dd=0.5)), "c0": weighted(dict(ctl=0.0, dd=0.0)),
    "leak1": leak(1), "leak2": leak(2), "leak3": leak(3), "leak2i": leak(2, True), "leak1i": leak(1, True),
    "leak2d": leak(2, uses=("data",)), "leak2id": leak(2, True, uses=("data",)),
    "ml": mlscale(), "mld": mlscale(uses=("data",)),
    "w1n": with_n0(weighted({}), 1.0), "leak2n": with_n0(leak(2), 1.0), "leak2in": with_n0(leak(2, True), 1.0),
    "leak2n.5": with_n0(leak(2), 0.5), "w0.5n": with_n0(weighted(dict(data=0.5)), 1.0),
}
def shipped_leak(uses, ctl_sq=0.15):
    """cpm.leak_metric as cpm.tone_metric makes it under DATA2G_CPM_LEAK=`uses` (the port of leak(1, k=0.5))."""
    def f(g, E, use="lock"):
        old = cpm.LEAK
        cpm.LEAK = frozenset(uses)
        try:
            return TONE_METRIC(g, E, use)
        finally:
            cpm.LEAK = old
    f.leak, f.ctl_sq = frozenset(uses), ctl_sq  # judge sets them for resoft's ctl_square and dd_metric
    return f


ARMS["L"] = shipped_leak(("ctl", "data", "dd"))
ARMS["LS"] = shipped_leak(("ctl", "data", "dd", "ctlsq"), 0.15)
ARMS["LS.1"] = shipped_leak(("ctl", "data", "dd", "ctlsq"), 0.1)
ARMS["Lsq"] = (shipped_leak(("ctl", "data", "dd")), N0, DD_SOFT, False)  # the control on the square law (leak), not coherent
ARMS["w1sq"] = (weighted({}), N0, DD_SOFT, False)
ARMS["Ld"] = shipped_leak(("data",))
ARMS["Ldd"] = shipped_leak(("data", "dd"))
for _k in (0.25, 0.5, 1.0):
    ARMS[f"leak2ik{_k:g}"] = leak(2, True, k=_k)
    ARMS[f"leak2k{_k:g}"] = leak(2, k=_k)
    ARMS[f"leak1ik{_k:g}"] = leak(1, True, k=_k)
    ARMS[f"leak1k{_k:g}"] = leak(1, k=_k)
    ARMS[f"leak2ik{_k:g}d"] = leak(2, True, uses=("data",), k=_k)
for _F in (4, 6, 8, 10, 14):
    ARMS[f"w0f{_F}"] = floor(weighted(dict(data=0.0)), _F)
    ARMS[f"w1f{_F}"] = floor(weighted({}), _F)
    ARMS[f"w0.5f{_F}"] = floor(weighted(dict(data=0.5)), _F)
    ARMS[f"leak2f{_F}"] = floor(leak(2), _F)
    ARMS[f"leak1f{_F}"] = floor(leak(1), _F)
    ARMS[f"leak2if{_F}"] = floor(leak(2, True), _F)
    ARMS[f"leak2if{_F}F"] = (floor(leak(2, True), _F), N0, dd_floor(_F))  # the DD passes' soft values floored too
    ARMS[f"w0f{_F}F"] = (floor(weighted(dict(data=0.0)), _F), N0, dd_floor(_F))
    ARMS[f"w1F{_F}"] = (weighted({}), N0, dd_floor(_F))


def resoft(r: dict) -> dict:
    """cpm.receive's soft values remade from its energies and bins under the cpm module as it is patched now."""
    g = cpm.GRIDS[r["band"]]
    E, nc = r["E"], r["n_ctl_slots"]
    sizes = [cpm.CTL[g.name].n_sym] * nc + [cpm.DATA_N // g.bits] * (r["n_cw"] - nc)
    soft, i = [], 0
    for j, n in enumerate(sizes):
        soft.append((cpm.llrs if j < nc else cpm.symbol_ll)(g, E[i:i + n]))
        i += n
    r = dict(r, soft=soft)
    if cpm.COH_CTL:
        for s in range(nc):
            if not cpm.ctl_square(g, E, s):
                soft[s] = cpm.dd_soft(r, cpm.dd_metric(r, {}, s), s)
    return r


def judge(b, r, arm: str) -> tuple:
    """-> (control decoded, data decoded on the square law alone, data decoded)."""
    a = ARMS[arm]
    metric, n0, dds, coh = (a + (DD_SOFT, True))[:4] if isinstance(a, tuple) else (a, N0, DD_SOFT, True)
    cpm.tone_metric, cpm._n0, cpm.dd_soft, cpm.COH_CTL = metric, n0, dds, coh
    cpm.LEAK, cpm.CTL_SQ = getattr(metric, "leak", frozenset()), getattr(metric, "ctl_sq", 0.15)
    try:
        r2 = resoft(r)
        first = bool(codes.decode_posterior(r["spec"], r2["soft"][-1])[1])
        rx = PHY.ModemRx(r2, {})
        ok = [rx.decode(i, s.mask_id, 0, None) == s.payload for i, s in enumerate(b.slots)]
        return int(ok[0]), int(first), int(ok[-1])
    finally:
        cpm.tone_metric, cpm._n0, cpm.dd_soft, cpm.COH_CTL = TONE_METRIC, N0, DD_SOFT, True
        cpm.LEAK, cpm.CTL_SQ = frozenset(), 0.15


# --- commands ------------------------------------------------------------------------

def _cache_one(args):
    b, r = received(*args)
    return args, b, r


def cache(a):
    jobs = [(m, c, float(s), 92000 + k) for m, c, s in (x.split(":") for x in a.cells) for k in range(a.trials)]
    with Pool(WORKERS) as p:
        out = p.map(_cache_one, jobs, chunksize=2)
    pickle.dump(out, open(a.out, "wb"))
    print(len(out), "trials,", sum(r is None for _, _, r in out), "not received")


def _eval_one(args):
    key, b, r, arms = args
    return key, {arm: judge(b, r, arm) if r is not None else (0, 0, 0) for arm in arms}


def table(res, arms, col):
    cells = list(dict.fromkeys(k[:3] for k, _ in res))
    print(f"{'':34s}" + " ".join(f"{a:>8s}" for a in arms))
    for c in cells:
        rs = [o for k, o in res if k[:3] == c]
        print(f"{c[0]:15s}{c[1]:9s}{c[2]:6.1f} n{len(rs):<3d}" + " ".join(f"{np.mean([o[a][col] for o in rs]):8.2f}" for a in arms))


def evaluate(a):
    data = pickle.load(open(a.cache, "rb"))
    with Pool(WORKERS) as p:
        res = p.map(_eval_one, [(k, b, r, a.arms) for k, b, r in data], chunksize=2)
    for name, col in (("control decoded", 0), ("data decoded, square law alone", 1), ("data decoded", 2)):
        print("==", name)
        table(res, a.arms, col)


def _run_one(args):
    name, chan, snr, seed, arms = args
    b, r = received(name, chan, snr, seed)
    return name, chan, snr, seed, int(r is not None), {arm: judge(b, r, arm) if r is not None else (0, 0, 0) for arm in arms}


def run(a):
    jobs = [(m, c, float(s), 70000 + k, a.arms) for m, c, ss in (x.split(":") for x in a.cells or grid_cells())
            for s in ss.split(",") for k in range(a.trials)]
    with Pool(WORKERS) as p, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "chan", "snr", "seed", "lock", "arm", "ctl", "first", "data"])
        for i, (m, c, s, seed, lock, out) in enumerate(p.imap_unordered(_run_one, jobs, chunksize=4)):
            for arm, v in out.items():
                w.writerow([m, c, s, seed, lock, arm, *v])
            if i % 500 == 0:
                f.flush()
                print(f"{i}/{len(jobs)}", file=sys.stderr, flush=True)
    summarize(a)


# fsk32r62's data 10% points per channel (r1/3, r1/2, r2/3), PEP cells (c32_rate_study, 2026-11-23), and the
# other grids' offsets from them (cpm_floor_study's e2e points): the fixed-burst grid's windows
C32 = {"awgn": (-16.2, -15.0, -14.2), "mpg": (-10.5, -8.1, -6.0), "mpp": (-12.6, -10.3, -7.3),
       "mpd": (-12.5, -10.2, -6.8), "hilat-m": (-12.3, -10.1, -7.1), "lowlat-d": (-11.6, -8.5, -4.9),
       "hilat-d": (-10.3, -6.7, -2.0), "nvis-d": (-11.2, -8.6, -3.5)}
GRID_DB = {"fsk32r62": 0.0, "fsk8r50": -2.5, "fsk16r25": -4.5}
HIGH = (10.0, 22.0)


def grid_cells():
    out = []
    for m in [m for m in MODES if m.startswith("fsk")]:
        g, rate = m.split("-")
        for c in CHANS:
            mid = round(C32[c][("r1/3", "r1/2", "r2/3").index(rate)] + GRID_DB[g])
            out.append(f"{m}:{c}:" + ",".join(str(float(v)) for v in list(range(mid - 4, mid + 4)) + list(HIGH)))
        if rate != "r1/3":  # an in-grid carrier: the floor (it sits higher) and up
            for c in ("awgn", "mpp"):
                mid = round(C32[c][("r1/3", "r1/2", "r2/3").index(rate)] + GRID_DB[g])
                out.append(f"{m}:{c}+c:" + ",".join(str(float(v)) for v in list(range(mid - 2, mid + 6)) + [0.0] + list(HIGH)))
    return out


def crossing(xs, fail, target=0.1):
    """The last crossing down through `target`."""
    out = float("nan")
    for i in range(len(xs) - 1):
        if fail[i] > target >= fail[i + 1]:
            out = xs[i] + (fail[i] - target) / (fail[i] - fail[i + 1]) * (xs[i + 1] - xs[i])
    return out


def summarize(a):
    rows = list(csv.DictReader(open(a.out)))
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    by = {}
    for r in rows:
        by.setdefault((r["mode"], r["chan"], r["arm"], float(r["snr"])), []).append((int(r["ctl"]), int(r["data"]), int(r["lock"])))
    cells = list(dict.fromkeys((r["mode"], r["chan"]) for r in rows))
    for title, col in (("data codeword", 1), ("control codeword", 0)):
        print(f"== {title}: 10% point (PEP cell dB), the first arm's and each other arm's less it; then failures at "
              f"{HIGH[0]:g} | {HIGH[1]:g} dB")
        print(f"{'':24s}" + " ".join(f"{x:>7s}" for x in arms) + "   " + " ".join(f"{x:>9s}" for x in arms))
        for m, c in cells:
            snrs = sorted(s for (mm, cc, aa, s) in by if (mm, cc, aa) == (m, c, arms[0]))
            pts, hi = [], []
            for arm in arms:
                fail = [1 - np.mean([v[col] for v in by[m, c, arm, s]]) for s in snrs]
                pts.append(crossing(snrs, fail))
                hi.append("|".join(f"{1 - np.mean([v[col] for v in by[m, c, arm, h]]):.2f}" for h in HIGH))
            print(f"{m:15s}{c:9s}" + f"{pts[0]:7.1f} " + " ".join(f"{p - pts[0]:+7.2f}" for p in pts[1:]) + "   "
                  + " ".join(f"{h:>9s}" for h in hi))
    lock = {}
    for (m, c, arm, s), v in by.items():
        if arm == arms[0]:
            lock.setdefault(m.split("-")[0], []).extend(x[2] for x in v)
    print("bursts received with the right header (every arm: the locks are the shipped receiver's):",
          ", ".join(f"{g} {np.mean(v):.3f}" for g, v in lock.items()))


def _profile_one(args):
    name, chan, snr, seed = args
    b, r = received(name, chan, snr, seed)
    if r is None:
        return name, chan, None
    E, t = r["E"], sent_tones(b)
    m, rows = E.shape[1], np.arange(len(t))
    acc, cnt = np.zeros(7), np.zeros(7)
    for d in range(-3, 4):
        k = t + d
        ok = (k >= 0) & (k < m)
        acc[d + 3], cnt[d + 3] = E[rows[ok], k[ok]].sum(), ok.sum()
    prev, nxt = np.roll(t, 1), np.roll(t, -1)
    far = np.abs(np.arange(m)[None] - t[:, None]) >= 4
    far[rows, prev] = far[rows, nxt] = False
    okp, okn = np.abs(prev - t) >= 4, np.abs(nxt - t) >= 4
    okp[0] = okn[-1] = False
    return name, chan, (acc, cnt, E[far].sum(), far.sum(), E[rows[okp], prev[okp]].sum(), okp.sum(),
                        E[rows[okn], nxt[okn]].sum(), okn.sum())


def profile(a):
    modes = ("fsk32r62-r1/2", "fsk8r50-r1/2", "fsk16r25-r1/2")
    with Pool(WORKERS) as p:
        res = p.map(_profile_one, [(m, c, a.snr, 91000 + k) for m in modes for c in CHANS for k in range(16)], chunksize=2)
    print(f"PEP cell {a.snr}: mean energy at offsets -3..+3 from the sent tone, and at the previous / next symbol's "
          "tone (4+ bins off), as shares of the sent tone's; the other bins' share")
    for m in modes:
        for c in CHANS:
            rs = [r[2] for r in res if r[0] == m and r[1] == c and r[2] is not None]
            if not rs:
                continue
            acc = sum(r[0] for r in rs) / sum(r[1] for r in rs)
            e0 = acc[3]
            f = lambda i: sum(r[i] for r in rs) / sum(r[i + 1] for r in rs) / e0  # noqa: E731
            print(f"{m:14s}{c:9s} n{len(rs):2d} " + " ".join(f"{v / e0:6.3f}" for v in acc)
                  + f" | prev {f(4):6.3f} next {f(6):6.3f} | rest {f(2):7.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profile", "cache", "eval", "run", "summarize"])
    ap.add_argument("cache", nargs="?")
    ap.add_argument("--snr", type=float, default=25.0)
    ap.add_argument("--cells", nargs="+", default=[])
    ap.add_argument("--arms", nargs="+", default=["w1"])
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--out", default="runs/spill_dev.pkl")
    a = ap.parse_args()
    {"profile": profile, "cache": cache, "eval": evaluate, "run": run, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
