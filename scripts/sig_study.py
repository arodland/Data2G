"""Keyed reply signatures (floor idea 2): a c16r25 control-only reply
replaced by one of M keyed tone sequences, detected in the expected slot.
Study only: no protocol change. docs: memory reply-signatures.

    python -m scripts.sig_study detect --out runs/sig/detect.npz [--trials 200]
    python -m scripts.sig_study noise --out runs/sig/noise_1.npz --n 20000 --seed 1
    python -m scripts.sig_study incumbent --out runs/sig/inc.csv [--trials 200]
    python -m scripts.sig_study report

The reply's start is the expected slot's plus U(-0.1, 0.1) s; the receiver
searches +-SLOT_S in T/4 steps at its CFO estimate (the true one plus
U(-1, 1) Hz) and +-R/16. Per (start, CFO) and hypothesis, two statistics:
  sq   sum of cpm.tone_llr on the sequence's tones (the header's metric)
  cohB sum over blocks of B symbols of |sum of the sequence's bins|^2 / (B N0)
Kept per trial and statistic: the sent hypothesis' best over the search, and
the best of the others (noise: the best and runner-up of all). A threshold
and margin are applied afterwards (report), with the threshold set from the
noise tail at a false-accept rate per slot.

The incumbent is today's reply: a c16r25 control-only burst at the same
start, locked by cpm.prime_lock in the same window (else the blind lock),
decoded blind, as an ARQ_DUP pair (its own burst), and primed with every
payload bit but 6 known (an optimistic stand-in for Station.expect).
SNR: mean power in 2500 Hz, as cpm_floor_study.
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data2g import cpm  # noqa: E402
from data2g.config import FS  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

G = cpm.GRIDS["c16r25"]
M = 128
LS = (24, 32, 48, 64)
BLOCKS = (4, 8)
STATS = ("sq",) + tuple(f"coh{b}" for b in BLOCKS)
SLOT_S = 0.15  # the search's half-width (ReplySlot.TOL_S)
JITTER_S = 0.10  # the reply's start about the slot's
CFO_ERR = 1.0  # Hz, the receiver's CFO estimate's error (uniform +-)
CHANS = {"awgn": None, "mpg": (0.1, 0.5), "mpp": (1.0, 2.0), "mpd": (2.0, 4.0)}
SNRS = {"awgn": np.arange(-26, -13, 1.0), "mpg": np.arange(-22, -5, 2.0), "mpp": np.arange(-24, -11, 1.0),
        "mpd": np.arange(-24, -11, 1.0)}


def family(key: int, L: int) -> np.ndarray:
    """(M, L) the keyed family's tone sequences."""
    return np.random.default_rng([0x5161, key, L]).integers(0, G.m, (M, L))


def waveform(seq: np.ndarray, cap: int = 2) -> np.ndarray:
    """A sequence -> audio as a CPM burst's (tones, glide, the cap's TX filter)."""
    return cpm.bandpass(G, cpm.tones(G, seq, cpm.tx_filter(G.name, cap).glide), cap)


def search(x: np.ndarray, fam: np.ndarray, lo: int, hi: int, cfo: float) -> dict:
    """Starts in [lo, hi] (T/4 steps), CFO +-R/16: per statistic, (M,) each
    hypothesis' best score."""
    L = fam.shape[1]
    phases = range(0, G.T, G.T // 4)
    n_pos = (hi - lo) // G.T + 1
    best = {s: np.full(M, -np.inf) for s in STATS}
    idx = np.arange(L)
    for p in phases:
        for df in (-1 / 16, 0.0, 1 / 16):
            Z = cpm.bins(G, x, lo + p, n_pos + L, cfo + df * G.rate)
            E = np.abs(Z) ** 2
            T_ = cpm.tone_llr(G, E)
            n0 = cpm._n0(E)
            for m in range(n_pos):
                if lo + p + m * G.T > hi:
                    continue
                rows = m + idx
                sq = T_[rows[None, :], fam].sum(axis=1)
                best["sq"] = np.maximum(best["sq"], sq)
                z = Z[rows[None, :], fam]  # (M, L)
                for b in BLOCKS:
                    nb = L // b
                    s = np.abs(z[:, :nb * b].reshape(M, nb, b).sum(axis=2)) ** 2 / (b * n0)
                    best[f"coh{b}"] = np.maximum(best[f"coh{b}"], s.sum(axis=1))
    return best


def detect_trial(args):
    import phy_session as PS

    chan, snr, seed = args
    rng = np.random.default_rng(seed)
    key = int(rng.integers(1 << 30))
    out = []
    for L in LS:
        fam = family(key, L)
        sent = int(rng.integers(M))
        x = waveform(fam[sent])
        ch = PS.ContinuousChannel("awgn", snr, seed, 120.0, *(CHANS[chan] or (None, None)))
        jit = int(rng.uniform(-JITTER_S, JITTER_S) * FS)
        x = np.concatenate([np.zeros(max(jit, 0) + int(SLOT_S * FS)), x])
        y = ch.apply(x, float(rng.uniform(5, 100)))
        nominal = int(PS.PAD_S * FS) + int(SLOT_S * FS) + max(jit, 0) - jit  # the slot's start; the reply's is + jit
        lo, hi = nominal - int(SLOT_S * FS), nominal + int(SLOT_S * FS)
        b = search(y, fam, lo, hi, PS.CFO_HZ + rng.uniform(-CFO_ERR, CFO_ERR))
        for s in STATS:
            other = np.delete(b[s], sent).max()
            out.append((chan, snr, seed, L, s, float(b[s][sent]), float(other)))
    return out


def noise_trial(args):
    seed, = args
    rng = np.random.default_rng([seed, 77])
    key = int(rng.integers(1 << 30))
    out = []
    for L in LS:
        fam = family(key, L)
        n = int((2 * SLOT_S + L / G.rate + 1) * FS)
        y = rng.normal(size=n)
        b = search(y, fam, int(0.5 * FS), int((0.5 + 2 * SLOT_S) * FS), 0.0)
        for s in STATS:
            v = np.sort(b[s])
            out.append((seed, L, s, float(v[-1]), float(v[-2])))
    return out


def incumbent_trial(args):
    import phy_session as PS
    from cpm_floor_study import receive_traced
    from data2g.arq import phy as PHY
    from data2g.arq.burst import Slot

    import outcome_data as O

    chan, snr, seed = args
    rng = np.random.default_rng(seed)
    mode = "fsk16r25-r1/2"
    b1 = replace(O.burst(mode, 1, rng), cap=2)
    s0 = b1.slots[0]
    b2 = replace(b1, slots=[s0, Slot(s0.mask_id, 1, s0.payload)])  # ARQ_DUP
    tmask = bytes([0xFF] * 3 + [0x03] + [0xFF] * (len(s0.payload) - 4))  # 6 bits unknown
    out = {}
    for name, b in (("single", b1), ("dup", b2)):
        x = PHY.tx_audio(b)
        ch = PS.ContinuousChannel("awgn", snr, seed, 120.0, *(CHANS[chan] or (None, None)))
        jit = int(rng.uniform(-JITTER_S, JITTER_S) * FS)
        x = np.concatenate([np.zeros(max(jit, 0) + int(SLOT_S * FS)), x])
        y = ch.apply(x, float(rng.uniform(5, 100)))
        nominal = int(PS.PAD_S * FS) + int(SLOT_S * FS) + max(jit, 0) - jit
        lock = cpm.prime_lock(G, y, nominal - int(SLOT_S * FS), nominal + int(SLOT_S * FS),
                              PS.CFO_HZ + rng.uniform(-CFO_ERR, CFO_ERR))
        r = cpm.receive(y, lock) if lock is not None else receive_traced(y, mode)
        ok_lock = r is not None and r["n_cw"] == len(b.slots)
        blind = primed = pair = 0
        if ok_lock:
            r = dict(r)
            r.pop("_soft", None)
            rx = PHY.ModemRx(r, {})
            blind = int(rx.decode(0, s0.mask_id, 0, None) == s0.payload)
            primed = int(blind or rx.decode_known(0, s0.mask_id, s0.payload, tmask) == s0.payload)
            if name == "dup":
                pair = int(blind or rx.decode_pair(0, s0.mask_id) == s0.payload)
                primed = int(pair or rx.decode_known(0, s0.mask_id, s0.payload, tmask, pair=True) == s0.payload)
        out[name] = (blind, pair, primed, cpm.burst_seconds(cpm.SPECS[mode], len(b.slots), name == "dup"))
    return chan, snr, seed, out


def run(fn, jobs, n):
    with Pool(n) as p:
        yield from p.imap_unordered(fn, jobs, chunksize=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["detect", "noise", "incumbent", "report", "probe"])
    ap.add_argument("--out")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--dir", default="runs/sig")
    ap.add_argument("--fa", type=float, default=1e-6, help="report: false accepts per slot")
    a = ap.parse_args()
    if a.cmd == "probe":
        import time
        t = time.time()
        print(detect_trial(("mpp", -18.0, 3))[:3], round(time.time() - t, 2), "s detect")
        t = time.time()
        print(noise_trial((1,))[:3], round(time.time() - t, 2), "s noise")
        t = time.time()
        print(incumbent_trial(("mpp", -12.0, 3)), round(time.time() - t, 2), "s incumbent")
        return
    if a.cmd == "detect":
        jobs = [(c, float(s), 90000 + i) for c in CHANS for s in SNRS[c] for i in range(a.trials)]
        rows = [r for out in run(detect_trial, jobs, a.jobs) for r in out]
        np.save(a.out, np.array(rows, dtype=object), allow_pickle=True)
    elif a.cmd == "noise":
        jobs = [(a.seed * 10_000_000 + i,) for i in range(a.n)]
        rows = [r for out in run(noise_trial, jobs, a.jobs) for r in out]
        np.save(a.out, np.array(rows, dtype=object), allow_pickle=True)
    elif a.cmd == "incumbent":
        jobs = [(c, float(s), 90000 + i) for c in CHANS for s in SNRS[c] if s >= -20 for i in range(a.trials)]
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["channel", "snr", "seed", "kind", "blind", "pair", "primed", "seconds"])
            for c, s, seed, out in run(incumbent_trial, jobs, a.jobs):
                for k, v in out.items():
                    w.writerow([c, s, seed, k, *v])
    else:
        report(a)


def tail_threshold(v: np.ndarray, fa: float) -> float:
    """The score a noise slot's best exceeds with probability fa: the empirical
    quantile if the sample reaches it, else a Gumbel fit to the top 10%."""
    v = np.sort(v)
    if fa * len(v) >= 10:
        return float(np.quantile(v, 1 - fa))
    top = v[int(0.9 * len(v)):]
    # exponential tail over the 90% point: P(V > t) = 0.1 exp(-(t - u) / beta)
    u, beta = top[0], float(np.mean(top - top[0]))
    return float(u + beta * np.log(0.1 / fa))


def x10(snrs, ok, target=0.1):
    f = 1 - np.asarray(ok)
    for i in range(len(snrs) - 1):
        if f[i] > target >= f[i + 1]:
            return snrs[i] + (f[i] - target) / (f[i] - f[i + 1]) * (snrs[i + 1] - snrs[i])
    return np.nan


def report(a):
    nz = np.concatenate([np.load(f, allow_pickle=True) for f in glob.glob(f"{a.dir}/noise_*.npy")])
    det = np.load(f"{a.dir}/detect.npy", allow_pickle=True)
    print(f"noise slots {len(nz) // (len(LS) * len(STATS))}; thresholds at {a.fa:g} false accepts per slot")
    thr = {}
    for L in LS:
        for s in STATS:
            v = np.array([r[3] for r in nz if r[1] == L and r[2] == s], float)
            thr[(L, s)] = tail_threshold(v, a.fa)
    inc = list(csv.DictReader(open(f"{a.dir}/inc.csv")))
    print("10% points (dB, mean power in 2500 Hz): signature (correct and over the threshold, the others under it) "
          "| incumbent single blind / primed, dup pair / primed")
    for c in CHANS:
        snrs = sorted({r[1] for r in det if r[0] == c})
        cells = []
        for L in LS:
            for s in STATS:
                ok = []
                for snr in snrs:
                    rs = [r for r in det if r[0] == c and r[1] == snr and r[3] == L and r[4] == s]
                    ok.append(np.mean([r[5] >= thr[(L, s)] and r[6] < thr[(L, s)] for r in rs]))
                cells.append(f"L{L} {s} {x10(snrs, ok):6.2f}")
        isn = sorted({float(r["snr"]) for r in inc if r["channel"] == c})
        def ip(kind, col):
            return x10(isn, [np.mean([int(r[col]) for r in inc if r["channel"] == c and float(r["snr"]) == s
                                      and r["kind"] == kind]) for s in isn])
        print(f"  {c:5s} " + "  ".join(cells))
        print(f"        incumbent single {ip('single', 'blind'):6.2f} / {ip('single', 'primed'):6.2f}, "
              f"dup {ip('dup', 'pair'):6.2f} / {ip('dup', 'primed'):6.2f}")
    secs = {k: float(next(r["seconds"] for r in inc if r["kind"] == k)) for k in ("single", "dup")}
    print("airtime: " + ", ".join(f"signature L{L} {L / G.rate + 2 * cpm.RAMP_S:.2f} s" for L in LS)
          + f"; incumbent single {secs['single']:.2f} s, dup {secs['dup']:.2f} s")


if __name__ == "__main__":
    main()
