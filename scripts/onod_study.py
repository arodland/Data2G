"""OFDM nods, burst level (docs/arq.md §3c; data2g/arq/onod.py).

  --noise N    each variant's best score over N white-noise slots
               (+-ReplySlot.TOL_S): the threshold at --fa per slot
  --detect     nods against the ACK they replace (blind, then primed), per
               channel and SNR (mean power in 2500 Hz)
  --false N    what is not a nod in the slot: the peer's real ACK, a data
               burst, a nod of the turn next to it; and what a stock
               receiver makes of a nod

    uv run --no-sync python scripts/onod_study.py --noise 200000 --jobs 18
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, hfchannel, modem  # noqa: E402
from data2g.arq import onod  # noqa: E402
from data2g.config import FS, LEADIN_SAMPLES, SUBMODES  # noqa: E402
from data2g.receiver import ReplySlot, receive_any  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from expect_header_study import expected_accept  # noqa: E402

TOL = int(ReplySlot.TOL_S * FS)
VARIANTS = [("w", 4), ("w", 9), ("n10", 10)]
ACKS = {"w": ("ack-4f", "ack-1f"), "n10": ("n10-ack-4f", "n4-ack-8f")}
DATA = {"w": "qpsk-r1/3", "n10": "n10-qpsk-r1/3"}


def fam_of(rng, band, k, turn=None):
    key = int(rng.integers(1 << 32))
    d, bn, ac = turn or (int(rng.integers(2)), int(rng.integers(8)), int(rng.integers(8)))
    return onod.family(band, k, key, d, bn, ac), (key, d, bn, ac)


def noise(args):
    band, k, seed, n = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        fam, _ = fam_of(rng, band, k)
        y = rng.standard_normal(FS // 4 + 2 * TOL + onod.samples(band, k) + FS // 4)
        sc = onod.scores(y, FS // 4, FS // 4 + 2 * TOL, rng.uniform(-50, 50), band, k, fam)
        out.append(float(np.max(sc[0])) if sc else 0.0)
    return out


def channel(x, chan, snr, seed, rng):
    lead = int(rng.uniform(0.3, 1.0) * FS)
    f0 = rng.uniform(-50, 50)
    x = np.concatenate([np.zeros(lead), x, np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=f0, ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    return y, lead + LEADIN_SAMPLES, f0


def slot(rng, start, slot_err):
    c = start + int(rng.uniform(-slot_err, slot_err) * FS)
    return max(0, c - TOL), c + TOL


def nod_trial(args):
    band, k, chan, snr, seed, slot_err, thr = args
    rng = np.random.default_rng(seed)
    fam, _ = fam_of(rng, band, k)
    m = int(rng.integers(onod.N_MEANINGS))
    y, start, f0 = channel(onod.waveform(band, k, fam[m]), chan, snr, seed, rng)
    lo, hi = slot(rng, start, slot_err)
    sc = onod.scores(y, lo, hi, f0 + rng.uniform(-1, 1), band, k, fam)
    if sc is None:
        return 0.0, 0.0, False
    order = np.argsort(sc[0])[::-1]
    return float(sc[0][order[0]]), float(sc[0][order[1]]), bool(order[0] == m)


def ack_trial(args):
    mode, n_cw, chan, snr, seed, slot_err = args
    spec = SUBMODES[mode]
    rng = np.random.default_rng(seed)
    x = modem.modulate([rng.bytes(codes.payload_bytes(spec)) for _ in range(n_cw)], spec)
    y, start, f0 = channel(x, chan, snr, seed, rng)

    def whole(r):
        try:
            return bool(all(modem.decode_received(r).crc_ok))
        except modem.SyncError:
            return False

    try:
        r = receive_any(y, lead=start - LEADIN_SAMPLES + FS // 4)
    except modem.SyncError:
        r = None
    if r is not None and (r["spec"].name, r["n_cw"]) == (mode, n_cw) and whole(r):
        return "blind"
    lo, hi = slot(rng, start, slot_err)
    lock = modem.prime_ofdm(y, lo, hi, f0 + rng.uniform(-1, 1), expected_accept(spec.sync_band))
    if lock is not None and (lock["spec"].name, lock["n_cw"]) == (mode, n_cw):
        try:
            if whole(modem.receive(y, known=dict(lock))):
                return "primed"
        except modem.SyncError:
            pass
    return "lost"


def false_trial(args):
    """-> (kind, accepted, best score): the listener's own family against
    something else in its slot."""
    band, k, kind, chan, snr, seed, thr = args
    rng = np.random.default_rng(seed)
    fam, (key, d, bn, ac) = fam_of(rng, band, k)
    if kind == "turn":  # the same session's nod for the turn before or after
        other = onod.family(band, k, key, d, (bn + int(rng.choice([-1, 1]))) % 8, (ac + int(rng.choice([-1, 0, 1]))) % 8)
        x = onod.waveform(band, k, other[int(rng.integers(onod.N_MEANINGS))])
    elif kind == "other":  # another session's
        x = onod.waveform(band, k, fam_of(rng, band, k)[0][int(rng.integers(onod.N_MEANINGS))])
    else:
        mode = DATA[band] if kind == "data" else ACKS[band][int(rng.integers(2))]
        spec = SUBMODES[mode]
        n_cw = 2 if kind == "data" else int(rng.integers(1, 4))
        x = modem.modulate([rng.bytes(codes.payload_bytes(spec)) for _ in range(n_cw)], spec)
    y, start, f0 = channel(x, chan, snr, seed, rng)
    lo, hi = slot(rng, start, 0.1)
    sc = onod.scores(y, lo, hi, f0 + rng.uniform(-1, 1), band, k, fam)
    best = float(np.max(sc[0])) if sc else 0.0
    return kind, onod.detect(y, lo, hi, f0, band, k, fam, thr) is not None, best


def stock_trial(args):
    """What a stock blind receiver makes of a nod: its lock, or None."""
    band, k, chan, snr, seed = args
    rng = np.random.default_rng(seed)
    fam, _ = fam_of(rng, band, k)
    y, start, f0 = channel(onod.waveform(band, k, fam[int(rng.integers(onod.N_MEANINGS))]), chan, snr, seed, rng)
    out = []
    try:
        r = receive_any(y, lead=start - LEADIN_SAMPLES + FS // 4)
        out.append(None if r is None else (r["spec"].name, r["n_cw"]))
    except modem.SyncError:
        out.append(None)
    try:  # a streaming receiver's first look: the head alone
        lock = modem.find_burst(y[:start + modem.receive_head(band) + FS // 4])
        out.append((lock["spec"].name, lock["n_cw"]))
    except modem.SyncError:
        out.append(None)
    return out


def tail(s: np.ndarray, fa: float) -> tuple[float, float]:
    """-> (the exponential tail's, the Gaussian tail's) score at `fa` per
    slot, each fitted over the top 1% of noise slots."""
    s = np.sort(s)
    n = len(s)
    q = s[int(n * 0.99)]
    top = s[s >= q]
    exp = q + float(np.mean(top - q)) * np.log(0.01 / fa)
    # log survival linear in score^2 (a maximum of Gaussians)
    surv = (n - np.searchsorted(s, top, side="left")) / n
    keep = surv > 5 / n
    a, b = np.polyfit(top[keep] ** 2, np.log(surv[keep]), 1)
    return exp, float(np.sqrt((np.log(fa) - b) / a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=int, default=0)
    ap.add_argument("--fa", type=float, default=1e-6)
    ap.add_argument("--detect", action="store_true")
    ap.add_argument("--false", type=int, default=0, dest="false_n")
    ap.add_argument("--at", nargs="+", default=["awgn:-12", "awgn:-10", "awgn:-8", "awgn:-6", "awgn:-4", "awgn:-2",
                                                "mpp:-8", "mpp:-6", "mpp:-4", "mpp:-2", "mpp:0", "mpp:3",
                                                "mpd:-8", "mpd:-6", "mpd:-4", "mpd:-2", "mpd:0", "mpd:3"])
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--acks", nargs="*", default=["ack-4f:1", "ack-4f:2", "ack-1f:2", "n10-ack-4f:1", "n10-ack-4f:2",
                                                  "n4-ack-8f:1"], help="mode:codewords, the ACKs --detect compares")
    ap.add_argument("--no-nods", action="store_true", help="--detect: the ACKs only")
    ap.add_argument("--slot-err", type=float, default=0.1)
    ap.add_argument("--thr", nargs="*", default=[], help="band:k:thr, in place of onod.THR")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", 2)))
    a = ap.parse_args()
    thr = dict(onod.THR)
    for t in a.thr:
        b, k, v = t.split(":")
        thr[(b, int(k))] = float(v)
    with Pool(a.jobs) as pool:
        if a.noise:
            for band, k in VARIANTS:
                parts = 8 * a.jobs
                s = np.array([v for p in pool.imap_unordered(noise, [(band, k, 90_000 + i, a.noise // parts)
                                                                     for i in range(parts)]) for v in p])
                np.save(Path("runs") / f"onod_noise_{band}_{k}.npy", s)
                e, g = tail(s, a.fa)
                qs = " ".join(f"{q:g}: {np.quantile(s, 1 - q):.3f}" for q in (1e-1, 1e-2, 1e-3, 1e-4))
                print(f"{band} K={k} ({onod.seconds(band, k):.3f} s): {len(s)} noise slots; {qs}; max {s.max():.3f}; "
                      f"at {a.fa:g}: exponential tail {e:.3f}, Gaussian tail {g:.3f}; over THR {thr[(band, k)]}: "
                      f"{int(np.sum(s >= thr[(band, k)]))}", flush=True)
        if a.detect:
            cells = [(c, float(s)) for c, s in (x.split(":") for x in a.at)]
            for band, k in [] if a.no_nods else VARIANTS:
                for chan, snr in cells:
                    raw = np.array(pool.map(nod_trial, [(band, k, chan, snr, 40_000 + i, a.slot_err, thr[(band, k)])
                                                        for i in range(a.trials)], chunksize=8))
                    np.save(Path("runs") / f"onod_det_{band}_{k}_{chan}_{snr:g}.npy", raw)
                    t = thr[(band, k)]
                    over = (raw[:, 0] >= t) & (raw[:, 1] < t)
                    print(f"nod {band} K={k:<2d} {chan:4s} {snr:+5.1f} | ok {np.mean(over & (raw[:, 2] > 0)):.3f}"
                          f" wrong {int(np.sum(over & (raw[:, 2] == 0)))} | median score {np.median(raw[:, 0]):.3f}",
                          flush=True)
            for mode, n_cw in [(m, int(n)) for m, n in (x.split(":") for x in a.acks)]:
                for chan, snr in cells:
                    out = Counter(pool.map(ack_trial, [(mode, n_cw, chan, snr, 40_000 + i, a.slot_err)
                                                       for i in range(a.trials)], chunksize=8))
                    print(f"ack {mode:10s} x{n_cw} {chan:4s} {snr:+5.1f} | blind {out['blind'] / a.trials:.3f}"
                          f" +primed {(out['blind'] + out['primed']) / a.trials:.3f}", flush=True)
        if a.false_n:
            for band, k in VARIANTS:
                for kind in ("ack", "data", "turn", "other"):
                    jobs = [(band, k, kind, chan, snr, 60_000 + i, thr[(band, k)])
                            for i, (chan, snr) in zip(range(a.false_n), [("awgn", 20.0), ("awgn", 0.0), ("awgn", -6.0),
                                                                         ("mpp", 10.0), ("mpp", 0.0), ("mpd", 0.0)]
                                                      * (a.false_n // 6 + 1))]
                    out = pool.map(false_trial, jobs, chunksize=16)
                    best = np.array([o[2] for o in out])
                    print(f"false {band} K={k:<2d} {kind:5s}: {sum(o[1] for o in out)} accepted of {len(out)}; best score "
                          f"median {np.median(best):.3f}, 99.9% {np.quantile(best, 0.999):.3f}, max {best.max():.3f}",
                          flush=True)
                n = max(600, a.false_n // 10)
                for chan, snr in (("awgn", 20.0), ("awgn", 0.0), ("mpp", 0.0)):
                    out = pool.map(stock_trial, [(band, k, chan, snr, 80_000 + i) for i in range(n)], chunksize=8)
                    for j, what in enumerate(("receive_any", "find_burst")):
                        locks = Counter(o[j] for o in out if o[j] is not None)
                        secs = [modem.burst_seconds(SUBMODES[m], c) for (m, c), v in locks.items() for _ in range(v)]
                        print(f"stock {band} K={k:<2d} {chan} {snr:+.0f} {what}: {sum(locks.values())} locks of {n}"
                              + (f"; claimed length median {np.median(secs):.1f} s, max {max(secs):.1f} s" if secs else ""),
                              flush=True)


if __name__ == "__main__":
    main()
