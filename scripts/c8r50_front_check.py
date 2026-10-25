"""Check a c8r50 front (scripts/c8r50_front_search.py) with the real
detector: cpm.FRONTS's, or --front's patched into data2g.cpm in this process.

  edge     find with its range ending 0.5-7.5 symbols before a clean burst's
           start (the period-early lock): wrong locks
  offpeak  the best sync score where the true start can't be chosen (the
           window cut short of it on either side), per SNR
  detect   a correct front lock (start within T/4, header right), AWGN and
           MPP at low SNR: no sensitivity lost
  stream   scripts/cpm_supersede_study.py's base receiver: fsk8r50 misreads

    uv run python scripts/c8r50_front_check.py --front 2,1,5,... --workers 8
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import codes, cpm, hfchannel

G = cpm.GRIDS["c8r50"]
FS = 8000


def install(front):
    if front is None:
        return
    cpm.FRONTS = dict(cpm.FRONTS, c8r50=tuple(front))
    for f in (cpm.layout, cpm.header_rows, cpm.mid_rows, cpm.hdr_span):
        f.cache_clear()


def burst(spec, n, rng):
    ctl = codes.encode(cpm.CTL[G.name], rng.bytes(codes.payload_bytes(cpm.CTL[G.name])), 0, 0)
    data = [codes.encode(spec, rng.bytes(codes.payload_bytes(spec)), 0, 0) for _ in range(n)]
    return cpm.modulate(spec, [ctl] + data, dup=False)


SPECS = [s for s in cpm.SPECS.values() if s.grid == "c8r50"]


def edge(seed):
    rng = np.random.default_rng(seed)
    bad = tot = 0
    for _ in range(4):
        spec, n = SPECS[rng.integers(2)], int(rng.integers(0, 6))
        x = burst(spec, n, rng)
        y = np.concatenate([np.zeros(8000), x, np.zeros(8000)]) + rng.normal(0, 0.05, len(x) + 16000)
        for k in range(1, 16):
            lock = cpm.find(G, y, hi=8000 - k * G.T // 2)
            tot += 1
            bad += lock is not None and not (abs(lock["start"] - 8000) < G.T // 4 and lock["spec"] is spec
                                             and lock["n_data"] == n)
    return bad, tot


def offpeak(args):
    """The best front score where the burst's own start can't be chosen: the
    window ending k/2 symbols short of its front's end, or starting k/2
    symbols past its start (k = 2..12: half a symbol off is the peak's own
    skirt, not a sidelobe), worst over k."""
    seed, snr = args
    rng = np.random.default_rng(seed)
    spec, n = SPECS[rng.integers(2)], int(rng.integers(0, 6))
    x = burst(spec, n, rng)
    s0 = 8000
    y = np.concatenate([np.zeros(s0), x, np.zeros(4000)])
    y = hfchannel.awgn(hfchannel.freq_shift(y, float(rng.uniform(-40, 40))), snr, seed=seed,
                       s_power=hfchannel.active_power(x))
    span = len(cpm.preamble_pattern(G)) * G.T
    out = []
    for k in range(2, 13):
        out.append(cpm.detect(G, y[s0 - FS:s0 + span - k * G.T // 2], front_only=True)[0])
        out.append(cpm.detect(G, y[s0 + k * G.T // 2:s0 + k * G.T // 2 + 2 * FS], front_only=True)[0])
    return max(out)


def detect(args):
    seed, snr, chan = args
    rng = np.random.default_rng(seed)
    spec, n = SPECS[rng.integers(2)], int(rng.integers(0, 6))
    x = burst(spec, n, rng)
    y = np.concatenate([np.zeros(8000), x, np.zeros(4000)])
    y = hfchannel.apply_channel(y, snr, float(rng.uniform(-40, 40)), fading_preset=None if chan == "awgn" else chan,
                                seed=seed)
    lock = cpm.find(G, y[:8000 + cpm.hdr_span("c8r50", 2) + FS], wait=True)
    if lock is not None and lock.get("wait"):
        lock = cpm.late_lock(G, y, lock)
    return lock is not None and abs(lock["start"] - 8000) < G.T // 4 and lock["spec"] is spec and lock["n_data"] == n


def stream(seed):
    sys.path.insert(0, str(Path(__file__).parent))
    import cpm_supersede_study as S

    S.VARIANTS = {"base": {}}
    heads, _ = S.job((seed, 10.0))
    _, bursts = S.stream(seed, 10.0)
    bad = tot = 0
    for h in heads:
        if not h["name"].startswith("fsk8r50"):
            continue
        b = [b for b in bursts if b["family"] == "cpm" and abs(h["t"] - b["at"] / FS) < 0.5]
        if b:
            tot += 1
            span = (b[0]["end"] - b[0]["at"]) / FS
            bad += not (h["name"] == b[0]["name"] and abs(h["t"] - b[0]["at"] / FS) < 0.03 and abs(h["span_s"] - span) < 0.1)
    return bad, tot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=lambda s: [int(v) for v in s.split(",")], default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stream-jobs", type=int, default=12)
    a = ap.parse_args()
    install(a.front)
    print("front", list(map(int, cpm.preamble_pattern(G))))
    with Pool(a.workers) as p:
        e = p.map(edge, range(20))
        print("edge: wrong locks %d of %d" % (sum(b for b, _ in e), sum(t for _, t in e)))
        for snr in (0.0, 10.0, 20.0):
            s = np.array(p.map(offpeak, [(k, snr) for k in range(40)]))
            print(f"offpeak {snr:4.0f} dB: max {s.max():.3f}, p90 {np.quantile(s, 0.9):.3f}, median {np.median(s):.3f}"
                  f" (sync threshold {cpm.SYNC_THRESHOLD['c8r50']})")
        for chan in ("awgn", "mpp"):
            row = []
            for snr in (-8.0, -6.0, -4.0, -2.0):
                ok = p.map(detect, [(k, snr, chan) for k in range(200)])
                row.append(f"{snr:.0f} dB {np.mean(ok):.3f}")
            print(f"detect {chan}: " + ", ".join(row))
        st = p.map(stream, range(9000, 9000 + a.stream_jobs))
        print("stream: fsk8r50 misreads %d of %d" % (sum(b for b, _ in st), sum(t for _, t in st)))
