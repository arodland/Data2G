"""CPM false locks per hour (cpm-acq-gap, 2026-10-26): the streaming
Receiver listening for one CPM grid (no OFDM bands) on audio holding no
burst of that grid: Gaussian noise, OFDM bursts (random modes, 0-30 dB,
AWGN/MPG, 0.5-3 s gaps) or the other grids' CPM bursts; or two stations'
bursts of the grid itself, overlapping at random (cpm-own: a lock not at a
real burst's start and word counts). Each false "header" event holds BUSY
for the length it claims.

    python -m scripts.cpm_false_locks --source noise --hours 1 [--master]

`--master`: a reference cpm.py (git show origin/master:data2g/cpm.py >
data2g/_cpm_master.py, not committed) in place of this branch's, for the A/B.

Found (2026-10-26, 8 h a source and grid, branch / master): noise 0 / 0,
OFDM 0 / 0, other grids 0 / 2, own grid 4 / 25.
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402

from data2g.config import FS  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

SOURCES = ("noise", "ofdm", "cpm-other", "cpm-own")
CHUNK_S = 120  # audio per job
FEED = FS // 10


def job(args):
    grid, source, seed, master = args
    if master:
        import data2g
        import data2g._cpm_master as old

        sys.modules["data2g.cpm"] = old
        data2g.cpm = old
    from data2g import cpm, hfchannel, modem
    from data2g.receiver import Receiver

    rng = np.random.default_rng([seed, list(cpm.GRIDS).index(grid), SOURCES.index(source)])
    n = CHUNK_S * FS
    real = []  # (start, spec, n_data): the grid's own bursts ("cpm-own")
    if source == "noise":
        y = rng.normal(0, 1, n)
    elif source == "cpm-own":  # two stations' bursts of this grid, each its own channel, overlapping at random
        from data2g import codes

        y = rng.normal(0, 1, n)
        g = cpm.GRIDS[grid]
        for _ in range(2):
            k = int(rng.uniform(0, 3) * FS)
            while k < n:
                spec = cpm.grid_specs(grid)[int(rng.integers(2))]
                nd = int(rng.integers(0, 4))
                ctl = codes.encode(cpm.CTL[grid], rng.bytes(codes.payload_bytes(cpm.CTL[grid])), 0, 0)
                x = cpm.modulate(spec, [ctl] + [codes.encode(spec, rng.bytes(codes.payload_bytes(spec)), 0, 0)
                                                for _ in range(nd)], dup=False)
                x = hfchannel.apply_channel(x, snr_db=None, freq_offset_hz=float(rng.uniform(-50, 50)),
                                            fading_preset=[None, "mpg"][int(rng.integers(2))], seed=int(rng.integers(1 << 30)))
                x *= 10 ** (float(rng.uniform(-5, 30)) / 20) / max(np.sqrt(np.mean(x ** 2)), 1e-12)
                m = min(len(x), n - k)
                y[k:k + m] += x[:m]
                real.append((k, spec, nd))
                k += len(x) + int(rng.uniform(1, 8) * FS)
        del g
    else:
        from cpm_acq_fit import burst
        from data2g.arq.modes import MODES, is_cpm

        pool = ([s for s in MODES.values() if not is_cpm(s)] if source == "ofdm"
                else [s for s in cpm.SPECS.values() if s.grid != grid])
        parts, k = [], 0
        while k < n:
            gap = np.zeros(int(rng.uniform(0.5, 3) * FS))
            x = burst(pool[int(rng.integers(len(pool)))], rng)
            x = hfchannel.apply_channel(x, snr_db=None, freq_offset_hz=float(rng.uniform(-50, 50)),
                                        fading_preset=[None, "mpg"][int(rng.integers(2))], seed=int(rng.integers(1 << 30)))
            x *= 10 ** (float(rng.uniform(0, 30)) / 20) / max(np.sqrt(np.mean(x ** 2)), 1e-12)
            parts += [gap, x]
            k += len(gap) + len(x)
        y = np.concatenate(parts)[:n] + rng.normal(0, 1, n)
    rx = Receiver(modem.Accept((), 0.0), cpm_grids=(grid,))
    hits = 0
    t = time.process_time()
    T = cpm.GRIDS[grid].T
    for i in range(0, n, FEED):
        for e in rx.feed(y[i:i + FEED]):
            if e[0] != "header":
                continue
            h = e[1]
            # a lock on a real burst (its start within 2 symbols, its word) is no false lock
            hits += not any(abs(h["start"] - s) <= 2 * T and h["spec"] == sp and h["n_data"] == nd and not h["dup"]
                            for s, sp, nd in real)
    return grid, source, hits, time.process_time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="noise", choices=SOURCES)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--grids", nargs="+", default=["c16r25", "c8r50", "c32r62"])
    ap.add_argument("--master", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    k = int(round(a.hours * 3600 / CHUNK_S))
    jobs = [(g, a.source, i, a.master) for g in a.grids for i in range(k)]
    out = {}
    with Pool(a.jobs) as p:
        for g, s, h, cpu in p.imap_unordered(job, jobs):
            o = out.setdefault(g, [0, 0.0])
            o[0] += h
            o[1] += cpu
    for g, (h, cpu) in out.items():
        print(f"{'master' if a.master else 'branch'} {g} {a.source}: {h} false locks in {a.hours:.2f} h, "
              f"{h / a.hours:.2f}/h; CPU {100 * cpu / (a.hours * 3600):.2f}% of a core")


if __name__ == "__main__":
    main()
