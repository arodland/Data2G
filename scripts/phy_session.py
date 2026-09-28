"""Gear-shifter phase G: whole ARQ sessions through the real modem.

scripts/linksim.py's event loop and sessions, with SimPhy swapped for
RealPhy: every burst is modulated (data2g.arq.phy.tx_audio), passed
through one continuous Watterson process for the session (hfchannel's
taps, generated once at a low rate and read at each burst's time) plus
noise, then received by the real receiver; slots decode with masked CRCs
and soft-bit combining (phy.ModemRx). Delivered bytes are checked against
what was written (linksim.run asserts it), so this is also an end-to-end
correctness test of the ARQ layer on the modem.

    uv run python scripts/phy_session.py --out runs/phy_session.csv
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import hfchannel, modem
from data2g.arq import phy as PHY
from data2g.config import BANDS, FS, LEADIN_SAMPLES, SUBMODES
from data2g.tnc import receive_any

sys.path.insert(0, str(Path(__file__).parent))
import linksim as L  # noqa: E402

PAD_S = 0.3  # noise the receiver sees around a burst
# DATA2G_PEP_REF_DB (studies): SNR against each burst's envelope peak, as the
# average-power SNR of a burst whose peak-to-average is this many dB (PEP-fair
# between modes of different clip headroom); unset: against its own average
PEP_REF_DB = float(os.environ["DATA2G_PEP_REF_DB"]) if os.environ.get("DATA2G_PEP_REF_DB") else None


def header_time(r: dict, t0: float) -> float:
    """When the receiver knew a burst's mode and length (its header read)."""
    if r.get("family") == "cpm":
        return t0 + (r["header_end"] - int(PAD_S * FS)) / FS
    sb = BANDS[r["spec"].sync_band]
    return t0 + (LEADIN_SAMPLES + sb.preamble_samples + modem.header_samples(r["spec"].sync_band)) / FS
CFO_HZ = 4.5  # the two rigs' frequency offset, both directions


class ContinuousChannel:
    """Two-path Watterson fading for a whole session, read at any time."""

    def __init__(self, chan: str, snr_db: float, seed: int, horizon: float, doppler=None, delay_ms=None):
        """`chan`: a preset name, unless `doppler` (Hz) and `delay_ms` are given."""
        self.snr_db, self.rng = snr_db, np.random.default_rng(seed)
        dop, dly = (doppler, delay_ms) if doppler is not None else L.PRESETS[chan]
        self.doppler, self.delay = dop, int(round(dly * 1e-3 * FS))
        if dop:
            self.rate = max(64 * dop, 8.0)
            n_low = int(np.ceil((horizon + 60) * self.rate)) + 2
            f = np.fft.fftfreq(n_low, 1 / self.rate)
            shape = np.exp(-(f**2) / (4 * (dop / 2) ** 2))
            norm = np.sqrt(2 * np.mean(shape**2))
            self.g = [np.fft.ifft(np.fft.fft(self.rng.normal(size=n_low) + 1j * self.rng.normal(size=n_low)) * shape) / norm
                      for _ in range(2)]

    def apply(self, x: np.ndarray, t0: float) -> np.ndarray:
        """Audio sent at t0 -> what the receiver hears, PAD_S of noise either side."""
        s_power = hfchannel.active_power(x)
        z = hfchannel._analytic(x)
        if PEP_REF_DB is not None:
            # a peak-limited transmitter (data2g-host sends every burst at a
            # full-scale peak): noise against the peak, so a burst whose
            # peak-to-average is REF has the cell's SNR, a lower one more
            s_power = np.max(np.abs(z) ** 2) / 2 / 10 ** (PEP_REF_DB / 10)
        if self.doppler:
            tl = (t0 + np.arange(len(x)) / FS) * self.rate
            g1, g2 = (np.interp(tl, np.arange(len(g)), g.real) + 1j * np.interp(tl, np.arange(len(g)), g.imag)
                      for g in self.g)
            z2 = np.concatenate([np.zeros(self.delay, dtype=complex), z[: len(z) - self.delay]])
            z = (z * g1 + z2 * g2) / np.sqrt(2)
        z = z * np.exp(2j * np.pi * CFO_HZ * (t0 + np.arange(len(x)) / FS))
        pad = np.zeros(int(PAD_S * FS))
        y = np.concatenate([pad, np.real(z), pad])
        return hfchannel.awgn(y, self.snr_db, seed=int(self.rng.integers(1 << 31)), s_power=s_power)


class RealPhy:
    def __init__(self, ch: ContinuousChannel):
        self.ch, self.decode_s = ch, []

    def hear(self, x, t0):
        """The receiver's result for audio x sent at t0 (either family), or None."""
        try:
            return receive_any(self.ch.apply(x, t0), lead=int(PAD_S * FS) + FS // 2)
        except modem.SyncError:  # ponytail: an OFDM header read past the audio
            return None

    def send(self, burst, t0):
        """linksim.SimPhy.send's contract, on the real modem."""
        x = PHY.tx_audio(burst)
        end = t0 + len(x) / FS
        t = time.perf_counter()
        r = self.hear(x, t0)
        if r is None:
            return end, None, None, None
        spec = r["spec"]
        t_hdr = header_time(r, t0)
        soft_r = r

        def make_rx(store, stats, rng):
            t1 = time.perf_counter()
            rx = PHY.ModemRx(soft_r, store)
            self.decode_s.append(time.perf_counter() - t1)
            return rx

        self.decode_s.append(time.perf_counter() - t)
        return end, (t_hdr, spec.name, r["n_cw"]), make_rx, PHY.measure(r)


def one(args):
    workload, policy, chan, snr, seed = args
    horizon = 300.0 if workload == "bulk" else 1800.0
    ch = ContinuousChannel(chan, snr, seed, horizon)
    phy = RealPhy(ch)
    steps = L.WORKLOADS[workload](random.Random(seed + 7))
    t0 = time.perf_counter()
    try:
        res = L.run(L.make_policy(policy), L.make_policy(policy), None, steps, seed=seed, horizon=horizon, phy=phy)
    except AssertionError as e:
        return dict(workload=workload, policy=policy, channel=chan, snr=snr, seed=seed, error=f"CORRUPT {e}"[:200])
    if workload == "bulk":
        score = res["delivered"] / horizon
    elif workload == "winlink":
        score = res["t"] if res["complete"] else horizon
    else:
        score = float(np.mean(res["latency"] + [horizon] * (len(steps) - len(res["latency"]))))
    sim = L.score(workload, policy, chan, snr, seed)
    return dict(workload=workload, policy=policy, channel=chan, snr=snr, seed=seed, score=round(score, 1),
                sim_score=round(sim, 1), bursts=res["bursts"], lost_sync=res["lost_sync"], timeouts=res["timeouts"],
                reason="|".join(res["reason"]), cpu_s=round(time.perf_counter() - t0),
                rx_cpu_max_s=round(max(phy.decode_s, default=0), 2), error="")


SCENARIOS = [("bulk", "shift", c, s) for c, s in (("awgn", 0), ("awgn", 12), ("mpg", 4), ("mpp", 8), ("mpd", 4), ("mpd", 16))] + [
    ("winlink", "shift", "awgn", 8), ("winlink", "shift", "mpp", 12),
    ("chat", "chat", "mpp", 8), ("bulk", "fixed:w48-qpsk-r1/3:20", "mpg", 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/phy_session.csv")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    jobs = [(w, p, c, s, seed) for w, p, c, s in SCENARIOS for seed in range(a.seeds)]
    rows = []
    with Pool(a.jobs) as pool, open(a.out, "w", newline="") as f:
        w = None
        for r in pool.imap_unordered(one, jobs):
            rows.append(r)
            if w is None:
                w = csv.DictWriter(f, ["workload", "policy", "channel", "snr", "seed", "score", "sim_score", "bursts",
                                       "lost_sync", "timeouts", "reason", "cpu_s", "rx_cpu_max_s", "error"])
                w.writeheader()
            w.writerow(r)
            f.flush()
            print(r, flush=True)
    summarize(a.out)


def summarize(path):
    from collections import defaultdict
    g = defaultdict(list)
    for r in csv.DictReader(open(path)):
        g[(r["workload"], r["policy"], r["channel"], r["snr"])].append(r)
    print(f"\n{'workload':8s} {'policy':24s} {'chan':5s} {'snr':>4s} |   PHY    sim | lost_sync timeouts | errors")
    for k in sorted(g):
        rs = [r for r in g[k] if not r["error"]]
        f = lambda c: np.mean([float(r[c]) for r in rs]) if rs else float("nan")  # noqa: E731
        print(f"{k[0]:8s} {k[1]:24s} {k[2]:5s} {k[3]:>4s} | {f('score'):6.1f} {f('sim_score'):6.1f} |"
              f" {f('lost_sync'):9.1f} {f('timeouts'):8.1f} | {sum(1 for r in g[k] if r['error'])}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(sys.argv[2])
    else:
        main()
