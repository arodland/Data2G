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

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import hfchannel, modem
from data2g import interference as INTF
from data2g.arq import phy as PHY
from data2g.config import BANDS, FS, LEADIN_SAMPLES, SNR_REF_BW_HZ, SUBMODES
from data2g.tnc import NoiseProfile, receive_any

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
PRE_S = 30.0  # each station's receiver listened this long before the session (its noise profile)


class ContinuousChannel:
    """Two-path Watterson fading for a whole session, read at any time, and
    each receiving station's own interference (data2g.interference)."""

    def __init__(self, chan: str, snr_db: float, seed: int, horizon: float, doppler=None, delay_ms=None,
                 interference=None):
        """`chan`: a preset name, unless `doppler` (Hz) and `delay_ms` are given.
        `interference`: (station 0's, station 1's) interference.Spec; None: clean."""
        self.snr_db, self.rng = snr_db, np.random.default_rng(seed)
        self.seed, self.horizon = seed, horizon
        specs = interference or (INTF.Spec(), INTF.Spec())
        # each timeline from -PRE_S: the stations listened before the session
        self.intf = [INTF.Interference(sp, seed * 2 + i, PRE_S + horizon + 60) for i, sp in enumerate(specs)]
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

    def interference(self, station: int, t0: float, n: int, sigma: float) -> np.ndarray:
        """Station's interference from session time t0 (PRE_S before 0 at the earliest)."""
        return self.intf[station].render(t0 + PRE_S, n, sigma)

    def sigma(self, s_power: float) -> float:
        """The floor's standard deviation (white over FS/2) for a burst of
        signal power s_power at the current SNR (hfchannel.awgn's)."""
        return float(np.sqrt(s_power * (FS / 2) / SNR_REF_BW_HZ / 10 ** (self.snr_db / 10)))

    def floor_sigma(self) -> float:
        """The floor's standard deviation between bursts: a full-scale burst's
        (peak 1, as data2g-host sends), against its peak with PEP_REF_DB."""
        return self.sigma(0.5 / 10 ** ((PEP_REF_DB or 3.0) / 10))

    def apply(self, x: np.ndarray, t0: float, rx: int | None = None) -> np.ndarray:
        """Audio sent at t0 -> what the receiver hears, PAD_S of noise either
        side; station `rx`'s interference added when given."""
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
        y = hfchannel.awgn(y, self.snr_db, seed=int(self.rng.integers(1 << 31)), s_power=s_power)
        if rx is not None and not self.intf[rx].spec.clean:
            y += self.interference(rx, t0 - PAD_S, len(y), self.sigma(s_power))
        return y


class StationNoise:
    """What each station's receiver hears between bursts (the floor and its
    interference, not while it transmits), fed to its NoiseProfile as the
    engine does on air: its own bursts and the radio's recovery after them,
    and the bursts whose header it heard, are not noise."""

    CHUNK_S = 1.0

    def __init__(self, ch: ContinuousChannel):
        self.ch = ch
        self.profile = [NoiseProfile(), NoiseProfile()]
        self.fed = [-PRE_S, -PRE_S]  # listening before the session, as a station does before it calls
        self.tx = [[], []]  # each station's own bursts, (start, end)
        self.rng = np.random.default_rng(np.random.SeedSequence([ch.seed, 77]))  # not the bursts' noise

    def sent(self, station: int, t0: float, end: float):
        self.tx[station].append((t0, end))

    def feed(self, station: int, t: float):
        """Station's receiver up to time t."""
        a = self.fed[station]
        if t <= a:
            return
        p, clean = self.profile[station], self.ch.intf[station].spec.clean
        spans, cur = [], a
        for s0, e0 in self.tx[station]:
            if e0 <= cur or s0 >= t:
                continue
            if s0 > cur:
                spans.append((cur, s0))
            cur = max(cur, e0)
            p.mark(e0, e0 + NoiseProfile.RECOVER_S)
        if cur < t:
            spans.append((cur, t))
        self.tx[station] = [(s0, e0) for s0, e0 in self.tx[station] if e0 > t - 10]
        sigma = self.ch.floor_sigma()
        for s0, e0 in spans:
            ts = s0
            while ts < e0:
                n = int(round(min(self.CHUNK_S, e0 - ts) * FS))
                if n <= 0:
                    break
                y = self.rng.normal(0, sigma, n)
                if not clean:
                    y += self.ch.interference(station, ts, n, sigma)
                p.feed(y, ts)
                ts += n / FS
        self.fed[station] = t


class RealPhy:
    def __init__(self, ch: ContinuousChannel):
        self.ch, self.decode_s = ch, []
        self.noise = StationNoise(ch)

    def listen(self, burst, t0: float, end: float) -> int:
        """-> the station that hears `burst` (direction 0 is station 0's: 1
        hears), its receiver fed up to t0; the sender's transmission noted."""
        sender = burst.slots[0].mask_id[1] & 1
        self.noise.sent(sender, t0, end)
        self.noise.feed(1 - sender, t0)
        return 1 - sender

    def heard_noise(self, rx: int, r, t0: float, end: float) -> dict | None:
        """The receiver's noise profile as it reads `r` (a heard burst is not noise)."""
        if r is not None:
            self.noise.profile[rx].mark(t0, end)  # as the engine: its lead-in to its end
        return self.noise.profile[rx].snapshot()

    def hear(self, x, t0, rx=None):
        """The receiver's result for audio x sent at t0 (either family), or None."""
        try:
            return receive_any(self.ch.apply(x, t0, rx), lead=int(PAD_S * FS) + FS // 2)
        except modem.SyncError:  # ponytail: an OFDM header read past the audio
            return None

    def send(self, burst, t0):
        """linksim.SimPhy.send's contract, on the real modem."""
        x = PHY.tx_audio(burst)
        end = t0 + len(x) / FS
        t = time.perf_counter()
        rx = self.listen(burst, t0, end)
        r = self.hear(x, t0, rx)
        noise = self.heard_noise(rx, r, t0, end)
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
        return end, (t_hdr, spec.name, r["n_cw"]), make_rx, dict(PHY.measure(r), noise=noise)


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
