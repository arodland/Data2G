"""One-way bulk speed test, off the clock: two real stacks (data2g.arq.engine:
receiver, ARQ session, gear shifter, PTT delay) stepped in 0.1 s blocks
through run.sh's path, with no sound devices and no waiting on real time.

Each direction is what the loopback does to it: the sender's host
(host.Interpolator to 48 kHz, clipped at full scale, zeros fed to the
engine while keyed), channel.py's Fader (CHANNEL != awgn), `--latency` of
delay, noise.py's Noise (SNR: the input's PEP over the noise in 3000 Hz, as
ARSFI's HFSimulator), tnc.Decimator back to 8 kHz. `--latency` stands for
what the sample clock can't see: TX lead, sound-card buffers and decode
backlog (see LATENCY_S).

A calls B, both send `--warm` bytes, then A writes `--bytes` of random
data at once; the time runs from that write until A holds every byte
acked (B's copy checked). As run.sh with A_BYTES=n, B_BYTES=0, without Pat.

    python scripts/cpu_profile/speedtest.py <awgn|mpg|mpp|mpd|mps | doppler_hz:delay_ms> <snr_db> [--bytes 20000] [--seed 0]
"""
import argparse
import sys
from pathlib import Path

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W.parent.parent))  # this checkout's data2g, not the venv's

from data2g import threads  # noqa: E402

threads.limit(1)

import numpy as np  # noqa: E402

from data2g.arq import session as S  # noqa: E402
from data2g.arq.engine import Engine  # noqa: E402
from data2g.config import FS  # noqa: E402
from data2g.hfchannel import FADING_PRESETS  # noqa: E402
from data2g.host import BW, Interpolator  # noqa: E402
from data2g.tnc import Decimator  # noqa: E402

from channel import Fader  # noqa: E402
from noise import BLK, Noise  # noqa: E402
from noise import FS as FS_DEV  # noqa: E402

BLOCK = FS // 10
# one way, audio out -> audio in. Measured on run.sh loopback recordings (8
# runs, 2026-09-29, under py-spy): each burst's end at the sender to the
# receiver's rx event, summed over both directions, averaged 1.06-1.31 s
# against 0.27 s here with no latency. Includes the decode backlog's tail
# (median alone: ~0.35 s).
LATENCY_S = 0.43


class Side:
    """One station: its engine, its host's resamplers and keying, and the
    path from its speaker to the other's microphone (fading, delay)."""

    def __init__(self, call, seed, fader, delay, record_dir=None):
        self.eng = Engine(call, seed=seed, record_dir=record_dir)
        self.interp, self.dec = Interpolator(FS_DEV), Decimator(FS_DEV)
        self.fader, self.keyed = fader, False
        self.fifo = np.zeros(delay)
        self.airtime = {}  # submode -> seconds on air (PTT delay excluded)

    def step(self, x):
        """48 kHz heard -> 48 kHz arriving at the other side (host.serve's loop)."""
        was = self.eng.tx and self.eng.tx[0]
        y, ptt = self.eng.step(self.dec(x) if not self.keyed else np.zeros(BLOCK))
        if self.eng.tx and self.eng.tx[0] is not was:  # a burst started
            sub = self.eng.tx[0].submode
            self.airtime[sub] = self.airtime.get(sub, 0.0) + (len(self.eng.tx[1]) - self.eng.ptt_delay) / FS
        out = np.clip(self.interp(y), -1, 1) if (ptt or self.keyed) else np.zeros(BLK)
        self.keyed = ptt
        if self.fader:
            out = self.fader(out)
        self.fifo = np.concatenate([self.fifo, out])
        heard, self.fifo = self.fifo[:BLK], self.fifo[BLK:]
        return heard

    def unacked(self):
        s = self.eng.session
        return len(s._pending_write) + (len(s.station.tx.buf) if s.station else 0)


def run(chan, snr, nbytes, seed=0, warm=32, cap=2, latency=LATENCY_S, limit_s=3600.0, record=None):
    """-> dict: phase reached ('done' or where it failed), times, rates."""
    fade = None
    if chan != "awgn":
        dop, dly = (map(float, chan.split(":")) if ":" in chan else
                    (FADING_PRESETS[chan].doppler_hz, FADING_PRESETS[chan].delay_ms))
        fade = lambda s: Fader(dop, dly, s)  # noqa: E731
    delay = int(round(latency * FS_DEV))
    a = Side("W1AW", 3 * seed + 1, fade and fade(3 * seed + 11), delay, record and Path(record, "rec_a"))
    b = Side("K2XYZ", 3 * seed + 2, fade and fade(3 * seed + 12), delay, record and Path(record, "rec_b"))
    noise = Noise(seed)
    rng = np.random.default_rng(seed + 1000)
    to_a, to_b = np.zeros(BLK), np.zeros(BLK)
    got_a, got_b = bytearray(), bytearray()

    def until(cond):
        nonlocal to_a, to_b
        while not cond():
            if a.eng.now > limit_s or S.CLOSED in (a.eng.session.state, b.eng.session.state):
                return False
            n_b, n_a = noise(snr)
            to_a, to_b = b.step(to_b + n_b), a.step(to_a + n_a)  # both hear the last block, then speak
            got_a.extend(a.eng.session.read())
            got_b.extend(b.eng.session.read())
        return True

    res = dict(channel=chan, snr=snr, seed=seed, bytes=nbytes)
    b.eng.listen()
    a.eng.connect("K2XYZ", cap)
    if not until(lambda: a.eng.session.state == b.eng.session.state == S.CONNECTED):
        return dict(res, phase="connect", t=a.eng.now)
    res["t_connect"] = a.eng.now
    wa, wb = rng.bytes(warm), rng.bytes(warm)
    a.eng.session.write(wa)
    b.eng.session.write(wb)
    if not until(lambda: len(got_b) >= warm and len(got_a) >= warm and a.unacked() == b.unacked() == 0):
        return dict(res, phase="exchange", t=a.eng.now)
    assert bytes(got_b) == wa and bytes(got_a) == wb, "warm-up data corrupted"
    data = rng.bytes(nbytes)
    got_b.clear()
    a.airtime.clear()
    t0 = a.eng.now
    a.eng.session.write(data)
    if not until(lambda: a.unacked() == 0 and len(got_b) >= nbytes):
        return dict(res, phase="bulk", t=a.eng.now, delivered=len(got_b))
    assert bytes(got_b) == data, "bulk data corrupted"
    dt = a.eng.now - t0
    return dict(res, phase="done", t=a.eng.now, seconds=dt, bps=8 * nbytes / dt, bpm=60 * nbytes / dt,
                airtime=a.airtime)


def top2(airtime: dict) -> str:
    """The sender's two most-used submodes, as % of its airtime."""
    tot = sum(airtime.values()) or 1.0
    return ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in sorted(airtime.items(), key=lambda kv: -kv[1])[:2])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("channel", help="awgn | mpg | mpp | mpd | mps | doppler_hz:delay_ms")
    ap.add_argument("snr", type=float, help="dB, PEP over noise in 3000 Hz")
    ap.add_argument("--bytes", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm", type=int, default=32, help="bytes each way before the bulk transfer")
    ap.add_argument("--bw", choices=["500", "1200", "2300", "2750"], default="2300", help="session bandwidth cap")
    ap.add_argument("--latency", type=float, default=LATENCY_S, help="one-way audio latency, s")
    ap.add_argument("--record", metavar="DIR", help="record both stations (DIR/rec_a, rec_b), as data2g-host does")
    ap.add_argument("--limit", type=float, default=3600.0, help="give up after this much simulated time, s")
    a = ap.parse_args()
    r = run(a.channel, a.snr, a.bytes, a.seed, a.warm, BW["BW" + a.bw], a.latency, a.limit, a.record)
    head = f"{a.channel} {a.snr:g} dB seed {a.seed}:"
    if r["phase"] == "done":
        print(f"{head} {a.bytes} B in {r['seconds']:.1f} s = {r['bps']:.0f} bit/s = {r['bpm']:.0f} B/min"
              f" (connected at {r['t_connect']:.1f} s; {top2(r['airtime'])})")
    else:
        extra = f", {r['delivered']} of {a.bytes} B delivered" if r["phase"] == "bulk" else ""
        print(f"{head} FAILED during {r['phase']} at {r['t']:.1f} s{extra}")
        sys.exit(1)


if __name__ == "__main__":
    main()
