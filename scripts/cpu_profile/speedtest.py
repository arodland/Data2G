"""One-way bulk speed test, off the clock: two real stacks (data2g.arq.engine:
receiver, ARQ session, gear shifter, PTT delay) stepped in 0.1 s blocks
through run.sh's path, with no sound devices and no waiting on real time.

Each direction is what the loopback does to it: the sender's host
(audio.Interpolator to 48 kHz, clipped at full scale, zeros fed to the
engine while keyed), channel.py's Fader (CHANNEL != awgn), `--latency` of
delay, noise.py's Noise (SNR: the input's PEP over the noise in 3000 Hz, as
ARSFI's HFSimulator), audio.Decimator back to 8 kHz. `--latency` stands for
what the sample clock can't see: TX lead, sound-card buffers and decode
backlog (see LATENCY_S).

A calls B, both send `--warm` bytes, then A writes `--bytes` of random
data at once; the time runs from that write until A holds every byte
acked (B's copy checked). As run.sh with A_BYTES=n, B_BYTES=0, without Pat.
With `--reverse`, B (the callee) writes the bulk data and A receives it: as
A_BYTES=0, B_BYTES=n. With `--both`, both write at once, A ceil(n/2) bytes
and B floor(n/2), and the time runs until both hold all of theirs acked (and
both copies checked).

    python scripts/cpu_profile/speedtest.py <awgn|mpg|mpp|mpd|mps | doppler_hz:delay_ms> <snr_db> [--bytes 20000] [--seed 0] [--reverse | --both]
"""
import argparse
import logging
import sys
from pathlib import Path

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W.parent.parent))  # this checkout's data2g, not the venv's

from data2g import threads  # noqa: E402

threads.limit(1)

import numpy as np  # noqa: E402

from data2g.arq import session as S  # noqa: E402
from data2g.arq.engine import Engine  # noqa: E402
from data2g.audio import Decimator, Interpolator  # noqa: E402
from data2g.config import FS  # noqa: E402
from data2g.hfchannel import FADING_PRESETS  # noqa: E402
from data2g.host import BW  # noqa: E402

from channel import Fader  # noqa: E402
from noise import BLK, Noise  # noqa: E402
from noise import FS as FS_DEV  # noqa: E402

BLOCK = FS // 10


def log_arg(ap):
    """--log-level: data2g's log (data2g-host's), to stderr. Unconfigured,
    Python prints its warnings bare (receiver "burst lost" lines)."""
    ap.add_argument("--log-level", default="ERROR", help="data2g's log on stderr (the host's is INFO)")


def direction_arg(ap):
    """--reverse / --both -> a.direction: who sends the bulk data."""
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--reverse", dest="direction", action="store_const", const="reverse", default="forward",
                   help="the callee sends the bulk data, the caller receives")
    g.add_argument("--both", dest="direction", action="store_const", const="both",
                   help="both send at once, half the bytes each (the caller the odd one)")


def direction_tag(direction):
    """'' for forward, else ' reverse' / ' both'."""
    return "" if direction == "forward" else f" {direction}"


def log_setup(a):
    logging.basicConfig(level=a.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
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
        idle = not self.eng.tx_samples  # a step starts a burst only from idle, and none is under a block
        y, ptt = self.eng.step(self.dec(x) if not self.keyed else np.zeros(BLOCK))
        if idle and self.eng.tx_samples:  # a burst started (tx itself only now: native builds it per access)
            sub = self.eng.tx[0].submode
            self.airtime[sub] = self.airtime.get(sub, 0.0) + (self.eng.tx_samples - self.eng.ptt_delay) / FS
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


def run(chan, snr, nbytes, seed=0, warm=32, cap=2, latency=LATENCY_S, limit_s=3600.0, record=None,
        direction="forward"):
    """-> dict: phase reached ('done' or where it failed), times, rates.
    direction: who sends the bulk data: 'forward' the caller (A), 'reverse'
    the callee (B), 'both' each half of it at once (A the odd byte); then
    seconds is until both are done, and seconds_a/seconds_b each one's."""
    assert direction in ("forward", "reverse", "both"), direction
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
    n_a = {"forward": nbytes, "reverse": 0, "both": (nbytes + 1) // 2}[direction]
    # (name, sender, its data, what the other side has read of it), for each side that sends
    flows = [(k, tx, rng.bytes(n), got) for k, tx, n, got in (("a", a, n_a, got_b), ("b", b, nbytes - n_a, got_a))
             if n]
    for _, tx, _, got in flows:
        got.clear()
        tx.airtime.clear()
    t0 = a.eng.now
    for _, tx, data, _ in flows:
        tx.eng.session.write(data)
    finished = {}  # name -> seconds from t0 until its sender held all of it acked and the other read it

    def all_done():
        for k, tx, data, got in flows:
            if k not in finished and tx.unacked() == 0 and len(got) >= len(data):
                finished[k] = a.eng.now - t0
        return len(finished) == len(flows)

    if not until(all_done):
        return dict(res, phase="bulk", t=a.eng.now, delivered=sum(len(got) for *_, got in flows))
    for _, _, data, got in flows:
        assert bytes(got) == data, "bulk data corrupted"
    dt = a.eng.now - t0
    airtime = {}  # pooled over the senders
    for _, tx, _, _ in flows:
        for k, v in tx.airtime.items():
            airtime[k] = airtime.get(k, 0.0) + v
    return dict(res, phase="done", t=a.eng.now, seconds=dt, bps=8 * nbytes / dt, bpm=60 * nbytes / dt,
                airtime=airtime, **{f"seconds_{k}": v for k, v in finished.items()})


def top2(airtime: dict) -> str:
    """The sender's (senders') two most-used submodes, as % of its airtime."""
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
    direction_arg(ap)
    log_arg(ap)
    a = ap.parse_args()
    log_setup(a)
    r = run(a.channel, a.snr, a.bytes, a.seed, a.warm, BW["BW" + a.bw], a.latency, a.limit, a.record, a.direction)
    head = f"{a.channel} {a.snr:g} dB seed {a.seed}{direction_tag(a.direction)}:"
    if r["phase"] == "done":
        each = (f"; A->B {r['seconds_a']:.1f} s, B->A {r['seconds_b']:.1f} s"
                if "seconds_a" in r and "seconds_b" in r else "")
        print(f"{head} {a.bytes} B in {r['seconds']:.1f} s = {r['bps']:.0f} bit/s = {r['bpm']:.0f} B/min"
              f" (connected at {r['t_connect']:.1f} s{each}; {top2(r['airtime'])})")
    else:
        extra = f", {r['delivered']} of {a.bytes} B delivered" if r["phase"] == "bulk" else ""
        print(f"{head} FAILED during {r['phase']} at {r['t']:.1f} s{extra}")
        sys.exit(1)


if __name__ == "__main__":
    main()
