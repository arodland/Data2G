"""Expected-slot timing in live stacks: speedtest.py's two engines (the
loopback path: resampling, fading, latency, PEP-referenced noise) for a
fixed time, the caller sending bulk data. Per session, a CSV row: bytes
delivered, the caller's timeouts, and its receiver's expected-slot work:
prime_lock tries and locks, locks found while a reply deadline was held
for them (engine._expect_hold), and resets (its own transmission) with
grids still untried (a lock the deadline cut off).

    python scripts/cpu_profile/primed_sessions.py --out runs/x.csv --cells awgn:-12,mpp:-10 --seeds 0-23 [--seconds 600] [--workers 12]

The SNR is speedtest's (PEP over noise in 3000 Hz): a loss-study cell
label (DATA2G_PEP_REF_DB=5) is about 4.2 dB lower.
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import speedtest as T  # first: it sets the thread caps before numpy loads

import numpy as np  # noqa: E402

from data2g.arq import session as S  # noqa: E402
from data2g.hfchannel import FADING_PRESETS  # noqa: E402
from channel import Fader  # noqa: E402
from noise import FS as FS_DEV  # noqa: E402
from noise import Noise  # noqa: E402

FIELDS = ["channel", "snr", "seed", "phase", "seconds", "delivered", "bps", "timeouts", "tries", "locks",
          "held_locks", "held", "cut", "cpm_replies"]


def watch(eng, c):
    """Count eng's expected-slot tries, locks and cut-off windows into c."""
    r = eng.receiver
    prime, reset = r._primed, r.reset

    def primed():
        c["tries"] += 1
        lock = prime()
        if lock is not None:
            c["locks"] += 1
            # the deadline was held this turn (new code only)
            c["held_locks"] += getattr(eng, "_expect_for", None) is not None and eng._expect_for == eng._tx_end
        return lock

    def reset_():
        c["cut"] += r.expected is not None  # own transmission over grids still untried
        reset()
    r._primed, r.reset = primed, reset_
    hold = getattr(eng, "_expect_hold", None)
    if hold is not None:
        def counted(t):
            before = eng.session._deadline
            hold(t)
            c["held"] += before is not None and eng.session._deadline != before
        eng._expect_hold = counted


def run(job):
    chan, snr, seed, seconds = job
    fade = None
    if chan != "awgn":
        p = FADING_PRESETS[chan]
        fade = lambda s: Fader(p.doppler_hz, p.delay_ms, s)  # noqa: E731
    delay = int(round(T.LATENCY_S * FS_DEV))
    a = T.Side("W1AW", 3 * seed + 1, fade and fade(3 * seed + 11), delay)
    b = T.Side("K2XYZ", 3 * seed + 2, fade and fade(3 * seed + 12), delay)
    c = dict.fromkeys(["tries", "locks", "held_locks", "held", "cut"], 0)
    watch(a.eng, c)
    noise = Noise(seed)
    to_a, to_b = np.zeros(T.BLK), np.zeros(T.BLK)
    got = bytearray()
    sent = {"cpm": 0}
    b.eng.listen()
    a.eng.connect("K2XYZ", T.BW["BW2300"])
    data = np.random.default_rng(seed + 1000).bytes(400_000)
    wrote = False
    phase = "connect"
    while a.eng.now < seconds:
        if S.CLOSED in (a.eng.session.state, b.eng.session.state):
            phase = "closed"
            break
        n_b, n_a = noise(snr)
        idle_b = not b.eng.tx_samples
        to_a, to_b = b.step(to_b + n_b), a.step(to_a + n_a)
        if idle_b and b.eng.tx_samples and b.eng.tx[0].submode.startswith("fsk"):
            sent["cpm"] += 1
        got.extend(b.eng.session.read())
        if not wrote and a.eng.session.state == S.CONNECTED:
            a.eng.session.write(data)
            wrote, phase = True, "bulk"
    st = a.eng.session.station
    stats = st.stats if st is not None else {}
    return dict(channel=chan, snr=snr, seed=seed, phase=phase, seconds=round(a.eng.now, 1), delivered=len(got),
                bps=round(8 * len(got) / max(a.eng.now, 1e-9), 2), timeouts=stats.get("timeouts", 0),
                cpm_replies=sent["cpm"], **c)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", required=True, help="channel:snr,...")
    ap.add_argument("--seeds", default="0-23")
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    cells = [(c.split(":")[0], float(c.split(":")[1])) for c in a.cells.split(",")]
    jobs = [(ch, snr, s, a.seconds) for ch, snr in cells for s in range(lo, hi + 1)]
    with open(a.out, "w", newline="") as f, Pool(a.workers, maxtasksperchild=4) as pool:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for row in pool.imap_unordered(run, jobs):
            w.writerow(row)
            f.flush()
            print(row, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
