"""False headers on data whose preamble was missed: random OFDM bursts
(12 s class, random payloads), preamble and header cut, 0.5-2 s gaps,
AWGN at 10 dB; the native Receiver with the 16 s Accept and
engine.receive_accept on the same audio. Pure noise reaches no header test
(0 in 24 h with either), so this is the case that measures them.

    uv run python scripts/accept_false_locks.py 12 60   # jobs, audio minutes each
"""

from data2g import threads

threads.limit(1)

import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "native/build/python")
import data2g_native as N

from data2g import codes, cpm, hfchannel, modem
from data2g.arq.engine import receive_accept
from data2g.arq.policy import slots_for
from data2g.config import BANDS, FS, LEADIN_SAMPLES
from data2g.modem import header_samples

BLK = FS // 50
NAMES = [n for n in modem.SUBMODES if "ack" not in n]


def job(args):
    seed, minutes = args
    rng = np.random.default_rng(seed)
    accs = {"old": modem.Accept.of(None, 16.0), "new": receive_accept()}
    rx = {k: N.receiver.Receiver(a, list(cpm.GRIDS), True) for k, a in accs.items()}
    out = {k: [] for k in rx}
    t = 0
    while t < minutes * 60 * FS:
        s = modem.SUBMODES[NAMES[rng.integers(len(NAMES))]]
        pb = codes.payload_bytes(s)
        x = modem.modulate([rng.bytes(pb) for _ in range(slots_for(s, 12.0))], s)
        x = x[LEADIN_SAMPLES + BANDS[s.sync_band].preamble_samples + header_samples(s.sync_band):]
        y = np.concatenate([np.zeros(int(rng.uniform(0.5, 2) * FS)), x])
        y = hfchannel.awgn(y, 10.0, seed=int(rng.integers(1 << 30)), s_power=hfchannel.active_power(x))
        for i in range(0, len(y), BLK):
            for k, r in rx.items():
                out[k] += [(e[1]["spec"], e[1]["n_cw"]) for e in r.feed(y[i:i + BLK]) if e[0] == "header"]
        t += len(y)
    return out


if __name__ == "__main__":
    jobs, minutes = int(sys.argv[1]), float(sys.argv[2])
    t0 = time.time()
    with Pool(jobs) as p:
        res = p.map(job, [(2000 + k, minutes) for k in range(jobs)])
    hours = jobs * minutes / 60
    for k in ("old", "new"):
        hs = [h for r in res for h in r[k]]
        print(f"{k}: {len(hs)} false headers in {hours:.1f} h = {len(hs)/hours:.1f}/h")
    cap = dict(modem.Accept.of(None, 16.0).max_cw)
    extra = [h for r in res for h in r["new"] if h[1] > cap[h[0]]]
    print(f"new headers past the old cap: {len(extra)} {sorted(set(extra))}")
    print(f"wall {time.time()-t0:.0f} s")
