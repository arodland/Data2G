"""crc_exposure.py's count through two whole engines (data2g.arq.engine,
the KISS personality on both): the streaming receiver, KISS's key probes
on every burst it hears, CQ checks, then the session. AWGN on the sample
clock; engines send every burst at a full-scale peak (PEP-referenced).
Scenarios: an ARQ session with data both ways; KISS UI traffic between
sessions.

    uv run --no-sync python scripts/crc_exposure_engine.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from test_engine import link  # noqa: E402
from test_kiss import frame  # noqa: E402

from data2g import codes  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402
from data2g.arq.engine import Engine  # noqa: E402
from data2g.kisslink import KissLink  # noqa: E402

counts = Counter()
_start, _init, _decode = Engine._start_tx, PHY.ModemRx.__init__, PHY.ModemRx.decode
on_air = {}  # engine -> its last burst


def start(self, burst, t):
    on_air.setdefault(id(self), []).append(burst)
    del on_air[id(self)][:-6]
    return _start(self, burst, t)


def init(self, r, store):
    """The burst heard: the other engine's latest with this mode and length
    (it may already be sending the next when this one's decode runs)."""
    _init(self, r, store)
    self._tx = None
    for e_id, bs in on_air.items():
        if e_id != HEARING[0]:
            for b in reversed(bs):
                if b.submode == r["spec"].name and len(b.slots) == r["n_cw"]:
                    self._tx = b
                    break


def decode(self, slot, mask_id, rv, key):
    p = _decode(self, slot, mask_id, rv, key)
    if slot < self.n_cw and self._tx is not None:
        spec = self._spec(slot)
        sent = self._tx.slots[slot] if slot < len(self._tx.slots) else None
        true = sent is not None and PHY.mask_value(sent.mask_id) == PHY.mask_value(mask_id)  # key 0: all mask 0
        got = "none" if p is None else ("right" if true and p == sent.payload else "WRONG")
        counts[(f"{spec.code}{codes.crc_bits(spec)} k{spec.k}", "true" if true else "wrong-mask", got)] += 1
    return p


HEARING = [None]
_step = Engine.step


def step(self, x):
    HEARING[0] = id(self)
    return _step(self, x)


Engine._start_tx, PHY.ModemRx.__init__, PHY.ModemRx.decode, Engine.step = start, init, decode, step


def run(args):
    scenario, snr, seconds = args
    counts.clear()
    a, b = Engine("W1AW", seed=1, kiss=KissLink()), Engine("K2XYZ", seed=2, kiss=KissLink())
    if scenario == "arq":
        b.listen()
        a.connect("K2XYZ", 2)
        a.session.write(np.random.default_rng(3).bytes(200000))
        b.session.write(np.random.default_rng(4).bytes(20000))
        link(a, b, snr, seconds, lambda: False)
    else:  # KISS: a UI frame from one, then a connected-mode frame from the other, 20 s apart
        t = 0
        while t < seconds:
            a.kiss.enqueue(frame("APRS", "W1AW", 0x03, b"!beacon " + bytes(40)))
            link(a, b, snr, 20, lambda: False, seed=t)
            b.kiss.enqueue(frame("W1AW", "K2XYZ", 0x00, b"x" * 200))
            link(a, b, snr, 20, lambda: False, seed=t + 1)
            t += 40
    return (scenario, snr, seconds), Counter(counts)


def main():
    jobs = [(s, snr, 600) for s in ("arq", "kiss") for snr in (0.0, 10.0)]
    with Pool(4) as pool:
        for (scen, snr, secs), c in pool.imap_unordered(run, jobs):
            print(f"== {scen} AWGN {snr:+.0f} dB: decodes per hour (both engines)", flush=True)
            for k, v in sorted(c.items()):
                print(f"  {k[0]:14s} {k[1]:10s} {k[2]:5s} {v * 3600 / secs:9.0f}", flush=True)


if __name__ == "__main__":
    main()
