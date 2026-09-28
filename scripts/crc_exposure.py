"""How many chances a session gives a CRC to accept the wrong thing
(companion to scripts/crc_study.py): every codeword decode an ARQ session
makes (scripts/phy_session.py's real modem, linksim's stations), by
- the code and CRC it decodes with;
- true mask (the one the slot was sent with) or wrong mask (a probe: a
  seq mapped to the wrong slot, another key);
- whether it returned a payload, and whether that payload was right.
Per hour of session, PEP-referenced (DATA2G_PEP_REF_DB, default 5 here).

    uv run --no-sync python scripts/crc_exposure.py
"""

import os

from data2g import threads  # noqa: E402

threads.limit(1)
os.environ.setdefault("DATA2G_PEP_REF_DB", "5")

import argparse
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import linksim as L  # noqa: E402
import phy_session as PS  # noqa: E402

from data2g import codes  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402

CELLS = (("mpg", -4.0), ("mpp", -4.0), ("mpg", 0.0), ("mpp", 0.0), ("mpg", 8.0))
last = [None]
counts = Counter()
_tx_audio, _receive_any, _init, _decode = PHY.tx_audio, PS.receive_any, PHY.ModemRx.__init__, PHY.ModemRx.decode


def tx_audio(burst):
    last[0] = burst
    return _tx_audio(burst)


def receive_any(*a, **k):
    r = _receive_any(*a, **k)
    if r is not None:
        r["_txb"] = last[0]
    return r


def init(self, r, store):
    _init(self, r, store)
    self._tx = r.get("_txb")


def decode(self, slot, mask_id, rv, key):
    p = _decode(self, slot, mask_id, rv, key)
    if slot < self.n_cw and self._tx is not None:
        spec = self._spec(slot)
        sent = self._tx.slots[slot] if slot < len(self._tx.slots) else None
        true = sent is not None and PHY.mask_value(sent.mask_id) == PHY.mask_value(mask_id)  # key 0: all mask 0
        got = "none" if p is None else ("right" if true and p == sent.payload else "WRONG")
        counts[(f"{spec.code}{codes.crc_bits(spec)} k{spec.k}", "true" if true else "wrong-mask", got)] += 1
    return p


PHY.tx_audio, PS.receive_any, PHY.ModemRx.__init__, PHY.ModemRx.decode = tx_audio, receive_any, init, decode


def one(args):
    chan, snr, seed, horizon = args
    counts.clear()
    ch = PS.ContinuousChannel(chan, snr, seed, horizon + 60)
    L.run(L.make_policy("shift+cpm"), L.make_policy("shift+cpm"), None, L.WORKLOADS["bulk"](random.Random(seed + 7)),
          seed=seed, horizon=horizon, phy=PS.RealPhy(ch))
    return (chan, snr), Counter(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--horizon", type=float, default=300.0)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    tot = {}
    with Pool(a.jobs) as pool:
        for cell, c in pool.imap_unordered(one, [(ch, s, k, a.horizon) for ch, s in CELLS for k in range(a.seeds)]):
            tot.setdefault(cell, Counter()).update(c)
    hours = a.seeds * a.horizon / 3600
    for cell in CELLS:
        print(f"== {cell[0]} {cell[1]:+.0f} dB: decodes per hour of session (both stations)")
        for k, v in sorted(tot.get(cell, {}).items()):
            print(f"  {k[0]:14s} {k[1]:10s} {k[2]:5s} {v / hours:9.0f}")


if __name__ == "__main__":
    main()
