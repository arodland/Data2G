"""Bursts for data2g-bench-rx (D2GB files), made with the data2g on PYTHONPATH
(each tree its own: protocol 13 vs 14).

    PYTHONPATH=<tree>:<tree>/scripts python scripts/pi_genburst.py OUT MODE:N_CW:CHAN:SNR:SEED[:dup] ...

N_CW counts every slot (CPM: control slot 0 + N_CW - 1 data; dup: control slots 0 and 1, ARQ_DUP). Channel:
phy_session.ContinuousChannel (average-power SNR, 2500 Hz; CFO_HZ; PAD_S).
"""
import struct
import sys
from pathlib import Path

import numpy as np

from data2g import codes, cpm
from data2g.arq import phy as PHY
from data2g.arq.burst import Slot, TxBurst, ctl_mask, data_mask
from data2g.arq.modes import MODES, ctl_spec, is_cpm
from data2g.config import FS
from data2g.receiver import search_span

import phy_session as PS


def make(out: Path, mode: str, n: int, chan: str, snr: float, seed: int, dup: bool = False):
    s = MODES[mode]
    rng = np.random.default_rng(seed)
    rand = lambda sp: bytes(rng.integers(0, 256, codes.payload_bytes(sp), dtype=np.uint8))  # noqa: E731
    if is_cpm(s):
        slots = [Slot(ctl_mask(1, 0, 7), 0, rand(ctl_spec(s)))]
        if dup:
            slots.append(Slot(ctl_mask(1, 0, 7), 1, slots[0].payload))
        slots += [Slot(data_mask(1, i, 7), 0, rand(s)) for i in range(len(slots), n)]
        band = s.grid
        head = int(PS.PAD_S * FS) + FS // 2 + search_span((), (s.grid,))
    else:
        slots = [Slot(data_mask(1, i, 7), 0, rand(s)) for i in range(n)]
        band = s.sync_band
        head = int(PS.PAD_S * FS) + FS // 2 + search_span((band,))
    b = TxBurst(mode, slots, 0, 2)
    y = PS.ContinuousChannel(chan, snr, seed, 120.0).apply(PHY.tx_audio(b), 10.0)
    pb = len(slots[-1].payload)
    n_ctl = 2 if dup else 1
    assert all(len(sl.payload) == pb for sl in slots[n_ctl:])
    name = f"{mode.replace('/', '_')}__{n}{'d' if dup else ''}__{chan}__{snr:g}__s{seed}.d2gb"
    with open(out / name, "wb") as f:
        f.write(b"D2GB" + struct.pack("<qqii", len(y), head, n, pb) + band.encode().ljust(8, b"\0"))
        f.write(np.asarray(y, "<f8").tobytes())
        for sl in slots:  # the control payload is shorter: padded (the bench compares slot 0 against it)
            f.write(sl.payload.ljust(pb, b"\0")[:pb])
        if is_cpm(s):
            f.write(mode.encode().ljust(16, b"\0") + struct.pack("<qdi", int(PS.PAD_S * FS), PS.CFO_HZ * 1.0, n_ctl))


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for a in sys.argv[2:]:
        m, n, ch, snr, seed, *dup = a.split(":")
        make(out, m, int(n), ch, float(snr), int(seed), dup == ["dup"])
