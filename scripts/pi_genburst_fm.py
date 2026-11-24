"""FM bursts for data2g-bench-rx (D2GB files, as scripts/pi_genburst.py), through
the FM simulator at a station's fixed audio level (phy_session.FmSessionChannel:
the drive for 15 kHz occupied set by a full-level QPSK burst on the same band,
each burst's own rms against it, so the mic path's per-mode gain is kept).

    DATA2G_PROFILE=fm PYTHONPATH=<tree>:<tree>/scripts python scripts/pi_genburst_fm.py OUT MODE:N_CW:COND:RADIO:CNR:SEED ...

RADIO mic or flat, which also sets the sender's tx_path.
"""
import struct
import sys
from pathlib import Path

import numpy as np

from data2g import codes, modem
from data2g.arq import phy as PHY
from data2g.arq.burst import Slot, TxBurst, data_mask
from data2g.arq.modes import MODES
from data2g.config import FS
from data2g.receiver import search_span

import phy_session as PS


def make(out: Path, mode: str, n: int, cond: str, radio: str, cnr: float, seed: int):
    s = MODES[mode]
    modem.TX_PATH = radio  # a mic radio's station sends tx_path mic, a flat one's flat
    rng = np.random.default_rng(seed)
    rand = lambda sp: bytes(rng.integers(0, 256, codes.payload_bytes(sp), dtype=np.uint8))  # noqa: E731
    ch = PS.FmSessionChannel(cond, radio, cnr, seed, 120.0)
    ref = f"{s.band}-qpsk-r21"  # the level reference: full-level QPSK on the same band
    ch.apply(PHY.tx_audio(TxBurst(ref, [Slot(data_mask(1, 0, 7), 0, rand(MODES[ref]))], 0, 2)), 5.0)
    slots = [Slot(data_mask(1, i, 7), 0, rand(s)) for i in range(n)]
    band = s.sync_band
    head = int(PS.PAD_S * FS) + FS // 2 + search_span((band,))
    y = ch.apply(PHY.tx_audio(TxBurst(mode, slots, 0, 2)), 10.0)
    pb = len(slots[-1].payload)
    name = f"{mode.replace('/', '_')}__{n}__{cond}-{radio}__{cnr:g}__s{seed}.d2gb"
    with open(out / name, "wb") as f:
        f.write(b"D2GB" + struct.pack("<qqii", len(y), head, n, pb) + band.encode().ljust(8, b"\0"))
        f.write(np.asarray(y, "<f8").tobytes())
        for sl in slots:
            f.write(sl.payload.ljust(pb, b"\0")[:pb])


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for a in sys.argv[2:]:
        m, n, cond, radio, cnr, seed = a.split(":")
        make(out, m, int(n), cond, radio, float(cnr), int(seed))
