"""A compact CONNECT's 4 robust tries (fsk16r25-r1/2, bit-identical), each
decoded alone vs combined with the failed tries before it
(ModemRx.decode_repeat, docs/arq.md §3a). Per cell: heard by try k.

    python -m scripts.connect_repeat_study 150

Found (2026-11-10, 150 a cell), heard by try 4 alone -> combined:
AWGN -16 dB 0 -> 46, -15 dB 0 -> 145, -14 dB 49 -> 150; MPP -14 dB 7 -> 91,
-12 dB 92 -> 147, -10 dB 150 -> 150; MPD -12 dB 75 -> 150. No wrong frame.
"""
import sys
from data2g import threads
threads.limit(1)
import numpy as np
from multiprocessing import Pool
from data2g import cpm, hfchannel
from data2g.config import FS
from data2g.arq import burst as B, frames as F, phy as PHY, link as L

MODE = "fsk16r25-r1/2"
TRIES = 4

def trial(a):
    chan, snr, seed = a
    rng = np.random.default_rng(seed)
    spec = cpm.SPECS[MODE] if isinstance(cpm.SPECS, dict) else next(s for s in cpm.SPECS if s.name == MODE)
    g = cpm.GRIDS[spec.grid] if hasattr(spec, "grid") else cpm.GRIDS["c16r25"]
    cpb = PHY.codes.payload_bytes(cpm.CTL[g.name])
    body = F.Connect(5, "W1AW", "K2XYZ", int(rng.integers(1 << 16)), 2, 10).pack()
    p = F.pack_connect(body) + bytes(cpb - F.COMPACT_BYTES)
    x = PHY.tx_audio(B.TxBurst(MODE, [B.Slot(B.COMPACT_CONNECT, 0, p)], 0))
    pw = hfchannel.active_power(x)
    store, alone, comb = {}, None, None
    locks = 0
    for k in range(TRIES):
        y = hfchannel.apply_channel(x, None, float(rng.uniform(-30, 30)), fading_preset=None if chan == "awgn" else chan,
                                    seed=seed * 10 + k)
        y = hfchannel.awgn(np.concatenate([np.zeros(FS // 2), y, np.zeros(FS // 2)]), snr, seed=seed * 10 + k + 5, s_power=pw)
        lock = cpm.find(g, y)
        if lock is None or abs(lock["start"] - FS // 2) > 2 * g.T:
            continue
        locks += 1
        rx = PHY.ModemRx(cpm.receive(y, lock), store)
        if rx.n_cw != 1:
            continue
        if rx.decode(0, B.COMPACT_CONNECT, 0, None) == p:
            alone = k if alone is None else alone
            comb = k if comb is None else comb
        elif comb is None:
            got = rx.decode_repeat(0, B.COMPACT_CONNECT, L.REPEAT_SESSION_KEY)
            if got == p:
                comb = k
            elif got is not None:
                return a, (alone, "WRONG", locks)
    return a, (alone, comb, locks)

if __name__ == "__main__":
    N = int(sys.argv[1])
    cells = [("awgn", s) for s in (-17, -16, -15, -14)] + [("mpp", s) for s in (-14, -12, -10, -8)] + [("mpd", s) for s in (-12, -9, -6)]
    with Pool(8) as pool:
        res = pool.map(trial, [c + (3000 + k,) for c in cells for k in range(N)], chunksize=2)
    print(f"chan snr | locks/try | connected by try 1..{TRIES}: alone | with combining | wrong")
    for c in cells:
        rs = [o for a, o in res if a[:2] == c]
        al = [sum(o[0] is not None and o[0] <= k for o in rs) for k in range(TRIES)]
        co = [sum(isinstance(o[1], int) and o[1] <= k for o in rs) for k in range(TRIES)]
        print(f"{c[0]:4s} {c[1]:4d} | {sum(o[2] for o in rs) / (TRIES * len(rs)):.2f} | {al} | {co} | {sum(o[1] == 'WRONG' for o in rs)}  (of {len(rs)})")
