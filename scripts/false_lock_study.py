"""Receiver.RELEASE_S: what letting go of a suspect, never-confirmed OFDM
lock buys and costs, through the native streaming Receiver, every arm on the
same audio (release_s 0 = the old behaviour).

  hold   false-lock cost: seconds the receiver stays busy with no signal on
         air (a false lock's claimed length), per source: data whose
         preamble was missed (accept_false_locks' case), CW IDs, Hand-like
         8-FSK near 1500 Hz; each followed by 20 s of noise alone
  valid  real bursts of every OFDM submode, AWGN/MPP/MPD/MPG, SNR -12..+4:
         the true burst received (its header found and kept to the end),
         per arm; a burst one arm loses that another receives is decoded
         offline (modem.demodulate) to see whether it delivered anything

    uv run python scripts/false_lock_study.py hold|valid [--jobs 6] [--n 4]
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, "native/build/python")
import data2g_native as N  # noqa: E402

from data2g import codes, cpm, cwid, hfchannel, modem  # noqa: E402
from data2g.arq.engine import receive_accept  # noqa: E402
from data2g.arq.policy import slots_for  # noqa: E402
from data2g.config import BANDS, FS, LEADIN_SAMPLES  # noqa: E402
from data2g.modem import header_samples  # noqa: E402

ARMS = (0.0, 1.0, 2.0, 3.0)
BLK = FS // 50
GAP_S = 20.0
OFDM = [n for n, s in modem.SUBMODES.items()]
CALLS = ["W1AW", "K2XYZ", "VK2ABC/P", "G4ABC", "JA1XYZ", "N0CALL"]


def receivers():
    out = []
    for r_s in ARMS:
        r = N.receiver.Receiver(receive_accept(), list(cpm.GRIDS), True)
        r.release_s = r_s
        out.append(r)
    return out


def source(kind: str, rng) -> tuple[np.ndarray, float]:
    """One false-lock source's signal (unit-ish level) and the SNR to add noise at."""
    if kind == "data":
        names = [n for n in OFDM if "ack" not in n]
        s = modem.SUBMODES[names[rng.integers(len(names))]]
        x = modem.modulate([rng.bytes(codes.payload_bytes(s)) for _ in range(slots_for(s, 12.0))], s)
        return x[LEADIN_SAMPLES + BANDS[s.sync_band].preamble_samples + header_samples(s.sync_band):], 10.0
    if kind == "cw":
        return cwid.audio(cwid.id_text(CALLS[rng.integers(len(CALLS))]), (15, 20, 30)[rng.integers(3)]), \
            float(rng.choice([0.0, 10.0]))
    freq = float(rng.choice([1470.0, 1530.0]))  # hand: Hand channels 0 and 1
    g = cpm.Grid(f"hand{freq:g}", 8, 6.25, center=freq)
    return cpm.tones(g, rng.integers(0, 8, 66)), float(rng.choice([5.0, 20.0]))


def hold_job(args):
    kind, seed, n = args
    rng = np.random.default_rng(seed)
    rx = receivers()
    busy_off = np.zeros(len(ARMS))  # s busy with nothing on air
    heads = np.zeros(len(ARMS), int)
    for _ in range(n):
        x, snr = source(kind, rng)
        y = np.concatenate([np.zeros(FS), x, np.zeros(int(GAP_S * FS))])
        y = hfchannel.awgn(y, snr, seed=int(rng.integers(1 << 30)), s_power=hfchannel.active_power(x))
        sig_end = FS + len(x)
        for i in range(0, len(y) - BLK + 1, BLK):
            for j, r in enumerate(rx):
                heads[j] += sum(k == "header" for k, _ in r.feed(y[i:i + BLK]))
                if i >= sig_end and r.busy:
                    busy_off[j] += BLK / FS
    return kind, busy_off, heads


def hold(a):
    cells = [(k, 100 * i + {"data": 1, "cw": 2, "hand": 3}[k], a.n) for k in ("data", "cw", "hand") for i in range(a.jobs)]
    with Pool(a.jobs) as p:
        res = p.map(hold_job, cells)
    print(f"busy with nothing on air, s per source ({a.n * a.jobs} each), release_s {ARMS}:")
    for k in ("data", "cw", "hand"):
        b = sum(r[1] for r in res if r[0] == k)
        h = sum(r[2] for r in res if r[0] == k)
        print(f"  {k:5s} " + "  ".join(f"{s:.1f}: {v:7.1f} s ({c} headers)" for s, v, c in zip(ARMS, b, h)))


def valid_job(args):
    name, ch, snr, n = args
    s = modem.SUBMODES[name]
    rng = np.random.default_rng(abs(hash((name, ch, snr))) % (1 << 30))
    got = np.zeros(len(ARMS), int)
    lost = []  # (arm, decoded offline) for bursts arm 0 received and an arm didn't
    for t in range(n):
        x = modem.modulate([rng.bytes(codes.payload_bytes(s)) for _ in range(slots_for(s, 8.0))], s)
        y = np.concatenate([np.zeros(FS), x, np.zeros(2 * FS)])
        y = hfchannel.apply_channel(y, snr, fading_preset=None if ch == "awgn" else ch, seed=1000 * t + 7,
                                    freq_offset_hz=float(rng.uniform(-20, 20)))
        rx = receivers()
        mine = [False] * len(ARMS)
        for i in range(0, len(y) - BLK + 1, BLK):
            for j, r in enumerate(rx):
                for k, d in r.feed(y[i:i + BLK]):
                    h = d["header"] if k == "burst" else None
                    if h is not None and d["rx"] is not None and h["spec"] == name and abs(h["start"] - FS) < FS // 2:
                        mine[j] = True
        got += mine
        for j in range(1, len(ARMS)):
            if mine[0] and not mine[j]:
                try:
                    b = modem.demodulate(y)
                    dec = sum(b.crc_ok)
                except modem.SyncError:
                    dec = 0
                lost.append((j, dec))
    return name, ch, snr, got, lost


def valid(a):
    cells = [(nm, ch, snr, a.n) for nm in OFDM for ch in ("awgn", "mpp", "mpd", "mpg") for snr in range(-12, 5, 2)]
    with Pool(a.jobs) as p:
        res = p.map(valid_job, cells, chunksize=1)
    tot = sum(r[3] for r in res)
    print(f"{len(cells) * a.n} real bursts; received with release_s {ARMS}: {tot.tolist()}")
    by = defaultdict(lambda: np.zeros(len(ARMS), int))
    for nm, ch, snr, g, _ in res:
        by[ch] += g
    for ch, g in by.items():
        print(f"  {ch:5s} {g.tolist()}")
    lost = [(nm, ch, snr, j, d) for nm, ch, snr, _, ls in res for j, d in ls]
    for j in range(1, len(ARMS)):
        mine = [x for x in lost if x[3] == j]
        print(f"release_s {ARMS[j]}: {len(mine)} received only without it; "
              f"{sum(1 for x in mine if x[4])} of them decode codewords offline")
        for x in mine[:20]:
            print(f"    {x[0]} {x[1]} {x[2]:+d} dB: {x[4]} codewords offline")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("hold", "valid"))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    {"hold": hold, "valid": valid}[a.what](a)
