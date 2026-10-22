"""Hand burst link study: one Hand through AWGN and the fading presets, per
tone spacing, at the same transmit power -> decode rate vs SNR, 10% points.

A Hand (the D2G net's show of hands; ~/d2g-hand-burst-response.md): 8-FSK,
continuous phase (cpm.tones), 63 payload bits + CRC-16 seeded with the
request tag, CA-SCL polar K=79 of E=162, a fixed random interleaver from
coded bits to tone symbols (a symbol's 3 bits fade together), and 54 data
symbols between two Welch Costas blocks:

    [sync 6][data 27][sync 6][data 27]   66 symbols

Spacings 6.25 / 12.5 / 25 Hz last 10.6 / 5.3 / 2.6 s: the faster ones fit
1 / 2 / 4 time slots in the same round, so the comparison is at the same
hands per round. SNR is in 2500 Hz, against the Hand's own (constant
envelope: PEP = average) power, so a shorter Hand carries less energy.

Receiver: net control knows the slot and channel; each trial puts the Hand
within +-0.2 s (--dt) and +-3 Hz of nominal, and the receiver searches +-0.25 s (--search)
(T/8 steps) x +-5 Hz (R/4 steps) on the sync blocks' tone shares, decodes
the best TOP candidates, and takes the first list path whose CRC passes.
A wrong payload that passes is counted apart (false).

Trials are seeded per trial index, the same seed at every SNR (paired).

    python scripts/hand_study.py --trials 200 --workers 4 --out runs/hand_study/link.json
"""

import argparse
import json
import os
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data2g import threads  # noqa: E402

threads.cap_env(1)

import numpy as np  # noqa: E402

from data2g import cpm, hfchannel, polar  # noqa: E402
from data2g.codes import POLAR_DESIGN_SNR_DB, POLAR_LIST  # noqa: E402
from data2g.config import FS  # noqa: E402

PAYLOAD, CRC, E = 63, 16, 162
K = PAYLOAD + CRC
SPACINGS = (6.25, 12.5, 25.0)
CHANNELS = ("awgn", "mpg", "mpp", "mpd", "mps")
TAG = 0x5A  # the request tag the CRC is seeded with
LEAD_S = 0.8  # silence either side of the nominal slot: more than any --dt
DT_S, DF_HZ = 0.2, 3.0  # where the Hand really lands
SEARCH_S, SEARCH_HZ = 0.25, 5.0  # where net control looks
TOP = 3


def crc16(bits: np.ndarray, init: int) -> np.ndarray:
    """CRC-16/CCITT (0x1021) of a bit vector, register seeded with `init`."""
    c = init
    for b in bits:
        fb = ((c >> 15) & 1) ^ int(b)
        c = ((c << 1) & 0xFFFF) ^ (0x1021 if fb else 0)
    return np.array([(c >> (15 - i)) & 1 for i in range(16)], np.uint8)


@lru_cache(maxsize=None)
def code() -> polar.PolarCode:
    return polar.PolarCode(K, E, design_snr_db=POLAR_DESIGN_SNR_DB)


@lru_cache(maxsize=None)
def interleaver() -> np.ndarray:
    return np.random.default_rng(0).permutation(E)


@lru_cache(maxsize=None)
def grid(rate: float) -> cpm.Grid:
    return cpm.Grid(f"hand{rate:g}", 8, rate)


SYNC = np.array(cpm.costas(8))  # Welch, order 6
N_DATA = E // 3
HALF = N_DATA // 2
SYNC_ROWS = np.r_[0:len(SYNC), len(SYNC) + HALF:2 * len(SYNC) + HALF]
DATA_ROWS = np.setdiff1d(np.arange(2 * len(SYNC) + N_DATA), SYNC_ROWS)
N_SYM = 2 * len(SYNC) + N_DATA


def info_bits(payload: np.ndarray) -> np.ndarray:
    return np.concatenate([payload, crc16(payload, TAG << 8)])


def modulate(g: cpm.Grid, payload: np.ndarray) -> np.ndarray:
    coded = code().encode(info_bits(payload))[0]
    sym = np.empty(N_SYM, int)
    sym[SYNC_ROWS] = np.tile(SYNC, 2)
    sym[DATA_ROWS] = cpm.to_tones(g, coded[interleaver()])
    return cpm.tones(g, sym)


def search(g: cpm.Grid, y: np.ndarray, nominal: int, span_s: float = SEARCH_S) -> list:
    """Top candidates (start, cfo) by the sync blocks' tone shares."""
    T = g.T
    cands = []
    for cfo in np.arange(-SEARCH_HZ, SEARCH_HZ + 1e-9, g.rate / 4):
        for start in range(nominal - int(span_s * FS), nominal + int(span_s * FS) + 1, T // 8):
            s = 0.0
            for r0 in (0, len(SYNC) + HALF):
                E_ = cpm._shares(cpm._energies(g, y, start + r0 * T, len(SYNC), cfo))
                s += E_[np.arange(len(SYNC)), SYNC].sum()
            cands.append((s, start, cfo))
    cands.sort(reverse=True)
    out = []
    for s, start, cfo in cands:  # distinct peaks: half a symbol or a quarter tone apart
        if all(abs(start - a) > T // 2 or abs(cfo - b) > g.rate / 4 + 1e-9 for a, b in out):
            out.append((start, cfo))
        if len(out) == TOP:
            break
    return out


def receive(g: cpm.Grid, y: np.ndarray, nominal: int, dec: polar.SCLDecoder,
            span_s: float = SEARCH_S) -> np.ndarray | None:
    """The payload of the first candidate and list path whose CRC passes, or None."""
    cands = search(g, y, nominal, span_s)
    llr = np.empty((len(cands), E))
    for i, (start, cfo) in enumerate(cands):
        En = cpm._energies(g, y, start, N_SYM, cfo)[DATA_ROWS]
        llr[i, interleaver()] = cpm.llrs(g, En)
    bits, _ = dec.decode(llr)
    for paths in bits:
        for u in paths:
            if np.array_equal(crc16(u[:PAYLOAD], TAG << 8), u[PAYLOAD:]):
                return u[:PAYLOAD]
    return None


def _arrival(g: cpm.Grid, rng, chan: str, seed: int, shift_hz: float = 0.0, dt_s: float = DT_S):
    """One Hand as received, noise-free: (payload, audio). Its own timing and
    CFO draw, its own fading, `shift_hz` off the wanted channel."""
    payload = rng.integers(0, 2, PAYLOAD).astype(np.uint8)
    dt, df = rng.uniform(-dt_s, dt_s), rng.uniform(-DF_HZ, DF_HZ)
    lead = int(LEAD_S * FS)
    pad = np.zeros(lead + int(round(dt * FS)))
    tx = np.concatenate([pad, modulate(g, payload), np.zeros(2 * lead - len(pad))])
    y = hfchannel.freq_shift(tx, shift_hz + df)
    if chan != "awgn":
        y = hfchannel.fading(y, chan, seed=seed)
    return payload, y


def trial(args) -> list:
    """Decode outcomes per SNR: 1 right, 0 none, -1 a wrong payload. With
    `nb_db`, a Hand `pitch` Hz either side, each nb_db over the wanted one."""
    rate, chan, seed, snrs, nb_db, pitch, dt_s, span_s = args
    g = grid(rate)
    rng = np.random.default_rng(seed)
    payload, y0 = _arrival(g, rng, chan, seed, dt_s=dt_s)
    s_power = 1.0  # cpm.tones is unit RMS and the taps unit power: SNR against the TX, as apply_channel
    if nb_db is not None:
        for k, side in enumerate((-1, 1)):
            y0 = y0 + 10 ** (nb_db / 20) * _arrival(g, rng, chan, seed + 1000003 * (k + 1), side * pitch, dt_s)[1]
    lead = int(LEAD_S * FS)
    dec = polar.SCLDecoder(code(), POLAR_LIST)
    out = []
    for snr in snrs:
        got = receive(g, hfchannel.awgn(y0, snr, seed=seed + 1, s_power=s_power), lead, dec, span_s)
        out.append(0 if got is None else 1 if np.array_equal(got, payload) else -1)
    return out


def point(snrs, ok, frac: float = 0.9) -> float | None:
    """SNR where the decode rate first reaches `frac` (linear between points)."""
    for i in range(1, len(snrs)):
        if ok[i - 1] < frac <= ok[i]:
            return snrs[i - 1] + (frac - ok[i - 1]) / (ok[i] - ok[i - 1]) * (snrs[i] - snrs[i - 1])
    return snrs[0] if ok and ok[0] >= frac else None


def demo():
    g = grid(12.5)
    payload = np.random.default_rng(1).integers(0, 2, PAYLOAD).astype(np.uint8)
    x = modulate(g, payload)
    lead = int(LEAD_S * FS)
    y = np.concatenate([np.zeros(lead + 400), x, np.zeros(lead - 400)])
    y = hfchannel.freq_shift(y, 2.0)
    got = receive(g, y, lead, polar.SCLDecoder(code(), POLAR_LIST))
    assert got is not None and np.array_equal(got, payload), "noiseless Hand did not decode"
    assert len(x) == N_SYM * g.T
    print("demo ok:", N_SYM, "symbols,", len(x) / FS, "s at 12.5 Hz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--snr", type=float, nargs=3, default=(-26.0, -4.0, 1.0), metavar=("LO", "HI", "STEP"))
    ap.add_argument("--spacings", type=float, nargs="+", default=SPACINGS)
    ap.add_argument("--channels", nargs="+", default=CHANNELS)
    ap.add_argument("--neighbours", type=float, nargs="+", default=None, metavar="DB",
                    help="a Hand each side, this many dB over the wanted one (one run per value)")
    ap.add_argument("--pitch", type=float, default=None, help="channel pitch, Hz (default: 9.6 x spacing, 60 Hz at 6.25)")
    ap.add_argument("--dt", type=float, default=DT_S, help="Hands land within +-this of nominal, s")
    ap.add_argument("--search", type=float, default=SEARCH_S, help="net control searches +-this, s")
    ap.add_argument("--out", default="runs/hand_study/link.json")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        return demo()
    snrs = list(np.arange(a.snr[0], a.snr[1] + 1e-9, a.snr[2]))
    nbs = a.neighbours or [None]
    pitch = lambda r: a.pitch or 9.6 * r  # noqa: E731  (the response's 60 / 120 Hz)
    jobs = [(r, c, s, snrs, nb, pitch(r), a.dt, a.search) for r in a.spacings for c in a.channels for nb in nbs
            for s in range(a.trials)]
    with Pool(a.workers) as pool:
        res = pool.map(trial, jobs, chunksize=4)
    table = {}
    for (r, c, _, _, nb, *_), out in zip(jobs, res):
        table.setdefault(f"{r:g}/{c}" + ("" if nb is None else f"/{nb:+g}"), []).append(out)
    summary = {}
    print(f"{'case':>16} {'10% pt':>7} {'1% pt':>6} {'false':>5}")
    for key, rows in table.items():
        m = np.array(rows)
        ok = list((m == 1).mean(axis=0))
        p10, p1 = point(snrs, ok, 0.9), point(snrs, ok, 0.99)
        summary[key] = dict(ok=ok, false=int((m == -1).sum()), p10=p10, p1=p1)
        f = lambda v: f"{v:7.1f}" if v is not None else "      -"  # noqa: E731
        print(f"{key:>16} {f(p10)} {f(p1)[1:]} {summary[key]['false']:5d}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(snrs=snrs, trials=a.trials, dt=a.dt, search=a.search, k=K, e=E, pitch={f"{r:g}": pitch(r) for r in a.spacings},
                                          summary=summary), indent=1))


if __name__ == "__main__":
    main()
