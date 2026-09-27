"""Polar SCL list size (codes.POLAR_LIST, 8) against 16 and 32, as
aicodix and modem73 use: the smallest ARQ data burst (ladder_study's
trial) around each polar mode's 10% point, every list size decoding the
same received audio. Polar codewords: the reply modes' and polar-k*'s,
and the CPM modes' control codeword (k=176, n=360). Then CPU per
codeword, and false accepts (a CRC16 match on noise: about L / 65536).

    uv run --no-sync python scripts/polar_list_study.py --out runs/polar_list.csv
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import codes, hfchannel
from data2g.arq import phy as PHY
from data2g.arq.modes import MODES, ctl_spec
from data2g.config import FS
from data2g.tnc import receive_any

sys.path.insert(0, str(Path(__file__).parent))
import outcome_data as O  # noqa: E402

NAMES = ("ack-1f", "ack-4f", "n4-ack-2f", "n4-ack-8f", "n10-ack-4f", "polar-k96-f4", "polar-k96-f8",
         "polar-k192-f8", "fsk16r25-r1/3", "fsk32r62-r1/2")
CHANNELS = ("awgn", "mpg", "mpp", "mpd")
LISTS = (8, 16, 32)
OFFSETS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0)  # dB from the 10% point

_decoders: dict = {}
_L = [codes.POLAR_LIST]
_ldpc = codes._decoder


def _decoder(spec, device):
    if spec.code != "polar":
        return _ldpc(spec, device)
    key = (spec, device, _L[0])
    if key not in _decoders:
        from data2g import polar

        _decoders[key] = polar.SCLDecoder(codes.polar_code(spec), _L[0], device=device)
    return _decoders[key]


codes._decoder = _decoder


def trial(args):
    """-> None (no burst received: sync, the same at every L), or per L the
    polar slots decoded right and whether every slot was."""
    name, chan, snr, seed = args
    rng = np.random.default_rng(seed)
    b = O.burst(name, 2, rng)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    x = np.concatenate([np.zeros(lead), PHY.tx_audio(b), np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=float(rng.uniform(-50, 50)), ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    try:
        r = receive_any(y, lead=FS)
    except Exception:  # noqa: BLE001 (a header read past the audio: a miss)
        return None
    if r is None or r["spec"].name != name or r["n_cw"] != 2:
        return None
    s = MODES[name]
    polar_slots = [i for i, sp in enumerate((ctl_spec(s), s)) if sp.code == "polar"]
    out = {}
    for L in LISTS:
        _L[0] = L
        rx = PHY.ModemRx(r, {})
        good = [rx.decode(i, sl.mask_id, 0, None) == sl.payload for i, sl in enumerate(b.slots)]
        out[L] = (sum(good[i] for i in polar_slots), len(polar_slots), all(good))
    return out


def noise_accepts(args):
    """CRC matches over `n` codewords of noise LLRs at list size L."""
    spec, L, n, seed = args
    _L[0] = L
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n // 64):
        soft = rng.normal(0, 2.0, (64, spec.coded_bits)).astype(np.float32)
        hits += int(codes.decode_llrs(spec, soft, crc_mask=0, index=np.arange(64))[1].sum())
    return hits


def cpu_per_cw(spec, L, batch: int) -> float:
    """CPU seconds per codeword, decoding `batch` at once (ModemRx: 1)."""
    _L[0] = L
    soft = np.random.default_rng(0).normal(2.0, 2.0, (batch, spec.coded_bits)).astype(np.float32)
    codes.decode_llrs(spec, soft)
    t0 = time.process_time()
    for _ in range(3):
        codes.decode_llrs(spec, soft)
    return (time.process_time() - t0) / 3 / batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ladder", default="runs/ladder_10pct.csv")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--noise", type=int, default=64000, help="noise codewords per (code, L)")
    ap.add_argument("--names", nargs="+", default=list(NAMES))
    a = ap.parse_args()
    p10 = {r["name"]: r for r in csv.DictReader(open(a.ladder))}
    rows = []
    with Pool(a.jobs) as pool:
        for name in a.names:
            for chan in CHANNELS:
                for off in OFFSETS:
                    snr = float(p10[name][chan]) + off
                    res = pool.map(trial, [(name, chan, snr, 7919 * k + int((snr + 100) * 1000))
                                           for k in range(a.trials)])
                    got = [r for r in res if r is not None]
                    for L in LISTS:
                        ok = sum(r[L][0] for r in got)
                        n = sum(r[L][1] for r in got)
                        rows.append(dict(name=name, channel=chan, snr=round(snr, 3), L=L, trials=a.trials,
                                         received=len(got), polar_cw=n, polar_fail=n - ok,
                                         burst_ok=sum(r[L][2] for r in got)))
                    print(name, chan, f"{snr:.2f}", len(got), [(L, rows[-3 + j]["polar_fail"], rows[-3 + j]["burst_ok"])
                                                               for j, L in enumerate(LISTS)], flush=True)
                with open(a.out, "w", newline="") as f:
                    w = csv.DictWriter(f, list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
        specs = {(s.k, s.coded_bits): s for s in (sp for m in NAMES for sp in (ctl_spec(MODES[m]), MODES[m]))
                 if s.code == "polar"}
        print("code k/n | L | CPU ms per codeword: alone, in a batch of 64 | noise CRC accepts per 1e4", flush=True)
        for (k, n), spec in sorted(specs.items()):
            for L in LISTS:
                hits = sum(pool.map(noise_accepts, [(spec, L, a.noise // a.jobs, 97 * j + L) for j in range(a.jobs)]))
                print(f"{k}/{n} | {L} | {1e3 * cpu_per_cw(spec, L, 1):.1f} {1e3 * cpu_per_cw(spec, L, 64):.2f} | "
                      f"{1e4 * hits / a.noise:.2f}", flush=True)


if __name__ == "__main__":
    main()
