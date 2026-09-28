"""CRC16 false accepts, measured (study only; no format change).

Every CRC16 code decodes today with the CRC as its only check: LDPC paths
(codes.decode_many, decode_buffer) drop the decoder's syndrome flag, and
polar's list picks the first of L paths whose CRC checks. Real-modem
bursts of plain codewords (mask 0) near and below each mode's 10% point;
per decoded codeword:

- right, or wrong: info-bit error weight; LDPC: syndrome satisfied or not
  (a wrong codeword the decoder converged to); polar: CRC passed on a
  wrong path (a true-mask false accept, observed directly).
- CRC16's verdict on a wrong decode is exact: the scrambler cancels in the
  error pattern, and CRC16 passes iff the error's payload part checks to
  its CRC part. CRC32's verdict on the same payload bits, alike.
- Flip statistic (a receiver-only check): the decoded codeword
  re-encoded, against the channel's hard decisions; D = sum |LLR| where
  they disagree / sum |LLR|.

    uv run --no-sync python scripts/crc_study.py --out runs/crc_study.npz
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import binascii
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import codes, hfchannel, modem
from data2g.arq import phy as PHY
from data2g.arq import policy as G
from data2g.arq.modes import MODES
from data2g.config import FS
from data2g.tnc import receive_any

NAMES = ("qpsk-r1/5", "w48-qpsk-r1/5", "n10-qpsk-r1/5", "n10-qpsk-r1/2", "n4-qpsk-r1/5", "ack-1f", "ack-4f",
         "n10-ack-4f", "polar-k96-f4")
CHANNELS = ("awgn", "mpp", "mpg")
OFFSETS = (-3.0, -2.0, -1.0, 0.0)  # dB from the smallest burst's 10% point


def crc_passes(err: np.ndarray, pb: int, n_crc: int) -> bool:
    """Whether a CRC of n_crc bits misses the info-bit error pattern `err`
    over pb payload bytes plus its CRC field: linear parts only (the CRC's
    init and the mask cancel between sent and decoded)."""
    e = np.packbits(err[: 8 * (pb + n_crc // 8)]).tobytes()
    p, r = e[:pb], int.from_bytes(e[pb:], "big")
    zero = bytes(pb)
    if n_crc == 16:
        return (binascii.crc_hqx(p, 0) ^ binascii.crc_hqx(zero, 0)) == r
    return (binascii.crc32(p) ^ binascii.crc32(zero)) == r


def trial(args):
    name, chan, snr, seed = args
    spec = MODES[name]
    rng = np.random.default_rng(seed)
    n = min(64, max(8, G.slots_for(spec, 12.0)))
    pb = codes.payload_bytes(spec)
    pays = [rng.bytes(pb) for _ in range(n)]
    lead = int(rng.uniform(0.3, 1.0) * FS)
    x = np.concatenate([np.zeros(lead), modem.modulate(pays, spec), np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=float(rng.uniform(-50, 50)), ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    try:
        r = receive_any(y, lead=FS)
    except Exception:  # noqa: BLE001
        return None
    if r is None or r["spec"].name != name or r["n_cw"] != n:
        return None
    soft = np.asarray(PHY.soft_bits(r))
    bits, ok = codes.decode_llrs(spec, soft, crc_mask=0, index=np.arange(n))
    truth = np.stack([codes.info_bits(spec, p, 0, i) for i, p in enumerate(pays)])
    coded = codes.encode_info(spec, bits.astype(np.uint8))  # the decoded codewords, mapping order
    rows = []
    n_crc = codes.crc_bits(spec)
    for i in range(n):
        err = bits[i].astype(np.uint8) ^ truth[i]
        w = int(err.sum())
        dis = (coded[i] == 1) != (soft[i] < 0)
        d = float(np.abs(soft[i])[dis].sum() / max(np.abs(soft[i]).sum(), 1e-12))
        rows.append((w, bool(ok[i]), d, w > 0 and crc_passes(err, pb, n_crc), w > 0 and crc_passes(err, pb - 2, 32)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ladder", default="runs/ladder_10pct.csv")
    ap.add_argument("--bursts", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--names", nargs="+", default=list(NAMES))
    a = ap.parse_args()
    p10 = {r["name"]: r for r in csv.DictReader(open(a.ladder))}
    save = {}
    print("mode chan snr | decodes, right, wrong: syndrome-ok/CRC16-passes/CRC32-passes | wrong weight median, min "
          "| D p99.9 right, D median wrong", flush=True)
    with Pool(a.jobs) as pool:
        for name in a.names:
            for chan in CHANNELS:
                for off in OFFSETS:
                    snr = float(p10[name][chan]) + off
                    res = [r for t in pool.map(trial, [(name, chan, snr, 7919 * k + 13) for k in range(a.bursts)])
                           if t is not None for r in t]
                    if not res:
                        continue
                    w, ok, d, c16, c32 = (np.array(v) for v in zip(*res))
                    wrong = w > 0
                    save[f"{name}|{chan}|{snr:.2f}"] = np.stack([w, ok, d, c16, c32]).astype(float)
                    dr, dw = d[~wrong], d[wrong]
                    print(f"{name} {chan} {snr:.2f} | {len(w)}, {int((~wrong).sum())}, {int(wrong.sum())}: "
                          f"{int((wrong & ok).sum())}/{int(c16.sum())}/{int(c32.sum())} | "
                          f"{int(np.median(w[wrong])) if wrong.any() else '-'}, {int(w[wrong].min()) if wrong.any() else '-'}"
                          f" | {np.quantile(dr, 0.999) if len(dr) else float('nan'):.3f}, "
                          f"{np.median(dw) if len(dw) else float('nan'):.3f}", flush=True)
                    np.savez(a.out, **save)


if __name__ == "__main__":
    main()
