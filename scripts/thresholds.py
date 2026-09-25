"""Threshold SNR of a submode candidate, per channel, on the GPU path
(channel_torch: real clipper, waveform-domain fading, numpy estimator;
exact LLRs; LDPC min-sum). Acquisition is assumed; the full numpy modem
reruns survivors later (plan step 5).

Criterion, every channel (decided 2026-09-23; AWGN was BER <= 1e-6
until then, which only cost time): PER <= 1e-2 over >= MIN_BURSTS
bursts, failing early once >= 200 bursts and >= 100 codeword errors put
it clearly above. CRC-undetected errors are counted separately; the
ARQ layer's goodput scoring is the eventual judge.

    uv run python scripts/thresholds.py --code ldpc --constellation gray-qam16 \\
        --frames 4 --k 960 --channels awgn mpp mpd
"""

import os

# Two BLAS/OpenMP threads, torch capped at 4, so a long GPU run leaves the CPU
# to the machine's owner (numpy and torch default to a thread per core).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import time

import numpy as np
import torch

torch.set_num_threads(4)

from data2g import codes, constellation
from data2g.channel_torch import CHANNELS, BurstChannel, llr
from data2g.config import DATA_SYMS_PER_FRAME, SubmodeSpec

MIN_BURSTS = 2000


def _payloads(spec, n, rng):
    nb = codes.payload_bytes(spec)
    raw = rng.integers(0, 256, (n, nb), dtype=np.uint8)
    return np.stack([codes.info_bits(spec, r.tobytes()) for r in raw])


class Sim:
    def __init__(self, spec: SubmodeSpec, device: str, burst_cws: int, batch: int = 64, spread: bool = True,
                 clip_setting: tuple | None = None, clip_consts: tuple | None = None):
        """burst_cws: codewords per simulated burst. It matters beyond
        speed: the estimator sees only that burst's pilots, so a one-
        codeword ACK must be simulated as the one-frame burst it is."""
        self.spec, self.device, self.batch, self.spread = spec, device, batch, spread
        self.n_cw = burst_cws
        self.n_f = self.n_cw * spec.frames_per_cw
        self.ch = BurstChannel(spec, self.n_f, device=device, clip_setting=clip_setting, clip_consts=clip_consts)
        pts = constellation.load(spec.constellation)
        self.m = constellation.bits_per_symbol(pts)
        self.points = torch.tensor(pts, dtype=torch.complex64, device=device)
        self.w = torch.tensor(1 << np.arange(self.m - 1, -1, -1), device=device)

    def run(self, chan: str, snr: float, rng: np.random.Generator, g: torch.Generator):
        """One batch of bursts -> (bits, bit errs, codewords, cw errs, undetected)."""
        s, b = self.spec, self.batch
        info = _payloads(s, b * self.n_cw, rng)
        coded = codes.encode_info(s, info).reshape(b, self.n_cw, -1)
        if self.spread:
            coded = codes.spread(coded, self.m)
        cb = torch.tensor(coded.reshape(b, self.n_f, DATA_SYMS_PER_FRAME, self.ch.nc, self.m), device=self.device)
        c = CHANNELS[chan]
        with torch.no_grad():
            y, h, var = self.ch.receive(self.ch.channel(self.ch.transmit(self.points[(cb * self.w).sum(-1)]), c, snr, g), c)
            l = llr(y, h, var, self.points).reshape(b, -1)
            l = (codes.despread(l, self.n_cw, self.m) if self.spread else l).reshape(b * self.n_cw, -1)
        est, _ = codes.decode_llrs(s, l, device=self.device)
        wrong = est != info
        cw_err = wrong.any(axis=1)
        ok = codes.crc_ok(s, est)
        return info.size, int(wrong.sum()), len(info), int((~ok | cw_err).sum()), int((ok & cw_err).sum())


def point(sim: Sim, chan: str, snr: float, seed: int = 0, verbose=False) -> dict:
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=sim.device).manual_seed(seed)
    tot = dict(bits=0, bit_err=0, cws=0, cw_err=0, undetected=0, bursts=0)
    t0 = time.time()
    while True:
        r = sim.run(chan, snr, rng, g)
        for k, v in zip(("bits", "bit_err", "cws", "cw_err", "undetected"), r):
            tot[k] += v
        tot["bursts"] += sim.batch
        per = tot["cw_err"] / tot["cws"]
        # Codewords in one burst fade together, so evidence is counted
        # in bursts: >= 200 before an early fail, MIN_BURSTS to pass.
        if tot["bursts"] >= 200 and tot["cw_err"] >= 100 and per > 3e-2:  # clearly failing
            tot["pass"] = False
            break
        if tot["bursts"] >= MIN_BURSTS:
            tot["pass"] = per <= 1e-2
            break
    tot["ber"] = tot["bit_err"] / tot["bits"]
    tot["per"] = tot["cw_err"] / tot["cws"]
    tot["secs"] = time.time() - t0
    if verbose:
        print(f"    {chan} {snr:6.2f} dB: BER {tot['ber']:.2e} PER {tot['per']:.2e} "
              f"undet {tot['undetected']} ({tot['bits']:.1e} bits, {tot['secs']:.0f}s) "
              f"{'PASS' if tot['pass'] else 'fail'}", flush=True)
    return tot


def threshold(sim: Sim, chan: str, start: float, verbose=True, step: float = 2.0) -> float:
    """Lowest passing SNR on a 0.25 dB grid: walk from `start` in steps that
    double while the walk keeps going the same way, until bracketed, then
    bisect. Assumes pass/fail is monotone in SNR. A good prior wants a small
    first step (0.5: a close start then costs ~2 points); the doubling keeps
    a bad one cheap."""
    lo, hi = None, None
    snr = start
    while lo is None or hi is None:
        ok = point(sim, chan, snr, verbose=verbose)["pass"]
        if ok:
            hi = snr
            if lo is None:
                snr -= step
        else:
            lo = snr
            if hi is None:
                snr += step
        step *= 2
        if snr > 45:
            return float("inf")
    while hi - lo > 0.25:
        mid = round((lo + hi) / 2 * 4) / 4
        if mid in (lo, hi):
            break
        if point(sim, chan, mid, verbose=verbose)["pass"]:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="ldpc")
    ap.add_argument("--constellation", required=True)
    ap.add_argument("--frames", type=int, required=True, help="frames per codeword")
    ap.add_argument("--k", type=int, required=True, help="info bits per codeword incl. CRC")
    ap.add_argument("--channels", nargs="+", default=["awgn", "mpg", "mpp", "mpd"])
    ap.add_argument("--band", default="w")
    ap.add_argument("--protograph", default="", help='LDPC mask, "bg2:runs/proto/bg2_r0.5.npy"')
    ap.add_argument("--start", type=float, default=10.0)
    ap.add_argument("--no-spread", action="store_true", help="codewords in contiguous frames")
    ap.add_argument("--burst-cws", type=int, help="codewords per burst (default: 1 for polar, else fill 8 frames)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    spec = SubmodeSpec(15, "candidate", a.code, a.constellation, a.frames, k=a.k,
                       protograph=a.protograph, band=a.band)
    burst = a.burst_cws or (1 if a.code == "polar" else max(1, 8 // a.frames))
    sim = Sim(spec, a.device, burst, batch=512 if burst * a.frames <= 2 else 64, spread=not a.no_spread)
    rate = a.k / spec.coded_bits
    print(f"{a.constellation} {a.code} k={a.k} n={spec.coded_bits} rate {rate:.3f}, "
          f"{a.k / (a.frames * spec.cu_per_frame):.3f} info bits/cu", flush=True)
    for chan in a.channels:
        t = threshold(sim, chan, a.start)
        print(f"  threshold {chan}: {t} dB", flush=True)


if __name__ == "__main__":
    main()
