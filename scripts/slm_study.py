"""Peak reduction before any waveform change: SLM (selected mapping, as
aicodix/modem's encoder) and ACE (active constellation extension,
Krongold & Jones 2003; the README TODO), alone and together.

- SLM: each OFDM data symbol (or each frame's five) goes under whichever
  of C carrier sign patterns gives the lowest envelope peak, pattern 0
  being "unchanged". The receiver would need the index (log2 C bits per
  symbol or per frame); here it arrives free.
- ACE: after each clip-and-filter pass, the data cells are projected back
  into their allowed regions (square QAM: an inner level returns to its
  place, scaled by the pass's clip gain; an outer one may only move
  outward), then one last clip and filter holds the peak. No receiver or
  format change. The TX bandpass (201 taps) smears each symbol past its
  cyclic prefix, and the projection chases that too: with nothing
  clipped, SDR falls from 37 to 30 dB (far above the 12-20 dB in play). Square QAM only here (the learned 64/256 sets would need
  their Voronoi cells).

Per band, headroom and method, through channel_torch's clipper and clean
receiver (as clip_constants.py):
- SDR: data error against the sent symbols, after the fitted gain.
- effective SDR: the same, less the outward part of the error on outer
  levels (that part only moves a point further from its decision
  boundaries). ACE moves outer points on purpose, so plain SDR undersells
  it; every method is scored both ways.
- PAPR at the 99.99th percentile, and the mean per-burst peak (dB).
Then, per headroom a mode uses today: each method's headroom for the same
effective SDR, and its PEP gain (today's peak - the method's, dB).

    uv run --no-sync python scripts/slm_study.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse

import numpy as np
import torch

from data2g import constellation
from data2g.channel_torch import CHANNELS, BurstChannel, _analytic
from data2g.config import BANDS, LEADIN_SAMPLES, LEADOUT_SAMPLES, M, NCP, NSYM, SUBMODES, SYMS_PER_FRAME, SubmodeSpec
from data2g.waveform import ofdm

N_F = 8
HEADROOMS = tuple(float(h) for h in np.arange(0, 6.01, 0.5))
# (label, SLM candidates, SLM per, ACE projection passes, overshoot of ACE's closing clip)
METHODS = (("today", 1, "symbol", 0, 0), ("slm8", 8, "symbol", 0, 0), ("slm32", 32, "symbol", 0, 0),
           ("slm128", 128, "symbol", 0, 0), ("slm32/frame", 32, "frame", 0, 0),
           ("slm128/frame", 128, "frame", 0, 0), ("ace3", 1, "symbol", 3, 2.0), ("ace3/k1", 1, "symbol", 3, 1.0),
           ("ace6", 1, "symbol", 6, 2.0), ("slm32+ace3", 32, "symbol", 3, 2.0),
           ("slm32+ace3/k1", 32, "symbol", 3, 1.0), ("slm32/frame+ace3", 32, "frame", 3, 2.0))


def select(x: np.ndarray, band: str, c: int, per: str, rng) -> np.ndarray:
    """(B, n_f, 5, nc) data symbols -> each symbol (per="symbol") or
    frame (per="frame") under the lowest-peak of c sign patterns."""
    if c == 1:
        return x
    mod = ofdm.band(band).mod  # (NSYM, nc): |x @ mod.T| is a symbol's envelope
    pats = rng.choice([-1.0, 1.0], size=(c, x.shape[-1]))
    pats[0] = 1.0
    out = np.empty_like(x)
    for i in range(len(x)):
        cand = x[i][:, :, None, :] * pats  # (n_f, 5, c, nc)
        peak = np.abs(cand @ mod.T).max(axis=-1)  # (n_f, 5, c)
        if per == "symbol":
            k = peak.argmin(axis=-1)
            out[i] = np.take_along_axis(cand, k[..., None, None], axis=2)[:, :, 0]
        else:
            k = peak.max(axis=1).argmin(axis=-1)
            out[i] = cand[np.arange(len(k)), :, k]
    return out


class AceChannel(BurstChannel):
    """BurstChannel whose clipper projects the data cells (ACE) after
    each pass. `umax`: the constellation's outer level per axis."""

    def __init__(self, *a, passes: int = 0, final: float = 2.0, umax: float = 1.0, **kw):
        super().__init__(*a, **kw)
        self.passes, self.final, self.umax = passes, final, umax
        frames = [f for f in range(self.n_air) if f != self.kc]
        j = torch.tensor([f * SYMS_PER_FRAME + s for f in frames for s in range(1, SYMS_PER_FRAME)])
        self.win = self.frames0 + j[:, None] * NSYM + NCP + torch.arange(M)[None]  # useful windows
        self.full = self.frames0 + j[:, None] * NSYM + torch.arange(NSYM)[None]  # whole symbols
        self.dem = self.mod[NCP:].conj()  # (M, nc)

    def transmit(self, data: torch.Tensor) -> torch.Tensor:
        self.sent = data.reshape(data.shape[0], -1, self.nc)
        return super().transmit(data)

    def _pass(self, x, thresh, k):
        z = _analytic(x)
        scale = torch.clamp(thresh / z.abs().clamp_min(1e-12), max=1.0)
        x = (z * (scale**k if k != 1.0 else scale)).real
        return torch.nn.functional.conv1d(x[:, None], self.taps.flip(0)[None, None], padding=100)[:, 0]

    def _project(self, x):
        X = self.sent
        got = (2.0 / M) * torch.einsum("bjm,mc->bjc", x[:, self.win].to(self.cdtype), self.dem)
        # per carrier: the bandpass shapes the band edges, pilots included
        g = (torch.sum(X.conj() * got, dim=1).real / torch.sum(X.abs() ** 2, dim=1))[:, None, :]

        def axis(u, v):
            t = g * u
            outer = u.abs() >= self.umax - 1e-9
            return torch.where(outer, t + torch.sign(u) * torch.clamp(torch.sign(u) * (v - t), min=0), t)

        want = torch.complex(axis(X.real, got.real), axis(X.imag, got.imag))
        x = x.clone()
        x[:, self.full] += torch.einsum("bjc,nc->bjn", want - got, self.mod).real
        return x

    def tx_condition(self, x: torch.Tensor) -> torch.Tensor:
        if not self.passes:
            return super().tx_condition(x)
        act = slice(LEADIN_SAMPLES, x.shape[1] - LEADOUT_SAMPLES)
        thresh = torch.sqrt(2 * x[:, act].pow(2).mean(dim=1, keepdim=True)) * 10 ** (self.headroom / 20)
        ks = list(self.overshoot) + [self.overshoot[-1]] * max(0, self.passes - len(self.overshoot))
        for k in ks[:self.passes]:
            x = self._project(self._pass(x, thresh, k))
        x = self._pass(x, thresh, self.final)
        return x / x[:, act].pow(2).mean(dim=1, keepdim=True).sqrt()


def measure(band: str, x: np.ndarray, headroom: float, passes: int, final: float, umax: float) -> tuple:
    """-> (SDR, effective SDR, PAPR 99.99%, mean per-burst peak), dB."""
    spec = SubmodeSpec(0, "pk", "ldpc", "gray-qam16", 1, band=band)
    ch = AceChannel(spec, N_F, dtype=torch.float64, clip_setting=(headroom, BANDS[band].clip_overshoot),
                    clip_consts=({}, 1.0, 0.0), passes=passes, final=final, umax=umax)
    tx = ch.transmit(torch.tensor(x))
    raw, h, _ = ch.receive(tx, CHANNELS["awgn"])
    raw, h = raw.numpy(), h.numpy()
    g = np.vdot(h * x, raw) / np.vdot(h * x, h * x)
    e = raw / (g * h) - x

    def inward(u, eu):
        outer = np.abs(u) >= umax - 1e-9
        return np.where(outer, np.minimum(0, np.sign(u) * eu), eu)

    e_eff = inward(x.real, e.real) + 1j * inward(x.imag, e.imag)
    p = np.mean(np.abs(x) ** 2)
    env2 = _analytic(tx[:, LEADIN_SAMPLES:tx.shape[1] - LEADOUT_SAMPLES]).abs().pow(2).numpy()
    return (10 * np.log10(p / np.mean(np.abs(e) ** 2)), 10 * np.log10(p / np.mean(np.abs(e_eff) ** 2)),
            10 * np.log10(np.quantile(env2, 0.9999) / env2.mean()),
            10 * np.log10(env2.max(axis=1).mean() / env2.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", type=int, default=64)
    ap.add_argument("--bands", nargs="+", default=["w48", "n10", "w", "n4"])
    ap.add_argument("--const", default="gray-qam16")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    pts = constellation.load(a.const)
    m, umax = constellation.bits_per_symbol(pts), float(np.max(np.abs(pts.real)))
    for band in a.bands:
        nc = ofdm.band(band).nc
        x = constellation.modulate(np.random.default_rng(1).integers(0, 2, a.bursts * N_F * 5 * nc * m),
                                   pts).reshape(a.bursts, N_F, 5, nc)
        used = sorted({float(s.headroom) for s in SUBMODES.values() if s.band == band})
        res, sel = {}, {}
        print(f"{band} ({nc} carriers, {a.const}): headroom method | SDR, effective SDR, PAPR 99.99%, peak (dB)",
              flush=True)
        for label, c, per, passes, final in METHODS:
            if (c, per) not in sel:
                sel[(c, per)] = select(x, band, c, per, np.random.default_rng(2))
            for hr in HEADROOMS:
                res[(label, hr)] = measure(band, sel[(c, per)], hr, passes, final, umax)
                print(f"  {hr:.1f} {label} | " + " ".join(f"{v:.2f}" for v in res[(label, hr)]), flush=True)
        print(f"{band}: per headroom in use, each method's headroom for today's effective SDR, "
              f"and its PEP gain (dB)", flush=True)
        for h0 in used:
            _, sdr0, _, peak0 = res[("today", h0)]
            out = []
            for label, *_ in METHODS[1:]:
                hs = [hr for hr in HEADROOMS if res[(label, hr)][1] >= sdr0]
                out.append(f"{label} {min(hs):.1f} {peak0 - res[(label, min(hs))][3]:+.2f}" if hs else f"{label} -")
            print(f"  today {h0:g} dB (eff. SDR {sdr0:.1f}): " + "; ".join(out), flush=True)


if __name__ == "__main__":
    main()
