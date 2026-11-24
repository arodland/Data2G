"""Differentiable burst channel for training constellations.

TX is `modem.modulate_bits` in torch: the numpy burst (preamble, header,
pilots) plus the data symbols, then the same clip-and-filter as
`dsp.tx_condition`, so gradients reach the constellation points through
the clipper. The channel is waveform-domain (two-path Watterson with
ITU-R F.1487 Gaussian Doppler, AWGN in the SNR_REF_BW_HZ convention),
so ISI past the CP and clip-noise leakage are in it too; SSTVAE's
replica faded per symbol and had neither.

RX assumes acquisition succeeded (no CFO, no clock error), places the
window as `equalizer.window_shift` would for the known delays, and runs
the real numpy `equalizer.estimate` on detached pilots. The estimate
depends on the constellation only through clip distortion on the
pilots, so it carries no gradient; keeping one estimator means training
and the modem cannot drift apart.

`tests/test_channel_torch.py` pins the TX against numpy and the whole
path against `scripts/eq_floor.py`'s measurement.

Training only (torch), not on the live path and not ported to native.
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import firwin

from . import config, constellation, equalizer, hfchannel, modem
from .config import (
    DATA_SYMS_PER_FRAME,
    DEMOD_BACKOFF,
    FCENTER,
    FS,
    LEADIN_SAMPLES,
    LEADOUT_SAMPLES,
    M,
    NCP,
    NSYM,
    PREAMBLE_CP,
    SNR_REF_BW_HZ,
    SYMS_PER_FRAME,
    SubmodeSpec,
)
from .waveform import ofdm


@dataclass(frozen=True)
class Channel:
    name: str
    spread_hz: float = 0.0  # 0 = no fading
    delay_ms: float = 0.0


CHANNELS = {"awgn": Channel("awgn"),
            **{k: Channel(k, p.doppler_hz, p.delay_ms) for k, p in hfchannel.FADING_PRESETS.items()}}


def _analytic(x: torch.Tensor) -> torch.Tensor:
    """scipy.signal.hilbert along the last axis."""
    n = x.shape[-1]
    h = torch.zeros(n, device=x.device, dtype=x.dtype)
    h[0] = 1
    h[1 : (n + 1) // 2] = 2
    if n % 2 == 0:
        h[n // 2] = 1
    return torch.fft.ifft(torch.fft.fft(x) * h)


class BurstChannel:
    def __init__(self, spec: SubmodeSpec, n_frames: int, device="cpu", dtype=torch.float32,
                 clip_setting: tuple | None = None, clip_consts: tuple | None = None):
        """`clip_setting` (headroom dB, overshoot[, ACE closing passes]) and
        `clip_consts` (as a config.CLIP entry) override the submode's, for
        clipper studies."""
        self.spec, self.n_f, self.device, self.dtype = spec, n_frames, device, dtype
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.cdtype = cdtype
        self.band = b = ofdm.band(spec.band)
        cs = clip_setting or (spec.headroom, b.spec.clip_overshoot, spec.ace)
        self.headroom, self.overshoot, self.ace = cs[0], cs[1], tuple(cs[2]) if len(cs) > 2 else ()
        self.clip_consts = clip_consts or spec.clip
        if self.ace:
            win, full = modem.ace_cells(spec, n_frames)
            self.ace_win, self.ace_full = torch.tensor(win, device=device), torch.tensor(full, device=device)
            pts = constellation.load(spec.constellation)
            self.points = torch.tensor(pts, dtype=cdtype, device=device)
            self.dirs = torch.tensor(constellation.ace_dirs(spec.constellation), dtype=cdtype, device=device)
        self.nc = b.nc
        self.kc = modem.copy_frame(spec.sync_band, n_frames)  # the header copy's frame (in self.base)
        self.n_air = n_frames + (self.kc is not None)
        zeros = np.zeros((n_frames, DATA_SYMS_PER_FRAME, b.nc), dtype=np.complex128)
        self.base = torch.tensor(modem.burst_waveform(zeros, spec), dtype=dtype, device=device)
        self.mod = torch.tensor(b.mod, dtype=cdtype, device=device)  # (NSYM, nc)
        sb = config.BANDS[spec.sync_band]  # preamble and header (BandSpec.sync)
        self.frames0 = LEADIN_SAMPLES + sb.preamble_samples + modem.header_samples(spec.sync_band)
        self.taps = torch.tensor(
            firwin(config.TX_FILTER_TAPS, sb.tx_bandpass, fs=FS, pass_zero=False), dtype=dtype, device=device
        )
        n = torch.arange(len(self.base), device=device)
        self.het = torch.exp(-2j * torch.pi * ((FCENTER * n) % FS).to(dtype) / FS).to(cdtype)
        self.demod = torch.tensor(b.demod, dtype=cdtype, device=device)  # (nc, M)

    # --- TX -------------------------------------------------------------
    def transmit(self, data: torch.Tensor) -> torch.Tensor:
        """(B, n_f, 5, nc) complex data symbols -> (B, n) clipped, unit-RMS."""
        b = data.shape[0]
        syms = torch.zeros(b, self.n_air, SYMS_PER_FRAME, self.nc, dtype=self.cdtype, device=self.device)
        frames = [f for f in range(self.n_air) if f != self.kc]
        syms[:, frames, 1:] = data
        wav = torch.einsum("bfsc,nc->bfsn", syms, self.mod).real.reshape(b, -1)
        x = self.base.expand(b, -1).clone()
        x[:, self.frames0 : self.frames0 + wav.shape[1]] += wav
        self.sent = data.reshape(b, -1, self.nc) if self.ace else None
        if self.ace:
            # the point each cell was sent as, a few bursts at a time: the
            # (cells x points) table was 2 GB for 2000 one-frame 256-point bursts
            self.sent_dirs = torch.cat([self.dirs[(s[..., None] - self.points).abs().argmin(dim=-1)]
                                        for s in self.sent.split(32)])
        return self.tx_condition(x)

    def _ace(self, x: torch.Tensor) -> torch.Tensor:
        """modem.ace_projector, batched."""
        X = self.sent
        got = (2.0 / M) * torch.einsum("bjm,mc->bjc", x[:, self.ace_win].to(self.cdtype), self.mod[NCP:].conj())
        g = (torch.sum(X.conj() * got, dim=1).real / torch.sum(X.abs() ** 2, dim=1))[:, None, :]
        new = constellation.ace_project(got, g * X, self.sent_dirs)
        x = x.clone()
        x[:, self.ace_full] += torch.einsum("bjc,nc->bjn", new - got, self.mod).real
        return x

    def _filter(self, x: torch.Tensor) -> torch.Tensor:
        """np.convolve(x, taps, "same") per row, by FFT. conv1d built a
        (bursts, taps, samples) buffer: 2.6 GB for 250 one-frame bursts, and
        clip_constants' 2000 at once took ~20 GB a process."""
        n, k = x.shape[-1], len(self.taps)
        y = torch.fft.irfft(torch.fft.rfft(x, n + k - 1) * torch.fft.rfft(self.taps, n + k - 1), n + k - 1)
        return y[..., (k - 1) // 2:(k - 1) // 2 + n]

    def tx_condition(self, x: torch.Tensor) -> torch.Tensor:
        """dsp.tx_condition, batched, power over the non-silent part."""
        act = slice(LEADIN_SAMPLES, x.shape[1] - LEADOUT_SAMPLES)
        power = x[:, act].pow(2).mean(dim=1, keepdim=True)
        thresh = torch.sqrt(2 * power) * 10 ** (self.headroom / 20)
        for i, k in enumerate(list(self.overshoot) + list(self.ace)):
            z = _analytic(x)
            scale = torch.clamp(thresh / z.abs().clamp_min(1e-12), max=1.0)
            if k != 1.0:
                scale = scale**k
            x = (z * scale).real
            x = self._filter(x)
            if self.ace and i < len(self.overshoot):
                x = self._ace(x)
        return x / x[:, act].pow(2).mean(dim=1, keepdim=True).sqrt()

    # --- channel --------------------------------------------------------
    def _taps(self, b: int, n: int, spread: float, g: torch.Generator) -> torch.Tensor:
        """hfchannel._gaussian_taps, batched: (b, n) unit-power complex."""
        lowrate = max(64 * spread, 8.0)
        n_low = int(np.ceil(n * lowrate / FS)) + 2
        w = torch.randn(b, n_low, generator=g, device=self.device, dtype=self.dtype) + 1j * torch.randn(
            b, n_low, generator=g, device=self.device, dtype=self.dtype
        )
        f = torch.fft.fftfreq(n_low, 1 / lowrate, device=self.device, dtype=self.dtype)
        shape = torch.exp(-(f**2) / (4 * (spread / 2) ** 2))
        w = torch.fft.ifft(torch.fft.fft(w) * shape) / torch.sqrt(2 * shape.pow(2).mean())
        pos = torch.arange(n, device=self.device, dtype=self.dtype) * (lowrate / FS)
        i0 = pos.floor().long().clamp(max=n_low - 2)
        a = (pos - i0).to(self.dtype)
        return w[:, i0] * (1 - a) + w[:, i0 + 1] * a  # unit power in expectation

    def channel(self, x: torch.Tensor, ch: Channel, snr_db: float, g: torch.Generator) -> torch.Tensor:
        """hfchannel.apply_channel: SNR against the transmitted (= average
        received) active power, so a burst in a fade is a low-SNR burst."""
        b, n = x.shape
        env = _analytic(x).abs()
        active = env > 0.1 * x.pow(2).mean(dim=1, keepdim=True).sqrt()
        s_power = (x.pow(2) * active).sum(dim=1, keepdim=True) / active.sum(dim=1, keepdim=True)
        if ch.spread_hz:
            z = _analytic(x)
            d = int(round(ch.delay_ms * 1e-3 * FS))
            z2 = torch.nn.functional.pad(z, (d, 0))[:, :n]
            g1, g2 = self._taps(b, n, ch.spread_hz, g), self._taps(b, n, ch.spread_hz, g)
            x = ((z * g1 + z2 * g2) / np.sqrt(2)).real
        sigma2 = s_power * (FS / 2) / SNR_REF_BW_HZ / 10 ** (snr_db / 10)
        return x + torch.randn(x.shape, generator=g, device=self.device, dtype=self.dtype) * sigma2.sqrt()

    # --- RX -------------------------------------------------------------
    def receive(self, y: torch.Tensor, ch: Channel, measure: bool = False,
                points: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> raw data symbols (B, n_f, 5, nc), and (no grad) channel
        estimate and per-cu noise variance of the same shape.

        `measure` (fmtorch): as modem.receive past acquisition, each burst's
        delay support from its own pilots (the FM radios' filters delay the
        audio), the preamble's noise per carrier (FM noise is coloured), and,
        with `points` and config.SYMBOL_NOISE, the per-symbol noise scale
        (clicks). Off: the HF path, unchanged."""
        b = y.shape[0]
        z = y.to(self.cdtype) * self.het
        d = int(round(ch.delay_ms * 1e-3 * FS))
        support = [(DEMOD_BACKOFF, d + DEMOD_BACKOFF)] * b  # apparent delays before the shift
        if measure:
            with torch.no_grad():
                hp0 = (self._demod(z, [0] * b)[:, ::SYMS_PER_FRAME]
                       / torch.tensor(self.band.pilot, dtype=self.cdtype, device=self.device)).cpu().numpy()
            bb = self.band.bb
            seen = [equalizer.delay_support(h.astype(np.complex128), bb=bb) for h in hp0]
            support = [equalizer.delay_support(h.astype(np.complex128), bb=bb, n0=n) for h, n in
                       zip(hp0, self._preamble_noise(z)[0])]
            shifts = [equalizer.window_shift(s) for s in seen]
        else:
            shifts = [equalizer.window_shift(s) for s in support]
        raw = self._demod(z, shifts)
        n0s, n0ks = self._preamble_noise(z)
        pilots = raw[:, ::SYMS_PER_FRAME] / torch.tensor(self.band.pilot, dtype=self.cdtype, device=self.device)
        frames = [f for f in range(self.n_air) if f != self.kc]
        data = raw[:, :-1].reshape(b, self.n_air, SYMS_PER_FRAME, self.nc)[:, frames, 1:]

        hp = pilots.detach().cpu().numpy().astype(np.complex128)
        yd = data.detach().cpu().numpy() if points is not None else None
        hs, vs = [], []
        for i in range(b):
            sup = (support[i][0] - shifts[i], support[i][1] - shifts[i])
            est = modem.data_channel(hp[i], sup, self.spec.band, n0s[i], clip=self.clip_consts,
                                     n0_pre_k=n0ks[i] if measure else None, n_frames=self.n_f)
            h_i = est["h"][frames]
            v_i = np.broadcast_to(modem.noise_var(est["h"], est) + est["mse"], est["h"].shape)[frames]
            if measure and points is not None and config.SYMBOL_NOISE:
                v_i = v_i * modem.symbol_noise_scale(yd[i], h_i, v_i, points)
            hs.append(h_i)
            vs.append(v_i)
        h = torch.tensor(np.stack(hs), dtype=self.cdtype, device=self.device)
        var = torch.tensor(np.stack(vs), dtype=self.dtype, device=self.device)
        return data, h, var

    def _demod(self, z: torch.Tensor, shifts: list) -> torch.Tensor:
        """Every symbol of every burst, burst i's window `shifts[i]` later: (B, S, nc)."""
        start = torch.tensor([self.frames0 + NCP - DEMOD_BACKOFF + s for s in shifts], device=self.device)
        idx = (
            start[:, None, None]
            + torch.arange(self.n_air * SYMS_PER_FRAME + 1, device=self.device)[None, :, None] * NSYM
            + torch.arange(M, device=self.device)[None, None, :]
        )
        win = torch.gather(z, 1, idx.reshape(len(shifts), -1)).reshape(idx.shape)  # (B, S, M)
        return (2.0 / M) * torch.einsum("bsm,cm->bsc", win, self.demod)

    def _preamble_noise(self, z: torch.Tensor) -> tuple[list, list]:
        """equalizer.preamble_noise and preamble_noise_k of each burst's repeats."""
        pidx = (
            LEADIN_SAMPLES + PREAMBLE_CP - NCP // 2
            + torch.arange(config.BANDS[self.spec.sync_band].preamble_repeats, device=self.device)[:, None] * M
            + torch.arange(M, device=self.device)[None, :]
        )
        reps = (2.0 / M) * torch.einsum("bsm,cm->bsc", z[:, pidx], self.demod)
        reps = (reps / torch.tensor(self.band.pilot, dtype=self.cdtype, device=self.device)).detach().cpu().numpy()
        return [equalizer.preamble_noise(r) for r in reps], [equalizer.preamble_noise_k(r) for r in reps]


def llr(y: torch.Tensor, h: torch.Tensor, var: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """constellation.llr in torch: (..., ) -> (..., m)."""
    m = int(np.log2(points.shape[0]))
    d = -(y[..., None] - h[..., None] * points).abs().pow(2) / var[..., None]  # (..., 2^m)
    lb = torch.tensor(
        (np.arange(2**m)[None, :] >> np.arange(m - 1, -1, -1)[:, None]) & 1, dtype=torch.bool, device=y.device
    )  # (m, 2^m)
    neg = torch.tensor(-torch.inf, device=y.device, dtype=d.dtype)
    dd = d[..., None, :]
    l0 = torch.logsumexp(torch.where(~lb, dd, neg), dim=-1)
    l1 = torch.logsumexp(torch.where(lb, dd, neg), dim=-1)
    return l0 - l1


def bmi(l: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """Bitwise mutual information, bits per bit: 1 - E log2(1 + e^(-s L))."""
    return 1.0 - torch.nn.functional.softplus(-(1.0 - 2.0 * bits) * l).mean() / np.log(2)
