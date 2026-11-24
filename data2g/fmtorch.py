"""Differentiable FM channel for training constellations (FM plan M4/M5).

`FmBurstChannel` is channel_torch.BurstChannel (the real TX, clipper and
the real numpy channel estimator) with fmchannel's FM chain in place of
HF's Watterson channel. The chain is not a statistical stand-in: it is
fmchannel.FmChannel.apply's signal path ported to torch, every filter an
exact linear operator (the scipy filter's own impulse response, applied by
FFT; resample_poly's own polyphase FIR), so the limiter, the TX radio's
pre-emphasis and band, FM modulation, the RF path, the IF filter, the
limiter-discriminator (clicks come out of it as they do in a radio),
de-emphasis and the RX band are all in the gradient's path. What does not
depend on the data (the path's fading gains, the noise draws) comes from
fmchannel itself, seeded as FmChannel.apply seeds them, so a burst's
output matches numpy's to float rounding (tests/test_fmtorch.py).

Measured residual of the chain (scripts/fm_residual.py) is reported there;
this module only reproduces the chain.

Training only (torch), not on the live path and not ported to native.
"""

import numpy as np
import torch
from scipy import signal

from . import config, fmchannel
from .channel_torch import CHANNELS, BurstChannel
from .config import FS, SubmodeSpec

OBW_HZ = 15_000.0  # the drive's occupied-bandwidth target, 26 dB down (FM plan Q15)


def _fftconv(x: torch.Tensor, h: torch.Tensor, n_out: int, offset: int = 0) -> torch.Tensor:
    """Full linear convolution of each row of x with h, samples offset..offset+n_out."""
    n = x.shape[-1] + h.shape[-1] - 1
    nfft = 1 << (n - 1).bit_length()
    if x.is_complex() or h.is_complex():
        y = torch.fft.ifft(torch.fft.fft(x, nfft) * torch.fft.fft(h, nfft))
    else:
        y = torch.fft.irfft(torch.fft.rfft(x, nfft) * torch.fft.rfft(h, nfft), nfft)
    y = y[..., offset:offset + n_out]
    if y.shape[-1] < n_out:
        y = torch.nn.functional.pad(y, (0, n_out - y.shape[-1]))
    return y


class _Iir:
    """A scipy IIR filter (lfilter b, a or sosfilt sos) as its impulse
    response over n samples: exact for any n-sample input (a causal
    filter's first n outputs need only its first n taps)."""

    def __init__(self, n: int, device, b=None, a=None, sos=None):
        d = np.zeros(n)
        d[0] = 1.0
        h = signal.sosfilt(sos, d) if sos is not None else signal.lfilter(b, a, d)
        self.h = torch.tensor(h, dtype=torch.float64, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _fftconv(x, self.h.to(x.real.dtype), x.shape[-1])


class _Resample:
    """scipy.signal.resample_poly(x, up, down) (window kaiser 5, as scipy's
    default), along the last axis."""

    def __init__(self, up: int, down: int, device):
        g = np.gcd(up, down)
        self.up, self.down = up // g, down // g
        half = 10 * max(self.up, self.down)
        h = signal.firwin(2 * half + 1, 1.0 / max(self.up, self.down), window=("kaiser", 5.0)) * self.up
        pre = self.down - half % self.down
        self.remove = (half + pre) // self.down
        self.h = torch.tensor(np.concatenate([np.zeros(pre), h]), dtype=torch.float64, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        n_out = -(-n * self.up // self.down)
        if self.up > 1:
            xs = torch.zeros(*x.shape[:-1], (n - 1) * self.up + 1, dtype=x.dtype, device=x.device)
            xs[..., :: self.up] = x
        else:
            xs = x
        full = _fftconv(xs, self.h.to(x.dtype), xs.shape[-1] + self.h.shape[-1] - 1)
        y = full[..., :: self.down][..., self.remove:self.remove + n_out]
        if y.shape[-1] < n_out:
            y = torch.nn.functional.pad(y, (0, n_out - y.shape[-1]))
        return y


def _fdelay(s: torch.Tensor, d: float) -> torch.Tensor:
    """fmchannel._delay: s delayed by d (fractional) samples, by FFT, padded."""
    if d == 0:
        return s
    pad = int(np.ceil(d)) + 8
    n = s.shape[-1] + pad
    f = torch.fft.fftfreq(n, device=s.device, dtype=torch.float64)
    return torch.fft.ifft(torch.fft.fft(s, n) * torch.exp(-2j * np.pi * f * d))[..., : s.shape[-1]]


def default_drive(spec: SubmodeSpec, radio: str, n_frames: int = 8, seed: int = 0) -> float:
    """The rms drive (Hz) at which a burst of `spec`'s band, random data in
    its constellation, occupies OBW_HZ 26 dB down (fmchannel.drive_for_bandwidth)."""
    from . import constellation, modem

    pts = constellation.load(spec.constellation)
    m = int(np.log2(len(pts)))
    nc = config.BANDS[spec.band].nc
    bits = np.random.default_rng(seed).integers(0, 2, n_frames * config.DATA_SYMS_PER_FRAME * nc * m)
    x = modem.modulate_bits(bits, spec)
    return fmchannel.drive_for_bandwidth(x[np.abs(x) > 0], radio, OBW_HZ, "db26")[0]


class FmBurstChannel(BurstChannel):
    """BurstChannel through an FM radio pair (fmchannel.RADIOS[radio]) and
    path (fmchannel.CONDITIONS[condition]); CNR in fmchannel's convention.
    `channel(x, cnr_db, g)`; `receive(y, points=)` as modem.receive past
    acquisition (measured delay support, per-carrier preamble noise, per-symbol
    noise with config.SYMBOL_NOISE)."""

    def __init__(self, spec: SubmodeSpec, n_frames: int, radio: str = "mic", condition: str = "static",
                 drive_dev_hz: float | None = None, carrier_mhz: float = 146.0, device="cpu",
                 dtype=torch.float32):
        super().__init__(spec, n_frames, device=device, dtype=dtype)
        self.radio, self.condition, self.carrier_mhz = radio, condition, carrier_mhz
        self.rad = r = fmchannel.RADIOS[radio]
        self.drive = drive_dev_hz if drive_dev_hz is not None else default_drive(spec, radio, n_frames)
        self.rf_fs = fmchannel.rf_rate(FS)
        self.up = self.rf_fs // FS
        n = len(self.base)
        self.n_rf = n * self.up
        # TX radio (fmchannel.tx_audio): pre-emphasis (mic), limiter, band
        self.pre = _Iir(n, device, *fmchannel._emphasis(FS, r.emphasis_us, inverse=False)) if r.path == "mic" else None
        self.de = _Iir(n, device, *fmchannel._emphasis(FS, r.emphasis_us, inverse=True)) if r.path == "mic" else None
        hi = min(r.audio_hi_hz, 0.45 * FS)
        self.lp = _Iir(n, device, sos=signal.butter(6 if r.path == "mic" else 4, hi, "lowpass", fs=FS, output="sos"))
        self.hp = _Iir(n, device, sos=signal.butter(2 if r.path == "mic" else 1, r.audio_lo_hz, "highpass", fs=FS,
                                                    output="sos"))
        self.if_filter = _Iir(self.n_rf, device, sos=signal.butter(r.if_order, r.if_bw_hz / 2, "lowpass", fs=self.rf_fs,
                                                                  output="sos"))
        self.rs_up, self.rs_down = _Resample(self.up, 1, device), _Resample(1, self.up, device)

    def _band(self, y):
        return self.hp(self.lp(y))

    def _processes(self, n: int, cnr_db: float, seeds: list):
        """Per burst of n audio samples, from fmchannel itself as
        FmChannel.apply draws them (fmchannel.apply_channel's seed and
        duration): the path's tap gains and flutter (b, taps(+1), n_rf), slow
        amplitude (b, n_rf), RF noise (b, n_rf), the RX audio chain's own noise (b, n)."""
        n_rf = n * self.up
        t = np.arange(n_rf) / self.rf_fs
        gains, slow, noise, ceil = [], [], [], []
        for s in seeds:
            ch = fmchannel.FmChannel(self.condition, self.radio, cnr_db, s, duration_s=n / FS + 2.0,
                                     carrier_mhz=self.carrier_mhz, drive_dev_hz=self.drive)
            gk = [ch._at(gt, t) for gt in ch._gains]
            gk.append(ch._at(ch._flutter, t) if ch._flutter is not None else np.zeros(n_rf, complex))
            gains.append(np.stack(gk))
            slow.append(ch._at(ch._slow_amp, t))
            rng = np.random.default_rng([ch.burst_seed, 0])
            sigma2 = 10 ** (-cnr_db / 10) * self.rf_fs / fmchannel.CNR_REF_BW_HZ
            noise.append(np.sqrt(sigma2 / 2) * (rng.normal(size=n_rf) + 1j * rng.normal(size=n_rf)))
            ceil.append(rng.normal(scale=10 ** (-self.rad.ceiling_db / 20), size=n)
                        if self.rad.ceiling_db is not None else np.zeros(n))
        dev = self.device
        return (torch.tensor(np.stack(gains), device=dev), torch.tensor(np.stack(slow), device=dev),
                torch.tensor(np.stack(noise), device=dev), torch.tensor(np.stack(ceil), device=dev))

    def channel(self, x: torch.Tensor, cnr_db: float, g: torch.Generator | None = None,
                seeds: list | None = None) -> torch.Tensor:
        """fmchannel.apply_channel(x, cnr_db, condition, radio, seed) for each
        row, the seeds drawn from `g` or given (key-up, CTCSS, squelch, clock
        offset and DC step off: the burst is acquired). Any length of x: a
        padded burst pairs with numpy's (scripts/fm_bmi_check.py)."""
        r = self.rad
        x = x.to(torch.float64)
        if seeds is None:
            seeds = torch.randint(0, 1 << 31, (x.shape[0],), generator=g).tolist()
        gains, slow, noise, ceil = self._processes(x.shape[-1], cnr_db, seeds)
        ref_rms = (x.pow(2).sum(-1, keepdim=True) / (x != 0).sum(-1, keepdim=True)).sqrt()
        y = x / ref_rms * self.drive
        if self.pre is not None:
            y = self.pre(y)
        if r.peak_dev_hz is not None:
            y = y.clamp(-r.peak_dev_hz, r.peak_dev_hz)
        dev = self._band(y)  # Hz of deviation
        f_inst = self.rs_up(dev)
        s = torch.exp(2j * np.pi * torch.cumsum(f_inst, -1) / self.rf_fs)
        taps = fmchannel.CONDITIONS[self.condition].taps
        z = sum(_fdelay(s, tap.delay_us * 1e-6 * self.rf_fs) * gains[:, k] for k, tap in enumerate(taps))
        z = (z + s * gains[:, -1]) * slow + noise
        # RX radio (fmchannel.rx_audio)
        z = self.if_filter(z)
        f = torch.angle(z[..., 1:] * z[..., :-1].conj()) * self.rf_fs / (2 * np.pi)
        f = torch.nn.functional.pad(f, (1, 0))
        out = self.rs_down(f)
        if self.de is not None:
            out = self.de(out)
        out = self._band(out) / self.drive + ceil
        return (out * ref_rms).to(self.dtype)

    def receive(self, y: torch.Tensor, points: np.ndarray | None = None):
        return super().receive(y, CHANNELS["awgn"], measure=True, points=points)
