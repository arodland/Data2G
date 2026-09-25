"""NumPy HF channel simulator: AWGN, Watterson-style fading, frequency
offset, sample-clock error. Operates on real passband audio at FS.

SNR is signal power relative to the noise power falling in
`config.SNR_REF_BW_HZ`.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .waveform import dsp
from .config import FS, SNR_REF_BW_HZ


@dataclass(frozen=True)
class FadingPreset:
    name: str
    doppler_hz: float  # two-sided Doppler spread
    delay_ms: float  # second-path delay


FADING_PRESETS = {
    "mpg": FadingPreset("mpg", 0.1, 0.5),  # good
    "mpp": FadingPreset("mpp", 1.0, 2.0),  # poor (CCIR)
    "mpd": FadingPreset("mpd", 2.0, 4.0),  # disturbed
    # NOT a CCIR preset -- measured, 2026-08-28, from four consecutive
    # mode C receptions over a ~4000 km path (wav-samples/). The CCIR
    # three tie Doppler to delay spread, and that path does not: its
    # envelope decorrelation time was 0.86-4.03 s (Doppler 0.05-0.3 Hz
    # by the same estimator, centred ~0.15) while its frequency
    # selectivity matched mpp/mpd's ~2 ms. So it fades an order of
    # magnitude slower than mpp while being just as selective, a
    # combination none of the three can express -- mpg is slow but
    # nearly flat. Slow fading is the harder case for the interleaver,
    # since a fade that outlasts a frame damages latents in correlated
    # blocks rather than sprinkling them. Fade depth here runs
    # 6.4-7.0 dB against the measured 4.8-6.3, so it is if anything
    # slightly pessimistic.
    "mps": FadingPreset("mps", 0.15, 2.0),  # slow + selective (measured)
}


def _analytic(x: np.ndarray) -> np.ndarray:
    return signal.hilbert(x)


def freq_shift(x: np.ndarray, df_hz: float) -> np.ndarray:
    # Phase reduced to one turn before exp(), for the reason in
    # dsp.wrap_cycles: over a whole transmission the unreduced argument
    # reaches tens of thousands of radians, where the result depends on
    # the platform's argument reduction rather than on the signal.
    n = np.arange(len(x))
    return np.real(_analytic(x) * np.exp(2j * np.pi * dsp.wrap_cycles(df_hz * n / FS)))


def sample_clock_offset(x: np.ndarray, ppm: float) -> np.ndarray:
    """Resample as if the far-end clock ran (1 + ppm*1e-6) fast.

    Band-limited (FFT) resampling. SSTVAE's np.interp version adds
    linear-interpolation distortion at ~-23 dB on this waveform, which
    its clipper (~12.7 dB SINR) happens to mask but a channel model
    must not have. The FFT form is circular; the lead-in/out silence
    absorbs the wrap.
    """
    return signal.resample(x, int(round(len(x) / (1 + ppm * 1e-6))))


def _butter_taps(n: int, doppler_hz: float, rng: np.random.Generator) -> np.ndarray:
    """SSTVAE's tap generator, kept for comparison with its numbers.

    Not the Watterson spectrum its label claims: measured, the 2 Hz
    setting has a 2-sigma spread of 3.0 Hz and a 99% bandwidth of 7.4 Hz
    (2nd-order Butterworth skirts), wider than the 6.9 Hz pilot rate.
    """
    lowrate = max(8 * doppler_hz, 1.0)
    n_low = int(np.ceil(n * lowrate / FS)) + 8
    g = rng.normal(size=n_low) + 1j * rng.normal(size=n_low)
    b, a = signal.butter(2, min(doppler_hz / (lowrate / 2), 0.99))
    g = signal.lfilter(b, a, g)
    g = g[4:]  # drop filter transient
    t_low = np.arange(len(g)) * (FS / lowrate)
    t = np.arange(n)
    tap = np.interp(t, t_low, g.real) + 1j * np.interp(t, t_low, g.imag)
    return tap / np.sqrt(np.mean(np.abs(tap) ** 2))


def _gaussian_taps(n: int, spread_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Rayleigh tap with the ITU-R F.1487 Doppler spectrum: Gaussian,
    frequency spread = 2 sigma. Shaped in the frequency domain at a low
    rate (circular, so no transient), then linearly interpolated; at 64x
    oversampling the interpolation error is < -60 dB.

    Unit power in expectation, not per realization: normalizing each
    realization (as SSTVAE does) divides a slow fade out of a short
    burst, which then simulates as no fade at all.
    """
    lowrate = max(64 * spread_hz, 8.0)
    n_low = int(np.ceil(n * lowrate / FS)) + 2
    f = np.fft.fftfreq(n_low, 1 / lowrate)
    shape = np.exp(-(f**2) / (4 * (spread_hz / 2) ** 2))
    g = np.fft.ifft(np.fft.fft(rng.normal(size=n_low) + 1j * rng.normal(size=n_low)) * shape)
    g /= np.sqrt(2 * np.mean(shape**2))
    t_low = np.arange(n_low) * (FS / lowrate)
    t = np.arange(n)
    return np.interp(t, t_low, g.real) + 1j * np.interp(t, t_low, g.imag)


_TAPS = {"gaussian": _gaussian_taps, "butter": _butter_taps}


def fading(
    x: np.ndarray, preset: str | FadingPreset, seed: int = 0, taps: str = "gaussian"
) -> np.ndarray:
    """Two independent equal-power Rayleigh paths (Watterson model).
    `taps="butter"` reproduces SSTVAE's harsher simulator."""
    p = FADING_PRESETS[preset] if isinstance(preset, str) else preset
    rng = np.random.default_rng(seed)
    z = _analytic(x)
    delay = int(round(p.delay_ms * 1e-3 * FS))
    g1 = _TAPS[taps](len(z), p.doppler_hz, rng)
    g2 = _TAPS[taps](len(z), p.doppler_hz, rng)
    z2 = np.concatenate([np.zeros(delay, dtype=complex), z[: len(z) - delay]])
    return np.real((z * g1 + z2 * g2) / np.sqrt(2))


def active_power(x: np.ndarray) -> float:
    """Mean power over the active portion (envelope above 10% of the
    overall RMS), so lead-in/out silence doesn't skew it."""
    env = np.abs(_analytic(x))
    active = env > 0.1 * np.sqrt(np.mean(x**2))
    return float(np.mean(x[active] ** 2) if active.any() else np.mean(x**2))


def awgn(x: np.ndarray, snr_db: float, seed: int = 0, s_power: float | None = None) -> np.ndarray:
    """Add white noise for the given SNR in a `SNR_REF_BW_HZ` bandwidth,
    against `s_power` (default: `x`'s own active power)."""
    rng = np.random.default_rng(seed)
    if s_power is None:
        s_power = active_power(x)
    # White noise over FS/2 Hz with total power sigma^2 puts
    # sigma^2 * SNR_REF_BW_HZ / (FS/2) into the reference bandwidth.
    sigma2 = s_power * (FS / 2) / SNR_REF_BW_HZ / 10 ** (snr_db / 10)
    return x + rng.normal(scale=np.sqrt(sigma2), size=len(x))


def zero_spans(x: np.ndarray, spans_s: list[tuple[float, float]]) -> np.ndarray:
    """Blank out time spans (seconds) — simulates lost/blocked frames."""
    y = x.copy()
    for a, b in spans_s:
        y[int(a * FS) : int(b * FS)] = 0.0
    return y


def apply_channel(
    x: np.ndarray,
    snr_db: float | None = None,
    freq_offset_hz: float = 0.0,
    ppm: float = 0.0,
    fading_preset: str | None = None,
    spans: list[tuple[float, float]] | None = None,
    seed: int = 0,
    taps: str = "gaussian",
) -> np.ndarray:
    y = x.astype(np.float64)
    # SNR is against the transmitted power, i.e. the average received
    # power (the taps are unit power in expectation), not this burst's:
    # a burst that lands in a fade is a low-SNR burst.
    s_power = active_power(y)
    if ppm:
        y = sample_clock_offset(y, ppm)
    if freq_offset_hz:
        y = freq_shift(y, freq_offset_hz)
    if fading_preset:
        y = fading(y, fading_preset, seed=seed, taps=taps)
    if spans:
        y = zero_spans(y, spans)
    if snr_db is not None:
        y = awgn(y, snr_db, seed=seed + 1, s_power=s_power)
    return y
