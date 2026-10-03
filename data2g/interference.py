"""Simulated interference for the session studies (docs/interference-plan.md):
what one station's receiver hears besides the Gaussian floor, over a whole
session, rendered for any time span from a seed (bursts and the gaps between
them see one consistent timeline).

Levels are relative to the floor (`sigma`: the white noise's standard
deviation over FS/2, given at render time), so a cell's SNR does not move
them.

- Impulse trains: Poisson-timed trains of impulses `gap_ms` apart, each a
  short white burst through the radio's passband (it rings 1-3 ms), its peak
  `height_db` over the floor's RMS (on air: ~22 dB, 1-3 ms, 20-40 ms apart).
- QRM sources: a steady carrier, an FSK-like hopper (8 tones across `bw_hz`,
  a new tone each symbol), or band-limited noise; `inr_db` is its power over
  the floor's in its own band (a carrier: in 50 Hz); on for episodes of mean
  `on_s`, `duty` of the time."""

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

from .config import FS

PASSBAND_HZ = (300.0, 2800.0)
CARRIER_REF_HZ = 50.0  # a carrier's INR is against the floor in this bandwidth
CHUNK = FS // 2  # band-limited noise is generated in chunks this long
_PASS = sps.firwin(49, PASSBAND_HZ, pass_zero=False, fs=FS)  # the impulses' ringing


@dataclass(frozen=True)
class Impulses:
    trains_per_min: float
    per_train: float = 4.0  # mean impulses in a train (1 + geometric)
    gap_ms: float = 40.0  # mean spacing within a train (exponential, at least 5 ms)
    height_db: float = 22.0  # median peak over the floor's RMS
    height_sd_db: float = 4.0
    length_ms: tuple = (0.1, 1.0)  # the white burst, before the passband rings it


@dataclass(frozen=True)
class Qrm:
    kind: str  # "carrier", "fsk" or "noise"
    f_hz: float  # centre
    bw_hz: float
    inr_db: float
    on_s: float  # mean episode length
    duty: float  # share of the time on


@dataclass(frozen=True)
class Spec:
    impulses: Impulses | None = None
    qrm: tuple = ()

    @property
    def clean(self) -> bool:
        return self.impulses is None and not self.qrm


@dataclass
class Interference:
    """One receiver's interference over [0, horizon) seconds."""

    spec: Spec
    seed: int
    horizon: float
    _imp: np.ndarray = field(init=False, repr=False)  # (start sample, height, length samples, seed) rows
    _episodes: list = field(init=False, repr=False)  # per QRM source: (starts, ends) in samples

    def __post_init__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1]))
        n_end = int(self.horizon * FS)
        rows = []
        im = self.spec.impulses
        if im is not None and im.trains_per_min > 0:
            t = 0.0
            while True:
                t += rng.exponential(60.0 / im.trains_per_min)
                if t * FS >= n_end:
                    break
                k = 1 + rng.geometric(1.0 / max(im.per_train, 1.0)) - 1
                s = t
                for _ in range(k):
                    rows.append((int(s * FS), rng.normal(im.height_db, im.height_sd_db),
                                 max(1, int(rng.uniform(*im.length_ms) * 1e-3 * FS)), int(rng.integers(1 << 31))))
                    s += max(0.005, rng.exponential(im.gap_ms * 1e-3))
        self._imp = np.array(sorted(rows), dtype=float).reshape(-1, 4)
        self._episodes = []
        for q in self.spec.qrm:
            starts, ends = [], []
            off_s = q.on_s * (1 - q.duty) / max(q.duty, 1e-6)
            t = rng.exponential(off_s) if q.duty < 1 else 0.0
            while t < self.horizon:
                on = rng.exponential(q.on_s) if q.duty < 1 else self.horizon
                starts.append(int(t * FS))
                ends.append(int(min(t + on, self.horizon) * FS))
                t += on + (rng.exponential(off_s) if q.duty < 1 else 0.0)
            self._episodes.append((np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)))

    def render(self, t0: float, n: int, sigma: float) -> np.ndarray:
        """The interference in [t0, t0 + n / FS), for a floor of standard deviation sigma."""
        y = np.zeros(n)
        if self.spec.clean or n <= 0:
            return y
        s0 = int(round(t0 * FS))
        s1 = s0 + n
        n0 = sigma**2 / (FS / 2)  # the floor's density, per Hz
        if len(self._imp):
            ring = len(_PASS)
            lo = np.searchsorted(self._imp[:, 0], s0 - 64 - ring)
            hi = np.searchsorted(self._imp[:, 0], s1)
            for start, h_db, length, seed in self._imp[lo:hi]:
                w = _impulse(int(length), int(seed))
                w *= sigma * 10 ** (h_db / 20) / np.max(np.abs(w))
                _add(y, w, int(start) - s0)
        for i, (q, (starts, ends)) in enumerate(zip(self.spec.qrm, self._episodes)):
            lo = np.searchsorted(ends, s0, side="right")
            hi = np.searchsorted(starts, s1)
            bw = CARRIER_REF_HZ if q.kind == "carrier" else q.bw_hz
            power = 10 ** (q.inr_db / 10) * n0 * bw
            for e in range(lo, hi):
                a, b = max(starts[e], s0), min(ends[e], s1)
                if a < b:
                    y[a - s0:b - s0] += _qrm(q, power, np.arange(a, b), int(starts[e]), (self.seed, i, e))
        return y


def _add(y, w, at):
    a, b = max(0, at), min(len(y), at + len(w))
    if a < b:
        y[a:b] += w[a - at:b - at]


def _impulse(length: int, seed: int) -> np.ndarray:
    burst = np.random.default_rng(seed).normal(size=length)
    return np.convolve(burst, _PASS)


def _qrm(q: Qrm, power: float, s: np.ndarray, ep_start: int, key: tuple) -> np.ndarray:
    """One source's signal at absolute samples s (within one episode)."""
    t = s / FS

    def stream(i):  # one stream per quantity, so any span draws the same values
        return np.random.default_rng(np.random.SeedSequence([*key, i]))

    if q.kind == "carrier":
        return np.sqrt(2 * power) * np.cos(2 * np.pi * q.f_hz * t + stream(0).uniform(0, 2 * np.pi))
    if q.kind == "fsk":
        ts = 8.0 / q.bw_hz  # 8 orthogonal tones across bw: spacing bw / 8, symbol 8 / bw
        k = ((s - ep_start) / FS / ts).astype(np.int64)
        n_sym = int(k.max()) + 1
        tone = stream(1).integers(0, 8, size=n_sym)
        phase = stream(2).uniform(0, 2 * np.pi, size=n_sym)
        f = q.f_hz + (tone[k] - 3.5) * q.bw_hz / 8
        return np.sqrt(2 * power) * np.cos(2 * np.pi * f * t + phase[k])
    if q.kind == "noise":
        out = np.empty(len(s))
        c = (s - ep_start) // CHUNK
        for ci in np.unique(c):
            sel = c == ci
            out[sel] = _noise_chunk(q, power, (*key, int(ci)))[(s[sel] - ep_start) % CHUNK]
        return out
    raise ValueError(f"QRM kind {q.kind!r}")


def _noise_chunk(q: Qrm, power: float, key: tuple) -> np.ndarray:
    w = np.random.default_rng(np.random.SeedSequence(list(key))).normal(size=CHUNK)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(CHUNK, 1 / FS)
    W[np.abs(f - q.f_hz) > q.bw_hz / 2] = 0
    x = np.fft.irfft(W, CHUNK)
    return x * np.sqrt(power / max(np.mean(x**2), 1e-30))


def describe(spec: Spec) -> str:
    """A short label, for data rows: '' clean, else 'imp<trains/min>x<per train>@<dB>' and
    '<kind><f>/<bw>+<inr>d<duty>' per QRM source, ';'-separated."""
    parts = []
    if spec.impulses is not None:
        im = spec.impulses
        parts.append(f"imp{im.trains_per_min:.1f}x{im.per_train:.1f}@{im.height_db:.0f}")
    parts += [f"{q.kind}{q.f_hz:.0f}/{q.bw_hz:.0f}+{q.inr_db:.0f}d{q.duty:.2f}" for q in spec.qrm]
    return ";".join(parts)


# --- the training distribution ----------------------------------------------------------

# Ranges of what draw() produces; the benchmark's held-out cells go outside them.
P_CLEAN = 0.35
TRAINS_PER_MIN = (1.0, 60.0)  # log-uniform
PER_TRAIN = (1.0, 8.0)
HEIGHT_DB = (15.0, 30.0)
QRM_MAX = 3
INR_DB = (0.0, 30.0)
ON_S = (0.5, 30.0)  # log-uniform
DUTY = (0.05, 0.8)
BW_HZ = {"carrier": (50.0, 50.0), "fsk": (50.0, 500.0), "noise": (200.0, 1000.0)}
KINDS = ("carrier", "fsk", "noise")


def draw(rng: np.random.Generator) -> Spec:
    """One station's interference for a training session."""
    if rng.random() < P_CLEAN:
        return Spec()
    imp = None
    if rng.random() < 0.5:
        imp = Impulses(trains_per_min=float(np.exp(rng.uniform(*np.log(TRAINS_PER_MIN)))),
                       per_train=float(rng.uniform(*PER_TRAIN)), height_db=float(rng.uniform(*HEIGHT_DB)))
    qrm = []
    for _ in range(int(rng.integers(0 if imp else 1, QRM_MAX + 1))):
        kind = str(rng.choice(KINDS))
        bw = float(rng.uniform(*BW_HZ[kind]))
        qrm.append(Qrm(kind, float(rng.uniform(PASSBAND_HZ[0] + bw / 2, PASSBAND_HZ[1] - bw / 2)), bw,
                       float(rng.uniform(*INR_DB)), float(np.exp(rng.uniform(*np.log(ON_S)))), float(rng.uniform(*DUTY))))
    return Spec(imp, tuple(qrm))
