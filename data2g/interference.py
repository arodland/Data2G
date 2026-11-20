"""Simulation only, not on the live path and not ported to native.

Simulated interference for the session studies (docs/interference-plan.md):
what one station's receiver hears besides the Gaussian floor, over a whole
session, rendered for any time span from a seed (bursts and the gaps between
them see one consistent timeline).

Levels are relative to the floor (`sigma`: the white noise's standard
deviation over FS/2, given at render time), so a cell's SNR does not move
them.

- Impulse trains: Poisson-timed trains of impulses `gap_ms` apart, each a
  short white burst through the radio's passband (it rings 1-3 ms), its peak
  `height_db` over the floor's RMS (2026-09/10 recordings: ~22 dB, 1-3 ms,
  20-40 ms apart; two nodes in 2026-11, behind a limiter: ~14 dB, mostly alone).
- QRM sources: a steady carrier, an FSK-like hopper (8 tones across `bw_hz`,
  a new tone each symbol), or band-limited noise; `inr_db` is its power over
  the floor's in its own band (a carrier: in 50 Hz); on for episodes of mean
  `on_s`, `duty` of the time.
- A receive passband (PASSBANDS, passband()): the radio's filter, which the
  session channel applies to signal and noise alike."""

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
    # noise only: its band (bw_hz wide) sweeps over span_hz about f_hz (a sine, period
    # sweep_s, phase from the seed); at most bw_hz: it stays put
    span_hz: float = 0.0
    sweep_s: float = 10.0


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
    _max_len: int = field(init=False, repr=False)  # longest impulse, samples (floor 64)
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
                k = rng.geometric(1.0 / max(im.per_train, 1.0))  # numpy counts trials, from 1
                s = t
                for _ in range(k):
                    rows.append((int(s * FS), rng.normal(im.height_db, im.height_sd_db),
                                 max(1, int(rng.uniform(*im.length_ms) * 1e-3 * FS)), int(rng.integers(1 << 31))))
                    s += max(0.005, rng.exponential(im.gap_ms * 1e-3))
        self._imp = np.array(sorted(rows), dtype=float).reshape(-1, 4)
        self._max_len = max(64, int(self._imp[:, 2].max())) if len(self._imp) else 64
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
            lo = np.searchsorted(self._imp[:, 0], s0 - self._max_len - ring)
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
        moving = q.span_hz > q.bw_hz
        out = np.empty(len(s), dtype=complex if moving else float)
        c = (s - ep_start) // CHUNK
        for ci in np.unique(c):
            sel = c == ci
            out[sel] = _noise_chunk(q, power, (*key, int(ci)), moving)[(s[sel] - ep_start) % CHUNK]
        if not moving:
            return out
        # the band's centre f_hz + a sin(2 pi t / T + p0): its phase integrates in closed form
        a, T = (q.span_hz - q.bw_hz) / 2, q.sweep_s
        p0 = np.random.default_rng(np.random.SeedSequence([*key[:2], 99])).uniform(0, 2 * np.pi)  # per source
        return np.real(out * np.exp(1j * (2 * np.pi * q.f_hz * t - a * T * np.cos(2 * np.pi * t / T + p0))))
    raise ValueError(f"QRM kind {q.kind!r}")


def _noise_chunk(q: Qrm, power: float, key: tuple, baseband: bool = False) -> np.ndarray:
    """CHUNK samples of noise in the source's band at `power`; baseband: complex, |f| under bw/2
    (its real part after a shift up has `power`)."""
    rng = np.random.default_rng(np.random.SeedSequence(list(key)))
    if baseband:
        W = np.fft.fft(rng.normal(size=CHUNK) + 1j * rng.normal(size=CHUNK))
        W[np.abs(np.fft.fftfreq(CHUNK, 1 / FS)) > q.bw_hz / 2] = 0
        z = np.fft.ifft(W)
        return z * np.sqrt(2 * power / max(np.mean(np.abs(z) ** 2), 1e-30))
    W = np.fft.rfft(rng.normal(size=CHUNK))
    f = np.fft.rfftfreq(CHUNK, 1 / FS)
    W[np.abs(f - q.f_hz) > q.bw_hz / 2] = 0
    x = np.fft.irfft(W, CHUNK)
    return x * np.sqrt(power / max(np.mean(x**2), 1e-30))


# The benchmark's cells (docs/interference-plan.md). In the training ranges:
# impulse trains, a noise source at the top edge (hurts the wide modes only),
# an FSK hopper mid-band (everyone), carriers. Held out (outside draw()'s
# ranges): longer, denser impulses; a wide, near-continuous noise source.
PRESETS = {
    "clean": Spec(),
    "imp": Spec(Impulses(20.0, per_train=4.0, height_db=22.0)),
    "edge": Spec(qrm=(Qrm("noise", 2400, 400, 20.0, 5.0, 0.5),)),
    "mid": Spec(qrm=(Qrm("fsk", 1500, 300, 15.0, 3.0, 0.4),)),
    "carriers": Spec(qrm=(Qrm("carrier", 800, 50, 25.0, 10.0, 0.7), Qrm("carrier", 2200, 50, 25.0, 10.0, 0.7))),
    "imp_long": Spec(Impulses(30.0, per_train=12.0, height_db=26.0, length_ms=(3.0, 5.0))),
    "qrm_wide": Spec(qrm=(Qrm("noise", 1700, 1200, 8.0, 8.0, 0.9),)),
}


def describe(spec: Spec) -> str:
    """A short label, for data rows: '' clean, else 'imp<trains/min>x<per train>@<dB>' and
    '<kind><f>/<bw>+<inr>d<duty>' per QRM source (a swept one '~<span>' after its bw), ';'-separated."""
    parts = []
    if spec.impulses is not None:
        im = spec.impulses
        parts.append(f"imp{im.trains_per_min:.1f}x{im.per_train:.1f}@{im.height_db:.0f}")
    parts += [f"{q.kind}{q.f_hz:.0f}/{q.bw_hz:.0f}" + (f"~{q.span_hz:.0f}" if q.span_hz > q.bw_hz else "")
              + f"+{q.inr_db:.0f}d{q.duty:.2f}" for q in spec.qrm]
    return ";".join(parts)


# --- the training distribution ----------------------------------------------------------

@dataclass(frozen=True)
class Ranges:
    """What draw() produces; the benchmark's held-out cells go outside v1's.
    The fields from height_sd_db on default to v1's draw (its random stream unchanged)."""

    p_clean: float = 0.35
    p_impulses: float = 0.5
    trains_per_min: tuple = (1.0, 60.0)  # log-uniform
    per_train: tuple = (1.0, 8.0)
    height_db: tuple = (15.0, 30.0)
    qrm_max: int = 3
    inr_db: tuple = (0.0, 30.0)
    on_s: tuple = (0.5, 30.0)  # log-uniform
    duty: tuple = (0.05, 0.8)
    height_sd_db: float = 4.0
    loud: float = 0.0  # share of impulse draws at loud_db instead (a radio without a limiter)
    loud_db: tuple = (20.0, 26.0)
    kind_p: tuple | None = None  # KINDS' shares; None: v1's stream (uniform, bw uniform over BW_HZ)
    # per kind, over the above and BW_HZ: inr_db, on_s, duty, bw_hz (log-uniform); noise also
    # moving (share swept: its bw from inst_bw_hz, log-uniform, over span_hz, uniform, sweep_s log-uniform)
    qrm: dict | None = None
    wander_db: float = 0.0  # the floor's wander (phy_session.ContinuousChannel), unless a study sets its own


# v1: interference_round{,2}.sh. mild (round 3): judgement, not fitted to the
# recordings (too few and unrepresentative): more clean receivers, fewer and
# weaker QRM sources on less of the time, slightly weaker impulses. With v1's
# data the model learned caution in clean low-SNR fading (-15..-25%).
# onair (#260): shapes and ranges from ~100 h of two volunteer nodes' noise
# (2026-11), through the same estimators as on the sim: impulses ~14 dB over a
# passband-limited floor (8.5-14.5 over the white one) and mostly alone, 2% of
# draws loud (a radio without a limiter); FSK-like hoppers the main kind, short
# and moderate; few, short carriers; noise mostly 150-1200 Hz wide, sweeping
# over up to 2400 Hz (on air the wide components are narrow at any instant and
# move), a third flat up to 2400 Hz; v1's occupancy; the floor wanders 1 dB.
# docs/interference-plan.md has the like-for-like numbers.
DRAWS = {
    "v1": Ranges(),
    "mild": Ranges(p_clean=0.5, p_impulses=0.4, trains_per_min=(1.0, 30.0), per_train=(1.0, 6.0),
                   height_db=(15.0, 26.0), qrm_max=2, inr_db=(0.0, 20.0), duty=(0.05, 0.5)),
    "onair": Ranges(p_impulses=0.9, trains_per_min=(3.0, 60.0), per_train=(1.0, 1.5), height_db=(8.5, 14.5),
                    height_sd_db=2.0, loud=0.02, qrm_max=5, kind_p=(0.08, 0.55, 0.37),
                    qrm=dict(carrier=dict(inr_db=(6.0, 20.0), on_s=(0.5, 3.0), duty=(0.05, 0.5)),
                             fsk=dict(bw_hz=(50.0, 450.0), inr_db=(3.0, 18.0), on_s=(0.3, 2.0), duty=(0.05, 0.5)),
                             noise=dict(bw_hz=(300.0, 2400.0), inr_db=(3.0, 16.0), on_s=(2.0, 15.0), duty=(0.1, 0.9),
                                        moving=0.7, inst_bw_hz=(150.0, 1200.0), span_hz=(300.0, 2400.0),
                                        sweep_s=(4.0, 30.0))),
                    wander_db=1.0),
}
BW_HZ = {"carrier": (50.0, 50.0), "fsk": (50.0, 500.0), "noise": (200.0, 1000.0)}
KINDS = ("carrier", "fsk", "noise")
DEFAULT_DRAWS = "onair"  # what new training data draws (session_data --interference)


def draw(rng: np.random.Generator, draws: str = "v1") -> Spec:
    """One station's interference for a training session, from DRAWS[draws]."""
    r = DRAWS[draws]
    if rng.random() < r.p_clean:
        return Spec()
    imp = None
    if rng.random() < r.p_impulses:
        trains = float(np.exp(rng.uniform(*np.log(r.trains_per_min))))
        per = float(rng.uniform(*r.per_train))
        h = r.loud_db if r.loud and rng.random() < r.loud else r.height_db
        imp = Impulses(trains_per_min=trains, per_train=per, height_db=float(rng.uniform(*h)),
                       height_sd_db=r.height_sd_db)
    qrm = []
    for _ in range(int(rng.integers(0 if imp else 1, r.qrm_max + 1))):
        if r.qrm is None:
            kind = str(rng.choice(KINDS))
            bw = float(rng.uniform(*BW_HZ[kind]))
            inr, on, duty = r.inr_db, r.on_s, r.duty
        else:
            kind = str(rng.choice(KINDS, p=r.kind_p))
            k = r.qrm[kind]
            bw = float(np.exp(rng.uniform(*np.log(k.get("bw_hz", BW_HZ[kind])))))
            inr, on, duty = k.get("inr_db", r.inr_db), k.get("on_s", r.on_s), k.get("duty", r.duty)
            if k.get("moving") and rng.random() < k["moving"]:
                bw = float(np.exp(rng.uniform(*np.log(k["inst_bw_hz"]))))
                span = float(rng.uniform(max(1.5 * bw, k["span_hz"][0]), max(1.5 * bw, k["span_hz"][1])))
                span = min(span, PASSBAND_HZ[1] - PASSBAND_HZ[0])
                sweep = float(np.exp(rng.uniform(*np.log(k["sweep_s"]))))
                qrm.append(Qrm(kind, float(rng.uniform(PASSBAND_HZ[0] + span / 2, PASSBAND_HZ[1] - span / 2)), bw,
                               float(rng.uniform(*inr)), float(np.exp(rng.uniform(*np.log(on)))),
                               float(rng.uniform(*duty)), span, sweep))
                continue
        qrm.append(Qrm(kind, float(rng.uniform(PASSBAND_HZ[0] + bw / 2, PASSBAND_HZ[1] - bw / 2)), bw,
                       float(rng.uniform(*inr)), float(np.exp(rng.uniform(*np.log(on)))),
                       float(rng.uniform(*duty))))
    return Spec(imp, tuple(qrm))


def draw_passband(rng: np.random.Generator) -> str | None:
    """One station's receive passband, PASSBAND_SHARES: an "ssb:..." filter of its
    own (draw_ssb), a PASSBANDS name, or None (flat)."""
    u, acc = rng.random(), 0.0
    for name, share in PASSBAND_SHARES:
        acc += share
        if u < acc:
            return draw_ssb(rng) if name == "ssb" else name
    return None


def draw_ssb(rng: np.random.Generator) -> str:
    """A current SSB receive filter: -6 dB edges at 150-300 Hz and 2.7-3.1 kHz
    above it, skirts to -40 dB over 60-140 Hz (low) and 100-300 Hz (high), a
    -1.5..+1.5 dB tilt across the passband. "ssb:lo:hi:skirt_lo:skirt_hi:tilt"."""
    lo = rng.uniform(150, 300)
    hi = lo + rng.uniform(2700, 3100)
    return f"ssb:{lo:.0f}:{hi:.0f}:{rng.uniform(60, 140):.0f}:{rng.uniform(100, 300):.0f}:{rng.uniform(-1.5, 1.5):.1f}"


def passband_table(name: str) -> tuple:
    """The (Hz, dB) points of a passband: PASSBANDS[name], or a draw_ssb filter's."""
    if not name.startswith("ssb:"):
        return PASSBANDS[name]
    lo, hi, sl, sh, tilt = (float(v) for v in name.split(":")[1:])
    return ((0, -60), (lo - sl, -40), (lo, -6), (lo + 50, -2), (lo + 150, -tilt / 2), (hi - 150, tilt / 2),
            (hi - 50, -2), (hi, -6), (hi + sh, -40), (FS / 2, -60))


def draw_session(seed: int, draws: str) -> tuple:
    """Both stations' interference Specs for a session's seed."""
    irng = np.random.default_rng(np.random.SeedSequence([seed, 5]))
    return draw(irng, draws), draw(irng, draws)


def session_passbands(seed: int) -> tuple:
    """Both stations' receive passbands for a session's seed (its own stream: the Specs don't move)."""
    prng = np.random.default_rng(np.random.SeedSequence([seed, 6]))
    return draw_passband(prng), draw_passband(prng)


# --- the receive passband ------------------------------------------------------------------

# A radio's receive filter: gain (dB) at frequencies (Hz), linear in dB between
# them. From the floor's shape on two volunteer nodes (2026-11, against its
# 500-2500 Hz median): wide is the one's (-4.7 at 300 Hz, -3 at 2400, -8 at
# 2700, -15 at 2800), narrow the other's narrower setting (-15 at 2400).
PASSBANDS = {
    "wide": ((0, -40), (150, -25), (200, -12), (300, -4.7), (400, -1), (500, 0), (700, 1), (1000, 0.8), (1500, 0),
             (2000, -1.2), (2400, -3), (2500, -3.6), (2600, -4.8), (2700, -8), (2800, -15), (2900, -25),
             (3000, -35), (FS / 2, -50)),
    "narrow": ((0, -40), (150, -15), (200, -6), (300, -2.5), (400, -0.8), (500, 0), (2000, -1.5), (2200, -5),
               (2400, -15), (2500, -22), (2600, -30), (2700, -40), (2800, -50), (FS / 2, -60)),
}


# Almost every radio receives through a current SSB filter (draw_ssb: 2.7-3.1 kHz
# wide); an old narrow one (the narrow node's, 2 kHz) a little, so the model copes;
# a few SDRs are as good as flat. "wide" (the other node's, ~2.4 kHz) is kept for
# studies but not drawn.
PASSBAND_SHARES = (("ssb", 0.85), ("narrow", 0.05))  # the rest (0.1) flat


def passband(y: np.ndarray, name: str | None) -> np.ndarray:
    """y through the receive passband `name` (passband_table; None: y itself), zero-phase."""
    if name is None or len(y) == 0:
        return y
    f_db = np.array(passband_table(name), dtype=float)
    f = np.fft.rfftfreq(len(y), 1 / FS)
    # ponytail: one circular FFT per block, so a block's last few ms leak into its first;
    # the session's blocks start and end in noise (PAD_S, StationNoise's 1 s chunks)
    return np.fft.irfft(np.fft.rfft(y) * 10 ** (np.interp(f, f_db[:, 0], f_db[:, 1]) / 20), len(y))
