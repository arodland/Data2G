"""Link prediction for the gear shifter.

- The outcome model (below; codes_data/outcome_predictor.npz, trained by
  scripts/train_outcome.py on real decodes, ARQ sessions included): per
  submode, P(its next burst is usable) and P(a codeword decodes | usable)
  from the receiver's last two measurements. Plain numpy at runtime.
- The measurement features it takes: effective MI per constellation
  (MIESM over the AWGN BICM capacity tables).
- The link abstraction (codes_data/link_abstraction.json): per submode,
  P(decode) as a sigmoid in effective MI, for scripts/linksim.py.

(An MI-forecasting MLP, link_predictor.npz, was the shifter's fallback
without the outcome model; it is gone, as the outcome model always ships.)
"""

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import SUBMODES

DATA = Path(__file__).parent.parent / "codes_data"
BANDS = ("w", "n10", "n4", "w48")
CONSTS = ("gray-qam4", "gray-qam16", "c64-snr18", "c256-snr26")


@lru_cache(maxsize=None)
def _capacity():
    d = np.load(DATA / "capacity_tables.npz")
    return d["grid"], {c: d[c] for c in CONSTS}


def effective_mi(h: np.ndarray, var: np.ndarray, const: str) -> float:
    """Mean AWGN BICM capacity (bits per coded bit) at |h|^2 / var over
    the channel uses: the MIESM feature (scripts/rx_audit.py measured it
    against the genie BMI of real LLRs: within +-0.01 median)."""
    grid, tables = _capacity()
    snr = 10 * np.log10(np.maximum(np.abs(h) ** 2 / var, 1e-6))
    return float(np.mean(np.interp(snr, grid, tables[const_family(const)])))


def capacity(snr_db, const: str):
    """AWGN BICM capacity (bits per coded bit) at snr_db, per channel use."""
    grid, tables = _capacity()
    return np.interp(snr_db, grid, tables[const_family(const)])


def const_family(name: str) -> str:
    """A submode's constellation -> the capacity table standing in for it
    (learned 64/256-point sets use the design ones' tables)."""
    if name.startswith("c64"):
        return "c64-snr18"
    if name.startswith("c256"):
        return "c256-snr26"
    return name


@lru_cache(maxsize=None)
def abstraction(path: str = str(DATA / "link_abstraction.json")) -> dict:
    return json.load(open(path))


# --- outcome model: P(decode) learned from real decodes -------------------------------
# codes_data/outcome_predictor.npz (scripts/outcome_data.py, train_outcome.py):
# per submode, P(a next burst of it is usable: synced, header right, first
# codeword decoded) and P(one of its other codewords decodes | usable), from
# the receiver's last two measurements. No link abstraction, clip-noise or
# estimation-loss model in the loop: the real receiver made the labels.

OUTCOME_MODES = tuple(SUBMODES)  # a model file without its own list (the OFDM-only v2)


NOISE_BANDS_HZ = ((350, 950), (950, 1300), (1300, 1750), (1750, 2100), (2100, 2700))  # tnc.NoiseProfile.BANDS_HZ
N_NOISE = 2 * len(NOISE_BANDS_HZ) + 2


@lru_cache(maxsize=None)
def band_span_hz(band: str) -> tuple[float, float]:
    """What a sync band (OFDM band or CPM grid) occupies, Hz."""
    from .. import cpm
    from ..waveform import ofdm

    if band in cpm.GRIDS:
        g = cpm.GRIDS[band]
        return g.f0 - g.bp, g.f0 + (g.m - 1) * g.rate + g.bp
    f = np.sort(ofdm.band(band).freqs)
    half = (f[1] - f[0]) / 2 if len(f) > 1 else 25.0
    return float(f[0] - half), float(f[-1] + half)


def noise_features(noise: dict | None, band: str) -> list[float]:
    """The receiver's noise profile (tnc.NoiseProfile.snapshot()) as model
    inputs: each sub-band's median over the noise in the band `band`'s burst
    was measured in (power-weighted over the sub-bands it overlaps), each
    sub-band's p90/median, log10(1 + impulses per minute), and 1; all 0
    without a profile."""
    if not noise:
        return [0.0] * N_NOISE
    db = np.asarray(noise["noise_db"], dtype=np.float64)
    lo, hi = band_span_hz(band)
    w = np.array([max(0.0, min(hi, b) - max(lo, a)) for a, b in NOISE_BANDS_HZ])
    ref = 10 * np.log10(np.sum(w * 10 ** (db / 10)) / np.sum(w)) if w.sum() > 0 else float(np.median(db))
    return [*(db - ref), *noise["noise_tail_db"], float(np.log10(1 + noise["impulses_per_min"])), 1.0]


def _band_level(db: np.ndarray, band: str) -> float:
    """Noise power over a band's span, power-weighted over the NOISE_BANDS_HZ it overlaps (dB)."""
    lo, hi = band_span_hz(band)
    w = np.array([max(0.0, min(hi, b) - max(lo, a)) for a, b in NOISE_BANDS_HZ])
    if w.sum() <= 0:
        return float(np.median(db))
    return float(10 * np.log10(np.sum(w * 10 ** (db / 10)) / np.sum(w)))


def noise_shift_db(noise: dict | None, measured_band: str, band: str, tail_weight: float = 0.5,
                   deadband_db: float = 1.0) -> float:
    """How much worse a burst in `band` should fare than the one measured in
    `measured_band`, from the receiver's noise profile, as an SNR drop (dB,
    never a gain): the median noise over its span above the measured
    band's, plus tail_weight of how much louder its often-loud moments (p90)
    are beyond that. Under deadband_db: 0 (a flat profile changes nothing)."""
    if not noise:
        return 0.0
    db, tail = np.asarray(noise["noise_db"], float), np.asarray(noise["noise_tail_db"], float)
    med = _band_level(db, band) - _band_level(db, measured_band)
    loud = _band_level(db + tail, band) - _band_level(db + tail, measured_band)
    shift = max(0.0, med) + tail_weight * max(0.0, loud - max(0.0, med))
    return shift if shift >= deadband_db else 0.0


def shifted(measured: dict, shift_db: float) -> dict:
    """The measurements as if the SNR were shift_db lower: snr_est, and each
    MI feature moved along its capacity curve."""
    if not shift_db:
        return measured
    grid, tables = _capacity()
    m = dict(measured, snr_est=measured["snr_est"] - shift_db)
    for c in CONSTS:
        t = tables[const_family(c)]
        snr = np.interp(measured[f"mi_{c}"], t, grid)  # the curve is increasing
        m[f"mi_{c}"] = float(np.interp(snr - shift_db, grid, t))
    return m


def outcome_inputs(measured: dict, band: str, gap: float, seconds: float, prev=None, bands=BANDS,
                   noise: bool = False) -> np.ndarray:
    """measured, prev: as inputs(); `seconds`: the next burst's length on air;
    `bands`: the model's band one-hot (OFDM bands, then CPM grids); `noise`:
    a model with the noise profile's inputs (measured["noise"], noise_features)."""
    x = [measured["snr_est"], np.log(0.05 + measured["spread_est"]), measured["delay_est_ms"]]
    x += [measured[f"mi_{c}"] for c in CONSTS]
    x += [measured.get("headroom", 0.0), np.log2(measured.get("frames", 16))]
    x += [float(band == b) for b in bands]
    pm, pb, age = prev if prev is not None else (measured, band, 0.0)
    x += [float(prev is not None), *(pm[f"mi_{c}"] for c in CONSTS), pm["snr_est"], np.log(0.05 + pm["spread_est"]),
          np.log2(1 + age), float(pb == band), np.log2(pm.get("frames", 16))]
    x += [gap, np.log2(seconds)]
    if noise:
        x += noise_features(measured.get("noise"), band)
    return np.array(x, dtype=np.float64)


@dataclass
class OutcomeMlp:
    mean: np.ndarray
    std: np.ndarray
    layers: list
    modes: tuple = OUTCOME_MODES  # its outputs' order
    bands: tuple = tuple(BANDS)  # its band one-hot
    noise: bool = False  # takes the noise profile's inputs (outcome_inputs)
    noise_lo: np.ndarray | None = None  # its noise inputs clipped to these (clip_noise)
    noise_hi: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """-> (..., 2 * len(OUTCOME_MODES)) logits: burst ok, then codeword ok."""
        if self.noise and self.noise_lo is not None:
            x = clip_noise(x, self.noise_lo, self.noise_hi)
        h = (x - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        return h


@lru_cache(maxsize=None)
def outcome_model(path: str = os.environ.get("DATA2G_OUTCOME_MODEL") or str(DATA / "outcome_predictor.npz")
                  ) -> OutcomeMlp | None:
    """The installed model, or DATA2G_OUTCOME_MODEL's file (studies: two
    models side by side, paired, without swapping the installed one)."""
    p = Path(path)
    if not p.exists():
        return None
    d = np.load(p)
    if "m0_mean" in d.files:  # an ensemble (scripts/train_outcome.py --ensemble)
        members = sorted({k.split("_", 1)[0] for k in d.files})
        return OutcomeEnsemble([_mlp({k.split("_", 1)[1]: d[k] for k in d.files if k.startswith(m + "_")})
                                for m in members])
    return _mlp({k: d[k] for k in d.files})


def clip_noise(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The noise profile's inputs (the last N_NOISE) clipped to [lo, hi], where
    there is a profile (its flag, the last, 1): a model reads interference
    stronger than it trained on as its strongest, not as something to
    extrapolate from (round 3: carriers past the training range took one
    from 359 to 3 bps). Without a profile the inputs stay 0."""
    x = np.array(x, dtype=np.float64)
    has = x[..., -1:] > 0.5
    x[..., -N_NOISE:] = np.where(has, np.clip(x[..., -N_NOISE:], lo, hi), x[..., -N_NOISE:])
    return x


def _mlp(d: dict) -> OutcomeMlp:
    n = sum(1 for k in d if k.startswith("W"))
    extra = {k: tuple(str(v) for v in d[k]) for k in ("modes", "bands") if k in d}  # older files: OFDM only
    clip = {k: d[k] for k in ("noise_lo", "noise_hi") if k in d}
    return OutcomeMlp(d["mean"], d["std"], [(d[f"W{i}"], d[f"b{i}"]) for i in range(n)],
                      noise=bool(d["noise_inputs"]) if "noise_inputs" in d else False, **clip, **extra)


# DATA2G_OUTCOME_LCB=k (studies): an ensemble's logits minus k times its
# members' standard deviation, a lower confidence bound. The shifter picks
# the best of many noisy predictions, so its pick is the one most likely
# over-predicted (on explored rows v10 is near calibrated; on its own picks
# it is not); this discounts where the members disagree, in any channel,
# instead of per-mode constants (LOGIT_OFFSETS).
LCB = float(os.environ.get("DATA2G_OUTCOME_LCB") or 0)


@dataclass
class OutcomeEnsemble:
    """Bootstrap members; their probabilities averaged (returned as logits).
    Where the data are thin the members disagree, and the average stays
    away from 0 and 1 (single models put 0.9+ on contradictory inputs)."""
    members: list

    @property
    def modes(self):
        return self.members[0].modes

    @property
    def bands(self):
        return self.members[0].bands

    @property
    def noise(self):
        return self.members[0].noise

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = np.array([np.clip(m(x), -40, 40) for m in self.members])
        p = np.clip(np.mean(1 / (1 + np.exp(-z)), axis=0), 1e-9, 1 - 1e-9)
        out = np.log(p / (1 - p))
        if LCB:
            out = out - LCB * np.std(z, axis=0)
        return out


def outcome_knows(submode: str) -> bool:
    m = outcome_model()
    return m is not None and submode in m.modes


# Logit offsets on P(burst usable), per submode, for the installed model.
# v7 needed six (2026-09-28), but they were a patch for a coverage loop:
# they kept modes out of the sessions later models trained on, so those
# never saw them fail, and they cost AWGN 0 dB 32% (16qam-r1/3 never
# picked). v12 keeps one: w48-16qam-r1/2 at MPG +8 dB, +6.7% (10/2 seeds),
# nothing elsewhere (old README in git history, outcome model v12).
# n10-256l-r3/4 (v12 + the n10 extension): tried below its threshold at AWGN
# 20 dB, BW500 (speedtrials); -0.5 trades some of 25 dB for it (the user's call, 2026-09-29)
LOGIT_OFFSETS = {"w48-16qam-r1/2": -1.0, "n10-256l-r3/4": -0.5}
# DATA2G_LOGIT_OFFSETS="mode:logit,..." (studies): this table instead, for
# whatever model is loaded ("" = none); unset, LOGIT_OFFSETS apply to the
# installed model only.
_ENV_OFFSETS = os.environ.get("DATA2G_LOGIT_OFFSETS")
if _ENV_OFFSETS is not None:
    LOGIT_OFFSETS = {m: float(v) for m, v in (e.rsplit(":", 1) for e in _ENV_OFFSETS.split(",") if e)}


def predict_outcome(measured: dict, band: str, gap: float, seconds: float, submodes=None,
                    prev=None) -> dict[str, tuple[float, float]]:
    """-> {submode: (P(burst usable), P(codeword decodes | usable))} for a
    next burst `seconds` long."""
    model = outcome_model()
    z = model(outcome_inputs(measured, band, gap, seconds, prev, model.bands, model.noise))
    n, idx = len(model.modes), {m: i for i, m in enumerate(model.modes)}
    if LOGIT_OFFSETS and (_ENV_OFFSETS is not None or not os.environ.get("DATA2G_OUTCOME_MODEL")):
        z = z.copy()
        for m, off in LOGIT_OFFSETS.items():
            if m in idx:
                z[idx[m]] += off
    p = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
    return {s.name: (float(p[idx[s.name]]), float(p[n + idx[s.name]]))
            for s in (submodes or SUBMODES.values()) if s.name in idx}  # a model knows the modes it was trained on
