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


def outcome_inputs(measured: dict, band: str, gap: float, seconds: float, prev=None, bands=BANDS) -> np.ndarray:
    """measured, prev: as inputs(); `seconds`: the next burst's length on air;
    `bands`: the model's band one-hot (OFDM bands, then CPM grids)."""
    x = [measured["snr_est"], np.log(0.05 + measured["spread_est"]), measured["delay_est_ms"]]
    x += [measured[f"mi_{c}"] for c in CONSTS]
    x += [measured.get("headroom", 0.0), np.log2(measured.get("frames", 16))]
    x += [float(band == b) for b in bands]
    pm, pb, age = prev if prev is not None else (measured, band, 0.0)
    x += [float(prev is not None), *(pm[f"mi_{c}"] for c in CONSTS), pm["snr_est"], np.log(0.05 + pm["spread_est"]),
          np.log2(1 + age), float(pb == band), np.log2(pm.get("frames", 16))]
    x += [gap, np.log2(seconds)]
    return np.array(x, dtype=np.float64)


@dataclass
class OutcomeMlp:
    mean: np.ndarray
    std: np.ndarray
    layers: list
    modes: tuple = OUTCOME_MODES  # its outputs' order
    bands: tuple = tuple(BANDS)  # its band one-hot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """-> (..., 2 * len(OUTCOME_MODES)) logits: burst ok, then codeword ok."""
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


def _mlp(d: dict) -> OutcomeMlp:
    n = sum(1 for k in d if k.startswith("W"))
    extra = {k: tuple(str(v) for v in d[k]) for k in ("modes", "bands") if k in d}  # older files: OFDM only
    return OutcomeMlp(d["mean"], d["std"], [(d[f"W{i}"], d[f"b{i}"]) for i in range(n)], **extra)


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
LOGIT_OFFSETS = {"w48-16qam-r1/2": -1.0}
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
    z = model(outcome_inputs(measured, band, gap, seconds, prev, model.bands))
    n, idx = len(model.modes), {m: i for i, m in enumerate(model.modes)}
    if LOGIT_OFFSETS and (_ENV_OFFSETS is not None or not os.environ.get("DATA2G_OUTCOME_MODEL")):
        z = z.copy()
        for m, off in LOGIT_OFFSETS.items():
            if m in idx:
                z[idx[m]] += off
    p = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
    return {s.name: (float(p[idx[s.name]]), float(p[n + idx[s.name]])) for s in (submodes or SUBMODES.values())}
