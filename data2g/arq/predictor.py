"""Link predictor (gear-shifter phase C): P(a codeword of submode m in
the next burst decodes | what the receiver measured on this burst).

Two parts:
- link abstraction (codes_data/link_abstraction.json, scripts/link_abstraction.py):
  P(decode) = sigmoid(slope_m (MI - MI50_m)), MI the codeword's effective
  mutual information, one curve per submode for every channel (within
  ~0.02 MI for most);
- a tiny MLP (codes_data/link_predictor.npz, scripts/train_predictor.py)
  giving the next burst's effective MI per (band, constellation) as a
  Gaussian (mean, std) from this burst's measurements. The mean is this
  burst's MI for the same constellation plus a learned correction in
  logit space, so persistence is the zero-effort answer. The std is what
  makes fast fading predict wide and slow fading narrow.
P(decode) = E over that Gaussian of the sigmoid (probit approximation).

Plain numpy at runtime: no torch. The same network exports to ONNX
(scripts/train_predictor.py --onnx), per the rule for anything shipped:
small enough for CPU, exportable to ONNX or LiteRT.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import SUBMODES

DATA = Path(__file__).parent.parent / "codes_data"
BANDS = ("w", "n10", "n4", "w48")
CONSTS = ("gray-qam4", "gray-qam16", "c64-snr18", "c256-snr26")
INPUTS = ("snr_est", "log_spread", "delay_est_ms", *(f"mi_{c}" for c in CONSTS), *(f"band_{b}" for b in BANDS),
          "gap", "log_frames", "headroom", "log_window",
          "has_prev", *(f"prev_mi_{c}" for c in CONSTS), "prev_snr_est", "prev_log_spread", "prev_log_age",
          "prev_same_band", "prev_log_frames")
TARGETS = tuple(f"next_{b}_{c}" for b in BANDS for c in CONSTS)
# per target: the input column holding this burst's MI for its
# constellation (the persistence skip in Mlp)
MI_COLS = [INPUTS.index(f"mi_{t.split('_', 2)[2]}") for t in TARGETS]


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


def inputs(measured: dict, band: str, gap: float, window: float = 16, prev=None) -> np.ndarray:
    """measured: snr_est, spread_est, delay_est_ms, mi_<const> (effective MI
    against thermal noise and estimation error only: the transmitter's clip
    noise belongs to the submode measured, not to the channel), frames
    (data frames the measurement spans: a 1-frame reply says less than a
    16-frame burst), headroom (the measured submode's clip headroom: wide
    bands estimate noise from pilots that carry its clip distortion);
    `window`: data frames of the burst being predicted (its MI averages over
    them, and a long one reaches further into a slow fade);
    `prev`: (measured, band, age s) of the peer burst before this one, or
    None: two bursts tell a steady channel (average them) from a slowly
    fading one (their difference), which one burst cannot."""
    x = [measured["snr_est"], np.log(0.05 + measured["spread_est"]), measured["delay_est_ms"]]
    x += [measured[f"mi_{c}"] for c in CONSTS]
    x += [float(band == b) for b in BANDS]
    x += [gap, np.log2(measured.get("frames", 16)), measured.get("headroom", 0.0), np.log2(window)]
    pm, pb, age = prev if prev is not None else (measured, band, 0.0)
    x += [float(prev is not None), *(pm[f"mi_{c}"] for c in CONSTS), pm["snr_est"],
          np.log(0.05 + pm["spread_est"]), np.log2(1 + age), float(pb == band), np.log2(pm.get("frames", 16))]
    return np.array(x, dtype=np.float64)


@dataclass
class Mlp:
    mean: np.ndarray
    std: np.ndarray
    layers: list  # [(W, b), ...]; tanh between, linear last

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = (x - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        n = len(TARGETS)
        m1 = np.clip(x[..., MI_COLS], 1e-3, 1 - 1e-3)
        z = np.log(m1 / (1 - m1)) + h[..., :n]  # persistence + learned correction, logit space
        mu = 1 / (1 + np.exp(-z))  # MI lives in (0, 1)
        sigma = np.exp(np.clip(h[..., n:], -6, 1))
        return mu, sigma


@lru_cache(maxsize=None)
def model(path: str = str(DATA / "link_predictor.npz")) -> Mlp:
    d = np.load(path)
    n = sum(1 for k in d.files if k.startswith("W"))
    return Mlp(d["mean"], d["std"], [(d[f"W{i}"], d[f"b{i}"]) for i in range(n)])


@lru_cache(maxsize=None)
def abstraction(path: str = str(DATA / "link_abstraction.json")) -> dict:
    return json.load(open(path))


@lru_cache(maxsize=None)
def _offsets():
    return json.load(open(DATA / "mi_offsets.json"))


def estimation_loss(band: str, doppler: float) -> float:
    """MI the receiver's channel estimate loses against the true channel's
    (codes_data/mi_offsets.json, scripts/linksim.py calibrate), in Doppler."""
    d = _offsets()
    names = ["awgn", "mpg", "mpp", "mpd"]
    return float(np.interp(doppler, [d["doppler"][c] for c in names],
                           [d["offsets"][f"{band}|{c}"] for c in names]))


def mode_mi(channel_mi, spec, doppler: float):
    """The effective MI a submode's decoder would see, from the channel's own
    MI for its constellation: to an equivalent SNR, plus the submode's clip
    noise, back to MI, less the receiver's estimation loss. The link
    abstraction's curves are in this MI."""
    from ..config import clip_consts

    grid, tables = _capacity()
    tab = tables[const_family(spec.constellation)]
    mi = np.clip(np.asarray(channel_mi, dtype=float), tab[0] + 1e-6, tab[-1] - 1e-6)
    snr = 10 ** (np.interp(mi, tab, grid) / 10)
    ratio = clip_consts(spec.band, spec.headroom)[2]
    eff = np.interp(10 * np.log10(1 / (1 / snr + ratio)), grid, tab)
    return np.maximum(0.0, eff - estimation_loss(spec.band, doppler))


def p_decode(mu: float, sigma: float, slope: float, mi50: float) -> float:
    """E[sigmoid(slope (MI - mi50))], MI ~ N(mu, sigma): the probit
    approximation sigmoid(slope (mu - mi50) / sqrt(1 + pi slope^2 sigma^2 / 8))."""
    z = slope * (mu - mi50) / np.sqrt(1 + np.pi * slope**2 * sigma**2 / 8)
    return float(1 / (1 + np.exp(-z)))


def predict(measured: dict, band: str, gap: float, submodes=None, window: float = 16, prev=None) -> dict[str, float]:
    """-> {submode name: P(a codeword decodes on the next burst)}. The
    channel MI's Gaussian goes through mode_mi at mu and mu +- sigma (the
    mapping is monotone), then the probit approximation."""
    mu, sigma = model()(inputs(measured, band, gap, window, prev))
    ab = abstraction()
    dop = measured["spread_est"]
    out = {}
    for s in (submodes or SUBMODES.values()):
        j = TARGETS.index(f"next_{s.band}_{const_family(s.constellation)}")
        lo, m, hi = mode_mi([mu[j] - sigma[j], mu[j], mu[j] + sigma[j]], s, dop)
        c = ab[s.name]
        out[s.name] = p_decode(float(m), float(max(hi - lo, 1e-6) / 2), c["slope"], c["mi50"])
    return out


# --- outcome model: P(decode) learned from real decodes -------------------------------
# codes_data/outcome_predictor.npz (scripts/outcome_data.py, train_outcome.py):
# per submode, P(a next burst of it is usable: synced, header right, first
# codeword decoded) and P(one of its other codewords decodes | usable), from
# the receiver's last two measurements. No link abstraction, clip-noise or
# estimation-loss model in the loop: the real receiver made the labels.

OUTCOME_MODES = tuple(SUBMODES)  # a model file without its own list (the OFDM-only v2)
OUTCOME_INPUTS = ("snr_est", "log_spread", "delay_est_ms", *(f"mi_{c}" for c in CONSTS), "headroom", "log_frames",
                  *(f"band_{b}" for b in BANDS), "has_prev", *(f"prev_mi_{c}" for c in CONSTS), "prev_snr_est",
                  "prev_log_spread", "prev_log_age", "prev_same_band", "prev_log_frames", "gap", "log_seconds")


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
def outcome_model(path: str = str(DATA / "outcome_predictor.npz")) -> OutcomeMlp | None:
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
        p = np.mean([1 / (1 + np.exp(-np.clip(m(x), -40, 40))) for m in self.members], axis=0)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))


def outcome_knows(submode: str) -> bool:
    m = outcome_model()
    return m is not None and submode in m.modes


def predict_outcome(measured: dict, band: str, gap: float, seconds: float, submodes=None,
                    prev=None) -> dict[str, tuple[float, float]]:
    """-> {submode: (P(burst usable), P(codeword decodes | usable))} for a
    next burst `seconds` long."""
    model = outcome_model()
    z = model(outcome_inputs(measured, band, gap, seconds, prev, model.bands))
    n, idx = len(model.modes), {m: i for i, m in enumerate(model.modes)}
    p = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
    return {s.name: (float(p[idx[s.name]]), float(p[n + idx[s.name]])) for s in (submodes or SUBMODES.values())}
