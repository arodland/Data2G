"""The noise profile as outcome-model inputs (predictor.noise_features)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from data2g.arq import predictor as P

NOISE = {"noise_db": [10.0, 0.0, 0.0, 0.0, 13.0], "noise_tail_db": [1.0, 1.5, 2.0, 2.5, 9.0], "impulses_per_min": 99.0}


def measured(snr=3.0, noise=None):
    m = dict(snr_est=snr, spread_est=0.3, delay_est_ms=0.5, headroom=0.0, frames=16)
    m.update({f"mi_{c}": 0.5 for c in P.CONSTS})
    if noise is not None:
        m["noise"] = noise
    return m


def test_no_profile_is_zeros():
    assert P.noise_features(None, "w") == [0.0] * P.N_NOISE


@pytest.mark.parametrize("band", ["w", "c16r25", "w48", "n4"])
def test_levels_are_relative_to_the_measured_band(band):
    """Each sub-band's level over the noise where the burst was measured: the
    power-weighted mean of the sub-bands its span overlaps (c16r25, 1250-1725
    Hz, sees only the quiet middle: the edges read +10 and +13 over it)."""
    f = P.noise_features(NOISE, band)
    lo, hi = P.band_span_hz(band)
    w = np.array([max(0, min(hi, b) - max(lo, a)) for a, b in P.NOISE_BANDS_HZ])
    ref = 10 * np.log10(np.sum(w * 10 ** (np.array(NOISE["noise_db"]) / 10)) / w.sum())
    np.testing.assert_allclose(f[:5], np.array(NOISE["noise_db"]) - ref, atol=1e-9)
    if band == "c16r25":
        assert ref == pytest.approx(0.0)
    assert f[5:10] == NOISE["noise_tail_db"] and f[10] == pytest.approx(2.0) and f[11] == 1.0


def test_inputs_grow_only_for_a_noise_model(reference):
    inputs = reference(P, "outcome_inputs")
    base = inputs(measured(noise=NOISE), "w", 2.5, 6.0)
    ext = inputs(measured(noise=NOISE), "w", 2.5, 6.0, noise=True)
    assert len(ext) == len(base) + P.N_NOISE and np.array_equal(ext[:len(base)], base)


def test_a_noise_model_predicts_with_and_without_a_profile(tmp_path, monkeypatch, reference):
    """A model file flagged noise_inputs takes the extended inputs; a burst
    with no profile yet reads as 'no profile', not an error. (Python's: the
    C++ predictor has the installed model compiled in.)"""
    monkeypatch.setattr(P, "outcome_inputs", reference(P, "outcome_inputs"))
    monkeypatch.setattr(P, "predict_outcome", reference(P, "predict_outcome"))
    n_in = len(P.outcome_inputs(measured(), "w", 2.5, 6.0, noise=True))
    modes = P.OUTCOME_MODES
    rng = np.random.default_rng(1)
    path = tmp_path / "m.npz"
    np.savez(path, mean=np.zeros(n_in), std=np.ones(n_in), modes=np.array(modes), bands=np.array(P.BANDS),
             noise_inputs=np.array(True), W0=rng.normal(0, 0.1, (n_in, 8)), b0=np.zeros(8),
             W1=rng.normal(0, 0.1, (8, 2 * len(modes))), b1=np.zeros(2 * len(modes)))
    P.outcome_model.cache_clear()
    monkeypatch.setattr(P, "outcome_model", lambda p=str(path): P._mlp(dict(np.load(path))))
    assert P.outcome_model().noise
    a = P.predict_outcome(measured(noise=NOISE), "w", 2.5, 6.0)
    b = P.predict_outcome(measured(), "w", 2.5, 6.0)
    assert set(a) == set(b) and a != b


def test_training_rows_carry_the_profile():
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import train_outcome as T

    row = {f"noise_db{i}": str(v) for i, v in enumerate(NOISE["noise_db"], 1)}
    row.update({f"noise_tail{i}": str(v) for i, v in enumerate(NOISE["noise_tail_db"], 1)})
    row["impulses_per_min"] = "99.0"
    assert T.noise_of(row) == NOISE
    assert T.noise_of({"noise_db1": ""}) is None
