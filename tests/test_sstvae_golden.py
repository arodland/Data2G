"""The waveform is SSTVAE's, sample for sample. Skipped without a
checkout at ~/code/SSTVAE (or $SSTVAE_PATH)."""

import os
import sys

import numpy as np
import pytest

path = os.environ.get("SSTVAE_PATH", os.path.expanduser("~/code/SSTVAE"))
if not os.path.isdir(os.path.join(path, "sstvae")):
    pytest.skip("no SSTVAE checkout", allow_module_level=True)
sys.path.insert(0, path)
from sstvae.modem import ofdm as s_ofdm  # noqa: E402
from sstvae.modem import dsp as s_dsp  # noqa: E402

from data2g.config import CLIP_HEADROOM_DB, NC  # noqa: E402
from data2g.waveform import dsp, ofdm  # noqa: E402


def test_pilot_and_preamble_identical():
    """Same pilot; the preamble is the same symbol, repeated 8 times here
    instead of 4 (config.PREAMBLE_REPEATS), so its tail matches SSTVAE's."""
    assert np.array_equal(ofdm.pilot_sequence(), s_ofdm.pilot_sequence())
    ours, theirs = ofdm.preamble_waveform(), s_ofdm.preamble_waveform()
    np.testing.assert_allclose(ours[-len(theirs):], theirs, atol=1e-12)


def test_symbols_and_clipper_identical(reference):
    rng = np.random.default_rng(0)
    s = rng.normal(size=(12, NC)) + 1j * rng.normal(size=(12, NC))
    x = reference(ofdm, "modulate_symbols")(s)
    assert np.array_equal(x, s_ofdm.modulate_symbols(s))
    want = s_dsp.tx_condition(x, CLIP_HEADROOM_DB)
    assert np.array_equal(reference(dsp, "tx_condition")(x, CLIP_HEADROOM_DB), want)
    # what is substituted (C++ under --native) sums in its own order: to rounding
    np.testing.assert_allclose(ofdm.modulate_symbols(s), x, rtol=0, atol=1e-13)
    np.testing.assert_allclose(dsp.tx_condition(x, CLIP_HEADROOM_DB), want, rtol=0, atol=1e-12)
