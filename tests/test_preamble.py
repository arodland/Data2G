"""The preamble and the matched-filter detector (sync.detection_stat).

Deliberately not statistical: detection margins are measured by
scripts/mf_detect_study.py and scripts/sync_floor.py, not asserted here.
"""

import numpy as np
import pytest

from data2g.config import FS, M, PREAMBLE_CP
from data2g.hfchannel import freq_shift
from data2g.waveform import ofdm, sync
from data2g.waveform.dsp import to_baseband


def test_the_preamble_is_periodic_with_M_throughout():
    """Every repeat must be the same symbol, including the double-length
    CP: the detector sums matched-filter power over repeats, and the
    header's channel reference averages them."""
    b = ofdm.band("n10")
    w = b.preamble_waveform()
    assert len(w) == b.spec.preamble_samples == PREAMBLE_CP + b.spec.preamble_repeats * M
    body = w[PREAMBLE_CP:]
    first = body[:M]
    for r in range(1, b.spec.preamble_repeats):
        np.testing.assert_allclose(body[r * M : (r + 1) * M], first, atol=1e-9)
    np.testing.assert_allclose(w[:PREAMBLE_CP], first[-PREAMBLE_CP:], atol=1e-9)


def test_noise_statistic_is_band_independent():
    """Over white noise the normalized repeat outputs are ~CN(0, 1), so
    |sum of R-1 neighbour products| has mean ~sqrt(pi (R-1)) / 2 on every
    band: one threshold serves them all."""
    z = to_baseband(np.random.default_rng(0).normal(size=4 * FS))
    for name in ("w", "n10"):
        S, _ = sync.detection_stat(z, ofdm.band(name))
        expect = np.sqrt(np.pi * (ofdm.band(name).spec.preamble_repeats - 1)) / 2
        assert abs(S.mean() / expect - 1) < 0.15, (name, S.mean())


@pytest.mark.parametrize("band", ["w", "n10", "w48"])
@pytest.mark.parametrize("offset_hz", [0.0, 6.0, -37.5, 143.0])
def test_acquire_finds_a_noisy_preamble_exactly(band, offset_hz):
    """Timing to the sample (a few on the narrow bands, whose correlation
    main lobe is FS / bandwidth wide) and CFO to a fraction of a Hz, on
    every band, off the CFO grid and several bins out."""
    b = ofdm.band(band)
    lead = 3000
    x = np.concatenate([np.zeros(lead), b.preamble_waveform(), np.zeros(3000)])
    x = freq_shift(x, offset_hz)
    x = x + np.random.default_rng(1).normal(scale=0.05 * np.std(x), size=len(x))
    acq = sync.acquire(to_baseband(x), band=b)
    assert abs(acq.preamble_start - lead) <= (1 if b.spec.nc >= 24 else 4)
    assert abs(acq.freq_offset - offset_hz) < 0.5
    assert acq.metric > b.spec.preamble_threshold


def test_noise_alone_is_not_a_preamble():
    """Pure noise: nothing crosses the threshold (a 4 s buffer; the
    threshold sits above the 1200 s noise peak)."""
    z = to_baseband(np.random.default_rng(2).normal(size=4 * FS))
    with pytest.raises(sync.SyncError, match="no preamble"):
        sync.acquire(z)


def test_search_reaches_150_hz_and_no_further():
    from data2g.config import ACQUIRE_REACH_HZ
    from data2g.waveform import sync

    g = sync._cfo_grid()
    assert ACQUIRE_REACH_HZ == 150 and g.min() == -150 and g.max() == 150
