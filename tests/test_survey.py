import numpy as np

from data2g import modem
from data2g.config import FS
from data2g.survey import survey


def test_survey_finds_a_carrier_only_the_wide_band_overlaps():
    """White noise plus a steady carrier at 2400 Hz, inside w48 only, and
    a Data2G burst in the middle: the burst is not noise, the carrier is."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=6 * FS)
    t = np.arange(len(x)) / FS
    x += 0.3 * np.cos(2 * np.pi * 2400 * t)
    b = modem.modulate([bytes(4)], "ack-1f")
    x[2 * FS : 2 * FS + len(b)] += 10 * b
    s = survey(x)
    # the worst of 24 carriers' floor estimates (~0.55 dB each) reads ~1 dB high on its own
    assert abs(s.band_excess_db("w")) < 2.0
    assert abs(s.band_excess_db("n10")) < 2.0
    assert s.band_excess_db("w48") > 6.0  # measured 8.9
    assert s.band_busy("w") > 0.02  # the burst shows as traffic, not as floor
