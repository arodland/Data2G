"""data2g.interference: the session sim's interference."""

import numpy as np
import pytest

from data2g import interference as I
from data2g.config import FS
from data2g.tnc import NoiseProfile

SIGMA = 0.01


def band_power(y, lo, hi):
    Y = np.abs(np.fft.rfft(y)) ** 2 / len(y) ** 2 * 2
    f = np.fft.rfftfreq(len(y), 1 / FS)
    return Y[(f >= lo) & (f < hi)].sum()


def test_clean_is_silent():
    assert not I.Interference(I.Spec(), 1, 60).render(3.0, FS, SIGMA).any()


@pytest.mark.parametrize("spec", [
    I.Spec(I.Impulses(30.0)),
    I.Spec(qrm=(I.Qrm("fsk", 1200, 300, 15, 2.0, 0.5), I.Qrm("noise", 2300, 400, 10, 1.0, 0.3),
                I.Qrm("carrier", 700, 50, 20, 3.0, 0.6))),
])
def test_any_span_renders_the_same(spec):
    """Random access: a span rendered alone is that span of a longer render."""
    it = I.Interference(spec, 7, 60)
    whole = it.render(10.0, 10 * FS, SIGMA)
    for a, n in ((0, FS // 3), (12345, 4567), (5 * FS + 17, 3 * FS)):
        np.testing.assert_allclose(it.render(10.0 + a / FS, n, SIGMA), whole[a:a + n], atol=1e-12)


@pytest.mark.parametrize("kind, bw", [("noise", 500.0), ("fsk", 400.0), ("carrier", 50.0)])
def test_inr_is_over_the_floor_in_its_band(kind, bw):
    """Always on (duty 1): its power over the floor's in its own band (a
    carrier: in CARRIER_REF_HZ) is inr_db."""
    it = I.Interference(I.Spec(qrm=(I.Qrm(kind, 1500, bw, 12.0, 1.0, 1.0),)), 3, 20)
    y = it.render(2.0, 8 * FS, SIGMA)
    floor = SIGMA**2 / (FS / 2) * (I.CARRIER_REF_HZ if kind == "carrier" else bw)
    got = 10 * np.log10(band_power(y, 1500 - bw, 1500 + bw) / floor)
    assert got == pytest.approx(12.0, abs=0.7)


def test_duty_and_episodes():
    it = I.Interference(I.Spec(qrm=(I.Qrm("carrier", 1000, 50, 20, 2.0, 0.3),)), 4, 3000)
    starts, ends = it._episodes[0]
    on = (ends - starts).sum() / (3000 * FS)
    assert on == pytest.approx(0.3, abs=0.05)
    assert np.mean(ends - starts) / FS == pytest.approx(2.0, rel=0.2)


def test_impulses_look_like_the_air_to_the_noise_profile():
    """Trains of impulses clearly over its 20 dB threshold: NoiseProfile
    counts about as many as were scheduled in the window it kept, and its
    band medians barely move."""
    spec = I.Spec(I.Impulses(trains_per_min=20.0, per_train=4.0, height_db=30.0, height_sd_db=2.0))
    it = I.Interference(spec, 5, 200)
    x = np.random.default_rng(6).normal(0, SIGMA, 120 * FS) + it.render(0.0, 120 * FS, SIGMA)
    p = NoiseProfile()
    for i in range(0, len(x), FS // 25):
        p.feed(x[i:i + FS // 25], i / FS)
    s = p.snapshot()
    kept = s["noise_blocks"] * NoiseProfile.BLOCK  # the last kept blocks end COMMIT_S before the end
    end = 120 * FS - int(NoiseProfile.COMMIT_S * FS)
    scheduled = np.sum((it._imp[:, 0] >= end - kept) & (it._imp[:, 0] < end))
    assert s["impulses_per_min"] == pytest.approx(scheduled * 60 * FS / kept, rel=0.3)
    assert np.ptp(s["noise_db"]) < 1.0


def test_draw_is_in_range():
    rng = np.random.default_rng(8)
    specs = [I.draw(rng) for _ in range(400)]
    assert 0.25 < np.mean([s.clean for s in specs]) < 0.45
    for s in specs:
        for q in s.qrm:
            assert I.PASSBAND_HZ[0] <= q.f_hz - q.bw_hz / 2 and q.f_hz + q.bw_hz / 2 <= I.PASSBAND_HZ[1]
            assert I.INR_DB[0] <= q.inr_db <= I.INR_DB[1] and I.DUTY[0] <= q.duty <= I.DUTY[1]
