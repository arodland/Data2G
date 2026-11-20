"""data2g.interference: the session sim's interference."""

import numpy as np
import pytest

from data2g import interference as I
from data2g.config import FS
from data2g.receiver import NoiseProfile

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
    I.Spec(qrm=(I.Qrm("noise", 1500, 300, 10, 3.0, 0.7, span_hz=2000, sweep_s=6.0),)),
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


def test_swept_noise_is_narrow_and_moves():
    """A swept noise source: about inr_db over the floor in its own (instantaneous) band,
    centred where the sine puts it, and it covers its span over a period."""
    q = I.Qrm("noise", 1500, 200, 12.0, 1.0, 1.0, span_hz=2000, sweep_s=8.0)
    it = I.Interference(I.Spec(qrm=(q,)), 3, 40)
    cents, pw = [], []
    for k in range(32):  # quarter-second looks over one period
        y = it.render(2.0 + k * 0.25, FS // 4, SIGMA)
        Y = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / FS)
        cents.append(np.sum(f * Y) / np.sum(Y))
        pw.append(np.mean(y**2))
    assert min(cents) < 700 and max(cents) > 2300  # 1500 -+ (2000 - 200) / 2
    floor = SIGMA**2 / (FS / 2) * q.bw_hz
    assert 10 * np.log10(np.mean(pw) / floor) == pytest.approx(12.0, abs=0.7)


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


@pytest.mark.parametrize("draws", sorted(I.DRAWS))
def test_draw_is_in_range(draws):
    r = I.DRAWS[draws]
    rng = np.random.default_rng(8)
    specs = [I.draw(rng, draws) for _ in range(400)]
    assert abs(np.mean([s.clean for s in specs]) - r.p_clean) < 0.1
    for s in specs:
        assert len(s.qrm) <= r.qrm_max
        for q in s.qrm:
            k = (r.qrm or {}).get(q.kind, {})
            inr, duty, on = k.get("inr_db", r.inr_db), k.get("duty", r.duty), k.get("on_s", r.on_s)
            width = max(q.bw_hz, q.span_hz)
            assert I.PASSBAND_HZ[0] <= q.f_hz - width / 2 and q.f_hz + width / 2 <= I.PASSBAND_HZ[1] + 1e-9
            assert inr[0] <= q.inr_db <= inr[1] and duty[0] <= q.duty <= duty[1] and on[0] <= q.on_s <= on[1]
            if q.span_hz > q.bw_hz:
                assert k["inst_bw_hz"][0] <= q.bw_hz <= k["inst_bw_hz"][1] and q.span_hz >= 1.5 * q.bw_hz
            elif "bw_hz" in k:
                assert k["bw_hz"][0] <= q.bw_hz <= k["bw_hz"][1]


def test_old_draws_reproduce():
    """v1 and mild draw what they drew before onair was added (earlier studies' data reproduces)."""
    import hashlib

    def key(s):
        im = s.impulses
        return repr(((im.trains_per_min, im.per_train, im.gap_ms, im.height_db, im.height_sd_db, im.length_ms)
                     if im else None, [(q.kind, q.f_hz, q.bw_hz, q.inr_db, q.on_s, q.duty) for q in s.qrm]))

    for draws, digest in (("v1", "f17b414daa66df17"), ("mild", "8036fd0b0e05481f")):
        rng = np.random.default_rng(123)
        specs = [I.draw(rng, draws) for _ in range(500)]
        assert hashlib.sha256("".join(map(key, specs)).encode()).hexdigest()[:16] == digest
        assert I.DRAWS[draws].wander_db == 0 and not any(q.span_hz for s in specs for q in s.qrm)


def test_onair_draw_shape():
    """onair: impulses mostly alone, a few loud draws; FSK hoppers the main QRM kind, carriers ~10%;
    noise mostly narrow and swept; passbands in their shares (ssb 0.85, narrow 0.05, flat the rest)."""
    rng = np.random.default_rng(9)
    specs = [I.draw(rng, "onair") for _ in range(2000)]
    imps = [s.impulses for s in specs if s.impulses]
    assert all(1.0 <= im.per_train <= 1.5 and im.height_sd_db == 2.0 for im in imps)
    loud = np.mean([im.height_db > 17 for im in imps])
    assert 0.005 < loud < 0.05
    kinds = [q.kind for s in specs for q in s.qrm]
    share = {k: kinds.count(k) / len(kinds) for k in I.KINDS}
    assert share["fsk"] > 0.5 and 0.05 < share["carrier"] < 0.12
    noise = [q for s in specs for q in s.qrm if q.kind == "noise"]
    assert np.mean([q.span_hz > q.bw_hz for q in noise]) == pytest.approx(0.7, abs=0.06)
    pbs = [I.draw_passband(rng) for _ in range(4000)]
    assert sum(str(p).startswith("ssb:") for p in pbs) / 4000 == pytest.approx(0.85, abs=0.02)
    assert pbs.count("narrow") / 4000 == pytest.approx(0.05, abs=0.015)
    assert pbs.count(None) / 4000 == pytest.approx(0.10, abs=0.015)
    assert len({p for p in pbs if p}) > 3000  # each station its own filter


def test_ssb_passband_draw():
    """draw_ssb: -6 dB edges at 150-300 Hz and 2.7-3.1 kHz apart; w48 (325-2725 Hz) above -6 dB, and from
    450 Hz up within the tilt (+-0.75 dB); -40 dB within 300 Hz of the top edge."""
    rng = np.random.default_rng(4)
    for _ in range(500):
        t = np.array(I.passband_table(I.draw_ssb(rng)), dtype=float)
        assert np.all(np.diff(t[:, 0]) >= 0)
        f = np.linspace(100, 3900, 3801)
        g = np.interp(f, t[:, 0], t[:, 1])
        lo, hi = f[np.argmax(g >= -6)], f[len(g) - 1 - np.argmax(g[::-1] >= -6)]
        assert 150 <= lo <= 301 and 2700 <= hi - lo <= 3101
        assert g[(f >= 325) & (f <= 2725)].min() > -6
        w48 = g[(f >= 450) & (f <= 2725)]
        assert w48.min() > -0.76 and w48.max() < 0.76
        assert np.interp(hi + 300, t[:, 0], t[:, 1]) <= -40 + 1e-9


@pytest.mark.parametrize("name", sorted(I.PASSBANDS) + ["ssb:220:3050:100:200:1.2"])
def test_passband_shape(name):
    """White noise through a passband: its gain per band is the table's (linear in dB between points)."""
    x = np.random.default_rng(2).normal(size=60 * FS)
    y = I.passband(x, name)
    table = np.array(I.passband_table(name), dtype=float)
    for f in (300, 1000, 2000, 2400, 2700):
        got = 10 * np.log10(band_power(y, f - 20, f + 20) / band_power(x, f - 20, f + 20))
        assert got == pytest.approx(np.interp(f, table[:, 0], table[:, 1]), abs=0.6)
    assert I.passband(x, None) is x


def _phy_session():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import phy_session

    return phy_session


def test_clean_channel_hears_as_before():
    """No interference: a burst through the channel is what it was without
    the receiver argument (the studies' clean cells reproduce)."""
    PS = _phy_session()
    x = np.random.default_rng(1).normal(0, 0.1, 4000)
    a, b = PS.ContinuousChannel("mpg", 3.0, 9, 60), PS.ContinuousChannel("mpg", 3.0, 9, 60)
    np.testing.assert_array_equal(a.apply(x, 5.0), b.apply(x, 5.0, rx=1))


def test_passband_is_per_station_on_signal_and_noise():
    """Station 1's narrow passband takes a 2700 Hz tone and the floor there down
    alike (in-band SNR unchanged); station 0 (flat) hears what a channel without
    passbands gives; the passband reaches the station's noise profile too."""
    PS = _phy_session()
    x = np.cos(2 * np.pi * 2700 * np.arange(2 * FS) / FS) * 0.1
    flat = PS.ContinuousChannel("awgn", 20.0, 5, 60)
    ch = PS.ContinuousChannel("awgn", 20.0, 5, 60, passband=(None, "narrow"))
    np.testing.assert_array_equal(ch.apply(x, 5.0, rx=0), flat.apply(x, 5.0, rx=0))
    ch = PS.ContinuousChannel("awgn", 20.0, 5, 60, passband=(None, "narrow"))
    a, b = flat.apply(x, 5.0, rx=1), ch.apply(x, 5.0, rx=1)
    assert 10 * np.log10(band_power(b, 2690, 2710) / band_power(a, 2690, 2710)) == pytest.approx(-40, abs=1)
    assert 10 * np.log10(band_power(b, 2600, 2650) / band_power(a, 2600, 2650)) == pytest.approx(-34, abs=2)
    sn = PS.StationNoise(ch)
    sn.feed(0, 30.0)
    sn.feed(1, 30.0)
    top = [sn.profile[i].snapshot()["noise_db"][4] - sn.profile[i].snapshot()["noise_db"][1] for i in (0, 1)]
    assert top[0] == pytest.approx(0, abs=1) and top[1] < -6


def test_each_station_hears_its_own_interference():
    """Impulses at station 1 only: its noise profile counts them, station 0's
    doesn't; a station's own transmission (and the radio's recovery after
    it) is not noise."""
    PS = _phy_session()
    spec = I.Spec(I.Impulses(trains_per_min=20.0, height_db=30.0, height_sd_db=2.0))
    ch = PS.ContinuousChannel("awgn", 5.0, 3, 120, interference=(I.Spec(), spec))
    sn = PS.StationNoise(ch)
    sn.sent(0, 20.0, 30.0)
    sn.feed(0, 60.0)
    sn.feed(1, 60.0)
    s0, s1 = sn.profile[0].snapshot(), sn.profile[1].snapshot()
    assert s0["impulses_per_min"] < 1 and s1["impulses_per_min"] > 20
    # PRE_S before the session to 57 s (COMMIT_S); station 0 less its burst (20-30 s) and the recovery after it
    kept = round((PS.PRE_S + 60 - NoiseProfile.COMMIT_S) * 10)
    assert s0["noise_blocks"] == min(NoiseProfile.WINDOW, kept - 100 - round(NoiseProfile.RECOVER_S * 10))
    assert s1["noise_blocks"] == min(NoiseProfile.WINDOW, kept)


def test_floor_wander_moves_the_tails_not_the_median():
    """A floor wandering 1 dB (WANDER_TAU_S): the profile's p90/median rises a
    little in every band, its median and band shape don't move; without
    wander the floor is untouched."""
    PS = _phy_session()
    snaps = []
    for wander in (0.0, 1.0):
        ch = PS.ContinuousChannel("awgn", 5.0, 4, 300, wander_db=wander)
        sn = PS.StationNoise(ch)
        sn.feed(0, 120.0)
        snaps.append(sn.profile[0].snapshot())
    flat, wand = snaps
    assert all(w > f + 0.2 for f, w in zip(flat["noise_tail_db"], wand["noise_tail_db"]))
    assert np.ptp(np.array(wand["noise_db"]) - np.array(flat["noise_db"])) < 0.5
    assert PS.ContinuousChannel("awgn", 5.0, 4, 60).floor_gain(0, 3.0, 10) == 1.0
