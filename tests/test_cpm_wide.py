"""The wide-bin toolbox grids (cpm.WIDE_GRIDS): bursts round-trip through the
blind lock and the square-law receiver, and stay out of what is on air."""

from dataclasses import replace

import numpy as np
import pytest

from data2g import codes, cpm


def test_toolbox_is_not_on_air():
    """The wide grids are on air only as DATA2G_CPM_ONAIR names them (none
    unset), and never in the shifter's choices (policy.allowed)."""
    from data2g.arq import policy
    from data2g.arq.modes import MODES

    on = set(cpm.ON_AIR_WIDE)
    assert set(cpm.GRIDS) == {"c16r25", "c8r50", "c32r62"} | on and len(cpm.CTL) == 3 + len(on)
    assert len(cpm.SPECS) == 6 + 2 * len(on)
    for name, g in cpm.WIDE_GRIDS.items():
        assert (name in cpm.GRIDS) == (name in on) and cpm.GRIDS[name] is g and g.L > 1
        assert g.T == g.L * g.nsub and g.baud * g.T == pytest.approx(8000)
        assert cpm.coherent(g) and not cpm.coherent(replace(g, guard=1))
    assert {n for n in cpm.WIDE_SPECS if n in MODES} == {n for n, s in cpm.WIDE_SPECS.items()
                                                         if cpm.WIDE_ON or s.grid in on}
    assert not any(s.name in cpm.WIDE_SPECS for cap in (0, 1, 2) for s in policy.allowed(cap))


@pytest.mark.parametrize("name", list(cpm.WIDE_SPECS))
def test_round_trip(name):
    spec = cpm.SPECS[name]
    g = cpm.GRIDS[spec.grid]
    rng = np.random.default_rng(3)
    ctl = cpm.CTL[g.name]
    pay = [bytes(rng.integers(0, 256, codes.payload_bytes(s), dtype=np.uint8)) for s in (ctl, spec)]
    x = cpm.modulate(spec, [codes.encode(ctl, pay[0], index=0), codes.encode(spec, pay[1], index=1)], False, cap=2)
    assert len(x) == cpm.layout(g.name, cpm.stream_symbols(g.name, 1, False)).n * g.T
    lead = 2400
    # -4 dB in 2500 Hz, 20 Hz off: well over both grids' floors
    y = np.concatenate([np.zeros(lead), x, np.zeros(2400)])
    y = y + rng.normal(scale=np.sqrt(1.6 * 10 ** 0.4), size=len(y))
    t = np.arange(len(y)) / 8000
    from scipy.signal import hilbert

    y = np.real(hilbert(y) * np.exp(2j * np.pi * 20.0 * t))
    lock = cpm.find(g, y[:lead + 4000 + cpm.hdr_span(g.name, 1) + g.T])
    assert lock is not None and lock["spec"] == spec and lock["n_data"] == 1 and not lock["dup"]
    assert abs(lock["start"] - lead) <= g.T // 8 and abs(lock["cfo"] - 20.0) < g.rate / 8
    r = cpm.receive(y, lock)
    assert r["Z"].shape == (len(r["E"]) + len(cpm.layout(g.name, len(r["E"])).sync_rows)
                            + 2 * cpm.hdr_len(g), g.L, g.m)  # sub-window bins: the coherent control and DD
    c = cpm.coh_read_header(g, y, lock["start"], lock["cfo"], 1)
    assert c[0] == spec and c[3] > cpm.COH_LOCK[g.name][0] and c[3] - c[4] > cpm.HDR_MARGIN[0]
    from data2g.arq import phy as PHY

    p = codes.decode_posterior(spec, r["soft"][1])[2]
    dd = cpm.dd_soft(r, cpm.dd_metric(r, {1: p}, 1), 1)
    assert dd.shape == np.asarray(r["soft"][1]).shape
    assert codes.decode_many(spec, dd[None], index=[1])[0] == (pay[1], True)
    assert PHY.ModemRx(dict(r), {})._dd(spec)
    assert codes.decode_many(ctl, np.asarray(r["soft"][0])[None], index=[0])[0] == (pay[0], True)
    assert codes.decode_many(spec, np.asarray(r["soft"][1])[None], index=[1])[0] == (pay[1], True)
    m = cpm.measure(g, r["E"], len(r["E"]))
    assert abs(m["snr_est"] - (-4.0)) < 1.5
    mid = cpm.find_mid(g, y)
    assert mid is not None and mid["spec"] == spec and abs(mid["start"] - lead) <= g.T // 8


def test_guard_drops_the_first_sub_window():
    g = replace(cpm.WIDE_GRIDS["c16w125"], guard=1)
    rng = np.random.default_rng(1)
    sym = rng.integers(0, g.m, 50)
    x = cpm.tones(g, sym)
    E = cpm._energies(g, x, 0, 50, 0.0)
    assert (E.argmax(axis=1) == sym).all()
    # the kept sub-window alone: garbage in the first changes nothing
    y = x.reshape(50, g.L, g.nsub).copy()
    y[:, 0] = rng.normal(size=(50, g.nsub))
    assert np.allclose(cpm._energies(g, y.reshape(-1), 0, 50, 0.0), E)
    assert cpm._noise_median(1) == np.log(2) and 1.6 < cpm._noise_median(2) < 1.7


def test_streaming_receiver_given_the_grids():
    """A Receiver handed the wide grids locks and receives their bursts
    (its default search does not hold them)."""
    from data2g import modem, receiver

    spec = cpm.SPECS["fsk16w125-r1/3"]
    g, ctl = cpm.GRIDS[spec.grid], cpm.CTL[spec.grid]
    rng = np.random.default_rng(5)
    pay = [bytes(rng.integers(0, 256, codes.payload_bytes(s), dtype=np.uint8)) for s in (ctl, spec)]
    x = cpm.modulate(spec, [codes.encode(ctl, pay[0], index=0), codes.encode(spec, pay[1], index=1)], False, cap=2)
    y = np.concatenate([np.zeros(8000), x, np.zeros(16000)])
    y = 0.1 * (y + rng.normal(scale=1.0, size=len(y)))
    for grids, want in ((tuple(cpm.WIDE_GRIDS), True), (tuple(cpm.GRIDS), False)):
        rx = receiver.Receiver(modem.Accept.of(None, 16.0), cpm_grids=grids)
        got = [ev for i in range(0, len(y), 800) for k, ev in rx.feed(y[i:i + 800]) if k == "burst" and ev["rx"]]
        assert bool(got) == want
        if want:
            r = got[0]["rx"]
            assert r["spec"] == spec
            assert codes.decode_many(spec, np.asarray(r["soft"][1])[None], index=[1])[0] == (pay[1], True)


def test_shared_coarse_rows_are_the_grids_own(pure):
    """c8w50's search rows summed from c8r50's symbols' energies equal its own transforms."""
    g, h = cpm.WIDE_GRIDS["c8w50"], cpm.GRIDS["c8r50"]
    assert cpm.shares_rows(g, h) and not cpm.shares_rows(cpm.WIDE_GRIDS["c16w125"], cpm.GRIDS["c32r62"])
    rng = np.random.default_rng(2)
    x = rng.normal(size=40000)
    src, own = cpm.CoarseTrack(h), cpm.CoarseTrack(g)
    shared = cpm.CoarseTrack(g, source=src)
    off, buf = 0, np.zeros(0)
    for i in range(0, len(x), 1777):  # a receiver's buffer: grows, and its head is trimmed
        buf = np.concatenate([buf, x[i:i + 1777]])
        if len(buf) > 12000:
            off, buf = off + 5003, buf[5003:]
        for t in (src, own, shared):
            t.feed(buf, off)
        for ph in range(4):
            assert shared.s[ph] == own.s[ph] and shared.E[ph].shape == own.E[ph].shape
            assert np.allclose(shared.E[ph], own.E[ph], rtol=1e-9, atol=1e-12)
    assert len(own.E[0]) > 10


@pytest.mark.parametrize("name", ["fsk8w50-r1/3", "fsk16w125-r1/3"])
def test_carrier_in_the_grid_gets_its_own_floor(name):
    """master's per-tone floors (tone_floors) on sub-window sums: a steady
    carrier on one tone is flagged, the burst still locks and decodes, and
    snr_est reads the burst, not the carrier."""
    spec = cpm.SPECS[name]
    g, ctl = cpm.GRIDS[spec.grid], cpm.CTL[spec.grid]
    rng = np.random.default_rng(7)
    pay = [bytes(rng.integers(0, 256, codes.payload_bytes(s), dtype=np.uint8)) for s in (ctl, spec)]
    x = cpm.modulate(spec, [codes.encode(ctl, pay[0], index=0), codes.encode(spec, pay[1], index=1)], False, cap=2)
    n = len(x) + 4800
    y0 = np.concatenate([np.zeros(2400), x, np.zeros(2400)]) + rng.normal(scale=np.sqrt(1.6 * 10 ** 0.4), size=n)  # -4 dB
    sigma2 = 1.6 * 10 ** 0.4
    # 18 dB over the noise in 50 Hz, 7 Hz off the tone
    car = np.sqrt(2 * sigma2 * 50 / 4000 * 10 ** 1.8) * np.cos(2 * np.pi * (g.f0 + 3 * g.rate + 7.0) * np.arange(n) / 8000)
    got = {}
    for tag, y in (("clean", y0), ("carrier", y0 + car)):
        lock = cpm.find(g, y[:2400 + 4000 + cpm.hdr_span(g.name, 2) + g.T])
        assert lock is not None and lock["spec"] == spec, tag
        r = cpm.receive(y, lock)
        E = r["E"][:ctl.n_sym]
        got[tag] = (cpm.tone_floors(E, cpm._n0(E, g.k), g.k), cpm.measure(g, r["E"], len(r["E"]))["snr_est"])
        assert codes.decode_many(ctl, np.asarray(r["soft"][0])[None], index=[0])[0] == (pay[0], True), tag
        assert codes.decode_many(spec, np.asarray(r["soft"][1])[None], index=[1])[0] == (pay[1], True), tag
    assert got["clean"][0] is None and got["carrier"][0] is not None and int(np.argmax(got["carrier"][0])) == 3
    assert abs(got["carrier"][1] - got["clean"][1]) < 3.0
