"""CPM sync: every grid's burst locks at its front, and no later sync block
passes for a front (c8r50's once repeated the front's Costas array)."""

import numpy as np
import pytest

from data2g import codes, cpm


def test_front_lock_and_mid_blocks_unlike_the_front():
    rng = np.random.default_rng(0)
    for name, spec in cpm.SPECS.items():
        if not name.endswith("r1/2"):
            continue
        g = cpm.GRIDS[spec.grid]
        ctl = codes.encode(cpm.CTL[g.name], bytes(codes.payload_bytes(cpm.CTL[g.name])), 0, 0)
        data = [codes.encode(spec, bytes(codes.payload_bytes(spec)), 0, 0) for _ in range(3)]
        x = cpm.modulate(spec, [ctl] + data, dup=False)
        y = np.concatenate([np.zeros(2000), x, np.zeros(2000)]) + rng.normal(0, 0.05, len(x) + 4000)
        lock = cpm.find(g, y)
        assert lock is not None and abs(lock["start"] - 2000) < g.T // 4 and lock["n_data"] == 3, (name, lock)
        # the front pattern's detector, over a burst with its front cut off
        late = y[2000 + (len(cpm.preamble_pattern(g)) + 4) * g.T:]
        lock = cpm.find(g, late)
        assert lock is None or lock["n_data"] == 3 and lock["spec"].name == name, (name, lock)


def test_a_search_range_ending_before_the_front_does_not_lock_a_period_early():
    """A streaming search whose range ends a few symbols before a burst's
    start: c8r50's front, one Costas array four times before protocol 13,
    locked one period early there (sync 0.76) and read another header
    (r1/2 x4 as r1/3 x6). No front may lock off its start."""
    rng = np.random.default_rng(0)
    for name, spec in cpm.SPECS.items():
        g = cpm.GRIDS[spec.grid]
        ctl = codes.encode(cpm.CTL[g.name], bytes(codes.payload_bytes(cpm.CTL[g.name])), 0, 0)
        data = [codes.encode(spec, bytes(codes.payload_bytes(spec)), 0, 0) for _ in range(3)]
        x = cpm.modulate(spec, [ctl] + data, dup=False)
        y = np.concatenate([np.zeros(8000), x, np.zeros(8000)]) + rng.normal(0, 0.05, len(x) + 16000)
        for k in range(1, 16):  # hi a half symbol to 7.5 symbols before the start
            lock = cpm.find(g, y, hi=8000 - k * g.T // 2)
            assert lock is None or (abs(lock["start"] - 8000) < g.T // 4 and lock["spec"].name == name
                                    and lock["n_data"] == 3), (name, k, lock)


def test_tx_filter_by_cap():
    """Each grid's TX filter under every cap it may be used in (cpm.TX_FILTERS):
    out of band under -37 dB (PSD against the mean between the outer tones)
    beyond the cap's edges around the tones' centre (Grid.f0 rounds to a
    tone: c8r50's sits at 1475 Hz), a lower peak where the cap is wider, and
    the unchanged receiver still locks and decodes."""
    from scipy import signal

    from data2g.arq import phy as PHY
    from data2g.arq import policy as G
    from data2g.arq.link import Slot, TxBurst, ctl_mask, data_mask
    from data2g.config import FS

    rng = np.random.default_rng(2)
    for name, spec in cpm.SPECS.items():
        if not name.endswith("r1/2"):
            continue
        g = cpm.GRIDS[spec.grid]
        peaks = []
        for cap, hz in G.CAP_HZ.items():
            if spec not in G.allowed(cap):
                continue
            slots = [Slot(ctl_mask(0, 0, 3), 0, rng.bytes(codes.payload_bytes(cpm.CTL[g.name])))]
            slots += [Slot(data_mask(0, i, 3), 0, rng.bytes(codes.payload_bytes(spec))) for i in range(1, 3)]
            b = TxBurst(name, slots, 0, cap)
            x = PHY.tx_audio(b)
            f, p = signal.welch(x, FS, nperseg=4096)
            ref = p[(f >= g.f0) & (f <= g.f0 + (g.m - 1) * g.rate)].mean()
            out = p[np.abs(f - (g.f0 + (g.m - 1) * g.rate / 2)) >= hz / 2].max()
            assert 10 * np.log10(out / ref) < -37, (name, cap, 10 * np.log10(out / ref))
            peaks.append(PHY.peak_db(name, cap))
            y = np.concatenate([np.zeros(2000), x, np.zeros(2000)]) + rng.normal(0, 0.05, len(x) + 4000)
            rx = PHY.ModemRx(cpm.receive(y, cpm.find(g, y)), {})
            assert [rx.decode(i, s.mask_id, 0, None) for i, s in enumerate(slots)] == [s.payload for s in slots], (name, cap)
        assert peaks == sorted(peaks, reverse=True), (name, peaks)


def test_pair_probe_on_a_data_slot_is_a_miss():
    """The link's blind ARQ_DUP probe (slot 0 failed alone: try slots 0+1 as
    one control codeword) must not combine a CPM data slot into control:
    it crashed a loss study (960 bits into a 360-bit buffer)."""
    from data2g.arq import phy as PHY
    from data2g.arq.link import Slot, TxBurst, ctl_mask, data_mask

    spec = cpm.SPECS["fsk8r50-r1/2"]
    b = TxBurst(spec.name, [Slot(ctl_mask(0, 0, 3), 0, bytes(codes.payload_bytes(cpm.CTL[spec.grid])))]
                + [Slot(data_mask(0, 1, 3), 0, bytes(codes.payload_bytes(spec)))], 0)
    x = PHY.tx_audio(b)
    y = np.concatenate([np.zeros(2000), x, np.zeros(2000)])
    y += np.random.default_rng(1).normal(0, 0.05, len(y))
    rx = PHY.ModemRx(cpm.receive(y, cpm.find(cpm.GRIDS[spec.grid], y)), {})
    assert rx.decode(0, b.slots[0].mask_id, 0, None) == b.slots[0].payload
    # slot 0 failed alone (another burst's mask here), its soft bits stored:
    # the probe must not combine slot 1's data codeword into them
    other = ctl_mask(1, 0, 3)
    assert rx.decode(0, other, 0, ("ctl", 0)) is None
    assert rx.decode(1, other, 1, ("ctl", 0)) is None
    assert rx.decode(1, b.slots[1].mask_id, 0, None) == b.slots[1].payload


def test_a_strong_cpm_burst_is_not_lost_to_a_weak_ofdm_header():
    """OFDM's search runs first and read strong CPM audio as a header
    (scores 0.25-0.29, under the suspect level) in 5 of 40 fsk8r50 bursts
    at MPG +10 dB: the CPM lock must win."""
    import sys
    from pathlib import Path

    from data2g import hfchannel
    from data2g.arq import phy as PHY
    from data2g.config import FS
    from data2g.receiver import receive_any

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import outcome_data as O

    for seed in (3, 9):
        rng = np.random.default_rng(seed)
        b = O.burst("fsk8r50-r1/2", 2, rng)
        lead = int(rng.uniform(0.3, 1) * FS)
        x = np.concatenate([np.zeros(lead), PHY.tx_audio(b), np.zeros(FS // 2)])
        y = hfchannel.apply_channel(x, snr_db=10.0, freq_offset_hz=float(rng.uniform(-50, 50)), ppm=10,
                                    fading_preset="mpg", seed=seed)
        r = receive_any(y, lead=FS)
        assert r is not None and r["spec"].name == "fsk8r50-r1/2", (seed, r and r["spec"].name)


def test_a_lock_before_the_buffer_is_a_miss(monkeypatch):
    """detect's fine timing (+-T/8) can move a coarse lock at sample 0 to a
    negative start. No tile alignment is then left to read, and find's max()
    over none crashed the host on air."""
    g = cpm.GRIDS["c8r50"]
    monkeypatch.setattr(cpm, "detect", lambda *a, **k: (1.0, -g.T // 8, 0.0))
    x = np.random.default_rng(0).normal(0, 0.05, 40 * g.T)
    assert cpm.find(g, x) is None


@pytest.mark.hf_only("seed-picked draws at an HF-calibrated SNR on HF channels; FM picks its own on fmchannel")
def test_cpm_control_padding_is_known_when_it_fails(monkeypatch):
    """CPM's control is a polar codeword: its payload enters the code
    reversed (codes.polar_order), so a short control's padding sits on the
    weakest positions. At -11 dB these draws fail plain and decode with the
    padding known (phy.POLAR_PIN_CUTS). Square-law control soft bits
    (cpm.COH_CTL off): the coherent ones decode these draws plain."""
    monkeypatch.setattr(cpm, "COH_CTL", False)
    from data2g import hfchannel
    from data2g.arq import frames as F
    from data2g.arq import phy as PHY
    from data2g.arq.link import Slot, TxBurst, ctl_mask, data_mask

    spec = cpm.SPECS["fsk8r50-r1/2"]
    cs = cpm.CTL[spec.grid]
    ctl = F.Control(F.Core(burst_seq=3, acted_on=2, cum=17, k=1, recommend=5), {F.T_REPLY: b"\x05"}).pack(
        codes.payload_bytes(cs))[0]
    data = bytes(np.random.default_rng(5).integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8))
    b = TxBurst(spec.name, [Slot(ctl_mask(0, 0, 7), 0, ctl), Slot(data_mask(0, 1, 7), 0, data)], 0)
    x = np.concatenate([np.zeros(4000), PHY.tx_audio(b), np.zeros(4000)])
    m = PHY.mask_value(ctl_mask(0, 0, 7))
    for seed in (3, 7, 9):
        y = hfchannel.apply_channel(x, snr_db=-11.0, seed=seed)
        r = cpm.receive(y, cpm.find(cpm.GRIDS[spec.grid], y))
        cands, usable = codes.decode_raw(cs, np.asarray(PHY.soft_bits(r)[0])[None], index=[0])
        assert codes.check(cs, cands[0], usable[0], m) is None  # plain fails
        rx = PHY.ModemRx(r, {})
        assert ctl in rx.raw(0)
        assert rx.decode(0, ctl_mask(0, 0, 7), 0, None) == ctl


def test_spread_tracks_doppler_through_symbol_jitter():
    """cpm.spread: a Rayleigh tone's energy per symbol plus white jitter as
    strong as the fading (what defeated the old lag-1 estimate: 3.0 always)."""
    from data2g import hfchannel

    g = cpm.GRIDS["c8r50"]
    rng = np.random.default_rng(3)
    for true in (0.5, 2.0, 6.0):
        est = []
        for _ in range(8):
            h = hfchannel._gaussian_taps(400 * g.T, true, rng)[:: g.T]
            e = np.abs(h) ** 2
            est.append(cpm.spread(g, e + rng.exponential(np.mean(e), len(e)), 0.0))
        assert abs(np.median(est) / true - 1) < 0.3, (true, est)
    # no correlation past lag 0: too fast if the energy fades (Rayleigh per
    # symbol), else no fading (a steady tone, the noise's own variance)
    n0, s = 0.1, 1.0
    awgn = s + n0 + rng.normal(0, np.sqrt(2 * n0 * s + n0 * n0), 400)
    assert cpm.spread(g, s * rng.exponential(1.0, 400) + n0, n0) == cpm.SPREAD_MAX_HZ
    assert cpm.spread(g, awgn, n0) == 0.0
    assert cpm.spread(g, np.ones(10), n0) == cpm.UNMEASURED_SPREAD_HZ  # too short


def _faded_control_energies(name, doppler, delay_ms, snr, seed):
    """A control-only burst (a robust CONNECT's shape) through two-path
    fading, at genie timing: its tone energies."""
    from data2g import hfchannel as H

    spec = cpm.SPECS[name]
    g = cpm.GRIDS[spec.grid]
    ctl = codes.encode(cpm.CTL[g.name], bytes(range(codes.payload_bytes(cpm.CTL[g.name]))), 0, 0)
    x = cpm.modulate(spec, [ctl], dup=False)
    y = H.awgn(H.fading(x, H.FadingPreset("p", doppler, delay_ms), seed=seed), snr, seed=seed + 1, s_power=H.active_power(x))
    return g, cpm.soft(g, spec, y, 0, 0.0, 0, False)[1]


@pytest.mark.hf_only("CPM: FM has none")
def test_spread_on_slow_selective_fading():
    """With multipath each tone has its own gain, so on slow fading the
    winning tone's energy varies with the data, not with time: no lag
    correlation, and the excess variance read SPREAD_MAX_HZ (fsk16r25, 2 ms
    / 0.1 Hz, these 40 seeds: 35; 2 ms / 1 Hz: 10; fsk8r50: 25 and 2), which
    is ROBUST_FAST's fast fading. The symbols on the same or adjacent tones
    80-160 ms apart are correlated there (SPREAD_NEAR_*): unmeasured.
    Really fast fading still reads it, 10 Hz too: its energies are
    correlated over a symbol or two, not over 80 ms (pairs from the next
    symbol on read 10 Hz as slow in about 1 burst in 10 of fsk16r25's and 1
    in 3 of fsk8r50's)."""
    def count(name, doppler, delay_ms, snr):
        est = []
        for seed in range(40):
            g, E = _faded_control_energies(name, doppler, delay_ms, snr, seed)
            est.append(cpm.measure(g, E, len(E))["spread_est"])
        return sum(e == cpm.SPREAD_MAX_HZ for e in est), sum(e == cpm.UNMEASURED_SPREAD_HZ for e in est)

    # (maxed, unmeasured) of 40; in the comments: as measured
    assert count("fsk16r25-r1/2", 0.1, 2.0, 2)[0] <= 5  # 3
    assert count("fsk16r25-r1/2", 1.0, 2.0, 2)[0] <= 5  # 2
    assert count("fsk8r50-r1/3", 0.1, 2.0, 2)[0] <= 3  # 1
    for name, doppler, delay_ms, least, most_unmeasured in (
            ("fsk16r25-r1/2", 10.0, 7.0, 37, 1),  # 38, 0 (38 without the check)
            ("fsk16r25-r1/2", 10.0, 2.0, 38, 1),  # 39, 0 (39)
            ("fsk16r25-r1/2", 30.0, 7.0, 39, 1),  # 40, 0 (40)
            ("fsk8r50-r1/3", 10.0, 7.0, 19, 1),  # 20, 0 (20; the rest are fits)
            ("fsk8r50-r1/3", 30.0, 7.0, 35, 4)):  # 37, 3 (40)
        maxed, unmeasured = count(name, doppler, delay_ms, 0)
        assert maxed >= least and unmeasured <= most_unmeasured, (name, doppler, delay_ms, maxed, unmeasured)


@pytest.mark.hf_only("CPM: FM has none")
def test_coarse_track_is_a_fresh_transform_of_the_stream(pure):  # (native's: test_native_cpm's test_coarse_track)
    """A receiver's CoarseTrack, fed 20 ms at a time from a buffer trimmed
    behind it: its rows are coarse_rows of the stream at each phase's
    stream-grid starts, as if transformed afresh (to rounding: pocketfft's
    result depends on the batch a row is transformed in), and detect
    reading them finds what detect over the same audio on that grid finds."""
    rng = np.random.default_rng(5)
    for g in cpm.GRIDS.values():
        x = rng.normal(0, 1, 4 * 8000)
        track, off, buf = cpm.CoarseTrack(g), 0, np.zeros(0)
        for i in range(0, len(x), 160):
            buf = np.concatenate([buf, x[i:i + 160]])
            if len(buf) > 8000:  # trimmed to the last second, as a receiver's buffer
                off, buf = off + len(buf) - 8000, buf[len(buf) - 8000:]
            track.feed(buf, off)
        a, b = off + 37, off + len(buf) - 11
        for s, E in track.rows(a, b):
            want = cpm.coarse_rows(g, x[a + s:b], track.extra)
            assert len(E) == len(want) and len(E) > 10 and (a + s) % (g.T // 4) == 0, g.name
            np.testing.assert_allclose(E, want, rtol=1e-12, atol=0)
        ph = min(s for s, _ in track.rows(a, b))  # the stream grid's first start at or after a
        kw = dict(front_only=True, fine=False)  # (the whole pattern is longer than the second held)
        got, ref = cpm.detect(g, x[a:b], track=track, x0=a, **kw), cpm.detect(g, x[a + ph:b], **kw)
        assert ref[0] > 0 and got[1:] == (ref[1] + ph, ref[2]) and got[0] == pytest.approx(ref[0], rel=1e-12), g.name


@pytest.fixture
def r23_offered(monkeypatch):
    """fsk32r62-r2/3 offered for one test (DATA2G_CPM_R23=1's effect on the mode tables)."""
    from data2g.arq import modes

    if cpm.R23:  # DATA2G_CPM_R23: offered already, in native too
        yield
        return
    caches = (cpm.valid_words, cpm.header_rows)
    old = [m for m in cpm.SPECS]
    monkeypatch.setitem(cpm.SPECS, cpm.R23_SPEC.name, cpm.R23_SPEC)
    monkeypatch.setitem(modes.MODES, cpm.R23_SPEC.name, cpm.R23_SPEC)
    for c in caches:
        c.cache_clear()
    yield
    monkeypatch.undo()
    assert list(cpm.SPECS) == old
    for c in caches:
        c.cache_clear()


# offered by this test alone (DATA2G_CPM_R23 unset): native reads the switch at load
@(pytest.mark.python_only("fsk32r62-r2/3 offered by a fixture, not DATA2G_CPM_R23") if not cpm.R23 else lambda f: f)
def test_r23_announced_at_index_4_and_older_words_unchanged(r23_offered):
    """fsk32r62-r2/3: its header words (index 4, 6 duplicated) are new ones,
    the two older modes' words are what they were, the recommendation codes
    of the older modes too, and a burst of it (control twice, two data
    codewords) locks and decodes."""
    from data2g.arq import modes
    from data2g.arq import phy as PHY
    from data2g.arq.link import Slot, TxBurst, ctl_mask, data_mask

    spec = cpm.SPECS["fsk32r62-r2/3"]
    g = cpm.GRIDS[spec.grid]
    assert (spec.index, spec.k, spec.coded_bits, codes.payload_bytes(spec)) == (4, 640, 960, 76)
    old = {cpm.header_word(i, n, d) for i in (0, 1) for d in (False, True) for n in range(cpm.MAX_DATA + 1)}
    words = [cpm.header_word(s.index, n, d) for s, n, d, _ in cpm.valid_words(g.name)]
    assert len(set(words)) == len(words) == 3 * 2 * (cpm.MAX_DATA + 1) and old < set(words)
    assert [modes.encode(m) & 15 for m in cpm.SPECS] == list(range(7)) and list(cpm.SPECS)[6] == spec.name
    assert modes.decode(modes.encode(spec.name)) == spec.name
    rng = np.random.default_rng(5)
    ctl = rng.bytes(codes.payload_bytes(cpm.CTL[g.name]))
    slots = [Slot(ctl_mask(0, 0, 3), 0, ctl), Slot(ctl_mask(0, 0, 3), 1, ctl)]
    slots += [Slot(data_mask(0, i, 3), 0, rng.bytes(codes.payload_bytes(spec))) for i in range(2, 4)]
    x = PHY.tx_audio(TxBurst(spec.name, slots, 0, 2))
    y = np.concatenate([np.zeros(2000), x, np.zeros(2000)]) + rng.normal(0, 0.05, len(x) + 4000)
    r = cpm.receive(y, cpm.find(g, y))
    assert r["spec"].name == spec.name and r["dup"] and r["n_cw"] == 4
    rx = PHY.ModemRx(r, {})
    assert [rx.decode(i, s.mask_id, 0, None) for i, s in enumerate(slots) if not s.rv] == [s.payload for s in slots if not s.rv]


def _carrier_burst(seed, snr_db=-8.0, carrier_hz=1573.4, inr_db=12.0):
    """A one-control fsk16r25 burst in white noise (SNR in 2500 Hz) with a
    steady carrier `inr_db` over the noise in 50 Hz (None: no carrier)."""
    from data2g.arq import burst as B, phy as PHY
    from data2g.config import FS

    rng = np.random.default_rng(seed)
    payload = rng.bytes(codes.payload_bytes(cpm.CTL["c16r25"]))
    mask = B.ctl_mask(0, 0, 5)
    x = PHY.tx_audio(B.TxBurst("fsk16r25-r1/2", [B.Slot(mask, 0, payload)], 0, 2))
    sigma2 = float(np.mean(x * x)) * (FS / 2) / 2500 / 10 ** (snr_db / 10)
    y = np.concatenate([np.zeros(4000), x, np.zeros(4000)]) + rng.normal(0, np.sqrt(sigma2), len(x) + 8000)
    if inr_db is not None:
        amp = np.sqrt(2 * sigma2 * 50 / (FS / 2) * 10 ** (inr_db / 10))
        y = y + amp * np.cos(2 * np.pi * carrier_hz * np.arange(len(y)) / FS + rng.uniform(0, 2 * np.pi))
    return y, mask, payload


def test_tone_floors_only_where_a_tone_stands_out():
    rng = np.random.default_rng(1)
    hits = 0
    for n in (22, 90, 240):  # a header read, a control codeword, a data codeword
        for _ in range(1000):
            E = rng.exponential(1.0, (n, 16))
            hits += cpm.tone_floors(E, cpm._n0(E)) is not None
    assert hits <= 3  # noise alone: under 1 in 1000
    E = rng.exponential(1.0, (90, 16))
    E[:, 5] += 8.0  # a carrier in tone 5
    nb = cpm.tone_floors(E, cpm._n0(E))
    assert nb is not None and nb[5] > 5 * cpm._n0(E)
    assert np.all(np.delete(nb, 5) == cpm._n0(E))


def test_a_steady_carrier_in_the_passband_does_not_stop_the_burst():
    """One N0 for every tone let a carrier win its tone in every symbol
    (tone_metric, coh_metric): no fsk16r25 burst decoded at 5 dB over the
    AWGN floor. With the tone's own floor (tone_floors) they do."""
    from data2g.arq import phy as PHY
    from data2g.receiver import receive_any

    for hz in (1330.0, 1573.4, 1690.0):
        for seed in range(3):
            y, mask, payload = _carrier_burst(seed, carrier_hz=hz)
            r = receive_any(y, lead=8000, cpm_grids=("c16r25",))
            assert r is not None and r["spec"].name == "fsk16r25-r1/2", (hz, seed)
            assert PHY.ModemRx(r, {}).decode(0, mask, 0, None) == payload, (hz, seed)


@pytest.mark.hf_only("CPM: FM has none")
def test_a_carrier_does_not_read_as_signal_in_snr_est():
    """measure() took each row's strongest tone as signal: a carrier that
    wins its tone read 2-8 dB high (12-24 dB in 50 Hz), and the gear
    shifter climbed. Scaled to the quiet tones' floor, Es scaled back up
    by the share kept, it reads within about 1 dB (a low read costs
    sessions too)."""
    from data2g.arq import phy as PHY
    from data2g.receiver import receive_any

    def est(seed, **kw):
        y, *_ = _carrier_burst(seed, **kw)
        return PHY.measure(receive_any(y, lead=8000, cpm_grids=("c16r25",)))["snr_est"]

    for seed in range(2):
        clean = est(seed, inr_db=None)
        for hz, inr, low in ((1573.4, 12.0, -1.0), (1330.0, 18.0, -1.5)):  # before: +2.0..+2.5, +5.4..+6.0
            d = est(seed, carrier_hz=hz, inr_db=inr) - clean
            assert low < d < 0.5, (seed, hz, inr, d)


def test_cpm_study_toggles_keep_cpm_in_python(monkeypatch):
    """with_native / --native under the study toggles in effect (on this branch
    most are on by default, so native_subs reads the settings, not the
    environment): the soft-value ones keep every provider that makes or decodes
    CPM soft values in Python (C++ has master's metric); fsk32r62-r2/3 offered
    or c8w50 on air keeps all but the OFDM and audio ones; scattered pilots keep
    all but base, waveform, cpm, codes and audio."""
    import native_subs

    from data2g import config

    for v in ("DATA2G_CONNECT_SEQ", "DATA2G_ROBUST_RULE", "DATA2G_ROBUST_FAST", "DATA2G_CPM_LEAK_K",
              "DATA2G_CPM_LEAK_Z", "DATA2G_CPM_CTL_SQ", "DATA2G_CPM_SPILL"):
        monkeypatch.delenv(v, raising=False)
    for mod, name, off in ((cpm, "ON_AIR_WIDE", ()), (cpm, "WIDE_ON", False), (cpm, "R23", False),
                           (cpm, "LEAK", frozenset()), (config, "SC_BANDS", ())):
        monkeypatch.setattr(mod, name, off)
    assert native_subs.study_skip() == set()
    monkeypatch.setattr(cpm, "LEAK", frozenset({"ctl", "data", "dd", "ctlsq"}))
    assert native_subs.study_skip() == native_subs.CPM_SOFT_PY
    monkeypatch.setattr(cpm, "R23", True)
    assert native_subs.study_skip() == set(native_subs.PROVIDERS) - native_subs.WIDE_NATIVE
    monkeypatch.setattr(cpm, "R23", False)
    monkeypatch.setattr(cpm, "ON_AIR_WIDE", ("c8w50",))
    assert {"cpm", "codes", "gear", "arq", "receiver", "phy", "engine"} <= native_subs.study_skip()
    monkeypatch.setattr(cpm, "ON_AIR_WIDE", ())
    monkeypatch.setattr(cpm, "LEAK", frozenset())
    monkeypatch.setattr(config, "SC_BANDS", ("w", "w48", "n10", "n4"))
    assert native_subs.study_skip() == set(native_subs.PROVIDERS) - native_subs.PILOTS_NATIVE


def test_leak_metric_only_where_the_spill_is_significant(monkeypatch):
    """The leak model (DATA2G_CPM_LEAK) leaves a codeword on today's metric
    unless its spill stands LEAK_Z standard errors out: AWGN codewords score
    exactly as without it; a tone leaking a fifth of its energy into each
    neighbour (c16r25 on 30 Hz) goes on leak_metric."""
    g = cpm.GRIDS["c16r25"]
    rng = np.random.default_rng(4)

    def codeword(leak, esn0=8.0, n=240):
        t = rng.integers(0, g.m, n)
        h = np.sqrt(esn0 / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
        y = (rng.normal(size=(n, g.m)) + 1j * rng.normal(size=(n, g.m))) / np.sqrt(2)
        y[np.arange(n), t] += h
        for d in (-1, 1):
            k = np.clip(t + d, 0, g.m - 1)
            y[np.arange(n), k] += np.sqrt(leak) * np.sqrt(esn0 / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
        return np.abs(y) ** 2

    awgn = [codeword(0.0, esn0) for esn0 in (0.5, 2.0, 8.0, 30.0) for _ in range(5)]
    leaky = codeword(0.2)
    today = [cpm.tone_metric(g, E, "data") for E in awgn]
    monkeypatch.setattr(cpm, "LEAK", frozenset({"data"}))
    assert sum(cpm.spill_significant(E) for E in awgn) <= 1
    assert sum(np.array_equal(cpm.tone_metric(g, E, "data"), t) for E, t in zip(awgn, today)) >= len(awgn) - 1
    assert cpm.spill_significant(leaky)
    assert np.array_equal(cpm.tone_metric(g, leaky, "data"), cpm.leak_metric(leaky, cpm.LEAK_K))


@pytest.mark.hf_only("CPM: FM has none")
def test_cpm_presence_detects_bursts_and_rejects_carriers_and_colored_noise():
    """cpm.present() hears a mid-burst CPM signal (even with a +20 dB carrier
    in-band) and rejects noise, steady/intermittent/dual carriers, CW IDs and
    narrow SSB-filtered noise; CoarseTrack.presence matches cpm.presence."""
    from scipy import signal as sps

    from data2g import cwid, hfchannel
    from data2g.config import FS

    rng = np.random.default_rng(11)
    L = 3 * FS
    t = np.arange(L) / FS
    n0 = 1.0 / (FS / 2)
    p20 = 10 ** (20.0 / 10) * n0 * 50.0

    # Real mid-burst CPM tones (with random CFO, and with a +20 dB 1500 Hz carrier)
    for gn, g in ((n, cpm.GRIDS[n]) for n in cpm.PRESENCE):  # (on-air wide grids have no rule)
        x = cpm.bandpass(g, cpm.tones(g, rng.integers(0, g.m, int(6 * g.rate))))
        y = hfchannel.apply_channel(x, -6.0, freq_offset_hz=37.0, seed=4)[FS:FS + L]
        assert cpm.present(g, y), gn
        # CoarseTrack on the same window (aligned to a multiple of T / 4) matches presence()
        tr = cpm.CoarseTrack(g)
        tr.feed(y, 0)
        assert tr.presence(0, L) == pytest.approx(cpm.presence(g, y), rel=1e-9, abs=1e-12), gn
        assert tr.present(0, L)
        sp = hfchannel.active_power(x)
        sigma = np.sqrt(sp / (10 ** (-6.0 / 10)) / (2500.0 / (FS / 2)))
        pc = 10 ** (20.0 / 10) * (sigma ** 2 / (FS / 2)) * 50.0
        assert cpm.present(g, y + np.sqrt(2 * pc) * np.cos(2 * np.pi * 1500.0 * t + 0.5)), f"{gn}+carrier"

    # Non-CPM signals that previously stuck BUSY on in PR #234
    sos = sps.butter(6, [400, 2200], btype="bandpass", fs=FS, output="sos")
    negatives = {
        "noise": rng.normal(size=L),
        "carrier_1500": rng.normal(size=L) + np.sqrt(2 * p20) * np.cos(2 * np.pi * 1500.0 * t + 0.7),
        "carrier_1531": rng.normal(size=L) + np.sqrt(2 * p20) * np.cos(2 * np.pi * 1531.25 * t + 0.7),
        "carrier_30pct": rng.normal(size=L) + (t < 0.9) * np.sqrt(2 * p20) * np.cos(2 * np.pi * 1500.0 * t + 0.7),
        "two_carriers": rng.normal(size=L) + np.sqrt(2 * p20) * (np.cos(2 * np.pi * 800.0 * t) + np.cos(2 * np.pi * 2200.0 * t)),
        "cw_20wpm": hfchannel.awgn(np.pad(cwid.audio(cwid.id_text("W1AW"), 20, freq=1500.0), (0, L))[:L], 20.0, seed=9),
        "ssb_400_2200": sps.sosfilt(sos, rng.normal(size=L + FS))[FS:],
    }
    for label, sig in negatives.items():
        for gn, g in cpm.GRIDS.items():  # (a grid without a rule reads absent)
            assert not cpm.present(g, sig), (label, gn)
