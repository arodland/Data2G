"""CPM sync: every grid's burst locks at its front, and no later sync block
passes for a front (c8r50's once repeated the front's Costas array)."""

import numpy as np

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


def test_cpm_control_padding_is_known_when_it_fails():
    """CPM's control is a polar codeword: its payload enters the code
    reversed (codes.polar_order), so a short control's padding sits on the
    weakest positions. At -11 dB these draws fail plain and decode with the
    padding known (phy.POLAR_PIN_CUTS)."""
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
    for seed in (3, 6, 7):
        y = hfchannel.apply_channel(x, snr_db=-11.0, seed=seed)
        r = cpm.receive(y, cpm.find(cpm.GRIDS[spec.grid], y))
        cands, usable = codes.decode_raw(cs, np.asarray(PHY.soft_bits(r)[0])[None], index=[0])
        assert codes.check(cs, cands[0], usable[0], m) is None  # plain fails
        rx = PHY.ModemRx(r, {})
        assert ctl in rx.raw(0)
        assert rx.decode(0, ctl_mask(0, 0, 7), 0, None) == ctl
