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
    from data2g.tnc import receive_any

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
