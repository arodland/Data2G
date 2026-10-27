"""data2g.arq.phy: masked CRCs and resend combining on the real modem."""

import numpy as np

from data2g import codes, hfchannel, modem
from data2g.arq import phy as PHY
from data2g.arq.link import Slot, TxBurst
from data2g.config import SUBMODES

SPEC = SUBMODES["w48-qpsk-r1/2"]


def burst(payloads, rv=0, key=7):
    return TxBurst(SPEC.name, [Slot((key, 0, i), rv, p) for i, p in enumerate(payloads)], 0)


def hear(b, snr=None, seed=0):
    x = np.concatenate([np.zeros(2400), PHY.tx_audio(b), np.zeros(2400)])
    return modem.receive(hfchannel.apply_channel(x, snr_db=snr, seed=seed) if snr is not None else x)


def test_mask_selects_the_slot_identity():
    rng = np.random.default_rng(0)
    pl = [bytes(rng.integers(0, 256, codes.payload_bytes(SPEC), dtype=np.uint8)) for _ in range(2)]
    rx = PHY.ModemRx(hear(burst(pl)), {})
    assert rx.decode(0, (7, 0, 0), 0, None) == pl[0]
    assert rx.decode(1, (7, 0, 0), 0, None) is None  # another seq
    assert rx.decode(1, (8, 0, 1), 0, None) is None  # another session
    assert rx.decode(1, (7, 1, 1), 0, None) is None  # the other direction
    assert rx.decode(1, (7, 0, 1), 0, None) == pl[1]


def test_session_traffic_reads_without_its_key():
    """Part 97: what's on air is plain once decoded. The scrambler seed is
    the burst position alone, so a listener without the session key reads
    every payload; only the masked CRC needs the key."""
    rng = np.random.default_rng(2)
    pl = [bytes(rng.integers(0, 256, codes.payload_bytes(SPEC), dtype=np.uint8)) for _ in range(3)]
    soft = PHY.soft_bits(hear(burst(pl, key=0xBEEF)))
    for i, p in enumerate(pl):
        payload, ok = codes.decode_many(SPEC, soft[i:i + 1], 0, index=i)[0]
        assert payload == p and not ok


def test_a_slot_decodes_once_for_every_mask(monkeypatch):
    """The scrambler is unkeyed, so asking a slot under several masks (the
    engine tries the session's, KISS's and mask 0 on every burst) costs
    one decode and a CRC check each."""
    rng = np.random.default_rng(3)
    pl = [bytes(rng.integers(0, 256, codes.payload_bytes(SPEC), dtype=np.uint8)) for _ in range(2)]
    r = hear(burst(pl))
    calls = []
    raw = codes.decode_raw
    monkeypatch.setattr(codes, "decode_raw", lambda *a, **k: calls.append(1) or raw(*a, **k))
    monkeypatch.setattr(PHY, "DD", False)
    rx = PHY.ModemRx(r, {})
    assert [rx.decode(0, m, 0, None) for m in [(8, 0, 0), (0, 0, 0), (7, 1, 0), (7, 0, 0)]] == [None] * 3 + [pl[0]]
    assert len(calls) == 1


def test_resend_in_another_slot_combines():
    """A seq first sent in slot 2 and resent at RV 1 in slot 0 (other
    scrambling: codes.flip aligns them) decodes from the pair."""
    rng = np.random.default_rng(1)
    pl = [bytes(rng.integers(0, 256, codes.payload_bytes(SPEC), dtype=np.uint8)) for _ in range(3)]
    store = {}
    first = PHY.ModemRx(hear(burst(pl), snr=0.5, seed=3), store)
    for seed in range(4, 20):  # a draw where slot 2 fails alone
        store.clear()
        if first.decode(2, (7, 0, 2), 0, ("p", 2)) is None:
            break
        first = PHY.ModemRx(hear(burst(pl), snr=0.5, seed=seed), store)
    assert ("p", 2) in store
    resend = TxBurst(SPEC.name, [Slot((7, 0, 2), 1, pl[2])], 0)
    r = hear(resend, snr=0.5, seed=21)
    assert PHY.ModemRx(r, {}).decode(0, (7, 0, 2), 1, ("p", 2)) is None  # the resend alone fails
    assert PHY.ModemRx(r, store).decode(0, (7, 0, 2), 1, ("p", 2)) == pl[2]


def test_failed_slot_combines_with_its_resend():
    """At an SNR where one transmission fails, the stored soft bits plus the
    RV 1 resend decode."""
    rng = np.random.default_rng(1)
    pl = [bytes(rng.integers(0, 256, codes.payload_bytes(SPEC), dtype=np.uint8)) for _ in range(4)]
    store = {}
    first = PHY.ModemRx(hear(burst(pl), snr=0.5, seed=3), store)
    got = [first.decode(i, (7, 0, i), 0, ("p", i)) for i in range(4)]
    assert got.count(None) >= 3 and len(store) == got.count(None)
    again = PHY.ModemRx(hear(burst(pl, rv=1), snr=0.5, seed=4), store)
    assert [again.decode(i, (7, 0, i), 1, ("p", i)) for i in range(4) if got[i] is None] == [
        p for p, g in zip(pl, got) if g is None]


def test_duplicated_control_pair_combines():
    """ARQ_DUP: control codeword 0 at RV 0 in slot 0 and RV 1 in slot 1. At
    an SNR where slot 0 alone mostly fails, the receiver's pair decode
    (link.Station._ctl_pair) mostly succeeds."""
    from data2g.arq import link as L
    from data2g.arq.policy import GearShifter

    rng = np.random.default_rng(5)
    alone = paired = 0
    st = L.Station(1, GearShifter(), key=7)
    # 16 seeds: the pair rate is ~0.69 (66 of 96), so 6 of 8 was a coin
    # flip on which noise draws the code saw
    for seed in range(16):
        pl = [bytes(rng.integers(0, 256, codes.payload_bytes(SPEC), dtype=np.uint8)) for _ in range(3)]
        b = TxBurst(SPEC.name, [Slot(L.ctl_mask(0, 0, 7), 0, pl[0]), Slot(L.ctl_mask(0, 0, 7), 1, pl[0]),
                                Slot(L.data_mask(0, 0, 7), 0, pl[1])], 0)
        rx = PHY.ModemRx(hear(b, snr=-1.5, seed=seed), {})
        alone += rx.decode(0, L.ctl_mask(0, 0, 7), 0, None) == pl[0]
        paired += st._ctl_pair(rx, 0, 0) == pl[0]
    assert alone <= 4 and paired >= 8, (alone, paired)


def test_control_padding_is_pinned_when_a_control_fails():
    """A control is a few bytes, zeroed to its payload (frames.Control.pack).
    At -1 dB, where the plain decode fails, the receiver decodes the slot
    again with the padding pinned (phy.PIN_CUTS) and gets it; raw() offers
    the same candidate (a broadcast control's group is read from it)."""
    from data2g.arq import frames as F
    from data2g.arq.burst import ctl_mask, data_mask

    ctl = F.Control(F.Core(burst_seq=3, acted_on=2, cum=17, k=1, recommend=5), {F.T_REPLY: b"\x05"}).pack(
        codes.payload_bytes(SPEC))[0]
    data = bytes(np.random.default_rng(5).integers(0, 256, codes.payload_bytes(SPEC), dtype=np.uint8))
    b = TxBurst(SPEC.name, [Slot(ctl_mask(0, 0, 7), 0, ctl), Slot(data_mask(0, 1, 7), 0, data)], 0)
    for seed in range(3):
        r = hear(b, snr=-1.0, seed=seed)
        cands, usable = codes.decode_raw(SPEC, PHY.soft_bits(r)[:1], index=[0])
        assert codes.check(SPEC, cands[0], usable[0], PHY.mask_value(ctl_mask(0, 0, 7))) is None  # plain fails
        rx = PHY.ModemRx(r, {})
        assert ctl in rx.raw(0)
        assert rx.decode(0, ctl_mask(0, 0, 7), 0, None) == ctl
        assert rx.decode(0, ctl_mask(0, 0, 8), 0, None) is None  # another session's key still fails


def test_long_control_is_pinned_where_its_padding_starts():
    """A control carrying an inline codeword (T_INLINE) ends past the fixed
    cuts: at 0 dB a 44 B one fails plain and with PIN_CUTS, and decodes
    leaning toward the padding its soft bits show (codes.pad_prior)."""
    from data2g.arq.burst import ctl_mask, data_mask

    rng = np.random.default_rng(5)
    pb = codes.payload_bytes(SPEC)
    ctl = bytes(rng.integers(1, 256, 44, dtype=np.uint8)) + bytes(pb - 44)
    data = bytes(rng.integers(0, 256, pb, dtype=np.uint8))
    b = TxBurst(SPEC.name, [Slot(ctl_mask(0, 0, 7), 0, ctl), Slot(data_mask(0, 1, 7), 0, data)], 0)
    for seed in range(3):
        r = hear(b, snr=0.0, seed=seed)
        prior = codes.pad_prior(SPEC, codes.info_llrs(SPEC, PHY.soft_bits(r)[0], 0))
        assert (prior[: 8 * 42] < 1).all() and (prior[8 * 50 : 8 * pb] > 10).all()
        soft, m = PHY.soft_bits(r)[:1], PHY.mask_value(ctl_mask(0, 0, 7))
        for cut in (None, *PHY.PIN_CUTS):  # plain and each fixed cut fail
            c, u = (codes.decode_raw(SPEC, soft, index=[0]) if cut is None else
                    codes.decode_raw_pinned(SPEC, soft, codes.zero_tail(SPEC, cut), 0))
            assert codes.check(SPEC, c[0], u[0], m) is None
        assert PHY.ModemRx(r, {}).decode(0, ctl_mask(0, 0, 7), 0, None) == ctl


def test_data_slots_are_not_pinned(monkeypatch):
    """Only a control mask (burst.ctl_mask) earns the retry: a failed data
    slot costs no extra decode."""
    from data2g.arq.burst import data_mask

    pl = [bytes(codes.payload_bytes(SPEC))] * 2  # all padding: pinning would rescue it
    r = hear(TxBurst(SPEC.name, [Slot(data_mask(0, i, 7), 0, p) for i, p in enumerate(pl)], 0), snr=-1.0, seed=0)
    calls = []
    pin = codes.pin
    monkeypatch.setattr(codes, "pin", lambda *a, **k: calls.append(1) or pin(*a, **k))
    monkeypatch.setattr(PHY, "DD", False)
    assert PHY.ModemRx(r, {}).decode(1, data_mask(0, 1, 7), 0, None) is None
    assert calls == []


def test_duplicated_control_pair_is_pinned_once_combined():
    """ARQ_DUP: at -4 dB a padded control fails alone even pinned, and the
    pair combined fails plain; combined and pinned (codes.pin_buffer), it
    decodes."""
    from data2g.arq import frames as F
    from data2g.arq import link as L
    from data2g.arq.policy import GearShifter

    ctl = F.Control(F.Core(burst_seq=3, acted_on=2, cum=17, k=1, recommend=5), {F.T_REPLY: b"\x05"}).pack(
        codes.payload_bytes(SPEC))[0]
    data = bytes(np.random.default_rng(5).integers(0, 256, codes.payload_bytes(SPEC), dtype=np.uint8))
    b = TxBurst(SPEC.name, [Slot(L.ctl_mask(0, 0, 7), 0, ctl), Slot(L.ctl_mask(0, 0, 7), 1, ctl),
                            Slot(L.data_mask(0, 0, 7), 0, data)], 0)
    st = L.Station(1, GearShifter(), key=7)
    m = PHY.mask_value(L.ctl_mask(0, 0, 7))
    for seed in range(3):
        r = hear(b, snr=-4.0, seed=seed)
        soft = PHY.soft_bits(r)
        buf = codes.combine(SPEC, codes.combine(SPEC, None, codes.flip(SPEC, 0, 0) * soft[0], 0),
                            codes.flip(SPEC, 1, 1) * soft[1], 1)
        assert not codes.decode_buffer(SPEC, buf, 1, m, index=codes.PLAIN)[0][1]  # the pair, plain, fails
        rx = PHY.ModemRx(r, {})
        assert rx.decode(0, L.ctl_mask(0, 0, 7), 0, None) is None  # alone, pinned, fails
        assert st._ctl_pair(rx, 0, 0) == ctl
