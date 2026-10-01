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
