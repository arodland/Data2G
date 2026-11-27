"""Wire protocol 1.0 extensibility and pre-1.0 retirements (docs/wire-extensibility-1.0.md):
- Unified 20-byte base-40 CONNECT on all modes (OFDM and CPM)
- Major version + caps/req negotiation, NAK_VERSION/NAK_CAPS detail, cross-mode CONNECT/CONNECT_ACK
- Primed CONNECT and CONNECT_ACK with default vs non-default caps
- Core.recommend == REC_EXT (63) + T_REC_EXT
- Critical (128..255) vs ignorable (1..127) TLVs in ARQ and Broadcast (plus reserved bit)
- Unknown valid-CRC PHY header indices treated cleanly as foreign air
"""

import random

import numpy as np
import pytest

from data2g import codes, config, constellation, cpm, kisslink, modem, receiver
from data2g.arq import burst as B
from data2g.arq import frames as F
from data2g.arq import link as L
from data2g.arq import modes as M
from data2g.arq import phy as PHY
from data2g.arq import session as S
from data2g.config import BANDS, FS, PROFILE, SUBMODES
from data2g.waveform import ofdm
from data2g.waveform.dsp import tx_condition

from test_arq import RandomPolicy
from test_nod import FullRx


class _FixedRecPolicy(RandomPolicy):
    def __init__(self, rec=0, mode="m46"):
        super().__init__(random.Random(1), change=0.0)
        self._rec = rec
        self._mode = mode

    def choose(self, station, escalation):
        return self._mode, 4

    def recommend(self, station):
        return self._rec, 1, None


class _ConnectPolicy(RandomPolicy):
    def __init__(self, mode="m46", robust_mode="c20"):
        super().__init__(random.Random(1), change=0.0)
        self._cmode = mode
        self._rmode = robust_mode

    def choose(self, station, escalation):
        return self._cmode, 4

    def payload_bytes(self, submode):
        return 20 if submode == "c20" else 23 if submode == "m23" else super().payload_bytes(submode)

    def ctl_payload_bytes(self, submode):
        return self.payload_bytes(submode)

    def max_ctl(self, submode):
        return 1 if submode == "c20" else 4

    def rv_cycle(self, submode):
        return 4

    def connect_mode(self, cap, tries=0, last=False):
        return self._cmode if tries == 0 else self._rmode

    def airtime(self, submode, n_cw, dup=False):
        return 0.5 * n_cw


def _rx(burst):
    return FullRx(burst)


def test_base40_callsign_and_connect_roundtrip():
    for call in ("", "A", "W1AW", "VE3/W1AW-1", "VARA KISS", "0123456789", "A-B/C D-9"):
        v = F.pack_call_b40(call)
        assert 0 <= v < F.CALL_B40_MAX < (1 << 54)
        assert F.unpack_call_b40(v) == call.upper().rstrip()
    with pytest.raises(ValueError):
        F.unpack_call_b40(F.CALL_B40_MAX)
    with pytest.raises(ValueError):
        F.unpack_call_b40(39)  # digit 39 is outside the 39-char alphabet
    c = F.Connect(1, "VE3/W1AW-1", "K2XYZ/P", 0xABCD, 2, 63, F.CAP_NODS | F.CAP_EXT | 0x1234, 0xA5)
    raw = c.pack()
    assert len(raw) == 20 == F.Connect.BYTES
    assert F.Connect.unpack(raw) == c
    assert F.Connect.calls(raw) == ("VE3/W1AW-1", "K2XYZ/P")


def test_unified_connect_single_codeword_on_all_modes():
    """CONNECT uses 20-byte COMPACT_CONNECT on OFDM and CPM alike, fitting 1
    codeword even on 23-byte FM connect modes (n10-qpsk-r1/5)."""
    for mode in ("m46", "m23", "c20"):
        a = S.Session("W1AW", _ConnectPolicy(mode=mode), rng=random.Random(1))
        b = S.Session("K2XYZ", _ConnectPolicy(mode=mode), rng=random.Random(2))
        b.listen()
        a.connect("K2XYZ", 2, 0.0)
        conn = a.poll(0.0)
        assert len(conn.slots) == 1 and conn.slots[0].mask_id == B.COMPACT_CONNECT
        a.on_tx_end(conn, 0.5)
        b.on_rx(_rx(conn), 0.5)
        ack = b.poll(0.5)
        assert b.state == S.CONNECTED and ack is not None and len(ack.slots) == 1
        a.on_rx(_rx(ack), 1.0)
        assert a.state == S.CONNECTED and a.station.nod_peer and b.station.nod_peer


def test_caps_and_req_negotiation_and_nak_details():
    """Unknown caps bits are ignored; unknown req bits yield NAK_CAPS with the
    unsupported mask; bad major version yields NAK_VERSION with min/max range;
    nods=False clears CAP_NODS so the peer never nods."""
    # 1. Unknown caps bit accepted; nods=False disables CAP_NODS
    a = S.Session("W1AW", _ConnectPolicy(), rng=random.Random(1))
    b = S.Session("K2XYZ", _ConnectPolicy(), rng=random.Random(2))
    a.caps = 0x0400  # unknown 1.x cap, CAP_NODS cleared
    b.nods = True
    b.listen()
    a.connect("K2XYZ", 2, 0.0)
    conn = a.poll(0.0)
    b.on_rx(_rx(conn), 0.5)
    ack = b.poll(0.5)
    assert b.state == S.CONNECTED and b.peer_caps == 0x0400 and not b.station.nod_peer
    a.on_rx(_rx(ack), 1.0)
    assert a.state == S.CONNECTED and (a.peer_caps & F.CAP_NODS) and a.station.nod_peer

    # 2. Unknown req bit in CONNECT -> NAK_CAPS with detail = unsupported mask
    a2 = S.Session("W1AW", _ConnectPolicy(), rng=random.Random(3))
    b2 = S.Session("K2XYZ", _ConnectPolicy(), rng=random.Random(4))
    a2.req = 0x05
    b2.listen()
    a2.connect("K2XYZ", 2, 0.0)
    conn2 = a2.poll(0.0)
    b2.on_rx(_rx(conn2), 0.5)
    nak = b2.poll(0.5)
    assert b2.state == S.LISTEN and nak is not None
    ctl = F.Control.unpack([s.payload for s in nak.slots])
    kn = F.ConnectNak.unpack(ctl.ext[F.T_SESS])
    assert kn.reason == S.NAK_CAPS and kn.detail == 0x05
    a2.on_rx(_rx(nak), 1.0)
    assert a2.state == S.CLOSED and a2.close_reason == "CAPS 0x05"

    # 3. Incompatible major version -> NAK_VERSION with detail = (MIN<<4)|MAX
    b3 = S.Session("K2XYZ", _ConnectPolicy(), rng=random.Random(5))
    b3.listen()
    a3 = S.Session("W1AW", _ConnectPolicy(), rng=random.Random(6))
    a3.connect("K2XYZ", 2, 0.0)
    orig = a3.poll(0.0)
    c = F.Connect.unpack(orig.slots[0].payload)
    bad_ver = F.Connect(2, c.caller, c.callee, c.nonce, c.cap, c.t_turn10, c.caps, 0).pack()
    bad_burst = B.TxBurst(orig.submode, [B.Slot(B.COMPACT_CONNECT, 0, bad_ver.ljust(len(orig.slots[0].payload), b"\0"))], 0)
    b3.on_rx(_rx(bad_burst), 0.5)
    nak3 = b3.poll(0.5)
    kn3 = F.ConnectNak.unpack(F.Control.unpack([s.payload for s in nak3.slots]).ext[F.T_SESS])
    assert kn3.reason == S.NAK_VERSION and kn3.detail == ((S.MIN_VERSION << 4) | S.MAX_VERSION)
    a3.on_rx(_rx(nak3), 1.0)
    assert a3.state == S.CLOSED and a3.close_reason == f"VERSION {S.MIN_VERSION}-{S.MAX_VERSION}"

    # 4. Unknown req bit in CONNECT_ACK closes caller with CAPS detail
    a4 = S.Session("W1AW", _ConnectPolicy(), rng=random.Random(7))
    a4.connect("K2XYZ", 2, 0.0)
    c4 = F.Connect.unpack(a4.poll(0.0).slots[0].payload)
    ack_body = F.ConnectAck(c4.nonce, 2, 10, F.CAP_NODS, 0x02).pack()
    ack_ctl = F.Control(F.Core(ftype=F.SESSION), {F.T_SESS: ack_body}).pack(46)
    ack_burst = B.TxBurst("m46", [B.Slot(B.ctl_mask(1, 0, 0), 0, ack_ctl[0])], 0)
    a4.on_rx(_rx(ack_burst), 1.0)
    assert a4.state == S.CLOSED and a4.close_reason == "CAPS 0x02"


def test_cross_mode_connect_and_connect_ack():
    """CONNECT sent in one mode (e.g. OFDM) answered by CONNECT_ACK in another
    mode (e.g. CPM) connects cleanly and transfers data."""
    a = S.Session("W1AW", _ConnectPolicy(mode="m46"), rng=random.Random(1))
    b = S.Session("K2XYZ", _ConnectPolicy(mode="c20"), rng=random.Random(2))
    b.listen()
    a.connect("K2XYZ", 2, 0.0)
    conn = a.poll(0.0)
    assert conn.submode == "m46"
    b.on_rx(_rx(conn), 0.5)
    ack = b.poll(0.5)
    assert b.state == S.CONNECTED
    # Re-package the CONNECT_ACK payload onto a CPM-sized 20-byte burst
    ctl_bytes = F.Control(F.Core(ftype=F.SESSION), {F.T_SESS: F.Control.unpack([s.payload for s in ack.slots]).ext[F.T_SESS]}).pack(20)
    cross_ack = B.TxBurst("c20", [B.Slot(B.ctl_mask(1, 0, 0), 0, ctl_bytes[0])], 0)
    a.on_rx(_rx(cross_ack), 1.0)
    assert a.state == S.CONNECTED and a.peer == "K2XYZ"


def test_rec_ext_sentinel_and_tlv_roundtrip():
    """REC_EXT (63) is reserved; extended mode recommendations >= 63 travel via
    T_REC_EXT (1 or 2 bytes) and malformed T_REC_EXT is dropped cleanly."""
    assert M.REC_EXT == 63 and M.decode(M.REC_EXT) is None
    for name in M.SELECTABLE:
        assert M.encode(name) < M.REC_EXT

    for rec in (63, 120, 300):
        a = L.Station(0, _FixedRecPolicy(rec=rec), master=True)
        b = L.Station(1, _FixedRecPolicy(rec=0))
        burst = a.build()
        ctl = F.Control.unpack([s.payload for s in burst.slots])
        assert ctl.core.recommend == M.REC_EXT and F.T_REC_EXT in ctl.ext
        assert b.handle(_rx(burst))
        assert b.peer_recommend == rec

    # Malformed T_REC_EXT (3 bytes) is dropped without changing state
    a = L.Station(0, _FixedRecPolicy(rec=0), master=True)
    b = L.Station(1, _FixedRecPolicy(rec=0))
    burst = a.build()
    ctl = F.Control.unpack([s.payload for s in burst.slots])
    ctl.core.recommend = M.REC_EXT
    ctl.ext[F.T_REC_EXT] = b"\x01\x02\x03"
    bad = B.TxBurst("m46", [B.Slot(B.ctl_mask(0, 0, 0), 0, ctl.pack(46)[0])], 0)
    assert not b.handle(_rx(bad))


def test_arq_tlv_ignorable_vs_critical():
    """ARQ Control.unpack and Station.handle accept unknown ignorable TLVs
    (1..127) and reject unknown critical TLVs (128..255)."""
    rng = random.Random(42)
    a = L.Station(0, _FixedRecPolicy(rec=0), master=True)
    b = L.Station(1, _FixedRecPolicy(rec=0))
    burst = a.build()
    ctl = F.Control.unpack([s.payload for s in burst.slots])

    # Unknown ignorable TLVs (types 50, 127) pass Control.unpack and Station.handle
    for t in (50, 99, 127):
        val = bytes(rng.randrange(256) for _ in range(rng.randint(0, 6)))
        c = F.Control(ctl.core, {**ctl.ext, t: val})
        ps = c.pack(46)
        unpacked = F.Control.unpack(ps)
        assert unpacked.ext[t] == val
        b_copy = L.Station(1, _FixedRecPolicy(rec=0))
        ok_burst = B.TxBurst("m46", [B.Slot(B.ctl_mask(0, i, 0), 0, p) for i, p in enumerate(ps)], 0)
        assert b_copy.handle(_rx(ok_burst))

    # Unknown critical TLVs (types 128, 200, 255) fail Control.unpack and are dropped by Station.handle
    for t in (128, 200, 255):
        c = F.Control(ctl.core, {**ctl.ext, t: b"\xaa\xbb"})
        ps = c.pack(46)
        with pytest.raises(ValueError):
            F.Control.unpack(ps)
        b_copy = L.Station(1, _FixedRecPolicy(rec=0))
        bad_burst = B.TxBurst("m46", [B.Slot(B.ctl_mask(0, i, 0), 0, p) for i, p in enumerate(ps)], 0)
        assert not b_copy.handle(_rx(bad_burst))


def test_broadcast_tlv_ignorable_vs_critical_and_reserved_bit():
    """Broadcast parse_tlvs skips unknown 1..127 TLVs, rejects unknown 128..255
    TLVs, and ignores the header's reserved bit (0x04)."""
    # parse_tlvs direct checks
    raw_ok = bytes([55, 2, 0xDE, 0xAD, 0])
    assert kisslink.parse_tlvs(raw_ok)[55] == b"\xde\xad"
    assert kisslink.read_control(kisslink.parse_tlvs(raw_ok))[0] == kisslink.PORT0_GROUP
    with pytest.raises(ValueError):
        kisslink.parse_tlvs(bytes([128, 1, 0x01, 0]))

    # End-to-end broadcast burst with reserved bit set + unknown ignorable TLV
    tx = kisslink.KissLink(cap=2)
    rx = kisslink.KissLink(cap=2)
    tx.enqueue(b"hello broadcast")
    burst = tx.next_burst()
    assert burst is not None
    # Verify sender clears reserved bit (0x04)
    c0 = bytearray(burst.slots[0].payload)
    assert (c0[0] & 0x04) == 0
    # Inject reserved bit 0x04 and an unknown ignorable TLV (type 77, len 2) before zero padding
    c0[0] |= 0x04
    # Find padding byte after existing TLVs
    i = 1
    while i + 2 <= len(c0) and c0[i]:
        i += 2 + c0[i + 1]
    assert i + 4 <= len(c0)
    c0[i:i + 4] = bytes([77, 2, 0x11, 0x22])
    mod_burst = B.TxBurst(burst.submode, [B.Slot(burst.slots[0].mask_id, 0, bytes(c0))] + burst.slots[1:], 0)
    x = PHY.tx_audio(mod_burst)
    y = np.concatenate([np.zeros(FS // 2), x, np.zeros(FS // 2)])
    r = receiver.receive_any(y, lead=FS)
    assert r is not None
    assert rx.on_burst(r) == [(0, b"hello broadcast")]

    # Same burst with an unknown critical TLV (type 130) must drop the burst
    c0_bad = bytearray(burst.slots[0].payload)
    c0_bad[i:i + 4] = bytes([130, 2, 0x11, 0x22])
    bad_burst = B.TxBurst(burst.submode, [B.Slot(burst.slots[0].mask_id, 0, bytes(c0_bad))] + burst.slots[1:], 0)
    x_bad = PHY.tx_audio(bad_burst)
    y_bad = np.concatenate([np.zeros(FS // 2), x_bad, np.zeros(FS // 2)])
    r_bad = receiver.receive_any(y_bad, lead=FS)
    assert r_bad is not None
    assert kisslink.KissLink(cap=2).on_burst(r_bad) is None
    with pytest.raises(ValueError):
        kisslink.parse_tlvs(bytes(c0_bad[1:]))


def test_unknown_phy_header_index_treated_as_foreign_air():
    """A valid-CRC PHY header using a spare (unknown) submode index on w, n10,
    or w48 is cleanly ignored by modem.demodulate and Receiver."""
    rng = np.random.default_rng(7)
    # Find one spare index per sync band present in this profile
    for sync_band in ("w", "n10", "w48"):
        if sync_band not in BANDS:
            continue
        used = ({s.index for s in SUBMODES.values() if s.sync_band == sync_band}
                | {idx for band, idx in modem.CONNECT_HEADER if band == sync_band})
        max_idx = (1 << (10 - config.CW_BITS.get(sync_band, 6))) - 1
        spare = next((idx for idx in range(max_idx, -1, -1) if idx not in used), None)
        if spare is None:
            continue
        # Modulate a clean burst waveform with the spare header index on that band
        b = ofdm.band(sync_band)
        hdr_bits = modem.header_bits(spare, 1, sync_band)
        hdr = constellation.modulate(hdr_bits, modem.QPSK)
        layout = modem.header_layout(sync_band)
        hsyms = np.empty((len(layout), b.nc), dtype=np.complex128)
        hsyms[layout] = b.pilot
        hsyms[~layout] = hdr.reshape(b.spec.header_syms, b.nc)
        # Build 2 dummy frames of QPSK data after the header
        n_f = 2
        syms = np.empty((n_f * config.SYMS_PER_FRAME + 1, b.nc), dtype=np.complex128)
        syms[::config.SYMS_PER_FRAME] = b.pilot
        syms[:-1].reshape(n_f, config.SYMS_PER_FRAME, b.nc)[:, 1:] = modem.QPSK[
            rng.integers(0, 4, (n_f, config.DATA_SYMS_PER_FRAME, b.nc))
        ]
        x = np.concatenate([
            np.zeros(config.LEADIN_SAMPLES),
            b.preamble_waveform(),
            b.modulate_symbols(hsyms),
            b.modulate_symbols(syms),
            np.zeros(config.LEADOUT_SAMPLES),
        ])
        x = tx_condition(x, b.spec.clip_headroom_db, b.spec.clip_overshoot,
                         active=slice(config.LEADIN_SAMPLES, len(x) - config.LEADOUT_SAMPLES),
                         bandpass=BANDS[sync_band].tx_bandpass)
        y = np.concatenate([np.zeros(FS // 2), x, np.zeros(FS // 2)]) + rng.normal(0, 0.01, len(x) + FS)
        with pytest.raises(modem.SyncError):
            modem.demodulate(y)
        rcv = receiver.Receiver(modem.Accept.of(None, 16.0), cpm_grids=() if PROFILE == "fm" else tuple(cpm.GRIDS))
        evs = []
        for k in range(0, len(y), FS // 10):
            evs += rcv.feed(y[k:k + FS // 10])
        assert not [e for e in evs if e[0] == "header"]


def test_retired_pre_1_0_tlvs_absent():
    for name in ("T_RESYNC", "T_NODS", "T_KEEP"):
        assert not hasattr(F, name)
    assert not hasattr(L, "KEEP")
    assert not hasattr(L, "Told")
