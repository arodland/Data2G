"""CW ID (data2g.cwid, docs/arq.md §7b): the keyed tone, CwSense, and the
engine sending one in and after a session, and outside one."""

import json

import numpy as np
import pytest

from data2g import codes, cpm, cwid
from data2g import hfchannel as H
from data2g.arq import engine as E
from data2g.arq import phy as PHY
from data2g.arq import session as S
from data2g.arq.burst import Slot, TxBurst, ctl_mask, data_mask
from data2g.arq.modes import MODES, ctl_spec
from data2g.config import FS
from test_engine import BLOCK, link


def test_audio_timing_and_shape():
    # PARIS: 50 dots a word, so 20 wpm keys "PARIS " 20 times a minute (less the last word gap)
    assert cwid.seconds("PARIS " * 20, 20) == pytest.approx(60 - 7 * 1.2 / 20 + cwid.RAMP / FS)
    x = cwid.audio(cwid.id_text("w1aw-7"), 20)
    assert cwid.id_text("w1aw-7") == "DE W1AW" and len(x) == round(cwid.seconds("DE W1AW", 20) * FS)
    assert abs(x[0]) < 1e-3 and abs(x[-1]) < 1e-3 and np.max(np.abs(x)) == pytest.approx(1.0, abs=1e-3)
    # constant envelope at the peak while keyed; raised-cosine edges keep it in band
    f = np.fft.rfftfreq(len(x), 1 / FS)
    p = np.abs(np.fft.rfft(x)) ** 2
    assert p[np.abs(f - cwid.FREQ_HZ) > 250].max() < 1e-4 * p.max()
    with pytest.raises(ValueError):
        cwid.audio("DE W1AW", 60)


def _active(x, center=cwid.FREQ_HZ):
    s = cwid.CwSense(center)
    return np.array([s.feed(x[i:i + 80]) for i in range(0, len(x), 80)])


def test_cwsense_hears_an_id_not_noise_or_narrow_bursts():
    x = cwid.audio("DE VK2ABC", 20)
    y = H.awgn(H.freq_shift(np.concatenate([np.zeros(FS), x, np.zeros(FS)]), 30.0), -3, seed=1, s_power=0.5)
    a = _active(y)
    t = (np.arange(len(a)) + 1) * 80 / FS
    on = t[a]
    assert on.min() < 1.0 + 0.5 and on.max() > 1.0 + len(x) / FS - 0.5  # from its first letters to its last
    assert not _active(np.random.default_rng(2).normal(size=60 * FS)).any()
    rng = np.random.default_rng(3)
    for name in ("n4-qpsk-r1/3", "n10-qpsk-r1/2", "fsk16r25-r1/3", "fsk8r50-r1/2", "w48-qpsk-r1/2"):
        spec = MODES[name]
        r = lambda sp: bytes(rng.integers(0, 256, codes.payload_bytes(sp), dtype=np.uint8))  # noqa: E731
        slots = [Slot(ctl_mask(0, 0, 5), 0, r(ctl_spec(spec)))] + [Slot(data_mask(0, i, 5), 0, r(spec)) for i in (1, 2)]
        b = PHY.tx_audio(TxBurst(name, slots, 0, 0 if "r25" in name or "r50" in name else 2))
        assert not _active(H.awgn(np.concatenate([np.zeros(800), b, np.zeros(800)]), 30, seed=4)).any(), name


def test_cwsense_ignores_a_steady_tone_starting_and_ending():
    """A steady carrier's start and end are one long key-down run, not an ID:
    a Hand-like 8-FSK burst (6.25 Hz, 66 symbols, steady envelope) on the
    centre's channels, and a plain 3 s tone, never read as CW. Before the
    keyed-run rule (CwSense.MIN_RUNS, MAX_RUN) every Hand at 1470 or 1530 Hz did."""
    rng = np.random.default_rng(5)
    pad = np.zeros(FS)
    for freq in (1470.0, 1530.0, 1590.0):
        x = cpm.tones(cpm.Grid(f"hand{freq:g}", 8, 6.25, center=freq), rng.integers(0, 8, 66))
        assert not _active(H.awgn(np.concatenate([pad, x, pad]), 20, seed=6)).any(), freq
    tone = np.sin(2 * np.pi * 1500 * np.arange(3 * FS) / FS)
    assert not _active(H.awgn(np.concatenate([pad, tone, pad]), 20, seed=7)).any()


def _cw_tx(path):
    return [e for e in map(json.loads, open(path / "events.jsonl")) if e["kind"] == "tx" and e["submode"] == "cw"]


@pytest.mark.parametrize("mode", ["add", "only"])
def test_cw_id_during_and_after_a_session(tmp_path, mode):
    """Each station's CW ID goes ahead of its turn when due and after the
    session; the peer holds for it (nothing times out) and data still flows.
    "only": no ID frames."""
    a = E.Engine("W1AW", seed=21, record_dir=tmp_path / "a")
    b = E.Engine("K2XYZ", seed=22, record_dir=tmp_path / "b")
    for e in (a, b):
        e.id_interval_s, e.cw_id = 3.0, mode
    b.listen()
    a.connect("K2XYZ", 2)
    assert link(a, b, 12, 60, lambda: a.session.state == S.CONNECTED and b.session.state == S.CONNECTED)
    key = a.session.station.key
    up, down = bytes(range(256)) * 40, bytes(range(255, -1, -1)) * 20
    a.session.write(up)
    b.session.write(down)
    got_a, got_b, ev = bytearray(), bytearray(), []

    def done():
        got_a.extend(a.session.read())
        got_b.extend(b.session.read())
        ev.extend(a.events() + b.events())
        return len(got_b) >= len(up) and len(got_a) >= len(down)
    assert link(a, b, 12, 300, done, seed=3)
    assert bytes(got_b) == up and bytes(got_a) == down
    assert a.session.station.stats.get("timeouts", 0) == 0
    assert _cw_tx(tmp_path / "a") and _cw_tx(tmp_path / "b")
    assert (f"ID W1AW {key}" in ev) == (mode == "add")
    n_a, n_b = len(_cw_tx(tmp_path / "a")), len(_cw_tx(tmp_path / "b"))
    a.session.disconnect()
    assert link(a, b, 12, 60, lambda: a.session.state == S.CLOSED and b.session.state == S.CLOSED, seed=4)
    # B's after its DISC_ACK, A's ID_GUARD_S after B's
    assert link(a, b, 12, 30, lambda: len(_cw_tx(tmp_path / "a")) > n_a and len(_cw_tx(tmp_path / "b")) > n_b, seed=5)


def test_cw_id_after_a_cq_frame():
    """Outside a session: the CW ID CW_PLAN_S after the first transmission,
    held by BUSY (another station's CW) until CW_FORCE_S; the next one
    CW_PLAN_S after it when a transmission follows it within CW_GAP_S."""
    a = E.Engine("W1AW", seed=1)
    a.cw_id = "only"
    qrm = cwid.audio("DE K2XYZ " * 40, 20)  # ~4 minutes keyed
    sent = []

    def run(until, busy_from=None):
        while a.now < until:
            x = np.zeros(BLOCK)
            if busy_from is not None and a.now >= busy_from:
                i = int((a.now - busy_from) * FS) % len(qrm)
                x = 0.3 * qrm.take(np.arange(i, i + BLOCK), mode="wrap")
            _, ptt = a.step(x)
            if ptt and (not sent or sent[-1][1] != a.tx_submode or a.now - sent[-1][0] > 30):
                sent.append((a.now, a.tx_submode))

    a.send_cq("W1AW", 2)
    run(1.0)
    cq = sent[0][0]
    run(cq + E.CW_FORCE_S + 20, busy_from=cq + E.CW_PLAN_S - 10)
    assert [m for _, m in sent] == [sent[0][1], "cw"]
    assert cq + E.CW_FORCE_S - 1 < sent[1][0] < cq + E.CW_FORCE_S + 2
    run(sent[1][0] + 60)
    a.send_cq("W1AW", 2)  # within CW_GAP_S of the ID: the next one CW_PLAN_S after the ID
    run(sent[1][0] + E.CW_PLAN_S + 20)
    assert [m for _, m in sent][2:] == [sent[0][1], "cw"]
    assert E.CW_PLAN_S - 1 < sent[3][0] - sent[1][0] < E.CW_PLAN_S + 2
