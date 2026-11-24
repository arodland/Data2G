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
from data2g.config import FM_TX_PEAK, FS, PROFILE
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
        if name not in MODES:  # FM has no n4
            continue
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


def test_cwsense_ignores_a_weak_steady_carrier():
    """A steady carrier near RATIO times the floor flickers over and under it
    with the noise: keyed by the run rule, but its quiet windows still hold
    the tone (CwSense.QUIET_MAX). At 8-12 dB over the noise in 50 Hz, on the
    centre and 73 Hz off it, one read as a CW ID 70-85% of the time, and a
    session under it held every timeout (on_cw) and delivered nothing."""
    n = 30 * FS
    for freq in (1500.0, 1573.4):
        for inr in (8, 10, 12):
            a = np.sqrt(2 * 50 / (FS / 2) * 10 ** (inr / 10))  # over unit-variance noise's power in 50 Hz
            y = np.random.default_rng(8).normal(size=n) + a * np.sin(2 * np.pi * freq * np.arange(n) / FS)
            assert _active(y).mean() < 0.02, (freq, inr)


@pytest.mark.xfail(strict=True, reason="CwSense reads a weak or fading carrier near the centre as a CW ID (#302)")
@pytest.mark.parametrize("preset", ["mpp", "mpd"])
def test_cwsense_under_a_fading_carrier(preset):
    """A steady, unkeyed carrier 73 Hz off the centre that fades (1 and 2 Hz
    Doppler), at a mean 10-18 dB over the noise in 50 Hz: it goes through
    the level where it flickers about RATIO, and reads as a CW ID 20-57% of
    the time. Each reading holds transmission and moves a reply deadline on
    (Session.on_cw). Bound: 15%, since a deep fade is a real gap at the
    floor and some readings will remain."""
    n = 30 * FS
    f = H.fading(np.sin(2 * np.pi * 1573.4 * np.arange(n) / FS), preset, seed=11)
    for inr in (10, 14, 18):
        a = np.sqrt(2 * 50 / (FS / 2) * 10 ** (inr / 10)) / np.sqrt(2 * np.mean(f ** 2))  # the mean, over unit-variance noise
        y = np.random.default_rng(12).normal(size=n) + a * f
        assert _active(y).mean() < 0.15, inr


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="CwSense reads a weak or fading carrier near the centre as a CW ID (#302)")
@pytest.mark.parametrize("fading", [None, "mpp", "mpd"])
def test_a_carrier_does_not_hold_the_connect_retry(fading):
    """A caller nobody answers, under an unkeyed 1573.4 Hz carrier 12 dB
    over the noise in 50 Hz (interference.Qrm), steady or fading: its first
    retry goes out within 16 s (a quiet channel: 8.3-10 s; a fading carrier
    still reads as a CW ID now and then, and each reading may hold it 2.5 s).
    Each false CW ID reading holds transmission and moves the reply deadline
    on (Session.on_cw): steady or fading, six seeds each, no retry was keyed
    in 30 s (in a session: none for 88 s, and nothing delivered)."""
    from data2g import interference as I

    seconds, block = 20.0, 80
    n = int(seconds * FS)
    sigma = np.sqrt((FS / 2) / 2500.0)  # the floor of a 0 dB signal
    carrier = I.Interference(I.Spec(qrm=(I.Qrm("carrier", 1573.4, 50.0, 12.0, seconds, 1.0),)), 1, seconds).render(0.0, n, sigma)
    x = np.random.default_rng(1).normal(0, sigma, n) + (H.fading(carrier, fading, seed=1) if fading else carrier)
    a = E.Engine("W1AW", seed=1)
    a.connect("K2XYZ", 2)
    keyed, was = [], False
    for i in range(0, n, block):
        _, ptt = a.step(x[i:i + block])
        if ptt and not was:
            keyed.append(i / FS)
        was = ptt
    assert len(keyed) >= 2 and keyed[1] < 16.0, keyed


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
    n_a = len(_cw_tx(tmp_path / "a"))  # (a CW ID may have gone ahead of A's DISC)
    # B's after its DISC_ACK, A's ID_GUARD_S after B's
    assert link(a, b, 12, 30, lambda: len(_cw_tx(tmp_path / "a")) > n_a and len(_cw_tx(tmp_path / "b")) > n_b, seed=5)
    # ...and not over B's: "add" sends its ID frame, then CW, which CwSense takes ~1 s to
    # hear, and A once keyed in that gap
    spans = {k: [(e["t"], e["t"] + e["seconds"]) for e in map(json.loads, open(tmp_path / k / "events.jsonl"))
                 if e["kind"] == "tx"] for k in "ab"}
    assert not [(x, y) for x in spans["a"] for y in spans["b"] if min(x[1], y[1]) - max(x[0], y[0]) > 0.1]  # (event t: a block late)


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


def _cw_id_peak(engine):
    engine.cw_id = "only"
    engine.send_cq("W1AW", 2)
    peak = 0.0
    while engine.now < E.CW_PLAN_S + 30:
        out, ptt = engine.step(np.zeros(BLOCK))
        if ptt and engine.tx_submode == "cw":
            peak = max(peak, float(np.abs(out).max()))
    return peak


def test_cw_id_level():
    """HF: the bursts' peak, full scale. FM: a burst's RMS at the sound card
    (phy.full_scale), where a ping set the radio's deviation; it went out
    11 dB over that."""
    want = 1.0 if PROFILE == "hf" else 2 ** 0.5 / FM_TX_PEAK
    assert _cw_id_peak(E.Engine("W1AW", seed=1)) == pytest.approx(want, rel=1e-3)
