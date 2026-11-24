"""FM's Hand: channels are time slots (data2g/hand.py, docs/hand-burst.md "FM")."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from data2g import fmchannel, hand
from data2g.arq.engine import Engine
from data2g.config import FS, PROFILE

sys.path.insert(0, str(Path(__file__).parent))
from test_hand import hand_round, ref_at  # noqa: E402

fm_only = pytest.mark.skipif(PROFILE != "fm", reason="FM's slots")

CALLS = ["W1AW", "K2XYZ", "N0CALL-7", "VE3ABC", "G4XYZ", "JA1ZZZ", "DL1ABC", "VK2DEF"]


def test_slot_vectors():
    """Frozen: clients compute the same slots (docs/hand-burst.md gives these)."""
    assert [hand.slot(c, 0x5A, 32) for c in CALLS] == [hand.slot(c.lower() + " ", 0x5A, 32) for c in CALLS]
    assert [hand.slot("W1AW", t, 32) for t in (0, 1, 90, 255)] == [26, 19, 5, 31]
    assert [hand.slot(c, 90, 16) for c in CALLS] == [5, 15, 0, 7, 11, 7, 1, 15]
    assert hand.slot("W1AW", 90, 1) == 0


def test_a_new_tag_reshuffles_a_collision():
    """A pair that shares a slot under one tag shares it under about 1 in m of
    the others (a CRC seeded with the tag would keep them together under all)."""
    m, rng = 32, np.random.default_rng(1)
    calls = [f"K{i % 10}{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 3))}" for i in range(60)]
    pairs = [(a, b) for i, a in enumerate(calls) for b in calls[i + 1:] if hand.slot(a, 7, m) == hand.slot(b, 7, m)]
    assert len(pairs) > 20
    again = np.mean([hand.slot(a, t, m) == hand.slot(b, t, m) for a, b in pairs for t in range(256) if t != 7])
    assert 0.5 / m < again < 2 / m, again


def test_clean_tags():
    """Net control's pick: the roster of 8 in 32 slots has about 39% of the
    tags clean (the birthday product), and each really is."""
    tags = hand.clean_tags(CALLS, 32)
    assert 60 < len(tags) < 140, len(tags)
    assert all(len({hand.slot(c, t, 32) for c in CALLS}) == 8 for t in tags)
    assert hand.clean_tags(CALLS, 4) == []


@fm_only
def test_slot_budget():
    """The slot holds a Hand, the longest allowed key-up before the next one's
    audio, and both neighbours at the search's edge; a Hand late enough to be
    sent still lands inside net control's search."""
    assert hand.SLOT_S >= hand.LEN_S + hand.PTT_MAX_S + 2 * hand.SEARCH_S
    assert hand.PTT_MAX_SAMPLES == round(hand.PTT_MAX_S * FS) and hand.SLOT_SAMPLES == round(hand.SLOT_S * FS)
    assert hand.LATE_S < hand.SEARCH_S
    assert hand.grid(hand.channel_freq(0)).f0 == 1200.0 and hand.M * hand.RATE == 800.0


@fm_only
def test_round_in_slots():
    """Three stations answer in slots 0, 3 and 5, each within the timing
    budget: all three decode, each with its own dt."""
    hands = [(0, 0x111, 3.0), (3, 0x333, 3.2), (5, 0x555, 2.85)]
    lines, _, _ = hand_round(Engine("W1AW", seed=1), 3.0, 0x5A, [0, 1, 2, 3, 4, 5], hands, noise=0.1)
    got = {int(w[1]): (int(w[2], 16), float(w[4])) for w in (l.split() for l in lines[:-1])}
    assert {c: p for c, (p, _) in got.items()} == {0: 0x111, 3: 0x333, 5: 0x555}, lines
    assert all(abs(got[c][1] - 1000 * (s - 3.0)) < 30 for c, _, s in hands), lines


@fm_only
def test_a_hand_in_the_wrong_slot_is_nobodys():
    """The CRC binds the slot: a Hand made for slot 5 that lands in slot 3's
    time fails there, and is not found in its own (empty) slot."""
    x = hand.modulate(0x5A, hand.payload_bits(0x777), 5)
    y = np.random.default_rng(0).normal(0, 0.1, hand.WINDOW_SAMPLES + hand.slot_offset(5))
    i = hand.SEARCH_SAMPLES + hand.slot_offset(3)
    y[i:i + len(x)] += x
    assert hand.decode_window(y, 0x5A, [3, 5], hand.SEARCH_SAMPLES) == []
    y = np.random.default_rng(0).normal(0, 0.1, len(y))
    i = hand.SEARCH_SAMPLES + hand.slot_offset(5)
    y[i:i + len(x)] += x
    assert [r["c"] for r in hand.decode_window(y, 0x5A, [3, 5], hand.SEARCH_SAMPLES)] == [5]


@fm_only
def test_send_waits_for_its_slot_and_holds_the_radio():
    """HAND SEND in channel 4 keys 4 slots after channel 0 would, and nothing
    of mine goes out in the slots before."""
    e = ref_at(Engine("W1AW", seed=1), 1.0)
    e.hand_send(1, 0x5A, 1.0, 4, 0x123)
    up = e._hand_tx[0]
    assert up == int(round(2.0 * FS)) - e.ptt_delay + 4 * hand.SLOT_SAMPLES
    e.send_cq("W1AW", 2)
    while e.n < up:
        assert not e.step(np.zeros(FS // 10))[1]
    assert e.step(np.zeros(FS // 10))[1] and e.take_hand_events() == ["HAND SENT 4"]


def _slot_env(ms: str, profile: str = "fm") -> subprocess.CompletedProcess:
    code = "from data2g import hand; print(hand.SLOT_SAMPLES * 1000 // hand.FS, hand.PTT_MAX_SAMPLES * 1000 // hand.FS)"
    env = {**os.environ, "DATA2G_PROFILE": profile, "DATA2G_HAND_SLOT_MS": ms}
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          cwd=Path(__file__).parent.parent)


def test_the_slot_is_set_from_the_environment():
    """DATA2G_HAND_SLOT_MS: a longer slot for slow radios, all of it the PTT
    delay's allowance. Under 1200 ms there is no room for the Hand and its
    search; HF has no slots and ignores it."""
    assert _slot_env("2500").stdout.split() == ["2500", "1300"]
    assert _slot_env("1200").stdout.split() == ["1200", "0"]
    for bad in ("1199", "5001", "long"):
        assert _slot_env(bad).returncode != 0
    assert _slot_env("2500", "hf").stdout.split() == ["0", "0"]


@fm_only
def test_a_slow_ptt_is_refused():
    """A key-up over the slot's budget would put my carrier on the slot
    before mine."""
    e = ref_at(Engine("W1AW", seed=1, ptt_delay_s=hand.PTT_MAX_S + 0.05), 1.0)
    with pytest.raises(ValueError, match="PTT"):
        e.hand_send(1, 0x5A, 1.0, 4, 0x123)
    ref_at(Engine("W1AW", seed=1, ptt_delay_s=hand.PTT_MAX_S), 1.0).hand_send(1, 0x5A, 1.0, 4, 0x123)


def _on_air(xs: list[tuple[np.ndarray, int, float]], n: int, cnr_db: float = 30.0, seed: int = 0) -> np.ndarray:
    """Stations keying on one FM channel, as one receiver hears them: each
    (audio, start sample, level dB) made a carrier of its own, summed at RF
    over the receiver's noise, then discriminated. A carrier is up only while
    its audio runs."""
    r = fmchannel.RADIOS["mic"]
    rf = fmchannel.rf_rate(FS)
    up = rf // FS
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(10 ** (-cnr_db / 10) * rf / fmchannel.CNR_REF_BW_HZ / 2)
    z = sigma * (rng.normal(size=n * up) + 1j * rng.normal(size=n * up))
    for k, (x, start, db) in enumerate(xs):
        s, _ = fmchannel.rf_signal(x, r, fs=FS)
        z[start * up:start * up + len(s)] += 10 ** (db / 20) * s * np.exp(2j * np.pi * (rng.random() + 300.0 * k * np.arange(len(s)) / rf))
    y = fmchannel.rx_audio(z, r, FS, rf, 1500.0)
    return y / np.sqrt(np.mean(y ** 2)) * 0.2


@fm_only
def test_on_one_fm_channel_slots_carry_what_keying_together_loses():
    """Why FM's channels are slots. Two stations through the FM channel model
    (their carriers 300 Hz apart, summed before the discriminator), one 6 dB
    down: in the same slot the stronger captures and the weaker is lost
    (0 of 20 at 3 dB down and beyond, docs/hand-burst.md); in neighbouring
    slots both decode. HF's side-by-side tones add."""
    tag, n = 0x5A, hand.WINDOW_SAMPLES + hand.slot_offset(1) + FS
    a = hand.modulate(tag, hand.payload_bits(0xAAA), 0)
    b0 = hand.modulate(tag, hand.payload_bits(0xBBB), 0)
    b1 = hand.modulate(tag, hand.payload_bits(0xBBB), 1)
    at = hand.SEARCH_SAMPLES
    doubled = _on_air([(a, at, 0.0), (b0, at + FS // 20, -6.0)], n)
    assert [r["payload"] for r in hand.decode_window(doubled, tag, [0], at)] == [0xAAA]
    slotted = _on_air([(a, at, 0.0), (b1, at + hand.slot_offset(1), -6.0)], n)
    assert [(r["c"], r["payload"]) for r in hand.decode_window(slotted, tag, [0, 1], at)] == [(0, 0xAAA), (1, 0xBBB)]


@fm_only
def test_neighbours_at_the_budgets_edge():
    """The slot's budget at its worst: slot 0's station as late as the search
    reaches, slot 1's as early, its carrier up PTT_MAX_S before its Hand (the
    leading silence is an unmodulated carrier). The two carriers never
    overlap, and both Hands decode."""
    tag, n = 0x5A, hand.WINDOW_SAMPLES + hand.slot_offset(1) + FS
    edge, ptt = hand.SEARCH_SAMPLES - FS // 100, int(hand.PTT_MAX_S * FS)
    a = hand.modulate(tag, hand.payload_bits(0xAAA), 0)
    b = np.concatenate([np.zeros(ptt), hand.modulate(tag, hand.payload_bits(0xBBB), 1)])
    at = hand.SEARCH_SAMPLES
    a0, b0 = at + edge, at + hand.slot_offset(1) - edge - ptt
    assert a0 + len(a) <= b0
    got = hand.decode_window(_on_air([(a, a0, 0.0), (b, b0, 0.0)], n), tag, [0, 1], at)
    assert [(r["c"], r["payload"]) for r in got] == [(0, 0xAAA), (1, 0xBBB)]
    assert abs(got[0]["dt_ms"] - 1000 * edge / FS) < 10 and abs(got[1]["dt_ms"] + 1000 * edge / FS) < 10
