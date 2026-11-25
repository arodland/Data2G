"""native/core/cwid against data2g.cwid, and CW IDs between the C++ and
Python engines (docs/arq.md §7b)."""

import json

import numpy as np
import pytest

from data2g import cwid
from data2g import hfchannel as H
from data2g.arq import engine as E
from data2g.arq import session as S
from data2g.config import FS
from test_cwid import _cw_id_peak
from test_engine import link


@pytest.mark.parametrize("text,wpm", [("DE W1AW", 20), ("DE VK2ABC/P", 15), ("DE K2XYZ", 33.3), ("cq de g4abc", 40)])
def test_audio_matches(native, text, wpm):
    assert native.cwid.id_text("w1aw-7") == cwid.id_text("w1aw-7")
    assert native.cwid.keying(text) == [tuple(k) for k in cwid.keying(text)]
    assert native.cwid.seconds(text, wpm) == cwid.seconds(text, wpm)
    np.testing.assert_allclose(native.cwid.audio(text, wpm), cwid.audio(text, wpm), atol=1e-12)


@pytest.mark.parametrize("snr,fading", [(None, None), (-6, None), (0, "mpp"), (10, "mpd")])
def test_sense_matches(native, snr, fading):
    """Window by window: active and hits_from."""
    x = np.concatenate([np.zeros(FS), cwid.audio("DE W1AW", 25), np.zeros(FS)])
    y = H.apply_channel(x, None, freq_offset_hz=20.0, fading_preset=fading, seed=3)
    if snr is not None:
        y = H.awgn(y, snr, seed=4, s_power=0.5)
    y = y + 1e-9 * np.random.default_rng(5).normal(size=len(y))  # no window of exact zeros
    py, cc = cwid.CwSense(), native.cwid.CwSense()
    for i in range(0, len(y), 80):
        a, b = py.feed(y[i:i + 80]), cc.feed(y[i:i + 80])
        assert a == b and py.hits_from == cc.hits_from, i
    assert py.hits_from > 0 or snr is not None  # (a weak or faded one may never go active)


@pytest.mark.parametrize("freq,inr", [(1500.0, 8), (1500.0, 12), (1573.4, 10), (1573.4, 12), (1530.0, 6)])
def test_sense_matches_under_a_steady_carrier(native, freq, inr):
    """Window by window under a weak steady carrier (CwSense.QUIET_MAX): the
    quiet-median rule decides here, where it never does on a clean ID."""
    n = 20 * FS
    a = np.sqrt(2 * 50 / (FS / 2) * 10 ** (inr / 10))
    y = np.random.default_rng(9).normal(size=n) + a * np.sin(2 * np.pi * freq * np.arange(n) / FS)
    py, cc = cwid.CwSense(), native.cwid.CwSense()
    on = 0
    for i in range(0, len(y), 80):
        a_, b_ = py.feed(y[i:i + 80]), cc.feed(y[i:i + 80])
        assert a_ == b_ and py.hits_from == cc.hits_from, i
        on += a_
    assert on < 0.03 * len(y) / 80


@pytest.mark.parametrize("preset,inr", [("mpp", 10), ("mpp", 14), ("mpd", 10), ("mpd", 18)])
def test_sense_matches_under_a_fading_carrier(native, preset, inr):
    """Window by window under a fading steady carrier, where the detector
    is at its margin (tests/test_cwid.py)."""
    n = 20 * FS
    f = H.fading(np.sin(2 * np.pi * 1573.4 * np.arange(n) / FS), preset, seed=11)
    a = np.sqrt(2 * 50 / (FS / 2) * 10 ** (inr / 10)) / np.sqrt(2 * np.mean(f ** 2))
    y = np.random.default_rng(12).normal(size=n) + a * f
    py, cc = cwid.CwSense(), native.cwid.CwSense()
    for i in range(0, len(y), 80):
        assert py.feed(y[i:i + 80]) == cc.feed(y[i:i + 80]) and py.hits_from == cc.hits_from, i


def _cw_tx(path):
    return [e for e in map(json.loads, open(path / "events.jsonl")) if e["kind"] == "tx" and e["submode"] == "cw"]


@pytest.mark.parametrize("cpp_calls", [True, False], ids=["cpp-calls-python", "python-calls-cpp"])
def test_cw_ids_between_cpp_and_python_engines(native, reference, cpp_calls, tmp_path):
    """CW IDs in place of ID frames, each way: the other implementation
    holds for them (nothing times out) and the transfer completes."""
    cpp = native.engine.Engine("W1AW", seed=1, record_dir=str(tmp_path / "cpp"))
    py = reference(E, "Engine")("K2XYZ", seed=2, record_dir=tmp_path / "py")
    a, b = (cpp, py) if cpp_calls else (py, cpp)
    for e in (a, b):
        e.id_interval_s, e.cw_id = 3.0, "only"
    b.listen()
    a.connect(b.call, 2)
    assert link(a, b, 12, 60, lambda: a.session.state == S.CONNECTED and b.session.state == S.CONNECTED)
    up, down = bytes(range(256)) * 30, bytes(range(255, -1, -1)) * 15
    a.session.write(up)
    b.session.write(down)
    got_a, got_b = bytearray(), bytearray()

    def done():
        got_a.extend(a.session.read())
        got_b.extend(b.session.read())
        return len(got_b) >= len(up) and len(got_a) >= len(down)
    assert link(a, b, 12, 300, done, seed=3)
    assert bytes(got_b) == up and bytes(got_a) == down
    assert a.session.station.stats.get("timeouts", 0) == 0
    assert _cw_tx(tmp_path / "cpp") and _cw_tx(tmp_path / "py")


def test_cw_id_level_matches(native, reference):
    assert _cw_id_peak(native.engine.Engine("W1AW", seed=1)) == pytest.approx(
        _cw_id_peak(reference(E, "Engine")("W1AW", seed=1)), rel=1e-9)
