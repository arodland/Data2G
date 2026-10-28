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
