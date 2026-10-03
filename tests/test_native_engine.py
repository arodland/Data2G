"""The C++ Engine (pure C++: GearPolicy, C++ random draws, its own TX
audio) against the Python Engine through test_engine's simulated channel,
both directions, a VARA session and KISS: the port plan's Phase 2 exit. And
the recorder's files, read the way the analysis scripts read them."""

import json

import numpy as np
import pytest

from data2g import kisslink
from data2g.arq import engine as E
from data2g.arq import session as S
from test_engine import link
from test_kiss import frame


def _pair(native, reference, cpp_calls, tmp_path, **kw):
    cpp = native.engine.Engine("W1AW", seed=1, record_dir=str(tmp_path / "cpp"), **kw.get("cpp", {}))
    py = reference(E, "Engine")("K2XYZ", seed=2, record_dir=tmp_path / "py", **kw.get("py", {}))
    return (cpp, py) if cpp_calls else (py, cpp)


@pytest.mark.parametrize("cpp_calls", [True, False], ids=["cpp-calls-python", "python-calls-cpp"])
def test_a_session_between_cpp_and_python_engines(native, reference, cpp_calls, tmp_path):
    a, b = _pair(native, reference, cpp_calls, tmp_path)
    b.listen()
    a.connect(b.call, 2)
    assert link(a, b, 12, 60, lambda: a.session.state == S.CONNECTED and b.session.state == S.CONNECTED)
    rng = np.random.default_rng(3)
    up = rng.bytes(1500) + (b"CQ CQ de W1AW QTH FN31 RST 599 GM OM 73 " * 40)[:1500]
    down = rng.bytes(400)
    a.session.write(up)
    b.session.write(down)
    got_a, got_b = bytearray(), bytearray()

    def done():
        got_a.extend(a.session.read())
        got_b.extend(b.session.read())
        return len(got_b) >= len(up) and len(got_a) >= len(down)
    assert link(a, b, 12, 240, done, seed=1)
    assert bytes(got_b) == up and bytes(got_a) == down
    a.session.disconnect()
    assert link(a, b, 12, 60, lambda: a.session.state == S.CLOSED and b.session.state == S.CLOSED, seed=2)
    for e, peer in ((a, b), (b, a)):
        ev = e.events()
        assert ev[0] == f"CONNECTED {peer.call}" and ev[-1].startswith("DISCONNECTED"), ev
    _same_records(tmp_path)


def _same_records(d):
    """The C++ recording has the Python one's files, kinds and fields."""
    keys = {}
    for side in ("cpp", "py"):
        ev = [json.loads(line) for line in open(d / side / "events.jsonl")]
        keys[side] = {e["kind"]: set(e) for e in ev}
        rx = [e for e in ev if e["kind"] == "rx"]
        assert rx and all(e["meas"] is None or set(e["meas"]) >= {"snr_est", "mi_gray-qam4"} for e in rx)
        assert all(e["noise"] is None or len(e["noise"]["noise_db"]) == 5 for e in rx)
        for e in rx:
            audio = np.load(d / side / e["file"])["audio"]
            assert audio.dtype == np.float32 and audio.ndim == 1 and len(audio)
        tx = [e for e in ev if e["kind"] == "tx"]
        assert tx and all(len(s["mask"]) == 3 and bytes.fromhex(s["payload"]) for e in tx for s in e["slots"])
        f16 = np.fromfile(d / side / "audio_in.f16", dtype=np.float16)
        assert len(f16) % (E.FS // 10) == 0 and np.isfinite(f16).all()
    assert keys["cpp"] == keys["py"]


@pytest.mark.parametrize("cpp_calls", [True, False], ids=["cpp-calls-python", "python-calls-cpp"])
def test_id_frames_between_cpp_and_python_engines(native, reference, cpp_calls, tmp_path):
    """ID frames (docs/arq.md §7a) each way: ahead of a turn when due, and
    once more after the session, heard by the other implementation."""
    a, b = _pair(native, reference, cpp_calls, tmp_path)
    a.id_interval_s = b.id_interval_s = 3.0
    b.listen()
    a.connect(b.call, 2)
    assert link(a, b, 12, 60, lambda: a.session.state == S.CONNECTED and b.session.state == S.CONNECTED)
    key = a.session.station.key
    up, down = bytes(range(256)) * 40, bytes(range(255, -1, -1)) * 20
    a.session.write(up)
    b.session.write(down)
    got_a, got_b, ev_a, ev_b = bytearray(), bytearray(), [], []

    def poll():
        got_a.extend(a.session.read())
        got_b.extend(b.session.read())
        ev_a.extend(a.events())
        ev_b.extend(b.events())
    assert link(a, b, 12, 240, lambda: poll() or (len(got_b) >= len(up) and len(got_a) >= len(down)), seed=3)
    assert bytes(got_b) == up and bytes(got_a) == down
    assert f"ID {a.call} {key}" in ev_b and f"ID {b.call} {key}" in ev_a
    a.session.disconnect()
    assert link(a, b, 12, 60, lambda: poll() or (a.session.state == S.CLOSED and b.session.state == S.CLOSED), seed=4)
    n_a, n_b = ev_a.count(f"ID {b.call} {key}"), ev_b.count(f"ID {a.call} {key}")
    assert link(a, b, 12, 30, lambda: poll() or (ev_a.count(f"ID {b.call} {key}") > n_a
                                                 and ev_b.count(f"ID {a.call} {key}") > n_b), seed=5)
    _same_records(tmp_path)


def test_kiss_between_cpp_and_python_engines(native, reference):
    cpp = native.engine.Engine("W1AW", seed=3, kiss=native.kisslink.KissLink())
    py = reference(E, "Engine")("K2XYZ", seed=4, kiss=kisslink.KissLink())
    heard = []
    to_py, to_cpp = frame("APRS", "W1AW", 0x03, b"!from C++"), frame("APRS", "K2XYZ", 0x03, b"!from Python")
    cpp.kiss.enqueue(to_py)
    assert link(cpp, py, 12, 30, lambda: to_py in py.kiss_rx)
    py.kiss.enqueue(to_cpp)
    assert link(cpp, py, 12, 30, lambda: heard.extend(cpp.take_kiss_rx()) or to_cpp in heard, seed=1)


def test_float16_and_json_numbers_as_numpy_and_json_write_them(native):
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 10000), rng.normal(0, 1e-5, 1000), [0.0, -0.0, 65504.0, 65519.9, 65520.0,
                        1e6, -1e6, 2.0 ** -24, 2.0 ** -25, 1 + 2.0 ** -11, 1 + 3 * 2.0 ** -11, np.inf, -np.inf]])
    with np.errstate(over="ignore"):  # 1e6 to inf, as meant
        want = x.astype(np.float16).view(np.uint16)
    assert np.array_equal(native.engine.to_f16(x), want)
    for v in (0.0, 1.0, 60.0, -2.5, 1e-5, 0.0001, 123456.789, 1e16, 1.5e300, 1 / 3, float("nan"), float("inf")):
        assert native.engine.json_num(v) == json.dumps(v)


def test_worker_mode_steps_and_stops_without_the_gil(native):
    """Worker mode (real time only, so no session here: test_engine.cpp
    holds one): steps release the GIL, and deleting the engine joins its
    worker."""
    e = native.engine.Engine("W1AW", seed=5, worker=True)
    rng = np.random.default_rng(6)
    for _ in range(30):
        out, ptt = e.step(rng.normal(0, 0.1, E.FS // 10))
        assert len(out) == E.FS // 10 and not ptt
    with pytest.raises(ValueError):
        e.receiver = object()
    del e
    with pytest.raises(ValueError):
        native.engine.Engine("W1AW", policy=lambda: None, worker=True)
