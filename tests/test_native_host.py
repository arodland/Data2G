"""data2g.host.Host: the C++ Host (native/core/host) against the Python one,
both over the C++ engine in its Python-compatible sync mode, so the same
seeds give the same session. Every command reply, notification and data
byte, step by step, must be identical. Skips if the module isn't built;
`pytest --native` errors instead."""

import conftest
import numpy as np
import pytest

from data2g import host
from data2g.arq import engine as E
from data2g.arq import session as S
from data2g.config import FS
from test_engine import BLOCK, link


@pytest.fixture
def classes(native, reference):
    engine = conftest._engine_substitutions(native)[(E, "Engine")]
    return engine, {"python": reference(host, "Host"), "cpp": conftest._host_substitutions(native)[(host, "Host")]}


def transcript(Engine, Host, credit=None):
    """A scripted pair of stations; everything each Host gave back, per step."""
    a, b = Host(Engine("NOCALL", seed=61), credit), Host(Engine("NOCALL", seed=62), credit)
    out, seen = [], {"a": [], "b": []}

    def take(tag):
        for name, h in (("a", a), ("b", b)):
            seen[name] += h.out_cmd
            if h.out_cmd or h.out_data:
                out.append((tag, name, list(h.out_cmd), bytes(h.out_data)))
            h.out_cmd.clear()
            h.out_data.clear()

    def pump(until):
        def f():
            a.after_step(bool(a.engine.tx))
            b.after_step(bool(b.engine.tx))
            take("step")
            return until()
        return f

    for line in ("VERSION", "MYCALL K2XYZ K2XYZ-1", "LISTEN CQ", "LISTEN ON", "BW1200", "CHAT OFF", "COMPRESSION ON",
                 "FOO", "", "LISTEN MAYBE", "CQFRAME K2XYZ 9999"):
        b.command(line)
    a.command("CQFRAME W1AW 500")
    take("cmd")
    assert link(a.engine, b.engine, 12, 30, pump(lambda: "CQFRAME W1AW 500" in seen["b"]), seed=1)
    a.command("bw500")
    a.command("CONNECT W1AW K2XYZ-1")
    a.command("CONNECT W1AW K2XYZ-1")  # a session under way: WRONG
    take("cmd")
    assert link(a.engine, b.engine, 12, 60, pump(lambda: b.engine.session.state == S.CONNECTED
                                                 and a.engine.session.state == S.CONNECTED), seed=2)
    up, down = np.random.default_rng(3).bytes(1500), b"F> BC\r" * 30
    a.data_in(up[:700])
    a.data_in(up[700:])
    b.data_in(down)
    take("data")
    got_a, got_b = bytearray(), bytearray()

    def done():
        got_a.extend(a.out_data)
        got_b.extend(b.out_data)
        return len(got_b) >= len(up) and len(got_a) >= len(down)

    def collect():  # take() would drop the data before done() sees it
        a.after_step(bool(a.engine.tx))
        b.after_step(bool(b.engine.tx))
        r = done()
        take("step")
        return r
    assert link(a.engine, b.engine, 12, 240, collect, seed=4)
    assert bytes(got_b) == up and bytes(got_a) == down
    b.client_gone()
    take("gone")
    assert link(a.engine, b.engine, 12, 60, pump(lambda: a.engine.session.state in (S.IDLE, S.CLOSED)
                                                 and b.engine.session.state in (S.IDLE, S.CLOSED)), seed=5)
    a.command("CONNECT W1AW K2XYZ")  # nobody listening now: tries, then ABORT
    assert link(a.engine, b.engine, 12, 5, pump(lambda: False), seed=6) is False
    a.command("ABORT")
    take("abort")
    for _ in range(int(1.2 * host.ALIVE_S * FS / BLOCK)):  # IAMALIVE, BUFFER repeats
        a.engine.step(np.zeros(BLOCK))
        a.after_step(False)
    take("idle")
    return out


@pytest.mark.parametrize("credit", [None, 100, 0])
def test_cpp_host_matches_python(classes, credit):
    Engine, hosts = classes
    py, cpp = (transcript(Engine, hosts[k], credit) for k in ("python", "cpp"))
    flat = [line for _, _, cmd, _ in py for line in cmd]
    for want in ("VERSION Data2G 0.1", "CQFRAME W1AW 500", "CONNECTED W1AW K2XYZ-1 500", "DISCONNECTED", "BUSY ON",
                 "PTT ON", "IAMALIVE", "WRONG"):
        assert want in flat, want
    assert any(line.startswith("MODE ") for line in flat) and any(line.startswith("BUFFER ") for line in flat)
    for i, (x, y) in enumerate(zip(py, cpp)):
        assert x == y, (i, x, y)
    assert len(py) == len(cpp)


def test_constants(native):
    N = native.host
    assert N.VERSION == host.VERSION and N.ALIVE_S == host.ALIVE_S and N.BUFFER_REPEAT_S == host.BUFFER_REPEAT_S
    assert all(N.ignored(c) for c in host.IGNORED) and not N.ignored("VERSION")
    assert {c: N.bw_cap(c) for c in host.BW} == host.BW and N.bw_cap("BW9") is None
