"""native arq (native/core/arq) against data2g/arq/{frames,link,session}.py.

The C++ is a line-for-line port, so the test is equality, not tolerance:
the same frames (and log lines) for the same inputs. Codecs are compared value for value;
two stations or two sessions are run through test_arq's fake channel
(loss, soft combining, duplicated receptions) as Python/Python,
C++/C++ and both mixed pairs, and every burst sent must be identical
across all four. The policies and random.Random stay Python objects that
the C++ calls in Python's order, so seeded runs draw the same numbers.
"""

import logging
import random
import zlib

import pytest

import conftest
from data2g.arq import frames as F
from data2g.arq import link as L
from data2g.arq import session as S

from test_arq import MODES, FakeRx, RandomPolicy, payload, text
from test_arq_session import Policy as SessionPolicy
from test_arq_session import run as session_run


def logged(caplog, fn, *args, **kw):
    """fn's result and the data2g.link / data2g.session lines it logged."""
    caplog.clear()
    out = fn(*args, **kw)
    return out, [(r.name, r.levelno, r.getMessage()) for r in caplog.records if r.name.startswith("data2g.")]


@pytest.fixture
def pure(monkeypatch):
    """The Python modules as written, whether or not --native substituted them."""
    for (module, attr), fn in conftest._originals.items():
        monkeypatch.setattr(module, attr, fn)


# --- codecs ---------------------------------------------------------------------------

def test_core_and_control(native, pure):
    a, rng = native.arq, random.Random(0)
    for _ in range(300):
        kw = dict(ftype=rng.randrange(4), n_ctl=rng.randint(1, 4), burst_seq=rng.randrange(8), acted_on=rng.randrange(8),
                  cum=rng.randrange(128), reply_lost=rng.random() < 0.5, k=rng.randrange(64),
                  recommend=rng.randrange(64), size_hint=rng.randrange(4))
        assert a.Core(**kw).pack() == F.Core(**kw).pack()
        raw = bytes(rng.randrange(256) for _ in range(rng.randint(0, 6)))
        assert vars(F.Core.unpack(raw)) == {k: getattr(a.Core.unpack(raw), k) for k in kw}
        ext = {t: bytes(rng.randrange(256) for _ in range(rng.randint(0, 12)))
               for t in rng.sample([F.T_NEW, F.T_RV, F.T_BITMAP, F.T_COMP, 200], rng.randint(0, 4))}
        for pb in (4, 20, 46):
            try:
                want = F.Control(F.Core(**kw), dict(ext)).pack(pb)
            except ValueError:
                with pytest.raises(ValueError):
                    a.Control(a.Core(**kw), dict(ext)).pack(pb)
                continue
            core = a.Core(**kw)
            assert a.Control(core, dict(ext)).pack(pb) == want and core.n_ctl == len(want)
            got = a.Control.unpack(want)
            assert got.ext == F.Control.unpack(want).ext and got.core.k == kw["k"]
    junk = [bytes([0, 0, 0, 0, F.T_NEW, 9, 1])]
    for codec in (F.Control, a.Control):
        with pytest.raises(ValueError):
            codec.unpack(junk)


def test_extension_payloads_callsigns_records(native, pure):
    a, rng = native.arq, random.Random(1)
    for _ in range(300):
        cum = rng.randrange(128)
        got = {rng.randrange(128) for _ in range(rng.randrange(20))}
        assert a.pack_bitmap(got, cum) == F.pack_bitmap(got, cum)
        b = bytes(rng.randrange(256) for _ in range(rng.randrange(10)))
        assert a.unpack_bitmap(b, cum) == F.unpack_bitmap(b, cum)
        rvs = [rng.randrange(4) for _ in range(rng.randrange(20))]
        assert a.pack_rv(rvs) == F.pack_rv(rvs)
        k = rng.randrange(4 * len(b) + 1)
        assert a.unpack_rv(b, k) == F.unpack_rv(b, k)
        flags = [rng.random() < 0.3 for _ in range(rng.randrange(30))]
        assert a.pack_flags(flags) == F.pack_flags(flags)
        assert a.unpack_flags(b, k) == F.unpack_flags(b, k)
    with pytest.raises(ValueError):  # T_RV shorter than k resends: Python's negative shift
        F.unpack_rv(b"\x00", 5)
    with pytest.raises(ValueError):
        a.unpack_rv(b"\x00", 5)
    for call in ("W1AW", "vk2abc-15", "G4ABC/P", "K", "AB12345678"):
        assert a.pack_call(call) == F.pack_call(call) and a.unpack_call(F.pack_call(call)) == call.upper()
    for bad in ("TOOLONGCALL1", "W1 AW", "W1_AW"):
        for codec in (F, a):
            with pytest.raises(ValueError):
                codec.pack_call(bad)
    data = bytes(rng.randrange(256) for _ in range(1000))
    assert a.to_records(data) == F.to_records(data)
    stream = F.to_records(data[:300]) + bytes(5) + F.to_records(data[300:])
    rp, rc = F.RecordReader(), a.RecordReader()
    for i in range(0, len(stream), 7):
        assert rc.feed(stream[i:i + 7]) == rp.feed(stream[i:i + 7])
    assert rc.delivered == rp.delivered == len(data)


def test_session_key_and_frame_desc(native, pure):
    for caller, callee, nonce in (("W1AW", "K2XYZ-7", 0), ("G4ABC/P", "VK2ABC", 65535), ("A", "B", 1234)):
        assert native.arq.session_key(caller, callee, nonce) == S.session_key(caller, callee, nonce)
    body = bytes([F.CONNECT, S.VERSION]) + F.pack_call("W1AW") + F.pack_call("K2XYZ") + bytes(4)
    for b in (body, bytes([F.DISC]), bytes([F.DISC_ACK]), bytes([99])):
        assert native.arq._frame_desc(b) == S._frame_desc(b)


# --- compression: interop with Python's zlib --------------------------------------------

def _cases():
    rng = random.Random(2)
    for i in range(60):
        hist = text(rng, rng.choice([0, 100, 4096])) if i % 3 else bytes(rng.randrange(256) for _ in range(500))
        data = payload(rng, rng.randint(1, 900), ["text", "mixed", "random"][i % 3])
        yield hist, data


def test_deflate_interoperates_with_python_zlib(native, pure):
    """Each side inflates the other's deflate. Compressed bytes are allowed to
    differ (only the peer decodes them); with the same zlib they don't."""
    a, same = native.arq, 0
    for hist, data in _cases():
        zc, zp = a.deflate(hist, data), F.deflate(hist, data)
        assert F.inflate(hist, zc + bytes(7)) == data
        assert a.inflate(hist, zp + bytes(7)) == data
        same += zc == zp
        for pb in (22, 46, 120):
            fc, fp = a.deflate_fit(hist, data, pb), F.deflate_fit(hist, data, pb)
            assert (fc is None) == (fp is None)
            if fc:
                assert F.inflate(hist, fc[1]) == data[:fc[0]]
                assert fc == fp or zc != zp
    print(f"identical deflate output: {same} of 60 (zlib {zlib.ZLIB_RUNTIME_VERSION} in Python)")
    assert same == 60 or zlib.ZLIB_RUNTIME_VERSION.split(".")[:2] != ["1", "3"]


def test_inflate_rejects_as_python_does(native, pure):
    hist = b"some history " * 50
    z = F.deflate(hist, text(random.Random(3), 2000))
    for bad in (z[:len(z) // 2], b"\xff" * 30, b"", F.deflate(b"", bytes(70000))):
        with pytest.raises(ValueError) as want:
            F.inflate(hist, bad)
        with pytest.raises(ValueError) as got:
            native.arq.inflate(hist, bad)
        assert str(got.value) == str(want.value)


# --- two stations through the fake channel ---------------------------------------------

def _burst(b):
    return b.submode, b.burst_seq, [(tuple(s.mask_id), s.rv, bytes(s.payload)) for s in b.slots]


def lockstep(seed, cls_a, cls_b, p_burst, p_cw, n_a, n_b, kind="random", modes=("m4", "m22", "m46"), max_cw=20,
             p_dup=0.1, max_turns=3000):
    """test_arq.run with chosen classes and duplicated receptions: -> (trace
    of every burst sent, outcome)."""
    rng = random.Random(seed)
    data_a, data_b = payload(rng, n_a, kind), payload(rng, n_b, kind)
    a = cls_a(0, RandomPolicy(random.Random(seed + 1), 0.2, modes, max_cw), master=True)
    b = cls_b(1, RandomPolicy(random.Random(seed + 2), 0.2, modes, max_cw))
    a.write(data_a)
    b.write(data_b)
    stores, stats, trace = {0: {}, 1: {}}, {"mismatch": 0}, []
    got_a, got_b = bytearray(), bytearray()
    burst, sender = a.build(), a
    for _ in range(max_turns):
        trace.append((sender.direction, _burst(burst)))
        receiver = b if sender is a else a
        ok = False
        if rng.random() >= p_burst:
            ok = receiver.handle(FakeRx(burst, rng, p_cw, stores[receiver.direction], stats))
            if rng.random() < p_dup:  # heard twice (a repeat that crossed)
                ok = receiver.handle(FakeRx(burst, rng, p_cw, stores[receiver.direction], stats)) or ok
        got_a += a.read()
        got_b += b.read()
        assert bytes(got_b) == data_a[:len(got_b)] and bytes(got_a) == data_b[:len(got_a)]
        if a.state == L.FAILED or b.state == L.FAILED:
            return trace, ("failed", a.fail_reason or b.fail_reason)
        if got_a == data_b and got_b == data_a and not a.tx.pending() and not b.tx.pending():
            return trace, ("done", a.tx.acked, b.tx.acked, dict(a.stats), dict(b.stats), stats["mismatch"])
        if ok:
            burst, sender = receiver.build(), receiver
            receiver.answered()
        else:
            burst, sender = a.on_timeout(), a
            if burst is None:
                return trace, ("failed", a.fail_reason)
    return trace, ("stuck",)


CELLS = [(0.0, 0.0, "random"), (0.1, 0.1, "text"), (0.2, 0.2, "mixed"), (0.3, 0.3, "mixed"), (0.5, 0.5, "text")]


@pytest.mark.parametrize("seed", range(20))
def test_stations_send_identical_bursts(native, pure, caplog, seed):
    """Abandons, resyncs, repeats, duplicated control, compression and lost
    links all occur over these seeds."""
    caplog.set_level(logging.INFO)
    p_burst, p_cw, kind = CELLS[seed % len(CELLS)]
    modes = [("m4", "m22", "m46"), ("m4", "m46", "c60", "c40")][seed % 2]
    py, cc = L.Station, native.arq.Station
    args = (p_burst, p_cw, 6000, 2000, kind, modes, [20, 64, 3][seed % 3])
    want, want_log = logged(caplog, lockstep, 900 + seed, py, py, *args)
    assert want[1][0] in ("done", "failed")
    got, got_log = logged(caplog, lockstep, 900 + seed, cc, cc, *args)
    assert got == want and got_log == want_log
    for pair in ((py, cc), (cc, py)):
        got = lockstep(900 + seed, *pair, *args)
        assert got[1] == want[1]
        assert got[0] == want[0]


def test_compression_off_matches(native, pure, monkeypatch):
    """link.COMPRESS read at each build, by the C++ too."""
    monkeypatch.setattr(L, "COMPRESS", False)
    py, cc = L.Station, native.arq.Station
    want = lockstep(5, py, py, 0.1, 0.1, 2000, 500, "text")
    assert want[1][0] == "done" and want[1][3]["cw_comp"] == 0
    assert lockstep(5, cc, cc, 0.1, 0.1, 2000, 500, "text") == want


# --- two sessions on the simulated clock ----------------------------------------------------

def recording(cls, trace, tag, dup_seed, p_dup):
    """cls with every burst it sends recorded, and some bursts it hears
    delivered twice."""
    dup = random.Random(dup_seed)

    class Recording(cls):
        def poll(self, now):
            b = super().poll(now)
            if b is not None:
                trace.append((tag, now, _burst(b)))
            return b

        def on_rx(self, rx, now):
            super().on_rx(rx, now)
            if dup.random() < p_dup:
                super().on_rx(rx, now)

    return Recording


def session_pair(monkeypatch, seed, cls_a, cls_b, p_dup=0.1, **kw):
    trace = []
    made = iter([recording(cls_a, trace, "a", seed + 10, p_dup), recording(cls_b, trace, "b", seed + 11, p_dup)])
    monkeypatch.setattr(S, "Session", lambda *a, **k: next(made)(*a, **k))
    r = session_run(seed, **kw)
    return trace, (r["got_a"], r["got_b"], r["t"], r["a"].state, r["b"].state, r["a"].close_reason,
                   r["b"].close_reason, r["mismatch"], r["collisions"], list(r["a"].events), list(r["b"].events))


@pytest.mark.parametrize("seed", range(12))
def test_python_and_cpp_sessions_interoperate(native, pure, monkeypatch, caplog, seed):
    """One Python and one C++ session (either way round) connect, transfer
    both ways and disconnect through loss and duplicated receptions, sending
    exactly the bursts two Python sessions send."""
    py, cc = S.Session, native.arq.Session
    kw = dict(p_burst=[0.0, 0.05, 0.15, 0.25][seed % 4], p_cw=[0.0, 0.1, 0.2][seed % 3])
    if seed in (7, 11):
        kw.update(n_a=50000, die_at=20.0)  # a dead link: both close, bounded
    caplog.set_level(logging.INFO)
    want, want_log = logged(caplog, session_pair, monkeypatch, seed, py, py, **kw)
    if seed not in (7, 11):
        assert want[1][0] and want[1][1] and want[1][3] == want[1][4] == S.CLOSED
    got, got_log = logged(caplog, session_pair, monkeypatch, seed, cc, cc, **kw)
    assert got == want and got_log == want_log
    for pair in ((py, cc), (cc, py)):
        got = session_pair(monkeypatch, seed, *pair, **kw)
        assert got[1] == want[1]
        assert got[0] == want[0]


def test_idle_wakes_and_chat_match(native, pure, monkeypatch):
    """The callee's wake (random backoff from the session's random.Random)
    and the caller's keepalive, lost wakes included."""
    py, cc = S.Session, native.arq.Session
    for kw in (dict(n_a=200, n_b=40, b_write_at=120.0, ack_loss_first=2),
               dict(n_a=200, n_b=40, b_write_at=120.0, ack_loss_first=4, chat=True)):
        want = session_pair(monkeypatch, 3, py, py, p_dup=0.0, **kw)
        for pair in ((cc, cc), (py, cc), (cc, py)):
            assert session_pair(monkeypatch, 3, *pair, p_dup=0.0, **kw) == want


def test_session_policy_is_the_python_object(native):
    pol = SessionPolicy(random.Random(1))
    s = native.arq.Session("W1AW", pol, rng=random.Random(2))
    assert s.policy is pol and s.state == "idle" and s.station is None
    s.connect("k2xyz", 2, 0.0)
    assert s.state == "connecting" and s.peer == "K2XYZ" and s._master
    b = s.poll(0.0)
    assert b.submode == "m46" and isinstance(b, L.TxBurst) and s.next_event() is None
    s.on_tx_end(b, 1.0)
    assert s.next_event() == 1.0 + s.t_turn + S.REPLY_START_S
    assert MODES["m46"][0] == len(b.slots[0].payload)


def test_session_on_timeout_override_and_tuning(native, monkeypatch):
    """scripts/linksim.py wraps a session's _on_timeout to count timeouts and
    scripts/idle_study.py patches session.py's constants: both reach the C++
    session, as they do the Python one."""
    monkeypatch.setattr(S, "REPLY_START_S", 4.0)
    monkeypatch.setattr(S, "CONNECT_TRIES", 2)
    for cls in (conftest._originals.get((S, "Session"), S.Session), native.arq.Session):
        s = cls("W1AW", SessionPolicy(random.Random(1)), rng=random.Random(2))
        seen, ot = [], s._on_timeout
        s._on_timeout = lambda now, ot=ot: (seen.append(now), ot(now))
        s.connect("K2XYZ", 2, 0.0)
        s.on_tx_end(s.poll(0.0), 1.0)
        assert s.next_event() == 1.0 + s.t_turn + 4.0
        s.poll(6.0)
        s.on_tx_end(s.poll(s.next_event()), 10.0)
        s.poll(s.next_event())
        assert seen == [6.0, 10.0 + s.t_turn + 4.0] and s.state == S.CLOSED and s.close_reason == "no answer"
