"""Phase 2 exit check (docs/native-port-plan.md): the ARQ state-agreement
fuzz (docs/arq.md §10) on mixed Python / C++ pairs.

Seeded random sessions in all four pairings (Py-Py, Py-C++, C++-Py,
C++-C++), at three levels:
- link: two Stations in lockstep through test_arq's fake channel;
- session: two Sessions on test_arq_session's simulated clock;
- engine: two Engines through test_engine's simulated audio channel.

Impairments, drawn per seed: asymmetric burst loss, codeword loss with
soft combining, duplicated receptions, late copies of a sender's earlier
burst (reordering; the in-order engine never shows it), delayed decode,
long fades, a peer that restarts mid-session (session: a fresh Session;
engine: abort), and CRC-valid but corrupted control codewords (link and
session, own tests).

Reference behaviours found here, the same in every pairing, are kept as
strict xfails rather than fixed (docs/native-port-plan.md, Findings):
malformed control raising, a flipped T_COMP bit, and reordered bursts
corrupting the stream (a run that corrupts after a late copy ends with
outcome LATE; anything else that corrupts fails).

Every run asserts:
(a) it ends in delivery or a bounded disconnect. A station's handled
    bursts without progress while it has data queued never exceed the
    watchdog's NO_PROGRESS_TURNS x (RESYNCS_BEFORE_FAIL + 1); turns
    without progress never exceed that times the dead-link bound; a
    session never stays connected LINK_LOST_S past the last control it
    decoded.
(b) each delivered stream is a prefix of what was sent at every point,
    and all of it at the end of a delivered run.
(c) link and session: every pairing sends exactly the bursts Py-Py sends
    (the C++ classes call the same Python policies and random.Random).
    The engine level runs the pure C++ Engine (its own policy and draws),
    so there only (a) and (b) apply.

The default run is a sample; `-m slow` runs the long sweep.
"""

import random

import numpy as np
import pytest

from data2g.arq import engine as E
from data2g.arq import frames as F
from data2g.arq import link as L
from data2g.arq import session as S
from data2g.config import SNR_REF_BW_HZ

from test_arq import MODES, FakeRx, RandomPolicy, payload
from test_arq_session import DECODE_S, HEADER_S, PTT_S
from test_arq_session import Policy as SessionPolicy
from test_engine import BLOCK
from test_native_arq_link import _burst, pure  # noqa: F401 (fixture)

WATCHDOG = L.NO_PROGRESS_TURNS * (L.RESYNCS_BEFORE_FAIL + 1)  # handled bursts without progress, then FAILED
# turns between progress: each counted handle can be separated by at most
# LINK_LOST_MISSES + 1 timeouts, and one more burst from the peer
TURN_BOUND = (WATCHDOG + 1) * (L.LINK_LOST_MISSES + 2)
KNOWN_RAISE = ("a CRC-valid malformed control frame raises out of Station.handle / Session.on_rx in both "
               "implementations (docs/native-port-plan.md, Findings): kept as reference behaviour pending a decision")


def impl(native, which):
    """(Station, Session) of one implementation. Python's are the
    originals, whether or not --native substituted them (the `pure`
    fixture has restored them by the time this runs)."""
    return (L.Station, S.Session) if which == "py" else (native.arq.Station, native.arq.Session)


PAIRS = [("py", "py"), ("py", "cpp"), ("cpp", "py"), ("cpp", "cpp")]


class Channel(FakeRx):
    """FakeRx, plus: one control codeword corrupted (CRC still passing)
    with probability p_corrupt. The draw is always made, so loss patterns
    don't depend on it."""

    def __init__(self, burst, rng, p_cw, store, stats, p_corrupt=0.0):
        super().__init__(burst, rng, p_cw, store, stats)
        self.bad = None
        if rng.random() < p_corrupt:
            n_ctl = sum(s.mask_id[2] >= F.SEQ_MOD for s in burst.slots)
            i = rng.randrange(n_ctl)
            n = len(burst.slots[i].payload)
            # mostly past the 4-byte core: the TLVs are where malformed lives
            pos = rng.randrange(4, n) if n > 4 and rng.random() < 0.7 else rng.randrange(n)
            self.bad = (i, pos, rng.randrange(1, 256))
            stats["corrupted"] = stats.get("corrupted", 0) + 1

    def decode(self, i, mask_id, rv, key):
        p = super().decode(i, mask_id, rv, key)
        if p is not None and self.bad and i == self.bad[0] and mask_id[2] >= F.SEQ_MOD:
            b = bytearray(p)
            b[self.bad[1]] ^= self.bad[2]
            p = bytes(b)
        return p


def progress_of(stations):
    """Rises whenever any stream moves: a delivery, or a base advanced."""
    return sum(s.rx.cum + s.tx.base for s in stations if s is not None)


# --- scenarios ----------------------------------------------------------------------------

def link_scenario(seed, corrupt=False):
    r = random.Random(seed * 7919 + 17)
    p_ab = r.choice([0.0, 0.0, 0.05, 0.1, 0.2, 0.3])
    sc = dict(
        seed=seed, p=(p_ab, p_ab if r.random() < 0.5 else r.choice([0.0, 0.05, 0.2, 0.4, 0.6])),
        p_cw=r.choice([0.0, 0.05, 0.1, 0.2, 0.3]), p_dup=r.choice([0.0, 0.05, 0.2]),
        p_late=r.choice([0.0, 0.0, 0.0, 0.03, 0.1]),
        fade=(r.randrange(0, 150), r.choice([3, 8, 12, 30])) if r.random() < 0.3 else None,
        kind=r.choice(["random", "text", "mixed"]),
        modes=r.choice([("m4", "m22", "m46"), ("m4", "m46", "c60", "c40"), ("m22",), ("m46", "c60")]),
        max_cw=r.choice([2, 3, 10, 20, 64]), change=r.choice([0.05, 0.2, 0.5]),
        n=(r.randrange(0, 6000), r.randrange(0, 2500)), p_corrupt=0.0)
    if corrupt:
        sc.update(p_corrupt=r.choice([0.02, 0.05, 0.1]), p_late=0.0, fade=None)
    return sc


def lockstep(sc, cls_a, cls_b, max_turns=30000):
    """Two stations, one burst per turn, the master retrying on timeouts.
    -> (trace of every burst sent, outcome, stats)."""
    seed = sc["seed"]
    rng = random.Random(seed)
    data_a, data_b = payload(rng, sc["n"][0], sc["kind"]), payload(rng, sc["n"][1], sc["kind"])
    pol = lambda k: RandomPolicy(random.Random(seed + k), sc["change"], sc["modes"], sc["max_cw"])
    a, b = cls_a(0, pol(1), master=True), cls_b(1, pol(2))
    a.write(data_a)
    b.write(data_b)
    stores, stats, trace = {0: {}, 1: {}}, {"mismatch": 0}, []
    got_a, got_b = bytearray(), bytearray()
    count = {0: 0, 1: 0}  # per station: handled bursts without progress, its data queued
    last_progress, prog = 0, progress_of((a, b))
    prev = {0: None, 1: None}  # each sender's previous burst (late copies)
    burst, sender = a.build(), a
    stats["max_count"] = stats["max_turns"] = 0
    for turn in range(max_turns):
        trace.append((sender.direction, _burst(burst)))
        receiver = b if sender is a else a
        fade = sc["fade"] and sc["fade"][0] <= turn < sc["fade"][0] + sc["fade"][1]
        heard = []
        if not fade and rng.random() >= sc["p"][sender.direction]:
            heard.append(burst)
            if rng.random() < sc["p_dup"]:
                heard.append(burst)
        if prev[sender.direction] is not None and rng.random() < sc["p_late"]:
            heard.append(prev[sender.direction])  # the previous burst, decoded late
            stats["late"] = stats.get("late", 0) + 1
        prev[sender.direction] = burst
        ok = False
        for x in heard:
            before = receiver.stats["rx_ok"]
            ok = receiver.handle(Channel(x, rng, sc["p_cw"], stores[receiver.direction], stats, sc["p_corrupt"])) or ok
            if receiver.state == L.FAILED:
                break
            now = progress_of((a, b))
            if now != prog:
                prog, last_progress, count = now, turn, {0: 0, 1: 0}
            elif receiver.stats["rx_ok"] > before and receiver.tx.pending():
                count[receiver.direction] += 1
                stats["max_count"] = max(stats["max_count"], count[receiver.direction])
                assert count[receiver.direction] <= WATCHDOG, ("no progress past the watchdog", turn, count)
        stats["max_turns"] = max(stats["max_turns"], turn - last_progress)
        assert turn - last_progress <= TURN_BOUND, ("turns without progress", turn, last_progress)
        got_a += a.read()
        got_b += b.read()
        if bytes(got_b) != data_a[:len(got_b)] or bytes(got_a) != data_b[:len(got_a)]:
            assert stats.get("late"), "stream corrupted"
            return trace, ("corrupted after a late burst",), stats  # test_late_burst_*
        if a.state == L.FAILED or b.state == L.FAILED:
            return trace, ("failed", a.fail_reason or b.fail_reason), stats
        if got_a == data_b and got_b == data_a and not a.tx.pending() and not b.tx.pending():
            return trace, ("done", a.tx.acked, b.tx.acked), stats
        if ok:
            burst, sender = receiver.build(), receiver
            receiver.answered()
        else:
            burst, sender = a.on_timeout(), a
            if burst is None:
                return trace, ("failed", a.fail_reason), stats
    return trace, ("stuck",), stats


def session_scenario(seed, corrupt=False):
    r = random.Random(seed * 104729 + 3)
    p_ab = r.choice([0.0, 0.0, 0.05, 0.15, 0.25])
    sc = dict(
        seed=seed, p=(p_ab, p_ab if r.random() < 0.5 else r.choice([0.0, 0.1, 0.3, 0.5])),
        p_cw=r.choice([0.0, 0.1, 0.2]), p_dup=r.choice([0.0, 0.05, 0.2]),
        p_late=r.choice([0.0, 0.0, 0.0, 0.03, 0.1]), delay=r.choice([0.0, 0.0, 0.5, 1.2]),
        fades=[(r.uniform(5, 300), r.choice([5.0, 20.0, 60.0, 120.0])) for _ in range(r.choice([0, 0, 1, 2]))],
        restart=(r.choice("ab"), r.uniform(3, 200)) if r.random() < 0.15 else None,
        n=(r.randrange(0, 4000), r.randrange(0, 1500)), chat=r.random() < 0.2, p_corrupt=0.0,
        b_write_at=r.uniform(30, 200) if r.random() < 0.2 else None)
    if corrupt:
        sc.update(p_corrupt=r.choice([0.02, 0.05, 0.1]), p_late=0.0, fades=[], restart=None)
    return sc


def sessions(sc, cls_a, cls_b, horizon=8000.0):
    """test_arq_session.run with the scenario's channel. -> (trace,
    outcome, stats)."""
    seed = sc["seed"]
    rng = random.Random(seed)
    data_a = bytes(rng.randrange(256) for _ in range(sc["n"][0]))
    data_b = bytes(rng.randrange(256) for _ in range(sc["n"][1]))
    a = cls_a("W1AW", SessionPolicy(random.Random(seed + 1)), rng=random.Random(seed + 2))
    b = cls_b("K2XYZ-7", SessionPolicy(random.Random(seed + 3)), rng=random.Random(seed + 4))
    for s in (a, b):
        s.set_chat(sc["chat"])
    b.listen()
    a.write(data_a)
    if sc["b_write_at"] is None:
        b.write(data_b)
    a.connect("K2XYZ-7", 2, 0.0)
    stores = {"a": {}, "b": {}}
    stats = {"mismatch": 0, "collisions": 0, "max_count": 0, "max_quiet": 0.0}
    trace, air, events = [], [], []
    who = {"a": a, "b": b}
    held = {"a": set(), "b": set()}
    got = {"a": bytearray(), "b": bytearray()}  # what a / b delivered (from the original instances only)
    orig = {"a": a, "b": b}
    count = {"a": 0, "b": 0}
    last_ok = {"a": 0.0, "b": 0.0}  # last control decoded by each, while connected
    prog = 0
    disconnect_asked = b_written = restarted = False
    t = 0.0
    while t < horizon:
        if sc["b_write_at"] is not None and not b_written and t >= sc["b_write_at"]:
            orig["b"].write(data_b)
            b_written = True
        if sc["restart"] and not restarted and t >= sc["restart"][1]:
            tag = sc["restart"][0]
            cls = cls_a if tag == "a" else cls_b
            fresh = cls(who[tag].call, SessionPolicy(random.Random(seed + 5)), rng=random.Random(seed + 6))
            if tag == "b":
                fresh.listen()
            who[tag], held[tag], restarted = fresh, set(), True
            events = [e for e in events if e[2] != tag]  # its pending receptions die with it
            stats["restarted"] = t
        for tag, other in (("a", "b"), ("b", "a")):
            me = who[tag]
            if held[tag]:
                continue
            burst = me.poll(t)
            if burst is None:
                continue
            trace.append((tag, round(t, 9), _burst(burst)))
            start = t + PTT_S
            busy = max([e for s, e, w in air[-4:] if w == other and e > start], default=None)
            if busy is not None:
                start = busy + 0.05
            end = start + me.policy.airtime(burst.submode, len(burst.slots))
            if any(s < end and start < e for s, e, _ in air[-4:]):
                stats["collisions"] += 1
            air.append((start, end, tag))
            events.append((end, "txend", tag, burst))
            lost = any(f0 <= start < f0 + fl for f0, fl in sc["fades"]) or rng.random() < sc["p"][tag == "b"]
            if not lost:
                held[other].add(id(burst))
                events.append((start + HEADER_S, "header", other, burst))
                events.append((end + DECODE_S + sc["delay"], "rx", other, burst))
                if rng.random() < sc["p_dup"]:
                    events.append((end + DECODE_S + sc["delay"], "rx", other, burst))
            if rng.random() < sc["p_late"]:  # a late copy, decoded after whatever comes next
                events.append((end + DECODE_S + sc["delay"] + rng.uniform(2.0, 6.0), "late", other, burst))
                stats["late"] = stats.get("late", 0) + 1
        for tag in "ab":
            got[tag] += orig[tag].read()
        if bytes(got["b"]) != data_a[:len(got["b"])] or bytes(got["a"]) != data_b[:len(got["a"])]:
            assert stats.get("late_heard"), "stream corrupted"
            return trace, ("corrupted after a late burst",), dict(stats, corrupted=True)  # test_late_burst_*
        if restarted:
            assert not who[sc["restart"][0]].read() or who[sc["restart"][0]] is orig[sc["restart"][0]]
        done = bytes(got["a"]) == data_b and bytes(got["b"]) == data_a and (b_written or sc["b_write_at"] is None)
        if done and not disconnect_asked:
            who["a"].disconnect()
            disconnect_asked = True
        if all(who[x].state in (S.CLOSED, S.IDLE, S.LISTEN) for x in "ab") and not events \
                and who["a"]._out is None and who["b"]._out is None:
            break
        for tag in "ab":
            if who[tag].state == S.CONNECTED:
                stats["max_quiet"] = max(stats["max_quiet"], t - last_ok[tag])
                # closes at LINK_LOST_S; a reply on air then may still land
                assert t - last_ok[tag] <= S.LINK_LOST_S + 15, ("connected past the dead-link bound", tag, t)
        nxt = [e[0] for e in events] + [x for x in (who[g].next_event() for g in "ab" if not held[g]) if x is not None]
        if sc["b_write_at"] is not None and not b_written:
            nxt.append(sc["b_write_at"])
        if sc["restart"] and not restarted:
            nxt.append(sc["restart"][1])
        if not nxt:
            break
        t = max(t, min(nxt))
        due = sorted([e for e in events if e[0] <= t], key=lambda e: e[0])
        events = [e for e in events if e[0] > t]
        for when, kind, tag, burst in due:
            me = who[tag]
            if kind == "txend":
                me.on_tx_end(burst, when)
                if not any(e[1] == "rx" and e[3] is burst for e in events):
                    for g in "ab":
                        held[g].discard(id(burst))
            elif kind == "header":
                me.on_header(burst.submode, len(burst.slots), when)
            else:
                if kind == "late":
                    stats["late_heard"] = stats.get("late_heard", 0) + 1
                st = me.station
                before = st.stats["rx_ok"] if st is not None else 0
                was = me.state
                me.on_rx(Channel(burst, rng, sc["p_cw"], stores[tag], stats, sc["p_corrupt"]), when)
                if kind == "rx":
                    held[tag].discard(id(burst))
                st = me.station
                if st is None:
                    continue
                if me.state == S.CONNECTED and (was != S.CONNECTED or st.stats["rx_ok"] > before):
                    last_ok[tag] = when
                now = progress_of([x.station for x in who.values()])
                if now != prog:
                    prog, count = now, {"a": 0, "b": 0}
                elif st.stats["rx_ok"] > before and st.tx.pending():
                    count[tag] += 1
                    stats["max_count"] = max(stats["max_count"], count[tag])
                    assert count[tag] <= WATCHDOG, ("no progress past the watchdog", tag, when)
    a, b = who["a"], who["b"]
    outcome = (bytes(got["a"]), bytes(got["b"]), round(t, 9), a.state, b.state, a.close_reason, b.close_reason,
               stats["mismatch"], stats["collisions"])
    return trace, outcome, dict(stats, done=bytes(got["a"]) == data_b and bytes(got["b"]) == data_a, t=t,
                                horizon=t >= horizon, data=(data_a, data_b))


# --- running the four pairings --------------------------------------------------------------

def run_pairs(fn, sc, native):
    """fn in all four pairings: every one sends Py-Py's bursts and ends the
    same way. A raise (corrupted control only) must happen in every
    pairing after the same bursts; the Python one is re-raised."""
    out = {}
    for pair in PAIRS:
        classes = [impl(native, w)[fn is sessions] for w in pair]
        try:
            out[pair] = fn(sc, *classes)
        except AssertionError:
            raise
        except Exception as e:  # noqa: BLE001
            if not sc["p_corrupt"]:
                raise
            out[pair] = e
    ref = out[("py", "py")]
    raised = {pair for pair, got in out.items() if isinstance(got, Exception)}
    assert not raised or len(raised) == 4, ("raised in some pairings only", raised)
    if raised:
        raise ref
    for pair, got in out.items():
        assert got[1] == ref[1], (pair, got[1], ref[1])
        assert got[0] == ref[0], (pair, "bursts differ")
    return ref


LATE = ("corrupted after a late burst",)


def check_link(sc, native):
    trace, outcome, stats = run_pairs(lockstep, sc, native)
    assert outcome[0] in ("done", "failed") or outcome == LATE, (outcome, stats)
    if not (sc["p_late"] or sc["p_corrupt"]):
        # only the clock or the watchdog ends a link, and nothing is ever mapped wrong
        assert outcome[0] == "done" or outcome[1] in ("link lost", "no progress"), outcome
        assert stats["mismatch"] == 0, stats
    return outcome, stats


def check_session(sc, native):
    trace, outcome, stats = run_pairs(sessions, sc, native)
    if outcome == LATE:
        return outcome, stats
    assert not stats["horizon"], ("still running at the horizon", outcome[3:7])
    if not sc["p_late"]:  # a late copy is answered like any burst, whoever is on air
        assert outcome[8] == 0, "collisions"
    if not (sc["p_late"] or sc["p_corrupt"]):
        # only clocks and the watchdog close a session; no protocol check ever trips
        assert "protocol" not in outcome[5] + outcome[6], outcome[5:7]
    if not stats["done"]:
        # a disconnect (or a restarted station that never heard of the session)
        assert outcome[3] in (S.CLOSED, S.IDLE) and outcome[4] in (S.CLOSED, S.LISTEN, S.IDLE), outcome[3:7]
    return outcome, stats


DEFAULT_LINK, SLOW_LINK = range(0, 200), range(200, 2200)
DEFAULT_SESSION, SLOW_SESSION = range(0, 100), range(100, 1100)


@pytest.mark.parametrize("seed", DEFAULT_LINK)
def test_link_fuzz(native, pure, seed):
    check_link(link_scenario(seed), native)


@pytest.mark.slow
@pytest.mark.parametrize("seed", SLOW_LINK)
def test_link_fuzz_sweep(native, pure, seed):
    check_link(link_scenario(seed), native)


@pytest.mark.parametrize("seed", DEFAULT_SESSION)
def test_session_fuzz(native, pure, seed):
    check_session(session_scenario(seed), native)


@pytest.mark.slow
@pytest.mark.parametrize("seed", SLOW_SESSION)
def test_session_fuzz_sweep(native, pure, seed):
    check_session(session_scenario(seed), native)


# --- CRC-valid corrupted control codewords --------------------------------------------------
# A false CRC accept (2^-16 per control codeword on noise) hands the
# station a control word that is wrong but well framed. Most are discarded
# (unparseable) or end the link on a protocol check. Two reference
# behaviours, the same in every pairing, are listed by seed so they stay
# visible: a raise (KNOWN_RAISE), and a flipped T_COMP bit, which no data
# CRC covers, delivering a deflated codeword raw (COMP_REASON). Regenerate
# the lists with `python tests/test_native_fuzz.py` after changing the
# scenarios or the channel.

COMP_REASON = ("reference behaviour: T_COMP flags ride in the control word only, so a CRC-valid corrupted "
               "control can deliver a deflated codeword as raw bytes (or the reverse): stream corrupted")
CORRUPT_LINK, SLOW_CORRUPT_LINK = range(2000, 2060), range(2060, 2560)
CORRUPT_SESSION, SLOW_CORRUPT_SESSION = range(3000, 3040), range(3040, 3340)
# 560 runs: 91 raise, 23 comp
KNOWN_LINK = {**dict.fromkeys([2001, 2003, 2005, 2008, 2010, 2017, 2043, 2044, 2049, 2051, 2052, 2056, 2058, 2067,
    2072, 2078, 2079, 2082, 2090, 2094, 2098, 2101, 2102, 2116, 2117, 2122, 2125, 2135, 2139, 2148, 2149, 2153,
    2154, 2156, 2181, 2189, 2196, 2200, 2208, 2209, 2212, 2218, 2227, 2233, 2238, 2243, 2245, 2248, 2258, 2259,
    2262, 2271, 2278, 2279, 2280, 2283, 2293, 2300, 2305, 2312, 2317, 2332, 2358, 2360, 2361, 2367, 2369, 2370,
    2378, 2385, 2404, 2408, 2418, 2423, 2426, 2433, 2438, 2447, 2448, 2466, 2482, 2484, 2487, 2498, 2502, 2504,
    2508, 2511, 2517, 2520, 2552], 'raise'), **dict.fromkeys([2047, 2048, 2075, 2121, 2129, 2174, 2178, 2179, 2183,
    2219, 2236, 2241, 2260, 2286, 2355, 2356, 2421, 2431, 2434, 2452, 2480, 2483, 2486], 'comp')}
# 340 runs: 32 raise, 0 comp
KNOWN_SESSION = {**dict.fromkeys([3003, 3015, 3040, 3057, 3059, 3065, 3066, 3071, 3085, 3103, 3104, 3113, 3116,
    3125, 3131, 3183, 3185, 3199, 3211, 3215, 3221, 3240, 3260, 3269, 3273, 3279, 3286, 3292, 3305, 3311, 3330,
    3336], 'raise'), **dict.fromkeys([], 'comp')}


RAISES = (ValueError, IndexError, KeyError)


def _marked(seeds, known):
    mark = {"raise": pytest.mark.xfail(strict=True, raises=RAISES, reason=KNOWN_RAISE),
            "comp": pytest.mark.xfail(strict=True, raises=AssertionError, reason=COMP_REASON)}
    return [pytest.param(s, marks=mark[known[s]]) if s in known else s for s in seeds]


@pytest.mark.parametrize("seed", _marked(CORRUPT_LINK, KNOWN_LINK))
def test_link_corrupt_control(native, pure, seed):
    check_link(link_scenario(seed, corrupt=True), native)


@pytest.mark.slow
@pytest.mark.parametrize("seed", _marked(SLOW_CORRUPT_LINK, KNOWN_LINK))
def test_link_corrupt_control_sweep(native, pure, seed):
    check_link(link_scenario(seed, corrupt=True), native)


@pytest.mark.parametrize("seed", _marked(CORRUPT_SESSION, KNOWN_SESSION))
def test_session_corrupt_control(native, pure, seed):
    check_session(session_scenario(seed, corrupt=True), native)


@pytest.mark.slow
@pytest.mark.parametrize("seed", _marked(SLOW_CORRUPT_SESSION, KNOWN_SESSION))
def test_session_corrupt_control_sweep(native, pure, seed):
    check_session(session_scenario(seed, corrupt=True), native)


@pytest.mark.xfail(strict=True, raises=RAISES, reason=KNOWN_RAISE)
@pytest.mark.parametrize("which", ["py", "cpp"])
def test_malformed_control_raises(native, pure, which):
    """The smallest known raise: a CRC-valid control word whose T_RV is
    shorter than its K resends."""
    cls = impl(native, which)[0]
    b = cls(1, Script([("m22", 1)]))
    ctl = F.Control(F.Core(k=5, acted_on=7), {F.T_RV: b"\x00"}).pack(22)
    slots = [L.Slot(L.ctl_mask(0, 0), 0, ctl[0])] + [L.Slot(L.data_mask(0, i), 0, bytes(22)) for i in range(5)]
    b.handle(clean(L.TxBurst("m22", slots, 0)))
    assert b.state in (L.ACTIVE, L.FAILED)


# --- late bursts: minimized reproducers -------------------------------------------------------
# A burst heard after a later one from the same sender (true reordering;
# the half-duplex engine decodes in order, so not expected on air) can
# corrupt the delivered stream, in both implementations: neither the
# abandon epoch nor the slicing is in the CRC mask, and a fresh build that
# answers a stale burst takes its stale ACK as current. docs/arq.md §10
# says reordered bursts are tested; nothing in the suite reordered before.

LATE_REASON = ("reference behaviour: a reordered burst can corrupt the delivered stream, though docs/arq.md §10 "
               "says reordering is covered; found by this fuzz, not fixed here")


class Script:
    """A policy that plays a list of (submode, max codewords), then repeats the last."""

    def __init__(self, plan):
        self.plan = list(plan)

    def choose(self, station, escalation):
        return self.plan.pop(0) if len(self.plan) > 1 else self.plan[0]

    def payload_bytes(self, m):
        return MODES[m][0]

    def rv_cycle(self, m):
        return MODES[m][1]


def clean(burst, lost=()):
    rx = FakeRx(burst, random.Random(0), 0.0, {}, {"mismatch": 0})
    for i in lost:
        rx.good[i] = False
    return rx


def finish(a, b, burst, data, turns=20):
    """Lossless lockstep from a's `burst` on. -> what b delivered."""
    got = bytearray()
    for _ in range(turns):
        b.handle(clean(burst))
        got += b.read()
        if L.FAILED in (a.state, b.state):
            break
        r = b.build()
        b.answered()
        a.handle(clean(r))
        if a.state == L.FAILED or not a.tx.pending():
            break
        burst = a.build()
        a.answered()
    return bytes(got)


def _stations(cls, plan_a):
    rng = random.Random(1)
    data = bytes(rng.randrange(256) for _ in range(400))
    a = cls(0, Script(plan_a), master=True)
    b = cls(1, Script([("m22", 1)]))
    a.write(data)
    return a, b, data


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LATE_REASON)
@pytest.mark.parametrize("which", ["py", "cpp"])
def test_late_burst_from_before_an_abandon(native, pure, which):
    """a sends P (m22, seqs 0-3); b loses seq 1 (cum 1) and answers. a
    switches to m46: Q abandons at 1 and re-slices; Q is lost. A late copy
    of P arrives: b delivers P's old 1-3 (cum 4), and its answer acks Q's
    re-sliced 1-3 to a. a switches back to m22: a new abandon at 4, equal
    to b's cumulative, so b takes it and joins two slicings."""
    a, b, data = _stations(impl(native, which)[0], [("m22", 5), ("m46", 5), ("m22", 5)])
    p = a.build()
    a.answered()
    assert b.handle(clean(p, lost=(2,))) and b.rx.cum == 1
    assert a.handle(clean(b.build()))
    b.answered()
    a.build()  # Q, lost
    a.answered()
    assert b.handle(clean(p)) and b.rx.cum == 4  # P again, late
    assert a.handle(clean(b.build())) and a.tx.base == 4
    b.answered()
    n = a.build()
    a.answered()
    got = finish(a, b, n, data)
    assert got == data[:len(got)] or L.FAILED in (a.state, b.state)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LATE_REASON)
@pytest.mark.parametrize("which", ["py", "cpp"])
def test_late_burst_with_a_stale_ack(native, pure, which):
    """b's reply R0 (cum 2) is heard again late, after b moved on to cum 4
    (its reply R1 lost). a answers R0 with a fresh build that switches
    mode: abandon at 2 (data b already delivered), lost. R1 then arrives
    late too: a's base goes to 4 under the new slicing, and its next switch
    abandons at 4, which b takes: b holds old 2-3, then new 4 on."""
    a, b, data = _stations(impl(native, which)[0], [("m22", 3), ("m22", 3), ("m46", 5), ("m22", 5)])
    assert b.handle(clean(a.build())) and b.rx.cum == 2
    a.answered()
    r0 = b.build()
    b.answered()
    assert a.handle(clean(r0))
    p1 = a.build()
    a.answered()
    assert b.handle(clean(p1)) and b.rx.cum == 4
    r1 = b.build()  # lost
    b.answered()
    assert a.handle(clean(r0))  # R0 again, late
    a.build()  # abandon at 2, lost
    a.answered()
    assert a.handle(clean(r1)) and a.tx.base == 4  # R1, late
    n = a.build()
    a.answered()
    got = finish(a, b, n, data)
    assert got == data[:len(got)] or L.FAILED in (a.state, b.state)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=COMP_REASON)
@pytest.mark.parametrize("which", ["py", "cpp"])
def test_flipped_comp_bit(native, pure, which):
    """The smallest T_COMP case: a's first burst carries one deflated
    codeword; its control arrives CRC-valid with the T_COMP bit cleared."""
    cls = impl(native, which)[0]
    a = cls(0, Script([("m46", 2)]), master=True)
    b = cls(1, Script([("m46", 1)]))
    data = b"CQ CQ de W1AW QTH FN31 RST 599 " * 20 + bytes(range(256))
    a.write(data)
    burst = a.build()
    a.answered()
    ctl = F.Control.unpack([burst.slots[0].payload])
    assert ctl.ext[F.T_COMP] == b"\x80"  # its one data codeword is deflated
    ctl.ext[F.T_COMP] = b"\x00"
    burst.slots[0] = L.Slot(burst.slots[0].mask_id, 0, F.Control(ctl.core, ctl.ext).pack(46)[0])
    got = finish(a, b, burst, data)  # the rest of the stream, lossless
    assert got == data[:len(got)] or L.FAILED in (a.state, b.state)


# --- engines through the simulated audio channel ----------------------------------------------
# test_engine.link with an impairment per direction: SNR, fades (signal
# gone, noise stays), a delay in blocks, and one side aborting mid-session
# (Engine.abort: no DISC) then listening again. Python Engine against the
# pure C++ Engine (GearPolicy and draws of its own), so no burst equality.

def engine_scenario(seed):
    r = random.Random(seed * 31337 + 5)
    return dict(seed=seed, snr=(r.choice([9, 12, 20]), r.choice([9, 12, 20])), delay=(r.randrange(3), r.randrange(3)),
                fades=[(r.uniform(3, 30), r.choice([1.5, 4.0, 10.0, 100.0]), r.randrange(3))
                       for _ in range(r.choice([0, 1, 2]))],
                abort=(r.choice("ab"), r.uniform(5, 25)) if r.random() < 0.25 else None,
                n=(r.randrange(1000, 8000), r.randrange(0, 1500)))


def engines(sc, a, b, limit_s=400.0):
    """-> (outcome, stats). Steps both engines in 0.1 s blocks."""
    rng = np.random.default_rng(sc["seed"])
    data = random.Random(sc["seed"])
    up = bytes(data.randrange(256) for _ in range(sc["n"][0] // 2)) + (b"CQ de W1AW QTH FN31 RST 599 73 " * 99)[
        :sc["n"][0] - sc["n"][0] // 2]
    down = bytes(data.randrange(256) for _ in range(sc["n"][1]))
    sigma = [np.sqrt((E.FS / 2) / SNR_REF_BW_HZ / 10 ** (s / 10)) for s in sc["snr"]]
    lines = [[np.zeros(BLOCK)] * (d + 1) for d in sc["delay"]]  # a->b, b->a
    got = {"a": bytearray(), "b": bytearray()}
    st = {"max_count": 0}
    count, last_ok, prog = {"a": 0, "b": 0}, {"a": 0.0, "b": 0.0}, 0
    phase, written, aborted, t = "connect", False, False, 0.0
    b.listen()
    a.connect(b.call, 2)
    eng = {"a": a, "b": b}
    for k in range(int(limit_s * 10)):
        t = k / 10
        gain = [0.0 if any(f0 <= t < f0 + fl and w in (d, 2) for f0, fl, w in sc["fades"]) else 2.2 for d in (0, 1)]
        oa, _ = a.step(gain[1] * lines[1][0] + rng.normal(0, sigma[1], BLOCK))
        ob, _ = b.step(gain[0] * lines[0][0] + rng.normal(0, sigma[0], BLOCK))
        lines = [lines[0][1:] + [oa], lines[1][1:] + [ob]]
        sa, sb = a.session, b.session
        if phase == "connect" and sa.state == S.CONNECTED and sb.state == S.CONNECTED:
            sa.write(up)
            sb.write(down)
            phase, written = "data", True
        if written:
            got["b"] += sb.read() if sb.station is not None else b""
            got["a"] += sa.read() if sa.station is not None else b""
        assert bytes(got["b"]) == up[:len(got["b"])] and bytes(got["a"]) == down[:len(got["a"])], "stream corrupted"
        if sc["abort"] and not aborted and written and t >= sc["abort"][1]:
            eng[sc["abort"][0]].abort()
            if sc["abort"][0] == "b":
                b.listen()
            aborted, st["aborted"] = True, t
        if phase == "data" and bytes(got["b"]) == up and bytes(got["a"]) == down:
            sa.disconnect()
            phase, st["delivered"] = "disc", t
        for tag in "ab":
            s = eng[tag].session
            if s.state == S.CONNECTED:
                ok = s.station.stats["rx_ok"]
                if ok > last_ok.get(tag + "n", 0):
                    last_ok[tag], last_ok[tag + "n"] = t, ok
                    stations = [x.session.station for x in (a, b) if x.session.state == S.CONNECTED]
                    now = progress_of(stations)
                    if now != prog:
                        prog, count = now, {"a": 0, "b": 0}
                    elif s.station.tx.pending():
                        count[tag] += 1
                        st["max_count"] = max(st["max_count"], count[tag])
                        assert count[tag] <= WATCHDOG, ("no progress past the watchdog", tag, t)
                assert t - last_ok[tag] <= S.LINK_LOST_S + 15, ("connected past the dead-link bound", tag, t)
            else:
                last_ok[tag], last_ok[tag + "n"] = t, 0
        settled = all(eng[g].session.state in (S.CLOSED, S.LISTEN, S.IDLE) for g in "ab")
        if settled and (phase != "connect" or t > 60):
            break
    return (phase, a.session.state, b.session.state, a.session.close_reason, b.session.close_reason), dict(
        st, t=t, settled=settled, delivered=bytes(got["b"]) == up and bytes(got["a"]) == down)


def check_engines(sc, native, reference):
    py = lambda call, seed: reference(E, "Engine")(call, seed=seed)
    cc = lambda call, seed: native.engine.Engine(call, seed=seed)
    for wa, wb in PAIRS:
        a = (py if wa == "py" else cc)("W1AW", sc["seed"] * 2 + 1)
        b = (py if wb == "py" else cc)("K2XYZ", sc["seed"] * 2 + 2)
        outcome, st = engines(sc, a, b)
        assert st["settled"], ("still running at the limit", (wa, wb), outcome)
        if not (sc["abort"] or any(fl > 60 for _, fl, _ in sc["fades"])):
            assert st["delivered"], ((wa, wb), outcome)


DEFAULT_ENGINE = (2, 4, 9)  # delivered through fades; the link-lost ones (~25 s each) are in the sweep


@pytest.mark.parametrize("seed", DEFAULT_ENGINE)
def test_engine_fuzz(native, pure, reference, seed):
    check_engines(engine_scenario(seed), native, reference)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [s for s in range(30) if s not in DEFAULT_ENGINE])
def test_engine_fuzz_sweep(native, pure, reference, seed):
    check_engines(engine_scenario(seed), native, reference)


def _scan(fn, seeds):
    """{seed: "raise" | "comp"} for the Py-Py runs that end either way."""
    out = {}
    for seed in seeds:
        try:
            fn(seed)
        except AssertionError as e:
            if "stream corrupted" not in str(e):
                raise
            out[seed] = "comp"
        except Exception:  # noqa: BLE001
            out[seed] = "raise"
    return out


if __name__ == "__main__":
    import logging
    import textwrap

    logging.disable(logging.WARNING)
    for name, fn, seeds in (
            ("KNOWN_LINK", lambda s: lockstep(link_scenario(s, True), L.Station, L.Station),
             [*CORRUPT_LINK, *SLOW_CORRUPT_LINK]),
            ("KNOWN_SESSION", lambda s: sessions(session_scenario(s, True), S.Session, S.Session),
             [*CORRUPT_SESSION, *SLOW_CORRUPT_SESSION])):
        known = _scan(fn, seeds)
        by = {k: [s for s, v in known.items() if v == k] for k in ("raise", "comp")}
        print(f"# {len(seeds)} runs: {len(by['raise'])} raise, {len(by['comp'])} comp")
        print(textwrap.fill(f"{name} = {{**dict.fromkeys({by['raise']}, 'raise'), "
                            f"**dict.fromkeys({by['comp']}, 'comp')}}", 116, subsequent_indent="    "))
