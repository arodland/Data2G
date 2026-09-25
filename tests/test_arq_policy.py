"""The gear shifter's pieces: recommendation encoding, the bandwidth cap,
and the link predictor's basic sanity (the evidence it is good is
scripts/linksim.py sweep, not these)."""

from types import SimpleNamespace

import numpy as np

from data2g.arq import policy as G
from data2g.arq import predictor as P
from data2g.config import SUBMODES


def test_recommend_roundtrip_every_submode():
    for name in SUBMODES:
        assert G.decode(G.encode(name)) == name


def test_500hz_cap_only_narrow_modes():
    assert all(s.band in ("n10", "n4") for s in G.allowed(0))
    assert G.FALLBACK[0] in {s.name for s in G.allowed(0)} and G.CONNECT[0] in {s.name for s in G.allowed(0)}


def measured(snr_db, doppler, band="w"):
    """Receiver features as a clean AWGN-ish burst at this SNR would give."""
    nc = {"w": 24, "n10": 10, "n4": 4, "w48": 48}[band]
    c_snr = snr_db + 10 * np.log10(2500 / 50 / nc)
    m = dict(snr_est=snr_db, spread_est=doppler, delay_est_ms=0.0)
    for c in P.CONSTS:
        m[f"mi_{c}"] = float(P.capacity(c_snr, c))
    return m


def test_predictor_monotone_in_snr_and_rate():
    lo = P.predict(measured(-2, 0.05), "w", 2.5)
    hi = P.predict(measured(12, 0.05), "w", 2.5)
    for name in ("qpsk-r1/2", "16qam-r1/2", "w48-16qam-r2/3"):
        assert hi[name] > lo[name]
    assert hi["qpsk-r1/5"] >= hi["16qam-r1/2"] - 1e-9  # a faster mode never predicts likelier


def station(cap, chat=False):
    return SimpleNamespace(cap=cap, last_rx_data=True, peer_recommend=None, peer_reply_recommend=None,
                           peer_size_hint=1, tx=SimpleNamespace(pending=lambda: True, base=0), rx=SimpleNamespace(buf={}),
                           chat=chat, peer_chat=False)


def test_chat_trades_rate_for_latency():
    """CHAT ON: a short burst in a mode at least as likely to decode."""
    out = {}
    for chat in (False, True):
        g = G.GearShifter()
        g.observe(measured(6, 0.1, "w48"), "w48-qpsk-r1/2", 0.0)
        rec, hint, _ = g.recommend(station(2, chat))
        out[chat] = (G.decode(rec), hint)
    assert out[True][1] <= out[False][1]
    p = P.predict(measured(6, 0.1, "w48"), "w48", 2.5)
    assert p[out[True][0]] >= p[out[False][0]] - 0.02


def test_shifter_respects_cap_and_falls_back():
    g = G.GearShifter()
    st = station(0)
    assert g.choose(st, 0)[0] == G.FALLBACK[0]  # nothing heard yet
    g.observe(measured(15, 0.1, "n10"), "n10-qpsk-r1/2", 0.0)
    rec, hint, _ = g.recommend(st)
    name = G.decode(rec)
    assert SUBMODES[name].band in ("n10", "n4")
    st.peer_recommend, st.peer_size_hint = rec, hint
    mode, n = g.choose(st, 0)
    assert mode == name and n >= 1
    assert g.choose(st, 1)[0] == G.FALLBACK[0]  # escalation: robust


def test_numpy_runtime_has_no_torch():
    import sys
    before = "torch" in sys.modules
    P.model.cache_clear()
    P.model()
    assert before or "torch" not in sys.modules


def test_inputs_match_model_with_and_without_history():
    m = dict(measured(5, 0.1), frames=8)
    for prev in (None, (m, "w", 4.0)):
        x = P.inputs(m, "w", 2.5, 16, prev)
        assert x.shape == (len(P.INPUTS),) == P.model().mean.shape


def test_online_bias_follows_outcomes():
    g = G.GearShifter()
    g.predicted = {"qpsk-r1/2": (0.5, 0.5)}
    for _ in range(5):
        g.outcome("qpsk-r1/2", 10, 10)
    assert g.bias["qpsk-r1/2"] > 1
    g.outcome("16qam-r1/2", 0, 10)  # no prediction for it: no update
    assert list(g.bias) == ["qpsk-r1/2"]  # per mode: qpsk-r1/3 is not vouched for


def test_a_mode_that_never_answers_is_left():
    """Sender-side strikes: the peer keeps recommending a mode whose bursts
    go unanswered (escalation after each) though polls get through; the
    sender must stop using it for a while instead of looping forever."""
    g = G.GearShifter()
    st = station(2)
    st.peer_recommend, st.peer_reply_recommend = G.encode("w48-64l-r2/3"), G.encode("ack-4f")
    assert g.choose(st, 0)[0] == "w48-64l-r2/3"
    g.choose(st, 1)  # its repeat timed out: a poll (fallback)
    assert g.choose(st, 0)[0] == G.CONNECT[2]  # the robust data mode, not 64-QAM again
    for _ in range(G.STRIKE_HOLD + 1):
        mode = g.choose(st, 0)[0]
    assert mode == "w48-64l-r2/3"  # the hold ends; it may be tried again


def test_a_family_whose_controls_fail_is_not_recommended():
    """Receiver-side strikes: two headers in a family with no control decoded
    and the family sits out the next recommendations."""
    g = G.GearShifter()
    g.observe(measured(30, 0.05, "w48"), "w48-qpsk-r1/2", 0.0)
    st = station(2)
    first = G.decode(g.recommend(st)[0])
    fam = G._family(SUBMODES[first])
    g.outcome(first, 0, 1)
    g.outcome(first, 0, 1)
    again = G.decode(g.recommend(st)[0])
    assert G._family(SUBMODES[again]) != fam
