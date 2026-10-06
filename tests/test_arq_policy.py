"""The gear shifter's pieces: recommendation encoding, the bandwidth cap,
and the link predictor's basic sanity (the evidence it is good is
scripts/linksim.py sweep, not these)."""

from types import SimpleNamespace

import numpy as np

from data2g.arq import policy as G
from data2g.arq import predictor as P
from data2g.arq.modes import MODES
from data2g.config import SUBMODES


def test_recommend_roundtrip_every_submode():
    for name in MODES:
        assert G.decode(G.encode(name)) == name


def test_500hz_cap_only_narrow_modes():
    assert all(G.width_hz(s) <= 500 for s in G.allowed(0))
    assert {s.band for s in G.allowed(0)} == {"n10", "n4", "c16r25", "c8r50"}
    assert G.FALLBACK[0] in {s.name for s in G.allowed(0)} and G.CONNECT[0] in {s.name for s in G.allowed(0)}


def measured(snr_db, doppler, band="w"):
    """Receiver features as a clean AWGN-ish burst at this SNR would give."""
    nc = {"w": 24, "n10": 10, "n4": 4, "w48": 48}[band]
    c_snr = snr_db + 10 * np.log10(2500 / 50 / nc)
    m = dict(snr_est=snr_db, spread_est=doppler, delay_est_ms=0.0)
    for c in P.CONSTS:
        m[f"mi_{c}"] = float(P.capacity(c_snr, c))
    return m


def joint(measured_, band, seconds=6.0):
    """P(usable) x P(codeword) per submode from the outcome model."""
    return {k: pb * pc for k, (pb, pc) in P.predict_outcome(measured_, band, 2.5, seconds).items()}


def test_predictor_monotone_in_snr_and_rate():
    lo, hi = joint(measured(-2, 0.05), "w"), joint(measured(12, 0.05), "w")
    for name in ("qpsk-r1/2", "16qam-r1/2", "w48-16qam-r2/3"):
        assert hi[name] > lo[name]
    assert lo["qpsk-r1/5"] >= lo["16qam-r1/2"]  # near the bottom, the robust mode is the likelier


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
    p = joint(measured(6, 0.1, "w48"), "w48")
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
    assert g.choose(st, 1)[0] == G.FALLBACK[0]  # escalation, no reply mode heard: the fallback
    st.peer_reply_recommend = G.encode("n10-qpsk-r1/3")
    assert [g.choose(st, e)[0] for e in (1, 2, 3)] == ["n10-qpsk-r1/3", G.ALT_POLL, "n10-qpsk-r1/3"]
    assert g.choose(st, 4) == (G.ROBUST_CONNECT, 1)  # control only in the robust mode
    g.observe(measured(-5, 0.1, "n10"), G.ROBUST_CONNECT, 1.0)
    assert g.choose(st, 1) == (G.ROBUST_CONNECT, 1)  # a robust poll is answered in its mode
    assert g.choose(st, 0)[0] == name  # not escalated: the recommendation


def test_numpy_runtime_has_no_torch():
    import sys
    before = "torch" in sys.modules
    P.outcome_model.cache_clear()
    P.predict_outcome(measured(5, 0.1), "w", 2.5, 6.0)
    assert before or "torch" not in sys.modules


def test_inputs_match_model_with_and_without_history():
    m = dict(measured(5, 0.1), frames=8)
    model = P.outcome_model()
    mean = (model.members[0] if isinstance(model, P.OutcomeEnsemble) else model).mean
    for prev in (None, (m, "w", 4.0)):
        x = P.outcome_inputs(m, "w", 2.5, 6.0, prev, model.bands)
        assert x.shape == mean.shape


def test_online_bias_follows_outcomes():
    g = G.GearShifter()
    g.predicted = {"qpsk-r1/2": (0.5, 0.5)}
    for _ in range(5):
        g.outcome("qpsk-r1/2", 10, 10)
    assert g.bias["qpsk-r1/2"] > 1
    g.outcome("16qam-r1/2", 0, 10)  # no prediction for it: no update
    assert list(g.bias) == ["qpsk-r1/2"]  # per mode: qpsk-r1/3 is not vouched for


def test_outcome_ensemble_averages_member_probabilities(tmp_path):
    """scripts/train_outcome.py --ensemble: members' files in one, loaded
    as an ensemble whose output is the logit of the mean probability."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import train_outcome as T

    n_in, n_out = 3, 2
    paths = []
    for i, b in enumerate((2.0, -2.0)):
        p = tmp_path / f"m{i}.npz"
        np.savez(p, mean=np.zeros(n_in), std=np.ones(n_in), W0=np.zeros((n_in, n_out)), b0=np.full(n_out, b),
                 modes=np.array(["a"]), bands=np.array(["w"]))
        paths.append(str(p))
    out = str(tmp_path / "ens.npz")
    T.combine(paths, out)
    m = P.outcome_model(out)
    assert isinstance(m, P.OutcomeEnsemble) and m.modes == ("a",)
    z = m(np.zeros(n_in))
    assert np.allclose(z, 0.0, atol=1e-9)  # mean of sigmoid(2), sigmoid(-2) is 0.5


def test_cpm_airtime_counts_a_duplicated_control_as_control():
    """ARQ_DUP's second control copy is a short control codeword: an
    fsk32r62-r1/2 x10 burst is 30.4 s on air, not the 32.6 s of 9 data."""
    from data2g.arq.link import Slot, TxBurst, ctl_mask, data_mask, dup_ctl
    g = G.GearShifter()
    assert abs(g.airtime("fsk32r62-r1/2", 10, True) - 30.4) < 0.1
    assert abs(g.airtime("fsk32r62-r1/2", 10) - 32.6) < 0.1
    ctl, data = [Slot(ctl_mask(0, 0), 0, b"")], [Slot(data_mask(0, i), 0, b"") for i in range(8)]
    assert dup_ctl(TxBurst("fsk32r62-r1/2", ctl + [Slot(ctl_mask(0, 0), 1, b"")] + data, 0))
    assert not dup_ctl(TxBurst("fsk32r62-r1/2", ctl + data, 0))


def test_control_is_not_a_decoded_data_codeword():
    """A usable burst whose 7 data codewords all failed: burst bias up,
    codeword bias down (counting the control, it scored 1/8)."""
    g = G.GearShifter()
    g.predicted = {"qpsk-r1/3": (0.5, 0.9)}
    g.outcome("qpsk-r1/3", 0, 7, usable=True)
    assert g.bias_burst["qpsk-r1/3"] > 0 and g.bias["qpsk-r1/3"] < -0.8
    g.outcome("qpsk-r1/3", 0, 0, usable=False)  # control lost: the burst bias only
    assert g.bias_burst["qpsk-r1/3"] < 0.5 and g.bias["qpsk-r1/3"] < -0.8


def test_cpm_cap_reply_hold_and_link_lost_price():
    from data2g.arq.modes import burst_seconds

    for m in ("fsk32r62-r1/2", "fsk8r50-r1/2"):
        s = G.MODES[m]
        assert burst_seconds(s, G.slots_for(s, G.SIZE_S[-1])) <= G.CPM_MAX_S
    g = G.GearShifter()
    g.log = [("w48-qpsk-r1/2", 2, "ack-1f")]
    quiet = g.reply_hold(SimpleNamespace(cap=2, misses=0, esc_floor=0), SimpleNamespace(submode="w48-qpsk-r1/2"))
    assert quiet < 2.0  # a short reply asked for: about the old 1.5 s
    robust = g.reply_hold(SimpleNamespace(cap=2, misses=1, esc_floor=4), SimpleNamespace(submode=G.ROBUST_CONNECT))
    assert robust >= burst_seconds(G.MODES[G.ROBUST_CONNECT], 1)  # its 5 s answer is not polled over


def test_robust_floor_sends_control_only_bursts_robust():
    g = G.GearShifter()
    st = station(2)
    st.peer_reply_recommend = G.encode("n4-ack-8f")
    st.tx = SimpleNamespace(pending=lambda: False, base=0)
    assert g.choose(st, 0)[0] == "n4-ack-8f"
    st.esc_floor = G.ROBUST_ESCALATION
    assert g.choose(st, 0) == (G.ROBUST_CONNECT, 1)
    st.tx = SimpleNamespace(pending=lambda: True, base=0)
    st.peer_recommend = G.encode("fsk32r62-r1/2")
    assert g.choose(st, 0)[0] == "fsk32r62-r1/2"  # data still follows the recommendation


def test_lost_data_steps_down_the_ladder():
    """LADDER_AFTER data bursts lost in a row in the mode I recommended: data
    goes only in modes LADDER_STEP_DB more robust on every channel; a further
    loss steps down from the mode that failed, a usable data burst climbs."""
    T, step = G.MODE_THRESHOLDS, G.LADDER_STEP_DB

    def below(a, b, by):  # a at least `by` dB more robust than b on every channel
        return all(x <= y - by for x, y in zip(T[a], T[b]))

    g = G.GearShifter()
    g.observe(measured(10, 0.1), "qpsk-r1/5", 0.0)
    st = station(2)
    for _ in range(G.LADDER_AFTER):
        data = G.decode(g.recommend(st)[0])
        g.outcome(data, 0, 0, usable=False)  # the peer's data burst in it: lost
    down = G.decode(g.recommend(st)[0])
    assert below(down, data, step)
    g.outcome(down, 0, 0, usable=False)
    lower = G.decode(g.recommend(st)[0])
    assert lower == down == min(T, key=lambda m: max(T[m])) or below(lower, down, step)
    for _ in range(20):  # data gets through: it climbs off the ladder
        g.outcome(G.decode(g.recommend(st)[0]), 3, 3, usable=True)
    assert g.ceiling is None


def test_link_features_count_lost_and_missed_peer_bursts():
    """The link history (predictor.N_LINK): heard-but-lost bursts from
    outcome(), missed ones from my timeouts, over the last LINK_HIST."""
    sh = G.GearShifter()
    st = SimpleNamespace(stats={"timeouts": 0})
    assert sh.link_features(st) is None
    sh.outcome("qpsk-r1/5", 3, 4, usable=True)
    sh.outcome("qpsk-r1/5", 0, 0, usable=False)
    st.stats["timeouts"] = 2
    assert sh.link_features(st) == [0.25, 0.5, 4 / G.LINK_HIST, 1.0]
    assert sh.link_features(st) == [0.25, 0.5, 4 / G.LINK_HIST, 1.0]  # timeouts counted once
    for _ in range(G.LINK_HIST):
        sh.outcome("qpsk-r1/5", 4, 4, usable=True)
    assert sh.link_features(st) == [0.0, 0.0, 1.0, 1.0]
