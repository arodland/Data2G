"""The gear shifter's pieces: recommendation encoding, the bandwidth cap,
and the link predictor's basic sanity (the evidence it is good is
scripts/linksim.py sweep, not these)."""

from types import SimpleNamespace

import numpy as np
import pytest

from data2g.arq import policy as G
from data2g.arq import predictor as P
from data2g.arq.modes import MODES
from data2g.config import SUBMODES


def test_recommend_roundtrip_every_submode():
    from data2g.arq.modes import SELECTABLE

    for name in SELECTABLE:  # what the ARQ may name on air (every mode on HF)
        assert G.decode(G.encode(name)) == name


def test_500hz_cap_only_narrow_modes():
    assert all(G.width_hz(s) <= 500 for s in G.allowed(0))
    from data2g import config

    # FM: n10 alone (no n4, no CPM: plan Q2)
    assert {s.band for s in G.allowed(0)} == ({"n10"} if config.PROFILE == "fm" else {"n10", "n4", "c16r25", "c8r50", "c8w50"})
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


@pytest.mark.hf_only("HF's outcome model (FM predicts from thresholds, M6)")
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


@pytest.mark.hf_only("the HF outcome model and gear shifter on an HF recording's measurements; FM's are M6's")
def test_chat_bulk_queue_goes_for_goodput():
    """CHAT ON with more queued than one burst carries (a file): codewords a
    burst loses are resent, so not every one need decode. At the 20 dB a
    client sending a file with CHAT ON measured (recordings/20261021-165549)
    a chat line goes robust; the file goes faster than 16-QAM."""
    m = dict(snr_est=20.3, spread_est=0.2, delay_est_ms=0.0, headroom=1.0, frames=76.0,
             **{f"mi_{c}": v for c, v in zip(P.CONSTS, (1.0, 1.0, 0.91, 0.74))})
    picks = {}
    for queued in (200, 60000):
        g = G.GearShifter()
        g.observe(m, "w48-16qam-r3/4", 0.0)
        g.observe_energy(m["snr_est"], 0.0)
        rec, _, _ = g.recommend(G.StationView(cap=2, chat=True, peer_queued=queued))
        picks[queued] = MODES[G.decode(rec)]
    rate = lambda s: s.k / s.frames_per_cw  # noqa: E731  info bits per frame, w48 alike
    assert rate(picks[60000]) > rate(picks[200])
    assert rate(picks[60000]) > rate(MODES["w48-16qam-r5/6"])


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
    for prev in (None, (m, "w", 4.0)):
        x = P.outcome_inputs(m, "w", 2.5, 6.0, prev, model.bands, model.noise, model.link, model.energy)
        assert model(x).shape == (2 * len(model.modes),)  # the model takes these inputs as built


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
    fsk32r62-r1/2 x10 burst is 28.1 s on air, not the 30.1 s of 9 data."""
    from data2g.arq.link import Slot, TxBurst, ctl_mask, data_mask, dup_ctl
    g = G.GearShifter()
    assert abs(g.airtime("fsk32r62-r1/2", 10, True) - 28.1) < 0.1
    assert abs(g.airtime("fsk32r62-r1/2", 10) - 30.1) < 0.1
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


def test_weak_path_reads_the_burst_s_own_spread():
    """weak_path() takes the last burst's own spread reading, not the one
    observe() carries in from an earlier burst when it reports unmeasured:
    while dialing the earlier one can be QRM, and its 9 Hz carried into a
    good CONNECT_ACK started a good path at the robust floor. Unmeasured
    is not fast (the SNR decides); a fast reading of its own is."""
    g = G.GearShifter()
    assert g.weak_path()  # nothing measured
    g.observe(measured(-2.0, 9.0), G.ROBUST_CONNECT, 1.0)  # QRM heard while dialing
    assert g.weak_path()
    g.observe(measured(5.0, G.UNMEASURED_SPREAD_HZ), G.ROBUST_CONNECT, 6.0)  # the ACK: good, spread unmeasured
    assert g.measured["spread_est"] == 9.0 and g.heard_spread is None  # carried for the predictor, not for this
    assert not g.weak_path()
    g.observe(measured(-9.0, G.UNMEASURED_SPREAD_HZ), G.ROBUST_CONNECT, 11.0)
    assert g.weak_path()  # weak by its SNR
    g.observe(measured(5.0, 12.0), G.ROBUST_CONNECT, 16.0)
    assert g.heard_spread == 12.0 and g.weak_path()  # fast, by its own reading
    g.observe(measured(5.0, 1.0), G.ROBUST_CONNECT, 21.0)
    assert not g.weak_path()


@pytest.mark.hf_only("names HF's CPM modes and their thresholds")
def test_a_robust_start_bounds_the_first_data_request():
    """A session started at the robust floor has one reading, the
    CONNECT_ACK's, and from it at -13.8 dB the model asks for fsk32r62-r1/2
    (steady-channel threshold -8 dB): the burst was lost and with it 9 of 16
    sessions at AWGN -15 dB. Until a data burst of the peer's is usable the
    request stays within ROBUST_START_DATA_MARGIN_DB of the reading and asks
    for the control twice; a reading the model's pick already fits is left
    alone. A session not started there asks as before."""
    def first(snr, started):
        g = G.GearShifter()
        g.observe(measured(snr, 0.0), G.ROBUST_CONNECT, 1.0)
        if started:
            g.robust_started()
        return g, G.decode(g.recommend(station(2))[0])

    assert first(-13.8, False)[1].startswith("fsk32r62")
    g, mode = first(-13.8, True)
    assert mode.startswith("fsk16r25") and g.want_dup
    assert G.MODE_THRESHOLDS[mode][0] <= -13.8 + G.ROBUST_START_DATA_MARGIN_DB
    assert first(-12.0, True)[1].startswith("fsk8r50")  # the margin moves with the reading
    assert first(-9.0, True)[1] == first(-9.0, False)[1] and first(-9.0, True)[1].startswith("fsk32r62")
    g.outcome(mode, 0, 1, usable=False)  # a data burst lost: still bounded
    m2 = G.decode(g.recommend(station(2))[0])
    assert g.start_bound and G.MODE_THRESHOLDS[m2][0] <= -13.8 + G.ROBUST_START_DATA_MARGIN_DB
    g.outcome(mode, 1, 1, usable=True)  # one usable: the model's own from here
    assert not g.start_bound
    g.observe(measured(-22.0, 0.0), G.ROBUST_CONNECT, 9.0)  # under every threshold: the most robust there is
    g.robust_started()
    mode = G.decode(g.recommend(station(2))[0])
    assert G.MODE_THRESHOLDS[mode][0] == min(G.MODE_THRESHOLDS[s.name][0] for s in G.allowed(2) if P.outcome_knows(s.name))


@pytest.mark.hf_only("names HF's n4 and CPM modes (FM has neither, plan Q2)")
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
    st.peer_recommend = None  # ... once there is one: until then control only, in the robust mode
    assert g.choose(st, 0) == (G.ROBUST_CONNECT, 1)
    st.esc_floor = 0
    assert g.choose(st, 0) == (G.FALLBACK[2], 2)


@pytest.mark.hf_only("HF's ladder thresholds (FM's are provisional, M6)")
def test_lost_data_steps_down_the_ladder():
    """LADDER_AFTER data bursts lost in a row in the mode I recommended: data
    goes only in modes LADDER_STEP_DB more robust on every channel; a further
    loss steps down from the mode that failed, a usable data burst climbs;
    climbed off it, one loss puts it back."""
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
    def climb_off():  # data gets through: it climbs off the ladder
        for _ in range(20):
            g.outcome(G.decode(g.recommend(st)[0]), 3, 3, usable=True)
            if g.ceiling is None:
                return
        raise AssertionError("still on the ladder")

    climb_off()
    top = G.decode(g.recommend(st)[0])
    g.outcome(top, 0, 0, usable=False)  # one loss right after the climb out
    assert g.ceiling is not None and below(G.decode(g.recommend(st)[0]), top, step)
    climb_off()
    g.outcome(G.decode(g.recommend(st)[0]), 3, 3, usable=True)  # a usable burst after the climb out
    g.outcome(G.decode(g.recommend(st)[0]), 0, 0, usable=False)
    assert g.ceiling is None  # one loss later: LADDER_AFTER again


def test_link_features_count_lost_and_missed_peer_bursts():
    """The link history (predictor.N_LINK): heard-but-lost bursts from
    outcome(), missed ones from my timeouts, over the last LINK_HIST."""
    sh = G.GearShifter()
    if hasattr(sh, "_n"):
        import pytest

        pytest.skip("the link history is Python-only (study inputs; the C++ shifter keeps none)")
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


def test_gate_takes_the_second_model_on_flat_low_snr_channels():
    """predictor.GATE: the gate's model when the median spread_est of the last
    GATE_HIST peer bursts is under GATE_SPREAD_HZ and snr_est under GATE_SNR_DB."""
    if P.GATE is None:
        import pytest

        pytest.skip("gate off (DATA2G_OUTCOME_GATE / DATA2G_OUTCOME_MODEL)")
    sh = G.GearShifter()
    for _ in range(P.GATE_HIST):
        sh.observe(measured(5.0, 0.05), "qpsk-r1/2", 1.0)
    assert sh.gate() is not None
    sh.observe(measured(P.GATE_SNR_DB + 1, 0.05), "qpsk-r1/2", 2.0)
    assert sh.gate() is None  # SNR over the gate
    for _ in range(3):
        sh.observe(measured(5.0, 2.0), "qpsk-r1/2", 3.0)
    assert sh.gate() is not P.outcome_model(P.GATE[0])  # fading: the median spread over the gate


def test_fast_region_takes_the_third_model_on_fast_fading_near_0_db():
    """predictor.FAST: its model when the median spread_est is at least
    FAST_SPREAD_HZ and snr_est in [FAST_SNR_LO_DB, FAST_SNR_HI_DB); the
    installed one outside."""
    if P.FAST is None:
        import pytest

        pytest.skip("fast region off (DATA2G_OUTCOME_FAST / DATA2G_OUTCOME_MODEL)")
    sh = G.GearShifter()
    for _ in range(P.GATE_HIST):
        sh.observe(measured(P.FAST_SNR_LO_DB + 1, 2.0), "qpsk-r1/2", 1.0)
    assert sh.gate() is P.outcome_model(P.FAST[0])
    sh.observe(measured(P.FAST_SNR_HI_DB, 2.0), "qpsk-r1/2", 2.0)
    assert sh.gate() is None  # SNR at the top edge
    sh.observe(measured(P.FAST_SNR_LO_DB - 0.5, 2.0), "qpsk-r1/2", 3.0)
    assert sh.gate() is None  # under the bottom edge


def test_cnp_outcome_model_takes_one_input_or_many(tmp_path):
    """predictor.OutcomeCNP (scripts/cnp_outcome.py): one input row gives one
    row of logits, as OutcomeMlp does, and agrees with the batched call."""
    import numpy as np

    rng = np.random.default_rng(0)
    n_in, n_desc, n_modes, z, h = 31, 5, 3, 4, 6
    p = dict(mean=np.zeros(n_in - 1, np.float32), std=np.ones(n_in - 1, np.float32))
    for name, shapes in (("enc", [(8, n_in - 1), (8, 8), (z, 8)]), ("head", [(h, z + n_desc + 1), (h, h), (2, h)])):
        for i, s in zip((0, 2, 4), shapes):
            p[f"{name}.{i}.weight"] = rng.normal(size=s).astype(np.float32)
            p[f"{name}.{i}.bias"] = rng.normal(size=s[0]).astype(np.float32)
    m = P.OutcomeCNP(p, rng.normal(size=(n_modes, n_desc)), 27, ("a", "b", "c"))
    x = rng.normal(size=(4, n_in))
    batch = m(x)
    assert batch.shape == (4, 2 * n_modes)
    np.testing.assert_allclose(m(x[1]), batch[1], rtol=1e-6)


def test_expected_reply_follows_what_the_peer_sent_last():
    """The reply I expect (the engine's carrier hold and missed-reply energy)
    is a data burst in the data mode I recommended only if the peer's last
    burst carried data; an idle peer (the IRS of a one-way transfer) answers
    control-only in the reply mode. peer_had_data was never set: always data."""
    g = G.GearShifter()
    g.observe(measured(10.0, 0.5), "qpsk-r1/2", 0.0)
    data, hint, reply = g.recommend(G.StationView(cap=2, last_rx_data=True))
    mode, secs = g.expected_reply()
    assert mode == G.decode(data) and secs == G.burst_seconds(G.MODES[mode], G.slots_for(G.MODES[mode], G.SIZE_S[hint]))
    data, hint, reply = g.recommend(G.StationView(cap=2, last_rx_data=False))
    mode, secs = g.expected_reply()
    assert mode == G.decode(reply) and secs == G.burst_seconds(G.MODES[mode], G.ctl_slots(G.MODES[mode]))
    assert secs < G.burst_seconds(G.MODES[G.decode(data)], G.slots_for(G.MODES[G.decode(data)], G.SIZE_S[hint]))


def test_robust_mode_follows_measured_spread():
    """In a session, robust polls and the robust floor go in ROBUST_FAST once
    the peer measured fast fading (fsk16r25 fails on 30 Hz Doppler, #244); a
    robust poll is answered in whichever robust mode it came in."""
    g = G.GearShifter()
    st = station(0)
    assert g.choose(st, 4) == (G.ROBUST_CONNECT, 1)  # nothing measured
    g.observe(measured(10, 12.0), "fsk32r62-r1/2", 0.0)
    assert g.choose(st, 4) == (G.ROBUST_FAST, 1)
    st.tx, st.esc_floor = SimpleNamespace(pending=lambda: False, base=0), G.ROBUST_ESCALATION
    assert g.choose(st, 0) == (G.ROBUST_FAST, 1)  # the robust floor too
    g.observe(measured(10, 1.0), G.ROBUST_CONNECT, 1.0)
    assert g.choose(st, 1) == (G.ROBUST_CONNECT, 1)  # answered in the peer's robust mode
    assert g.choose(st, 4) == (G.ROBUST_CONNECT, 1)
    g.observe(measured(10, 12.0), G.ROBUST_FAST, 2.0)
    assert g.choose(st, 1) == (G.ROBUST_FAST, 1)
    from data2g.arq.modes import burst_seconds

    hold = g.reply_hold(SimpleNamespace(cap=0, misses=0, esc_floor=0), SimpleNamespace(submode=G.ROBUST_FAST))
    assert hold >= max(burst_seconds(G.MODES[m], 1) for m in G.ROBUST_MODES)


@pytest.mark.hf_only("an HF channel (mpp) against the HF predictor's training rows")
def test_live_gap_is_the_training_rows_gap():
    """The shifter's gap input live (GearShifter.gap) against the one its
    training rows get (scripts/session_data.py: t0 - snap["t"], from the
    measured peer burst to the start of the burst predicted), in a 120 s
    session on the real modem, MPP 0 dB. Live it is an estimate: my burst's
    end plus a reply's start. It is never late; a reply is within ~0.1 s
    (a control spilling into a second codeword: ~0.5 s); a poll after a
    timeout or a held reply starts later than estimated."""
    import random
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import linksim as L
    import outcome_data as O
    import phy_session as PS
    import session_data as SD

    pairs = []

    class Shifter(SD.Explorer):
        def recommend(self, station):
            out = super().recommend(station)
            if self.measured is not None:
                self.snap["live"] = self.gap(G.view(station))
            return out

    class Phy(SD.DataPhy):
        def send(self, burst, t0):
            snap, n = self.pols[1 - burst.slots[0].mask_id[1]].snap, len(self.out)
            res = super().send(burst, t0)
            if len(self.out) > n:  # a training row
                pairs.append(self.out[-1]["gap"] - snap["live"])
            return res

    seed, snr, horizon = 5, 0.0, 120.0
    doppler, delay = L.PRESETS["mpp"]
    pols = [Shifter(random.Random(seed * 2 + i)) for i in range(2)]
    ch = PS.ContinuousChannel(O.family(doppler), snr, seed, horizon + 60, doppler=doppler, delay_ms=delay)
    L.run(pols[0], pols[1], None, L.WORKLOADS["bulk"](random.Random(seed + 7)), seed=seed, horizon=horizon,
          phy=Phy(ch, dict(seed=seed), pols, snr, 0.0, []), cap=2)
    d = np.array(pairs)
    assert len(d) >= 15
    assert d.min() >= -0.1 and np.median(np.abs(d)) <= 0.1 and np.mean(np.abs(d) <= 0.6) >= 0.8, np.round(d, 2)


def test_harq_boost_db_has_no_phantom_gain_from_dead_burst():
    """A near-zero-MI reception contributes ~0 mutual information on fresh
    parity as well as Chase: harq_boost_db gives ~0 dB boost when the stored
    burst was noise, and positive IR gain when both receptions have signal."""
    s = MODES["w48-16qam-r2/3"]
    m_dead = float(P.capacity(-25.0, "gray-qam16"))
    m_near = float(P.capacity(7.0, "gray-qam16"))
    assert P.harq_boost_db(s, [(0, m_dead)], m_near) < 0.05
    assert P.harq_boost_db(s, [(0, m_near)], m_near) > 4.0
    if "fsk16r25-r1/3" in MODES:
        sc = MODES["fsk16r25-r1/3"]
        mc = float(P.capacity(-5.0, "gray-qam4"))
        assert P.harq_boost_db(sc, [(0, float(P.capacity(-25.0, "gray-qam4")))], mc) < 0.05
        assert P.harq_boost_db(sc, [(0, mc)], mc) > 2.5


@pytest.mark.hf_only("tests HF ladder transitions on a marginal 0/M burst vs deep fade")
def test_harq_credit_holds_mode_on_marginal_loss_and_drops_on_deep_fade():
    """Issue #360: a 0/M burst (held == 0) 1-2 dB below RV0 threshold steps
    bias[cur] down; without HARQ credit recommend() switches down and abandons
    the M stored RV0 codewords, while with HARQ credit RV1's combining gain
    keeps cur. On a deep SNR drop where RV1 also fails, recommend() drops."""
    from data2g.arq import link as L

    class FakeRx:
        def __init__(self, submode, slots, ok_set):
            self.submode, self.n_cw = submode, len(slots)
            self._slots, self._ok = slots, ok_set

        def decode(self, slot, mask_id, rv, key):
            s = self._slots[slot]
            return s.payload if (slot in self._ok and s.mask_id == mask_id and s.rv == rv) else None

        def forget(self, key):
            pass

    def make_b():
        bb = L.Station(1, G.GearShifter(), master=False, key=7)
        bb.policy.observe(m_hi, "w48-qpsk-r1/2", 1.0)
        bb.build()
        bb.policy.observe(m_dip, cur, 4.0)
        assert bb.handle(FakeRx(cur, tx1.slots, set(range(n_ctl))))
        return bb

    a = L.Station(0, G.GearShifter(), master=True, key=7)
    b0 = L.Station(1, G.GearShifter(), master=False, key=7)
    # Initial exchange at 10 dB AWGN so B recommends an upper mode, then A sends incompressible data
    m_hi = measured(10.0, 0.05, "w48")
    b0.policy.observe(m_hi, "w48-qpsk-r1/2", 1.0)
    ack0 = b0.build()
    a.policy.observe(m_hi, ack0.submode, 2.0)
    assert a.handle(FakeRx(ack0.submode, ack0.slots, set(range(len(ack0.slots)))))
    a.write(np.random.default_rng(0).bytes(8000))
    tx1 = a.build()
    cur = tx1.submode
    n_ctl = sum(1 for s in tx1.slots if s.mask_id[2] >= 128)
    n_data = len(tx1.slots) - n_ctl
    assert n_data >= 3

    # Burst arrives 2 dB lower: control decodes, 0/n_data data codewords decode (held == 0)
    m_dip = measured(8.0, 0.05, MODES[cur].band)
    b = make_b()
    assert len(b.rx.buf) == 0 and len(b.rx.harq) == n_data and b.rx.submode == cur

    st = G.view(b)
    assert st.held == 0 and st.rx_mode == cur and len(st.harq) == n_data
    # Without HARQ credit, shifter switches away from cur; with HARQ credit, it keeps cur
    rec_no = G.decode(make_b().policy.recommend(st._replace(harq=()))[0])
    rec_yes = G.decode(b.policy.recommend(st)[0])
    if not G.HARQ_CREDIT:
        assert rec_yes == rec_no
        return
    assert rec_no != cur and rec_yes == cur
    # CHAT ON ignores HARQ credit
    assert G.decode(make_b().policy.recommend(st._replace(chat=True))[0]) == G.decode(
        make_b().policy.recommend(st._replace(chat=True, harq=()))[0])

    # Deep drop (-6 dB): even RV1 cannot rescue cur, so shifter drops to a robust mode
    m_deep = measured(-6.0, 0.05, MODES[cur].band)
    b.policy.observe(m_deep, cur, 8.0)
    rec_deep = G.decode(b.policy.recommend(G.view(b))[0])
    assert rec_deep != cur

    # Control-only poll in ack-4f preserves rx.submode and rx.harq
    poll = a.on_timeout(allow_repeat=False)
    b.policy.observe(m_dip, poll.submode, 10.0)
    assert b.handle(FakeRx(poll.submode, poll.slots, set(range(len(poll.slots)))))
    assert b.rx.submode == cur and len(b.rx.harq) == n_data

    # Resend decoding all slots clears rx.harq and rx.submode
    reply = b.build()
    a.policy.observe(m_hi, reply.submode, 11.0)
    assert a.handle(FakeRx(reply.submode, reply.slots, set(range(len(reply.slots)))))
    # Force A to stay in cur so it resends without abandon, then decode all slots at B
    a.peer_recommend = G.encode(cur)
    tx2 = a.build()
    b.policy.observe(m_hi, tx2.submode, 13.0)
    assert b.handle(FakeRx(tx2.submode, tx2.slots, set(range(len(tx2.slots)))))
    assert not b.rx.harq and not b.rx.buf and b.rx.submode is None

    # Abandon clears rx.harq and rx.submode
    ack2 = b.build()
    a.policy.observe(m_hi, ack2.submode, 14.0)
    assert a.handle(FakeRx(ack2.submode, ack2.slots, set(range(len(ack2.slots)))))
    a.write(np.random.default_rng(1).bytes(4000))
    a.peer_recommend = G.encode(cur)
    tx3 = a.build()
    b.policy.observe(m_dip, tx3.submode, 15.0)
    assert b.handle(FakeRx(tx3.submode, tx3.slots, set(range(sum(1 for s in tx3.slots if s.mask_id[2] >= 128)))))
    assert b.rx.harq and b.rx.submode == tx3.submode
    b_ack = b.build()
    a.policy.observe(m_dip, b_ack.submode, 16.0)
    assert a.handle(FakeRx(b_ack.submode, b_ack.slots, set(range(len(b_ack.slots)))))
    a.peer_recommend = G.encode("qpsk-r1/5" if tx3.submode != "qpsk-r1/5" else "qpsk-r1/3")
    tx_ab = a.build()
    b.policy.observe(m_dip, tx_ab.submode, 18.0)
    assert b.handle(FakeRx(tx_ab.submode, tx_ab.slots, set(range(len(tx_ab.slots)))))
    assert not b.rx.harq and b.rx.submode is None


def test_harq_boost_db_rv_wrap_uses_latest_rv():
    """After wrapping the 4-RV cycle ([0, 1, 2, 3, 0]), the next resend is RV 1
    ((txs[-1][0] + 1) % cyc), not RV 0 ((max + 1) % cyc): when [0, 1) already
    holds a strong reception and [1, 2) only a weak one, ending at RV 0 puts
    the next transmission on fresh parity [1, 2) for a much larger boost than
    ending the same multiset of stored RVs at RV 3 (which resends RV 0)."""
    s = MODES["w48-16qam-r2/3"]
    m_lo = float(P.capacity(-10.0, "gray-qam16"))
    m_hi = float(P.capacity(4.0, "gray-qam16"))
    txs_end0 = [(0, m_lo), (1, m_lo), (2, m_lo), (3, m_lo), (0, m_hi)]  # next resend is RV 1
    txs_end3 = [(0, m_hi), (0, m_lo), (1, m_lo), (2, m_lo), (3, m_lo)]  # same stored slices, next resend is RV 0
    assert P.harq_boost_db(s, txs_end0, m_hi) > P.harq_boost_db(s, txs_end3, m_hi) + 2.0


@pytest.mark.hf_only("tests HF GearShifter min_success gate with pending HARQ resends")
def test_harq_credit_applies_before_min_success():
    """With min_success > 0, a mode whose RV0 probability alone dropped below
    min_success still passes the gate when pending HARQ resends lift the
    burst's effective per-slot decode probability above min_success."""
    from data2g.arq import link as L

    class FakeRx:
        def __init__(self, submode, slots, ok_set):
            self.submode, self.n_cw = submode, len(slots)
            self._slots, self._ok = slots, ok_set

        def decode(self, slot, mask_id, rv, key):
            s = self._slots[slot]
            return s.payload if (slot in self._ok and s.mask_id == mask_id and s.rv == rv) else None

        def forget(self, key):
            pass

    m_hi = measured(10.0, 0.05, "w48")
    a = L.Station(0, G.GearShifter(), master=True, key=7)
    b0 = L.Station(1, G.GearShifter(), master=False, key=7)
    b0.policy.observe(m_hi, "w48-qpsk-r1/2", 1.0)
    ack0 = b0.build()
    a.policy.observe(m_hi, ack0.submode, 2.0)
    assert a.handle(FakeRx(ack0.submode, ack0.slots, set(range(len(ack0.slots)))))
    a.write(np.random.default_rng(0).bytes(8000))
    tx1 = a.build()
    cur = tx1.submode
    n_ctl = sum(1 for s in tx1.slots if s.mask_id[2] >= 128)
    m_dip = measured(8.0, 0.05, MODES[cur].band)

    def make_b():
        bb = L.Station(1, G.GearShifter(min_success=0.88), master=False, key=7)
        bb.policy.observe(m_hi, "w48-qpsk-r1/2", 1.0)
        bb.build()
        bb.policy.observe(m_dip, cur, 4.0)
        assert bb.handle(FakeRx(cur, tx1.slots, set(range(n_ctl))))
        return bb

    b_yes = make_b()
    st = G.view(b_yes)
    rec_no = G.decode(make_b().policy.recommend(st._replace(harq=()))[0])
    rec_yes = G.decode(b_yes.policy.recommend(st)[0])
    if not G.HARQ_CREDIT:
        assert rec_yes == rec_no
        return
    p_rv0 = b_yes.policy.predicted[cur][0] * b_yes.policy.predicted[cur][1]
    assert p_rv0 < 0.88  # RV0 alone is below min_success; only the HARQ bonus passes the gate
    assert rec_no != cur and rec_yes == cur


def test_note_harq_requires_all_mi_keys_without_skipping_refine_measured():
    """A Policy.measured dict missing one of the four MI keys skips rx.harq
    without throwing, while refine_measured still runs update_measured."""
    from data2g.arq import link as L

    class PartialPolicy(L.Policy):
        def __init__(self, measured_dict):
            self.measured = measured_dict
            self.updated = []

        def choose(self, station, escalation):
            return "w48-qpsk-r1/2", 4

        def payload_bytes(self, submode):
            return 32

        def rv_cycle(self, submode):
            return 4

        def update_measured(self, m):
            self.updated.append(m)
            self.measured = m

    class FakeRx:
        def __init__(self, submode, slots, ok_set, upd=None):
            self.submode, self.n_cw = submode, len(slots)
            self._slots, self._ok, self._upd = slots, ok_set, upd

        def decode(self, slot, mask_id, rv, key):
            s = self._slots[slot]
            return s.payload if (slot in self._ok and s.mask_id == mask_id and s.rv == rv) else None

        def forget(self, key):
            pass

        def measure_data(self, m):
            return self._upd

    full = measured(8.0, 0.05, "w48")
    missing_one = {k: v for k, v in full.items() if k != "mi_c256-snr26"}

    a = L.Station(0, PartialPolicy(dict(full)), master=True, key=3)
    a.write(np.random.default_rng(2).bytes(200))
    tx = a.build()
    n_ctl = sum(1 for s in tx.slots if s.mask_id[2] >= 128)
    assert len(tx.slots) > n_ctl

    # Missing one MI key: measure_data runs (returning None), rx.harq stays empty without KeyError
    b1 = L.Station(1, PartialPolicy(dict(missing_one)), master=False, key=3)
    assert b1.handle(FakeRx(tx.submode, tx.slots, set(range(n_ctl)), upd=None))
    assert not b1.rx.harq and b1.rx.submode is None

    # measure_data refines measured to include all four MI keys: update_measured runs before note_harq
    b2 = L.Station(1, PartialPolicy(dict(full)), master=False, key=3)
    upd = dict(full, snr_est=5.0)
    assert b2.handle(FakeRx(tx.submode, tx.slots, set(range(n_ctl)), upd=upd))
    assert len(b2.policy.updated) == 1 and b2.policy.updated[0]["snr_est"] == 5.0
    assert len(b2.rx.harq) == len(tx.slots) - n_ctl and b2.rx.submode == tx.submode
