"""FM ping / level check (data2g.arq.ping; FM plan M7)."""

import numpy as np
import pytest

from data2g import config, fmchannel, modem
from data2g.arq import burst as B
from data2g.arq import frames as F
from data2g.arq import ping as PG
from data2g.arq.engine import Engine
from data2g.arq.phy import tx_audio
from data2g.arq.policy import GearShifter
from data2g.config import FS, SNR_REF_BW_HZ

pytestmark = pytest.mark.skipif(config.PROFILE != "fm", reason="FM only")

M = dict(snr_db=21.3, rms_dbfs=-18.2, peak_dbfs=-3.1, clip_permille=2, thirds_db=[24.0, 20.5, 15.0], dist_db=0.7,
         noise_tilt_db=2.5)


def _ping(cap=1, report=None):
    mode = PG.PING_MODES[cap]
    pb = GearShifter().payload_bytes(mode)
    payloads = F.Control(F.Core(ftype=F.SESSION), {F.T_PING: PG.ping_body("N0CALL", "W1AW", report, pb)}).pack(pb)
    return B.TxBurst(mode, [B.Slot(B.ctl_mask(0, i, 0), 0, p) for i, p in enumerate(payloads)], 0, cap), payloads


def test_wire_round_trip_and_quantisation():
    _, payloads = _ping(report=PG.pack_report(M))
    assert len(payloads) == F.MAX_CTL  # the body fills the control: 4 codewords to measure on
    frm, to, rep = PG.unpack_ping(F.Control.unpack(payloads).ext[F.T_PING])
    got = PG.unpack_report(rep)
    assert (frm, to) == ("N0CALL", "W1AW")
    for k in ("snr_db", "rms_dbfs", "peak_dbfs", "dist_db", "noise_tilt_db"):
        assert abs(got[k] - M[k]) <= 0.25, k
    assert got["clip_permille"] == 2 and np.allclose(got["thirds_db"], M["thirds_db"], atol=0.25)
    assert PG.unpack_ping(F.Control.unpack(_ping()[1]).ext[F.T_PING])[2] is None


def _measure(radio, path, hot=1.0, cnr=20.0, cap=1):
    b, payloads = _ping(cap)
    x = np.concatenate([np.zeros(FS // 2), tx_audio(b, tx_path=path, level=1.0), np.zeros(FS // 2)])
    drive = {"mic": 1460.0, "flat": 2350.0}[radio] * hot * PG.tx_level(path)
    y = fmchannel.FmChannel("static", radio, cnr, seed=1, duration_s=len(x) / FS + 2, drive_dev_hz=drive).apply(x)
    return PG.measure(modem.receive(y), payloads, PG.level(0.1 * y))


@pytest.mark.parametrize("radio, path, hot, want", [
    ("mic", "mic", 1.0, "level OK"),
    ("mic", "flat", 1.0, "limiter"),  # a mic radio set flat: full level into its limiter
    ("mic", "mic", 2.0, "limiter"),  # 6 dB too hot
    ("flat", "flat", 1.0, "level OK"),
    ("flat", "mic", 1.0, "tx_path flat"),  # a 9600 port set mic: untilted
])
def test_verdicts_on_the_simulator(radio, path, hot, want):
    v = PG.verdict(_measure(radio, path, hot), path)
    assert any(want in s for s in v), v
    if want == "level OK":
        assert v == ["level OK"]


def test_two_stations_ping_and_reply():
    """A pings B over a noisy link; B (listening, no session) answers once;
    A gets PINGACK (both measurements) and a verdict."""
    a, b = Engine("N0CALL", seed=1), Engine("W1AW", seed=2)
    b.listen()
    a.send_ping("W1AW", 1)
    rng, block = np.random.default_rng(0), FS // 10
    sigma = np.sqrt((FS / 2) / SNR_REF_BW_HZ / 10 ** (25 / 10)) * 0.2  # 25 dB under the FM full-scale rms
    a_out = b_out = np.zeros(block)
    a_ev, b_ev = [], []
    for _ in range(int(20 * FS / block)):
        a_out, _ = a.step(b_out + rng.normal(0, sigma, block))
        b_out, _ = b.step(a_out + rng.normal(0, sigma, block))
        a_ev += a.events()
        b_ev += b.events()
        if any(e.startswith("PINGVERDICT") for e in a_ev):
            break
    assert any(e.startswith("PING N0CALL W1AW snr=") for e in b_ev), b_ev
    acks = [e for e in a_ev if e.startswith("PINGACK W1AW far snr=")]
    assert len(acks) == 1 and " near snr=" in acks[0], a_ev
    assert any(e.startswith("PINGVERDICT W1AW ") for e in a_ev)


@pytest.mark.python_only("the profile is compiled into native (config::PROFILE)")
def test_hf_ignores_t_ping(monkeypatch):
    """The frame is FM's: an engine on another profile never reads it."""
    from data2g.arq import engine as E

    monkeypatch.setattr(E, "PROFILE", "hf")
    with pytest.raises(ValueError):
        Engine("N0CALL").send_ping("W1AW", 1)


def test_host_ping_command():
    """PING call [NARROW|WIDE]: OK and a burst queued; a callsign that won't pack is WRONG."""
    from data2g import host

    h = host.Host(Engine("W1AW", seed=1))
    h.command("PING K2XYZ WIDE")
    h.command("PING BAD!CALL")
    assert h.out_cmd == ["OK", "WRONG"]
    assert [b.submode for b in h.engine._extra] == ["f108-qpsk-r21"]
