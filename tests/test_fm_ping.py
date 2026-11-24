"""FM ping / level check (data2g.arq.ping, data2g.arq.probe; FM plan M7)."""

import numpy as np
import pytest
from scipy import signal

from data2g import config, fmchannel
from data2g.arq import frames as F
from data2g.arq import ping as PG
from data2g.arq import probe as PR
from data2g.arq.engine import Engine
from data2g.config import FS, SNR_REF_BW_HZ

pytestmark = pytest.mark.skipif(config.PROFILE != "fm", reason="FM only")

M = dict(lo_hz=300.0, hi_hz=3350.0, snr_nb=24.5, snr_top=-11.0, snr_tilt=0.5, noise_tilt=-1.0,
         snr=[36.0, 34.5, 29.0, 23.0, 17.0, 34.0], sinad=[14.5, 30.0, 35.0, 28.0, 22.5, 30.0],
         rms_dbfs=-21.5, peak_dbfs=-8.0, clip_permille=0)


def test_wire_round_trip_and_quantisation():
    rep = PR.pack_report(M)
    assert len(rep) == PR.REPORT_BYTES
    body = PG.ping_body("N0CALL", "W1AW", rep)
    pb = 55  # f55-qpsk-r21: the identity and its report in one codeword
    assert len(F.Control(F.Core(ftype=F.SESSION), {F.T_PING: body}).pack(pb)) == 1
    frm, to, got = PG.unpack_ping(body)
    assert (frm, to) == ("N0CALL", "W1AW")
    got = PR.unpack_report(got)
    for k in ("lo_hz", "hi_hz", "snr_nb", "snr_top", "snr_tilt", "noise_tilt", "rms_dbfs", "peak_dbfs"):
        assert abs(got[k] - M[k]) <= 25.0 if k.endswith("hz") else abs(got[k] - M[k]) <= 0.25, k
    assert np.allclose(got["snr"], M["snr"], atol=0.25) and np.allclose(got["sinad"], M["sinad"], atol=0.25)
    assert PG.unpack_ping(PG.ping_body("N0CALL", "W1AW", None))[2] is None


def test_probe_fits_the_sound_card():
    """At the ping's level the probe's peaks stay under full scale (config.FM_TX_PEAK)."""
    for path in ("mic", "flat"):
        assert np.abs(PR.waveform(path, PG.tx_level(path))).max() < config.FM_TX_PEAK


def _report(radio, path, hot=1.0, cnr=30.0, seed=1):
    x = np.concatenate([np.zeros(FS // 2), PR.waveform(path, 1.0), np.zeros(FS // 2)])
    drive = {"mic": 1460.0, "flat": 2350.0}[radio] * hot * PG.tx_level(path)
    y = fmchannel.FmChannel("static", radio, cnr, seed=seed, duration_s=len(x) / FS + 2, drive_dev_hz=drive).apply(x)
    return PR.unpack_report(PR.pack_report(PR.summary(PR.analyse(y[FS // 2:]), PG.level(0.1 * y))))


@pytest.mark.parametrize("radio, path, hot, cnr, want", [
    ("mic", "mic", 1.0, 30, ["NARROW only", "level OK"]),
    ("mic", "mic", 0.5, 30, ["NARROW only", "level OK"]),
    ("mic", "mic", 1.0, 12, ["NARROW only", "level OK"]),
    ("mic", "mic", 2.0, 30, ["NARROW only", "lower the TX level about 6 dB"]),
    ("mic", "mic", 4.0, 30, ["NARROW only", "lower the TX level about 12 dB"]),
    ("mic", "flat", 1.0, 30, ["NARROW only", "set tx_path mic"]),  # a mic radio set flat
    ("flat", "flat", 1.0, 30, ["WIDE works", "level OK"]),
    ("flat", "flat", 0.3, 30, ["WIDE works", "level OK"]),
    ("flat", "flat", 1.0, 12, ["WIDE works", "level OK"]),
    ("flat", "flat", 2.0, 30, ["WIDE works", "lower the TX level about 6 dB"]),
    ("flat", "mic", 1.0, 30, ["set tx_path flat", "level OK"]),  # a 9600 port set mic: untilted
])
def test_verdicts_on_the_simulator(radio, path, hot, cnr, want):
    v = PR.verdict(_report(radio, path, hot, cnr), path)
    for w in want:
        assert any(w in s for s in v), (w, v)


def test_varied_link():
    """The reference step and its repeat apart: no level advice from this probe."""
    v = PR.verdict(dict(M, sinad=[14.5, 22.0, 35.0, 28.0, 22.5, 27.0]), "mic")
    assert any("varied" in s for s in v) and not any(w in s for s in v for w in ("level OK", "lower", "raise")), v


def _pair(lowpass_hz=None, seconds=40):
    """A pings B over noise (and, if given, a lowpass: a narrow path)."""
    a, b = Engine("N0CALL", seed=1), Engine("W1AW", seed=2)
    b.listen()
    a.send_ping("W1AW", 2)
    rng, block = np.random.default_rng(0), FS // 10
    sigma = np.sqrt((FS / 2) / SNR_REF_BW_HZ / 10 ** (25 / 10)) * 0.2  # 25 dB under the FM full-scale rms
    sos = signal.butter(8, lowpass_hz, fs=FS, output="sos") if lowpass_hz else None
    zi = [signal.sosfilt_zi(sos) * 0, signal.sosfilt_zi(sos) * 0] if sos is not None else None

    def path(x, i):
        if sos is None:
            return x
        y, zi[i] = signal.sosfilt(sos, x, zi=zi[i])
        return y

    a_out = b_out = np.zeros(block)
    a_ev, b_ev = [], []
    for _ in range(int(seconds * FS / block)):
        a_out, _ = a.step(path(b_out, 0) + rng.normal(0, sigma, block))
        b_out, _ = b.step(path(a_out, 1) + rng.normal(0, sigma, block))
        a_ev += a.events()
        b_ev += b.events()
        if sum(e.startswith("PINGVERDICT") for e in a_ev) >= 2:
            break
    return a_ev, b_ev


def test_two_stations_ping_and_reply():
    """One round trip: B (listening, no session) measures the probe and
    answers once; A gets PINGACK (both measurements) and the verdict."""
    a_ev, b_ev = _pair()
    assert any(e.startswith("PING N0CALL W1AW band=") for e in b_ev), b_ev
    acks = [e for e in a_ev if e.startswith("PINGACK W1AW far band=")]
    assert len(acks) == 1 and " near band=" in acks[0], a_ev
    v = [e for e in a_ev if e.startswith("PINGVERDICT W1AW ")]
    # an AWGN link, no radio: S/N follows the drive, so "level OK" or "raise"
    assert any("WIDE works" in e for e in v) and any("level OK" in e or "raise" in e for e in v), v


def test_narrow_path_recognised():
    """Over a path that stops near 3.3 kHz, the probe says so: nothing has to decode."""
    a_ev, _ = _pair(lowpass_hz=3300)
    v = [e for e in a_ev if e.startswith("PINGVERDICT W1AW ")]
    assert any("NARROW only" in e for e in v), a_ev


@pytest.mark.python_only("the profile is compiled into native (config::PROFILE)")
def test_hf_ignores_t_ping(monkeypatch):
    """The frame is FM's: an engine on another profile never reads it."""
    from data2g.arq import engine as E

    monkeypatch.setattr(E, "PROFILE", "hf")
    with pytest.raises(ValueError):
        Engine("N0CALL").send_ping("W1AW", 1)


def test_host_ping_command():
    """PING call [NARROW|WIDE]: OK and a ping queued (f55 identity, probe behind); a callsign that won't pack is WRONG."""
    from data2g import host

    h = host.Host(Engine("W1AW", seed=1))
    h.command("PING K2XYZ WIDE")
    h.command("PING BAD!CALL")
    assert h.out_cmd == ["OK", "WRONG"]
    assert [(b.submode, b.probe) for b in h.engine._extra] == [("f55-qpsk-r21", True)]


@pytest.mark.parametrize("up, down", [(1001, 1000), (999, 1000)])
def test_probe_survives_sample_clock_offset(up, down):
    """+-1000 ppm between the two sound cards (a USB card's 100-300 ppm flattened the staircase on air before
    probe._periods took the slip out): the staircase still steps."""
    x = np.concatenate([PR.waveform("flat", 1.0), np.zeros(800)])
    y = signal.resample_poly(x, up, down)
    y = y + np.random.default_rng(0).normal(0, 0.01, len(y))
    m = PR.summary(PR.analyse(y), PG.level(0.1 * y))
    assert m["snr"][PR.REF] - m["snr"][4] > 10, m["snr"]


@pytest.mark.parametrize("heard", [0.0, 0.5])
def test_exact_zeros_in_the_window_still_report(heard):
    """A digital squelch, a dropped block or the history's gap over our own
    transmission puts exact zeros in the probe's window: 0 dB there, a
    report that packs (0/0 stopped the host)."""
    y = PR.waveform("flat", 1.0) + np.random.default_rng(0).normal(0, 0.01, PR.n_samples())
    y[int(heard * len(y)):] = 0.0
    m = PR.summary(PR.analyse(y), PG.level(0.1 * y))
    assert np.isfinite([m[k] for k in ("snr_nb", "snr_top", "snr_tilt", "noise_tilt")] + m["snr"] + m["sinad"]).all(), m
    assert m["snr"][-1] == 0.0 and len(PR.pack_report(m)) == PR.REPORT_BYTES
