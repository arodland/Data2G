"""FM's own bands, f55 (450..3150 Hz) and f108 (300..5650 Hz): every
submode round-trips clean on both TX paths (modem.TX_PATH), and through the
FM channel simulator at high CNR on the radio path it is meant for. FM
profile only."""

import numpy as np
import pytest

from data2g import codes, config, constellation, fmchannel, modem
from data2g.config import FS, SUBMODES

pytestmark = pytest.mark.skipif(config.PROFILE != "fm", reason="FM's bands exist under DATA2G_PROFILE=fm only")
NAMES = [n for n, s in SUBMODES.items() if s.band in ("f55", "f108")]


def _burst(spec, seed, n_cw=2, tx_path=None):
    rng = np.random.default_rng(seed)
    sent = [rng.bytes(codes.payload_bytes(spec)) for _ in range(n_cw)]
    return sent, np.concatenate([np.zeros(FS // 2), modem.modulate(sent, spec, tx_path=tx_path), np.zeros(FS // 2)])


def _decoded(y, spec):
    r = modem.receive(y)
    assert r["spec"].name == spec.name
    return modem.decode_received(r)


@pytest.mark.parametrize("tx_path", modem.TX_PATHS)
@pytest.mark.parametrize("name", NAMES)
def test_round_trip_clean(name, tx_path):
    spec = SUBMODES[name]
    sent, x = _burst(spec, 1, tx_path=tx_path)
    b = _decoded(x + 1e-3 * np.random.default_rng(0).standard_normal(len(x)), spec)
    assert b.payloads == sent and all(b.crc_ok)


# narrow on the mic path (the radio model passes 300..3000 Hz, so f55's top
# carriers ride its skirt) and a 9600 port, wide on a 9600 port; each sent on
# its radio's TX path, at a station's level (the mic path's top modes quieter)
@pytest.mark.parametrize("name,radio", [("f55-qpsk-r40", "mic"), ("f55-16qam-r47", "mic"), ("f55-256l-r67", "mic"),
                                        ("f55-16qam-r47", "flat"), ("f55-256l-r80", "flat"),
                                        ("f108-qpsk-r41", "flat"), ("f108-64l-r59", "flat"), ("f108-256l-r75", "flat")])
def test_through_the_fm_channel(name, radio):
    spec = SUBMODES[name]
    sent, x = _burst(spec, 2, tx_path=radio)
    m = constellation.bits_per_symbol(constellation.load(spec.constellation))
    gain = config.FM_MIC_GAIN[m] if radio == "mic" else 1.0
    drive = fmchannel.drive_for_bandwidth(x[np.abs(x) > 0], radio, 15_000.0)[0] * gain
    y = fmchannel.apply_channel(x, 36.0, "static", radio, seed=3, drive_dev_hz=drive)
    b = _decoded(y, spec)
    assert b.payloads == sent and all(b.crc_ok)


@pytest.mark.skipif(config.EST_WINDOW == 0, reason="per-window channel estimates only (config.EST_WINDOW)")
def test_snr_est_reads_the_whole_bursts_noise():
    """A burst of several estimate windows whose noise rises 10 dB after the
    first: snr_est's noise is every window's, as the decoder's is (it was the
    first window's alone, 6 dB optimistic here)."""
    spec = SUBMODES["f55-qpsk-r21"]
    rng = np.random.default_rng(1)
    assert modem.frames_on_air(spec, 6) > 3 * config.EST_WINDOW
    x = modem.modulate([rng.bytes(codes.payload_bytes(spec)) for _ in range(6)], spec)
    noise = rng.normal(0, 0.05, len(x))
    noise[len(x) * 2 // 5:] *= 10 ** 0.5
    est = modem.receive(x + noise)["est"]
    assert abs(10 * np.log10(est["n0_snr"] / est["n0"])) < 1.0
