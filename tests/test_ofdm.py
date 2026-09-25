import numpy as np

from data2g.config import NC, NCP
from data2g.waveform import ofdm
from data2g.waveform.dsp import to_baseband


def test_symbol_loopback_via_baseband():
    rng = np.random.default_rng(1)
    n_sym = 20
    s = (rng.normal(size=(n_sym, NC)) + 1j * rng.normal(size=(n_sym, NC))) / np.sqrt(2)
    pad = np.zeros((2, NC), dtype=complex)
    z = to_baseband(ofdm.modulate_symbols(np.vstack([pad, s, pad])))
    for i in range(n_sym):
        got = ofdm.demod_window(z, (2 + i) * (160 + NCP) + NCP)
        snr = 10 * np.log10(np.mean(np.abs(s[i]) ** 2) / np.mean(np.abs(got - s[i]) ** 2))
        assert snr > 35, f"symbol {i}: {snr:.1f} dB"
