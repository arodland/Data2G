"""DFT-matrix OFDM: complex carrier amplitudes <-> waveform samples.

Passband modulation generates the real transmit waveform directly.
Demodulation operates on the complex baseband signal produced by
dsp.to_baseband, where carrier k sits at bin (k - 11) * RS Hz.
"""

import numpy as np

from dataclasses import dataclass
from functools import cached_property, lru_cache

from ..config import (
    BANDS, FS, M, NCP, NSYM, FCENTER, PILOT_PHASE_DEN, PREAMBLE_CP,
    BandSpec,
)

# Every frequency here is an integer number of Hz and every sample index
# is an integer, so `n*f/FS` has an exact integer remainder and the
# phasor depends only on `(n*f) mod FS`. Reducing first, in integer
# arithmetic, is exact and keeps the argument to exp() under one turn.
#
# Without it |theta| reaches 262 rad, where one ulp is 5.7e-14, so the
# phasors carried ~3e-14 of error and *which* entries rounded which way
# was an accident of numpy's complex-array arithmetic. Two reasons that
# was worth fixing, neither of them the accuracy itself:
#
#   * sin/cos of a large argument disagree between libms -- and between
#     x86-64 and Apple silicon -- by far more than they do near zero,
#     because implementations differ in how far they carry argument
#     reduction. This made the tables non-reproducible across platforms.
#   * The C++ port checks itself against these values, and a tolerance
#     sized by the reference's error rather than the port's is a much
#     weaker statement. See docs/native-app.md.
def _phasor(cycles_num: np.ndarray, sign: int = 1) -> np.ndarray:
    """exp(sign * 2j*pi * cycles_num / FS) for integer `cycles_num`."""
    return np.exp(sign * 2j * np.pi * (np.asarray(cycles_num) % FS) / FS)


@dataclass(frozen=True, eq=False)
class Band:
    """Everything carrier-specific for one config.BandSpec."""

    spec: BandSpec
    freqs: np.ndarray  # passband carrier frequencies, Hz (integers)
    bb: np.ndarray  # the same at baseband, multiples of RS
    mod: np.ndarray  # (NSYM, nc) passband modulation, phase ref at n = NCP
    demod: np.ndarray  # (nc, M) baseband demodulation over one useful window
    pilot: np.ndarray  # (nc,) unit-magnitude pilot

    @property
    def nc(self) -> int:
        return self.spec.nc

    def modulate_symbols(self, symbols: np.ndarray) -> np.ndarray:
        """(n_sym, nc) complex -> real waveform (n_sym * NSYM,)."""
        return np.real(self.mod @ symbols.T).T.reshape(-1)

    def demod_window(self, z: np.ndarray, start: int, backoff: int = 0) -> np.ndarray:
        """Demodulate one useful window of baseband signal starting at
        `start` (nominal first useful sample), `backoff` earlier into the
        CP; the resulting phase slope is absorbed by pilot equalization.
        Factor 2 undoes the real->analytic amplitude halving."""
        s = start - backoff
        win = z[s : s + M]
        if len(win) < M:
            win = np.pad(win, (0, M - len(win)))
        return (2.0 / M) * (self.demod @ win)

    def preamble_waveform(self) -> np.ndarray:
        """Real passband preamble: the pilot symbol, periodic with M over
        the whole block (double-length CP + the band's repeats)."""
        n = np.arange(self.spec.preamble_samples) - PREAMBLE_CP
        return np.real(_phasor(np.outer(n, self.freqs)) @ self.pilot)

    def preamble_template(self) -> np.ndarray:
        """Complex baseband replica of the preamble (timing correlation).
        Read-only, built once: every search and refine asked for it (7% of
        a ladder trial's CPU)."""
        return self._preamble_template

    @cached_property
    def _preamble_template(self) -> np.ndarray:
        n = np.arange(self.spec.preamble_samples) - PREAMBLE_CP
        t = 0.5 * (_phasor(np.outer(n, self.bb)) @ self.pilot)
        t.flags.writeable = False
        return t


@lru_cache(maxsize=None)
def band(name: str = "w") -> Band:
    s = BANDS[name]
    freqs = s.freqs.astype(np.int64)
    bb = freqs - FCENTER
    n_sym = np.arange(NSYM) - NCP
    # The pilot as an exact rational turn: see _phasor for why a phase
    # that is a property of the libm rather than of the format is a hazard.
    num = np.asarray(s.pilot_num) % PILOT_PHASE_DEN
    return Band(
        spec=s,
        freqs=freqs,
        bb=bb,
        mod=_phasor(np.outer(n_sym, freqs)),
        demod=_phasor(np.outer(bb, np.arange(M)), -1),
        pilot=np.exp(2j * np.pi * num / PILOT_PHASE_DEN),
    )


# Wide-band aliases (SSTVAE's names), for callers that predate bands.
_W = band("w")
CARRIER_FREQS, BASEBAND_FREQS = _W.freqs, _W.bb
MOD_MATRIX, DEMOD_MATRIX = _W.mod, _W.demod
modulate_symbols, demod_window = _W.modulate_symbols, _W.demod_window
preamble_waveform, preamble_template = _W.preamble_waveform, _W.preamble_template


def pilot_sequence() -> np.ndarray:
    return _W.pilot
