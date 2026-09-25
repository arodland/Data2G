"""Passive noise survey across the passband, for the gear shifter.

A burst in one band says nothing about the noise in a wider band's extra
spectrum, where HF noise is rarely flat and a neighbour may sit. The
receiver's audio always holds the whole passband, so it can look:

    s = survey(x)                 # any stretch of received audio
    s.band_floor_db("w48")        # noise per carrier over w48's carriers
    s.band_excess_db("w48")       # worst carrier above the passband's median floor

Per 50 Hz bin (the carrier grid, from M-sample segments): the floor is a
low quantile of segment power over time, so bursts, whether ours or
anyone's, do not count as noise as long as the bin is quiet some of the
time; `busy` is the fraction of segments well above that floor (an
intermittent neighbour shows there rather than in the floor).
"""

from dataclasses import dataclass

import numpy as np

from .config import BANDS, FS, M, RS

QUANTILE = 0.2  # of an exponential: scaled by -ln(1 - q) to its mean
BUSY_DB = 6.0
LO_HZ, HI_HZ = 300, 2700


@dataclass
class Survey:
    freqs: np.ndarray  # bin centres, Hz
    floor_db: np.ndarray  # noise power per bin, dB (arbitrary reference)
    busy: np.ndarray  # fraction of segments BUSY_DB above the floor

    def _bins(self, band: str) -> np.ndarray:
        return np.isin(self.freqs, BANDS[band].freqs)

    def band_floor_db(self, band: str) -> float:
        """Mean noise power over the band's carriers (dB, linear mean)."""
        return float(10 * np.log10(np.mean(10 ** (self.floor_db[self._bins(band)] / 10))))

    def band_excess_db(self, band: str) -> float:
        """The band's worst carrier above the passband's median floor."""
        return float(np.max(self.floor_db[self._bins(band)]) - np.median(self.floor_db))

    def band_busy(self, band: str) -> float:
        return float(np.max(self.busy[self._bins(band)]))


def survey(x: np.ndarray) -> Survey:
    """x: real audio at FS. Needs a few seconds for a stable floor."""
    n = len(x) // M
    seg = np.asarray(x[: n * M], dtype=np.float64).reshape(n, M)
    p = np.abs(np.fft.rfft(seg, axis=1)) ** 2  # (n, M/2 + 1), bins RS apart
    freqs = np.arange(p.shape[1]) * RS
    keep = (freqs >= LO_HZ) & (freqs <= HI_HZ)
    p, freqs = p[:, keep], freqs[keep]
    floor = np.quantile(p, QUANTILE, axis=0) / -np.log(1 - QUANTILE) + 1e-300
    busy = np.mean(p > floor * 10 ** (BUSY_DB / 10), axis=0)
    return Survey(freqs=freqs, floor_db=10 * np.log10(floor), busy=busy)


assert FS % M == 0 and FS // M == RS  # bins land on the carrier grid
