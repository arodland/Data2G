"""Acquisition: preamble detection, timing, and carrier frequency offset.

The preamble is one pilot symbol repeated (BandSpec.preamble_repeats). Each
repeat is matched-filtered against the band's own template on a grid of
CFO hypotheses (STEP_HZ apart, out to +-ACQUIRE_REACH_HZ),
and successive repeats' outputs are correlated with each other:

    D[n] = max_f  | sum_{r>=1} c_f[n + CP + r M] conj(c_f[n + CP + (r-1) M]) | / noise

c_f being the correlation with the template shifted by f and noise its
mean power over noise (a low quantile over the buffer, lowest bin).

- Over noise D has one distribution whatever the band, so a narrow band
  keeps the SNR its narrowness buys. The lag-M autocorrelation this
  replaced (runs/sync_lagm_backup.py) worked on the raw signal, needed a
  threshold of 0.634 on n4 against 0.278 wide, and spent it.
- Neighbouring repeats only: 2 Hz Doppler turns the phase ~1% of a
  cycle in 20 ms, while a coherent sum over the whole preamble
  decorrelates (why a longer preamble detected worse before).
- Differential rather than a sum of |c|^2: a burst's own data and
  isolated pilot symbols also correlate with the template, but not
  coherently from one repeat to the next. With the power sum, a
  preamble in an mpd fade lost to its own data (n4, 0 dB: 14 of 40
  bursts locked thousands of samples late; 5 with this), and it missed
  more at low SNR (n4 AWGN -10 dB: 49% against 5%).

The peaks one repeat either side (6/7 of the main one) are the periodic
preamble's; modem.receive reads the header at +-1, +-2 repeats.

Nothing here decides whether a peak is a preamble beyond the noise
threshold: at high SNR a transmission's own data symbols clear it too.
acquire() returns the best peak and runner-ups, and the header (exact ML
+ CRC-6 + score floor, modem.receive) is the gate.
"""

from dataclasses import dataclass, field as dataclass_field

import numpy as np
from scipy import signal

from ..config import (
    ACQUIRE_REACH_HZ,
    FIRST_PATH_FRAC,
    FIRST_PATH_SEARCH,
    FS,
    M,
    PREAMBLE_CP,
)
from . import ofdm

# CFO grid. A repeat is 20 ms, so a residual of STEP_HZ / 2 costs
# sinc^2(0.125), 0.2 dB, at the worst point between grid lines.
STEP_HZ = 12.5
# noise: this quantile of |c_f|^2 over the buffer, scaled to the mean of
# an exponential, lowest over the CFO bins (detection_stat). ponytail: one
# buffer-wide figure, needing >= 20% of the buffer noise-only in some
# bin; a live receiver with a long ring buffer and changing noise wants a
# sliding minimum instead.
NOISE_QUANTILE = 0.2


class SyncError(Exception):
    pass


@dataclass
class Acquisition:
    preamble_start: int  # index of first preamble sample (CP start)
    freq_offset: float  # Hz
    metric: float  # detection_stat at the peak (config.PREAMBLE_THRESHOLDS)
    # Runner-up (start, freq) hypotheses for the header to check: other
    # CFO bins at the same start, best first (a 4-carrier band shifted one
    # bin still overlaps itself in 3 carriers), then other peaks in time.
    alternatives: list = dataclass_field(default_factory=list)


NOISE_REF_HZ = (-625.0, 625.0)  # far bins for the noise level only (not searched)


def _cfo_grid(reach: float = ACQUIRE_REACH_HZ) -> np.ndarray:
    return np.arange(-reach, reach + STEP_HZ / 2, STEP_HZ)


def _repeat_corr(z: np.ndarray, t: np.ndarray, f: float) -> np.ndarray:
    """c_f[n] = sum_k z[n+k] conj(t[k] e^{j 2 pi f k / FS})."""
    tf = t * np.exp(2j * np.pi * f * np.arange(M) / FS)
    return signal.fftconvolve(z, np.conj(tf[::-1]), mode="valid")


def _repeat_corrs(z: np.ndarray, t: np.ndarray, freqs) -> np.ndarray:
    """_repeat_corr for every f in `freqs` (multiples of STEP_HZ): one
    forward FFT of z and of the template, then per f one inverse FFT of
    their product with the template's spectrum rolled by f. With the FFT
    length a multiple of FS / STEP_HZ every f is a whole number of bins,
    and a frequency-shifted template's spectrum is the unshifted one's,
    rolled, times a constant phase (3x fewer transforms than a
    fftconvolve per f: the receiver's search was ~40% of a burst's CPU)."""
    from scipy import fft

    per = round(FS / STEP_HZ)
    n = len(z) + M - 1
    L = per * fft.next_fast_len(-(-n // per))
    Z = fft.fft(z, L)
    G0 = fft.fft(np.conj(t[::-1]), L)
    out = np.empty((len(freqs), len(z) - M + 1), dtype=np.complex128)
    for i, f in enumerate(freqs):
        shift = round(f / STEP_HZ) * (L // per)
        c = fft.ifft(Z * np.roll(G0, shift), L)[M - 1 : n - M + 1]
        out[i] = c * np.exp(-2j * np.pi * f * (M - 1) / FS)
    return out


def detection_stat(z: np.ndarray, band=None, reach: float = ACQUIRE_REACH_HZ,
                   repeats: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """-> S (n_freqs, n_starts): the noise-normalized differential
    statistic per CFO hypothesis and candidate preamble start, and the
    hypotheses (Hz). `repeats`: the band's unless given."""
    S, q, freqs = _raw_stat(z, band, reach, repeats)
    # White noise is the same in every bin, so one level for all: the
    # lowest bin's, from bins the buffer's signal does not reach. Per-bin
    # levels failed when one burst filled most of the buffer: an n4 burst
    # (89% of a test buffer) raised every bin overlapping its 4 carriers,
    # the true CFO's among them, and n10 locked 400 Hz off at 30 dB.
    # NOISE_REF_HZ bins are for this level only: at +-150 Hz every searched
    # bin overlaps a narrow burst, noise read high, and n10 mpd lost ~0.4%.
    return S / q.min(), freqs


def _raw_stat(z, band=None, reach=ACQUIRE_REACH_HZ, repeats=None, levels_from: int | None = None, outs: list | None = None):
    """-> (S before noise normalization, each bin's noise level (the
    NOISE_REF_HZ bins last), the searched hypotheses). `levels_from`: the
    levels from correlation outputs from there on only, by one partition
    (StreamDetector: a chunk's new outputs, not its overlap with the last).
    `outs`: gets each searched bin's matched filter outputs, c_f."""
    band = band or ofdm.band("w")
    repeats = repeats or band.spec.preamble_repeats
    t = band.preamble_template()[PREAMBLE_CP : PREAMBLE_CP + M]
    t = t / np.linalg.norm(t)
    freqs = _cfo_grid(reach)
    n_out = len(z) - PREAMBLE_CP - repeats * M + 1
    S = np.empty((len(freqs), n_out))
    q = np.empty(len(freqs) + len(NOISE_REF_HZ))
    for i, c in enumerate(_repeat_corrs(z, t, list(freqs) + list(NOISE_REF_HZ))):
        p = np.abs(c) ** 2
        if levels_from is None:
            q[i] = np.quantile(p, NOISE_QUANTILE) / -np.log(1 - NOISE_QUANTILE)
        else:
            pn = p[levels_from:]
            k = int(NOISE_QUANTILE * (len(pn) - 1))
            q[i] = np.partition(pn, k)[k] / -np.log(1 - NOISE_QUANTILE)
        q[i] = max(q[i], 1e-12 * np.mean(p) + 1e-300)  # silence-only buffers (tests)
        if i < len(freqs):
            if outs is not None:
                outs.append(c)
            d = c[M:] * np.conj(c[:-M])  # each window against the one before it
            S[i] = np.abs(sum(d[PREAMBLE_CP + (r - 1) * M : PREAMBLE_CP + (r - 1) * M + n_out]
                              for r in range(1, repeats)))
    return S, q, freqs


class StreamDetector:
    """detection_stat for a stream, each sample's statistic computed once.

    A streaming receiver searched its whole buffer (1.9 s) every 0.25 s,
    on three bands: each start's statistic was recomputed ~8 times, and the
    search was 70-90% of a listening host's CPU. S at a start depends only
    on the preamble's span of audio after it, and on no carrier phase (the
    phase cancels in the products), so feeding new baseband audio computes
    S for the new starts alone. The noise level can't be the whole buffer's
    quantile any more: each fed chunk's per-bin level is kept, and a bin's
    level is the median of the last CHUNKS chunks (about the buffer's span),
    then the lowest bin's as before (detection_stat).

    Its matched filter outputs are kept too (C, from stream index c0): the
    frame pilot is the preamble's repeat symbol, so modem.find_copy's
    mid-burst search reuses them rather than filtering again."""

    CHUNKS = 8

    def __init__(self, band, reach: float = ACQUIRE_REACH_HZ):
        self.band, self.reach = band, reach
        self.span = PREAMBLE_CP + band.spec.preamble_repeats * M
        self.reset()

    def reset(self):
        self.tail = np.zeros(0, dtype=np.complex128)  # the last span - 1 samples fed
        self.S = np.zeros((len(_cfo_grid(self.reach)), 0))
        self.s0 = 0  # stream index of the start S[:, 0] is for
        self.fed = 0  # stream index one past the last sample fed
        self.levels: list = []  # per chunk, each bin's noise level
        self.C = np.zeros((len(_cfo_grid(self.reach)), 0), dtype=np.complex128)  # c_f per stream start
        self.c0 = 0  # stream index of C[:, 0]

    def feed(self, z: np.ndarray):
        """The next baseband samples of the stream (contiguous)."""
        if not len(self.S[0]) and not len(self.tail):
            self.s0 = self.fed
        self.fed += len(z)
        new = len(z)
        z = np.concatenate([self.tail, z])
        if len(z) < self.span + M:
            self.tail = z
            return
        # the level from the new outputs, at least 2000 (0.25 s) of the latest
        outs: list = []
        S, q, _ = _raw_stat(z, self.band, self.reach, levels_from=max(0, len(z) - M + 1 - max(new, 2000)), outs=outs)
        zs = self.fed - len(z)  # stream index of z[0], so of c[0]
        if not len(self.C[0]):
            self.c0 = zs
        have = self.c0 + len(self.C[0])  # outputs overlap the last chunk's: append the new ones
        self.C = np.concatenate([self.C, np.array(outs)[:, max(0, have - zs):]], axis=1)
        self.S = np.concatenate([self.S, S], axis=1)
        self.levels = (self.levels + [q])[-self.CHUNKS:]
        self.tail = z[len(S[0]):]

    def trim(self, start: int):
        """Drop the statistic of starts before stream index `start`."""
        k = min(max(0, start - self.s0), len(self.S[0]))
        self.S, self.s0 = self.S[:, k:], self.s0 + k
        k = min(max(0, start - self.c0), len(self.C[0]))
        self.C, self.c0 = self.C[:, k:], self.c0 + k

    def level(self) -> float | None:
        """The noise level stat() normalizes by (None before any)."""
        return float(np.median(np.array(self.levels), axis=0).min()) if self.levels else None

    def stat(self, lo: int, hi: int) -> np.ndarray:
        """Normalized S for stream starts [lo, hi) (-1 where not computed)."""
        out = np.full((len(self.S), max(0, hi - lo)), -1.0)
        a, b = max(lo, self.s0), min(hi, self.s0 + len(self.S[0]))
        if b > a and self.levels:
            q = self.level()
            out[:, a - lo:b - lo] = self.S[:, a - self.s0:b - self.s0] / q
        return out


def first_path(
    power: np.ndarray,
    peak: int,
    search: int = FIRST_PATH_SEARCH,
    frac: float = FIRST_PATH_FRAC,
    cyclic: bool = False,
) -> int:
    """The earliest local maximum within `search` samples *ahead* of
    `peak` that still holds `frac` of its power. `peak` itself if there
    is none.

    `power` is a correlation power profile against a known reference --
    the pilot fold for the blind path, |template correlation|**2 for the
    preamble path -- and `peak` its argmax.

    Why this exists rather than the argmax: on a multipath channel the
    argmax is the *strongest* path, which is not the *first* one, and
    which of the two is stronger changes as the channel fades. Syncing
    to a late path pushes the early path's energy in front of the
    demodulation window, where the cyclic prefix cannot cover it. See
    config.FIRST_PATH_SEARCH for the measurements and for why the caller
    must keep scoring at the argmax rather than here.

    The local-maximum requirement is load-bearing and not tidiness. A
    plain "earliest bin above the threshold" walks down the argmax's own
    correlation skirt and returns a position a few samples early on
    *every* channel, single-path ones included -- measured, that costs
    0.27 dB at mpd while still fixing the two-path case. Requiring a
    local maximum makes the single-path answer exactly the argmax again,
    which is what leaves awgn and mpg bit-identical.
    """
    n = len(power)
    thr = frac * power[peak]
    for d in range(search, 0, -1):
        i = peak - d
        if cyclic:
            i %= n
        elif i < 1:
            continue
        lo, hi = (i - 1) % n, (i + 1) % n
        if power[i] >= thr and power[i] >= power[lo] and power[i] >= power[hi]:
            return int(i)
    return int(peak)


# Runner-ups for the header to check (modem.receive): other CFO bins at
# the detection, and later detections in time.
#
# Detections are taken in time order (_crossings), not strongest first.
# Every OFDM symbol on the 50 Hz grid is M-periodic within itself (that
# is what its CP is), so a burst's own pilots and data also score on S:
# a 4-tone symbol at about a quarter of the pilot's correlation power,
# which leaves n4's preamble only ~10 dB above its own burst. Fading flat
# across 200 Hz moves the level more than that within a 1.4 s burst, so
# strongest-first lost 7.5% of n4 mpp and mpd bursts at 32 dB (the
# preamble ranked 11th-12th among the burst's peaks). A live receiver
# meets the preamble first anyway. (Also tried and dropped: ranking by
# phase coherence, which one strong product fools, and by an
# energy-normalized form, which favours starts overlapping the silence
# before a preamble.)
ALTERNATIVES = 3
TIME_ALTERNATIVES = 4


def _refine(z: np.ndarray, band, n: int, f: float) -> tuple[int, float]:
    """(start, CFO) from a detection at (n, f).

    Timing: the first path (config.FIRST_PATH_SEARCH) of the repeat-summed
    matched-filter power around n.

    CFO: the phase advance between the matched-filter outputs of
    successive repeats (in-band noise only), modulo 50 Hz, the alias
    nearest the grid's f (off by <= STEP_HZ / 2)."""
    t = band.preamble_template()[PREAMBLE_CP : PREAMBLE_CP + M]
    t = t / np.linalg.norm(t)
    R, n_pre = band.spec.preamble_repeats, band.spec.preamble_samples
    lo = max(0, n - FIRST_PATH_SEARCH - 8)
    hi = min(len(z) - n_pre, n + 8)
    if hi <= lo:
        return n, f
    c = _repeat_corr(z[lo : hi + n_pre], t, f)
    m = np.arange(hi - lo + 1)
    prof = sum(np.abs(c[m + PREAMBLE_CP + r * M]) ** 2 for r in range(R))
    start = lo + first_path(prof, int(np.argmax(prof)))
    cs = c[start - lo + PREAMBLE_CP + M * np.arange(R)]
    d = np.sum(cs[1:] * np.conj(cs[:-1]))
    if np.abs(d) == 0:
        return start, f
    # c_f removes f within a window only, so successive outputs advance by
    # the whole CFO, which they measure modulo FS / M; the grid picks the alias
    period = FS / M
    res = (np.angle(d) / (2 * np.pi) * period - f + period / 2) % period - period / 2
    return start, f + res


def _crossings(D: np.ndarray, threshold: float, span: int, limit: int) -> list[int]:
    """Detections in time order, as a streaming receiver meets them: at
    each threshold crossing, the peak within the next `span` samples
    (a preamble's length, so the crossing of its early partial overlap
    still finds it); the next search starts after that window."""
    out, n = [], 0
    while len(out) < limit:
        above = np.flatnonzero(D[n:] >= threshold)
        if not len(above):
            break
        c = n + int(above[0])
        out.append(c + int(np.argmax(D[c : c + span])))
        n = c + span
    return out


def acquire(
    z: np.ndarray,
    threshold: float | None = None,
    reach: float = ACQUIRE_REACH_HZ,
    search: tuple[int, int] | None = None,
    band=None,
    S: np.ndarray | None = None,
) -> Acquisition:
    """Find the preamble in baseband signal z (modem.to_baseband output,
    unfiltered: the matched filter is the band filter).

    `search` optionally restricts the preamble hunt to a [start, end)
    sample range (the rest of the signal is still used for frames).
    `S`: detection_stat's statistic for z's starts, precomputed
    (StreamDetector; -1 marks starts not to search)."""
    band = band or ofdm.band("w")
    if threshold is None:
        threshold = band.spec.preamble_threshold
    n_pre = band.spec.preamble_samples
    if len(z) < n_pre + 2 * M:
        raise SyncError("signal too short")
    if S is None:
        S, freqs = detection_stat(z, band, reach)
    else:
        S, freqs = S.copy(), _cfo_grid(reach)
    if search is not None:
        s0, s1 = max(0, int(search[0])), min(S.shape[1], int(search[1]))
        if s1 - s0 < 1:
            raise SyncError(f"empty search window {search}")
        out = (np.arange(S.shape[1]) < s0) | (np.arange(S.shape[1]) >= s1)
        S[:, out] = -1.0
    if S.max() < threshold:
        raise SyncError(f"no preamble found (peak {S.max():.1f} < {threshold:g})")
    peaks = _crossings(S.max(axis=0), threshold, n_pre, 1 + TIME_ALTERNATIVES)
    n = peaks[0]
    i = int(np.argmax(S[:, n]))
    peak = float(S[i, n])
    start, f_hat = _refine(z, band, n, freqs[i])

    # runner-ups: other CFO bins at this start more than a grid step
    # either side of the winner's, then the later crossings
    hyps = []
    col = S[:, n].copy()
    col[max(0, i - 2) : i + 3] = -1
    for _ in range(ALTERNATIVES):
        j = int(np.argmax(col))
        if col[j] < threshold:
            break
        hyps.append(_refine(z, band, n, freqs[j]))
        col[max(0, j - 2) : j + 3] = -1
    for n in peaks[1:]:
        hyps.append(_refine(z, band, n, freqs[int(np.argmax(S[:, n]))]))
    return Acquisition(
        preamble_start=start, freq_offset=float(f_hat), metric=peak,
        alternatives=[(st, float(f)) for st, f in hyps],
    )
