"""OFDM nods (docs/arq.md §3c): a nod (§3b) for the OFDM regime. A
band's preamble, then K keyed QPSK symbols where a header would be (a pilot
after every 5, as a header's), then nothing: one of M keyed sequences,
detected where the listener expects the reply, not decoded.

The family is keyed as nod.family (session key, direction, the nod's burst
number and the burst it acts on, mod 8), so a nod matches only the turn it
answers; to any other receiver its symbols are noise after a preamble. Each
sequence is redrawn until no valid header word of the band matches it over
SCREEN (what a stock receiver reads there).

Variants (band, K): ("w", 9) 0.47 s, a header and its copy frame's, and
("n10", 10) 0.50 s are the ones sent (K); ("w", 4) 0.33 s, a header's
length, was studied and needs 3 dB more (scripts/onod_study.py).

Native: arq/onod.hpp.
"""

from functools import lru_cache

import numpy as np

from .. import constellation, equalizer, modem
from ..config import (
    BANDS, DATA_SYMS_PER_FRAME, FS, LEADIN_SAMPLES, LEADOUT_SAMPLES, M as M_SAMPLES, NCP, NSYM, PREAMBLE_CP, SUBMODES,
)
from ..waveform import dsp, ofdm
from ..waveform import sync as _sync

M = 32  # sequences in a family
N_MEANINGS = 20  # nod.CODEBOOK's; the rest reserved (a detection of one is none)
SCREEN = 0.25
MASK64 = (1 << 64) - 1
PEAKS = modem.PRIME_PEAKS
# Acceptance: the best sequence's normalized correlation at least THR and
# the runner-up under it. THR per (band, K): 1e-6 false accepts per slot
# (+-ReplySlot.TOL_S) on white noise: the exponential tail over the top 1% of
# 600000 noise slots' best (scripts/onod_study.py --noise; their maxima 0.419,
# 0.278, 0.389; a Gaussian tail gives 0.398, 0.274, 0.393).
THR = {("w", 4): 0.42, ("w", 9): 0.29, ("n10", 10): 0.42}
K = {"w": 9, "n10": 10}  # the keyed symbols of a nod on each sync band


def band_of(submode: str) -> str | None:
    """The sync band a control-only reply in `submode` nods on: an OFDM ACK
    mode's, where one is defined (K); None otherwise."""
    s = SUBMODES.get(submode)
    return s.sync_band if s is not None and "ack" in submode and s.sync_band in K else None


def layout(k: int) -> np.ndarray:
    """Per on-air symbol after the preamble: True for a pilot. The keyed
    symbols in groups of 5 (the first the remainder), a pilot after each."""
    first = k - DATA_SYMS_PER_FRAME * ((k - 1) // DATA_SYMS_PER_FRAME)
    out = [False] * first + [True]
    for _ in range((k - first) // DATA_SYMS_PER_FRAME):
        out += [False] * DATA_SYMS_PER_FRAME + [True]
    return np.array(out)


def samples(band: str, k: int) -> int:
    """From the preamble's first sample to the last pilot's end."""
    return BANDS[band].preamble_samples + len(layout(k)) * NSYM


def seconds(band: str, k: int) -> float:
    return (LEADIN_SAMPLES + samples(band, k) + LEADOUT_SAMPLES) / FS


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


@lru_cache(maxsize=512)
def family(band: str, k: int, key: int, direction: int, bn8: int, acted8: int) -> np.ndarray:
    """(M, 2 * nc * K) +-1: the keyed family's bit sequences (+1 = bit 0)."""
    n = 2 * BANDS[band].nc * k
    hdr = modem._valid_signs(band)
    nh = hdr.shape[1]
    state = 0x0F0D << 48 | (key & 0xFFFFFFFF) << 16 | (direction & 1) << 8 | (bn8 & 7) << 4 | (acted8 & 7)
    out = np.empty((M, n), dtype=np.float32)
    for m in range(M):
        while True:
            bits = np.empty(-(-n // 64) * 64, dtype=np.float32)
            for i in range(0, len(bits), 64):
                state, z = _splitmix64(state)
                bits[i:i + 64] = [1.0 - 2.0 * ((z >> j) & 1) for j in range(64)]
            s = bits[:n]
            if n < nh or float(np.max(hdr @ s[:nh])) / nh <= SCREEN:
                break
        out[m] = s
    out.flags.writeable = False
    return out


def waveform(band: str, k: int, seq: np.ndarray) -> np.ndarray:
    """A sequence (+-1) -> unit-RMS audio, conditioned as an ACK mode's burst."""
    b = ofdm.band(band)
    lay = layout(k)
    syms = np.empty((len(lay), b.nc), dtype=np.complex128)
    syms[lay] = b.pilot
    syms[~lay] = constellation.modulate((seq < 0).astype(np.uint8), modem.QPSK).reshape(k, b.nc)
    x = np.concatenate([np.zeros(LEADIN_SAMPLES), b.preamble_waveform(), b.modulate_symbols(syms),
                        np.zeros(LEADOUT_SAMPLES)])
    return dsp.tx_condition(x, 0.0, b.spec.clip_overshoot, active=slice(LEADIN_SAMPLES, len(x) - LEADOUT_SAMPLES),
                            bandpass=BANDS[band].tx_bandpass)


def read(z: np.ndarray, start: int, band: str, k: int) -> np.ndarray:
    """Soft bits of the K keyed symbols of a nod whose preamble starts at
    `start` (z: baseband, frequency-corrected), as modem._read_header reads
    a header: each symbol's channel interpolated between its nearest
    references (the preamble's last repeats, the pilots)."""
    b = ofdm.band(band)
    R = b.spec.preamble_repeats
    ref = min(R, modem.REF_REPEATS)
    back = modem.HEADER_BACKOFF
    h_pre = np.array([b.demod_window(z, start + PREAMBLE_CP + r * M_SAMPLES, back) for r in range(R - ref, R)]).mean(axis=0) / b.pilot
    lay = layout(k)
    h0 = start + b.spec.preamble_samples
    t_sym = h0 + np.arange(len(lay)) * NSYM + NCP + M_SAMPLES / 2
    y_all = np.array([b.demod_window(z, int(t - M_SAMPLES / 2), back) for t in t_sym])
    u0 = start + PREAMBLE_CP + (R - ref) * M_SAMPLES
    t_p = np.concatenate([[u0 + ref * M_SAMPLES / 2], t_sym[lay]])
    h_p = np.concatenate([h_pre[None], y_all[lay] / b.pilot])
    j = np.clip(np.searchsorted(t_p, t_sym[~lay]) - 1, 0, len(t_p) - 2)
    a = ((t_sym[~lay] - t_p[j]) / (t_p[j + 1] - t_p[j]))[:, None]
    h_p = equalizer.freq_smooth(h_p, (back, back + NCP), b.bb)[0]
    ys, hs = y_all[~lay], (1 - a) * h_p[j] + a * h_p[j + 1]
    return constellation.llr(ys, hs, np.ones(ys.shape), modem.QPSK).reshape(-1)


def scores(x: np.ndarray, lo: int, hi: int, cfo: float, band: str, k: int, fam: np.ndarray) -> tuple | None:
    """Each sequence's best normalized correlation over the reads of a nod
    whose preamble starts in [lo, hi] (x's samples) at a CFO near `cfo`:
    modem.prime_ofdm's search (the window's strongest detections, each at
    +-1, +-2 repeats). -> ((M,) scores, (M,) starts, (M,) CFOs), or None."""
    z0 = dsp.freq_correct(dsp.to_baseband(np.asarray(x, dtype=np.float64)), cfo)
    b = ofdm.band(band)
    need = samples(band, k)
    S, freqs = _sync.detection_stat(z0, b, modem.PRIME_REACH_HZ)
    a, e = max(0, lo), min(S.shape[1], hi + 1)
    if e <= a:
        return None
    D = S[:, a:e].max(axis=0)
    best = np.full(M, -np.inf)
    where = np.zeros(M, dtype=np.int64)
    f_at = np.zeros(M)
    for _ in range(PEAKS):
        n = int(np.argmax(D))
        if D[n] <= 0:
            break
        D[max(0, n - M_SAMPLES):n + M_SAMPLES] = -1.0
        n += a
        start, f = _sync._refine(z0, b, n, float(freqs[int(np.argmax(S[:, n]))]))
        zb = dsp.freq_correct(z0, f)
        for al in (0, -1, 1, -2, 2):
            st = start + al * M_SAMPLES
            if not 0 <= st <= len(z0) - need:
                continue
            soft = read(zb, st, band, k).astype(np.float32)
            sc = (fam @ soft) / (np.sqrt(float(np.sum(soft**2)) * len(soft)) + 1e-12)
            up = sc > best
            best = np.where(up, sc, best)
            where[up] = st
            f_at[up] = cfo + f
    return best, where, f_at


def detect(x: np.ndarray, lo: int, hi: int, cfo: float, band: str, k: int, fam: np.ndarray,
           thr: float | None = None) -> tuple[int, int, float, float] | None:
    """A nod of `fam` in the slot -> (meaning, start, score, CFO), or None:
    none over the threshold, a runner-up over it too, or a reserved one."""
    got = scores(x, lo, hi, cfo, band, k, fam)
    if got is None:
        return None
    sc, where, f_at = got
    order = np.argsort(sc)[::-1]
    t = THR[(band, k)] if thr is None else thr
    if sc[order[0]] < t or sc[order[1]] >= t or order[0] >= N_MEANINGS:
        return None
    m = int(order[0])
    return m, int(where[m]), float(sc[m]), float(f_at[m])


if __name__ == "__main__":  # a self-check: a clean nod detects, noise does not
    rng = np.random.default_rng(1)
    for band, k in THR:
        fam = family(band, k, 1234, 1, 3, 5)
        x = np.concatenate([np.zeros(3000), waveform(band, k, fam[7]), np.zeros(FS)])
        x = x + rng.normal(0, 0.05, len(x))
        got = detect(x, 3000 + LEADIN_SAMPLES - 1200, 3000 + LEADIN_SAMPLES + 1200, 0.0, band, k, fam)
        assert got is not None and got[0] == 7, got
        assert detect(rng.normal(size=len(x)), 1800, 4200, 0.0, band, k, fam) is None
        print(band, k, f"{seconds(band, k):.3f} s", "ok", got)
