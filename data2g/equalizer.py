"""Pilot-based channel estimation for one burst, all pilots at once.

Replaces SSTVAE's per-frame Catmull-Rom interpolation. The whole burst
is buffered before decoding, so every estimate can use every pilot:

- `residual_cfo`: common phase rotation between consecutive pilots,
  summed over the burst. Fixes acquisition CFO errors of a few Hz that
  fading during the 88 ms preamble can cause.
- `delay_support`: which delays hold channel energy, from the burst-
  averaged power delay profile. Used to place the FFT window so every
  path lands inside the cyclic prefix, and as the frequency-correlation
  model below.
- `estimate`: 2-D LMMSE (Li/Cimini/Sollenberger robust form). Across
  carriers, a projection onto the delay support; in time, Wiener
  interpolation over the nearest pilots with a Gaussian (ITU-R F.1487)
  Doppler correlation whose spread is measured from the pilots. Returns
  the per-cu estimate MSE so the demapper can discount it.
"""

from functools import lru_cache

import numpy as np

from .config import FRAME_SAMPLES, FS, NC, NCP, RS, SYMS_PER_FRAME
from .waveform import ofdm

FRAME_S = FRAME_SAMPLES / FS
BB_FREQS = ofdm.BASEBAND_FREQS.astype(np.float64)  # wide band's, Hz; functions take a band's `bb`

# Delay grid for the support search, in samples relative to the demod
# window's own reference. Covers every delay the CP could hold and some
# margin either side, since timing may be off by a path before placement.
_DELAYS = np.arange(-2 * NCP, 2 * NCP + 1)

# Time-interpolation window: this many pilots either side.
TIME_TAPS = 4
# The same for equalizer.refine, whose data rows are 6 times denser.
DD_TAPS = 2
# Doppler spread assumed when the burst is too short to measure one.
DEFAULT_SPREAD_HZ = 2.0
# Per-carrier noise for the demapper (narrowband interference: a carrier
# on one tone): per_carrier_noise, with this many samples' worth of prior.
PER_CARRIER_NOISE = True
NOISE_SHAPE_PRIOR = 8.0


def residual_cfo(h_pilot: np.ndarray) -> float:
    """Hz. Unambiguous within +-1/(2*FRAME_S) = +-3.47 Hz."""
    d = np.sum(h_pilot[1:] * np.conj(h_pilot[:-1]))
    return float(np.angle(d) / (2 * np.pi * FRAME_S)) if np.abs(d) > 0 else 0.0


def delay_profile(h_pilot: np.ndarray, bb: np.ndarray = BB_FREQS) -> np.ndarray:
    """Burst-averaged power at each delay in _DELAYS (matched filter,
    Hann-tapered across carriers to keep sidelobes out of the support)."""
    nc = len(bb)
    w = np.hanning(nc + 2)[1:-1]
    steer = np.exp(-2j * np.pi * np.outer(bb, _DELAYS) / FS)
    g = (h_pilot * w) @ np.conj(steer)  # (F, D)
    return np.mean(np.abs(g) ** 2, axis=0)


def delay_support(h_pilot: np.ndarray, floor_db: float = -15.0, bb: np.ndarray = BB_FREQS) -> tuple[int, int]:
    """(first, last) delay in samples holding power within `floor_db` of
    the strongest, as local maxima of the profile (a lone path's
    mainlobe is ~10 samples wide and must not read as spread)."""
    p = delay_profile(h_pilot, bb)
    thr = p.max() * 10 ** (floor_db / 10)
    peaks = [
        i for i in range(1, len(p) - 1)
        if p[i] >= thr and p[i] >= p[i - 1] and p[i] >= p[i + 1]
    ]
    if not peaks:
        peaks = [int(np.argmax(p))]
    return int(_DELAYS[peaks[0]]), int(_DELAYS[peaks[-1]])


def window_shift(support: tuple[int, int]) -> int:
    """Samples to move the demod window later so every path in `support`
    lies inside the cyclic prefix, centred so timing drift has margin
    both ways.

    `support` is in apparent delay: what `delay_profile` reports, i.e. a
    path's delay past the window start. The window is interference-free
    for that path when its apparent delay a satisfies 0 <= a - s <= NCP
    after a shift s; the centre of the interval every path allows is
    s = (a_first + a_last - NCP) / 2. When the spread exceeds the CP no
    shift is clean, and the centre is still the least bad.
    """
    return int(round((support[0] + support[1] - NCP) / 2))


@lru_cache(maxsize=1024)
def _support_basis(bb_bytes: bytes, d0: int, d1: int) -> tuple:
    """(orthonormal basis of the support's delays across the carriers, its
    rank, the uncapped rank): geometry only, so computed once per band and
    support (the SVD was 7% of a Pat exchange's CPU, once per header read)."""
    bb = np.frombuffer(bb_bytes)
    nc = len(bb)
    slack = 4  # see _freq_smooth
    d = np.arange(d0 - slack, d1 + slack + 1)
    B = np.exp(-2j * np.pi * np.outer(bb, d) / FS)
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    r_full = int(np.sum(s > s[0] * 1e-2))
    r = min(r_full, nc - 2)
    U = np.ascontiguousarray(U[:, :r])
    U.setflags(write=False)
    return U, r, r_full


def _freq_smooth(h_pilot: np.ndarray, support: tuple[int, int], bb: np.ndarray = BB_FREQS) -> tuple:
    """Project each pilot vector onto the delay support. Returns the
    smoothed pilots, the per-carrier noise variance from the residual
    (inf where the support would take all but < 2 dimensions, i.e. narrow
    bands), the rank, and per carrier the fraction of its noise the
    residual keeps (1 - leverage).

    The rank is capped at nc - 2. That binds only on narrow bands (4
    carriers span 200 Hz, a 5 ms delay resolution, so any support takes
    all 4) and keeps some frequency smoothing there: a two-path channel is
    exactly rank 2 in frequency, and an uncapped projection on 4 carriers
    smooths nothing (measured, n4 polar on mpd: -7.5 dB capped, -6.25
    not). Noise then comes from the preamble instead (preamble_noise):
    the residual after a capped projection holds channel the cap left
    out (it cost n4 1 dB on mpp)."""
    nc = len(bb)
    d0, d1 = support
    # A mainlobe's worth of slack around the measured support. Narrow
    # bands can not resolve delay (a path's lobe is FS / bandwidth wide:
    # 16 samples on n10, 40 on n4), so a 2-4 ms two-path channel reads as
    # one delay and this projection loses the second path: signal power
    # reads ~20 dB low on n4 mpd at 20 dB (scripts/rx_audit.py). Padding
    # the support by one resolution cell fixed that and made decoding
    # WORSE on every narrow candidate, AWGN included (runs/
    # ladder_narrowpad.csv, 2026-09-24: +0.25-0.75 dB typical, n4 16-QAM
    # k1280 AWGN +2.5): the tight projection's noise averaging is worth
    # more than its bias at the SNRs these modes run at. Per-burst model
    # order selection could have both; untried.
    U, r, r_full = _support_basis(np.asarray(bb, dtype=np.float64).tobytes(), int(d0), int(d1))
    hs = (h_pilot @ np.conj(U)) @ U.T
    n0 = float(np.mean(np.abs(h_pilot - hs) ** 2) * nc / (nc - r)) if nc - r_full >= 2 else np.inf
    return hs, n0, r, 1 - np.sum(np.abs(U) ** 2, axis=1)


def preamble_noise(h_repeats: np.ndarray) -> float:
    """Per-carrier noise variance from the preamble's repeats (one LS
    channel estimate each, 20 ms apart): half the mean squared change
    between neighbours. The channel's own change over 20 ms is ~18 dB
    down even at mpd's 2 Hz. Thermal noise only: the repeats are
    identical, so their clip distortion cancels. Used where the pilot
    residual has no room (narrow bands); estimating from the change
    between frame pilots instead counted fading as noise."""
    return float(np.mean(preamble_noise_k(h_repeats)))


def preamble_noise_k(h_repeats: np.ndarray) -> np.ndarray:
    """preamble_noise per carrier."""
    return np.mean(np.abs(np.diff(h_repeats, axis=0)) ** 2, axis=0) / 2


def per_carrier_noise(power_k: np.ndarray, samples: int) -> np.ndarray:
    """Noise variance per carrier from `power_k`, each carrier's noise
    estimate averaged over `samples` observations (a mean of `samples`
    exponentials in Gaussian noise). A carrier reading above the 99th
    percentile of that, against the median carrier, is interfered with:
    it gets its own estimate, shrunk by NOISE_SHAPE_PRIOR samples toward
    the band's. Every other carrier gets the band's level, the mean over
    the clean carriers, so on a clean channel this is the old band-wide
    estimate. (Letting every carrier keep its own reading when above the
    band cost AWGN: carriers high by chance went timid, n10 at -5 dB
    0.45 -> 0.43 decoded. Averaging the interferer into the band's level
    made every clean carrier timid: w48 QPSK with a tone 3 dB under the
    signal 0.50 -> 0.21.)"""
    from scipy.stats import gamma

    q = gamma.ppf(0.99, samples) / samples
    hot = power_k > float(np.median(power_k)) * samples / (samples - 1 / 3) * q
    if hot.all():
        hot[:] = False
    band = float(np.mean(power_k[~hot]))
    own = (samples * power_k + NOISE_SHAPE_PRIOR * band) / (samples + NOISE_SHAPE_PRIOR)
    return np.where(hot, np.maximum(own, band), band)


def _doppler_corr(dt: np.ndarray, spread_hz: float) -> np.ndarray:
    sigma = spread_hz / 2
    return np.exp(-2 * (np.pi * sigma * dt) ** 2)


def measure_spread(hs: np.ndarray, n0_s: float) -> float:
    """Doppler spread (2 sigma, Hz) from the lag-1-frame correlation of
    the smoothed pilots, noise-corrected. Default when unmeasurable."""
    if len(hs) < 8:
        return DEFAULT_SPREAD_HZ
    p = np.mean(np.abs(hs) ** 2) - n0_s
    if p <= 0:
        return DEFAULT_SPREAD_HZ
    rho = np.abs(np.mean(hs[1:] * np.conj(hs[:-1]))) / p
    rho = float(np.clip(rho, 1e-3, 0.9999))
    sigma = np.sqrt(-np.log(rho) / 2) / (np.pi * FRAME_S)
    return float(np.clip(2 * sigma, 0.02, 4.0))


def estimate(h_pilot: np.ndarray, support: tuple[int, int], bb: np.ndarray = BB_FREQS,
             n0_pre: float = np.inf, n0_pre_k: np.ndarray | None = None) -> dict:
    """h_pilot (F+1, NC): LS estimates at every pilot, the closing one
    included; `n0_pre` a noise estimate from the preamble
    (preamble_noise). Returns h (F, 5, NC) at the data symbols, mse (F, 5, NC),
    n0 (thermal noise variance per carrier, in the same units), n0_k (the
    same per carrier: narrowband interference raises its carriers; from the
    pilot residual, or `n0_pre_k` (preamble_noise_k) where that has no room),
    signal power, and the measured Doppler spread."""
    n_p = len(h_pilot)
    n_f = n_p - 1
    nc = len(bb)
    hs, n0, r, keep = _freq_smooth(h_pilot, support, bb)
    # The residual's when it has room (wide band: it includes the pilots'
    # own clip distortion, which the preamble's estimate cannot see:
    # identical repeats clip identically, so it cancels in their
    # difference; taking the smaller of the two cost the wide band 1.25 dB
    # on mpd). The preamble's otherwise.
    n0_k = None
    if np.isfinite(n0):
        # each carrier's residual keeps 1 - its leverage of the noise (the
        # band edges keep more): without this, AWGN read edge carriers as
        # noisier (n10 at -5 dB: 0.45 -> 0.38 decoded)
        n0_k = per_carrier_noise(np.mean(np.abs(h_pilot - hs) ** 2, axis=0) / np.maximum(keep, 1e-3), n_p)
    elif n0_pre_k is not None and len(n0_pre_k) == nc:
        n0_k = per_carrier_noise(n0_pre_k, 7)
    if not np.isfinite(n0):
        n0 = n0_pre
    if not np.isfinite(n0):
        raise ValueError("no noise estimate: pass n0_pre on a band this narrow")
    n0_s = n0 * r / nc  # noise left on a smoothed pilot
    spread = measure_spread(hs, n0_s)
    p_sig = max(float(np.mean(np.abs(hs) ** 2) - n0_s), 1e-12)

    t_p = np.arange(n_p) * FRAME_S
    offs = np.arange(1, SYMS_PER_FRAME) / SYMS_PER_FRAME * FRAME_S
    h = np.zeros((n_f, SYMS_PER_FRAME - 1, nc), dtype=np.complex128)
    mse = np.zeros((n_f, SYMS_PER_FRAME - 1, nc))
    for f in range(n_f):
        lo = max(0, min(f - TIME_TAPS + 1, n_p - 2 * TIME_TAPS))
        j = np.arange(lo, min(n_p, lo + 2 * TIME_TAPS))
        Rpp = p_sig * _doppler_corr(t_p[j, None] - t_p[None, j], spread) + n0_s * np.eye(len(j))
        t = f * FRAME_S + offs
        Rdp = p_sig * _doppler_corr(t[:, None] - t_p[None, j], spread)  # (5, J)
        W = np.linalg.solve(Rpp, Rdp.T).T  # (5, J)
        h[f] = W @ hs[j]
        mse[f] = np.maximum(p_sig - np.real(np.sum(W * np.conj(Rdp), axis=1)), 0)[:, None]
    if n0_k is None or not PER_CARRIER_NOISE:
        n0_k = np.full(nc, n0)
    return {"h": h, "mse": mse, "n0": n0, "n0_k": n0_k, "p_sig": p_sig, "spread_hz": spread}


def time_shift_phase(shift: np.ndarray, bb: np.ndarray = BB_FREQS) -> np.ndarray:
    """Per-carrier phasor that undoes a demod-window shift of `shift`
    samples (array, per frame): a window moved later by s multiplies
    carrier k by exp(+2j*pi*f_k*s/FS)."""
    return np.exp(-2j * np.pi * np.outer(shift, bb) / FS)



def refine(h_pilot: np.ndarray, t_pilot: np.ndarray, z: np.ndarray, w: np.ndarray, t_rows: np.ndarray,
           support: tuple[int, int], est: dict, bb: np.ndarray = BB_FREQS) -> tuple[np.ndarray, np.ndarray]:
    """Decision-directed re-estimate: the pilots plus data cells whose
    symbols are (softly) known, as extra pilots.

    h_pilot (P, NC): LS at the pilots, at times t_pilot. z (R, NC): LS at
    every data symbol row, frame by frame (times t_rows (F, S), R = F S),
    in the pilots' units; w (R, NC)
    its inverse noise variance, 0 where nothing is known. Pilot rows are
    smoothed as in `estimate`; data rows by LMMSE across carriers with
    the delay support as prior (a row may know only a comb of carriers:
    one codeword's), keeping each carrier's posterior variance. Then per
    carrier, every data row is Wiener-interpolated in time over the
    nearby pilot and data rows, each with its own noise. `est`:
    estimate()'s result for p_sig, spread and n0. Returns h and mse
    (F, S, NC), in the pilots' units."""
    nc = len(bb)
    U, r, _ = _support_basis(np.asarray(bb, dtype=np.float64).tobytes(), int(support[0]), int(support[1]))
    p_sig, spread = est["p_sig"], est["spread_hz"]
    hs_p = (h_pilot @ np.conj(U)) @ U.T
    d = np.arange(support[0] - 4, support[1] + 5)  # _support_basis's slack
    B = np.exp(-2j * np.pi * np.outer(bb, d) / FS)
    Rf = p_sig / len(d) * (B @ B.conj().T)
    # data rows, batched by which carriers they know (after decoding: all)
    tr = t_rows.ravel()
    rows = np.flatnonzero(np.any(w > 0, axis=1))
    vals_d = np.zeros((len(rows), nc), dtype=np.complex128)
    var_d = np.zeros((len(rows), nc))
    known = w[rows] > 0
    pattern = np.packbits(known, axis=1)
    _, group = np.unique(pattern, axis=0, return_inverse=True)
    for gi in np.unique(group):
        sel = np.flatnonzero(group == gi)
        k = known[sel[0]]
        i_ = rows[sel]
        S = Rf[np.ix_(k, k)][None] + (1 / w[i_][:, k])[:, :, None] * np.eye(k.sum())
        G = np.linalg.solve(S, np.broadcast_to(Rf[k], (len(sel), k.sum(), nc))).conj().swapaxes(1, 2)  # (n, NC, K)
        vals_d[sel] = np.einsum("nck,nk->nc", G, z[i_][:, k])
        var_d[sel] = np.maximum(p_sig - np.real(np.sum(G * Rf[:, k].conj()[None], axis=2)), 1e-9 * p_sig)
    # one extra observation, uncorrelated with everything, pads the windows
    vals = np.concatenate([hs_p, vals_d, np.zeros((1, nc))])
    var = np.concatenate([np.full((len(hs_p), nc), est["n0"] * r / nc), var_d, np.full((1, nc), p_sig)])
    times = np.concatenate([t_pilot, tr[rows], [1e9]])
    # time: one window per frame (every obs within DD_TAPS frames of the
    # frame's middle), all its data rows at once; frames and carriers
    # stacked into one solve per chunk
    mid = t_rows.mean(axis=1)
    near = np.abs(times[None, :-1] - mid[:, None]) <= DD_TAPS * FRAME_S  # (F, O)
    J = int(near.sum(axis=1).max())
    idx = np.full((len(mid), J), len(times) - 1)
    for f, row in enumerate(near):
        o = np.flatnonzero(row)
        idx[f, : len(o)] = o
    F, S_ = t_rows.shape
    h = np.zeros((F, S_, nc), dtype=np.complex128)
    mse = np.zeros((F, S_, nc))
    for c in range(0, F, 16):
        ix = idx[c:c + 16]
        tj = times[ix]  # (f, J)
        Rt = p_sig * _doppler_corr(tj[:, :, None] - tj[:, None, :], spread)
        Rpp = Rt[:, None] + var[ix].transpose(0, 2, 1)[..., None] * np.eye(J)  # (f, NC, J, J)
        Rdp = p_sig * _doppler_corr(t_rows[c:c + 16][:, :, None] - tj[:, None, :], spread)  # (f, S, J)
        W = np.linalg.solve(Rpp, np.broadcast_to(Rdp.swapaxes(1, 2)[:, None], (len(ix), nc, J, S_)))  # (f, NC, J, S)
        h[c:c + 16] = np.einsum("fcjs,fjc->fsc", W, vals[ix])
        mse[c:c + 16] = np.maximum(p_sig - np.einsum("fcjs,fsj->fsc", W, Rdp), 0.0)
    return h, mse
