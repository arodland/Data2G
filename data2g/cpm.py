"""Single-carrier constant-envelope modes (continuous-phase M-FSK,
noncoherent) with Costas-array sync and a header: the ARQ's bottom rungs.
Prototyped in the cpm-modes fork (its scripts/cpm_study.py, cpm_spectrum.py):
4-7 dB better than OFDM of similar rate on fading, PEP-fair, 10% points.

Signal: M tones spaced by the symbol rate R (h = 1), continuous phase, the
OFDM bands' TX bandpass over the tones with clip-and-filter at no headroom,
its margin (and a glide at each tone change) by the session's bandwidth cap
(TX_FILTERS: envelope peak 0.1-0.6 dB over average; beyond the segment or
band edge under -37 dB). A grid (tones, spacing) is what OFDM calls a sync
band.

Codewords go through data2g.codes like OFDM's (CRC masks, scrambler, RVs,
soft combining): a CpmSpec quacks like a config.SubmodeSpec there. Burst:

    [front Costas block][header][ctl][ctl RV1?][data ...][mid block][header][data ...]

- ctl: the grid's short control codeword (polar, CTL_K of CTL_N), one per
  burst, twice (RV 0, RV 1) when the ARQ duplicates control (ARQ_DUP);
- data: 0..MAX_DATA codewords of the mode (960 bits): GF(M) LDPC, one
  code symbol per tone (data2g.gfldpc; protocol 14, was binary LDPC on
  bit LLRs), decoded on each tone's likelihoods (symbol_ll);
- two sync blocks: the front, and a mid block SPACING_S of symbols later
  (after the data, on a burst shorter than that). Nothing else in the burst
  tracks timing or CFO: 30 ppm over a 90 s burst is ~22 samples of T = 320.
  Later blocks (every SPACING_S) were dropped on 2026-10-23: no receiver
  path read them, and they were ~8% of a long burst's air;
- header (twice, after each block): 16 bits as OFDM's: index = mode index
  + 2 x duplicated, n field = data codewords + 1, CRC-6. The receiver
  locks on the front and the first copy (find); if that copy faded, on
  both once the second is in (late_lock); if the front was missed, on the
  mid block and both copies (find_mid). A short burst's second copy sits
  after its data, so each word is read at its own rows.

Native: native/core/cpm (grids and specs compiled in, tables::CPM_*).
Parity: tests/test_native_cpm.py; under `pytest --native`
tests/native_subs/cpm.py substitutes by name, private _energies, _shares
and _peak_ratio included.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import permutations

import numpy as np

from .config import CLIP_OVERSHOOT, FS

PREAMBLE_S = 0.45  # the front sync block
BLOCK_S = 0.25  # the mid sync block, about
SPACING_S = 2.5  # symbols from the front block to the mid one, about (MPG fades last ~1-3 s)
HDR_S = 0.25  # a header copy, about
HDR_COPIES = 2  # one after each sync block
MAX_DATA = 8  # data codewords per burst
PEAK_RATIO = 0.7  # a new front's design bar on _peak_ratio off its start (c8r50_front_search); no lock gate
RAMP_S = 0.01  # amplitude ramp at a burst's ends (key clicks)
DATA_N = 960
CTL_K, CTL_N = 184, 360  # 20 bytes of control (CRC-24); 360 bits fill whole symbols at 3, 4 and 5 bits per tone


@dataclass(frozen=True)
class Grid:
    """Tones and their spacing: a sync band (its modes share sync, header,
    and the short control codeword)."""
    name: str
    m: int  # tones
    rate: float  # symbols/s = tone spacing (Hz)
    center: float = 1500.0
    bp: float = 50.0  # nominal margin beyond the outer tones, Hz: the width the caps judge (TX_FILTERS: the TX's)
    clip_db: float = 0.0  # clip-and-filter headroom

    @property
    def T(self) -> int:  # samples per symbol
        return int(round(FS / self.rate))

    @property
    def bits(self) -> int:
        return int(np.log2(self.m))

    @property
    def f0(self) -> float:  # lowest tone, a multiple of R
        return round((self.center - (self.m - 1) * self.rate / 2) / self.rate) * self.rate

    @property
    def bandwidth(self) -> float:
        return self.m * self.rate + 2 * self.bp


GRIDS = {g.name: g for g in [
    Grid("c16r25", 16, 25.0),  # 400 Hz of tones: a 500 Hz segment
    Grid("c8r50", 8, 50.0),  # the same
    Grid("c32r62", 32, 62.5, bp=150.0),  # 2000 Hz of tones, 2300 Hz wide (the 2400 Hz cap)
]}


@dataclass(frozen=True)
class TxFilter:
    """A grid's TX filter: the bandpass margin beyond the outer tones (Hz),
    the frequency trajectory smoothed (raised cosine) over this fraction of
    a symbol at each tone change, and clip-and-filter passes (overshoot as
    config.CLIP_OVERSHOOT, its last factor repeated). Receivers don't see it."""
    bp: float
    glide: float = 0.0
    passes: int = 3

    @property
    def overshoot(self) -> tuple:
        return CLIP_OVERSHOOT + CLIP_OVERSHOOT[-1:] * (self.passes - len(CLIP_OVERSHOOT))


# The TX filter by bandwidth cap code (arq.policy.CAP_HZ: 0 500 Hz, 1 1200,
# 2 2400). SSB PEP is the envelope, and a narrow margin rings at every tone
# jump; a wider margin where the cap leaves room, more passes and a short
# glide take the envelope to within 0.1-0.6 dB of its average. scripts/
# cpm_papr_study.py (2026-10-07), end to end, 150 trials a point, PEP-fair
# 10% points against the filter before (bp 50 / 150, 3 passes, no glide):
#   c8r50   500 Hz -0.3 AWGN / -0.8 MPP; wider -1.0 / -1.0 (bp 350: no more)
#   c16r25  500 Hz -0.85 / -0.4;        wider -0.9 / -0.9 (bp 350: no more)
#   c32r62  -0.1 / -0.5
# Out of band under -37 dB beyond the cap's edges around the tones' centre
# (tests/test_cpm.py; OFDM: -3 to -5). c32r62 is allowed at 2400 Hz only.
# The receivers are unchanged; the energy inputs read each cap's peak
# (arq.phy.peak_db).
TX_FILTERS = {
    "c16r25": (TxFilter(75.0, 0.2, 6), TxFilter(150.0, 0.1, 6), TxFilter(150.0, 0.1, 6)),
    "c8r50": (TxFilter(75.0, 0.2, 6), TxFilter(150.0, 0.1, 6), TxFilter(150.0, 0.1, 6)),
    "c32r62": (TxFilter(200.0, 0.1, 6),) * 3,
}


def tx_filter(grid: str, cap: int) -> TxFilter:
    return TX_FILTERS[grid][cap]


@dataclass(frozen=True)
class CpmSpec:
    """A CPM mode (or a grid's control codeword), shaped like
    config.SubmodeSpec where data2g.codes (codes.CodeSpec) and the ARQ
    read it."""
    name: str
    grid: str
    index: int  # in its grid's header (-1: the control codeword)
    code: str  # "gf" (data: GF(M) LDPC) | "polar" (control)
    k: int  # info bits incl. CRC
    coded_bits: int
    frames_per_cw: int = 0
    clip_headroom_db: float | None = None
    family: str = field(default="cpm", compare=False)

    @property
    def band(self) -> str:
        return self.grid

    @property
    def sync_band(self) -> str:
        return self.grid

    @property
    def constellation(self) -> str:
        return f"fsk{GRIDS[self.grid].m}"

    @property
    def headroom(self) -> float:
        return 0.0

    @property
    def bits_per_cu(self) -> int:
        return GRIDS[self.grid].bits

    @property
    def n_sym(self) -> int:
        return self.coded_bits // self.bits_per_cu


def _grid_name(g: Grid) -> str:
    return f"fsk{g.m}r{int(g.rate)}"


SPECS = {s.name: s for g in GRIDS.values() for s in (
    CpmSpec(f"{_grid_name(g)}-r1/3", g.name, 0, "gf", 320, DATA_N),
    CpmSpec(f"{_grid_name(g)}-r1/2", g.name, 1, "gf", 480, DATA_N),
)}
CTL = {g: CpmSpec(f"{_grid_name(GRIDS[g])}-ctl", g, -1, "polar", CTL_K, CTL_N) for g in GRIDS}


# Locks (cpm-acq-gap, 2026-10-26; scripts/cpm_floor_study.py's stage arms,
# scripts/cpm_acq_fit.py, scripts/cpm_false_locks.py). detect's coarse score
# is only a CPU floor for its fine stage (about the median a 0.25 s hop of
# noise scores: the coarse grid's T/8 and R/8 quantisation cost lost fronts
# 0.03-0.05 there); a lock is judged by tone_llr summed over the known
# symbols, the best valid word's: front + header copy 1 (find, LOCK_LLR[0]),
# front + both copies (late_lock, [1]), front + mid block + both copies
# (find_mid, MID_LLR), each block floored at 0 (_segments), and the word's
# margin over the runner-up (HDR_MARGIN). That replaced the share thresholds
# and PEAK_RATIO's gate (fronts at the floor read 0.6-0.7 there): the old
# gates lost 1-1.7 dB to the genie on fading for c8r50/c16r25 (fronts below
# PEAK_RATIO and SYNC_THRESHOLD) and 0.7-0.9 dB for c32r62 (both copies right
# but under HEADER2_THRESHOLD). Largest score on 1000 buffers a source (noise,
# OFDM, other grids, the grid's own cut bursts; 0-30 dB, AWGN/MPG): front
# paths 15-20, mid 16-20; bursts the old gates lost read 20-30 (c8r50/c16r25)
# and 50-80 (c32r62). Streaming Receiver, 8 h a source and grid: 0 false
# locks on noise, OFDM and other grids (old: 2 on other grids), 4 on two
# stations' own-grid traffic (old: 25). The margins: a strong front once
# carried a faded copy's wrong word (0.1-1 apart) past the floor.
SYNC_THRESHOLD = {"c16r25": 0.07, "c8r50": 0.12, "c32r62": 0.05}
MID_THRESHOLD = {"c16r25": 0.14, "c8r50": 0.20, "c32r62": 0.07}
LOCK_LLR = {"c16r25": (24.0, 22.0), "c8r50": (24.0, 22.0), "c32r62": (24.0, 22.0)}
MID_LLR = {"c16r25": 24.0, "c8r50": 24.0, "c32r62": 24.0}
HDR_MARGIN = (8.0, 4.0)  # the read word's LLR over the runner-up's: one copy, both
LLR_EPS = 0.02  # tone_llr's outlier share: a row whose tone the model rejects costs at most -ln(EPS)
WAIT_LLR = {"c16r25": 0.0, "c8r50": 0.0, "c32r62": 0.0}  # a front waits for its second copy from this LLR
# An early lock (front + header copy 1) is looked at again once its mid block
# and second header copy are in: confirm() under this, the burst is not
# there (the streaming Receiver lets it go). The lock was never checked
# again, and held BUSY and this station's replies for the length its word
# claimed (c16r25: up to 82 s) on a burst that stopped or a lock on another
# station's data. scripts/cpm_confirm_study.py (3000 bursts a grid, AWGN /
# MPP / MPD from the floor to 12 dB over it; early locks under the floor, and
# how many of those decoded whole, of 1734 / 1956 / 1858 that did):
#   floor   c16r25           c8r50            c32r62         cut     on data
#     0     7, 0 decoded    13, 3 decoded    13, 0        97-100%   99-100%
#    -2     5, 0             4, 0             9, 0        89-100%   98.5-100%
#    -4     3, 0             2, 0             7, 0        75-99%    96-100%
# cut: the burst stops after its first header copy; on data: the lock's word
# at a start inside a burst's data. None: never.
CONFIRM_LLR = -2.0


def grid_specs(grid: str) -> list:
    return [s for s in SPECS.values() if s.grid == grid]


# --- sync patterns -------------------------------------------------------------------

def _is_costas(p) -> bool:
    n = len(p)
    for d in range(1, n):
        v = [p[i + d] - p[i] for i in range(n - d)]
        if len(set(v)) < len(v):
            return False
    return True


def _welch(p: int) -> np.ndarray:
    g = next(g for g in range(2, p) if len({pow(g, i, p) for i in range(p - 1)}) == p - 1)
    return np.array([pow(g, i, p) - 1 for i in range(p - 1)])


def _prime(q: int) -> bool:
    return q > 1 and all(q % d for d in range(2, int(q**0.5) + 1))


@lru_cache(maxsize=None)
def costas(m: int) -> tuple:
    """A Costas array using at most m tones: Welch (order p - 1, p prime)
    when one uses nearly all of them, else the first found by search."""
    p = max(q for q in range(3, m + 2) if _prime(q))
    if p - 1 >= m - 2 or m > 10:
        return tuple(_welch(p))
    return next(q for q in permutations(range(m)) if _is_costas(q))


# c8r50's 8 tones hold a Costas array of 6 only, and tiled to PREAMBLE_S it
# matched one period off (18 of 24 symbols, sync 0.76, peak ratio 0.94): a
# search range ending before a burst's start locked a period early and read
# another header (24 of 223 fsk8r50 locks, scripts/cpm_supersede_study.py).
# This front is searched instead (scripts/c8r50_front_search.py, protocol
# 13's header words): each tone 3 times, every other placement and whole-bin
# shift at most 5 of 24 on itself, 6 with a header after it.
FRONTS = {"c8r50": (2, 1, 1, 3, 7, 7, 3, 6, 6, 4, 3, 6, 0, 5, 4, 0, 7, 0, 2, 1, 2, 5, 4, 5)}


def preamble_pattern(g: Grid) -> np.ndarray:
    """The front sync block, ~PREAMBLE_S: a Costas array, or FRONTS's."""
    return np.array(FRONTS.get(g.name, costas(g.m)))


def mid_block(g: Grid) -> np.ndarray:
    """The mid sync block, ~BLOCK_S: the largest Welch Costas array that
    fits, its tones spread over the band (scaling keeps the Costas property)."""
    target = max(4, round(BLOCK_S * g.rate))
    p = max(q for q in range(3, min(target, g.m) + 2) if _prime(q))
    c = _welch(p) * (g.m // (p - 1))
    if np.array_equal(c, costas(g.m)):
        # c8r50's front array until protocol 13 (tiled: FRONTS): a mid block
        # then matched half the front pattern (score 0.50) and whole-burst
        # searches locked on it. Time-reversed (still Costas): 0.17; kept,
        # and FRONTS's c8r50 front was searched against it (3 of 12 at most)
        c = c[::-1]
    return np.tile(c, max(1, round(target / len(c))))


# --- header --------------------------------------------------------------------------

def hdr_len(g: Grid) -> int:
    return max(6, round(HDR_S * g.rate))


def header_word(index: int, n_data: int, dup: bool) -> int:
    """16 bits as the OFDM header's: index (4) | n field (6) | CRC-6; index
    = mode index + 2 x duplicated control, n field = data codewords + 1."""
    from .modem import crc6

    v = ((index + 2 * dup) << 6) | n_data
    return (v << 6) | crc6(v)


@lru_cache(maxsize=None)
def header_symbols(grid: str, word: int) -> np.ndarray:
    """A word's header tones: pseudo-random per (grid, word). Decoding is
    ML over the grid's few valid words, so a random M-ary code is strong."""
    g = GRIDS[grid]
    return np.random.default_rng([g.m, int(g.rate * 100), word]).integers(0, g.m, hdr_len(g))


@lru_cache(maxsize=None)
def valid_words(grid: str) -> tuple:
    """(spec, n_data, dup, tones) of every word this grid's header carries."""
    return tuple((s, n, d, header_symbols(grid, header_word(s.index, n, d))) for s in grid_specs(grid)
                 for d in (False, True) for n in range(MAX_DATA + 1))


# --- layout --------------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    """Rows (symbol index from the burst's start) of a burst's parts."""
    n: int
    sync_rows: np.ndarray
    sync_tones: np.ndarray
    hdr_rows: tuple  # one array of rows per header copy
    data_rows: np.ndarray  # every codeword's symbols, in slot order
    front: int  # sync symbols in the first block (an early lock's)


def stream_symbols(grid: str, n_data: int, dup: bool) -> int:
    return (1 + dup) * CTL[grid].n_sym + n_data * DATA_N // GRIDS[grid].bits


@lru_cache(maxsize=None)
def layout(grid: str, n_sym: int) -> Layout:
    """Sync blocks, header copies and the codewords' `n_sym` symbols."""
    g = GRIDS[grid]
    H = hdr_len(g)
    D = max(1, round(SPACING_S * g.rate))
    segs = [min(D, n_sym - i) for i in range(0, max(n_sym, 1), D)]
    rows, tones_, hdr, data, r = [], [], [], [], 0
    for i, d in enumerate(segs):
        if i < HDR_COPIES:  # a sync block before each header copy, none after
            b = preamble_pattern(g) if i == 0 else mid_block(g)
            rows += range(r, r + len(b))
            tones_ += list(b)
            r += len(b)
            hdr.append(np.arange(r, r + H))
            r += H
        data += range(r, r + d)
        r += d
    while len(hdr) < HDR_COPIES:  # a short burst: its second header copy after one more block
        b = mid_block(g)
        rows += range(r, r + len(b))
        tones_ += list(b)
        r += len(b)
        hdr.append(np.arange(r, r + H))
        r += H
    return Layout(r, np.array(rows), np.array(tones_), tuple(hdr), np.array(data, int), len(preamble_pattern(g)))


def burst_seconds(spec, n_cw: int, dup: bool = False) -> float:
    """On-air length of a burst of n_cw slots (control included, as the ARQ
    counts them)."""
    g = GRIDS[spec.grid]
    n_data = max(0, n_cw - 1 - dup)
    return layout(spec.grid, stream_symbols(spec.grid, n_data, dup)).n * g.T / FS + 2 * RAMP_S


# --- transmit ------------------------------------------------------------------------

def _gray(m):
    return np.array([i ^ (i >> 1) for i in range(m)])


def to_tones(g: Grid, bits: np.ndarray) -> np.ndarray:
    idx = bits.reshape(-1, g.bits) @ (1 << np.arange(g.bits)[::-1])
    return _gray(g.m)[idx]


def tones(g: Grid, sym: np.ndarray, glide: float = 0.0) -> np.ndarray:
    """Tone indices -> constant-envelope audio (unit RMS), phase continuous;
    `glide`: TxFilter.glide."""
    T = g.T
    a = np.repeat(sym.astype(float), T)
    n = int(glide * T)
    if n > 1:
        w = np.hanning(n + 2)[1:-1]
        a = np.convolve(np.pad(a, n, mode="edge"), w / w.sum(), mode="same")[n:-n]
    x = np.sqrt(2) * np.cos(2 * np.pi * np.cumsum(g.f0 + a * g.rate) / FS)
    n_ramp = int(RAMP_S * FS)
    ramp = (1 - np.cos(np.pi * (np.arange(n_ramp) + 0.5) / n_ramp)) / 2
    x[:n_ramp] *= ramp
    x[-n_ramp:] *= ramp[::-1]
    return x


def bandpass(g: Grid, x: np.ndarray, cap: int = 0) -> np.ndarray:
    """The OFDM bands' TX filter over the tones +- the cap's margin, clip and filter (TX_FILTERS)."""
    from .waveform.dsp import tx_condition

    f = tx_filter(g.name, cap)
    return tx_condition(x, g.clip_db, f.overshoot, bandpass=(g.f0 - f.bp, g.f0 + (g.m - 1) * g.rate + f.bp))


def modulate(spec: CpmSpec, coded: list, dup: bool, cap: int = 0) -> np.ndarray:
    """Every slot's coded bits (data2g.codes.encode's, mapping order; the
    control codeword first, twice if dup) -> audio, filtered for the
    bandwidth cap `cap` (TX_FILTERS)."""
    g = GRIDS[spec.grid]
    n_data = len(coded) - 1 - dup
    stream = np.concatenate([to_tones(g, c) for c in coded])
    lay = layout(spec.grid, len(stream))
    sym = np.empty(lay.n, int)
    sym[lay.sync_rows] = lay.sync_tones
    h = header_symbols(spec.grid, header_word(spec.index, n_data, dup))
    for rows in lay.hdr_rows:
        sym[rows] = h
    sym[lay.data_rows] = stream
    return bandpass(g, tones(g, sym, tx_filter(g.name, cap).glide), cap)


# --- receive -------------------------------------------------------------------------

def _sq(Z: np.ndarray) -> np.ndarray:
    """|Z|^2 as re^2 + im^2 (within an ulp of np.abs(Z) ** 2, without its hypot)."""
    return Z.real ** 2 + Z.imag ** 2


def _energies(g: Grid, x: np.ndarray, start: int, n_sym: int, cfo: float, extra: int = 0) -> np.ndarray:
    """(n_sym, m + 2 extra) tone energies of symbols from `start`, CFO removed."""
    T = g.T
    seg = x[max(0, start):start + n_sym * T]
    if start < 0:
        seg = np.concatenate([np.zeros(-start), seg])
    if len(seg) < n_sym * T:
        seg = np.pad(seg, (0, n_sym * T - len(seg)))
    t = (start + np.arange(len(seg))) / FS
    z = seg * np.exp(-2j * np.pi * (g.f0 - extra * g.rate + cfo) * t)  # lowest bin at DC
    Z = np.fft.fft(z.reshape(n_sym, T), axis=1)
    return _sq(Z[:, :g.m + 2 * extra])


def _shares(E: np.ndarray) -> np.ndarray:
    """Each symbol's energies as shares of its total: raw energy let a strong
    stretch of data outscore a sync block in a fade (MPG, 20 dB: locked 4 s
    late); a share can't be inflated."""
    return E / (E.sum(axis=1, keepdims=True) + 1e-30)


def coarse_rows(g: Grid, x: np.ndarray, extra: int) -> np.ndarray:
    """detect's coarse energies of x's whole symbols: (n, 4, m + 2 extra),
    [k, q] symbol k's tones from extra bins below the grid at CFO fraction
    q / 4, as shares. One rfft per symbol zero-padded to 4T, bins R / 4
    apart: a tone at f0 + k R is bin 4 (f0 / R + k) (f0 a whole number of
    bins), so fraction q / 4 is that bin + q, what mixing by it and a
    T-point FFT would give (|Z|^2 drops the mix's phase)."""
    T = g.T
    n = len(x) // T
    if n == 0:
        return np.zeros((0, 4, g.m + 2 * extra))
    F = np.fft.rfft(x[:n * T].reshape(n, T), n=4 * T, axis=1)
    P = F.real ** 2 + F.imag ** 2
    bins = 4 * (round(g.f0 / g.rate) - extra + np.arange(g.m + 2 * extra))
    return np.stack([_shares(P[:, bins + q]) for q in range(4)], axis=1)


class CoarseTrack:
    """A streaming receiver's coarse_rows of grid g as audio arrives, on the
    stream's T/4 timing grid: each hop's searches (find, find_mid) read the
    rows instead of transforming their windows again (each window holds
    2-4 hops of audio already transformed). Detection as detect's on x,
    its placements on the stream's grid instead of the window's."""

    def __init__(self, g: Grid, reach_hz: float = 150.0):
        self.g, self.extra = g, int(np.ceil(reach_hz / g.rate))
        self.s: list[int | None] = [None] * 4  # per phase: the stream sample of its first row
        self.E = [np.zeros((0, 4, g.m + 2 * self.extra)) for _ in range(4)]

    def feed(self, buf: np.ndarray, off: int):
        """`buf` holds stream samples off.. (a receiver's buffer): rows for its
        new whole symbols; rows of samples it no longer holds dropped."""
        T, end = self.g.T, off + len(buf)
        for ph in range(4):
            s, E = self.s[ph], self.E[ph]
            if s is None or s < off:
                first = off + (ph * T // 4 - off) % T
                E = E[(first - s) // T:] if s is not None else E  # the rows still in buf
                s = first
            nxt = s + len(E) * T
            if end - nxt >= T:
                E = np.concatenate([E, coarse_rows(self.g, buf[nxt - off:], self.extra)])
            self.s[ph], self.E[ph] = s, E

    def rows(self, a: int, b: int) -> list:
        """detect's per-phase (first symbol's sample from a, rows) of the
        whole symbols in stream samples [a, b)."""
        T, out = self.g.T, []
        for ph in range(4):
            s, E = self.s[ph], self.E[ph]
            if s is None:
                continue
            k0 = max(0, -((s - a) // T))  # the first row starting at or after a
            k1 = min(len(E), (b - s) // T)  # rows ending by b
            out.append((s + k0 * T - a, E[k0:max(k0, k1)]))
        return out


def detect(g: Grid, x: np.ndarray, reach_hz: float = 150.0, fine=True, front_only: bool = False, n_sym: int = 0,
           floor: float = -1.0, mid: bool = False, track: "CoarseTrack | None" = None, x0: int = 0):
    """-> (score, burst start sample, cfo Hz): the sync pattern's best
    placement. Coarse: symbol energies at T/4 hops and quarter-bin CFO
    steps, as shares, summed over the sync symbols of the shortest burst
    (`n_sym` stream symbols; default a control-only burst's) at every start
    and whole-bin shift. Score: the mean share on the pattern (0..1).
    `front_only`: the first block alone (an early lock); `mid`: the mid
    block alone, its first sample for the burst's start (find_mid);
    `floor`: no fine stage under this score (the search's CPU on noise).
    `track`: its rows for the coarse stage instead (x: stream samples from
    `x0`, rows on the stream's timing grid)."""
    lay = layout(g.name, n_sym or stream_symbols(g.name, 0, False))
    rows, tones_ = lay.sync_rows, lay.sync_tones
    if front_only:
        rows, tones_ = rows[:lay.front], tones_[:lay.front]
    if mid:
        tones_ = mid_block(g)
        rows = np.arange(len(tones_))
    span = rows[-1] + 1
    T = g.T
    extra = int(np.ceil(reach_hz / g.rate))
    best = (-1.0, 0, 0.0)
    # per timing phase: (its first symbol's sample in x, coarse_rows of its symbols)
    if track is None:
        C = [(ph * T // 4, coarse_rows(g, x[ph * T // 4:], extra)) for ph in range(4)]
    else:
        assert track.extra == extra
        C = track.rows(x0, x0 + len(x))
    for q, frac in enumerate((0.0, 0.25, 0.5, 0.75)):
        for off, E4 in C:
            n = len(E4)
            if n < span:
                continue
            E = E4[:, q]
            # every start and whole-bin shift at once: S[dk, j] = sum over the
            # pattern's rows r of E[r + j, tone_r + extra + dk] (a loop over
            # dk and r took ~1/3 of a listening host's CPU on its tiny arrays)
            J = n - span + 1
            Er = E[np.asarray(rows)[:, None] + np.arange(J)]  # (rows, J, bins)
            cols = np.asarray(tones_)[:, None] + extra + np.arange(-extra, extra + 1)  # (rows, dk)
            S = Er[np.arange(len(rows))[:, None], :, cols].sum(axis=0)  # (dk, J)
            k = int(np.argmax(S))
            if S.flat[k] / len(rows) > best[0]:
                dk, j = divmod(k, J)
                best = (float(S.flat[k]) / len(rows), off + j * T, (dk - extra) * g.rate + frac * g.rate)
    score, s0, cfo = best
    if fine and score >= floor:  # timing to T/32, CFO to R/16
        cand = []
        for dt in range(-T // 8, T // 8 + 1, max(1, T // 32)):
            for df in np.arange(-0.125, 0.126, 0.0625) * g.rate:
                E = _energies(g, x, s0 + dt, span, cfo + df)
                cand.append((E[rows, tones_].sum() / E[rows].sum(), s0 + dt, cfo + df))
        _, s0, cfo = max(cand)
    return score, s0, cfo


# One N0 for every tone lets a steady carrier in the passband (QRM: a
# broadcast carrier, a birdie, another mode's idle tone) win its tone in
# every symbol: the square law reads it as signal, and coh_metric, which
# looks for a constant phasor, more so. A tone carries signal in ~1/m of a
# codeword's rows, so its median energy over the rows is its own floor.
# tone_floors() gives it to tone_metric and coh_metric once it stands
# FLOOR_SIGMAS standard errors (a median of n exponentials: 1.44 / sqrt(n),
# relative) over the tones' median floor; with no such tone both are as
# before. On noise alone a tone passes in under 1 of 1000 calls at 6 (8, 16
# and 32 tones; 22, 90 and 240 rows: a header read, a control, a data
# codeword); at 5, up to 6 of 1000.
FLOOR_SIGMAS = 6.0


def tone_floors(E: np.ndarray, n0: float) -> np.ndarray | None:
    """(n, m) tone energies and the common N0 -> (m,) per-tone noise floors,
    or None where no tone's median stands out (FLOOR_SIGMAS): N0, times the
    tone's median over the tones' median where it does."""
    med = np.median(E, axis=0)
    ref = np.median(med)
    out = med > ref * (1 + FLOOR_SIGMAS * 1.44 / np.sqrt(len(E)))
    if not out.any():
        return None
    return np.where(out, n0 * (med / ref), n0)


def _tone_gain(E: np.ndarray) -> tuple:
    """tone_metric's (gain, offset): E * gain - offset. One N0: the square
    law's Es / (N0 (N0 + Es)) and None; per-tone floors (tone_floors): (m,)
    gains Es / (N_m (N_m + Es)) and offsets ln(1 + Es / N_m), which are a
    per-symbol constant only when the floors are equal."""
    top = E.max(axis=1)
    n0 = np.median(np.sort(E, axis=1)[:, :-1]) / np.log(2)  # exponential: median = N0 ln 2
    n0 = max(n0, 1e-12 * np.mean(top) + 1e-300)  # noise-free tests
    nb = tone_floors(E, n0)
    if nb is not None:
        Ew = E * (n0 / nb)  # whitened, for Es and the spill: the carrier's tone is not the strongest by itself
        es = max(np.mean(Ew.max(axis=1)) - n0, 1e-3 * n0)
        nb = nb + spill(Ew, n0, es) * es
        return es / (nb + es) / nb, np.log1p(es / nb)
    es = max(np.mean(top) - n0, 1e-3 * n0)
    n0 += spill(E, n0, es) * es
    # es / (n0 (n0 + es)) without its underflow: a codeword past the audio's
    # end (zero-padded: a header claiming more than arrived) reads as erasures
    return es / (n0 + es) / n0, None


def tone_metric(g: Grid, E: np.ndarray) -> np.ndarray:
    """(n_sym, m) tone energies of one codeword -> (n_sym, m) log P(E | tone
    sent) up to a per-symbol constant. Noncoherent square-law metric, exact
    for Rayleigh fading: E_m g, g = Es / (N0 (N0 + Es)), Es and N0 the
    codeword's, N0 plus the Doppler spill (spill()); a tone whose own floor
    stands out of N0 (tone_floors: a carrier) is scored against that floor."""
    gain, off = _tone_gain(E)
    return E * gain if off is None else E * gain - off


def spill(E: np.ndarray, n0: float, es: float) -> float:
    """The share of Es each neighbour of a symbol's strongest tone holds over
    N0, 0..1: fast fading (30 Hz on c16r25's 25 Hz tones) spreads a tone into
    its neighbours, which the median N0 never sees (2 of 15 bins), so the
    metric's confidence grew with SNR while the right tone won no more rows,
    and the lock's header score fell under LOCK_LLR (#244: no lock 20% at
    -5 dB, 60% at +20 dB, hilat). Counting it as noise caps the confidence
    by the spread. ~0 on AWGN and slow fading."""
    a = E.argmax(axis=1)
    P = np.pad(E, ((0, 0), (1, 1)), constant_values=n0)  # an edge tone's missing neighbour: noise
    r = np.arange(len(E))
    return float(np.clip(np.mean(P[r, a] + P[r, a + 2] - 2 * n0) / 2 / es, 0.0, 1.0))


def llrs(g: Grid, E: np.ndarray) -> np.ndarray:
    """(n_sym, m) tone energies of one codeword -> (n_sym * bits,) coded-bit
    LLRs (positive: 0) in mapping order (data2g.codes.combine / decode_many's),
    marginals of tone_metric: the control codeword's."""
    return bit_llrs(g, tone_metric(g, E))


def bit_llrs(g: Grid, metric: np.ndarray) -> np.ndarray:
    """(n_sym, m) tone log-likelihoods -> llrs's coded-bit LLRs (their marginals)."""
    label = np.argsort(_gray(g.m))  # tone -> the bit group it carries
    out = np.empty((len(metric), g.bits))
    for b in range(g.bits):
        bit = (label >> (g.bits - 1 - b)) & 1
        out[:, b] = np.logaddexp.reduce(metric[:, bit == 0], axis=1) - np.logaddexp.reduce(metric[:, bit == 1], axis=1)
    return out.reshape(-1)


def symbol_ll(g: Grid, E: np.ndarray) -> np.ndarray:
    """(n_sym, m) tone energies of one codeword -> (n_sym * m,) symbol
    log-likelihoods in mapping order, symbol-major, value v's on tone
    gray(v) (to_tones): a GF(M) codeword's soft values (data2g.codes)."""
    return tone_metric(g, E)[:, _gray(g.m)].reshape(-1)




@lru_cache(maxsize=None)
def header_rows(grid: str, copies: int) -> tuple:
    """Per valid word, the rows of its first `copies` header copies: the
    second's depend on the word's length when the burst is short."""
    return tuple(np.concatenate(layout(grid, stream_symbols(grid, n, d)).hdr_rows[:copies])
                 for _, n, d, _ in valid_words(grid))


def tone_llr(g: Grid, E: np.ndarray) -> np.ndarray:
    """(rows, m) energies -> per row and tone, the log-likelihood ratio of
    that tone sent against a uniformly random one: llrs's square-law metric,
    es and N0 from these rows. A front or header row scores ~0 on noise, more
    on its tone as SNR grows, and strongly negative where another tone is
    strong (a burst's data: what PEAK_RATIO's share test was for)."""
    G = tone_metric(g, E)
    L = G - np.logaddexp.reduce(G, axis=1, keepdims=True) + np.log(g.m)
    return np.logaddexp(np.log1p(-LLR_EPS) + L, np.log(LLR_EPS))


def _segments(L: np.ndarray, rows: np.ndarray, t: np.ndarray, H: int) -> float:
    """tone_llr over header copies (rows: H each, tones t), each copy's sum
    floored at 0: a faded copy is no evidence either way."""
    return sum(max(float(L[rows[i:i + H], t].sum()), 0.0) for i in range(0, len(rows), H))


def read_header(g: Grid, x: np.ndarray, s0: int, cfo: float, copies: int = HDR_COPIES):
    """-> (spec, n_data, dup, score, runner-up score, front): ML over the
    grid's valid words of tone_llr summed over the front block and the first
    `copies` header copies (1: all an early lock has), each word's at its own
    rows, each block's sum floored at 0 (_segments); `front`: the front
    block's sum, unfloored."""
    rows = header_rows(g.name, copies)
    L = tone_llr(g, _energies(g, x, s0, max(r[-1] for r in rows) + 1, cfo))
    f = preamble_pattern(g)
    front = float(L[np.arange(len(f)), f].sum())
    words = valid_words(g.name)
    H = hdr_len(g)
    scores = max(front, 0.0) + np.array([_segments(L, r, t, H) for r, (*_, t) in zip(rows, words)])
    order = np.argsort(scores)[::-1]
    spec, n, d, _ = words[order[0]]
    return spec, n, d, float(scores[order[0]]), float(scores[order[1]]), front


def soft(g: Grid, spec: CpmSpec, x: np.ndarray, s0: int, cfo: float, n_data: int, dup: bool):
    """-> (per-slot soft bits in mapping order, the stream's tone energies)."""
    n_sym = stream_symbols(g.name, n_data, dup)
    lay = layout(g.name, n_sym)
    E = _energies(g, x, s0, lay.n, cfo)[lay.data_rows]
    sizes = [CTL[g.name].n_sym] * (1 + dup) + [DATA_N // g.bits] * n_data
    out, i = [], 0
    for j, n in enumerate(sizes):
        out.append((llrs if j < 1 + dup else symbol_ll)(g, E[i:i + n]))
        i += n
    return out, E


REFINE_SYMBOLS = 300  # refine's look: the burst's first symbols (the full burst gained nothing more, 2026-10-26)


def refine(g: Grid, x: np.ndarray, s0: int, cfo: float, spec: CpmSpec, n_data: int, dup: bool) -> tuple:
    """A lock's (start, cfo) -> the noncoherent likelihood's best over the
    burst's first REFINE_SYMBOLS symbols: known tones (sync blocks, header
    copies of the word read) by llrs's metric, the rest (control, data) by
    its log-sum over tones; 5 x 5 grids of T/16 x R/16, then T/64 x R/64.
    detect's fine stage reads the front alone, and its residual error cost
    the data 0.5 dB at the AWGN floor (c16r25; 2 Hz or T/32 off: 0.2 dB)."""
    lay = layout(g.name, stream_symbols(g.name, n_data, dup))
    h = header_symbols(g.name, header_word(spec.index, n_data, dup))
    n = min(lay.n, REFINE_SYMBOLS)
    known = np.full(n, -1)
    for rows, t in [(lay.sync_rows, lay.sync_tones)] + [(r, h) for r in lay.hdr_rows]:
        keep = rows < n
        known[rows[keep]] = np.asarray(t)[keep]
    kr, ur = np.flatnonzero(known >= 0), np.flatnonzero(known < 0)
    E0 = _energies(g, x, s0, n, cfo)
    tot = E0.sum()
    # the metric's weight fixed at the lock's placement (each candidate's own: 0.2 dB worse, c8r50 AWGN)
    gain, off = _tone_gain(E0) if tot > 0 else (0.0, None)  # per-tone floors: the tones' own gains and offsets
    w = float((E0 * gain).sum() / tot) if tot > 0 and off is None else 0.0
    T, N = g.T, n * g.T

    def ll(z):  # the likelihood of n symbols of mixed samples, as _energies(...) * w would score them
        G = _sq(np.fft.fft(z.reshape(n, T), axis=1)[:, :g.m])
        G = G * w if off is None else G * gain - off
        return G[kr, known[kr]].sum() + np.logaddexp.reduce(G[ur], axis=1).sum()
    best, bv = (s0, cfo), None
    for st, sf in ((max(1, g.T // 16), g.rate / 16), (max(1, g.T // 64), g.rate / 64)):
        s, f = best
        # _energies mixes by absolute sample, so one mix per CFO serves every start. The stage mixes once at
        # its centre CFO; each other CFO is that times a rotation, exp(-2 pi i sf t / FS) to the power j,
        # whose phase repeats every FS / sf samples (native cpm::refine, the same arithmetic)
        a0, M = s - 2 * st, 4 * st + N
        seg = np.zeros(M)
        lo, hi = max(0, a0), min(len(x), a0 + M)
        if hi > lo:
            seg[lo - a0:hi - a0] = x[lo:hi]
        base = seg * np.exp(-2j * np.pi * (g.f0 + f) * ((a0 + np.arange(M)) / FS))
        P = FS / sf
        Lt = int(P) if P == np.floor(P) and P <= M else M
        tab = np.exp(1j * (-2 * np.pi * (sf * np.arange(Lt) / FS)))
        r1 = np.exp(1j * (-2 * np.pi * np.fmod(sf * a0 / FS, 1.0))) * tab[np.arange(M) % Lt]
        r2 = r1 * r1
        z = {-2: base * np.conj(r2), -1: base * np.conj(r1), 0: base, 1: base * r1, 2: base * r2}
        cands = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                # the second stage's centre is the first stage's best, already scored
                v = bv if (bv is not None and i == 0 and j == 0) else ll(z[j][(i + 2) * st:(i + 2) * st + N])
                cands.append((v, (s + i * st, f + j * sf)))
        bv, best = max(cands, key=lambda c: c[0])  # the first best, as max(..., key=ll) took it
    return best


# Doppler spread from the winning tone's energy (spread()): autocovariance at
# lags 1..SPREAD_LAGS. The lag-1 correlation alone read 6-19 Hz on 0.5 Hz
# fading (each symbol's energy jitters with the modulation and noise: white,
# so it sits at lag 0 only), which the old 0-3 Hz clip pinned at 3.0 on every
# channel. Measured (2026-11-18, two-path 2 ms, -15..+20 dB): 0.5-4 Hz within
# about 20% on every grid; c8r50 and c32r62 to 10 Hz; past what the grid
# resolves, no correlation survives lag 1 and it reads SPREAD_MAX_HZ, if the
# energy varies at all: AWGN has no correlation past lag 0 either. So with
# none, the excess variance decides: var(top) less the noise's (2 n0 s +
# n0^2, chi-square) over s^2 reads 0.00 on AWGN and 0.28-1.06 on fading of
# 0.1-30 Hz at -10 dB and up (c32r62: -5); under SPREAD_EXCESS_MIN it is
# no fading seen, 0.0. (At -15 dB the noise term swamps it and fast fading
# reads 0.0 too: only the CPM floor works there either way.)
# Native: cpm::measure.
SPREAD_LAGS = 8
SPREAD_MIN_SYMS = 16  # shorter: unmeasured (UNMEASURED_SPREAD_HZ: the shifter carries its last)
UNMEASURED_SPREAD_HZ = 2.0  # equalizer.DEFAULT_SPREAD_HZ, policy.UNMEASURED_SPREAD_HZ
SPREAD_KEEP = 0.05  # lags used: up to the first under this fraction of lag 1's
SPREAD_MAX_HZ = 12.0
SPREAD_EXCESS_MIN = 0.2
SPREAD_SIG = 2.5  # the fit needs lag 1's correlation over SPREAD_SIG / sqrt(n): white energies pass by chance otherwise
# and an excess variance of at least this: on AWGN at high SNR the noise's
# variance is so small that the burst's own energy pattern passed SPREAD_SIG
# (c8r50 +20 dB read 2.6 Hz)
SPREAD_EXCESS_FIT = 0.05
# With multipath every tone has its own gain and the data changes tone each
# symbol, so on slow selective fading the energies look white too: no fit, and
# the excess variance read SPREAD_MAX_HZ (control-only fsk16r25, 200 seeds:
# 2 ms / 0.1 Hz 152, 2 ms / 1 Hz 29; flat 1 Hz none). So before that reading,
# the correlation over the pairs whose winning tones are at most one apart
# and that are SPREAD_NEAR_S apart in time: over SPREAD_NEAR_Z / sqrt(pairs),
# the fading is slow and the spread unmeasured. The pairs start 80 ms apart,
# not at the next symbol: 10 Hz fading is still correlated over a symbol or
# two (lags 1-3 read 25 of fsk16r25's 188 right readings at 7 ms / 10 Hz as
# slow, and 30 of fsk8r50's 88), and by 80 ms it is not, where 1 Hz fading
# holds to 160 ms and past. Of 200, fsk16r25 / fsk8r50, SPREAD_MAX_HZ
# readings before -> after: 2 ms / 0.1 Hz 152 -> 4 / 105 -> 0; 2 ms / 1 Hz
# +2 dB 29 -> 8 / 4 -> 0; 7 ms / 1 Hz -8 dB 68 -> 32 / 41 -> 28; and the
# right ones, 7 ms / 10 Hz 188 -> 180 / 88 -> 88; 2 ms / 10 Hz 182 -> 179 /
# 65 -> 62; 7 ms / 30 Hz 198 -> 198 / 194 -> 188 (the test's own chance
# level at SPREAD_NEAR_Z: about 2%). The near pairs against the far ones at
# the same lags did worse both ways (7 ms / 1 Hz 68 -> 60, 30 Hz 198 -> 182):
# with multipath the far tones fade apart whatever the rate.
SPREAD_NEAR_S = (0.08, 0.16)
SPREAD_NEAR_Z = 2.0


def _near_lags(g: Grid) -> range:
    """SPREAD_NEAR_S in symbols (c16r25: 2-4, c8r50: 4-8, c32r62: 5-10)."""
    return range(max(1, int(np.ceil(SPREAD_NEAR_S[0] * g.rate - 1e-9))), int(np.floor(SPREAD_NEAR_S[1] * g.rate + 1e-9)) + 1)


def _slow_by_near_tones(g: Grid, t: np.ndarray, v: float, tones: np.ndarray) -> bool:
    num, n = 0.0, 0
    for lag in _near_lags(g):
        if lag >= len(t):
            break
        ok = np.abs(tones[lag:] - tones[:-lag]) <= 1
        num += float(np.sum(t[lag:][ok] * t[:-lag][ok]))
        n += int(ok.sum())
    return n > 0 and v > 0 and num / n / v * np.sqrt(n) > SPREAD_NEAR_Z


def spread(g: Grid, top: np.ndarray, n0: float, tones: np.ndarray | None = None) -> float:
    """Doppler spread (2 sigma, Hz) from per-symbol energies (noise n0 per
    bin): a least-squares fit of log c(k) = a - b k^2 (Gaussian Doppler:
    c(k) = A exp(-(2 pi sigma k T)^2)) over the lags 1.. whose
    autocovariance c(k) holds up; with none, too fast or no fading, by the
    excess variance, unless the symbols on the same or adjacent tones
    (`tones`: each symbol's winning tone) say slow: then unmeasured."""
    n = len(top)
    if n < SPREAD_MIN_SYMS:
        return UNMEASURED_SPREAD_HZ
    mean = np.mean(top)
    t = top - mean
    c = np.array([np.mean(t[k:] * t[:n - k]) for k in range(1, SPREAD_LAGS + 1)])
    v = np.mean(t * t)
    s = max(mean - n0, 1e-12)
    excess = (v - (2 * n0 * s + n0 * n0)) / (s * s)
    m = 0
    while (excess >= SPREAD_EXCESS_FIT and c[0] > SPREAD_SIG / np.sqrt(n) * v and m < SPREAD_LAGS
           and c[m] > SPREAD_KEEP * c[0]):
        m += 1
    if m < 2:
        if excess < SPREAD_EXCESS_MIN:
            return 0.0
        return UNMEASURED_SPREAD_HZ if tones is not None and _slow_by_near_tones(g, t, v, tones) else SPREAD_MAX_HZ
    x = np.arange(1, m + 1, dtype=np.float64) ** 2
    y = np.log(c[:m])
    mx, my = np.mean(x), np.mean(y)
    b = -np.sum((x - mx) * (y - my)) / np.sum((x - mx) ** 2)
    return float(min(np.sqrt(max(b, 0.0)) * g.rate / np.pi, SPREAD_MAX_HZ))


def measure(g: Grid, E: np.ndarray, n_sym: int) -> dict:
    """The gear shifter's measurements, as arq.phy.measure gives them for
    OFDM: SNR in the 2500 Hz reference, Doppler spread (spread(), from the
    winning tone's energy over time), and `snr`, per symbol, from the
    winning tone's energy over the noise bins (phy.measure makes the
    effective-MI features of it). With a carrier (tone_floors), snr_est
    reads the tones scaled down to the quietest quarter's floor, Es back up
    by the share kept: fsk16r25 -8 dB reads +0.1 dB at a carrier 12 dB in
    50 Hz, -0.5 at 18, -3 at 24, where the carrier read as signal read
    +2.3, +5.8, +8. Native cpm::measure."""
    from .config import FRAME_SAMPLES, SNR_REF_BW_HZ

    n0 = max(np.median(np.sort(E, axis=1)[:, :-1]) / np.log(2), 1e-300)
    top = E.max(axis=1)
    snr = np.maximum(top / n0 - 1, 1e-6)
    es_n0 = max(float(np.mean(top) - n0), 1e-12) / n0
    if tone_floors(E, n0) is not None:
        # a carrier: for snr_est only, its tone and the ones it leaks into (under
        # tone_floors' bar, but they raise the tones' median) each scaled to the
        # quietest quarter's floor; the signal, ~1/m of every tone's rows, keeps
        # `kept` of its energy, so Es is scaled back up by that. The per-symbol
        # snr and the spread stay as read: scaled rows looked like fading and the
        # gear shifter left c32r62 (-37%, awgn -12, carrier 12 dB)
        med = np.median(E, axis=0)
        q = np.quantile(med, 0.25)
        r = q / np.maximum(med, q)
        Ew = E * r
        n0w = max(np.median(np.sort(Ew, axis=1)[:, :-1]) / np.log(2), 1e-300)
        es_n0 = max(float(np.mean(Ew.max(axis=1)) - n0w) / float(np.mean(r)), 1e-12) / n0w
    return dict(snr_est=10 * np.log10(es_n0 * g.rate / SNR_REF_BW_HZ), spread_est=spread(g, top, n0, E.argmax(axis=1)),
                delay_est_ms=0.0, headroom=0.0, frames=max(1.0, n_sym * g.T / FRAME_SAMPLES), snr=snr)


def _peak_ratio(g: Grid, x: np.ndarray, s0: int, cfo: float, mid: bool = False) -> float:
    """The front pattern's (`mid`: the mid block's) share over each symbol's
    strongest share there: ~1 on a front at any SNR (0.82 and up, -10 to 25
    dB), low where the pattern met a burst's data by chance (on a clean
    signal each symbol is one tone, and 5 of 24 on the pattern cleared the
    sync threshold)."""
    f = mid_block(g) if mid else preamble_pattern(g)
    E = _shares(_energies(g, x, s0, len(f), cfo))
    return float(E[np.arange(len(f)), f].mean() / E.max(axis=1).mean())


def find(g: Grid, x: np.ndarray, threshold: float | None = None, reach_hz: float = 150.0, front_only: bool = True,
         lo: int = 0, hi: int | None = None, wait: bool = False, track: CoarseTrack | None = None, x0: int = 0):
    """-> early lock {spec, n_data, dup, start, end, score, header score} of
    a burst of grid g in x, or None: the sync pattern (front block alone by
    default) scoring over `threshold` (default SYNC_THRESHOLD; given: no
    lock floor either), then the front and its first header copy
    (read_header: LOCK_LLR[0], HDR_MARGIN[0]), or under that, a front over
    WAIT_LLR, both copies (late_lock) once x holds the second. `wait`: before
    then, a candidate {wait, start, cfo, score, header_end: where x must
    reach} for late_lock instead of None.
    `end`: one past its last sample, in x's samples (it may run past x).
    `lo`, `hi`: only starts in [lo, hi) are searched (a streaming receiver
    searches each start once: detect's cost is the audio it is given).
    `track`, `x0`: detect's (x the stream from sample x0)."""
    floor = SYNC_THRESHOLD[g.name] if threshold is None else threshold
    if lo or hi is not None:
        lay0 = layout(g.name, stream_symbols(g.name, 0, False))
        span = ((lay0.sync_rows[lay0.front - 1] if front_only else lay0.sync_rows[-1]) + 2) * g.T
        a = max(0, lo - g.T)
        b = len(x) if hi is None else min(len(x), hi + span + g.T)
        score, s0, cfo = detect(g, x[a:b], reach_hz=reach_hz, front_only=front_only, floor=floor, track=track,
                                x0=x0 + a)
        s0 += a
        if not lo <= s0 < (len(x) if hi is None else hi):
            return None
    else:
        score, s0, cfo = detect(g, x, reach_hz=reach_hz, front_only=front_only, floor=floor, track=track, x0=x0)
    if score < (SYNC_THRESHOLD[g.name] if threshold is None else threshold):
        return None
    if s0 + hdr_span(g.name, 1) > len(x):
        return None  # its header copy is still arriving
    if s0 < 0:
        return None  # fine timing moved a lock at x's first sample before it: its front is cut off
    if threshold is not None:
        return _lock(g, s0, cfo, score, read_header(g, x, s0, cfo, copies=1), 1)
    h, ok = _lock_header(g, x, s0, cfo, 1)
    if ok:
        return _lock(g, s0, cfo, score, h, 1)
    if h[5] < WAIT_LLR[g.name]:
        return None  # noise, or a burst's data or a later sync block, its front missed
    # a front, its first header copy faded: both copies, once the second is in
    cand = dict(wait=True, band=g.name, start=s0, cfo=cfo, score=score, header_end=s0 + hdr_span(g.name, 2),
                family="cpm")
    if cand["header_end"] <= len(x):
        return late_lock(g, x, cand)
    return cand if wait else None


def late_lock(g: Grid, x: np.ndarray, cand: dict):
    """find's candidate (x holding its header_end) -> its lock from the front
    and both header copies (LOCK_LLR), or None."""
    h, ok = _lock_header(g, x, cand["start"], cand["cfo"], HDR_COPIES)
    if not ok:
        return None
    return _lock(g, cand["start"], cand["cfo"], cand["score"], h, HDR_COPIES)


def find_mid(g: Grid, x: np.ndarray, lo: int = 0, hi: int | None = None, track: CoarseTrack | None = None,
             x0: int = 0):
    """-> find's lock of a burst whose front was missed, from its mid block
    (the sync block before header copy 2) and both header copies, or None.
    Block 1's start in [lo, hi); each word puts it at its own row (a short
    burst's after its data), so each is read at the start it implies, and
    a start before x is no reading. `track`, `x0`: detect's."""
    T, nb = g.T, len(mid_block(g))
    a = max(0, lo - T)
    b = len(x) if hi is None else min(len(x), hi + (nb + 1) * T)
    score, p, cfo = detect(g, x[a:b], mid=True, floor=MID_THRESHOLD[g.name], track=track, x0=x0 + a)
    p += a
    if score < MID_THRESHOLD[g.name] or not lo <= p < (len(x) if hi is None else hi):
        return None
    if p + (nb + hdr_len(g)) * T > len(x):
        return None  # its second header copy is still arriving
    h = read_mid(g, x, p, cfo)
    if h is None or h[3] < MID_LLR[g.name] or h[3] - h[4] < HDR_MARGIN[1]:
        return None
    return _lock(g, h[5], cfo, score, h, HDR_COPIES)


def read_mid(g: Grid, x: np.ndarray, p: int, cfo: float):
    """A mid block at p -> (spec, n_data, dup, score, runner-up, start) or
    None: ML over the valid words of tone_llr summed over every known row
    (the mid block; front and both header copies as read_header floors
    them), each word read at the start its mid block row implies (a start
    before x is no reading)."""
    T = g.T
    words, rows = valid_words(g.name), header_rows(g.name, HDR_COPIES)
    n = max(r[-1] for r in rows) + 1
    f, mb = preamble_pattern(g), mid_block(g)
    L, scores = {}, np.full(len(words), -np.inf)
    for k, ((*_, t), r, b1) in enumerate(zip(words, rows, mid_rows(g.name))):
        s = p - b1 * T
        if s >= 0:
            if s not in L:
                L[s] = tone_llr(g, _energies(g, x, s, n, cfo))
            Ls = L[s]
            scores[k] = (max(float(Ls[np.arange(len(f)), f].sum()), 0.0) + float(Ls[b1 + np.arange(len(mb)), mb].sum())
                         + _segments(Ls, r, t, hdr_len(g)))
    order = np.argsort(scores)[::-1]
    if not np.isfinite(scores[order[0]]):
        return None
    spec, nd, d, _ = words[order[0]]
    return spec, nd, d, float(scores[order[0]]), float(scores[order[1]]), p - mid_rows(g.name)[order[0]] * T


# Expected-slot lock (docs/arq.md §3a): a station waiting for its peer's reply
# knows when it should start and what CFO the peer's rig has, so a burst
# that no blind path locked is looked for at those starts only (PRIME_STEP)
# and CFOs (PRIME_CFO, fractions of the tone spacing), with every row the
# header word fixes (front, mid block, both copies; each block floored at 0,
# as _segments). Scored on noise, OFDM bursts and the other grids' bursts in
# a +-0.1 s window, 27000 windows: largest 15.1 (scripts/primed_study.py,
# runs/primed-rx/plock_fa.csv); real bursts the blind paths lost read 20-60.
PRIME_LLR = 16.0
PRIME_STEP = 4  # starts every T / PRIME_STEP
PRIME_CFO = (-1 / 16, 0.0, 1 / 16)


def prime_lock(g: Grid, x: np.ndarray, lo: int, hi: int, cfo: float):
    """A burst of grid g starting in [lo, hi] (x's samples), its CFO near
    `cfo`, any valid word: find's lock dict, or None under PRIME_LLR. x must
    hold the second header copy of a start at hi."""
    words = valid_words(g.name)
    rows = header_rows(g.name, HDR_COPIES)
    f, mb, H = preamble_pattern(g), mid_block(g), hdr_len(g)
    n = max(r[-1] for r in rows) + 1
    # a start a whole symbol later reads the same symbols one row on: per
    # phase (a fraction of T) and CFO, one run of tone_llr over the window
    # and the rows after it (its scale from all of them), sliced per start
    phases = range(0, g.T, max(1, g.T // PRIME_STEP))
    n_pos = (hi - lo) // g.T + 1
    LL = {(p, df): tone_llr(g, _energies(g, x, lo + p, n + n_pos, cfo + df * g.rate)) for p in phases for df in PRIME_CFO}
    best = (-np.inf, 0, 0.0, 0)  # score, start, cfo, word
    for s0, p, m in ((lo + m * g.T + p, p, m) for m in range(n_pos) for p in phases):
        if s0 > hi:
            continue
        for df in PRIME_CFO:
            L = LL[(p, df)][m:m + n]
            front = max(float(L[np.arange(len(f)), f].sum()), 0.0)
            for k, ((*_, t), r, b1) in enumerate(zip(words, rows, mid_rows(g.name))):
                sc = front + max(float(L[b1 + np.arange(len(mb)), mb].sum()), 0.0) + _segments(L, r, t, H)
                if sc > best[0]:
                    best = (sc, s0, cfo + df * g.rate, k)
    sc, s0, c, k = best
    if sc < PRIME_LLR:
        return None
    spec, nd, d, _ = words[k]
    return dict(_lock(g, s0, c, sc, (spec, nd, d, sc, sc), HDR_COPIES), primed=True)


def confirm(g: Grid, x: np.ndarray, lock: dict) -> float:
    """A lock's second look, once x holds its second header copy: tone_llr
    summed over the mid block and header copy 2 at the locked word's rows and
    tones, unfloored. ~0 where they faded or the burst stopped, positive on
    the burst, strongly negative where other tones are strong (a lock on
    someone's data)."""
    lay = layout(g.name, stream_symbols(g.name, lock["n_data"], lock["dup"]))
    L = tone_llr(g, _energies(g, x, lock["start"], int(lay.hdr_rows[1][-1]) + 1, lock["cfo"]))
    mb = mid_block(g)
    t = header_symbols(g.name, header_word(lock["spec"].index, lock["n_data"], lock["dup"]))
    return float(L[int(lay.sync_rows[lay.front]) + np.arange(len(mb)), mb].sum() + L[lay.hdr_rows[1], t].sum())


def confirm_end(lock: dict) -> int:
    """One past the last sample of the lock's second header copy (confirm's
    span), in its start's samples."""
    g = GRIDS[lock["band"]]
    lay = layout(g.name, stream_symbols(g.name, lock["n_data"], lock["dup"]))
    return lock["start"] + (int(lay.hdr_rows[1][-1]) + 1) * g.T


def reread(g: Grid, x: np.ndarray, lock: dict) -> tuple:
    """A lock's header read again at its start with every known row in x ->
    (spec, n_data, dup, score, runner-up): ML over the valid words of tone_llr
    over the front (floored at 0), each word's mid block and both header
    copies (read_mid's sum, at a known start)."""
    words, rows = valid_words(g.name), header_rows(g.name, HDR_COPIES)
    L = tone_llr(g, _energies(g, x, lock["start"], max(int(r[-1]) for r in rows) + 1, lock["cfo"]))
    f, mb, H = preamble_pattern(g), mid_block(g), hdr_len(g)
    front = max(float(L[np.arange(len(f)), f].sum()), 0.0)
    scores = np.array([front + float(L[b1 + np.arange(len(mb)), mb].sum()) + _segments(L, r, t, H)
                       for (*_, t), r, b1 in zip(words, rows, mid_rows(g.name))])
    order = np.argsort(scores)[::-1]
    spec, n, d, _ = words[order[0]]
    return spec, n, d, float(scores[order[0]]), float(scores[order[1]])


@lru_cache(maxsize=None)
def mid_rows(grid: str) -> tuple:
    """Per valid word, the row its mid block starts at."""
    return tuple(int(layout(grid, stream_symbols(grid, n, d)).sync_rows[len(preamble_pattern(GRIDS[grid]))])
                 for _, n, d, _ in valid_words(grid))


@lru_cache(maxsize=None)
def hdr_span(grid: str, copies: int) -> int:
    """Samples from a burst's start to the end of its first `copies` header
    copies, the latest any word puts them."""
    return int(max(r[-1] for r in header_rows(grid, copies)) + 1) * GRIDS[grid].T


def _lock(g: Grid, s0: int, cfo: float, score: float, h: tuple, copies: int) -> dict:
    """find's lock dict from a header read at s0 over `copies` copies."""
    spec, n, d, hs, h2 = h[:5]
    lay = layout(g.name, stream_symbols(g.name, n, d))
    return dict(spec=spec, n_data=n, dup=d, start=s0, cfo=cfo, score=score, header_score=hs, header_margin=hs - h2,
                end=s0 + lay.n * g.T, header_end=s0 + (lay.hdr_rows[copies - 1][-1] + 1) * g.T, band=g.name,
                family="cpm", copies=copies)


def receive(x: np.ndarray, lock: dict) -> dict:
    """An early lock's burst, whole in x (same sample origin) -> the receive
    result data2g.arq.phy reads: spec, n_cw (slots, control included), the
    per-slot soft bits and the tone energies."""
    g = GRIDS[lock["band"]]
    s0, cfo = refine(g, x, lock["start"], lock["cfo"], lock["spec"], lock["n_data"], lock["dup"])
    soft_, E = soft(g, lock["spec"], x, s0, cfo, lock["n_data"], lock["dup"])
    Z = bins(g, x, s0, layout(g.name, stream_symbols(g.name, lock["n_data"], lock["dup"])).n, cfo)
    r = dict(family="cpm", spec=lock["spec"], band=g.name, n_cw=len(soft_), n_ctl_slots=1 + lock["dup"],
             dup=lock["dup"], soft=soft_, E=E, Z=Z, cfo=lock["cfo"], preamble_start=lock["start"],
             header_end=lock.get("header_end"))
    if COH_CTL:  # the control slots on the coherent metric (sync and header rows known, the rest square-law decided)
        for s in range(r["n_ctl_slots"]):
            soft_[s] = dd_soft(r, dd_metric(r, {}, s), s)
    return r


# --- coherent detection --------------------------------------------------------------
# h = 1 with continuous phase: every symbol starts at the same carrier phase,
# which the square law throws away. A channel estimate per row from its
# neighbours' (known or decided) tones turns the metric coherent, and a poor
# estimate falls back to the square law by itself (coh_metric's model).
# scripts/cpm_floor_study.py (2026-11-11, branch cpm-coherent, 200 trials,
# 10% points vs the shipped receiver): coherent DD decoding fsk32r62 -0.7 to
# -1.8 dB, fsk8r50 -0.5 to -1.8; fsk16r25 with the either-gate lock below
# awgn -1.24 mpg -1.57 mpp -0.73 mpd -0.47 (noncoherent lock: -0.2 to -0.45).

COH_W = 8  # coh_metric's channel window for data (+- rows, the row left out): best of 1-64 on fading
COH_HDR_W = 4  # the header's (front + copy: ~22 rows)
# Lock gates on coh_read_header's scale (one copy, both), grids calibrated:
# 6000 noise windows' maxima (26.2, 33.3); (32, 36) gave 1 false lock in 8 h
# on noise and on own-grid traffic, (34, 38) none in 16 h of either
# (scripts/cpm_false_locks.py --coherent either). The noncoherent gates
# (LOCK_LLR) still lock alone: either reading passing locks.
# DATA2G_CPM_COH_LOCK=0: the noncoherent gates alone.
# c8r50: noise maxima (26.3, 32.0), but (34, 38) and (36, 40) gave 2 false
# locks in 16 h of own-grid traffic (0 without): early locks at a real
# burst's start reading another word (margin 11.9) where two overlap, so
# its one-copy margin is 14 (AWGN -14 dB: one-copy locks 0.41, square law
# 0.28; fading even); c32r62: (32.4, 38.7), so (36, 42): 9 own-grid, 1
# noise, 1 OFDM in 16 / 16 / 8 h, as many as without (scripts/
# coh_lock_cal.py, 6000 windows a grid; runs/cohcal).
# Per grid: (score one copy, both, margin one copy, both).
COH_LOCK = ({"c16r25": (34.0, 38.0, HDR_MARGIN[0], HDR_MARGIN[1]), "c8r50": (36.0, 40.0, 14.0, HDR_MARGIN[1]),
             "c32r62": (36.0, 42.0, HDR_MARGIN[0], HDR_MARGIN[1])}
            if os.environ.get("DATA2G_CPM_COH_LOCK", "1") == "1" else {})
# The coherent reading runs only where the noncoherent score reaches this (one
# copy, both): run at every detect placement it doubled listening CPU on the
# Pi 4 (CPM 5.7 -> 15.3% of a core on noise; docs/pi-bench-coherent.md). Of
# the calibration's coherent locks the noncoherent gate missed, the lowest
# noncoherent score was 14 (one copy) / 8 (both); at (12, 10) none / under
# 0.2% are lost, and the read runs at 2-10% / 10-42% of noise placements.
COH_PREGATE = (12.0, 10.0)
# The control codeword's soft bits on dd_metric with no decoded data (known
# rows, square-law decisions elsewhere), from the start: 10% points, genie
# lock, unprimed and single, vs the square law: awgn fsk16r25 -1.19,
# fsk8r50 -1.46, fsk32r62 -1.03 dB; fading -0.2 to -1.0; flutter / hilat
# within +-0.12 (after the data's DD, as ModemRx's soft then gives it:
# awgn -1.70, -1.89, -1.13). DATA2G_CPM_COH_CTL=0: the square law.
COH_CTL = os.environ.get("DATA2G_CPM_COH_CTL", "1") == "1"


def bins(g: Grid, x: np.ndarray, start: int, n_sym: int, cfo: float) -> np.ndarray:
    """(n_sym, m) complex tone bins from `start`, CFO removed: _energies before the square."""
    T = g.T
    idx = start + np.arange(n_sym * T)
    seg = np.where((idx >= 0) & (idx < len(x)), x[np.clip(idx, 0, len(x) - 1)], 0.0)
    z = seg * np.exp(-2j * np.pi * (g.f0 + cfo) * idx / FS)
    return np.fft.fft(z.reshape(n_sym, T), axis=1)[:, :g.m]


def coh_metric(Z: np.ndarray, P: np.ndarray, W: int, n0: float) -> np.ndarray:
    """(n, m) complex bins of consecutive rows and P (n, m) the probability
    of each tone sent (one-hot where known) -> (n, m) tone log-likelihoods,
    coherent: Z_k = h_i + CN(0, N0 + s2) on the sent tone, noise alone
    elsewhere, h_i = sum over rows j != i within +-W of sum_k P_j(k) Z_j[k],
    s2 its error (the residual's mean less N0). The residual timing's
    per-tone phase ramp and the residual CFO come out first, from the
    decided tones' consecutive pairs weighted by their probabilities."""
    n, m = Z.shape
    r = np.arange(n)
    t = P.argmax(axis=1)
    c = P[r, t]
    a = Z[r, t]
    d = np.diff(t)
    p = a[1:] * np.conj(a[:-1]) * c[1:] * c[:-1]  # ~ exp(i (nu - 2 pi d delta))
    grid = np.linspace(-0.5, 0.5, 401)
    dv, inv = np.unique(d, return_inverse=True)  # the pairs summed per tone step first: the phase depends on d alone
    q = np.zeros(len(dv), complex)
    np.add.at(q, inv, p)
    delta = grid[np.argmax(np.abs(np.exp(2j * np.pi * np.outer(grid, dv)) @ q))]
    nu = np.angle(np.sum(p * np.exp(2j * np.pi * delta * d)))
    Z = Z * np.exp(2j * np.pi * delta * np.arange(m))[None] * np.exp(-1j * nu * r)[:, None]
    a = (P * Z).sum(axis=1)
    k = np.ones(2 * W + 1)
    k[W] = 0.0
    h = np.convolve(a, k, "same") / np.maximum(np.convolve(np.ones(n), k, "same"), 1)
    res = _sq(Z - h[:, None])
    nb = tone_floors(_sq(Z), n0)
    if nb is not None:  # per-tone floors: Z_k = h_i + CN(0, N_k + s2) on the sent tone, CN(0, N_k) elsewhere
        s2 = max(float(np.mean((P * (res - nb[None])).sum(axis=1))), n0 * 1e-3)
        return _sq(Z) / nb - res / (nb + s2) - np.log1p(s2 / nb)
    v = max(float(np.mean((P * res).sum(axis=1))), n0 * (1 + 1e-3))
    return _sq(Z) / n0 - res / v


def _n0(E: np.ndarray) -> float:
    """tone_metric's noise level: all but each row's strongest tone, exponential."""
    return max(float(np.median(np.sort(E, axis=1)[:, :-1])) / np.log(2), 1e-30)


def _coh_run_llr(g: Grid, Z: np.ndarray, t: np.ndarray, n0: float) -> np.ndarray:
    """Per-row tone_llr of known tones t over a run of consecutive rows, on coh_metric."""
    P = np.zeros(Z.shape)
    P[np.arange(len(t)), t] = 1.0
    L = coh_metric(Z, P, COH_HDR_W, n0)
    L = L - np.logaddexp.reduce(L, axis=1, keepdims=True) + np.log(g.m)
    L = np.logaddexp(np.log1p(-LLR_EPS) + L, np.log(LLR_EPS))
    return L[np.arange(len(t)), t]


def coh_read_header(g: Grid, x: np.ndarray, s0: int, cfo: float, copies: int = HDR_COPIES):
    """read_header on coh_metric: per word, the front and its first copy as
    one run, the block before its second copy and that copy as another,
    each run's sum floored at 0. Same tuple (front: run 1's front rows)."""
    words, rows = valid_words(g.name), header_rows(g.name, copies)
    n = max(r[-1] for r in rows) + 1
    Z = bins(g, x, s0, n, cfo)
    n0 = _n0(_sq(Z))
    f, mb, H = preamble_pattern(g), mid_block(g), hdr_len(g)
    F = len(f)
    scores, front = [], 0.0
    for (_, _, _, t), r in zip(words, rows):
        la = _coh_run_llr(g, Z[np.concatenate([np.arange(F), r[:H]])], np.concatenate([f, t]), n0)
        front = max(front, float(la[:F].sum()))
        sc = max(float(la.sum()), 0.0)
        if copies > 1:
            r2 = r[H:]
            rb = np.concatenate([np.arange(r2[0] - len(mb), r2[0]), r2])
            sc += max(float(_coh_run_llr(g, Z[rb], np.concatenate([mb, t]), n0).sum()), 0.0)
        scores.append(sc)
    s = np.array(scores)
    o = np.argsort(s)[::-1]
    spec, nd, d, _ = words[o[0]]
    return spec, nd, d, float(s[o[0]]), float(s[o[1]]), front


def _lock_header(g: Grid, x: np.ndarray, s0: int, cfo: float, copies: int) -> tuple:
    """-> (read_header's tuple, passed): the noncoherent reading if it passes
    LOCK_LLR (copies 1: [0] gates, else [1]); else, if the grid has COH_LOCK
    and the noncoherent score reaches COH_PREGATE, the coherent reading if it
    passes COH_LOCK; else the noncoherent one, not passed."""
    i = 0 if copies == 1 else 1
    h = read_header(g, x, s0, cfo, copies)
    if h[3] >= LOCK_LLR[g.name][i] and h[3] - h[4] >= HDR_MARGIN[i]:
        return h, True
    if g.name in COH_LOCK and h[3] >= COH_PREGATE[i]:
        c = coh_read_header(g, x, s0, cfo, copies)
        if c[3] >= COH_LOCK[g.name][i] and c[3] - c[4] >= COH_LOCK[g.name][2 + i]:
            return c, True
    return h, False


def slot_rows(g: Grid, n_data: int, dup: bool) -> tuple:
    """Per slot (control first), its rows in the burst."""
    lay = layout(g.name, stream_symbols(g.name, n_data, dup))
    sizes = [CTL[g.name].n_sym] * (1 + dup) + [DATA_N // g.bits] * n_data
    edges = np.cumsum([0] + sizes)
    return tuple(lay.data_rows[a:b] for a, b in zip(edges[:-1], edges[1:]))


def dd_metric(r: dict, post: dict, slot: int) -> np.ndarray:
    """cpm.receive's result, `post` {data slot: decode_posterior's GF
    log-probabilities, as sent} and a slot -> (its rows, m) coh_metric over
    the burst rows from its first less COH_W to its last plus COH_W (timing
    ramp and CFO fitted there): sync and header rows known, those slots'
    rows their posteriors, every other row its square-law probabilities.
    Local to the slot, so a receiver can run it once the slot and COH_W
    rows more are in, while the burst's later codewords arrive."""
    g, spec = GRIDS[r["band"]], r["spec"]
    n_data, dup = r["n_cw"] - r["n_ctl_slots"], r["dup"]
    lay = layout(g.name, stream_symbols(g.name, n_data, dup))
    rows = slot_rows(g, n_data, dup)
    a, b = max(0, rows[slot][0] - COH_W), min(lay.n, rows[slot][-1] + 1 + COH_W)
    Z = r["Z"][a:b]
    E = _sq(Z)
    G = tone_metric(g, E)
    P = np.exp(G - G.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    h = header_symbols(g.name, header_word(spec.index, n_data, dup))
    known = np.concatenate([lay.sync_rows] + list(lay.hdr_rows))
    kt = np.concatenate([lay.sync_tones] + [h] * len(lay.hdr_rows))
    keep = (known >= a) & (known < b)
    P[known[keep] - a] = 0.0
    P[known[keep] - a, kt[keep]] = 1.0
    gray = _gray(g.m)
    for s, v in post.items():
        rr = rows[s]
        keep = (rr >= a) & (rr < b)
        if not keep.any():
            continue
        lp = np.asarray(v).reshape(-1, g.m)[keep]
        q = np.exp(lp - lp.max(axis=1, keepdims=True))
        q /= q.sum(axis=1, keepdims=True)
        P[rr[keep][:, None] - a, gray[None]] = q  # value v on tone gray(v)
    return coh_metric(Z, P, COH_W, _n0(E))[rows[slot] - a]


def dd_soft(r: dict, metric: np.ndarray, slot: int) -> np.ndarray:
    """Slot `slot`'s soft values from its dd_metric (control: bit LLRs, data: symbol log-likelihoods)."""
    g = GRIDS[r["band"]]
    L = metric
    if slot < r["n_ctl_slots"]:
        return np.clip(bit_llrs(g, L), -30, 30)
    L = np.clip(L - L.max(axis=1, keepdims=True), -60, 0)
    return L[:, _gray(g.m)].reshape(-1)
