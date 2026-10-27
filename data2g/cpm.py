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
    return np.abs(Z[:, :g.m + 2 * extra]) ** 2


def _shares(E: np.ndarray) -> np.ndarray:
    """Each symbol's energies as shares of its total: raw energy let a strong
    stretch of data outscore a sync block in a fade (MPG, 20 dB: locked 4 s
    late); a share can't be inflated."""
    return E / (E.sum(axis=1, keepdims=True) + 1e-30)


def detect(g: Grid, x: np.ndarray, reach_hz: float = 150.0, fine=True, front_only: bool = False, n_sym: int = 0,
           floor: float = -1.0, mid: bool = False):
    """-> (score, burst start sample, cfo Hz): the sync pattern's best
    placement. Coarse: symbol energies at T/4 hops and quarter-bin CFO
    steps, as shares, summed over the sync symbols of the shortest burst
    (`n_sym` stream symbols; default a control-only burst's) at every start
    and whole-bin shift. Score: the mean share on the pattern (0..1).
    `front_only`: the first block alone (an early lock); `mid`: the mid
    block alone, its first sample for the burst's start (find_mid);
    `floor`: no fine stage under this score (the search's CPU on noise)."""
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
    for frac in (0.0, 0.25, 0.5, 0.75):
        # mixed once per CFO fraction, sliced per timing phase (as
        # _energies mixes: the ramp is by absolute sample, so a slice of it
        # is what _energies(x, off, ...) would build; 4x fewer exps)
        zf = x * np.exp(-2j * np.pi * (g.f0 - extra * g.rate + frac * g.rate) * np.arange(len(x)) / FS)
        for ph in range(4):
            off = ph * T // 4
            n = (len(x) - off) // T
            if n < span:
                continue
            Z = np.fft.fft(zf[off:off + n * T].reshape(n, T), axis=1)
            E = _shares(np.abs(Z[:, :g.m + 2 * extra]) ** 2)
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


def tone_metric(g: Grid, E: np.ndarray) -> np.ndarray:
    """(n_sym, m) tone energies of one codeword -> (n_sym, m) log P(E | tone
    sent) up to a per-symbol constant. Noncoherent square-law metric, exact
    for Rayleigh fading: E_m g, g = Es / (N0 (N0 + Es)), Es and N0 the
    codeword's."""
    top = E.max(axis=1)
    n0 = np.median(np.sort(E, axis=1)[:, :-1]) / np.log(2)  # exponential: median = N0 ln 2
    n0 = max(n0, 1e-12 * np.mean(top) + 1e-300)  # noise-free tests
    es = max(np.mean(top) - n0, 1e-3 * n0)
    # es / (n0 (n0 + es)) without its underflow: a codeword past the audio's
    # end (zero-padded: a header claiming more than arrived) reads as erasures
    return E * (es / (n0 + es) / n0)


def llrs(g: Grid, E: np.ndarray) -> np.ndarray:
    """(n_sym, m) tone energies of one codeword -> (n_sym * bits,) coded-bit
    LLRs (positive: 0) in mapping order (data2g.codes.combine / decode_many's),
    marginals of tone_metric: the control codeword's."""
    metric = tone_metric(g, E)
    label = np.argsort(_gray(g.m))  # tone -> the bit group it carries
    out = np.empty((len(E), g.bits))
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
    w = float(tone_metric(g, E0).sum() / tot) if tot > 0 else 0.0

    def ll(c):
        G = _energies(g, x, c[0], n, c[1]) * w
        return G[kr, known[kr]].sum() + np.logaddexp.reduce(G[ur], axis=1).sum()
    best = (s0, cfo)
    for st, sf in ((max(1, g.T // 16), g.rate / 16), (max(1, g.T // 64), g.rate / 64)):
        s, f = best
        best = max(((s + i * st, f + j * sf) for i in range(-2, 3) for j in range(-2, 3)), key=ll)
    return best


def measure(g: Grid, E: np.ndarray, n_sym: int) -> dict:
    """The gear shifter's measurements, as arq.phy.measure gives them for
    OFDM: SNR in the 2500 Hz reference, a fading-rate proxy for spread
    (symbol energies' decorrelation), and `snr`, per symbol, from the
    winning tone's energy over the noise bins (phy.measure makes the
    effective-MI features of it). Native cpm::measure."""
    from .config import FRAME_SAMPLES, SNR_REF_BW_HZ

    top = E.max(axis=1)
    n0 = max(np.median(np.sort(E, axis=1)[:, :-1]) / np.log(2), 1e-300)
    snr = np.maximum(top / n0 - 1, 1e-6)
    es = max(float(np.mean(top) - n0), 1e-12)
    rho = np.corrcoef(top[:-1], top[1:])[0, 1] if len(top) > 3 and np.std(top) > 0 else 1.0
    spread = float(np.clip(np.sqrt(max(-np.log(max(rho, 1e-3)), 0)) * g.rate / np.pi, 0.0, 3.0))
    return dict(snr_est=10 * np.log10(es / n0 * g.rate / SNR_REF_BW_HZ), spread_est=spread, delay_est_ms=0.0,
                headroom=0.0, frames=max(1.0, n_sym * g.T / FRAME_SAMPLES), snr=snr)


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
         lo: int = 0, hi: int | None = None, wait: bool = False):
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
    searches each start once: detect's cost is the audio it is given)."""
    floor = SYNC_THRESHOLD[g.name] if threshold is None else threshold
    if lo or hi is not None:
        lay0 = layout(g.name, stream_symbols(g.name, 0, False))
        span = ((lay0.sync_rows[lay0.front - 1] if front_only else lay0.sync_rows[-1]) + 2) * g.T
        a = max(0, lo - g.T)
        b = len(x) if hi is None else min(len(x), hi + span + g.T)
        score, s0, cfo = detect(g, x[a:b], reach_hz=reach_hz, front_only=front_only, floor=floor)
        s0 += a
        if not lo <= s0 < (len(x) if hi is None else hi):
            return None
    else:
        score, s0, cfo = detect(g, x, reach_hz=reach_hz, front_only=front_only, floor=floor)
    if score < (SYNC_THRESHOLD[g.name] if threshold is None else threshold):
        return None
    if s0 + hdr_span(g.name, 1) > len(x):
        return None  # its header copy is still arriving
    if s0 < 0:
        return None  # fine timing moved a lock at x's first sample before it: its front is cut off
    h = read_header(g, x, s0, cfo, copies=1)
    if threshold is not None or (h[3] >= LOCK_LLR[g.name][0] and h[3] - h[4] >= HDR_MARGIN[0]):
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
    h = read_header(g, x, cand["start"], cand["cfo"], copies=HDR_COPIES)
    if h[3] < LOCK_LLR[g.name][1] or h[3] - h[4] < HDR_MARGIN[1]:
        return None
    return _lock(g, cand["start"], cand["cfo"], cand["score"], h, HDR_COPIES)


def find_mid(g: Grid, x: np.ndarray, lo: int = 0, hi: int | None = None):
    """-> find's lock of a burst whose front was missed, from its mid block
    (the sync block before header copy 2) and both header copies, or None.
    Block 1's start in [lo, hi); each word puts it at its own row (a short
    burst's after its data), so each is read at the start it implies, and
    a start before x is no reading."""
    T, nb = g.T, len(mid_block(g))
    a = max(0, lo - T)
    b = len(x) if hi is None else min(len(x), hi + (nb + 1) * T)
    score, p, cfo = detect(g, x[a:b], mid=True, floor=MID_THRESHOLD[g.name])
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
                family="cpm")


def receive(x: np.ndarray, lock: dict) -> dict:
    """An early lock's burst, whole in x (same sample origin) -> the receive
    result data2g.arq.phy reads: spec, n_cw (slots, control included), the
    per-slot soft bits and the tone energies."""
    g = GRIDS[lock["band"]]
    s0, cfo = refine(g, x, lock["start"], lock["cfo"], lock["spec"], lock["n_data"], lock["dup"])
    soft_, E = soft(g, lock["spec"], x, s0, cfo, lock["n_data"], lock["dup"])
    return dict(family="cpm", spec=lock["spec"], band=g.name, n_cw=len(soft_), n_ctl_slots=1 + lock["dup"],
                dup=lock["dup"], soft=soft_, E=E, cfo=lock["cfo"], preamble_start=lock["start"],
                header_end=lock.get("header_end"))
