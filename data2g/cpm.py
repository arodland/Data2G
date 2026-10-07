"""Single-carrier constant-envelope modes (continuous-phase M-FSK,
noncoherent) with Costas-array sync and a header: the ARQ's bottom rungs.
Prototyped in the cpm-modes fork (scripts/cpm_study.py, cpm_spectrum.py):
4-7 dB better than OFDM of similar rate on fading, PEP-fair, 10% points.

Signal: M tones spaced by the symbol rate R (h = 1), continuous phase, the
OFDM bands' TX bandpass over the tones with clip-and-filter at no headroom,
its margin (and a glide at each tone change) by the session's bandwidth cap
(TX_FILTERS: envelope peak 0.1-0.6 dB over average; beyond the segment or
band edge under -37 dB). A grid (tones, spacing) is what OFDM calls a sync
band.

Codewords go through data2g.codes like OFDM's (CRC masks, scrambler, RVs,
soft combining): a CpmSpec quacks like a config.SubmodeSpec there. Burst:

    [front Costas block][header][ctl][ctl RV1?][data ...][short block][header][data ...][short block] ...

- ctl: the grid's short control codeword (polar, CTL_K of CTL_N), one per
  burst, twice (RV 0, RV 1) when the ARQ duplicates control (ARQ_DUP);
- data: 0..MAX_DATA LDPC codewords of the mode (n = 960);
- sync blocks every SPACING_S of symbols, at positions independent of the
  burst's length, so the receiver finds them before it knows the length;
- header (twice, after the first two blocks): 16 bits as OFDM's: index =
  mode index + 2 x duplicated, n field = data codewords + 1, CRC-6.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import permutations

import numpy as np

from .config import CLIP_OVERSHOOT, FS

PREAMBLE_S = 0.45  # the front sync block
BLOCK_S = 0.25  # the later sync blocks, about
SPACING_S = 2.5  # symbols between sync blocks, about (MPG fades last ~1-3 s)
HDR_S = 0.25  # a header copy, about
HDR_COPIES = 2  # after the first two sync blocks
MAX_DATA = 8  # data codewords per burst
PEAK_RATIO = 0.7  # a lock's floor on _peak_ratio: fronts 0.82+, locks inside a burst 0.59 at most
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
    config.SubmodeSpec where data2g.codes and the ARQ read it."""
    name: str
    grid: str
    index: int  # in its grid's header (-1: the control codeword)
    code: str  # "ldpc" (data) | "polar" (control)
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
    CpmSpec(f"{_grid_name(g)}-r1/3", g.name, 0, "ldpc", 320, DATA_N),
    CpmSpec(f"{_grid_name(g)}-r1/2", g.name, 1, "ldpc", 480, DATA_N),
)}
CTL = {g: CpmSpec(f"{_grid_name(GRIDS[g])}-ctl", g, -1, "polar", CTL_K, CTL_N) for g in GRIDS}


# Early lock (find): the front block's score, then its header copy's.
# Provisional, from 120 buffers each (2026-09-25): 1.1x the largest score
# noise or an OFDM burst gave; CPM bursts near their thresholds (fading mix)
# scored at least 0.098 / 0.151 / 0.053 in 99%, so a few bursts at the very
# bottom miss their early lock (the fork measured 66-100% at threshold).
SYNC_THRESHOLD = {"c16r25": 0.11, "c8r50": 0.20, "c32r62": 0.085}
HEADER_THRESHOLD = {"c16r25": 0.20, "c8r50": 0.29, "c32r62": 0.11}


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


def preamble_pattern(g: Grid) -> np.ndarray:
    """The front sync block: Costas tiled to ~PREAMBLE_S."""
    c = np.array(costas(g.m))
    return np.tile(c, max(1, int(np.ceil(PREAMBLE_S * g.rate / len(c)))))


def mid_block(g: Grid) -> np.ndarray:
    """A later sync block, ~BLOCK_S: the largest Welch Costas array that
    fits, its tones spread over the band (scaling keeps the Costas property)."""
    target = max(4, round(BLOCK_S * g.rate))
    p = max(q for q in range(3, min(target, g.m) + 2) if _prime(q))
    c = _welch(p) * (g.m // (p - 1))
    if np.array_equal(c, costas(g.m)):
        # the front's own array (c8r50): a mid block then matched half the
        # front pattern (score 0.50) and whole-burst searches locked on it.
        # Time-reversed (still Costas): 0.17
        c = c[::-1]
    return np.tile(c, max(1, round(target / len(c))))


# --- header --------------------------------------------------------------------------

def hdr_len(g: Grid) -> int:
    return max(6, round(HDR_S * g.rate))


def header_word(index: int, n_data: int, dup: bool) -> int:
    """16 bits as the OFDM header's: index (4) | n field (6) | CRC-6; index
    = mode index + 2 x duplicated control, n field = data codewords + 1."""
    from .modem import _crc6

    v = ((index + 2 * dup) << 6) | n_data
    return (v << 6) | _crc6(v)


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
        b = preamble_pattern(g) if i == 0 else mid_block(g)
        rows += range(r, r + len(b))
        tones_ += list(b)
        r += len(b)
        if i < HDR_COPIES:
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
           floor: float = -1.0):
    """-> (score, burst start sample, cfo Hz): the sync pattern's best
    placement. Coarse: symbol energies at T/4 hops and quarter-bin CFO
    steps, as shares, summed over the sync symbols of the shortest burst
    (`n_sym` stream symbols; default a control-only burst's) at every start
    and whole-bin shift. Score: the mean share on the pattern (0..1).
    `front_only`: the first block alone (an early lock); `floor`: no fine
    stage under this score (the search's CPU on noise)."""
    lay = layout(g.name, n_sym or stream_symbols(g.name, 0, False))
    rows, tones_ = lay.sync_rows, lay.sync_tones
    if front_only:
        rows, tones_ = rows[:lay.front], tones_[:lay.front]
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


def llrs(g: Grid, E: np.ndarray) -> np.ndarray:
    """(n_sym, m) tone energies of one codeword -> (n_sym * bits,) coded-bit
    LLRs (positive: 0) in mapping order (data2g.codes.combine / decode_many's). Noncoherent square-law metric, exact for
    Rayleigh fading: log P(E | tone m) = E_m g + const, g = Es / (N0 (N0 + Es))."""
    top = E.max(axis=1)
    n0 = np.median(np.sort(E, axis=1)[:, :-1]) / np.log(2)  # exponential: median = N0 ln 2
    n0 = max(n0, 1e-12 * np.mean(top) + 1e-300)  # noise-free tests
    es = max(np.mean(top) - n0, 1e-3 * n0)
    # es / (n0 (n0 + es)) without its underflow: a codeword past the audio's
    # end (zero-padded: a header claiming more than arrived) reads as erasures
    metric = E * (es / (n0 + es) / n0)
    label = np.argsort(_gray(g.m))  # tone -> the bit group it carries
    out = np.empty((len(E), g.bits))
    for b in range(g.bits):
        bit = (label >> (g.bits - 1 - b)) & 1
        out[:, b] = np.logaddexp.reduce(metric[:, bit == 0], axis=1) - np.logaddexp.reduce(metric[:, bit == 1], axis=1)
    return out.reshape(-1)




def read_header(g: Grid, x: np.ndarray, s0: int, cfo: float, copies: int = HDR_COPIES):
    """-> (spec, n_data, dup, score, runner-up score): ML over the grid's
    valid words, shares summed over the first `copies` header copies (1: all
    an early lock has)."""
    lay = layout(g.name, stream_symbols(g.name, 0, False))
    rows = np.concatenate(lay.hdr_rows[:copies])
    E = _shares(_energies(g, x, s0, rows[-1] + 1, cfo))[rows]
    words = valid_words(g.name)
    scores = np.array([E[np.arange(len(rows)), np.tile(t, copies)].mean() for *_, t in words])
    order = np.argsort(scores)[::-1]
    spec, n, d, _ = words[order[0]]
    return spec, n, d, float(scores[order[0]]), float(scores[order[1]])


def soft(g: Grid, spec: CpmSpec, x: np.ndarray, s0: int, cfo: float, n_data: int, dup: bool):
    """-> (per-slot soft bits in mapping order, the stream's tone energies)."""
    n_sym = stream_symbols(g.name, n_data, dup)
    lay = layout(g.name, n_sym)
    E = _energies(g, x, s0, lay.n, cfo)[lay.data_rows]
    sizes = [CTL[g.name].n_sym] * (1 + dup) + [DATA_N // g.bits] * n_data
    out, i = [], 0
    for n in sizes:
        out.append(llrs(g, E[i:i + n]))
        i += n
    return out, E


def measure(g: Grid, E: np.ndarray, n_sym: int) -> dict:
    """The gear shifter's inputs, as arq.phy.measure gives them for OFDM:
    per-symbol SNR from the winning tone's energy over the noise bins,
    effective MI per constellation over it, SNR in the 2500 Hz reference,
    and a fading-rate proxy for spread (symbol energies' decorrelation)."""
    from .arq import predictor as P
    from .config import FRAME_SAMPLES, SNR_REF_BW_HZ

    top = E.max(axis=1)
    n0 = max(np.median(np.sort(E, axis=1)[:, :-1]) / np.log(2), 1e-300)
    snr = np.maximum(top / n0 - 1, 1e-6)
    es = max(float(np.mean(top) - n0), 1e-12)
    rho = np.corrcoef(top[:-1], top[1:])[0, 1] if len(top) > 3 and np.std(top) > 0 else 1.0
    spread = float(np.clip(np.sqrt(max(-np.log(max(rho, 1e-3)), 0)) * g.rate / np.pi, 0.0, 3.0))
    out = dict(snr_est=10 * np.log10(es / n0 * g.rate / SNR_REF_BW_HZ), spread_est=spread, delay_est_ms=0.0,
               headroom=0.0, frames=max(1.0, n_sym * g.T / FRAME_SAMPLES))
    for c in P.CONSTS:
        out[f"mi_{c}"] = P.effective_mi(np.sqrt(snr), np.ones_like(snr), c)
    return out


def _peak_ratio(g: Grid, x: np.ndarray, s0: int, cfo: float) -> float:
    """The front pattern's share over each symbol's strongest share there:
    ~1 on a front at any SNR (0.82 and up, -10 to 25 dB), low where the
    pattern met a burst's data by chance (on a clean signal each symbol is
    one tone, and 5 of 24 on the pattern cleared the sync threshold)."""
    f = preamble_pattern(g)
    E = _shares(_energies(g, x, s0, len(f), cfo))
    return float(E[np.arange(len(f)), f].mean() / E.max(axis=1).mean())


def find(g: Grid, x: np.ndarray, threshold: float | None = None, reach_hz: float = 150.0, front_only: bool = True,
         lo: int = 0, hi: int | None = None):
    """-> early lock {spec, n_data, dup, start, end, score, header score} of
    a burst of grid g in x, or None: the sync pattern (front block alone by
    default) scoring over `threshold` (default SYNC_THRESHOLD; given: no
    header floor either), then its first header copy (HEADER_THRESHOLD).
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
    if s0 + (layout(g.name, stream_symbols(g.name, 0, False)).hdr_rows[0][-1] + 1) * g.T > len(x):
        return None  # its header copy is still arriving
    # a tiled front (c8r50: a 6-symbol Costas array four times) also matches
    # whole periods late, with the header's first symbols standing in for the
    # last tile: fsk8r50-r1/3 locked one period late in 3-8% of bursts at
    # MPP -3 dB. The alignment whose header reads best is the burst's.
    period = len(costas(g.m)) * g.T
    tiles = len(preamble_pattern(g)) * g.T // period
    starts = [s for s in (s0 - k * period for k in range(tiles)) if s >= 0]
    if not starts:
        return None  # fine timing moved a lock at x's first sample before it: its front is cut off
    s0, (spec, n, d, hs, h2) = max(((s, read_header(g, x, s, cfo, copies=1)) for s in starts),
                                   key=lambda c: c[1][3])
    if _peak_ratio(g, x, s0, cfo) < PEAK_RATIO:
        return None  # a burst's data or a later sync block, its front missed
    if threshold is None and hs < HEADER_THRESHOLD[g.name]:
        return None
    lay = layout(g.name, stream_symbols(g.name, n, d))
    return dict(spec=spec, n_data=n, dup=d, start=s0, cfo=cfo, score=score, header_score=hs, header_margin=hs - h2,
                end=s0 + lay.n * g.T, header_end=s0 + (lay.hdr_rows[0][-1] + 1) * g.T, band=g.name, family="cpm")


def receive(x: np.ndarray, lock: dict) -> dict:
    """An early lock's burst, whole in x (same sample origin) -> the receive
    result data2g.arq.phy reads: spec, n_cw (slots, control included), the
    per-slot soft bits and the tone energies."""
    g = GRIDS[lock["band"]]
    soft_, E = soft(g, lock["spec"], x, lock["start"], lock["cfo"], lock["n_data"], lock["dup"])
    return dict(family="cpm", spec=lock["spec"], band=g.name, n_cw=len(soft_), n_ctl_slots=1 + lock["dup"],
                dup=lock["dup"], soft=soft_, E=E, cfo=lock["cfo"], preamble_start=lock["start"],
                header_end=lock.get("header_end"))
