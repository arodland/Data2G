"""Hand burst: the D2G net's show of hands (docs/hand-burst.md).

One 8-FSK continuous-phase burst per station, tone spacing 6.25 Hz
(Hand-50: if your rig works for FT8, it works for this), constant
envelope. 63 payload bits + CRC-16 seeded with the 8-bit request tag (so
a Hand can't answer another net's call), CRC-aided polar K=79 of E=162
(data2g.polar, as the ACK submode), a fixed interleaver (seed 0) over the
54 three-bit data symbols between two Welch Costas blocks:

    [sync 6][data 27][sync 6][data 27]   66 symbols, 10.56 s

Net control knows each Hand's slot and channel, so the decoder skips the
blind search: the sync blocks' tone shares over nominal +-0.6 s x +-5 Hz
(the response's timing margins: host latencies), the top 3 candidates each
nailed in on a T/16 x rate/16 grid before its soft bits (a quarter symbol
off costs ~0.7-1 dB at the threshold), polar list decoding, first CRC pass
wins (false decodes ~2.6e-4 per attempt measured, about 1% of 40-channel
rounds).

Channels: 40 at 60 Hz pitch centred on 1500 Hz (channel 0 at 330 Hz). An
answerer shifts its Hand by the cfo of net control's ref burst (HAND
SEND), which lands it in its channel as net control hears it; the +-5 Hz
search covers the rest. Ordering the channel map by SNR and leaving gaps
at jumps over 5 dB (the near-far rule) is the client's job.

The host primitive over this: docs/arq.md §9 (HAND SEND, HAND LISTEN, the
REF and HAND lines), driven from arq.engine.Engine. Prototyped and
measured in scripts/hand_study.py (branch hand-burst-sim).

Native: native/core/hand (tests/test_native_hand.py checks parity; the
interleaver and Costas order are frozen by tools/gen_native_tables.py).
"""

from functools import lru_cache

import numpy as np

from . import cpm, hfchannel, polar
from .codes import POLAR_DESIGN_SNR_DB, POLAR_LIST
from .config import FS
from .waveform import dsp

M, RATE = 8, 6.25
PAYLOAD = 63  # payload bits
K = PAYLOAD + 16  # + CRC-16 (0x1021, register seeded with the request tag)
E = 162  # coded bits: rate 0.49, 54 symbols of 3 bits
N_DATA = E // 3
SYNC = np.array(cpm.costas(M))  # Welch, order 6
SYNC_ROWS = np.r_[0:len(SYNC), len(SYNC) + N_DATA // 2:2 * len(SYNC) + N_DATA // 2]
DATA_ROWS = np.setdiff1d(np.arange(2 * len(SYNC) + N_DATA), SYNC_ROWS)
N_SYM = 2 * len(SYNC) + N_DATA  # 66
LEN_S = N_SYM / RATE  # 10.56 s

N_CHANNELS, PITCH, CENTER = 40, 60.0, 1500.0  # channel c -> CENTER + (c - 19.5) * PITCH
SEARCH_S, SEARCH_HZ, TOP = 0.6, 5.0, 3
# A round's grids. ROUND_FULL ships. ROUND_CHEAP is what --hand-round cheap
# picks on a station that needs the round for about half the CPU: one fewer
# candidate (the third is a near-duplicate of the first two), a T/4 search
# step the nailing recovers, and no CFO refinement. Measured, the preset costs
# 0.2-0.4 dB: the two grids are free, the CFO refinement is what you pay for
# (docs/hand-burst.md).
ROUND_FULL = dict(top=TOP, step_div=8, nail_cfo=True)
ROUND_CHEAP = dict(top=2, step_div=4, nail_cfo=False)
ROUNDS = {"full": ROUND_FULL, "cheap": ROUND_CHEAP}
LATE_S = 0.5  # a scheduled Hand this late past its slot still goes; later is dropped (busy)

SEARCH_SAMPLES = int(round(SEARCH_S * FS))
LEN_SAMPLES = int(round(LEN_S * FS))
WINDOW_SAMPLES = 2 * SEARCH_SAMPLES + LEN_SAMPLES  # net control's capture, +-0.6 s around the slot


def channel_freq(c: int) -> float:
    return CENTER + (c - (N_CHANNELS - 1) / 2) * PITCH


def payload_bits(p: int) -> np.ndarray:
    """63 bits, most significant first."""
    p = int(p) & ((1 << PAYLOAD) - 1)
    return np.unpackbits(np.frombuffer(p.to_bytes(8, "big"), np.uint8))[1:]


def payload_int(bits: np.ndarray) -> int:
    return int.from_bytes(np.packbits(bits).tobytes(), "big") >> 1


def crc16(bits: np.ndarray, init: int) -> np.ndarray:
    """CRC-16/CCITT (0x1021) of a bit vector, register seeded with `init`
    (the request tag in its high byte: the tag authenticates without bits)."""
    c = init
    for b in bits:
        fb = ((c >> 15) & 1) ^ int(b)
        c = ((c << 1) & 0xFFFF) ^ (0x1021 if fb else 0)
    return np.array([(c >> (15 - i)) & 1 for i in range(16)], np.uint8)


@lru_cache(maxsize=None)
def code() -> polar.PolarCode:
    return polar.PolarCode(K, E, design_snr_db=POLAR_DESIGN_SNR_DB)


@lru_cache(maxsize=None)
def interleaver() -> np.ndarray:
    return np.random.default_rng(0).permutation(E)


@lru_cache(maxsize=None)
def grid(freq: float) -> cpm.Grid:
    return cpm.Grid(f"hand{freq:g}", M, RATE, center=freq)


class HandBurst:
    """Stand-in for a TxBurst in Engine.tx while a Hand is on air: the
    end-of-tx path reads .submode (the host's MODE line) and hands it to
    kiss.on_sent, whose equality test against its inflight never matches."""
    submode = "hand"


def modulate(tag: int, payload: np.ndarray, freq: float) -> np.ndarray:
    """63 payload bits -> unit-RMS Hand audio at `freq` (Hz, tone centre)."""
    g = grid(freq)
    coded = code().encode(np.concatenate([payload, crc16(payload, tag << 8)]))[0]
    sym = np.empty(N_SYM, int)
    sym[SYNC_ROWS] = np.tile(SYNC, 2)
    sym[DATA_ROWS] = cpm.to_tones(g, coded[interleaver()])
    return cpm.tones(g, sym)


def tx_audio(tag: int, payload: int, channel: int, cfo: float) -> np.ndarray:
    """A Hand as it goes out: channel `channel`, shifted by the ref burst's
    cfo, envelope-normalised for the sound card (arq.engine._start_tx's rule)."""
    x = modulate(tag, payload_bits(payload), channel_freq(channel))
    if cfo:
        x = hfchannel.freq_shift(x, cfo)
    return x / np.max(dsp.envelope(x))


def mix(g: cpm.Grid, y: np.ndarray, cfo: float) -> np.ndarray:
    """The window mixed down once for a cfo: cpm._energies' own factor at
    absolute sample times, so its symbol bins are bit for bit what a slice of
    this gives (cpm._energies re-mixes the audio per candidate: 427 times a
    channel here, where 7 suffice)."""
    return y * np.exp(-2j * np.pi * (g.f0 + cfo) * (np.arange(len(y)) / FS))


def bins(z: np.ndarray, start: int, T: int, m: int, n_sym: int) -> np.ndarray:
    """(n_sym, m) tone energies of the symbols from `start`, out of a mixed
    window: cpm._energies' segment, zero-padded the same way past its ends."""
    seg = z[max(0, start):start + n_sym * T]
    if start < 0:
        seg = np.concatenate([np.zeros(-start, complex), seg])
    if len(seg) < n_sym * T:
        seg = np.pad(seg, (0, n_sym * T - len(seg)))
    return np.abs(np.fft.fft(seg.reshape(n_sym, T), axis=1)[:, :m]) ** 2


def sync_shares(z: np.ndarray, start: int, T: int, m: int) -> float:
    """Both sync blocks' share of their own energy at `start`: the search's
    and the nailing's metric (a share, so a strong stretch of data can't
    outscore a sync block in a fade)."""
    s = 0.0
    for r0 in (0, len(SYNC) + N_DATA // 2):
        E_ = cpm._shares(bins(z, start + r0 * T, T, m, len(SYNC)))
        s += E_[np.arange(len(SYNC)), SYNC].sum()
    return s


def search(g: cpm.Grid, y: np.ndarray, nominal: int, rnd: dict = ROUND_FULL) -> list:
    """Top (start, cfo) candidates by the sync blocks' tone shares."""
    T = g.T
    cands = []
    for cfo in np.arange(-SEARCH_HZ, SEARCH_HZ + 1e-9, g.rate / 4):
        z = mix(g, y, cfo)
        for start in range(nominal - SEARCH_SAMPLES, nominal + SEARCH_SAMPLES + 1, T // rnd["step_div"]):
            cands.append((sync_shares(z, start, T, g.m), start, cfo))
    cands.sort(reverse=True)
    out = []
    for s, start, cfo in cands:  # distinct peaks: half a symbol or a quarter tone apart
        if all(abs(start - a) > T // 2 or abs(cfo - b) > g.rate / 4 + 1e-9 for a, b in out):
            out.append((start, cfo))
        if len(out) == rnd["top"]:
            break
    return out


def nail(g: cpm.Grid, y: np.ndarray, start: int, cfo: float, rnd: dict = ROUND_FULL):
    """A candidate -> (start, cfo, data-row energies) nailed in on T/16 x
    rate/16 steps: the search's T/8 step can land a quarter symbol off, and
    that leaks every symbol's energy across two bins: ~4 dB of the SNR
    reading, and the same leak in the soft bits, which is why nailing comes
    before the decode rather than after the CRC. Metric: the data rows'
    winning shares; the sync blocks' would do at 6 dB but picks the wrong
    quarter symbol at the -20 dB threshold (measured: 47% -> 48% decoded,
    where the data rows' give 58%). ROUND_CHEAP skips the rate/16 steps and
    nails on the search's own CFO grid."""
    T = g.T
    best = (start, cfo, bins(mix(g, y, cfo), start, T, g.m, N_SYM)[DATA_ROWS], -np.inf)
    dfs = (-g.rate / 16, 0.0, g.rate / 16) if rnd["nail_cfo"] else (0.0,)
    for df in dfs:
        z = mix(g, y, cfo + df)
        for dt in range(-(T // 2), T // 2 + 1, T // 16):
            En = bins(z, start + dt, T, g.m, N_SYM)[DATA_ROWS]
            v = float(cpm._shares(En).max(axis=1).sum())
            if v > best[3]:
                best = (start + dt, cfo + df, En, v)
    return best[0], best[1], best[2]


def receive(g: cpm.Grid, y: np.ndarray, nominal: int, tag: int, dec: polar.SCLDecoder, rnd: dict = ROUND_FULL):
    """The first candidate and list path whose CRC passes -> (payload bits,
    snr_est dB, start sample), or None. Every candidate is nailed first, and
    its data-row energies serve twice: the soft bits, and the SNR
    (cpm.measure's scale: average power in 2500 Hz)."""
    cands = search(g, y, nominal, rnd)
    if not cands:
        return None
    llr = np.empty((len(cands), E))
    ens = []
    for i, (start, cfo) in enumerate(cands):
        start, cfo, En = nail(g, y, start, cfo, rnd)
        llr[i, interleaver()] = cpm.llrs(g, En)
        ens.append((start, En))
    bits, _ = dec.decode(llr)
    for i, paths in enumerate(bits):
        for u in paths:
            if np.array_equal(crc16(u[:PAYLOAD], tag << 8), u[PAYLOAD:]):
                start, En = ens[i]
                return u[:PAYLOAD], cpm.measure(g, En, N_DATA)["snr_est"], start
    return None


def decode_window(y: np.ndarray, tag: int, channels: list[int], nominal: int, rnd: dict = ROUND_FULL) -> list:
    """Net control's round: decode every channel in the captured window.
    `nominal`: a Hand's expected start sample in y (its slot, as this host
    sent it); the search covers +-SEARCH_S of it. -> dicts with c, payload
    (63-bit int), snr (dB), dt_ms (start minus nominal).

    ponytail: a full grid search per channel, ~3 s of CPU for 40 measured on
    a desktop (the window is mixed once per cfo, not once per candidate:
    cpm._energies re-mixes, mix()/bins() here don't). This is the reference
    the tests and studies run; a station runs native/core/hand, whose round
    is ~1 s on the same machine (docs/hand-burst.md, "As shipped")."""
    dec = polar.SCLDecoder(code(), POLAR_LIST)
    out = []
    for c in channels:
        got = receive(grid(channel_freq(c)), y, nominal, tag, dec, rnd)
        if got is not None:
            bits, snr, start = got
            out.append(dict(c=c, payload=payload_int(bits), snr=float(snr),
                            dt_ms=1000.0 * (start - nominal) / FS))
    return out
