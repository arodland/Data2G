"""Burst modem: codeword payloads <-> passband audio.

Burst on air: silence | preamble | header | n_cw codewords of whole frames
(on w and w48 a header copy's frame among them) | closing pilot | silence.
Frames are SSTVAE's (1 pilot + 5 data symbols, every carrier of the
submode's band). The closing pilot is one extra pilot symbol so the last
frame is interpolated between two pilots rather than extrapolated from
one; it costs 24 ms and matters most to short bursts. A submode's frames
may be on other carriers than its preamble and header, which are its sync
band's (config.BandSpec.sync).

Sections, in file order:
- burst timing: header_layout .. burst_seconds, arithmetic on config.
- header: a (N, 16) linear code (codes_data/header_code*.npy) over
  submode | n_cw - 1 | CRC-6, ML-decoded over the valid words only; Accept
  narrows which words count.
- transmit: modulate, modulate_bits, burst_waveform, ACE.
- receive, search: _best_header tries each preamble hypothesis at +-2
  repeat alignments, reading a header at each (_read_header). Offline,
  receive() searches x itself. A streaming receiver (receiver.Receiver) calls
  find_burst, confirms the lock with pilot_coherence, finds a burst whose
  preamble faded with find_copy (from the header copy), then hands
  receive() the head or the lock.
- receive, frames: _demod_frames (sample-clock tracking from the pilots,
  timing slips), receive() up to the channel estimate and raw symbols.
- decode: data_channel, noise_var, soft_bits, burst_soft and slot_soft
  (arq/phy's, on its DD re-estimate too), snr_db, symbol_times,
  decode_received.

The receive side's results are dicts typed by Lock, HeaderRead, Received
and DataEstimate (after Burst). receive()'s Received is what receiver.Receiver
and arq/phy build on; decode_received is the plain decode of it (no CRC
masks).
Channel estimation is data2g.equalizer (burst-wide, 2-D LMMSE), not
SSTVAE's Catmull-Rom.

Native: native/core/modem/modem.hpp (this module; results as structs),
modem/timing.hpp (burst timing), generated/modem.hpp (the constants and
header codes, written by tools/gen_native_tables.py).
"""

import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np

from . import codes, config, constellation, equalizer
from .config import (
    BANDS,
    BY_INDEX,
    CLIP,
    ClipConsts,
    DATA_SYMS_PER_FRAME,
    DEMOD_BACKOFF,
    FRAME_SAMPLES,
    FS,
    HEADER_COPY_AFTER,
    HEADER_COPY_BANDS,
    LEADIN_SAMPLES,
    LEADOUT_SAMPLES,
    CW_BITS,
    M,
    max_codewords,
    NCP,
    NSYM,
    PREAMBLE_CP,
    PROFILE_TAG,
    PROTOCOL_VERSION,
    RS,
    SC_BANDS,
    SNR_REF_BW_HZ,
    SUBMODES,
    SYMBOL_NOISE,
    SYMS_PER_FRAME,
    SubmodeSpec,
)
from .waveform import dsp, ofdm
from .waveform import sync as _sync
from .waveform.sync import SyncError

# The receiver's constants are mirrored into native/core/generated/modem.hpp
# (tools/gen_native_tables.py modem_hpp names them): regenerate after a change.
REF_REPEATS = 4
# The header is read before the delay profile is known, at acquisition's
# timing: window centred in the CP, so timing may be off by +-16 samples
# either way. (SSTVAE's DEMOD_BACKOFF, 6, allows only 6 late; with 4
# carriers the template's main lobe is ~40 samples wide.)
HEADER_BACKOFF = NCP // 2
# Minimum normalized ML correlation for a header to count. The ML search
# runs over valid words only (decode_header), so every decode parses and
# this floor is all that rejects a header read off noise, whose best valid
# word scores 0.30-0.36 (99.9%, 16 submodes per band). Narrow bands: 0.30,
# far below their true headers at their sync floors (genie reads at -12 dB
# n4 AWGN: none lost). w48: its detector fires on wide-band bursts, where
# its headers read off the wide signal outscored true wide headers near
# -8 dB, and its own submodes never run below 0 dB. w: 0.25 (was none).
# Without one, data audio whose preamble was lost (a fade, or a buffer
# joined mid-burst) detected as a burst 40/40 times, with a random header
# whose length stretched the ARQ's deadlines; a steady tone did too
# (SSTVAE's 1b05ba4 found the same false lock). Scores (runs/w_header_gate.npy):
#   floor   data false locks   tone false locks   true headers lost
#   0.25         52%                0.8%               0.3%
#   0.28         11%                0%                 1.3%
#   0.33          0.7%              0%                 5.1%
# 0.33 cost the w band 1.75 dB of AWGN sync floor and up to 3 dB on fading
# (w carries the robust ACKs); 0.25 cost nothing measurable, the supersede
# search to make its false locks cheap. On air it wasn't: behind a 500 Hz
# receive filter the w detector fired on every window and 0.25-0.32 false
# headers held BUSY. With the header copy (two copies combined), 0.33:
#   false locks  500 Hz capture 225 -> 5 of 360 windows; data without its
#                preamble 476 -> 13 of 900
#   true lost    ack-4f AWGN -8 +0.7 points, MPP -4 +1.8, MPD -3 +1.5,
#                MPG -3 +3.3; qpsk-r1/5 MPP -2 +0.3 (and most wrong headers go)
STREAM_COMMIT_SCORE = 0.5  # a header this good commits even while another band's is still arriving
# a streaming receiver commits on the first header copy alone at this
# score; under it, it waits for the second copy (wrong words read 0.25-0.35)
COPY_COMMIT_SCORE = 0.45
# w48: 0.26 since its header went to 4 symbols (384 bits): its noise reads
# top out at 0.248 (3000 measured), its reads off w bursts at 0.243 (-8..+30
# dB); true w48 headers score 0.35 median at -6 dB (0.30 had cut ~20% there).
HEADER_MIN_SCORE = {"w": 0.33, "n10": 0.30, "w48": 0.26}
# FM's f55 and f108 (FM profile): set below with the pilot constants
# Handicap on the header score of every hypothesis but acquisition's own
# (its runner-ups, and the +-1, +-2 repeat alignments): near -8 dB wide a
# true header scores ~0.26 and noise read at some other hypothesis beat
# it. Wide AWGN -8 dB end to end: fail 6.25% -> 3.5% (0.08: 4.0%); no
# change on mpd, where the other hypotheses do the rescuing.
ALT_PENALTY = 0.04
# bands that carry a header, i.e. whose preambles the receiver looks for
SYNC_BANDS = [name for name, b in BANDS.items() if not b.sync]


@dataclass
class Burst:
    submode: SubmodeSpec
    payloads: list[bytes]
    crc_ok: list[bool]
    freq_offset: float
    preamble_start: int
    snr_db: float  # pilot-based, in the SNR_REF_BW_HZ convention
    soft: np.ndarray | None = None  # (n_cw, coded_bits) soft bits, mapping order: codes.combine for resends


# --- result types -------------------------------------------------------------
# The receive side's results are plain dicts (native returns the same keys:
# modem.hpp's structs, converted in native/bindings/bind_modem.cpp); these
# say what they hold. Positions are sample indices into the caller's buffer.
# "band" is the sync band in a Lock or HeaderRead, the data band in a
# Received and DataEstimate (they differ for n4, which syncs on n10).

class CopyRef(TypedDict):
    word: int  # the 16-bit header word the copy read
    pc: int  # the copy frame's first sample


class Lock(TypedDict):
    """find_burst's or find_copy's: where a burst is (it may run past the
    buffer). Native modem::Lock."""

    spec: SubmodeSpec
    n_cw: int
    start: int  # the preamble's first sample
    score: float  # the header's normalized ML correlation
    band: str  # the sync band
    end: int  # one past the closing pilot
    p0: int  # the first frame's first sample
    cfo: float  # Hz
    copy: NotRequired[CopyRef]  # set by find_copy only
    connect_hash: NotRequired[int]  # set for CONNECT_HEADER words (docs/pending-header.md)


class HeaderRead(TypedDict):
    """_read_header's (and _copy_header's): a header read at one start.
    Native modem::HeaderRead."""

    word: int
    hdr: tuple[SubmodeSpec, int] | None  # (spec, n_cw); None under the score floor
    score: float
    pending_copy: bool  # the header copy is not yet in the buffer
    y: np.ndarray  # (header_syms, nc) the header's data symbols
    y_all: np.ndarray  # every header symbol, pilots among them (header_layout)
    h_pre: np.ndarray  # (nc,) channel from the preamble's last REF_REPEATS repeats
    h_first: np.ndarray  # (nc,) channel at the first frame pilot
    p0: int  # the first frame's first sample
    start: int  # the preamble's first sample
    band: str  # the sync band
    n0_pre: float  # noise from the preamble (equalizer.preamble_noise)
    n0_pre_k: np.ndarray  # the same per carrier


class DataEstimate(equalizer.ChannelEstimate):
    """data_channel's: the pilots' estimate scaled to the channel the data
    see (h and mse already include gain). Native modem::DataEstimate."""

    clip_ratio: float  # clip noise over |h|^2 (noise_var)
    gain: float  # data over pilot amplitude (Bussgang; config.CLIP)
    band: str  # the data band


class Received(TypedDict):
    """receive()'s: a burst synchronized and equalized, up to soft bits.
    Native modem::Received. cpm.receive's dict is another shape, marked
    family="cpm"; arq/phy adds "_soft" (its soft bits, cached)."""

    spec: SubmodeSpec
    n_cw: int
    raw: np.ndarray  # (n_f, 6, nc) data frames, the header copy's dropped; [:, 0] each frame's pilot
    est: DataEstimate  # h, mse (n_f, 5, nc): aligned with raw[:, 1:]
    acq: _sync.Acquisition
    band: str  # the data band
    hp: np.ndarray  # (frames on air + 1, nc) pilots, the copy frame's and the closing one included
    kc: int | None  # the header copy's frame among those on air (copy_frame)
    cfo: float  # Hz, the residual included
    p0: int  # the first frame's first sample
    shift: int  # samples the demod window was moved (equalizer.window_shift)
    steps: np.ndarray  # (frames on air + 1,) the drift tracker's step total per frame
    slips: np.ndarray  # (frames on air + 1,) each frame's timing jump (_find_slip)
    phi_ref: float  # the preamble channel's phase step across carriers (_bin_phase_step)
    support: tuple[int, int]  # the estimate's delay support (noise-gated), after the shift
    seen_support: tuple[int, int]  # the same ungated: the window shift's, the reported delay's
    preamble_start: int
    score: float  # the header's


# --- burst timing (native: modem/timing.hpp) ----------------------------------

def _hosts(band: str) -> bool:
    """Whether the header closes with a pilot of its own: other bands' frames
    follow it, or frames without a pilot symbol (config.SC_BANDS)."""
    return band in SC_BANDS or any(b.sync == band for b in BANDS.values())


# Scattered pilots on HF 6-symbol OFDM bands (config.SC_BANDS):
# - Wide bands (w, w48, nc >= 24): frame-staggered intra-frame combs (fk3d2: K=3, symbols 0/2/4, comb fa % 3)
#   estimated by the Slepian-mode separable 2-D LMMSE.
# - Narrow bands (n10, n4): one pilot per carrier per frame (gold: SC_STRIDES).
# After each ACE clip pass, scattered pilots are restored onto their constellation points (SC_RESTORE_ACE = 1.0);
# plain (non-ACE) modes keep pilots as clipped (0.0) to preserve PEP.
SC_RESTORE_ACE = 1.0


def sc_spec(spec, n_f: int) -> bool:
    """Whether a burst of `n_f` data frames of this submode carries scattered pilots (the header says both)."""
    return spec.band in SC_BANDS and n_f > 0


SC_SPREADS_SLOW = (0.1, 0.25, 0.5, 1.0, 1.5)
SC_LOO_TOL = 0.1
SC_COPY_SUBBINS = 5


def sc_cells(nc: int) -> tuple:
    """A scattered-pilot frame's (6, nc) cells, flat: (the pilots', their
    carriers, the data cells' in sending order, per carrier its pilot's).
    Used on narrow bands (n10, n4) whose gold mask repeats every frame."""
    return _sc_cells(nc)


def _sc_intra_spec(nc: int) -> tuple[int, tuple[int, ...], int] | None:
    """Intra-frame comb layout on wide bands (nc >= 24, fk3d2): (carrier stride K,
    pilot symbols in a frame, frame period in combs); n10/n4 keep gold."""
    if nc < 24:
        return None
    return 3, (0, 2, 4), 3


def sc_period(nc: int) -> int:
    """Period in data frames of the scattered-pilot cell mask."""
    sp = _sc_intra_spec(nc)
    return sp[2] if sp is not None else 1


@lru_cache(maxsize=None)
def _sc_frame_cells_cached(nc: int, f_mod: int) -> tuple:
    sp = _sc_intra_spec(nc)
    if sp is None:
        P, PK, Dc, _ = _sc_cells(nc)
        return P, PK, Dc
    K, syms, n_combs = sp
    N = SYMS_PER_FRAME * nc
    sym, k = np.divmod(np.arange(N), nc)
    c = f_mod % n_combs
    pil = np.isin(sym, syms) & (k % K == c)
    P = np.flatnonzero(pil)
    assert len(P) == nc, (nc, len(P))
    return P, k[P], np.flatnonzero(~pil)


def sc_frame_cells(nc: int, f: int = 0) -> tuple:
    """Data frame `f`'s (P, PK, Dc): `nc` pilot indices, their carriers, and `5 * nc` data indices."""
    return _sc_frame_cells_cached(nc, f % sc_period(nc))


SC_STRIDES = {24: 49, 48: 97, 10: 37, 4: 17}


@lru_cache(maxsize=None)
def _sc_cells(nc: int) -> tuple:
    """Every gold-stride layout is the first nc terms of cell i * g mod 6 nc."""
    N = SYMS_PER_FRAME * nc
    _, k = np.divmod(np.arange(N), nc)
    g = SC_STRIDES[nc]
    pil = np.zeros(N, dtype=bool)
    pil[np.arange(nc) * g % N] = True
    P = np.flatnonzero(pil)
    assert np.array_equal(np.sort(k[P]), np.arange(nc)), "a pilot layout must give each carrier one pilot a frame"
    own = np.empty(nc, dtype=int)
    own[k[P]] = P
    return P, k[P], np.flatnonzero(~pil), own


def copy_open(b) -> np.ndarray:
    """The symbol opening a scattered-pilot band's header-copy frame: see copy_close."""
    return np.conj(b.pilot)


def copy_close(b) -> np.ndarray:
    """The symbol opening and closing a scattered-pilot band's header-copy frame (this, the 4 header symbols,
    this): the pilot's conjugate (the same envelope), not the pilot. With the pilot at both ends the frame was the
    first header's own shape (the preamble's last repeat, the header, the header pilot), so a preamble hypothesis
    two frames and a symbol early read the real header as its copy (0.96 on a clean burst; with conj(pilot) at
    the close alone still 0.62, the open end's channel being right) and held the streaming receiver past the real
    burst (tests/test_receiver.py false lock, seed 0). conj(pilot) correlates with the pilot at about
    1 / sqrt(nc)."""
    return np.conj(b.pilot)


def _sc_pil(band: str, n_air: int, kc: int | None) -> np.ndarray | None:
    """Per on-air frame (and the closing pilot) and pilot slot 0..nc-1, (symbol, carrier) when intra-frame comb,
    or per-carrier symbol (n_air + 1, nc) for 1-pilot-per-carrier layouts; None on pilot-row bands."""
    if band not in SC_BANDS:
        return None
    nc = BANDS[band].nc
    if _sc_intra_spec(nc) is not None:
        pil = np.zeros((n_air + 1, nc, 2), dtype=int)
        pil[:, :, 1] = np.arange(nc)
        for fa in range(n_air):
            if fa != kc:
                P, PK, _ = sc_frame_cells(nc, fa)
                pil[fa, :, 0], pil[fa, :, 1] = P // nc, PK
        return pil
    pil = np.tile(sc_cells(nc)[3] // nc, (n_air + 1, 1))
    pil[n_air] = 0
    if kc is not None:
        pil[kc] = 0
    return pil


def header_layout(band: str) -> np.ndarray:
    """Per on-air header symbol: True for a pilot. A pilot precedes every
    5 header symbols after the first 5 (the preamble fronts those), the
    data frames' own structure, so a long header can follow the channel:
    n4's 24 header symbols span 576 ms, and with pilots only at its ends
    they lost 69-75 of 200 bursts on mpd at -5..-3 dB, and a few even on
    AWGN (residual CFO rotating across the span)."""
    out = []
    for i in range(BANDS[band].header_syms):
        if i and i % DATA_SYMS_PER_FRAME == 0:
            out.append(True)
        out.append(False)
    if _hosts(band):
        # the first frame pilot may be on another band's carriers, so the
        # header closes with a pilot of its own
        out.append(True)
    return np.array(out)


def header_samples(band: str) -> int:
    return len(header_layout(band)) * NSYM


def copy_frame(band: str, n_f: int) -> int | None:
    """Which on-air frame carries the header copy (config.HEADER_COPY_*) of
    a burst of n_f data frames on sync band `band`; None: no copy."""
    return min(HEADER_COPY_AFTER, n_f) if band in HEADER_COPY_BANDS and n_f >= COPY_MIN_FRAMES else None


COPY_MIN_FRAMES = 0  # a study hook (scripts/burst_trim.py patches it): shorter bursts carry no copy


def frames_on_air(spec: SubmodeSpec, n_cw: int) -> int:
    n_f = n_cw * spec.frames_per_cw
    return n_f + (copy_frame(spec.sync_band, n_f) is not None)


def burst_end(p0: int, spec: SubmodeSpec, n_cw: int) -> int:
    """One past the closing pilot of a burst whose first frame starts at p0."""
    return p0 + (frames_on_air(spec, n_cw) * SYMS_PER_FRAME + 1) * NSYM


def head_samples(band: str) -> int:
    """From a preamble's first sample, the audio that holds every header copy
    and the pilot after the last (a header decision needs no more)."""
    n = BANDS[band].preamble_samples + header_samples(band) + NSYM
    return n + (HEADER_COPY_AFTER + 1) * FRAME_SAMPLES if band in HEADER_COPY_BANDS else n


def receive_head(band: str) -> int:
    """From the first sample of a preamble a streaming receiver found, the
    audio to pass as receive(head=...): head_samples, plus room for
    _best_header's reads up to two repeats late (2 M) and a symbol and a
    repeat to spare."""
    return head_samples(band) + NSYM + 3 * M


def burst_seconds(spec: SubmodeSpec, n_cw: int) -> float:
    """On-air length of an n_cw-codeword burst, lead-in/out silence included."""
    sb = BANDS[spec.sync_band]
    n = (LEADIN_SAMPLES + sb.preamble_samples + header_samples(spec.sync_band)
         + (frames_on_air(spec, n_cw) * SYMS_PER_FRAME + 1) * NSYM + LEADOUT_SAMPLES)
    return n / FS


# --- header -----------------------------------------------------------------
# Per band: 16 bits coded to 2 * nc * header_syms bits on QPSK, decoded by
# exact ML (config: header). Codes from scripts/design_header.py.

QPSK = constellation.gray_qam(2)


@lru_cache(maxsize=None)
def header_code(band: str) -> np.ndarray:
    name = "header_code.npy" if band == "w" else f"header_code_{band}.npy"
    g = np.load(Path(__file__).parent / "codes_data" / name)
    b = BANDS[band]
    assert g.shape == (16, 2 * b.nc * b.header_syms), (band, g.shape)
    return g


def crc6(v: int) -> int:
    """CRC-6 (x^6 + x + 1) over 10 bits, register seeded with the version
    and the profile (config.PROFILE_TAG).
    On air in every header word, v << 6 | crc6(v): the OFDM header's and
    the CPM header's (cpm.header_word)."""
    reg = (PROTOCOL_VERSION ^ PROFILE_TAG) & 0x3F
    for i in range(9, -1, -1):
        fb = ((reg >> 5) & 1) ^ ((v >> i) & 1)
        reg = ((reg << 1) & 0x3F) ^ (0x3 if fb else 0)
    return reg


def _word_bits(word: int) -> np.ndarray:
    return (word >> np.arange(15, -1, -1)) & 1


def _split(v: int, band: str) -> tuple[int, int]:
    """A header's 10-bit value -> (submode index, n_cw) on this sync band."""
    b = CW_BITS.get(band, 6)
    return v >> b, (v & ((1 << b) - 1)) + 1


def header_bits(submode: int, n_cw: int, band: str = "w") -> np.ndarray:
    """-> coded header bits for the band. Word: submode (4) | n_cw - 1 (6)
    | CRC-6, 5 | 5 on n10 (config.CW_BITS). Was 8 + CRC-4 until measured:
    tried at 5 alignments and on every band whose detector fires, a 4-bit
    check let wrong headers through after the true one failed (11 of 200
    wide ACKs at -6 dB on mpd read as a burst of hundreds of codewords)."""
    b = CW_BITS.get(band, 6)
    if not 1 <= n_cw <= 1 << b:
        raise ValueError(f"1..{1 << b} codewords per burst on {band}")
    if not 0 <= submode < 1 << (10 - b):
        raise ValueError(f"submode index {submode} does not fit {band}'s header")
    v = (submode << b) | (n_cw - 1)
    return (_word_bits((v << 6) | crc6(v)) @ header_code(band)) % 2


def _signs(words: np.ndarray, band: str) -> np.ndarray:
    """(W,) 16-bit words -> (W, N) +-1 codewords."""
    msgs = (words[:, None] >> np.arange(15, -1, -1)) & 1
    return (1 - 2 * ((msgs @ header_code(band)) % 2)).astype(np.float32)


@lru_cache(maxsize=None)
def _header_signs(band: str) -> np.ndarray:
    """(65536, N) +-1 for every message, message index = the 16-bit word."""
    return _signs(np.arange(2**16), band)


@dataclass(frozen=True)
class Accept:
    """Which headers a receiver takes, beyond validity: some submodes
    only, a cap on codewords per burst, a score floor over
    HEADER_MIN_SCORE. Fewer valid words also means noise matches them
    less well (Gaussian noise's best-of-896 scores 0.364 at 99.99%,
    best-of-64 0.325; true wide headers at -6 dB AWGN score >= 0.385 in
    99% of bursts, at -7.5 dB >= 0.29)."""

    max_cw: tuple  # ((submode name, most codewords per burst), ...)
    min_score: float = 0.0

    @classmethod
    def of(cls, names=None, max_secs: float | None = None, min_score: float = 0.0) -> "Accept":
        """`names` (default: every submode), each capped at the codewords
        a burst of at most `max_secs` on air holds (a submode that can't
        fit one is left out)."""
        out = []
        for n in names or SUBMODES:
            s = SUBMODES[n]
            cw = max_codewords(s.sync_band)
            if max_secs is not None:
                fixed = BANDS[s.sync_band].preamble_samples + header_samples(s.sync_band) + NSYM
                fixed += FRAME_SAMPLES * (s.sync_band in HEADER_COPY_BANDS)
                cw = min(cw, int((max_secs * FS - fixed) // (s.frames_per_cw * FRAME_SAMPLES)))
            if cw >= 1:
                out.append((n, cw))
        return cls(tuple(out), min_score)

    @property
    def bands(self) -> list[str]:
        return sorted({SUBMODES[n].sync_band for n, _ in self.max_cw})


# Dedicated CONNECT header words (docs/pending-header.md): spare pseudo-submode
# indices whose n_cw field carries a hash of the destination call (the burst is
# always 1 codeword). Active from HF protocol 16 / FM protocol 6.
CONNECT_HEADER: dict[tuple[str, int], tuple[str, int]] = {
    ("w", 15): ("qpsk-r1/5", 6),
    ("n10", 31): ("n10-qpsk-r1/3", 5),
    **({("n10", 30): ("n10-qpsk-r1/5", 5)} if config.PROFILE == "fm" else {}),
}
_CONNECT_BY_MODE = {mode: (band, idx, bits) for (band, idx), (mode, bits) in CONNECT_HEADER.items()}


def _connect_headers_active() -> bool:
    return PROTOCOL_VERSION >= (6 if config.PROFILE == "fm" else 16)


def connect_header_for(name: str) -> tuple[str, int, int] | None:
    """(sync_band, pseudo_index, hash_bits) if `name` carries a dedicated CONNECT header, else None."""
    return _CONNECT_BY_MODE.get(name) if _connect_headers_active() else None


def connect_hash_of(word: int, band: str) -> int | None:
    """Callee-callsign hash carried by OFDM header word `word` on `band`, or None for a regular burst."""
    if not _connect_headers_active():
        return None
    sub, n_val = _split(word >> 6, band)
    return n_val - 1 if (band, sub) in CONNECT_HEADER else None


@lru_cache(maxsize=None)
def _valid_words(band: str, accept: Accept | None = None) -> np.ndarray:
    """Words with a right CRC-6 and a submode defined on the band (and
    taken by `accept`), including CONNECT_HEADER pseudo-indices."""
    v = np.arange(2**10)
    active = _connect_headers_active()
    if accept is None:
        idx = {i for b, i in BY_INDEX if b == band}
        if active:
            idx |= {i for b, i in CONNECT_HEADER if b == band}
        v = v[[_split(int(x), band)[0] in idx for x in v]]
    else:
        lim = {SUBMODES[n].index: cw for n, cw in accept.max_cw if SUBMODES[n].sync_band == band}
        if active:
            for (b, pi), (m, bits) in CONNECT_HEADER.items():
                if b == band and SUBMODES[m].index in lim and lim[SUBMODES[m].index] >= 1:
                    lim[pi] = 1 << bits
        v = v[[(i := _split(int(x), band))[0] in lim and i[1] <= lim[i[0]] for x in v]]
    return (v << 6) | np.array([crc6(int(x)) for x in v], dtype=np.int64).reshape(-1)


@lru_cache(maxsize=None)
def _valid_signs(band: str, accept: Accept | None = None) -> np.ndarray:
    # the valid words only: ~1/70 of _header_signs' table, built on the
    # receiver's first header
    return _signs(_valid_words(band, accept), band)


def decode_header(soft: np.ndarray, band: str = "w",
                  accept: Accept | None = None) -> tuple[int, tuple[SubmodeSpec, int], float]:
    """Soft header bits (positive = 0) -> (ML word among the valid ones,
    parsed, normalized correlation of the winner, ~0..1; the caller holds
    it to HEADER_MIN_SCORE). ML over valid words rather than ML over all
    65536 then the CRC: genie-sync header failures at -8 dB AWGN, wide
    band, 14% -> 2.3% (scripts/header_study.py)."""
    words = _valid_words(band, accept)
    corr = _valid_signs(band, accept) @ soft.astype(np.float32)
    i = int(np.argmax(corr))
    word = int(words[i])
    score = float(corr[i] / (np.sqrt(np.sum(soft**2) * len(soft)) + 1e-12))
    sub, n_cw = _split(word >> 6, band)
    if _connect_headers_active() and (band, sub) in CONNECT_HEADER:
        return word, (SUBMODES[CONNECT_HEADER[(band, sub)][0]], 1), score
    return word, (BY_INDEX[(band, sub)], n_cw), score


# --- transmit ---------------------------------------------------------------

# The station's radio path on the FM profile (config.FM_*): "mic" (the default:
# its quieter top modes are safe on a flat radio, where flat's settings would
# overdrive a mic radio's limiter) or "flat" (a 9600 port). Set by the station
# (data2g-host --tx-path); what the transmitter sends, never what it receives.
TX_PATHS = ("mic", "flat")
TX_PATH = "mic"


@lru_cache(maxsize=None)
def tx_tilt_fir(band: str) -> np.ndarray:
    """The flat path's TX tilt (config.FM_TX_TILT_DB_OCT): linear-phase FIR,
    amplitude (f / fc)^(dB per octave / 6.02) over the band (fc its geometric
    centre), held flat outside it (scripts/fm_tilt_study.py's, measured with it)."""
    from scipy import signal

    b = BANDS[band]
    lo = config.CARRIER0 + RS * b.k0
    hi = lo + RS * (b.nc - 1)
    f = np.linspace(0, FS / 2, 513)
    g = (np.clip(f, lo, hi) / np.sqrt(lo * hi)) ** (config.FM_TX_TILT_DB_OCT / (20 * np.log10(2)))
    return signal.firwin2(255, f, g, fs=FS)


def modulate(payloads: list[bytes], submode: str | SubmodeSpec, rvs=None, tx_path: str | None = None,
             connect_hash: int | None = None) -> np.ndarray:
    """Codeword payloads (each exactly codes.payload_bytes long) -> unit-RMS
    audio (FM, mic path: the top modes quieter, config.FM_MIC_GAIN). `rvs`:
    each codeword's redundancy version (codes.rv_positions), all 0 by
    default. `tx_path`: FM's radio path (default TX_PATH)."""
    spec = SUBMODES[submode] if isinstance(submode, str) else submode
    if not 1 <= len(payloads) <= max_codewords(spec.sync_band):
        raise ValueError(f"1..{max_codewords(spec.sync_band)} codewords per burst, got {len(payloads)}")
    bits = np.stack([codes.encode(spec, p, rv, index=i) for i, (p, rv) in enumerate(zip(payloads, rvs or [0] * len(payloads)))])
    return modulate_bits(codes.spread(bits, spec.bits_per_cu, codes.spread_group(spec)), spec, tx_path,
                         connect_hash=connect_hash)


def modulate_bits(bits: np.ndarray, spec: SubmodeSpec, tx_path: str | None = None,
                  connect_hash: int | None = None) -> np.ndarray:
    """Coded bits in burst order (whole codewords, after `codes.spread`)
    -> unit-RMS audio. The codec-free half of `modulate`, for
    measurements that send raw bits."""
    b = ofdm.band(spec.band)
    pts = constellation.load(spec.constellation)
    data = constellation.modulate(bits, pts).reshape(-1, DATA_SYMS_PER_FRAME, b.nc)
    x = burst_waveform(data, spec, connect_hash=connect_hash)
    project = None
    closing = spec.ace
    if spec.ace:
        m = constellation.bits_per_symbol(pts)
        idx = (bits.reshape(-1, m) @ (1 << np.arange(m - 1, -1, -1))).reshape(data.shape)
        project = ace_projector(spec, data, constellation.ace_dirs(spec.constellation)[idx])
    path = TX_PATH if tx_path is None else tx_path
    if path not in TX_PATHS:
        raise ValueError(f"tx_path: {path!r}, not one of {TX_PATHS}")
    fm = config.PROFILE == "fm"
    m = constellation.bits_per_symbol(pts)
    # FM flat path: sent unclipped from 16-QAM up (a headroom no burst reaches)
    headroom = 40.0 if fm and path == "flat" and m >= config.FM_FLAT_NOCLIP_BITS else spec.headroom
    # the sync band's filter: it contains the data band (BandSpec.sync)
    y = dsp.tx_condition(
        x, headroom, b.spec.clip_overshoot,
        active=slice(LEADIN_SAMPLES, len(x) - LEADOUT_SAMPLES), bandpass=BANDS[spec.sync_band].tx_bandpass,
        project=project, closing=closing,
    )
    if not fm:
        return y
    if path == "mic":
        return y * config.FM_MIC_GAIN[m]
    act = slice(LEADIN_SAMPLES, len(y) - LEADOUT_SAMPLES)
    t = np.convolve(y, tx_tilt_fir(spec.sync_band), "same")
    return t * np.sqrt(np.mean(y[act] ** 2) / np.mean(t[act] ** 2))  # rms kept (the tilt studies')


def ace_cells(spec: SubmodeSpec, n_f: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample indices of a burst's data symbols (n_f data frames): (their
    useful windows (n_f * 5, M), the whole symbols (n_f * 5, NSYM))."""
    kc = copy_frame(spec.sync_band, n_f)
    frames = [f for f in range(n_f + (kc is not None)) if f != kc]
    j = np.array([f * SYMS_PER_FRAME + s for f in frames for s in range(not sc_spec(spec, n_f), SYMS_PER_FRAME)])
    f0 = LEADIN_SAMPLES + BANDS[spec.sync_band].preamble_samples + header_samples(spec.sync_band)
    full = f0 + j[:, None] * NSYM + np.arange(NSYM)
    return full[:, NCP:], full


def ace_projector(spec: SubmodeSpec, data: np.ndarray, dirs: np.ndarray):
    """Active constellation extension (constellation.ace_dirs), for
    tx_condition: demodulate each data cell of the clipped burst, move it
    into its point's allowed region (the point scaled by the carrier's
    clip gain), and add the difference back as those carriers."""
    b = ofdm.band(spec.band)
    win, full = ace_cells(spec, len(data))
    X, D = data.reshape(-1, b.nc), dirs.reshape(-1, b.nc, 2)
    hold = None
    if sc_spec(spec, len(data)):  # the pilots among the cells
        n_f, per = len(data), sc_period(b.nc)
        kc = copy_frame(spec.sync_band, n_f)
        norm = np.array([f for f in range(n_f + (kc is not None)) if f != kc])
        X6 = np.empty((n_f, SYMS_PER_FRAME * b.nc), dtype=np.complex128)
        D6 = np.zeros(X6.shape + (2,), dtype=np.complex128)
        hold = np.zeros(X6.shape, dtype=bool)
        df, dr = data.reshape(n_f, -1), dirs.reshape(n_f, -1, 2)
        for f_ in range(per):
            idx = np.flatnonzero(norm % per == f_)
            if len(idx):
                P, PK, Dc = sc_frame_cells(b.nc, f_)
                X6[idx[:, None], P], X6[idx[:, None], Dc] = b.pilot[PK], df[idx]
                D6[idx[:, None], Dc], hold[idx[:, None], P] = dr[idx], True
        X, D, hold = X6.reshape(-1, b.nc), D6.reshape(-1, b.nc, 2), hold.reshape(-1, b.nc)
    dem = b.mod[NCP:].conj()

    def project(x):
        got = (2.0 / M) * (x[win] @ dem)
        # per carrier: the TX bandpass shapes the band edges, pilots included
        g = np.sum(X.conj() * got, axis=0).real / np.sum(np.abs(X) ** 2, axis=0)
        x = x.copy()
        fix = constellation.ace_project(got, g * X, D) - got if spec.ace else np.zeros_like(got)
        if hold is not None and spec.ace:
            fix[hold] = SC_RESTORE_ACE * (g * X - got)[hold]
        x[full] += np.real(fix @ b.mod.T)
        return x

    return project


def burst_waveform(data: np.ndarray, spec: SubmodeSpec, connect_hash: int | None = None) -> np.ndarray:
    """(n_f, 5, nc) data symbols -> the unclipped burst waveform."""
    b = ofdm.band(spec.band)
    n_f = len(data)
    n_cw = n_f // spec.frames_per_cw
    sb = ofdm.band(spec.sync_band)
    if connect_hash is not None:
        ch = connect_header_for(spec.name)
        if ch is None or n_cw != 1 or not 0 <= connect_hash < (1 << ch[2]):
            raise ValueError(f"connect_hash={connect_hash} invalid for {spec.name} x{n_cw}")
        hbits = header_bits(ch[1], connect_hash + 1, spec.sync_band)
    else:
        hbits = header_bits(spec.index, n_cw, spec.sync_band)
    hdr = constellation.modulate(hbits, QPSK)
    kc = copy_frame(spec.sync_band, n_f)
    if sc_spec(spec, n_f):
        per, df = sc_period(b.nc), data.reshape(n_f, -1)
        norm = np.array([f for f in range(n_f + (kc is not None)) if f != kc])
        fr = np.empty((n_f, SYMS_PER_FRAME * b.nc), dtype=np.complex128)
        for f_ in range(per):
            idx = np.flatnonzero(norm % per == f_)
            if len(idx):
                P, PK, Dc = sc_frame_cells(b.nc, f_)
                fr[idx[:, None], P], fr[idx[:, None], Dc] = b.pilot[PK], df[idx]
        fr = fr.reshape(n_f, SYMS_PER_FRAME, b.nc)
        if kc is not None:  # the header copy's frame: copy_close, the 4 header symbols, copy_close
            fr = np.insert(fr, kc, np.concatenate([copy_open(b)[None], hdr.reshape(-1, b.nc), copy_close(b)[None]]), axis=0)
        syms = np.concatenate([fr.reshape(-1, b.nc), b.pilot[None]])
    else:
        if kc is not None:  # the header copy: a frame of its own (data band = sync band here)
            h = hdr.reshape(sb.spec.header_syms, sb.nc)
            # (a scattered-pilot band's copy frame ends on a pilot whatever the submode: _copy_llr)
            row = np.concatenate([h, copy_close(sb)[None]]) if spec.band in SC_BANDS else np.resize(h, (DATA_SYMS_PER_FRAME, sb.nc))
            data = np.insert(data, kc, row, axis=0)
            n_f += 1
        syms = np.empty((n_f * SYMS_PER_FRAME + 1, b.nc), dtype=np.complex128)
        syms[::SYMS_PER_FRAME] = b.pilot  # frame pilots and the closing pilot
        syms[:-1].reshape(n_f, SYMS_PER_FRAME, b.nc)[:, 1:] = data
        if kc is not None and spec.band in SC_BANDS:
            syms[kc * SYMS_PER_FRAME] = copy_open(b)  # a scattered-pilot band's copy frame, whatever the submode
    layout = header_layout(spec.sync_band)
    hsyms = np.empty((len(layout), sb.nc), dtype=np.complex128)
    hsyms[layout] = sb.pilot
    hsyms[~layout] = hdr.reshape(sb.spec.header_syms, sb.nc)
    return np.concatenate([
        np.zeros(LEADIN_SAMPLES),
        sb.preamble_waveform(),
        sb.modulate_symbols(hsyms),
        b.modulate_symbols(syms),
        np.zeros(LEADOUT_SAMPLES),
    ])


# --- receive ----------------------------------------------------------------

def _bin_phase_step(h: np.ndarray) -> float:
    return float(np.angle(np.sum(h[1:] * np.conj(h[:-1]))))


# in samples, set in time (40 and 240 at 8 kHz): a slip is a sound-card buffer, whatever the rate
SLIP_MIN = FS // 200  # 5 ms: under it, a path fading in or out (CCIR mpd's 4 ms) looks like one
SLIP_MAX = 3 * FS // 100  # 30 ms: a 1024-frame buffer at 48 kHz is 21 ms
SLIP_TRIGGER = 0.5  # a pilot this unlike the last one's prompts a search
SLIP_ACCEPT = 0.45  # each pilot after a slip, this like the last one at the new offset (and twice as like as before)


def _coherence(h: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """|<ref, h>| / (|ref| |h|) along h's last axis: 1 for the same channel
    seen again (common phase aside), near 0 once a timing jump tilts it."""
    return np.abs(h @ np.conj(ref)) / (np.linalg.norm(h, axis=-1) * np.linalg.norm(ref) + 1e-30)


def _find_slip(z: np.ndarray, p: int, n_left: int, prev: np.ndarray, b) -> int:
    """A jump in the audio's timing just before the frame at p (samples
    inserted: positive; dropped: negative), or 0. A sender's playback
    underrun inserts a buffer's worth of silence (recordings/20261013-234857:
    85 samples, 512 at 48 kHz, between frames 2 and 3); the drift tracker
    moves 2 samples a frame at most, and the burst's channel estimate
    (whole-burst support, CFO) is lost from there on.

    The pilot at p and the next ones (n_left: up to two more, the closing
    pilot counting) are tried at every offset within SLIP_MAX against the
    previous frame's pilot `prev`; the best joint offset is taken only if
    each of them is twice as like `prev` there as at p, and SLIP_ACCEPT like
    it (simulated: no false slip in 1200 bursts, AWGN to mpd, -3 to 20 dB)."""
    hf = b.demod_window(z, p + NCP, DEMOD_BACKOFF) / b.pilot
    if _coherence(hf, prev) >= SLIP_TRIGGER:
        return 0
    lo = max(-SLIP_MAX, -p)
    offs = np.arange(lo, SLIP_MAX + 1)
    at, base = [], []
    for j in range(1 + min(2, n_left)):
        a = p + j * FRAME_SAMPLES + NCP - DEMOD_BACKOFF
        seg = z[a + lo : a + SLIP_MAX + M]
        seg = np.pad(seg, (0, len(offs) - 1 + M - len(seg)))
        h = (2.0 / M) * (np.lib.stride_tricks.sliding_window_view(seg, M) @ b.demod.T) / b.pilot  # every offset's pilot
        at.append(_coherence(h, prev))
        base.append(at[-1][-lo])
    score = np.sum(at, axis=0)
    score[np.abs(offs) < SLIP_MIN] = -np.inf
    k = int(np.argmax(score))
    ok = all(c[k] >= max(2 * c0, SLIP_ACCEPT) for c, c0 in zip(at, base))
    return int(offs[k]) if ok else 0


def _demod_frames(
    z: np.ndarray, p: int, n_f: int, shift: int, phi_ref: float,
    steps_in: np.ndarray | None = None, band: str = "w", slips_in: np.ndarray | None = None,
    pil: np.ndarray | None = None, close_at: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Demodulate n_f frames and the closing pilot, window moved `shift`
    samples later than nominal.

    Sample-clock drift is followed SSTVAE's way (pilot phase slope across
    carriers, slow EMA, +-2 sample steps), but each step's phase is then
    undone so the symbols share one timing reference and the channel
    looks smooth to the interpolator; SSTVAE interpolated straight across
    the steps. A jump in the audio's timing (_find_slip) moves the window
    with it, its phase left alone: the channel after it is the one before.
    Returns raw (n_f, 6, NC), pilots (n_f + 1, NC), the per-frame step
    total, and each frame's slip. `steps_in` replays a previous run's
    steps instead of tracking (scripts/eq_floor.py's genie uses it), and
    `slips_in` its slips (searched for only when neither is given).
    `close_at`: the header-copy frame of a scattered-pilot band, whose
    first symbol is copy_close, not the pilot. `pil` (_sc_pil): scattered pilots. A frame's "pilot" is then each
    carrier's own, from three symbols 48 ms apart; the drift is read
    between a symbol's pilots (3 carriers apart), and slips are not
    searched for.
    """
    b = ofdm.band(band)
    phi_ref = phi_ref + 2 * np.pi * RS * shift / FS  # the shift's own slope
    p = p + shift
    raw = np.zeros((n_f + 1, SYMS_PER_FRAME, b.nc), dtype=np.complex128)
    steps = np.zeros(n_f + 1)
    slips = np.zeros(n_f + 1, dtype=np.int64)
    search = steps_in is None and slips_in is None and pil is None
    ar = np.arange(b.nc)
    intra = pil is not None and pil.ndim == 3
    if pil is not None and not intra:  # a symbol's neighbouring pilots: the drift's phase slope is read between them
        row = pil[np.argmax(pil.any(axis=1))]
        pr = [(a_, b_) for s in np.unique(row) for a_, b_ in zip(np.flatnonzero(row == s)[:-1], np.flatnonzero(row == s)[1:])]
        k1, k2 = (np.array(v, dtype=int) for v in zip(*pr)) if pr else (np.zeros(0, int), np.zeros(0, int))
    tau_ema, powers, total = 0.0, [], 0
    for f in range(n_f + 1):
        n_s = SYMS_PER_FRAME if f < n_f else 1
        if slips_in is not None:
            slips[f] = slips_in[f]
        elif search and 0 < f < n_f:
            slips[f] = _find_slip(z, p, n_f - f, raw[f - 1, 0] / b.pilot, b)
        p += int(slips[f])
        if p + n_s * NSYM > len(z):
            raise SyncError("burst truncated")
        for s in range(n_s):
            raw[f, s] = b.demod_window(z, p + s * NSYM + NCP, DEMOD_BACKOFF)
        steps[f] = total
        if steps_in is not None:
            if f < n_f:
                total = int(steps_in[f + 1])
                p += int(steps_in[f + 1] - steps_in[f])
            p += FRAME_SAMPLES
            continue
        if intra:
            ps, pk = pil[f, :, 0], pil[f, :, 1]
            hf = raw[f, ps, pk] / (copy_open(b)[pk] if f == close_at else b.pilot[pk])
        else:
            hf = (raw[f, 0] if pil is None else raw[f, pil[f], ar]) / (copy_open(b) if f == close_at else b.pilot)
        powers.append(float(np.mean(np.abs(hf) ** 2)))
        if powers[-1] > 0.1 * np.median(powers):
            if intra and f != close_at and f < n_f:
                acc, lags = 0j, []
                for s in np.unique(ps):
                    m = ps == s
                    ks, hs = pk[m], hf[m]
                    if len(ks) > 1:
                        dk = ks[1:] - ks[:-1]
                        acc += np.sum(hs[1:] * np.conj(hs[:-1]) * np.exp(-1j * dk * phi_ref))
                        lags.extend(dk)
                d = float(np.angle(acc) / np.mean(lags)) if lags else 0.0
            elif pil is not None and not intra and pil[f].any():
                d = (np.angle(np.sum(hf[k2] * np.conj(hf[k1]) * np.exp(-1j * (k2 - k1) * phi_ref))) / np.mean(k2 - k1)
                     if len(k1) else 0.0)
            else:
                d = np.angle(np.exp(1j * (_bin_phase_step(hf) - phi_ref)))
            tau_ema += 0.02 * (-d * FS / (2 * np.pi * RS) - tau_ema)
            if abs(tau_ema) >= 2:
                step = int(np.clip(round(tau_ema), -2, 2))
                p += step
                total += step
                tau_ema -= step
        p += FRAME_SAMPLES
    raw *= equalizer.time_shift_phase(steps, b.bb)[:, None, :]
    if intra:
        hp = raw[np.arange(n_f + 1)[:, None], pil[:, :, 0], pil[:, :, 1]] / b.pilot[pil[:, :, 1]]
        if close_at is not None:
            hp[close_at] *= b.pilot[pil[close_at, :, 1]] / copy_open(b)[pil[close_at, :, 1]]
    else:
        hp = (raw[:, 0] if pil is None else raw[np.arange(n_f + 1)[:, None], pil, ar]) / b.pilot
        if close_at is not None:
            hp[close_at] *= b.pilot / copy_open(b)
    return raw[:n_f], hp, steps, slips


def _read_header(z: np.ndarray, start: int, band: str = "w", accept: Accept | None = None) -> HeaderRead:
    """Header decode assuming the preamble starts at `start`."""
    b = ofdm.band(band)
    # Channel reference from the last REF_REPEATS repeats only. Detection
    # wants the whole preamble; a channel reference averaged over 320 ms
    # (16 repeats) is smeared and stale by the header under 2 Hz Doppler.
    R = b.spec.preamble_repeats
    ref = min(R, REF_REPEATS)
    u0 = start + PREAMBLE_CP + (R - ref) * M
    h_reps = np.array([
        b.demod_window(z, start + PREAMBLE_CP + r * M, HEADER_BACKOFF) for r in range(R)
    ]) / b.pilot
    h_pre = h_reps[R - ref:].mean(axis=0)
    # Header symbols sit between the preamble and the first frame pilot,
    # with pilots among them on long headers (header_layout): each symbol's
    # channel is interpolated in time between its nearest pilots (SSTVAE
    # used the preamble alone).
    h0 = start + b.spec.preamble_samples
    p0 = h0 + header_samples(band)
    h_first = b.demod_window(z, p0 + NCP, HEADER_BACKOFF) / b.pilot
    layout = header_layout(band)
    t_sym = h0 + np.arange(len(layout)) * NSYM + NCP + M / 2  # centres
    y_all = np.array([b.demod_window(z, int(t - M / 2), HEADER_BACKOFF) for t in t_sym])
    # the first frame pilot anchors the end, unless another band's frames
    # may follow (then the header's own closing pilot does)
    t_end, h_end = ([], []) if _hosts(band) else ([p0 + NCP + M / 2], [h_first])
    t_p = np.concatenate([[u0 + ref * M / 2], t_sym[layout], t_end])
    h_p = np.concatenate([h_pre[None], y_all[layout] / b.pilot] + [h[None] for h in h_end])
    j = np.clip(np.searchsorted(t_p, t_sym[~layout]) - 1, 0, len(t_p) - 2)
    a = ((t_sym[~layout] - t_p[j]) / (t_p[j + 1] - t_p[j]))[:, None]
    # frequency-smoothed onto a CP-long delay support (as the backed-off
    # window sees it) before interpolating: genie-sync header failures,
    # wide band, -8 dB AWGN 43% -> 14%, mpd 0 dB 4.7% -> 3.2%
    h_p = equalizer.freq_smooth(h_p, (HEADER_BACKOFF, HEADER_BACKOFF + NCP), b.bb)[0]
    ys, hs = y_all[~layout], (1 - a) * h_p[j] + a * h_p[j + 1]
    soft = constellation.llr(ys, hs, np.ones(ys.shape), QPSK)
    floor = max(HEADER_MIN_SCORE[band], accept.min_score if accept else 0.0)
    word, hdr, score = decode_header(soft, band, accept)
    pending = False
    if band in HEADER_COPY_BANDS:
        # the second copy, at each frame it can be in: a decode counts only if
        # the burst it describes carries its copy there
        best = None
        for kc in range(1, HEADER_COPY_AFTER + 1):
            extra = _copy_llr(z, p0 + kc * FRAME_SAMPLES, band, len(ys))
            if extra is None:
                pending = True
                continue
            w2, h2, s2 = decode_header(soft + extra, band, accept)
            if copy_frame(band, h2[1] * h2[0].frames_per_cw) == kc and s2 >= floor and (best is None or s2 > best[2]):
                best = (w2, h2, s2)
        if best is not None:
            (word, hdr, score), pending = best, False
    if score < floor:
        hdr = None
    return dict(word=word, hdr=hdr, score=score, pending_copy=pending, y=ys, y_all=y_all, h_pre=h_pre, h_first=h_first,
                p0=p0, start=start, band=band, n0_pre=equalizer.preamble_noise(h_reps),
                n0_pre_k=equalizer.preamble_noise_k(h_reps))


def _copy_llr(z: np.ndarray, p: int, band: str, n_hdr: int) -> np.ndarray | None:
    """LLRs of the header copy in the frame starting at p (its channel
    interpolated between its pilot and the next), in the first copy's
    order; the frame's last symbol repeats the header's first. None: not
    yet in z."""
    b = ofdm.band(band)
    if p + FRAME_SAMPLES + NSYM > len(z):
        return None
    win = [b.demod_window(z, p + s * NSYM + NCP, HEADER_BACKOFF) for s in range(SYMS_PER_FRAME + 1)]
    if band in SC_BANDS:  # pilot, the header's 4 symbols, pilot
        h0, h1 = win[0] / copy_open(b), win[5] / copy_close(b)
        a = (np.arange(1, 5) / 5)[:, None]
        ys = np.array(win[1:5])
        return constellation.llr(ys, (1 - a) * h0 + a * h1, np.ones(ys.shape), QPSK)
    h0, h1 = win[0] / b.pilot, win[SYMS_PER_FRAME] / b.pilot
    a = (np.arange(1, SYMS_PER_FRAME) / SYMS_PER_FRAME)[:, None]
    ys, hs = np.array(win[1:SYMS_PER_FRAME]), (1 - a) * h0 + a * h1
    llr = constellation.llr(ys, hs, np.ones(ys.shape), QPSK).reshape(DATA_SYMS_PER_FRAME, -1)
    out = llr[:n_hdr].copy()
    for i in range(n_hdr, DATA_SYMS_PER_FRAME):
        out[i % n_hdr] += llr[i]  # row i repeats header symbol i mod n_hdr (burst_waveform's np.resize)
    return out.reshape(-1)


def _best_header(z0: np.ndarray, bands=None, complete: bool = True, accept: Accept | None = None,
                 stats: dict | None = None, final: bool = False) -> tuple:
    """-> (header read, acquisition, frequency-corrected z) of the best
    CRC-valid header in baseband `z0`, over `bands` (default: every sync
    band). `complete`: the burst the header claims must fit in `z0`;
    False for a streaming receiver, which waits for the rest. `final`
    (with complete False): z0 is the head of a burst that has wholly
    arrived (receive's `head`), so nothing waits: a hypothesis past its end
    is not read and a weak first copy stands. Waiting there refused
    headers under STREAM_COMMIT_SCORE whenever an acquisition alternative
    lay near the window's end: 4 of AG7EW's bursts on air, 0.42-0.49,
    two of them decodable (recordings/20261002-232711).

    Every band's detector runs; each detection has its header read, and
    the best CRC-valid header decode decides the band. (A wide preamble
    is M-periodic through a narrow filter too, so detection alone can
    not tell bands apart; the template and header codes can.)

    The preamble is periodic, so its template correlation has sidelobes
    a whole repeat either side at (R-1)/R of the peak: measured at 16
    repeats, ~8% of mpd detections locked one repeat off and read the
    header from the wrong symbols. The header settles that too: it is
    read at the acquired start and at +-1, +-2 repeats."""
    good, detected, waiting = [], 0, False
    for name in bands or (accept.bands if accept else SYNC_BANDS):
        try:
            acq_b = _sync.acquire(z0, band=ofdm.band(name), S=stats.get(name) if stats else None)
        except SyncError:
            continue
        detected += 1
        # the header and the first frame pilot after it must be in the buffer
        hdr_end = BANDS[name].preamble_samples + header_samples(name) + NSYM
        # the winner and acquisition's runner-ups (other CFO bins, other
        # peaks): the header, not the detector, decides which is a preamble
        hyps = [(acq_b.preamble_start, acq_b.freq_offset)] + acq_b.alternatives
        for h, (start, f) in enumerate(hyps):
            # streaming: a header not yet wholly in the buffer at every shift
            # read below waits for more audio. Reading the shifts that do fit
            # read garbage off a real burst's half-arrived header, which
            # passed the w floor and hid the real header (audio loopback).
            if not complete and start + 2 * M > len(z0) - hdr_end:
                waiting |= not final
                continue
            zb = dsp.freq_correct(z0, f)
            for k in (0, -1, 1, -2, 2):
                if not 0 <= start + k * M <= len(z0) - hdr_end:
                    continue
                r = _read_header(zb, start + k * M, name, accept)
                if not complete and not final and r["pending_copy"] and r["score"] < COPY_COMMIT_SCORE:
                    waiting = True  # a weak first copy waits for the second
                    continue
                # a header claiming more than the buffer holds is not this burst's
                if r["hdr"] is not None and (not complete or burst_end(r["p0"], *r["hdr"]) <= len(z0)):
                    rank = r["score"] - (ALT_PENALTY if h or k else 0.0)
                    good.append((rank, r, replace(acq_b, preamble_start=start, freq_offset=f), zb))
    if not good:
        raise SyncError("header decode failed" if detected else "no preamble found")
    # streaming: while some band's header is still arriving, a weak one
    # from another band doesn't commit (the w detector fires on a narrow
    # preamble too, and its shorter header read garbage off an n10 CQ frame)
    if waiting and max(g[0] for g in good) < STREAM_COMMIT_SCORE:
        raise SyncError("a header is still arriving")
    _, hd, acq, z = max(good, key=lambda g: g[0])
    return hd, acq, z


def find_burst(x: np.ndarray, bands=None, accept: Accept | None = None, stats: dict | None = None) -> Lock:
    """Where the first burst in `x` is, from its preamble and header
    alone, so a streaming receiver knows how much audio to wait for
    (`stats`: per band, sync.StreamDetector's statistic for x's starts).
    The burst may run past the end of `x`. Raises SyncError."""
    hd, acq, _ = _best_header(dsp.to_baseband(np.asarray(x, dtype=np.float64)), bands, complete=False, accept=accept,
                              stats=stats)
    spec, n_cw = hd["hdr"]
    out: Lock = dict(spec=spec, n_cw=n_cw, start=hd["start"], score=hd["score"], band=hd["band"],
                     end=burst_end(hd["p0"], spec, n_cw), p0=hd["p0"], cfo=acq.freq_offset)
    if (ch := connect_hash_of(hd["word"], hd["band"])) is not None:
        out["connect_hash"] = ch
    return out


# A burst's frame pilots, PILOT_PAIRS consecutive pairs, are there when both
# (scripts/pilot_metric_study.py, 2026-11-10):
# - pilot_coherence's mean is over PILOT_NOISE: they repeat from frame to
#   frame. Any steady narrowband signal does too (a tone's leakage turns by
#   the same phase in every carrier), and alone this read a tone, a CW
#   key-down or a Hand as pilots 88-100% of the time;
# - the pattern (pilot_coherence's `pattern`) is over PILOT_PATTERN: with the
#   known pilot divided out, adjacent carriers agree (a smooth channel). A
#   tone's leakage, noise or someone's data leave the pilot's pseudo-random
#   pattern. PILOT_PATTERN is noise's 90th percentile; PILOT_NOISE then
#   noise's 99th of the coherence where the pattern passes (1% of noise
#   windows pass both, as the coherence alone was set).
# Against the coherence alone, windows read as pilots: real bursts at -6 dB
# MPD w 0.66 -> 0.83, n4 0.41 -> 0.61 (every band and channel up or level);
# tones and Hands at +20 dB 0.88-1.00 -> 0.00-0.03, at 0 dB -> 0.06-0.11.
PILOT_PAIRS = 4
PILOT_NOISE = {"w": 0.24, "n10": 0.38, "n4": 0.59, "w48": 0.18}
PILOT_PATTERN = {"w": 0.318, "n10": 0.504, "n4": 0.876, "w48": 0.226}
# FM's bands, measured as the others were (scripts/band_consts.py, 3000 reads each,
# 2026-11-17): pilot pattern 90% / coherence 99% on white noise f55 0.208 / 0.191,
# f108 0.148 / 0.142; header reads on noise top out at 0.222 (f55) and 0.153 (f108),
# off other bands' bursts 0.218 and 0.153 (the longer codes read lower).
# Header floors re-measured on what an FM receiver hands the modem
# (scripts/fm_header_floor.py, 2026-11-17, blanked: 12000 reads per band over
# mic and flat hiss and idle carriers at 8 dB, plus 12348 off other bands'
# bursts): FM audio reads higher than white noise, max f55 0.276 (flat, idle
# carrier), f108 0.175, so white noise's 0.23 / 0.16 let noise through. The
# floors sit just over those maxima, as w48's does. True headers (lowest rung,
# 60 trials, genie start): none lost at +2 dB static on either path; f55 mic
# static at 0 dB loses 43% (0.23: 5%), under that rung's 1.25 dB 10% knee;
# near the mobile knees (8 dB up) no change.
if "f55" in BANDS:
    HEADER_MIN_SCORE.update({"f55": 0.28, "f108": 0.18})
    PILOT_NOISE.update({"f55": 0.19, "f108": 0.14})
    PILOT_PATTERN.update({"f55": 0.208, "f108": 0.148})


def pilot_coherence(x: np.ndarray, lock: Lock, n_max: int = 8, latest: bool = False, pattern: bool = False):
    """For a streaming lock (find_burst's dict, same sample origin as x):
    per consecutive pair of the frame pilots already in x (the first n_max
    pairs, or with `latest` the newest), |sum over carriers of y_f
    conj(y_f-1)| / sqrt(energies). A real burst's pilots repeat through a
    slowly varying channel; a false lock's 'pilots' are noise or someone
    else's data, near 1/sqrt(carriers). With `pattern`: -> (that list, the
    pattern), the pattern being, with the known pilot divided out, the
    adjacent carriers' products summed over those frames (their phase is
    the CFO's in every frame), each carrier pair at unit weight, |sum| / pairs:
    near 1 for a smooth channel, near 1/sqrt(pairs) for anything else,
    however strong one carrier is (PILOT_PATTERN)."""
    spec = lock["spec"]
    b = ofdm.band(spec.band)
    total = frames_on_air(spec, lock["n_cw"]) + 1  # frame pilots and the closing one
    avail = sum(lock["p0"] + f * FRAME_SAMPLES + NCP + M <= len(x) for f in range(total))
    f1 = avail if latest else min(avail, n_max + 1)
    f0 = max(0, f1 - n_max - 1)
    if f1 - f0 < 2:
        return ([], 0.0) if pattern else []
    lo = max(0, lock["p0"] + f0 * FRAME_SAMPLES - 2 * NSYM)
    hi = lock["p0"] + (f1 - 1) * FRAME_SAMPLES + 3 * NSYM
    z = dsp.freq_correct(dsp.to_baseband(np.asarray(x[lo:hi], dtype=np.float64)), lock["cfo"])
    y = [b.demod_window(z, lock["p0"] - lo + f * FRAME_SAMPLES + NCP, DEMOD_BACKOFF) for f in range(f0, f1)]
    kc = copy_frame(spec.sync_band, lock["n_cw"] * spec.frames_per_cw)
    if spec.band in SC_BANDS and kc is not None and f0 <= kc < f1:
        y[kc - f0] = y[kc - f0] * b.pilot / copy_open(b)  # its first symbol is copy_close
    sc = sc_spec(spec, lock["n_cw"] * spec.frames_per_cw)
    intra = _sc_intra_spec(b.nc) if sc else None
    if intra is not None:
        K, syms, _ = intra
        ds = syms[1] - syms[0]
        # pair ds-spaced pilot symbols within each frame (and into full-row copy/closing frames), grouped to ~nc cells
        segs = []
        a_lag = np.zeros(b.nc - K, dtype=np.complex128)
        for fa in range(f0, f1):
            p_fa = lock["p0"] - lo + fa * FRAME_SAMPLES
            if fa in (kc, total - 1):
                ref = copy_open(b) if fa == kc else b.pilot
                h0 = b.demod_window(z, p_fa + NCP, DEMOD_BACKOFF) / ref
                segs.append((fa * SYMS_PER_FRAME, np.arange(b.nc), h0))
                a_lag += h0[K:] * np.conj(h0[:-K])
            else:
                _, PK, _ = sc_frame_cells(b.nc, fa)
                ks = PK[: b.nc // K]
                for s_ in syms:
                    hs = b.demod_window(z, p_fa + s_ * NSYM + NCP, DEMOD_BACKOFF)[ks] / b.pilot[ks]
                    segs.append((fa * SYMS_PER_FRAME + s_, ks, hs))
                    if len(ks) > 1:
                        a_lag[ks[:-1]] += hs[1:] * np.conj(hs[:-1])
        pairs = []
        for idx in range(1, len(segs)):
            s0, k0, h0 = segs[idx - 1]
            s1, k1, h1 = segs[idx]
            if s1 - s0 == ds:
                common, i0, i1 = np.intersect1d(k0, k1, return_indices=True)
                if len(common):
                    pairs.append((h0[i0], h1[i1]))
        if not pairs:
            return ([], 0.0) if pattern else []
        # group K consecutive comb-pairs so each coherence reading spans ~nc cells (noise floor matches PILOT_NOISE)
        grp = min(K, len(pairs))
        c = []
        for g0 in range(max(1, len(pairs) - grp + 1)):
            u = np.concatenate([pairs[j][0] for j in range(g0, g0 + grp)])
            v = np.concatenate([pairs[j][1] for j in range(g0, g0 + grp)])
            c.append(float(np.abs(np.vdot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30)))
        if not pattern:
            return c
        act = np.abs(a_lag) > 0
        if act.any():
            aa = a_lag[act]
            return c, float(np.abs(np.sum(aa / np.abs(aa))) / len(aa))
        return c, 0.0
    if sc:  # each carrier's own pilot (symbols 0, 2, 4), but the copy's frame and the closing pilot
        kc = copy_frame(spec.sync_band, lock["n_cw"] * spec.frames_per_cw)
        for i, f in enumerate(range(f0, f1)):
            if f not in (kc, total - 1):
                at = sc_cells(b.nc)[3] // b.nc
                for j in np.unique(at[at > 0]):
                    y[i][at == j] = b.demod_window(z, lock["p0"] - lo + f * FRAME_SAMPLES + j * NSYM + NCP,
                                                   DEMOD_BACKOFF)[at == j]
    c = [float(np.abs(np.sum(y[f] * np.conj(y[f - 1])))
               / (np.sqrt(np.sum(np.abs(y[f]) ** 2) * np.sum(np.abs(y[f - 1]) ** 2)) + 1e-30))
         for f in range(1, len(y))]
    if not pattern:
        return c
    zs = [v * np.conj(b.pilot) for v in y]
    lag = 3 if sc and _sc_combs(b.nc) else 1  # k3d2: neighbours in one symbol's comb
    a = sum(z[lag:] * np.conj(z[:-lag]) for z in zs)
    return c, float(np.abs(np.sum(a / (np.abs(a) + 1e-30))) / len(a))


# Mid-burst acquisition from the header copy (w, w48; scripts/copy_acq_study.py),
# for a burst whose preamble was lost in a fade: the frame pilots after it
# give the frame grid, the copy gives the header.
COPY_PAIRS = 3  # frame-pilot pairs per product window
COPY_GRIDS = 4  # frame grids (phases) tried per search
COPY_EARLIER = (0, NCP // 2, NCP)  # read the copy this much before a peak (its strongest path may be late)
# mean frame-pilot coherence a copy lock needs over its burst: on noise
# locks read 0.11-0.23 (47 in 2 h), true locks that decoded 0.41-0.73 (58)
COPY_COHERENCE = 0.35
# ... and their pattern (pilot_coherence's, over the whole burst): a CW ID's
# key-downs and an n10 burst's pilots are coherent too, and their "copy" read
# a word over the header floor (scripts/copy_pattern_study.py: 10 s of each at
# 0-20 dB, band w, 6 and 20 locks of 90, pattern 0.03-0.51). Right copy locks
# (head wiped, -8..8 dB, AWGN/MPP/MPD) read 0.65 and up on w (662), 0.53 and
# up on w48 (543; 2 under 0.6, neither decoded).
COPY_PATTERN = 0.6
# find_copy's normalized pilot peak worth reading headers at: noise-only
# 1.9 s buffers (the stream's) reach 11.9 (2400 of them, w and w48); bursts
# the copy lock went on to decode, 16.6-91 (16, MPP/MPD low end). Without it
# the search tripled a listening receiver's CPU (4.4 -> 13.7 s per minute).
COPY_DETECT = 14.0


def find_copy(x: np.ndarray, band: str, accept: Accept | None = None, C: np.ndarray | None = None,
              level: float | None = None, D: np.ndarray | None = None, fold: np.ndarray | None = None,
              comb: bool | None = None, decim: int = 1, row_level: np.ndarray | None = None) -> Lock | None:
    """A burst in real audio `x` found from its frame pilots and header copy
    alone (the preamble and first header lost): a Lock with "copy" (what
    receive() needs to rebuild the header), or None. The burst
    may run past the end of x. `C`, `level`: the preamble matched filter's
    outputs for x's starts (sync.StreamDetector.C) and noise level, if a
    stream has them; computed here otherwise. `D`: their pair_products
    (StreamDetector.products_from), instead of C. `fold`: step 1's fold of
    D's whole frames, if kept running (FoldTrack.fold, rolled to D's start).
    `row_level`: an intra-frame comb layout's noise level per CFO bin
    (CombStream.row_levels; computed here otherwise), for _copy_row_weights.

    1. The matched filter's outputs one frame apart, correlated as the
       preamble's are one repeat apart (the frame pilot is the preamble's
       repeat symbol), over COPY_PAIRS pairs, then folded over every frame
       per grid phase: a burst's pilots add coherently (each pair turns by
       the same CFO), its data symbols (M-periodic too) at random.
    2. Per grid (the COPY_GRIDS best phases), the CFO: the fold's phase (the
       CFO modulo 1 / FRAME_S), every alias within the bin (the copy read
       interpolates its channel over a frame, and a half-bin error turns it
       0.9 cycles).
    3. The copy read at every frame of the grid; a word counts if its own
       length puts its copy frame there and it clears the header floor.
    4. The best-scoring word whose burst's frame pilots are coherent
       (COPY_COHERENCE; the live BUSY floors pass noise) and keep the
       pilot's pattern (COPY_PATTERN: a CW ID or a narrower band's burst
       is coherent too)."""
    if (C is not None or D is not None) and level is None:
        raise ValueError("find_copy: C or D needs its noise level")
    if comb is None and band in SC_BANDS:
        if band not in HEADER_COPY_BANDS:  # no copy to find: _copy_llr reads a copy band's frame
            find_copy.peak = 0.0
            return None
        return find_copy(x, band, accept, comb=True)
    b = ofdm.band(band)
    z0 = dsp.to_baseband(np.asarray(x, dtype=np.float64))
    freqs = _sync.cfo_grid()
    intra = _sc_intra_spec(b.nc) if comb else None
    cfo_lag = (intra[1][1] - intra[1][0]) * NSYM if intra is not None else FRAME_SAMPLES
    if comb and D is not None:  # a stream's (CombStream)
        freqs = _intra_freqs(intra[2]) if intra is not None else _comb_freqs()
    if comb and D is None:
        Cs = [(j, _sync.repeat_corrs(z0, t, list(freqs))) for j, t in _comb_filters(band)]
        if intra is not None:
            freqs = _intra_freqs(intra[2])
            P1, level = _intra_frame_prods(Cs, intra, 0)
            row_level = np.min([np.quantile(np.abs(Cc) ** 2, _sync.NOISE_QUANTILE, axis=1) for _, Cc in Cs], axis=0)
            D = _sum_frames(P1, FRAME_SAMPLES, COPY_PAIRS)
        else:
            freqs = _comb_freqs()
            C = _comb_coherent(Cs, freqs)
            level = _sync.noise_level(C)
    if C is None and D is None:
        C = _sync.repeat_corrs(z0, _sync.unit_template(b), list(freqs))
        level = _sync.noise_level(C)
    if D is None:
        D = _sync.pair_products(C, FRAME_SAMPLES, COPY_PAIRS)
    FR = FRAME_SAMPLES // decim
    m = D.shape[1] // FR
    if m < 1:
        return None
    if fold is None:
        fold = D[:, :m * FR].reshape(len(D), m, FR).sum(axis=1)
    if comb:
        # a steady tone's products are the same at every frame phase (a comb's filter, a third of the band's
        # carriers, lets more of a tone outside the band through than the whole pilot's): out with each CFO bin's
        # mean over the phases, or it holds every grid's place and the burst's phase is never read (PR #308
        # review, item 8: mpp with tones at 800 and 2200 Hz, qpsk-r1/2's headers 8/17/20 of 24 at -2/0/+2 dB)
        fold = fold - fold.mean(axis=1, keepdims=True)
        w = _copy_row_weights(row_level, intra[2]) if intra is not None and row_level is not None else None
    else:
        w = None
    mag = np.abs(fold if w is None else fold * w[:, None]).max(axis=0)
    # on noise each product's magnitude is about the level, and the m x
    # COPY_PAIRS of them add at random: the header reads (nearly all of the
    # search's CPU) only past COPY_DETECT. Left on the function for tests
    # (instrumentation, not an interface: native returns it by out-param)
    find_copy.peak = float(mag.max() / (level * np.sqrt(m * COPY_PAIRS)))
    if find_copy.peak < COPY_DETECT:
        return None
    grids = []
    for ph_d in np.argsort(-mag):
        ph = int(ph_d) * decim
        if all(min(abs(ph - g), FRAME_SAMPLES - abs(ph - g)) >= NCP for g in grids):
            grids.append(ph)
            if len(grids) == COPY_GRIDS:
                break
    sb = BANDS[band]
    floor = max(HEADER_MIN_SCORE[band], accept.min_score if accept else 0.0)
    found = []
    for ph in grids:
        col = ph // decim
        i = int(np.argmax(np.abs(fold[:, col])))
        # weighed, the bin the tone spares can outscore the burst's own (every 4th: a neighbour's 0.6-0.9 of its
        # pilots' product over less noise), and the copy is read off the wrong CFO: both bins' (825 Hz: 89 of 144
        # found by the weighed bin alone, 93 by both, 94 unweighed)
        bins = [i] if w is None else list(dict.fromkeys([int(np.argmax(np.abs(fold[:, col]) * w)), i]))
        for f in [f for i in bins for f in _cfo_aliases(fold[i, col], float(freqs[i]), cfo_lag)]:
            z = dsp.freq_correct(z0, f)
            for early in COPY_EARLIER:
                for pc in range((ph - early - NCP) % FRAME_SAMPLES, len(z0) - FRAME_SAMPLES, FRAME_SAMPLES):
                    llr = _copy_llr(z, pc, band, sb.header_syms)
                    if llr is None:
                        continue
                    word, (spec, n_cw), score = decode_header(llr, band, accept)
                    kc = copy_frame(band, n_cw * spec.frames_per_cw)
                    p0 = pc - (kc or 0) * FRAME_SAMPLES
                    start = p0 - header_samples(band) - sb.preamble_samples
                    if kc is not None and score >= floor and start >= 0:
                        lk: Lock = dict(spec=spec, n_cw=n_cw, start=start, score=score, band=band,
                                        end=burst_end(p0, spec, n_cw), p0=p0, cfo=f,
                                        copy=dict(word=word, pc=pc))
                        if (ch := connect_hash_of(word, band)) is not None:
                            lk["connect_hash"] = ch
                        found.append(lk)
    for lock in sorted(found, key=lambda k: -k["score"]):
        if comb:
            c, pat = _copy_rows(z0, lock)
        else:
            c, pat = pilot_coherence(x, lock, n_max=10**4, pattern=True)
        if len(c) >= 2 and np.mean(c) >= COPY_COHERENCE and pat >= COPY_PATTERN:
            return lock
    return None


# A steady tone raises the comb filters' noise in most CFO bins (every 4th is spared), 1.5-26x the quietest on w and
# w48 with +25 dB tones at 800+2200, 825 or 1500 Hz; on noise alone the bins are within 1.06x of each other (mpp
# -2 dB, seeds 1-2). Past this spread find_copy weighs each bin's fold by the quietest bin's level over its own.
SC_COPY_BIN_SPREAD = 1.25


def _copy_row_weights(row_level: np.ndarray, n_combs: int) -> np.ndarray | None:
    """An intra-frame comb layout's fold rows (n_combs x the CFO bins), each weighed by the quietest bin's noise
    level over its own, or None when the bins' levels are within SC_COPY_BIN_SPREAD of each other. Without it a
    tone's cross terms with the noise (the products' tone-only part is the phase mean find_copy takes out) win the
    max over bins at random phases and take the COPY_GRIDS places (fk3d2, w qpsk-r1/2 and 16qam-r1/3 with the
    preamble and first header erased, mpp -4..0 dB, tones at 800 and 2200 Hz: 97 -> 109 of 144 found, at 825 Hz
    94 -> 93; without tones 109, unchanged; mix 111 and 117)."""
    lv = np.asarray(row_level, dtype=np.float64)
    lo = float(lv.min())
    if not lo > 0 or float(lv.max()) <= SC_COPY_BIN_SPREAD * lo:
        return None
    return np.tile(lo / lv, n_combs)


def _comb_filters(band: str) -> list:
    """find_copy's comb matched filters, [(j, unit template)]."""
    b = ofdm.band(band)
    tpl = b.preamble_template()[PREAMBLE_CP:PREAMBLE_CP + M]
    dem = np.exp(-2j * np.pi * np.outer(b.bb, np.arange(M)) / FS)
    intra = _sc_intra_spec(b.nc)
    if intra is not None:
        K, _, n_combs = intra
        out = []
        for c in range(n_combs):
            t = (dem.conj().T * (np.arange(b.nc) % K == c)) @ (dem @ tpl) / M
            out.append((c, t / np.linalg.norm(t)))
        return out
    at = sc_cells(b.nc)[3] // b.nc
    out = []
    for j in np.unique(at):
        t = (dem.conj().T * (at == j)) @ (dem @ tpl) / M
        out.append((int(j), t / np.linalg.norm(t)))
    return out


def _intra_freqs(n_combs: int) -> np.ndarray:
    """Standard 25 CFO bins, repeated across `n_combs` cyclic frame-comb phases (no sub-bin expansion)."""
    return np.tile(_sync.cfo_grid(), n_combs)


def _sum_frames(P: np.ndarray, F: int, pairs: int) -> np.ndarray:
    """Sum `pairs` consecutive frame-spaced values: D[:, t] = sum_{j < pairs} P[:, t + j * F]."""
    n = max(0, P.shape[1] - (pairs - 1) * F)
    if not n:
        return np.zeros((len(P), 0), dtype=np.complex128)
    return sum((P[:, j * F:j * F + n] for j in range(pairs)), np.zeros((len(P), n), dtype=np.complex128))


def _intra_frame_prods(Cs: list, intra: tuple, stream_s0: int, decim: int = 1) -> tuple[np.ndarray, float]:
    """Per-frame intra-comb lag products on the 25 CFO bins, cyclic-muxed by `(stream_s0 + t) // FRAME_SAMPLES`
    when frame-staggered so each burst frame takes only its active comb."""
    _, syms, n_combs = intra
    step = (syms[1] - syms[0]) * NSYM // decim
    span = (FRAME_SAMPLES if n_combs == 1 else syms[-1] * NSYM) // decim
    FR_d = FRAME_SAMPLES // decim
    n = min(c.shape[1] - span for _, c in Cs)
    if n <= 0:
        return np.zeros((len(_intra_freqs(n_combs)), 0), dtype=np.complex128), 1.0
    if n_combs == 1:
        C0 = Cs[0][1]
        P = C0[:, step:step + n] * np.conj(C0[:, :n]) + C0[:, 2 * step:2 * step + n] * np.conj(C0[:, step:step + n])
        return P, float(np.sqrt(2.0) * _sync.noise_level(C0))
    Qs = []
    n_lags = len(syms) - 1
    for _, Cc in Cs:
        q = sum(Cc[:, (m + 1) * step:(m + 1) * step + n] * np.conj(Cc[:, m * step:m * step + n]) for m in range(n_lags))
        Qs.append(q)
    Q = np.stack(Qs, axis=0)  # (n_combs, n_bins, n)
    f_idx = (stream_s0 // decim + np.arange(n)) // FR_d
    rows = [Q[(c0 + f_idx) % n_combs, :, np.arange(n)].T for c0 in range(n_combs)]  # each (n_bins, n)
    lvl = float(np.sqrt(n_lags) * min(_sync.noise_level(Cc) for _, Cc in Cs))
    return np.concatenate(rows, axis=0), lvl


def _comb_freqs() -> np.ndarray:
    """The coherent comb search's CFOs: the grid's bins at SC_COPY_SUBBINS offsets inside each (STEP_HZ: 12.5 Hz
    turns 0.6 cycles over 96 ms)."""
    offs = (np.arange(SC_COPY_SUBBINS) - (SC_COPY_SUBBINS - 1) / 2) * _sync.STEP_HZ / SC_COPY_SUBBINS
    return np.array([f + o for f in _sync.cfo_grid() for o in offs])


def _comb_coherent(Cs: list, freqs: np.ndarray, decim: int = 1) -> np.ndarray:
    """The combs' outputs [(j, C_j)] added within the frame, each turned back by its symbol's offset at each of
    `freqs` (_comb_freqs): the frame's whole pilot energy in one output, for every start all combs cover."""
    k = len(freqs) // len(Cs[0][1])
    n = min(c.shape[1] - (j * NSYM) // decim for j, c in Cs)
    return np.array([sum(c[i // k, (j * NSYM) // decim:(j * NSYM) // decim + n] * np.exp(-2j * np.pi * f * j * NSYM / FS)
                         for j, c in Cs)
                     for i, f in enumerate(freqs)])


SC_COPY_DECIM = 2  # CombStream decimation (2 is Nyquist-exact on w/w48)


class CombStream:
    """find_copy's coherent comb outputs (_comb_coherent) for a stream, as sync.StreamDetector keeps the whole
    pilot's: each start's computed once as audio is fed, not once per search hop over the buffer (mix's
    listening: the comb search was ~90% of a listening receiver's CPU, each start filtered ~8 times). The noise
    level likewise from each fed chunk's new outputs, per bin, the median of the last CHUNKS chunks' (the
    StreamDetector's rule; find_copy's offline level is the whole window's quantile). The pair products are
    extended as asked for and kept KEEP_BEHIND past a trim (a running FoldTrack subtracts what leaves)."""

    CHUNKS = 8
    KEEP_BEHIND = 4000

    def __init__(self, band: str):
        self.band = band
        self.intra = _sc_intra_spec(BANDS[band].nc)
        self.decim = SC_COPY_DECIM if SC_COPY_DECIM in (1, 2, 4) else 2
        self.filters = _comb_filters(band)
        if self.intra is not None:
            _, syms, n_combs = self.intra
            self.freqs = _intra_freqs(n_combs)
            self.lag = FRAME_SAMPLES if n_combs == 1 else syms[-1] * NSYM
        else:
            self.freqs = _comb_freqs()
            self.lag = max(j for j, _ in self.filters) * NSYM  # a start's output needs this and M samples after it
        self.reset()

    def reset(self):
        self.tail = np.zeros(0, dtype=np.complex128)
        self.fed = 0
        self.levels: list = []
        self.C = np.zeros((len(self.freqs), 0), dtype=np.complex128)
        self.c0 = 0
        self.D = np.zeros((len(self.freqs), 0), dtype=np.complex128)
        self.d0 = 0

    def restart_at(self, pos: int):
        self.reset()
        self.fed = pos

    def feed(self, z: np.ndarray):
        """The next baseband samples of the stream (contiguous; dsp.to_baseband with their stream index)."""
        new = len(z)
        self.fed += new
        z = np.concatenate([self.tail, z])
        zs0 = self.fed - len(z)
        D = self.decim
        lead = (D - (zs0 % D)) % D
        z_a = z[lead:]
        n_raw = len(z_a) - M + 1 - self.lag
        if n_raw < D:
            self.tail = z
            return
        n_d = n_raw // D
        n = n_d * D
        zs = zs0 + lead
        Cs = [(j, _sync.repeat_corrs(z_a, t, list(_sync.cfo_grid()))[:, ::D]) for j, t in self.filters]
        qwin_d = max(new, 2000) // D
        if self.intra is not None:
            C, _ = _intra_frame_prods(Cs, self.intra, zs, decim=D)
            n_lags = 2 if self.intra[2] == 1 else len(self.intra[1]) - 1
            per_comb = []
            for _, Cc in Cs:
                p = np.abs(Cc[:, max(0, Cc.shape[1] - qwin_d):]) ** 2
                k = int(_sync.NOISE_QUANTILE * (p.shape[1] - 1))
                per_comb.append(np.partition(p, k, axis=1)[:, k] / -np.log(1 - _sync.NOISE_QUANTILE))
            self.levels = (self.levels + [np.sqrt(n_lags) * np.min(per_comb, axis=0)])[-self.CHUNKS:]
        else:
            C = _comb_coherent(Cs, self.freqs, decim=D)
            p = np.abs(C[:, max(0, C.shape[1] - qwin_d):]) ** 2
            k = int(_sync.NOISE_QUANTILE * (p.shape[1] - 1))
            self.levels = (self.levels + [np.partition(p, k, axis=1)[:, k] / -np.log(1 - _sync.NOISE_QUANTILE)])[-self.CHUNKS:]
        if not self.C.shape[1]:
            self.c0 = zs
        self.C = np.concatenate([self.C, C[:, :n_d]], axis=1)
        self.tail = z[lead + n:]

    def level(self) -> float | None:
        return float(np.median(np.array(self.levels), axis=0).min()) if self.levels else None

    def row_levels(self) -> np.ndarray | None:
        """Per CFO bin, the noise level (level()'s before its min over bins); find_copy's row_level."""
        return np.median(np.array(self.levels), axis=0) if self.levels else None

    def trim(self, start: int):
        D = self.decim
        k = min(max(0, (start - self.c0) // D), self.C.shape[1])
        self.C, self.c0 = self.C[:, k:], self.c0 + k * D
        k = min(max(0, (start - self.KEEP_BEHIND - self.d0) // D), self.D.shape[1])
        self.D, self.d0 = self.D[:, k:], self.d0 + k * D

    def _extend(self):
        D = self.decim
        FR_d = FRAME_SAMPLES // D
        if self.d0 + self.D.shape[1] * D < self.c0:
            self.D, self.d0 = self.D[:, :0], self.c0
        have = (self.d0 - self.c0) // D + self.D.shape[1]
        if self.intra is not None:
            want = max(0, self.C.shape[1] - (COPY_PAIRS - 1) * FR_d)
            if want > have:
                self.D = np.concatenate([self.D, _sum_frames(self.C[:, have:], FR_d, COPY_PAIRS)], axis=1)
            return
        want = max(0, self.C.shape[1] - COPY_PAIRS * FR_d)
        if want > have:
            self.D = np.concatenate([self.D, _sync.pair_products(self.C[:, have:], FR_d, COPY_PAIRS)], axis=1)

    def products_from(self, start: int) -> tuple[int, np.ndarray]:
        """(a, the pair products for stream starts from a = max(start, c0) on)."""
        self._extend()
        D = self.decim
        want = max(self.c0, start)
        diff = want - self.d0
        k = min(max(0, (diff + D - 1) // D if diff > 0 else 0), self.D.shape[1])
        return self.d0 + k * D, self.D[:, k:]

    def kept_products(self) -> tuple[int, np.ndarray]:
        self._extend()
        return self.d0, self.D


def _copy_rows(z0: np.ndarray, lock: Lock) -> tuple:
    """find_copy's gates for a scattered-pilot burst, from its copy frame's two whole rows (the pilot, and the
    closing, both copy_close, 120 ms apart) where pilot_coherence reads each carrier's own pilot from another symbol (their
    pattern's neighbours three carriers apart, on k3d2; hilat-m's real copies read 0.3-0.5 against COPY_PATTERN
    0.6): -> ([coherence, coherence], pattern), pilot_coherence's shapes."""
    b = ofdm.band(lock["band"])
    z = dsp.freq_correct(z0, lock["cfo"])
    pc = lock["copy"]["pc"]
    if pc + SYMS_PER_FRAME * NSYM > len(z):
        return [], 0.0
    r0 = b.demod_window(z, pc + NCP, DEMOD_BACKOFF) / copy_open(b)
    r5 = b.demod_window(z, pc + (SYMS_PER_FRAME - 1) * NSYM + NCP, DEMOD_BACKOFF) / copy_close(b)
    c = float(np.abs(np.vdot(r5, r0)) / (np.linalg.norm(r0) * np.linalg.norm(r5) + 1e-30))
    a = r0[1:] * np.conj(r0[:-1]) + r5[1:] * np.conj(r5[:-1])
    return [c, c], float(np.abs(np.sum(a / (np.abs(a) + 1e-30))) / len(a))


def _pilot_fold(D: np.ndarray, t0: int, m: int) -> np.ndarray:
    """find_copy's step 1 from start t0: per CFO bin and frame phase, the
    pair products D (sync.pair_products of the matched filter's outputs,
    COPY_PAIRS pairs a frame apart) folded over m frames."""
    return D[:, t0:t0 + m * FRAME_SAMPLES].reshape(len(D), m, FRAME_SAMPLES).sum(axis=1)


class FoldTrack:
    """_pilot_fold kept running over a window of a stream's pair products
    that slides forward (the receiver's search hops): per CFO bin and
    absolute frame phase (stream index mod FRAME_SAMPLES), the window's sum.
    Each fold() adds the products that entered the window and subtracts
    those that left, in stream order, a frame phase's run at a time;
    every RENEW folds, or when the window moves any other way, it is
    summed afresh (the subtractions' rounding doesn't build up)."""

    RENEW = 64

    def __init__(self):
        self.reset()

    def reset(self):
        self.R = None
        self.lo = self.hi = 0  # the window summed: stream starts [lo, hi)
        self.n = 0  # folds since the last fresh sum

    def _acc(self, D: np.ndarray, a: int, u0: int, u1: int, sign: int, period: int = FRAME_SAMPLES):
        F, u = period, u0
        while u < u1:
            q = u % F
            k = min(u1 - u, F - q)
            if sign > 0:
                self.R[:, q:q + k] += D[:, u - a:u - a + k]
            else:
                self.R[:, q:q + k] -= D[:, u - a:u - a + k]
            u += k

    def fold(self, D: np.ndarray, a: int, lo: int, hi: int, period: int = FRAME_SAMPLES) -> np.ndarray:
        """The window [lo, hi) of stream starts (D: pair products from
        stream start a, covering it), summed by frame phase: R[:, q] for
        absolute phase q. np.roll(R, -(lo % period), axis=1) is
        _pilot_fold's from lo."""
        if (self.R is None or self.R.shape[1] != period or self.n >= self.RENEW or not self.lo <= lo <= self.hi <= hi
                or self.lo < a or hi > a + D.shape[1]):
            self.R = np.zeros((len(D), period), dtype=np.complex128)
            self._acc(D, a, lo, hi, 1, period)
            self.n = 0
        else:
            self._acc(D, a, self.hi, hi, 1, period)
            self._acc(D, a, self.lo, lo, -1, period)
            self.n += 1
        self.lo, self.hi = lo, hi
        return self.R


# Receiver.data2g_on_air's OFDM rule (pilot_present, scripts/presence_study.py):
# Stage 1 (running FoldTrack): fold contrast at least PRESENCE_CONTRAST[m -
# PRESENCE_MIN_M] over m frames (noise passes 2e-4/hop on each band), and peak
# at least PRESENCE_PEAK. Stage 2 (fold_verify, only when Stage 1 passes):
# known-pilot coherence over PILOT_NOISE and pilot pattern over
# min(PRESENCE_PATTERN_CAP, PRESENCE_PATTERN_SCALE * PILOT_PATTERN) (~99th
# percentile of noise: rejects impulse trains and QRM power steps to <0.2%
# while keeping mid-burst OFDM detection across w, w48, n10 and n4).
PRESENCE_PEAK = 8.0
PRESENCE_CONTRAST = (4.9, 4.9, 4.9, 4.75, 4.35, 4.25, 4.05, 3.9)
PRESENCE_CONTRAST_N4_SCALE = 0.75  # n4's 4-carrier gold comb has wider sidelobes (noise max 2.30, n4 -3 dB 3.75)
PRESENCE_MIN_M = 2
PRESENCE_PATTERN_SCALE = 1.4
PRESENCE_PATTERN_CAP = 0.876


def presence_window(dcols: int, window: int, decim: int = 1, pad_frames: int = COPY_PAIRS) -> tuple[int, int]:
    """(t0, m): the newest `window` outputs' pilot fold, over m whole frames
    from pair product t0 (`dcols` of them, at stride `decim`; m < 1: too few)."""
    FR_d = FRAME_SAMPLES // decim
    n_d = dcols + pad_frames * FR_d  # the outputs the products were made from
    W_d = min(n_d, window // decim)
    return n_d - W_d, (W_d - pad_frames * FR_d) // FR_d


def fold_presence(fold: np.ndarray, level: float, m: int, row_w: np.ndarray | None = None) -> tuple[float, float]:
    """(peak, contrast) of an m-frame pilot fold (any rotation of its phases
    reads the same): find_copy's normalized peak, and the peak over the
    fold's median over frame phases."""
    mag = np.abs(fold if row_w is None else fold * row_w[:, None]).max(axis=0)
    return float(mag.max() / (level * np.sqrt(m * COPY_PAIRS))), float(mag.max() / (np.median(mag) + 1e-300))


def presence_verdict(peak: float, contrast: float, m: int, band: str = "") -> bool:
    """Stage 1 of pilot_present: PRESENCE_PEAK, and PRESENCE_CONTRAST for m frames."""
    scale = PRESENCE_CONTRAST_N4_SCALE if (band == "n4" and band in SC_BANDS) else 1.0
    return (m >= PRESENCE_MIN_M and peak >= PRESENCE_PEAK
            and contrast >= scale * PRESENCE_CONTRAST[min(m - PRESENCE_MIN_M, len(PRESENCE_CONTRAST) - 1)])


def pilot_presence(D: np.ndarray, level: float, window: int) -> tuple[float, float]:
    """Stage 1 (peak, contrast) of the fold over the newest `window` matched
    filter outputs (`D`: their pair_products; `level`: noise level). Receiver
    keeps the fold running instead (FoldTrack)."""
    t0, m = presence_window(D.shape[1], window)
    if m < 1:
        return 0.0, 0.0
    return fold_presence(_pilot_fold(D, t0, m), level, m)


def _verify_band(z0: np.ndarray, ph: int, n_frames: int, band: str, cfo: float) -> bool:
    b = ofdm.band(band)
    z = dsp.freq_correct(z0, cfo)
    if band in SC_BANDS:
        at = sc_cells(b.nc)[3] // b.nc
        y = []
        for k in range(n_frames):
            v = np.empty(b.nc, dtype=np.complex128)
            for j in np.unique(at):
                w = b.demod_window(z, ph + k * FRAME_SAMPLES + int(j) * NSYM, DEMOD_BACKOFF)
                v[at == j] = w[at == j]
            y.append(v)
    else:
        y = [b.demod_window(z, ph + k * FRAME_SAMPLES, DEMOD_BACKOFF) for k in range(n_frames)]
    c = [float(np.abs(np.sum(y[k] * np.conj(y[k - 1])))
               / (np.sqrt(np.sum(np.abs(y[k]) ** 2) * np.sum(np.abs(y[k - 1]) ** 2)) + 1e-30))
         for k in range(1, len(y))]
    zs = [v * np.conj(b.pilot) for v in y]
    a = sum(z_[1:] * np.conj(z_[:-1]) for z_ in zs)
    pat = float(np.abs(np.sum(a / (np.abs(a) + 1e-30))) / len(a))
    pat_floor = min(PRESENCE_PATTERN_CAP, PRESENCE_PATTERN_SCALE * PILOT_PATTERN[band])
    return float(np.mean(c)) > PILOT_NOISE[band] and pat > pat_floor


def _verify_intra(z0: np.ndarray, ph: int, n_frames: int, band: str, cfo: float, c_start: int, intra: tuple) -> bool:
    b = ofdm.band(band)
    nc = b.nc
    K, syms, n_combs = intra
    z = dsp.freq_correct(z0, cfo)
    pairs = []
    a_lag = np.zeros(nc - K, dtype=np.complex128)
    for k in range(n_frames):
        c = (c_start + k) % n_combs
        ks = np.arange(c, nc, K)
        hs = [b.demod_window(z, ph + k * FRAME_SAMPLES + s * NSYM, DEMOD_BACKOFF)[ks] / b.pilot[ks] for s in syms]
        for h in hs:
            a_lag[ks[:-1]] += h[1:] * np.conj(h[:-1])
        for m_ in range(len(hs) - 1):
            pairs.append((hs[m_], hs[m_ + 1]))
    if not pairs:
        return False
    grp = min(K, len(pairs))
    cs = []
    for g0 in range(max(1, len(pairs) - grp + 1)):
        u = np.concatenate([pairs[j][0] for j in range(g0, g0 + grp)])
        v = np.concatenate([pairs[j][1] for j in range(g0, g0 + grp)])
        cs.append(float(np.abs(np.vdot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30)))
    act = np.abs(a_lag) > 0
    pat = float(np.abs(np.sum(a_lag[act] / np.abs(a_lag[act]))) / act.sum()) if act.any() else 0.0
    pat_floor = min(PRESENCE_PATTERN_CAP, PRESENCE_PATTERN_SCALE * PILOT_PATTERN[band])
    return float(np.mean(cs)) > PILOT_NOISE[band] and pat > pat_floor


def fold_verify(x: np.ndarray, band: str, fold: np.ndarray, fold_phase0: int = 0,
                decim: int = 1, row_w: np.ndarray | None = None) -> bool:
    """Stage 2 of pilot_present: verify that `fold`'s peak in real audio `x`
    (`x[0]` at stream index `fold_phase0`, 0 for a relative fold) carries
    `band`'s known pilots (fk3d2 intra-frame combs on w/w48, gold scattered
    pilots on n10/n4, whole pilot rows on FM). Rejects impulse trains and QRM
    power steps that clear Stage 1."""
    wfold = fold if row_w is None else fold * row_w[:, None]
    mag = np.abs(wfold).max(axis=0)
    q = int(np.argmax(mag))
    fi = int(np.argmax(np.abs(wfold[:, q])))
    ph = (q * decim - fold_phase0) % FRAME_SAMPLES
    sc = band in SC_BANDS
    intra = _sc_intra_spec(BANDS[band].nc) if sc else None
    if intra is not None:
        _, syms, n_combs = intra
        n_frames = (len(x) - ph - syms[-1] * NSYM - M) // FRAME_SAMPLES + 1
        if n_frames < 2:
            return False
        freqs = _intra_freqs(n_combs)
        c0 = fi // len(_sync.cfo_grid())
        c_start = (c0 + (fold_phase0 + ph) // FRAME_SAMPLES) % n_combs
        z0 = dsp.to_baseband(np.asarray(x, dtype=np.float64))
        d = fold[fi, q]
        cfo_lag = (syms[1] - syms[0]) * NSYM
        for f in _cfo_aliases(d, float(freqs[fi]), cfo_lag):
            if _verify_intra(z0, ph, n_frames, band, f, c_start, intra):
                return True
        return False
    max_j = int((sc_cells(BANDS[band].nc)[3] // BANDS[band].nc).max()) if sc else 0
    n_frames = (len(x) - ph - max_j * NSYM - M) // FRAME_SAMPLES + 1
    if n_frames < 2:
        return False
    freqs = _comb_freqs() if sc else _sync.cfo_grid()
    z0 = dsp.to_baseband(np.asarray(x, dtype=np.float64))
    d = fold[fi, q]
    centre = float(freqs[fi])
    for f in _cfo_aliases(d, centre):
        if _verify_band(z0, ph, n_frames, band, f):
            return True
    if not sc and band == "n10" and "n4" in BANDS:
        # n4's 4-carrier pilot peaks against n10's template at +100 Hz (0.62)
        # and -137.5 Hz (0.52), so n4's true CFO sits at centre - 100 or + 137.5
        for df in (-100.0, 137.5):
            c4 = centre + df
            if abs(c4) <= _sync.ACQUIRE_REACH_HZ + 25.0:
                for f in _cfo_aliases(d, c4):
                    if _verify_band(z0, ph, n_frames, "n4", f):
                        return True
    return False


def pilot_present(D: np.ndarray, level: float, window: int, x: np.ndarray | None = None,
                  band: str | None = None, decim: int = 1, stream_lo: int = 0,
                  row_w: np.ndarray | None = None) -> bool:
    """pilot_presence's verdict over the newest `window` outputs (fewer when
    there are fewer: the threshold rises as the window shrinks). When `x` (same
    origin as D) and `band` are given, also runs Stage 2 (fold_verify)."""
    sc = band is not None and band in SC_BANDS
    intra = _sc_intra_spec(BANDS[band].nc) if sc else None
    pad_fr = (COPY_PAIRS - 1) if intra is not None else COPY_PAIRS
    FR_d = FRAME_SAMPLES // decim
    t0_d, m = presence_window(D.shape[1], window, decim, pad_fr)
    if m < PRESENCE_MIN_M:
        return False
    fold = D[:, t0_d:t0_d + m * FR_d].reshape(len(D), m, FR_d).sum(axis=1)
    if sc:
        fold = fold - fold.mean(axis=1, keepdims=True)
    if not presence_verdict(*fold_presence(fold, level, m, row_w), m, band or ""):
        return False
    if x is None or band is None:
        return True
    shift = (stream_lo // decim + t0_d) % FR_d
    if shift:
        fold = np.roll(fold, shift, axis=1)
    return fold_verify(x[t0_d * decim:], band, fold, stream_lo + t0_d * decim, decim, row_w)


def _cfo_aliases(d: complex, centre: float, lag: int = FRAME_SAMPLES) -> list[float]:
    """CFOs whose turn over `lag` samples is d's phase, within the bin at
    `centre`. A frame-pair product carries the whole CFO modulo FS / lag
    (6.94 Hz at a frame; 13.89 or 20.83 Hz for intra-frame comb pairs)."""
    alias = FS / lag
    frac = float(np.angle(d)) / (2 * np.pi) * alias
    k0 = round((centre - frac) / alias)
    return [frac + k * alias for k in (k0 - 1, k0, k0 + 1) if abs(frac + k * alias - centre) <= _sync.STEP_HZ / 2 + 0.5]


def _copy_header(z: np.ndarray, lock: Lock) -> HeaderRead:
    """_read_header's dict for a copy lock (z frequency-corrected): the header
    it read, its phase reference from the copy frame's pilot (the preamble
    faded)."""
    band = lock["band"]
    b = ofdm.band(band)
    hd = _read_header(z, lock["start"], band)
    # (a scattered-pilot band's copy frame opens on copy_open, not the pilot)
    hd.update(hdr=(lock["spec"], lock["n_cw"]), word=lock["copy"]["word"], score=lock["score"],
              h_pre=b.demod_window(z, lock["copy"]["pc"] + NCP, HEADER_BACKOFF) / (copy_open(b) if band in SC_BANDS else b.pilot))
    return hd


def receive(x: np.ndarray, bands=None, accept: Accept | None = None, head: int | None = None,
            copy: Lock | None = None, known: dict | None = None, partial: bool = False) -> Received:
    """Everything up to soft bits for the first burst in `x`: the
    synchronisation state (so an analysis script can replay it on
    another signal) and the equalizer output. `bands`: the sync bands to
    look for (default: all). `head`: the preamble and header lie in the
    first `head` samples (a streaming receiver found them there): search
    only those. Acquisition over a whole 11 s burst took 1-4 s of CPU,
    nearly all of receive's time. `copy`: find_copy's lock (same sample
    origin as x): no preamble search, the header its copy read. `known`:
    {start, cfo, spec, n_cw} of a burst whose timing the receiver knows
    already (data2g.bulk): no search, the header neither read nor needed,
    its known word the coarse CFO's reference. `partial`: `x` holds the
    burst only so far (a streaming receiver decoding as it arrives): the
    air frames whose closing pilot is in, `frames_in` data frames of them
    (receive_partial's caller decodes the slots ready_slots names)."""
    z0 = dsp.to_baseband(np.asarray(x, dtype=np.float64))
    if known is not None:
        ks, kn = known["spec"], known["n_cw"]
        z = dsp.freq_correct(z0, known["cfo"])
        hd = _read_header(z, known["start"], ks.sync_band)
        v = (ks.index << CW_BITS.get(ks.sync_band, 6)) | (kn - 1)
        hd.update(hdr=(ks, kn), word=(v << 6) | crc6(v))
        acq = _sync.Acquisition(known["start"], known["cfo"], 0.0)
        if burst_end(hd["p0"], ks, kn) > len(z0):
            raise SyncError("burst runs past the buffer")
    elif copy is not None:
        if copy["end"] > len(z0):
            raise SyncError("burst runs past the buffer")
        z = dsp.freq_correct(z0, copy["cfo"])
        hd = _copy_header(z, copy)
        acq = _sync.Acquisition(copy["start"], copy["cfo"], 0.0)
    elif head is None:
        hd, acq, z = _best_header(z0, bands, accept=accept)
    else:
        hd, acq, _ = _best_header(z0[:head], bands, complete=False, accept=accept, final=True)
        z = dsp.freq_correct(z0, acq.freq_offset)
        spec, n_cw = hd["hdr"]
        if not partial and burst_end(hd["p0"], spec, n_cw) > len(z0):
            raise SyncError("burst runs past the buffer")
    band = hd["band"]  # the sync band; frames are on spec.band's carriers
    b = ofdm.band(band)
    spec, n_cw = hd["hdr"]
    db = spec.band
    word, hdr_y, h_pre, h_first, p0 = hd["word"], hd["y"], hd["h_pre"], hd["h_first"], hd["p0"]

    n_f = n_cw * spec.frames_per_cw
    kc = copy_frame(band, n_f)
    n_air = n_f + (kc is not None)  # the header copy's frame among the data frames
    if partial:  # the frames in so far, each with its closing pilot (and room for the window's shift)
        n_air = min(n_air, max(0, (len(z0) - p0 - NSYM - 2 * NCP) // FRAME_SAMPLES))
        if n_air < 1:
            raise SyncError("no whole frame yet")
        if kc is not None and kc >= n_air:
            kc = None  # its copy not in yet
    n_in = n_air - (kc is not None)  # data frames in
    phi_ref = _bin_phase_step(h_pre)

    # Residual CFO. The pilots measure it finely but only modulo the
    # pilot rate (6.94 Hz), and fading's random FM puts the preamble's own
    # estimate past half of that now and then (measured: 1 burst in 10 on
    # mpp, 6.8 Hz off). The alias is picked by a coarse estimate from the
    # header's symbols and the first frame pilot, known once the header
    # has decoded: consecutive symbols 24 ms apart, so unambiguous to
    # +-20.8 Hz. (Picking it by how well the data fit the constellation
    # instead chose a wrong alias in a third of 1-frame QPSK bursts at
    # 0 dB AWGN.)
    layout = header_layout(band)
    ref_syms = np.empty((len(layout), b.nc), dtype=np.complex128)
    ref_syms[layout] = b.pilot
    ref_syms[~layout] = constellation.modulate(_word_bits(word) @ header_code(band) % 2, QPSK).reshape(-1, b.nc)
    # (not `known`: that is the argument, a bulk burst's, and the alias choice below reads it)
    sc, scb = sc_spec(spec, n_f), db in SC_BANDS
    if sc and partial:
        # ponytail: a burst still arriving has no closing pilot symbol, and the frame after the last one in starts
        # with scattered cells, not a pilot row; decoding while it arrives (config.DECODE_EARLY, FM) waits for the end
        raise SyncError("scattered pilots: the whole burst only")
    pil = _sc_pil(db, n_air, kc) if sc else None
    ref = list(hd["y_all"] * np.conj(ref_syms)) + ([h_first] if db == band and not sc else [])
    runs = [ref]
    if kc is not None:
        # the header copy's frame is known symbols too (pilot, copy, next
        # pilot): the estimate survives a fade on the first copy
        pc = p0 + kc * FRAME_SAMPLES
        ys = [b.demod_window(z, pc + s * NSYM + NCP, HEADER_BACKOFF) for s in range(SYMS_PER_FRAME + (not scb))]
        h = ref_syms[~layout]
        ref2 = [copy_open(b), *h, copy_close(b)] if scb else [b.pilot, *np.resize(h, (DATA_SYMS_PER_FRAME, b.nc)), b.pilot]
        runs.append([y * np.conj(r) for y, r in zip(ys, ref2)])
    # Leave carriers hit by a steady tone or its off-carrier sidelobes out of
    # coarse and residual_cfo (issue #330): a +25 dB tone at 800 Hz otherwise
    # pulls coarse across the +-3.47 Hz alias boundary (-6.94 Hz jump) and biases
    # residual_cfo by ~0.5 Hz. Bit-identical when no carrier is flagged.
    noisy_sync = equalizer.noisy_carriers(hd["n0_pre_k"])
    hot_sync = equalizer.hot_carriers(np.array([row for run in runs for row in run])) | noisy_sync
    if hot_sync.all():
        hot_sync[:] = False
    if hot_sync.any():
        keep_sync = ~hot_sync
        d = sum(np.sum((run[i + 1] * np.conj(run[i]))[keep_sync]) for run in runs for i in range(len(run) - 1))
    else:
        d = sum(np.sum(run[i + 1] * np.conj(run[i])) for run in runs for i in range(len(run) - 1))
    coarse = float(np.angle(d) / (2 * np.pi * NSYM / FS))
    # slips are searched for once, here (residual CFO left: the search is blind to a common phase), and replayed
    raw0, hp, _, slips = _demod_frames(z, p0, n_air, 0, phi_ref, band=db, pil=pil, close_at=kc if scb else None)
    k0_off = BANDS[db].k0 - b.spec.k0
    exclude_db = noisy_sync[k0_off : k0_off + BANDS[db].nc]
    if db != band:
        # n4 (4 carriers hosted on n10's 10): carry the wider sync band's power
        # check too so a mid-carrier tone hitting 2 of 4 carriers cannot mask
        # itself in n4's 4-carrier median.
        exclude_db = exclude_db | hot_sync[k0_off : k0_off + BANDS[db].nc]
    if sc and _sc_intra_spec(BANDS[db].nc) is not None:
        fine, cfo_dt = _sc_residual_cfo(raw0, hp, kc if scb else None, n_air, db, exclude_db)
        ref_cfo = 0.0 if known is not None else coarse
        cfo_res = fine + round((ref_cfo - fine) * cfo_dt) / cfo_dt
    else:
        fine = equalizer.residual_cfo(hp, exclude_db)
        # a known burst's CFO is the same transmitter's, measured seconds ago: its
        # header may be the part that faded, so the alias nearest no change
        cfo_res = resolve_alias(fine, 0.0 if known is not None else coarse)
    z = dsp.freq_correct(z, cfo_res)
    raw, hp, _, _ = _demod_frames(z, p0, n_air, 0, phi_ref, band=db, slips_in=slips, pil=pil, close_at=kc if scb else None)
    def _sup_rows(rows: np.ndarray) -> np.ndarray:
        hot_s = equalizer.hot_carriers(rows)
        if len(exclude_db) == rows.shape[1]:
            hot_s = hot_s | exclude_db
        return np.where(hot_s, 0.0, rows) if hot_s.any() and not hot_s.all() else rows

    if sc:  # the support from whole pilot symbols: a frame's scattered pilots are 96 ms apart
        hp = _sup_rows(_sc_full(z, hd, raw, hp, kc, 0, db)[0])
    # the window and the reported delay on the support as before the noise
    # gate (the gear shifter was trained on it); the estimate on the gated one
    seen = equalizer.delay_support(hp, bb=ofdm.band(db).bb)
    support = equalizer.delay_support(hp, bb=ofdm.band(db).bb, n0=hd["n0_pre"])
    if sc:  # a handful of pilot symbols: the ungated support picks up noise
        seen = support
    shift = equalizer.window_shift(seen)
    raw, hp, steps, _ = _demod_frames(z, p0, n_air, shift, phi_ref, band=db, slips_in=slips, pil=pil, close_at=kc if scb else None)
    support = (support[0] - shift, support[1] - shift)
    seen = (seen[0] - shift, seen[1] - shift)
    if sc:
        # once more through the moved window: with a path outside the first one, a pilot symbol's profile
        # has lobes of its own (mpd, locked on the later path: one 32 samples before the earlier)
        support = equalizer.delay_support(_sup_rows(_sc_full(z, hd, raw, hp, kc, shift, db)[0]), bb=ofdm.band(db).bb, n0=hd["n0_pre"])
        more = equalizer.window_shift(support)
        if more:
            shift += more
            raw, hp, steps, _ = _demod_frames(z, p0, n_air, shift, phi_ref, band=db, slips_in=slips, pil=pil, close_at=kc if scb else None)
            support = (support[0] - more, support[1] - more)
        seen = support
        raw, est, info = _sc_estimate(z, hd, raw, hp, kc, n_f, shift, support, spec)
        return dict(
            spec=spec, n_cw=n_cw, raw=raw, est=est, acq=acq, band=db, hp=hp, kc=kc, sc=info, frames_in=n_in,
            cfo=acq.freq_offset + cfo_res, p0=p0, shift=shift, steps=steps, slips=slips, phi_ref=phi_ref, support=support,
            seen_support=seen, preamble_start=hd["start"], score=hd["score"],
        )
    # the copy frame's pilot keeps the pilot grid regular for the channel
    # estimate; its symbols are then dropped from the data
    est = data_channel(hp, support, db, hd["n0_pre"], spec.clip,
                       n0_pre_k=hd["n0_pre_k"] if db == band else None, n_frames=n_f)
    if kc is not None:
        raw = np.delete(raw, kc, axis=0)
        est["h"], est["mse"] = np.delete(est["h"], kc, axis=0), np.delete(est["mse"], kc, axis=0)
        if np.ndim(est.get("n0_k")) == 3:  # per frame (config.EST_WINDOW)
            est["n0_k"] = np.delete(est["n0_k"], kc, axis=0)
    return dict(
        spec=spec, n_cw=n_cw, raw=raw, est=est, acq=acq, band=db, hp=hp, kc=kc, frames_in=n_in,
        cfo=acq.freq_offset + cfo_res, p0=p0, shift=shift, steps=steps, slips=slips, phi_ref=phi_ref, support=support, seen_support=seen,
        preamble_start=hd["start"], score=hd["score"],
    )


def _sc_residual_cfo(raw0: np.ndarray, hp: np.ndarray, kc: int | None, n_air: int, db: str,
                     exclude_db: np.ndarray | None = None) -> tuple[float, float]:
    """Fine residual CFO (Hz) and pair lag (s) from intra-frame ds-spaced pilot comb pairs."""
    b = ofdm.band(db)
    nc = b.nc
    intra = _sc_intra_spec(nc)
    assert intra is not None
    K, syms, n_combs = intra
    ds = syms[1] - syms[0]
    dt = ds * NSYM / FS
    u0s, u1s, ks_all = [], [], []
    for fa in range(n_air):
        if fa == kc:
            continue
        ks = sc_frame_cells(nc, fa)[1][: nc // K]
        scale = b.pilot[ks]
        rows = [raw0[fa, s_, ks] / scale for s_ in syms]
        for m in range(len(syms) - 1):
            u0s.append(rows[m])
            u1s.append(rows[m + 1])
            ks_all.append(ks)
        if fa + 1 in (kc, n_air):
            u0s.append(rows[-1])
            u1s.append(hp[fa + 1, ks])
            ks_all.append(ks)
        elif n_combs == 1 and fa + 1 < n_air:
            u0s.append(rows[-1])
            u1s.append(raw0[fa + 1, 0, ks] / scale)
            ks_all.append(ks)
    if not u0s:
        return 0.0, dt
    u0, u1, kk = np.concatenate(u0s), np.concatenate(u1s), np.concatenate(ks_all)
    cnt = np.bincount(kk, minlength=nc)
    act = cnt > 0
    pk = np.zeros(nc)
    pk[act] = np.bincount(kk, 0.5 * (np.abs(u0) ** 2 + np.abs(u1) ** 2), minlength=nc)[act] / cnt[act]
    hot = (~act) | (pk > equalizer.HOT_CARRIER * float(np.median(pk[act])))
    if exclude_db is not None and len(exclude_db) == nc:
        hot = hot | np.asarray(exclude_db, dtype=bool)
    if hot[act].all():
        hot = ~act
    prod_k = np.bincount(kk, (u1 * np.conj(u0)).real, minlength=nc) + 1j * np.bincount(
        kk, (u1 * np.conj(u0)).imag, minlength=nc)
    d = np.sum(prod_k[~hot])
    if int(cnt.max()) >= equalizer.HOT_INNOV_MIN_PILOTS - 1 and np.abs(d) > 0:
        ph = d / np.abs(d)
        innov_k = np.zeros(nc)
        innov_k[act] = np.bincount(kk, np.abs(u1 - ph * u0) ** 2, minlength=nc)[act] / cnt[act]
        hot2 = hot | (innov_k > equalizer.HOT_INNOV * float(np.median(innov_k[act])))
        if not hot2[act].all() and not np.array_equal(hot2, hot):
            d = np.sum(prod_k[~hot2])
    return (float(np.angle(d) / (2 * np.pi * dt)) if np.abs(d) > 0 else 0.0), dt


def _sc_full(z: np.ndarray, hd: HeaderRead, raw: np.ndarray, hp: np.ndarray, kc: int | None, shift: int, db: str) -> tuple:
    """A scattered-pilot burst's whole pilot symbols, through the data's
    window: (LS (P, nc), their seconds from the first frame's start): the
    preamble's last REF_REPEATS repeats' mean, the header's closing pilot, the
    copy frame's two, the burst's closing one (hp's rows, _demod_frames')."""
    b = ofdm.band(hd["band"])
    R, p0 = b.spec.preamble_repeats, hd["p0"]
    ref = min(R, REF_REPEATS)
    u = hd["start"] + PREAMBLE_CP + np.arange(R - ref, R) * M
    # (frames on a narrower band than their header's: n4 on n10's: its carriers of the sync band's symbols)
    col = BANDS[db].k0 - b.spec.k0 + np.arange(BANDS[db].nc)
    rows = [(np.mean([b.demod_window(z, int(a) + shift, DEMOD_BACKOFF) for a in u], axis=0) / b.pilot)[col]]
    t = [(u.mean() - NCP - p0) / FS]  # as symbol starts, like the frames'
    rows.append((b.demod_window(z, p0 - NSYM + NCP + shift, DEMOD_BACKOFF) / b.pilot)[col])
    t.append(-NSYM / FS)
    if kc is not None:
        rows += [hp[kc], raw[kc, SYMS_PER_FRAME - 1] / copy_close(b)]  # (a copy frame: data band = sync band)
        t += [kc * FRAME_SAMPLES / FS, (kc * FRAME_SAMPLES + (SYMS_PER_FRAME - 1) * NSYM) / FS]
    return np.array(rows + [hp[-1]]), np.array(t + [(len(hp) - 1) * FRAME_SAMPLES / FS])


def _sc_combs(nc: int) -> bool:
    """Whether the band's pilots are k3d2's three combs (symbols 0, 2, 4; carriers 0, 1, 2 mod 3)."""
    if _sc_intra_spec(nc) is not None:
        return False
    P, PK = sc_cells(nc)[:2]
    return nc % 3 == 0 and np.array_equal(P // nc, 2 * (PK % 3))


SC_FAST_HZ = 3.0  # the symbol-to-symbol spread estimate takes over from the frame-to-frame one past this


def _sc_pilot_counts(nc: int, frames: np.ndarray) -> np.ndarray:
    if sc_period(nc) == 1 and _sc_intra_spec(nc) is None:
        return np.full(nc, len(frames), dtype=int)
    out = np.zeros(nc, dtype=int)
    per = sc_period(nc)
    for f_ in range(per):
        cnt = int(np.sum(frames % per == f_))
        if cnt:
            out += cnt * np.bincount(sc_frame_cells(nc, f_)[1], minlength=nc)
    return out


def _sc_estimate(z, hd, raw, hp, kc, n_f, shift, support, spec) -> tuple:
    """receive()'s raw and estimate from scattered pilots, in the shapes the
    pilot-row frames give ((n_f, 6, nc) with a row of the frame's pilots first
    and its 5 nc data cells in sending order after; h and mse (n_f, 5, nc)),
    plus what a decision-directed pass needs (sc_refine)."""
    db = spec.band
    b = ofdm.band(db)
    nc, n_air = b.nc, len(raw)
    per = sc_period(nc)
    intra = _sc_intra_spec(nc)
    c = ClipConsts(*(spec.clip or CLIP[db]))
    g = c.gain  # the pilots clip with the data at any length (measured within 2%; ACE modes' whole pilot symbols 7% off)
    norm = np.array([f for f in range(n_air) if f != kc])
    cells = raw[norm].reshape(n_f, -1)
    est: DataEstimate = {
        "h": np.empty((0, 0, 0), dtype=np.complex128),
        "mse": np.empty((0, 0, 0)),
        "n0": hd["n0_pre"],
        "n0_k": np.empty(0),
        "n0_snr": hd["n0_pre"],
        "p_sig": 1.0,
        "spread_hz": 1.0,
        "filter_spread_hz": 1.0,
        "slack": equalizer.SUPPORT_SLACK,
        "clip_ratio": c.ratio,
        "gain": c.gain_for(n_f),
        "band": db,
    }
    ls = np.empty((n_f, nc), dtype=np.complex128)
    for f_ in range(per):
        idx = np.flatnonzero(norm % per == f_)
        if len(idx):
            P_f, PK_f, _ = sc_frame_cells(nc, f_)
            ls[idx] = cells[idx[:, None], P_f] / b.pilot[PK_f]
    ratio = _sc_cal(spec)[1]
    sb = ofdm.band(hd["band"])  # the preamble's band (n4's frames follow n10's)
    R = sb.spec.preamble_repeats
    reps = np.array([sb.demod_window(z, hd["start"] + PREAMBLE_CP + r * M, HEADER_BACKOFF) for r in range(R)])
    V = np.vander(np.arange(R) / R, 4)
    n0 = hd["n0_pre"]
    if R > 4:
        n0_k = np.sum(np.abs(reps - V @ np.linalg.lstsq(V, reps, rcond=None)[0]) ** 2, axis=0) / (R - 4)
        n0 = float(np.min(equalizer.per_carrier_noise(n0_k, R - 4)))
    # per-carrier pilot power for hot-carrier masking in p_y
    n_p_norm = _sc_pilot_counts(nc, norm)
    pk_sum = np.zeros(nc)
    for f_ in range(per):
        idx = np.flatnonzero(norm % per == f_)
        if len(idx):
            PK_f = sc_frame_cells(nc, f_)[1]
            pk_sum += np.bincount(PK_f, np.sum(np.abs(ls[idx]) ** 2, axis=0), minlength=nc)
    act_p = n_p_norm > 0
    pk = np.zeros(nc)
    pk[act_p] = pk_sum[act_p] / n_p_norm[act_p]
    med_pk = float(np.median(pk[act_p]))
    cool_ls = np.zeros_like(ls, dtype=bool)
    for f_ in range(per):
        idx = np.flatnonzero(norm % per == f_)
        if len(idx):
            PK_f = sc_frame_cells(nc, f_)[1]
            cool_ls[idx] = (pk[PK_f] <= SC_HOT_POWER * med_pk)[None, :]
    p_y = max(float(np.mean(np.abs(ls[cool_ls]) ** 2)) - n0, 1e-12) / (1 + ratio)
    n0_y = n0 + ratio * p_y  # the scattered pilots': thermal + clip
    p_sig = p_y / g**2
    est.update(n0=n0, n0_snr=n0, n0_k=np.full(nc, n0), p_sig=p_sig)
    max_comb_d = 52  # fk3d2 (K=3) single-comb Nyquist window (52 samples = 6.5 ms)
    slack = int(np.clip((max_comb_d - (support[1] - support[0])) // 2, 0, equalizer.SUPPORT_SLACK))
    est["slack"] = slack if nc >= 24 else _sc_slack(nc)
    # slow or fast by same-carrier pilots `per` frames apart (or intra-frame ds symbols apart on intra layouts)
    pair = np.flatnonzero(np.diff(norm) == 1)
    rho, rho_dt = 0.0, equalizer.FRAME_S
    hm = hp[norm]
    pair_p = np.array([i for i in range(len(norm) - per) if norm[i + per] - norm[i] == per], dtype=int)
    if len(pair_p):
        rho = abs(np.vdot(hm[pair_p], hm[pair_p + per])) / (len(pair_p) * nc * p_y)
        rho_dt = per * equalizer.FRAME_S
    if intra is not None and rho <= SC_SLOW_RHO and len(hm):
        K_in, syms_in, _ = intra
        w_in = nc // K_in
        rho_in = abs(np.vdot(hm[:, :w_in], hm[:, -w_in:])) / (len(hm) * w_in * p_y)
        if rho_in > 0.60:
            rho, rho_dt = rho_in, (syms_in[-1] - syms_in[0]) * NSYM / FS
    loo_rule = nc >= 24
    if loo_rule:
        spread = None
    elif rho > SC_SLOW_RHO:
        spread = float(np.clip(SC_WIDEN * np.sqrt(-2 * np.log(min(rho, 0.99999))) / (np.pi * rho_dt), 0.05, SC_FAST_HZ))
    else:
        spread = None
    full, t_full = _sc_full(z, hd, raw, hp, kc, shift, db)
    # observations: (symbol from the first frame's start, carrier, kind 0 scattered | 1 whole | 2 the preamble's mean)
    sc_syms = np.empty((n_f, nc), dtype=float)
    sc_ks = np.empty((n_f, nc), dtype=int)
    for f_ in range(per):
        idx = np.flatnonzero(norm % per == f_)
        if len(idx):
            P_f, PK_f, _ = sc_frame_cells(nc, f_)
            sc_syms[idx] = norm[idx, None] * SYMS_PER_FRAME + (P_f // nc)[None, :]
            sc_ks[idx] = PK_f[None, :]
    o_sym = np.concatenate([sc_syms.ravel(), np.repeat(np.round(t_full * FS / NSYM, 2), nc)])
    o_k = np.concatenate([sc_ks.ravel(), np.tile(np.arange(nc), len(full))])
    kind = np.ones(len(full), dtype=int)
    kind[0] = 2
    o_kind = np.concatenate([np.zeros(n_f * nc, dtype=int), np.repeat(kind, nc)])
    o_val = np.concatenate([(ls / g).ravel(), full.ravel()])
    sup = support
    slep = nc >= 24
    # When delay span fits inside single-comb Nyquist (52 samples for K=3), _sc_slepian handles all spreads;
    # wider supports on K=3 (e.g. nvis-d's 56 samples at 1 Hz) keep _sc_separable for slow spreads <= SC_SEP_HZ.
    slep_all = slep and intra is not None and (sup[1] - sup[0]) <= max_comb_d
    fast_est = _sc_slepian if slep else _sc_joint

    # detect steady tones on carriers with zero scattered pilots via whole pilot rows
    u_mask = n_p_norm == 0
    hot_u = np.zeros(nc, dtype=bool)
    extra_u = np.zeros(nc)
    if u_mask.any() and len(full) > 1:
        from scipy.stats import gamma

        f_rows = full[1:]
        hs_f, _, r_f, keep_f = equalizer.freq_smooth(f_rows, sup, b.bb, est["slack"])
        if nc - r_f >= 2:
            n_fr = len(f_rows)
            pk_f = np.mean(np.abs(f_rows - hs_f) ** 2, axis=0) / np.maximum(keep_f, 1e-3)
            qf = gamma.ppf(0.999, n_fr) / n_fr
            ex_f = np.maximum(pk_f - float(np.median(pk_f)), 0.0) / g**2
            hot_u = u_mask & (pk_f > float(np.median(pk_f)) * n_fr / max(n_fr - 1 / 3, 0.5) * qf) & (
                ex_f >= SC_HOT_EXCESS * (n0_y / g**2))
            extra_u = np.where(hot_u, ex_f, 0.0)

    kw_init = (1 + extra_u / (n0_y / g**2)) if hot_u.any() else None
    hot_pow = act_p & (pk > SC_HOT_POWER * med_pk)
    kw_pick = kw_init
    if hot_pow.any():
        ex_p = np.where(hot_pow, np.maximum(pk - med_pk, 0.0) / g**2, extra_u)
        kw_pick = 1 + ex_p / (n0_y / g**2)

    if spread is None:
        cand = [v for v in SC_SPREADS if v >= (SC_FAST_HZ if len(pair) else equalizer.DEFAULT_SPREAD_HZ)]
        if loo_rule:
            cand = [*SC_SPREADS_SLOW, *SC_SPREADS]
        search = _sc_interior(norm, o_sym[o_kind > 0], n_air)
        n_p_search = _sc_pilot_counts(nc, search)

        def pick(est_fn, cand):
            prep = _sc_slepian_prep(o_sym, o_k, o_kind, o_val, db, sup, n0_y / g**2 / p_sig, n0 / p_sig, kw_pick) \
                if est_fn is _sc_slepian else None
            fits = [(est_fn(o_sym, o_k, o_kind, o_val, search, db, sup, v, n0_y / g**2 / p_sig, n0 / p_sig, p_sig, kw_pick,
                            _prep=prep) if prep is not None else
                     est_fn(o_sym, o_k, o_kind, o_val, search, db, sup, v, n0_y / g**2 / p_sig, n0 / p_sig, p_sig, kw_pick))[2:]
                    for v in cand]
            valid = n_p_search > 0
            i = int(np.argmin([np.sum(f[0][valid & (~hot_pow)]) for f in fits]))
            hot_s = _sc_hot(*fits[i], n_p_search, n0_y / g**2)[0] | hot_pow
            keep_s = valid & (~hot_s)
            if not keep_s.any():
                keep_s = valid
            loo = np.array([np.sum(f[0][keep_s]) for f in fits])
            if not loo_rule:
                return cand[int(np.argmin(loo))]
            i = int(np.argmin(loo))
            return max(v for v, e in zip(cand, loo) if e <= loo[i] + SC_LOO_TOL * max(loo[i] - np.sum(fits[i][1][keep_s]), 0.0))

        if slep_all:
            spread = pick(_sc_slepian, cand)
        elif SC_SEP_HZ and loo_rule:
            spread = pick(_sc_separable, cand)
            if spread > SC_SEP_HZ:
                spread = pick(fast_est, [v for v in cand if v > SC_SEP_HZ])
        else:
            spread = pick(fast_est, cand)
    sep = loo_rule and spread <= SC_SEP_HZ
    est.update(spread_hz=min(spread, SC_SEP_REPORT_HZ if sep else 4.0), filter_spread_hz=spread)

    def joint(kw=None):
        est_fn = _sc_slepian if slep_all else (_sc_separable if sep else fast_est)
        return est_fn(o_sym, o_k, o_kind, o_val, norm, db, sup, spread, n0_y / g**2 / p_sig, n0 / p_sig, p_sig, kw)

    h6, mse6, loo_k, model_k = joint(kw_init)
    hot, extra = _sc_hot(loo_k, model_k, n_p_norm, n0_y / g**2)
    if hot_u.any():
        hot = hot | hot_u
        extra = np.where(hot_u, extra_u, extra)
    kw = kw_init
    for _ in range(SC_HOT_PASSES if (hot & act_p).any() else 0):
        kw = 1 + extra / (n0_y / g**2)
        h6, mse6, loo_k, model_k = joint(kw)
        extra = np.where(hot & act_p, np.maximum(extra + (loo_k - model_k) / np.maximum(n_p_norm, 1), 0.0), extra)
    cool = (~hot) & act_p
    if not cool.any():
        cool = act_p
    excess = max(float(np.sum(loo_k[cool] - model_k[cool])), 0.0) / max(int(np.sum(n_p_norm[cool])), 1)
    if excess > 0.25 * n0_y / g**2:
        n0, n0_y = n0 + excess * g**2, n0_y + excess * g**2
        extra = np.maximum(extra - excess, 0.0)
        h6, mse6, _, _ = joint(None if kw is None else 1 + extra / (n0_y / g**2))
    est.update(n0=n0, n0_snr=n0, n0_k=np.full(nc, n0))
    if hot.any():  # per data cell, as h: a frame's cells (n_f, 5, nc) are in sending order, not by carrier
        nk_base = n0 + extra * g**2
        nk_arr = np.empty((n_f, DATA_SYMS_PER_FRAME, nc))
        for f_ in range(per):
            idx = np.flatnonzero(norm % per == f_)
            if len(idx):
                Dc_f = sc_frame_cells(nc, f_)[2]
                nk_arr[idx] = nk_base[Dc_f % nc].reshape(1, DATA_SYMS_PER_FRAME, nc)
        est["n0_k"] = nk_arr
    h_flat, m_flat = h6.reshape(n_f, -1), mse6.reshape(n_f, -1)
    hj = np.empty((n_f, DATA_SYMS_PER_FRAME, nc), dtype=np.complex128)
    mj = np.empty((n_f, DATA_SYMS_PER_FRAME, nc))
    rv = np.empty((n_f, SYMS_PER_FRAME, nc), dtype=np.complex128)
    pil_k = _sc_pil(db, n_air, kc)
    rv[:, 0] = hp[norm] * (b.pilot[pil_k[norm, :, 1]] if pil_k is not None and pil_k.ndim == 3 else b.pilot)
    for f_ in range(per):
        idx = np.flatnonzero(norm % per == f_)
        if len(idx):
            Dc_f = sc_frame_cells(nc, f_)[2]
            hj[idx] = h_flat[idx[:, None], Dc_f].reshape(len(idx), DATA_SYMS_PER_FRAME, nc)
            mj[idx] = m_flat[idx[:, None], Dc_f].reshape(len(idx), DATA_SYMS_PER_FRAME, nc)
            rv[idx, 1:] = cells[idx[:, None], Dc_f].reshape(len(idx), DATA_SYMS_PER_FRAME, nc)
    z6 = np.zeros((n_air, SYMS_PER_FRAME * nc), dtype=np.complex128)
    w6 = np.zeros(z6.shape)
    t_rows = (np.arange(n_air)[:, None] * SYMS_PER_FRAME + np.arange(SYMS_PER_FRAME)) * NSYM / FS
    info = dict(z=z6, w=w6, t_rows=t_rows, norm=norm, h=hj, mse=mj, full=full, ls=ls, n0_y=n0_y)
    est["h"], est["mse"] = g * hj, g**2 * mj
    return rv, est, info


SC_SPREADS = (2.0, 3.0, 4.5, 7.0, 10.0, 15.0)  # Hz: the fast spread's candidates where no comb measures it (gold)
SC_SLOW_RHO = 0.35  # frame-apart pilot correlation over this: the spread from it (under ~3.2 Hz)
SC_WIDEN = 1.25  # that spread, widened: too narrow a filter costs more than too wide (equalizer.filter_spread)
SC_FRAMES = 4  # the joint estimate's reach, frames either side, at most (0.7 / spread otherwise)
# ... and at most 2 from SC_JOINT_REACH2_HZ up (w48's 2 Hz mpd took 1.8-2.3 s against TURN_S 1.3 on a CM4 with 3;
# 1.0 s with 2, 10% points within 0.04 dB, see the commit)
SC_JOINT_REACH2_HZ = 2.0
_SC_WEIGHTS: dict = {}  # _sc_joint's, by pattern position and quantized statistics


def _sc_slack(nc: int, support: tuple[int, int] | None = None) -> int:
    """The joint estimate's delay-support padding, samples either side: equalizer.SUPPORT_SLACK, or on a narrow
    band its delay resolution (FS / bandwidth: 16 on n10, 40 on n4), where a second path inside it reads as one
    delay (equalizer.support_slack's reason: unpadded, n10 on mpp held to -8 dB of estimate error)."""
    if nc < 24:
        return int(round(FS / (nc * RS)))
    if support is not None:
        return int(np.clip((52 - (support[1] - support[0])) // 2, 0, equalizer.SUPPORT_SLACK))
    return equalizer.SUPPORT_SLACK


SC_HOT_POWER = 8.0  # _sc_estimate's p_y leaves out carriers this many times the median's power (a tone)
SC_SEARCH_MIN = 4  # minimum interior frames for spread search (0: every frame)


def _sc_interior(frames: np.ndarray, whole: np.ndarray, n_air: int) -> np.ndarray:
    """The spread search's frames: those whose widest reach (SC_FRAMES frames either side) holds every frame's
    pilots and no whole pilot symbol (whole: their symbols), which all share one pattern of observations, so one
    solve a candidate spread; every frame when fewer than SC_SEARCH_MIN do."""
    reach = SYMS_PER_FRAME * SC_FRAMES + 3
    c = frames * SYMS_PER_FRAME + 2.5
    inner = (c - reach >= 0) & (c + reach <= n_air * SYMS_PER_FRAME)
    inner &= ~np.any(np.abs(whole[None, :] - c[:, None]) < reach, axis=1)
    return frames[inner] if SC_SEARCH_MIN and inner.sum() >= SC_SEARCH_MIN else frames


SC_HOT_PASSES = 1
SC_HOT_EXCESS = 1.0


def _sc_hot(loo_k, model_k, n_p_k, nn: float) -> tuple:
    """Carriers whose pilots' leave-one-out error (loo_k, model_k) is over the 99th percentile of
    n_p_k observations' scatter about the median carrier's (equalizer.per_carrier_noise's test), and by at least
    SC_HOT_EXCESS x the pilots' noise `nn`: -> (hot, their error past the model per pilot (0 elsewhere))."""
    from scipy.stats import gamma

    n_p = np.broadcast_to(np.asarray(n_p_k, dtype=float), loo_k.shape)
    act = n_p > 0
    if not act.any():
        return np.zeros(loo_k.shape, dtype=bool), np.zeros_like(loo_k)
    n_eff = np.maximum(n_p, 1.0)
    t = np.zeros_like(loo_k)
    t[act] = loo_k[act] / np.maximum(model_k[act], 1e-30)
    q = gamma.ppf(0.99, n_eff) / n_eff
    extra = np.where(act, np.maximum(loo_k - model_k, 0.0) / n_eff, 0.0)
    hot = act & (t > float(np.median(t[act])) * n_eff / np.maximum(n_eff - 1 / 3, 0.5) * q) & (extra >= SC_HOT_EXCESS * nn)
    if hot[act].all():
        hot[:] = False
    return hot, np.where(hot, extra, 0.0)


SC_SEP_HZ = 1.5
SC_SEP_REPORT_HZ = 1.0
_SC_SEP: dict = {}  # _sc_separable's time weights, by pattern and quantized statistics


def _sc_separable(o_sym, o_k, o_kind, o_val, frames, band: str, support, spread: float, nr_sc: float,
                  nr_full: float, p_sig: float, kw=None) -> tuple:
    """_sc_joint's estimate, separably (its arguments and returns): per carrier in time from its own
    observations (Wiener, Gaussian Doppler correlation, the same reach), then per symbol across carriers
    through the delay support (LMMSE), the time step's outputs taken as unbiased observations with the noise
    their posterior variance implies (e / (1 - e), the prior's unit power)."""
    b = ofdm.band(band)
    nc = b.nc
    q = lambda v, n: round(np.log2(max(v, 1e-9)) * n)  # noqa: E731
    qs, q1, q2 = q(spread, 8), q(nr_sc, 6), q(nr_full, 6)
    spread, nr = 2 ** (qs / 8), np.array([2 ** (q1 / 6), 2 ** (q2 / 6), 2 ** (q2 / 6) / REF_REPEATS])
    kq = None if kw is None else np.array([q(v, 2) for v in kw])
    nk = np.ones(nc) if kq is None else 2 ** (kq / 2)
    reach = SYMS_PER_FRAME * float(np.clip(np.ceil(0.7 / spread / equalizer.FRAME_S), 1, SC_FRAMES)) + 3
    slack = _sc_slack(nc)
    d = np.arange(support[0] - slack, support[1] + slack + 1)
    Bd = np.exp(-2j * np.pi * np.outer(b.bb, d) / FS)
    Rf = Bd @ Bd.conj().T / len(d)
    tsym = NSYM / FS
    S = SYMS_PER_FRAME
    F = len(frames)
    per = sc_period(nc)
    p_syms_by_mod = []
    for f_ in range(per):
        P_f, PK_f, _ = sc_frame_cells(nc, f_)
        p_syms_by_mod.append([(P_f // nc)[PK_f == k] for k in range(nc)])
    xh = np.zeros((F, S, nc), dtype=np.complex128)
    ev = np.full((F, S, nc), 1e8)
    pil_obs = []
    for k in range(nc):
        m = o_k == k
        ts, kd, vs = o_sym[m], o_kind[m], o_val[m]
        order = np.argsort(ts, kind="stable")
        ts, kd, vs = ts[order], kd[order], vs[order]
        for i, f in enumerate(frames):
            lo, hi = np.searchsorted(ts, [f * S + 2.5 - reach, f * S + 2.5 + reach])
            if lo == hi:
                continue
            rel, kk = ts[lo:hi] - f * S, kd[lo:hi]
            psyms = p_syms_by_mod[int(f) % per][k]
            key = (qs, q1, q2, int(kq[k]) if kq is not None else 0, psyms.tobytes(), rel.tobytes(), kk.tobytes())
            got = _SC_SEP.get(key)
            if got is None:
                A = equalizer._doppler_corr((rel[:, None] - rel[None, :]) * tsym, spread)
                A[np.diag_indices_from(A)] += nr[kk] * nk[k]
                Bm = equalizer._doppler_corr((np.arange(S)[:, None] - rel[None, :]) * tsym, spread)
                W = np.linalg.solve(A, Bm.T).T  # (S, n), real
                e = np.clip(1 - np.sum(W * Bm, axis=1), 1e-9, 0.999)
                cols = np.array([int(np.flatnonzero((rel == s_p) & (kk == 0))[0]) for s_p in psyms], dtype=int)
                got = (W / (1 - e)[:, None], e / (1 - e), cols)
                if len(_SC_SEP) >= 4096:
                    _SC_SEP.clear()
                _SC_SEP[key] = got
            Wu, nu, cols = got
            v_win = vs[lo:hi]
            xh[i, :, k] = Wu @ v_win
            ev[i, :, k] = nu
            for s_p, col in zip(psyms, cols):
                pil_obs.append((i, k, int(s_p), float(Wu[int(s_p), col]), v_win[col]))
    h = np.empty((F, S, nc), dtype=np.complex128)
    mse = np.empty((F, S, nc))
    loo, model = np.zeros(nc), np.zeros(nc)
    cache: dict = {}
    for i in range(F):
        for s_ in range(S):
            e = ev[i, s_]
            ck = e.tobytes()
            G = cache.get(ck)
            if G is None:
                G = np.linalg.solve(Rf + np.diag(e), Rf).conj().T
                cache[ck] = G = (G, np.maximum(np.real(np.diag(Rf) - np.sum(G * Rf.conj(), axis=1)), 0.0))
            h[i, s_] = G[0] @ xh[i, s_]
            mse[i, s_] = p_sig * G[1]
    for i, k, s_p, ownt_v, yp_v in pil_obs:
        own = min(float(np.real(cache[ev[i, s_p].tobytes()][0][k, k])) * ownt_v, 0.99)
        loo[k] += abs((yp_v - h[i, s_p, k]) / (1 - own)) ** 2
        model[k] += p_sig * nr[0] * nk[k] / (1 - own)
    return h, mse, loo, model


@lru_cache(maxsize=512)
def _slepian_basis(bb_bytes: bytes, d0: int, d1: int, slack: int) -> tuple:
    """Orthonormal DPSS/Slepian basis Uf (nc, r) and eigenvalues lam (r,) of the delay-support covariance Rf.
    For uniform carrier grids uses Grunbaum's (1981) commuting symmetric tridiagonal matrix whose eigenvalues
    are strictly simple (avoiding arbitrary rotations inside Rf's near-degenerate top eigenspace)."""
    bb = np.frombuffer(bb_bytes)
    nc = len(bb)
    d = np.arange(d0 - slack, d1 + slack + 1)
    W = len(d)
    Bd = np.exp(-2j * np.pi * np.outer(bb, d) / FS)
    Rf = Bd @ Bd.conj().T / W
    df = float(bb[1] - bb[0]) if nc > 1 else float(RS)
    M_eff = FS / df if df > 0 else 0.0
    if nc > 1 and W < M_eff and np.allclose(np.diff(bb), df, rtol=0.0, atol=1e-6):
        k = np.arange(nc, dtype=np.float64)
        diag = -np.cos(np.pi * W / M_eff) * np.cos(np.pi * (2.0 * k - (nc - 1)) / M_eff)
        off = np.sin(np.pi * (k[:-1] + 1.0) / M_eff) * np.sin(np.pi * (nc - 1.0 - k[:-1]) / M_eff)
        T = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        ev_t, V = np.linalg.eigh(T)
        V = V[:, np.argsort(-ev_t, kind="stable")]
        d_mid = 0.5 * ((d0 - slack) + (d1 + slack))
        evecs = np.exp(-2j * np.pi * bb * d_mid / FS)[:, None] * V
        evals = np.real(np.einsum("ij,ik,kj->j", evecs.conj(), Rf, evecs))
    else:
        evals, evecs = np.linalg.eigh(Rf)
        order = np.argsort(-evals, kind="stable")
        evals, evecs = evals[order], evecs[:, order]
    keep = evals > 1e-4 * evals[0]
    lam = np.ascontiguousarray(evals[keep])
    Uf = np.ascontiguousarray(evecs[:, keep])
    Uf_abs2 = np.ascontiguousarray(np.abs(Uf) ** 2)
    rem_diag = np.ascontiguousarray(np.maximum(np.real(np.diag(Rf)) - Uf_abs2 @ lam, 0.0))
    Rf = np.ascontiguousarray(Rf)
    for a in (lam, Uf, Uf_abs2, rem_diag, Rf):
        a.setflags(write=False)
    return Rf, Uf, lam, Uf_abs2, rem_diag


def _sc_slepian_prep(o_sym, o_k, o_kind, o_val, band: str, support, nr_sc: float, nr_full: float, kw=None) -> tuple:
    """Frequency projection onto the Slepian subspace (independent of Doppler spread)."""
    b = ofdm.band(band)
    nc = b.nc
    q = lambda v, n: round(np.log2(max(v, 1e-9)) * n)  # noqa: E731
    q1, q2 = q(nr_sc, 6), q(nr_full, 6)
    nr = np.array([2 ** (q1 / 6), 2 ** (q2 / 6), 2 ** (q2 / 6) / REF_REPEATS])
    kq = None if kw is None else np.array([q(v, 2) for v in kw])
    nk = np.ones(nc) if kq is None else 2 ** (kq / 2)
    d0, d1 = int(support[0]), int(support[1])
    intra = _sc_intra_spec(nc)
    max_d = 79 if (intra is not None and intra[0] == 2) else 52
    if d1 - d0 > max_d:
        mid = (d0 + d1) // 2
        d0, d1 = mid - max_d // 2, mid - max_d // 2 + max_d
    slack = _sc_slack(nc, (d0, d1))
    Rf, Uf, lam, Uf_abs2, rem_diag = _slepian_basis(
        np.asarray(b.bb, dtype=np.float64).tobytes(), d0, d1, int(slack))
    r = len(lam)
    order = np.argsort(o_sym, kind="stable")
    os_s, ok_s, okd_s, ov_s = o_sym[order], o_k[order], o_kind[order], o_val[order]
    u_syms, starts = np.unique(os_s, return_index=True)
    ends = np.append(starts[1:], len(os_s))
    Q = len(u_syms)
    y_tilde = np.empty((Q, r), dtype=np.complex128)
    nu_row = np.empty((Q, r))
    Wu_list = [None] * Q
    row_cache: dict = {}
    kq_bytes = None if kq is None else kq.tobytes()
    full_k = np.arange(nc)
    for q_idx in range(Q):
        a, b_ = starts[q_idx], ends[q_idx]
        ks, kd, yq = ok_s[a:b_], okd_s[a:b_], ov_s[a:b_]
        rkey = (ks.tobytes(), kd.tobytes(), kq_bytes)
        rg = row_cache.get(rkey)
        if rg is None:
            if kq is None and len(ks) == nc and np.array_equal(ks, full_k) and np.all(kd == kd[0]):
                Wu = Uf.conj().T
                nu = np.full(r, nr[kd[0]])
            else:
                eta = nr[kd] * nk[ks]
                B = Rf[ks, :] @ Uf
                G = np.linalg.solve(Rf[np.ix_(ks, ks)] + np.diag(eta), B).conj().T
                c = np.clip(np.real(np.sum(G * B.T, axis=1)), 1e-8 * lam, (1.0 - 1e-7) * lam)
                Wu = G * (lam / c)[:, None]
                nu = np.minimum(lam * (lam - c) / c, 1e8 * lam)
            row_cache[rkey] = rg = (Wu, nu)
        Wu, nu = rg
        Wu_list[q_idx] = Wu
        y_tilde[q_idx] = Wu @ yq
        nu_row[q_idx] = nu
    return nc, nr, nk, Uf, lam, Uf_abs2, rem_diag, u_syms, starts, ends, ok_s, okd_s, ov_s, y_tilde, nu_row, Wu_list


def _sc_slepian(o_sym, o_k, o_kind, o_val, frames, band: str, support, spread: float, nr_sc: float, nr_full: float,
                p_sig: float, kw=None, _prep=None) -> tuple:
    """Slepian-mode separable 2-D LMMSE: project each pilot symbol's comb onto orthonormal eigenvectors of Rf
    (unbiased mode observations + posterior noise), then run r decoupled 1-D real Wiener solves in time."""
    if _prep is None:
        _prep = _sc_slepian_prep(o_sym, o_k, o_kind, o_val, band, support, nr_sc, nr_full, kw)
    nc, nr, nk, Uf, lam, Uf_abs2, rem_diag, u_syms, starts, ends, ok_s, okd_s, ov_s, y_tilde, nu_row, Wu_list = _prep
    qs = round(np.log2(max(spread, 1e-9)) * 8)
    spread = 2 ** (qs / 8)
    reach = SYMS_PER_FRAME * float(np.clip(np.ceil(0.7 / spread / equalizer.FRAME_S), 1, SC_FRAMES)) + 3
    tsym = NSYM / FS
    S = SYMS_PER_FRAME
    F = len(frames)
    r = len(lam)
    h = np.empty((F, S, nc), dtype=np.complex128)
    mse = np.empty((F, S, nc))
    loo, model = np.zeros(nc), np.zeros(nc)
    t_cache: dict = {}
    s_grid = np.arange(S)[:, None]
    for i, f in enumerate(frames):
        lo, hi = np.searchsorted(u_syms, [f * S + 2.5 - reach, f * S + 2.5 + reach])
        rel = u_syms[lo:hi] - f * S
        nu_win = nu_row[lo:hi]
        tkey = (rel.tobytes(), nu_win.tobytes())
        tg = t_cache.get(tkey)
        if tg is None:
            J = len(rel)
            Kt = equalizer._doppler_corr((rel[:, None] - rel[None, :]) * tsym, spread)
            Kdt = equalizer._doppler_corr((s_grid - rel[None, :]) * tsym, spread)
            A_t = np.broadcast_to(Kt, (r, J, J)).copy()
            idx_j = np.arange(J)
            A_t[:, idx_j, idx_j] += (nu_win / lam).T
            W_t = np.linalg.solve(A_t, np.broadcast_to(Kdt.T, (r, J, S))).swapaxes(1, 2)  # (r, S, J)
            e_mode = lam[:, None] * np.maximum(1.0 - np.sum(W_t * Kdt[None, :, :], axis=2), 0.0)
            mse_f = p_sig * (rem_diag[None, :] + e_mode.T @ Uf_abs2.T)
            t_cache[tkey] = tg = (W_t, mse_f)
        W_t, mse_f = tg
        a_hat = np.einsum("rsj,jr->sr", W_t, y_tilde[lo:hi])
        h[i] = a_hat @ Uf.T
        mse[i] = mse_f
        for j in range(len(rel)):
            rj = rel[j]
            if 0.0 <= rj < S and rj == int(rj):
                q_idx = lo + j
                a, b_ = starts[q_idx], ends[q_idx]
                if okd_s[a] == 0:
                    s_p = int(rj)
                    ks, yq, Wu = ok_s[a:b_], ov_s[a:b_], Wu_list[q_idx]
                    own = np.minimum(np.real(np.sum(Uf[ks, :].T * (W_t[:, s_p, j][:, None] * Wu), axis=0)), 0.99)
                    np.add.at(loo, ks, np.abs((yq - h[i, s_p, ks]) / (1.0 - own)) ** 2)
                    np.add.at(model, ks, p_sig * nr[0] * nk[ks] / (1.0 - own))
    return h, mse, loo, model


def _sc_joint(o_sym, o_k, o_kind, o_val, frames, band: str, support, spread: float, nr_sc: float, nr_full: float,
              p_sig: float, kw=None) -> tuple:
    """2-D LMMSE of the channel at every cell of `frames` (on-air frame
    indices) from the observations within reach in time, all carriers."""
    b = ofdm.band(band)
    nc = b.nc
    q = lambda v, n: round(np.log2(max(v, 1e-9)) * n)  # noqa: E731
    qs, q1, q2 = q(spread, 8), q(nr_sc, 6), q(nr_full, 6)
    spread, nr = 2 ** (qs / 8), np.array([2 ** (q1 / 6), 2 ** (q2 / 6), 2 ** (q2 / 6) / REF_REPEATS])
    kq = None if kw is None else np.array([q(v, 2) for v in kw])  # to 1.5 dB
    nk = np.ones(nc) if kq is None else 2 ** (kq / 2)
    reach = SYMS_PER_FRAME * float(np.clip(np.ceil(0.7 / spread / equalizer.FRAME_S), 1,
                                           2 if spread >= SC_JOINT_REACH2_HZ else SC_FRAMES)) + 3
    slack = _sc_slack(nc)
    d = np.arange(support[0] - slack, support[1] + slack + 1)
    Rf = None
    tsym = NSYM / FS
    cs, ck = np.divmod(np.arange(SYMS_PER_FRAME * nc), nc)
    loo, model = np.zeros(nc), np.zeros(nc)
    h = np.empty((len(frames), SYMS_PER_FRAME * nc), dtype=np.complex128)
    mse = np.empty(h.shape)
    order = np.argsort(o_sym, kind="stable")
    o_sym, o_k, o_kind, o_val = o_sym[order], o_k[order], o_kind[order], o_val[order]
    for i, f in enumerate(frames):
        P, PK, _ = sc_frame_cells(nc, int(f))
        lo, hi = np.searchsorted(o_sym, [f * SYMS_PER_FRAME + 2.5 - reach, f * SYMS_PER_FRAME + 2.5 + reach])
        rel, k, kd = o_sym[lo:hi] - f * SYMS_PER_FRAME, o_k[lo:hi], o_kind[lo:hi]
        key = (band, int(support[0]), int(support[1]), slack, qs, q1, q2, rel.tobytes(), k.tobytes(), kd.tobytes(),
               None if kq is None else kq.tobytes())
        got = _SC_WEIGHTS.get(key)
        if got is None:
            if Rf is None:
                Bd = np.exp(-2j * np.pi * np.outer(b.bb, d) / FS)
                Rf = Bd @ Bd.conj().T / len(d)
            A = equalizer._doppler_corr((rel[:, None] - rel[None, :]) * tsym, spread) * Rf[k[:, None], k[None, :]]
            A[np.diag_indices_from(A)] += nr[kd] * nk[k]
            Bm = equalizer._doppler_corr((cs[:, None] - rel[None, :]) * tsym, spread) * Rf[ck[:, None], k[None, :]]
            W = np.linalg.solve(A, Bm.conj().T).conj().T
            col = np.array([np.flatnonzero((rel == c // nc) & (k == kk) & (kd == 0))[0] for c, kk in zip(P, PK)])
            got = (W, np.maximum(1 - np.real(np.sum(W * Bm.conj(), axis=1)), 0.0), col, np.real(W[P, col]))
            if len(_SC_WEIGHTS) >= 256:
                _SC_WEIGHTS.clear()
            _SC_WEIGHTS[key] = got
        h[i] = got[0] @ o_val[lo:hi]
        mse[i] = p_sig * got[1]
        own = np.minimum(got[3], 0.99)
        np.add.at(loo, PK, np.abs((o_val[lo:hi][got[2]] - h[i][P]) / (1 - own)) ** 2)
        np.add.at(model, PK, p_sig * nr[0] * nk[PK] / (1 - own))
    return h.reshape(len(frames), SYMS_PER_FRAME, nc), mse.reshape(len(frames), SYMS_PER_FRAME, nc), loo, model


_SC_CAL: dict = {}


def _cal_payload(n: int, state: list) -> bytes:
    """n bytes of splitmix64 (state[0] its counter): _sc_cal's payloads, which native (modem/scattered.cpp) makes
    the same way; numpy's generator has no port there."""
    out = b""
    m = (1 << 64) - 1
    while len(out) < n:
        state[0] = (state[0] + 0x9E3779B97F4A7C15) & m
        z = state[0]
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
        out += (z ^ (z >> 31)).to_bytes(8, "little")
    return out[:n]


def _sc_cal(spec) -> tuple:
    """A submode's scattered pilots as its transmitter leaves them, measured
    once on clean bursts of its own: (their gain over the data cells', their
    distortion over their own power)."""
    if codes.payload_bytes(spec) <= 0:
        return (1.0, spec.clip.ratio)
    key = (spec.name, SC_RESTORE_ACE, tuple(SC_BANDS))
    if key not in _SC_CAL:
        _SC_CAL[key] = (1.0, spec.clip.ratio)  # while measuring
        b = ofdm.band(spec.band)
        state = [0]
        num_p = np.zeros(b.nc, dtype=np.complex128)
        den_p = np.zeros(b.nc)
        num_d = np.zeros(b.nc, dtype=np.complex128)
        den_d = np.zeros(b.nc)
        parts = []
        for _ in range(2):
            n_cw = min(4, max_codewords(spec.sync_band))
            bits = np.stack([codes.encode(spec, _cal_payload(codes.payload_bytes(spec), state), 0, index=i)
                             for i in range(n_cw)])
            x = modulate_bits(codes.spread(bits, spec.bits_per_cu), spec)
            r = receive(np.concatenate([np.zeros(800), x, np.zeros(800)]),
                        known=dict(start=800 + LEADIN_SAMPLES, cfo=0.0, spec=spec, n_cw=n_cw))
            X = constellation.modulate(codes.spread(bits, spec.bits_per_cu), constellation.load(spec.constellation))
            Y = data_symbols(r["raw"]).reshape(len(r["raw"]), -1)
            X = X.reshape(Y.shape)
            ls, norm = r["sc"]["ls"], r["sc"]["norm"]
            for idx, fa in enumerate(norm):
                _, PK_f, Dc_f = sc_frame_cells(b.nc, int(fa))
                yx = Y[idx] * X[idx].conj()
                num_d += np.bincount(Dc_f % b.nc, yx.real, b.nc) + 1j * np.bincount(Dc_f % b.nc, yx.imag, b.nc)
                den_d += np.bincount(Dc_f % b.nc, np.abs(X[idx]) ** 2, b.nc)
                num_p += np.bincount(PK_f, ls[idx].real, b.nc) + 1j * np.bincount(PK_f, ls[idx].imag, b.nc)
                den_p += np.bincount(PK_f, minlength=b.nc)
            parts.append((ls, norm))
        p_mask = den_p > 0
        gd = num_d / den_d
        gp = np.zeros(b.nc, dtype=np.complex128)
        gp[p_mask] = num_p[p_mask] / den_p[p_mask]
        err = sum(float(np.sum(np.abs(ls[idx] - gp[sc_frame_cells(b.nc, int(fa))[1]]) ** 2))
                  for ls, norm in parts for idx, fa in enumerate(norm))
        tot_gp = sum(float(np.sum(np.abs(gp[sc_frame_cells(b.nc, int(fa))[1]]) ** 2))
                     for _, norm in parts for fa in norm)
        _SC_CAL[key] = (float(np.real(np.vdot(gd[p_mask], gp[p_mask]) / np.vdot(gd[p_mask], gd[p_mask]))),
                        err / max(tot_gp, 1e-30))
    return _SC_CAL[key]


def sc_refine(r: dict, z: np.ndarray | None, w: np.ndarray | None) -> tuple:
    """Decision-directed pass on a scattered-pilot burst: data cells z with
    inverse variances w ((n_f, 5, nc) as est["h"], the pilots' units) ->
    h, mse (n_f, 5, nc), the pilots' units."""
    i, b = r["sc"], ofdm.band(r["band"])
    norm = i["norm"]
    n_f = len(norm)
    z6, w6 = i["z"].copy(), i["w"].copy()
    zf, wf, hf = z.reshape(n_f, -1), w.reshape(n_f, -1), i["h"].reshape(n_f, -1)
    for idx, fa in enumerate(norm):
        Dc = sc_frame_cells(b.nc, int(fa))[2]
        z6[fa, Dc] = np.where(wf[idx] > 0, zf[idx] - hf[idx], 0)
        w6[fa, Dc] = wf[idx]
    prior = dict(r["est"], p_sig=float(np.mean(i["mse"])))
    dh, mse = equalizer.refine(np.zeros((0, b.nc), dtype=np.complex128), np.zeros(0), z6.reshape(-1, b.nc),
                               w6.reshape(-1, b.nc), i["t_rows"], r["support"], prior, b.bb)
    dh6, mse6 = dh.reshape(len(i["t_rows"]), -1), mse.reshape(len(i["t_rows"]), -1)
    out_h = np.empty((n_f, DATA_SYMS_PER_FRAME, b.nc), dtype=np.complex128)
    out_m = np.empty((n_f, DATA_SYMS_PER_FRAME, b.nc))
    for idx, fa in enumerate(norm):
        Dc = sc_frame_cells(b.nc, int(fa))[2]
        out_h[idx] = dh6[fa, Dc].reshape(DATA_SYMS_PER_FRAME, b.nc)
        out_m[idx] = mse6[fa, Dc].reshape(DATA_SYMS_PER_FRAME, b.nc)
    return i["h"] + out_h, out_m


def resolve_alias(fine: float, coarse: float) -> float:
    """The alias of `fine` (known modulo the pilot rate) nearest `coarse`."""
    return fine + round((coarse - fine) * equalizer.FRAME_S) / equalizer.FRAME_S


def data_channel(h_pilot: np.ndarray, support: tuple[int, int], band: str = "w",
                 n0_pre: float = np.inf, clip: ClipConsts | None = None, n0_pre_k=None, n_frames: int | None = None,
                 _whole: bool = False) -> DataEstimate:
    """equalizer.estimate, rescaled from the pilots' channel to the one
    the data see. `clip` overrides config.CLIP[band] (clipper studies).
    `n_frames`: data frames, when the pilots include a header copy's."""
    from .config import EST_WINDOW

    n_air = len(h_pilot) - 1
    if EST_WINDOW and not _whole and n_air > EST_WINDOW:
        # studies (FM plan, decoding while a burst arrives): each window of
        # frames from its own pilots and one either side, as a receiver that
        # estimates as frames arrive would see them
        parts = []
        for a in range(0, n_air, EST_WINDOW):
            lo, hi = max(0, a - 1), min(n_air, a + EST_WINDOW + 1)
            e = data_channel(h_pilot[lo:hi + 1], support, band, n0_pre, clip, n0_pre_k, n_frames or n_air, _whole=True)
            w = slice(a - lo, a - lo + min(EST_WINDOW, n_air - a))
            nk = np.broadcast_to(np.asarray(e.get("n0_k", e["n0"]), dtype=float), (e["h"].shape[2],))
            parts.append((e, e["h"][w], e["mse"][w], np.broadcast_to(nk, (len(e["h"][w]), 1, len(nk)))))
        est = dict(parts[0][0])
        est["h"] = np.concatenate([p[1] for p in parts])
        est["mse"] = np.concatenate([p[2] for p in parts])
        est["n0_k"] = np.concatenate([p[3] for p in parts])
        for k in ("n0", "n0_snr", "p_sig", "spread_hz"):
            est[k] = float(np.mean([p[0][k] for p in parts]))
        return est
    est = equalizer.estimate(h_pilot, support, bb=ofdm.band(band).bb, n0_pre=n0_pre, n0_pre_k=n0_pre_k)
    c = ClipConsts(*(clip or CLIP[band]))
    est["clip_ratio"] = c.ratio
    g = c.gain_for(len(h_pilot) - 1 if n_frames is None else n_frames)
    est["h"] = est["h"] * g
    est["mse"] = est["mse"] * g**2
    est["gain"] = g
    est["band"] = band
    return est


def noise_var(h: np.ndarray, est: DataEstimate) -> np.ndarray:
    """Per-cu variance of thermal plus clip noise; add the estimate's own
    MSE when `h` is an estimate.

    Clip noise also leaks between carriers (it is not cyclic within a
    symbol), which this ignores. It only shows in notches deeper than
    ~25 dB, where it sets the high-SNR floor on slow selective channels;
    a constant leak term measured no better. Per-symbol cyclic clipping
    would remove it at the source (plan step 8)."""
    return est.get("n0_k", est["n0"]) + est["clip_ratio"] * np.abs(h) ** 2


def symbol_noise_scale(y: np.ndarray, h: np.ndarray, var: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(n_f, 5, 1) factors >= 1 for the data symbols' noise variance: each
    symbol's mean over carriers of |y - h x|^2 / var to its nearest point x,
    against the burst's median symbol. On Gaussian noise every symbol sits
    near the median (decision errors pull all of them down alike); a
    symbol a click hit stands far above it. ponytail: hard decisions, one
    figure per symbol; a click inside a symbol's CP is caught too (it is in
    no window and moves nothing)."""
    d = np.abs(y[..., None] - h[..., None] * points).min(axis=-1)  # (n_f, 5, nc)
    r = np.mean(d**2 / var, axis=-1, keepdims=True)
    return np.maximum(1.0, r / max(float(np.median(r)), 1e-12))


def soft_bits(raw: np.ndarray, h: np.ndarray, var: np.ndarray, spec: SubmodeSpec) -> np.ndarray:
    """LLRs from data symbols `raw`, channel `h` and per-cu noise `var`."""
    return constellation.llr(data_symbols(raw), h, var, constellation.load(spec.constellation))


def data_symbols(raw: np.ndarray) -> np.ndarray:
    """(n_f, 5, nc): receive()'s raw past each frame's pilot, aligned with est["h"]."""
    return raw[:, 1:]


def burst_soft(r: Received, est: DataEstimate | None = None) -> np.ndarray:
    """(n_cw, coded_bits) soft bits in mapping order on `est` (r["est"], or
    arq/phy's DD re-estimate): thermal, clip and estimation noise."""
    spec, est = r["spec"], est or r["est"]
    var = _data_var(r, est)
    return np.asarray(codes.despread(soft_bits(r["raw"], est["h"], var, spec), r["n_cw"], spec.bits_per_cu,
                                     codes.spread_group(spec)))


def _data_var(r: Received, est: DataEstimate) -> np.ndarray:
    """Each data cell's noise variance: thermal, clip and estimation, and
    per symbol (SYMBOL_NOISE) what clicks added."""
    var = np.broadcast_to(noise_var(est["h"], est) + est["mse"], est["h"].shape)
    if SYMBOL_NOISE:
        pts = constellation.load(r["spec"].constellation)
        var = var * symbol_noise_scale(data_symbols(r["raw"]), est["h"], var, pts)
    return var


def slot_cells(r: Received, slot: int) -> np.ndarray:
    """The data cells (flat index over the whole burst's (frames, 5, nc))
    holding `slot`'s symbols, in its order: codes.spread's layout, round-robin
    over the burst or over the slot's window (config.SPREAD_WINDOW)."""
    spec, n = r["spec"], r["n_cw"]
    syms = n * spec.coded_bits // spec.bits_per_cu
    return codes.despread(np.arange(syms), n, 1, codes.spread_group(spec))[slot]


def ready_slots(r: Received, margin: int = 1) -> list[int]:
    """The slots whose every symbol is in the frames received so far
    (r["frames_in"]), `margin` frames short of the last, so each has a
    pilot after it for the channel estimate; a complete burst's are all."""
    spec, n = r["spec"], r["n_cw"]
    n_f = n * spec.frames_per_cw
    frames = r.get("frames_in", n_f)
    have = (frames if frames >= n_f else frames - margin) * DATA_SYMS_PER_FRAME * BANDS[spec.band].nc
    return [s for s in range(n) if slot_cells(r, s).max() < have]


def slot_soft(r: Received, est: DataEstimate, slot: int) -> np.ndarray:
    """burst_soft(r, est)[slot] alone: codes.spread deals codeword symbols
    round-robin, so the slot's are every n_cw-th of the burst's (of its
    window's: config.SPREAD_WINDOW). On a partial burst, a ready_slots one."""
    var = _data_var(r, est)
    cells = slot_cells(r, slot)  # the slot's, in its order (within what is in, on a partial burst)
    pick = lambda a: a.reshape(-1)[cells]  # noqa: E731
    return constellation.llr(pick(data_symbols(r["raw"])), pick(est["h"]), pick(var),
                             constellation.load(r["spec"].constellation))


def snr_db(est: equalizer.ChannelEstimate, band: str) -> float:
    """Pilot SNR (dB) on `band`'s carriers, in the SNR_REF_BW_HZ convention."""
    return 10 * np.log10(est["p_sig"] / est.get("n0_snr", est["n0"]) * BANDS[band].nc * RS / SNR_REF_BW_HZ)


def symbol_times(r: Received) -> tuple[np.ndarray, np.ndarray]:
    """On-air seconds from the first frame pilot of each data symbol, (n_f,
    5) as est["h"]'s, and of each of r["hp"]'s pilots; the header copy's
    frame accounted for."""
    n_f, kc = len(r["raw"]), r["kc"]
    air = np.arange(n_f) + (0 if kc is None else (np.arange(n_f) >= kc))
    t_rows = (air[:, None] * SYMS_PER_FRAME + np.arange(1, SYMS_PER_FRAME)) * NSYM / FS
    t_pilot = np.arange(len(r["hp"])) * SYMS_PER_FRAME * NSYM / FS
    return t_rows, t_pilot


def demodulate(x: np.ndarray, bands=None, accept: Accept | None = None) -> Burst:
    """First burst found in `x`. Raises SyncError if none decodes."""
    return decode_received(receive(x, bands, accept))


def decode_received(r: Received) -> Burst:
    """receive()'s result -> the decoded Burst (plain modem use: no CRC masks)."""
    spec = r["spec"]
    soft = burst_soft(r)
    decoded = codes.decode_many(spec, soft)
    return Burst(
        submode=spec,
        payloads=[d[0] for d in decoded],
        crc_ok=[d[1] for d in decoded],
        freq_offset=r["cfo"],
        preamble_start=r["preamble_start"],
        snr_db=float(snr_db(r["est"], spec.band)),
        soft=soft,
    )
