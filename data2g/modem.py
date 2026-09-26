"""Burst modem: codeword payloads <-> passband audio.

Burst: silence | preamble | header | n_cw codewords of whole frames |
closing pilot | silence. Frames are SSTVAE's (1 pilot + 5 data symbols,
every carrier of the submode's band). The closing pilot is one extra pilot symbol so
the last frame is interpolated between two pilots rather than
extrapolated from one; it costs 24 ms and matters most to short bursts.

RX follows SSTVAE's `Modem.demodulate`: preamble acquisition, channel
reference from the preamble, soft-combined Golay header, per-frame
sample-clock tracking from the pilot phase slope. Channel estimation is
data2g.equalizer (burst-wide, 2-D LMMSE), not SSTVAE's Catmull-Rom.
"""

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import codes, constellation, equalizer
from .config import (
    BANDS,
    CLIP,
    clip_consts,
    DATA_SYMS_PER_FRAME,
    DEMOD_BACKOFF,
    FRAME_SAMPLES,
    FS,
    HEADER_COPY_AFTER,
    HEADER_COPY_BANDS,
    LEADIN_SAMPLES,
    LEADOUT_SAMPLES,
    M,
    MAX_CODEWORDS,
    NCP,
    NSYM,
    PREAMBLE_CP,
    PROTOCOL_VERSION,
    RS,
    SNR_REF_BW_HZ,
    SUBMODES,
    SYMS_PER_FRAME,
    SubmodeSpec,
)
from .waveform import ofdm
from .waveform.dsp import freq_correct, to_baseband, tx_condition
from .waveform.sync import SyncError, acquire

__all__ = ["modulate", "demodulate", "receive", "Burst", "SyncError"]

PILOT = ofdm.pilot_sequence()  # wide band's; per-band code uses band.pilot
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
# (w carries the robust ACKs); 0.25 costs nothing measurable. The data
# false locks it lets through are made cheap by tnc.Receiver's supersede
# search (a better header later replaces a pending one), not by the floor.
STREAM_COMMIT_SCORE = 0.5  # a header this good commits even while another band's is still arriving
# a streaming receiver commits on the first header copy alone at this
# score; under it, it waits for the second copy (wrong words read 0.25-0.35)
COPY_COMMIT_SCORE = 0.45
# w48: 0.26 since its header went to 4 symbols (384 bits): its noise reads
# top out at 0.248 (3000 measured), its reads off w bursts at 0.243 (-8..+30
# dB); true w48 headers score 0.35 median at -6 dB (0.30 had cut ~20% there).
HEADER_MIN_SCORE = {"w": 0.25, "n10": 0.30, "w48": 0.26}
# Handicap on the header score of every hypothesis but acquisition's own
# (its runner-ups, and the +-1, +-2 repeat alignments): near -8 dB wide a
# true header scores ~0.26 and noise read at some other hypothesis beat
# it. Wide AWGN -8 dB end to end: fail 6.25% -> 3.5% (0.08: 4.0%); no
# change on mpd, where the other hypotheses do the rescuing.
ALT_PENALTY = 0.04
# (sync band name, header index) -> submode
BY_INDEX = {(s.sync_band, s.index): s for s in SUBMODES.values()}
# bands that carry a header, i.e. whose preambles the receiver looks for
SYNC_BANDS = [name for name, b in BANDS.items() if not b.sync]


def _hosts(band: str) -> bool:
    """Whether other bands' frames follow this band's header."""
    return any(b.sync == band for b in BANDS.values())


@dataclass
class Burst:
    submode: SubmodeSpec
    payloads: list[bytes]
    crc_ok: list[bool]
    freq_offset: float
    preamble_start: int
    snr_db: float  # pilot-based, in the SNR_REF_BW_HZ convention
    soft: np.ndarray | None = None  # (n_cw, coded_bits) soft bits, mapping order: codes.combine for resends


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
    return min(HEADER_COPY_AFTER, n_f) if band in HEADER_COPY_BANDS else None


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


def burst_seconds(spec: SubmodeSpec, n_cw: int) -> float:
    """On-air length of an n_cw-codeword burst, lead-in/out silence included."""
    sb = BANDS[spec.sync_band]
    n = (LEADIN_SAMPLES + sb.preamble_samples + header_samples(spec.sync_band)
         + (frames_on_air(spec, n_cw) * SYMS_PER_FRAME + 1) * NSYM + LEADOUT_SAMPLES)
    return n / FS


def _crc6(v: int) -> int:
    """CRC-6 (x^6 + x + 1) over 10 bits, register seeded with the version."""
    reg = PROTOCOL_VERSION & 0x3F
    for i in range(9, -1, -1):
        fb = ((reg >> 5) & 1) ^ ((v >> i) & 1)
        reg = ((reg << 1) & 0x3F) ^ (0x3 if fb else 0)
    return reg


def _word_bits(word: int) -> np.ndarray:
    return (word >> np.arange(15, -1, -1)) & 1


def header_bits(submode: int, n_cw: int, band: str = "w") -> np.ndarray:
    """-> coded header bits for the band. Word: submode (4) | n_cw - 1 (6)
    | CRC-6. Was 8 + CRC-4 until measured: tried at 5 alignments and on
    every band whose detector fires, a 4-bit check let wrong headers
    through after the true one failed (11 of 200 wide ACKs at -6 dB on
    mpd read as a burst of hundreds of codewords)."""
    if not 1 <= n_cw <= MAX_CODEWORDS:
        raise ValueError(f"1..{MAX_CODEWORDS} codewords per burst")
    v = (submode << 6) | (n_cw - 1)
    return (_word_bits((v << 6) | _crc6(v)) @ header_code(band)) % 2


@lru_cache(maxsize=None)
def _header_signs(band: str) -> np.ndarray:
    """(65536, N) +-1 for every message, message index = the 16-bit word."""
    msgs = (np.arange(2**16)[:, None] >> np.arange(15, -1, -1)) & 1
    return (1 - 2 * ((msgs @ header_code(band)) % 2)).astype(np.float32)


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
            cw = MAX_CODEWORDS
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


@lru_cache(maxsize=None)
def _valid_words(band: str, accept: Accept | None = None) -> np.ndarray:
    """Words with a right CRC-6 and a submode defined on the band (and
    taken by `accept`)."""
    v = np.arange(2**10)
    if accept is None:
        v = v[np.isin(v >> 6, [i for b, i in BY_INDEX if b == band])]
    else:
        lim = {SUBMODES[n].index: cw for n, cw in accept.max_cw if SUBMODES[n].sync_band == band}
        v = v[[(x >> 6) in lim and (x & 0x3F) < lim[x >> 6] for x in v]]
    return (v << 6) | np.array([_crc6(int(x)) for x in v], dtype=np.int64).reshape(-1)


@lru_cache(maxsize=None)
def _valid_signs(band: str, accept: Accept | None = None) -> np.ndarray:
    return _header_signs(band)[_valid_words(band, accept)]


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
    v = word >> 6
    return word, (BY_INDEX[(band, v >> 6)], (v & 0x3F) + 1), score


# --- transmit ---------------------------------------------------------------

def modulate(payloads: list[bytes], submode: str | SubmodeSpec, rvs=None) -> np.ndarray:
    """Codeword payloads (each exactly codes.payload_bytes long) -> unit-RMS
    audio. `rvs`: each codeword's redundancy version (codes.rv_positions),
    all 0 by default."""
    spec = SUBMODES[submode] if isinstance(submode, str) else submode
    if not 1 <= len(payloads) <= MAX_CODEWORDS:
        raise ValueError(f"1..{MAX_CODEWORDS} codewords per burst, got {len(payloads)}")
    bits = np.stack([codes.encode(spec, p, rv, index=i) for i, (p, rv) in enumerate(zip(payloads, rvs or [0] * len(payloads)))])
    return modulate_bits(codes.spread(bits, spec.bits_per_cu), spec)


def modulate_bits(bits: np.ndarray, spec: SubmodeSpec) -> np.ndarray:
    """Coded bits in burst order (whole codewords, after `codes.spread`)
    -> unit-RMS audio. The codec-free half of `modulate`, for
    measurements that send raw bits."""
    b = ofdm.band(spec.band)
    data = constellation.modulate(bits, constellation.load(spec.constellation))
    x = burst_waveform(data.reshape(-1, DATA_SYMS_PER_FRAME, b.nc), spec)
    # the sync band's filter: it contains the data band (BandSpec.sync)
    return tx_condition(
        x, spec.headroom, b.spec.clip_overshoot,
        active=slice(LEADIN_SAMPLES, len(x) - LEADOUT_SAMPLES), bandpass=BANDS[spec.sync_band].tx_bandpass,
    )


def burst_waveform(data: np.ndarray, spec: SubmodeSpec) -> np.ndarray:
    """(n_f, 5, nc) data symbols -> the unclipped burst waveform."""
    b = ofdm.band(spec.band)
    n_f = len(data)
    n_cw = n_f // spec.frames_per_cw
    sb = ofdm.band(spec.sync_band)
    hdr = constellation.modulate(header_bits(spec.index, n_cw, spec.sync_band), QPSK)
    kc = copy_frame(spec.sync_band, n_f)
    if kc is not None:  # the header copy: a frame of its own (data band = sync band here)
        h = hdr.reshape(sb.spec.header_syms, sb.nc)
        row = np.concatenate([h, h[:DATA_SYMS_PER_FRAME - len(h)]])
        data = np.insert(data, kc, row, axis=0)
        n_f += 1
    syms = np.empty((n_f * SYMS_PER_FRAME + 1, b.nc), dtype=np.complex128)
    syms[::SYMS_PER_FRAME] = b.pilot  # frame pilots and the closing pilot
    syms[:-1].reshape(n_f, SYMS_PER_FRAME, b.nc)[:, 1:] = data
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


def _demod_frames(
    z: np.ndarray, p: int, n_f: int, shift: int, phi_ref: float,
    steps_in: np.ndarray | None = None, band: str = "w",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Demodulate n_f frames and the closing pilot, window moved `shift`
    samples later than nominal.

    Sample-clock drift is followed SSTVAE's way (pilot phase slope across
    carriers, slow EMA, +-2 sample steps), but each step's phase is then
    undone so the symbols share one timing reference and the channel
    looks smooth to the interpolator; SSTVAE interpolated straight across
    the steps. Returns raw (n_f, 6, NC), pilots (n_f + 1, NC), and the
    per-frame step total. `steps_in` replays a previous run's steps
    instead of tracking (scripts/eq_floor.py's genie uses it).
    """
    b = ofdm.band(band)
    phi_ref = phi_ref + 2 * np.pi * RS * shift / FS  # the shift's own slope
    p = p + shift
    raw = np.zeros((n_f + 1, SYMS_PER_FRAME, b.nc), dtype=np.complex128)
    steps = np.zeros(n_f + 1)
    tau_ema, powers, total = 0.0, [], 0
    for f in range(n_f + 1):
        n_s = SYMS_PER_FRAME if f < n_f else 1
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
        hf = raw[f, 0] / b.pilot
        powers.append(float(np.mean(np.abs(hf) ** 2)))
        if powers[-1] > 0.1 * np.median(powers):
            d = np.angle(np.exp(1j * (_bin_phase_step(hf) - phi_ref)))
            tau_ema += 0.02 * (-d * FS / (2 * np.pi * RS) - tau_ema)
            if abs(tau_ema) >= 2:
                step = int(np.clip(round(tau_ema), -2, 2))
                p += step
                total += step
                tau_ema -= step
        p += FRAME_SAMPLES
    raw *= equalizer.time_shift_phase(steps, b.bb)[:, None, :]
    return raw[:n_f], raw[:, 0] / b.pilot, steps


def _read_header(z: np.ndarray, start: int, band: str = "w", accept: Accept | None = None) -> dict:
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
    h_p = equalizer._freq_smooth(h_p, (HEADER_BACKOFF, HEADER_BACKOFF + NCP), b.bb)[0]
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
    h0, h1 = win[0] / b.pilot, win[SYMS_PER_FRAME] / b.pilot
    a = (np.arange(1, SYMS_PER_FRAME) / SYMS_PER_FRAME)[:, None]
    ys, hs = np.array(win[1:SYMS_PER_FRAME]), (1 - a) * h0 + a * h1
    llr = constellation.llr(ys, hs, np.ones(ys.shape), QPSK).reshape(DATA_SYMS_PER_FRAME, -1)
    out = llr[:n_hdr].copy()
    for i in range(n_hdr, DATA_SYMS_PER_FRAME):
        out[i - n_hdr] += llr[i]
    return out.reshape(-1)


def _best_header(z0: np.ndarray, bands=None, complete: bool = True, accept: Accept | None = None) -> tuple:
    """-> (header read, acquisition, frequency-corrected z) of the best
    CRC-valid header in baseband `z0`, over `bands` (default: every sync
    band). `complete`: the burst the header claims must fit in `z0`;
    False for a streaming receiver, which waits for the rest.

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
            acq_b = acquire(z0, band=ofdm.band(name))
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
                waiting = True
                continue
            zb = freq_correct(z0, f)
            for k in (0, -1, 1, -2, 2):
                if not 0 <= start + k * M <= len(z0) - hdr_end:
                    continue
                r = _read_header(zb, start + k * M, name, accept)
                if not complete and r["pending_copy"] and r["score"] < COPY_COMMIT_SCORE:
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


def find_burst(x: np.ndarray, bands=None, accept: Accept | None = None) -> dict:
    """Where the first burst in `x` is, from its preamble and header
    alone, so a streaming receiver knows how much audio to wait for:
    {spec, n_cw, start (first preamble sample), end (one past the closing
    pilot)}. The burst may run past the end of `x`. Raises SyncError."""
    hd, acq, _ = _best_header(to_baseband(np.asarray(x, dtype=np.float64)), bands, complete=False, accept=accept)
    spec, n_cw = hd["hdr"]
    return dict(spec=spec, n_cw=n_cw, start=hd["start"], score=hd["score"], band=hd["band"],
                end=burst_end(hd["p0"], spec, n_cw), p0=hd["p0"], cfo=acq.freq_offset)


# a burst's frame pilots, n_pairs consecutive pairs: pilot_coherence's mean
# over them at or under this is noise (each band's 99th percentile on noise:
# 1/sqrt(carriers) scaled). Real bursts: 0-2% under it at 0 dB, 10-16% at
# -4 dB; locks on noise: 99% under within 4 pairs (0.6 s).
PILOT_PAIRS = 4
PILOT_NOISE = {"w": 0.30, "n10": 0.44, "n4": 0.68, "w48": 0.20}


def pilot_coherence(x: np.ndarray, lock: dict, n_max: int = 8, latest: bool = False) -> list[float]:
    """For a streaming lock (find_burst's dict, same sample origin as x):
    per consecutive pair of the frame pilots already in x (the first n_max
    pairs, or with `latest` the newest), |sum over carriers of y_f
    conj(y_f-1)| / sqrt(energies). A real burst's pilots repeat through a
    slowly varying channel; a false lock's 'pilots' are noise or someone
    else's data, near 1/sqrt(carriers)."""
    spec = lock["spec"]
    b = ofdm.band(spec.band)
    total = frames_on_air(spec, lock["n_cw"]) + 1  # frame pilots and the closing one
    avail = sum(lock["p0"] + f * FRAME_SAMPLES + NCP + M <= len(x) for f in range(total))
    f1 = avail if latest else min(avail, n_max + 1)
    f0 = max(0, f1 - n_max - 1)
    if f1 - f0 < 2:
        return []
    lo = max(0, lock["p0"] + f0 * FRAME_SAMPLES - 2 * NSYM)
    hi = lock["p0"] + (f1 - 1) * FRAME_SAMPLES + 3 * NSYM
    z = freq_correct(to_baseband(np.asarray(x[lo:hi], dtype=np.float64)), lock["cfo"])
    y = [b.demod_window(z, lock["p0"] - lo + f * FRAME_SAMPLES + NCP, DEMOD_BACKOFF) for f in range(f0, f1)]
    return [float(np.abs(np.sum(y[f] * np.conj(y[f - 1])))
                  / (np.sqrt(np.sum(np.abs(y[f]) ** 2) * np.sum(np.abs(y[f - 1]) ** 2)) + 1e-30))
            for f in range(1, len(y))]


def receive(x: np.ndarray, bands=None, accept: Accept | None = None, head: int | None = None) -> dict:
    """Everything up to soft bits for the first burst in `x`: the
    synchronisation state (so an analysis script can replay it on
    another signal) and the equalizer output. `bands`: the sync bands to
    look for (default: all). `head`: the preamble and header lie in the
    first `head` samples (a streaming receiver found them there): search
    only those. Acquisition over a whole 11 s burst took 1-4 s of CPU,
    nearly all of receive's time."""
    z0 = to_baseband(np.asarray(x, dtype=np.float64))
    if head is None:
        hd, acq, z = _best_header(z0, bands, accept=accept)
    else:
        hd, acq, _ = _best_header(z0[:head], bands, complete=False, accept=accept)
        z = freq_correct(z0, acq.freq_offset)
        spec, n_cw = hd["hdr"]
        if burst_end(hd["p0"], spec, n_cw) > len(z0):
            raise SyncError("burst runs past the buffer")
    band = hd["band"]  # the sync band; frames are on spec.band's carriers
    b = ofdm.band(band)
    spec, n_cw = hd["hdr"]
    db = spec.band
    word, hdr_y, h_pre, h_first, p0 = hd["word"], hd["y"], hd["h_pre"], hd["h_first"], hd["p0"]

    n_f = n_cw * spec.frames_per_cw
    kc = copy_frame(band, n_f)
    n_air = n_f + (kc is not None)  # the header copy's frame among the data frames
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
    known = list(hd["y_all"] * np.conj(ref_syms)) + ([h_first] if db == band else [])
    runs = [known]
    if kc is not None:
        # the header copy's frame is known symbols too (pilot, copy, next
        # pilot): the estimate survives a fade on the first copy
        pc = p0 + kc * FRAME_SAMPLES
        ys = [b.demod_window(z, pc + s * NSYM + NCP, HEADER_BACKOFF) for s in range(SYMS_PER_FRAME + 1)]
        h = ref_syms[~layout]
        ref2 = [b.pilot, *np.concatenate([h, h[:DATA_SYMS_PER_FRAME - len(h)]]), b.pilot]
        runs.append([y * np.conj(r) for y, r in zip(ys, ref2)])
    d = sum(np.sum(run[i + 1] * np.conj(run[i])) for run in runs for i in range(len(run) - 1))
    coarse = float(np.angle(d) / (2 * np.pi * NSYM / FS))
    _, hp, _ = _demod_frames(z, p0, n_air, 0, phi_ref, band=db)
    fine = equalizer.residual_cfo(hp)
    cfo_res = resolve_alias(fine, coarse)
    z = freq_correct(z, cfo_res)
    _, hp, _ = _demod_frames(z, p0, n_air, 0, phi_ref, band=db)
    support = equalizer.delay_support(hp, bb=ofdm.band(db).bb)
    shift = equalizer.window_shift(support)
    raw, hp, steps = _demod_frames(z, p0, n_air, shift, phi_ref, band=db)
    support = (support[0] - shift, support[1] - shift)
    # the copy frame's pilot keeps the pilot grid regular for the channel
    # estimate; its symbols are then dropped from the data
    est = data_channel(hp, support, db, hd["n0_pre"], clip_consts(db, spec.headroom),
                       n0_pre_k=hd["n0_pre_k"] if db == band else None, n_frames=n_f)
    if kc is not None:
        raw = np.delete(raw, kc, axis=0)
        est["h"], est["mse"] = np.delete(est["h"], kc, axis=0), np.delete(est["mse"], kc, axis=0)
    return dict(
        spec=spec, n_cw=n_cw, raw=raw, est=est, acq=acq, band=db,
        cfo=acq.freq_offset + cfo_res, p0=p0, shift=shift, steps=steps, phi_ref=phi_ref, support=support,
        preamble_start=hd["start"], score=hd["score"],
    )


def resolve_alias(fine: float, coarse: float) -> float:
    """The alias of `fine` (known modulo the pilot rate) nearest `coarse`."""
    return fine + round((coarse - fine) * equalizer.FRAME_S) / equalizer.FRAME_S


def data_channel(h_pilot: np.ndarray, support: tuple[int, int], band: str = "w",
                 n0_pre: float = np.inf, clip: tuple | None = None, n0_pre_k=None, n_frames: int | None = None) -> dict:
    """equalizer.estimate, rescaled from the pilots' channel to the one
    the data see. `clip` overrides config.CLIP[band] (clipper studies).
    `n_frames`: data frames, when the pilots include a header copy's."""
    est = equalizer.estimate(h_pilot, support, bb=ofdm.band(band).bb, n0_pre=n0_pre, n0_pre_k=n0_pre_k)
    gains, default, ratio = clip or CLIP[band]
    est["clip_ratio"] = ratio
    g = gains.get(len(h_pilot) - 1 if n_frames is None else n_frames, default)
    est["h"] = est["h"] * g
    est["mse"] = est["mse"] * g**2
    est["band"] = band
    return est


def noise_var(h: np.ndarray, est: dict) -> np.ndarray:
    """Per-cu variance of thermal plus clip noise; add the estimate's own
    MSE when `h` is an estimate.

    Clip noise also leaks between carriers (it is not cyclic within a
    symbol), which this ignores. It only shows in notches deeper than
    ~25 dB, where it sets the high-SNR floor on slow selective channels;
    a constant leak term measured no better. Per-symbol cyclic clipping
    would remove it at the source (plan step 8)."""
    return est.get("n0_k", est["n0"]) + est["clip_ratio"] * np.abs(h) ** 2


def soft_bits(raw: np.ndarray, h: np.ndarray, var: np.ndarray, spec: SubmodeSpec) -> np.ndarray:
    """LLRs from data symbols `raw`, channel `h` and per-cu noise `var`."""
    return constellation.llr(raw[:, 1:], h, var, constellation.load(spec.constellation))


def demodulate(x: np.ndarray, bands=None, accept: Accept | None = None) -> Burst:
    """First burst found in `x`. Raises SyncError if none decodes."""
    return decode_received(receive(x, bands, accept))


def decode_received(r: dict) -> Burst:
    """receive()'s result -> the decoded Burst (plain modem use: no CRC masks)."""
    spec, est = r["spec"], r["est"]
    var = noise_var(est["h"], est) + est["mse"]
    soft = codes.despread(soft_bits(r["raw"], est["h"], var, spec), r["n_cw"], spec.bits_per_cu)
    decoded = codes.decode_many(spec, np.asarray(soft))
    return Burst(
        submode=spec,
        payloads=[d[0] for d in decoded],
        crc_ok=[d[1] for d in decoded],
        freq_offset=r["cfo"],
        preamble_start=r["preamble_start"],
        snr_db=float(10 * np.log10(est["p_sig"] / est["n0"] * BANDS[spec.band].nc * RS / SNR_REF_BW_HZ)),
        soft=np.asarray(soft),
    )
