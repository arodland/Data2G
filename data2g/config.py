"""Waveform, burst and submode constants.

The waveform half is SSTVAE's (`sstvae/config.py`, PROTOCOL_VERSION 4),
copied rather than imported so this project owns its on-air format; the
measurements behind each number are recorded there. Two deliberate
differences: the beacon carrier is reclaimed for data (all 24 carriers
carry payload), and the preamble repeat count is ours to raise.
"""

import os
from dataclasses import dataclass

FS = 8000  # audio sample rate, Hz

# --- OFDM waveform (identical to SSTVAE) ------------------------------------
RS = 50  # carrier spacing == symbol rate of one carrier, Hz
NC = 24  # carriers, all of them data
M = FS // RS  # useful symbol length, samples (160)
NCP = 32  # cyclic prefix, samples (4 ms)
NSYM = M + NCP  # 192 samples, 24 ms
CARRIER0 = 950  # Hz; occupied ~925..2125 Hz
FCENTER = 1500  # Hz; baseband conversion frequency

SYMS_PER_FRAME = 6  # 1 pilot + 5 data
DATA_SYMS_PER_FRAME = SYMS_PER_FRAME - 1
FRAME_SAMPLES = SYMS_PER_FRAME * NSYM  # 1152 = 144 ms

# --- preamble / acquisition (SSTVAE values; see its config.py) --------------
# DATA2G_* overrides exist for studies only; the on-air value is the
# default. Both need the matching receiver on the far end.
# 8 was chosen (2026-09-22, scripts/sync_sweep.py) with the lag-M
# detector, under which longer preambles decorrelated across themselves
# at 2 Hz Doppler and detected worse on mpd. The matched-filter
# detector (sync.py) compares neighbouring repeats only, so that no
# longer holds; 16 is being measured (runs/sync_floor_r16.csv). SSTVAE: 4.
# The default for every band (BandSpec.preamble_repeats); the env
# override sets it for all of them, for studies.
PREAMBLE_REPEATS = int(os.environ.get("DATA2G_PREAMBLE_REPEATS", 8))
PREAMBLE_CP = 2 * NCP
# Detection threshold on sync.detection_stat (noise-normalized, the same
# distribution over noise whatever the band), per repeat count: 1.1x the
# largest noise peak over 1200 s (scripts/mf_detect_study.py noise).
# 8 repeats, +-150 Hz search (runs/noise_peaks_150hz.log; the same with
# sync.NOISE_REF_HZ, runs/noise_peaks_150hz_refbins.log): w 21.2, n10
# 19.5, w48 21.9; one threshold for all (per-20 s-chunk 90% peaks agree
# within 0.6 across bands, so the bands' 1200 s maxima differ by luck).
# At +-625 Hz it was 25.5 (w 22.7, n10 23.1, w48 21.9). (16 and 32 repeats
# were measured at 31.0 and 38.5 under an older noise estimate and are not
# used.) The lag-M thresholds this replaced are in runs/config_lagm_backup.py.
PREAMBLE_THRESHOLDS = {8: 24.1}
# Frequency-offset search, +-Hz. SSTVAE searched +-625 Hz because it cost
# nothing there; here it cost ~5x the detector's CPU against +-150 Hz, and
# noise-only peaks ran 5-10% higher (2026-09-24). A mode up to 2400 Hz
# wide can't use the room anyway: +-150 Hz already pushes w48's edge
# carriers into the SSB filter's skirts.
ACQUIRE_REACH_HZ = 150.0
FIRST_PATH_SEARCH = NCP
FIRST_PATH_FRAC = 0.5

# --- header -----------------------------------------------------------------
# 16 bits: submode (4) | n_codewords - 1 (6) | CRC-6, encoded with a
# (192, 16) linear code (codes_data/header_code.npy, d_min 70, picked by
# scripts/design_header.py) onto 4 Gray-QPSK symbols across all 24
# carriers, decoded by exact ML over all 65536 codewords. Replaced two
# Golay(24,12) words with repeats: at the calibrated 16-repeat preamble
# that header was the binding failure on every channel (-6 dB AWGN: 175
# of 200 lost there), and repeating it more made mpd worse (a longer
# span to interpolate the channel over).
# The CRC is seeded with PROTOCOL_VERSION, so a receiver of another
# version (or SSTVAE) accepts a header only by 1-in-64 chance.
HEADER_SYMS = 4
PROTOCOL_VERSION = 12  # 10: first frozen submode table; n4 on n10 sync (2026-09-23); 11: header copy;
# 12: polar codewords carry CRC-24 (k + 8, payloads kept); n10's header 5 | 5 (CW_BITS, 2026-10-06,
# in place: development, one operator)
# A second header copy, time-diverse, on the 4-symbol headers (w, w48): a
# frame of its own (pilot, the 4 header symbols, the first again) after
# data frame HEADER_COPY_AFTER, or after the last on a shorter burst. The
# receiver combines the copies' LLRs. scripts/header_diversity.py (2000
# bursts a cell): w48 MPP -1 dB header loss 8.5% -> 2.4%, MPD -1 14.5% ->
# 3.8%; w wrong headers accepted at MPD -3 dB 10.2% -> 4.0%. Spaced beat
# contiguous (an 8-symbol header: w48 MPP -1 4.8%); 1-4 frames later alike.
HEADER_COPY_BANDS = ("w", "w48")
HEADER_COPY_AFTER = 2
MAX_CODEWORDS = 64  # the header carries n_cw - 1 in 6 bits (CW_BITS: fewer on n10)
# The header word's 10 bits split per sync band: submode | n_cw - 1, 6 bits
# of count unless listed. n10's 16 indices (n10 and n4) were full; its
# bursts never need 64 codewords (at 16 s, MAX_BURST_S, the most is 55
# n4-ack-2f, which the shifter never plans past 12 s), so it takes 5 | 5.
CW_BITS = {"n10": 5}


def max_codewords(sync_band: str) -> int:
    """Codewords a burst's header can announce on this sync band."""
    return 1 << CW_BITS.get(sync_band, 6)

LEADIN_SAMPLES = 800  # 100 ms of silence before the preamble
LEADOUT_SAMPLES = 800

# --- TX conditioning (SSTVAE values; plan step 8 revisits per submode) ------
CLIP_HEADROOM_DB = 1.0
CLIP_OVERSHOOT = (1.0, 1.5, 2.0)
TX_BANDPASS = (850.0, 2200.0)
# What the clipper does to data symbols, as seen by a receiver whose
# channel estimate comes from the pilots, per band and headroom:
# codes_data/clip_constants.json, measured through channel_torch on a
# clean channel (scripts/clip_constants.py --write regenerates it; do it
# whenever the clipper changes). Identical for QAM 4..256: after the IDFT
# every constellation looks Gaussian.
#
# Gain: data arrive at 0.72-0.99 of the pilot-referenced channel, because
# the data clip hard and the ~1 dB-PAPR pilots barely at all (Bussgang
# gain; ignoring it cost 16-QAM 0.1 BMI at 20 dB). A one-frame burst is
# mostly preamble, header and pilots, which shifts the clip threshold,
# hence a separate 1-frame gain. SDR: distortion power on each carrier
# relative to the wanted data power after that gain, flat from 1 to 60
# frames; made at TX, so it fades with its carrier (|h|^2 at RX).
# At the stock 1 dB: w 13.1 dB, n10 13.5, n4 15.1 (fewer carriers, less
# Gaussian, less clipping), w48 12.3.
DEMOD_BACKOFF = 6

SNR_REF_BW_HZ = 2500.0

# Minimized-crest-factor pilot, 0.99 dB envelope PAPR. SSTVAE's config.py
# records why this and not Zadoff-Chu.
PILOT_PHASE_DEN = 1024
PILOT_PHASE_NUM = (
    725, 497, 359, 322, 193, 849,
    710, 345, 960, 628, 347, 570,
    551, 678, 448, 713, 839, 90,
    236, 545, 1020, 403, 985, 304,
)


# --- bands ------------------------------------------------------------------
@dataclass(frozen=True)
class BandSpec:
    """A contiguous run of carriers on the 50 Hz grid (carrier k at
    CARRIER0 + 50 k Hz; k may run past NC for narrow bands placed
    elsewhere), with everything that depends on it. The preamble
    identifies the band; the header's submode index is per band."""

    name: str
    k0: int  # first carrier on the grid
    nc: int  # carriers
    pilot_num: tuple  # pilot phases, NUM / PILOT_PHASE_DEN turns
    header_syms: int  # header code length = 2 * nc * header_syms bits
    preamble_repeats: int = PREAMBLE_REPEATS
    # TX clipper, per band (plan step 8 tunes these); CLIP holds what the
    # RX needs to know about the result (scripts/clip_constants.py).
    clip_headroom_db: float = CLIP_HEADROOM_DB
    clip_overshoot: tuple = CLIP_OVERSHOOT
    # Another band whose preamble, header and header index space this
    # band's bursts use; only their frames are on this band's carriers.
    # "": its own.
    sync: str = ""

    @property
    def sync_band(self) -> str:
        return self.sync or self.name

    @property
    def freqs(self):
        import numpy as np

        return CARRIER0 + RS * (self.k0 + np.arange(self.nc))

    @property
    def tx_bandpass(self) -> tuple[float, float]:
        """Post-clip filter: the carriers plus 75 Hz each side, except the
        wide band's, which is SSTVAE's."""
        if self.name == "w":
            return TX_BANDPASS
        f = self.freqs
        return float(f[0] - 75), float(f[-1] + 75)

    @property
    def cu_per_frame(self) -> int:
        return self.nc * DATA_SYMS_PER_FRAME

    @property
    def preamble_samples(self) -> int:
        return PREAMBLE_CP + self.preamble_repeats * M

    @property
    def preamble_threshold(self) -> float:
        """On sync.detection_stat."""
        return PREAMBLE_THRESHOLDS[self.preamble_repeats]


# --- submodes ---------------------------------------------------------------
@dataclass(frozen=True)
class SubmodeSpec:
    index: int  # sent in the header: 0..15 (0..31 on n10, CW_BITS)
    name: str
    code: str  # "ldpc" | "polar"
    constellation: str  # name for data2g.constellation.load
    frames_per_cw: int
    k: int = 0  # info bits per codeword incl. CRC
    band: str = "w"  # key into BANDS
    clip_headroom_db: float | None = None  # None: the band's
    # active constellation extension in the TX clipper: the overshoot of
    # its closing passes; () none (scripts/pick_headroom.py)
    ace: tuple = ()

    @property
    def bits_per_cu(self) -> int:
        from .constellation import bits_per_symbol, load

        return bits_per_symbol(load(self.constellation))

    @property
    def cu_per_frame(self) -> int:
        return BANDS[self.band].cu_per_frame

    @property
    def sync_band(self) -> str:
        """Band of this submode's preamble and header (BandSpec.sync)."""
        return BANDS[self.band].sync_band

    @property
    def headroom(self) -> float:
        """Clip headroom this submode transmits with: its own if set
        (scripts/pick_headroom.py), else its band's."""
        return BANDS[self.band].clip_headroom_db if self.clip_headroom_db is None else self.clip_headroom_db

    @property
    def coded_bits(self) -> int:
        return self.cu_per_frame * self.frames_per_cw * self.bits_per_cu


# Narrow-band pilots from scripts/design_pilot.py (PAPR by its metric,
# which gives SSTVAE's wide pilot 0.98 dB).
PILOT_N4 = (518, 821, 829, 542)  # 1.83 dB
PILOT_N10 = (480, 471, 862, 780, 503, 490, 593, 127, 601, 62)  # 1.25 dB
PILOT_W48 = (  # 0.98 dB
    1021, 910, 952, 927, 968, 883, 915, 730, 618, 666, 431, 53, 996, 1022, 815, 278,
    77, 519, 263, 311, 918, 546, 399, 690, 927, 609, 100, 34, 390, 663, 873, 422,
    1004, 904, 229, 366, 529, 880, 1012, 422, 611, 98, 709, 78, 423, 772, 293, 849,
)

BANDS = {
    # SSTVAE's 24 carriers, 950..2100 Hz.
    "w": BandSpec("w", 0, NC, PILOT_PHASE_NUM, 4),
    # 500 Hz and 200 Hz, both centred on the wide band's 1525 Hz. Header
    # lengths keep the code near 192 bits.
    "n10": BandSpec("n10", 7, 10, PILOT_N10, 10),
    # n4 syncs on n10 (preamble, header, index space; 2026-09-23): 200 Hz
    # fades as a whole, and its own sync floor on mpp was 8.75 dB against
    # n10's 0.25 (runs/sync_floor.csv). Its own preamble and 24-symbol
    # header are no longer sent; header_syms stays for the record.
    "n4": BandSpec("n4", 10, 4, PILOT_N4, 24, sync="n10"),
    # 2400 Hz, 350..2700 Hz, centred on 1525 Hz like the others; data
    # submodes only (the ACKs stay on the narrower bands). Its header is 2
    # symbols (192 bits, the wide band's code).
    # 4 header symbols (was 2: the same 192 bits in half w's airtime, half its
    # energy; its headers failed 10-23% at -5..-6 dB AWGN where w's never did,
    # the whole of its 3 dB worse sync floor; runs/w48_sync_diag.log)
    "w48": BandSpec("w48", -12, 48, PILOT_W48, 4),
}


def _load_clip_table():
    import json
    from pathlib import Path

    d = json.loads((Path(__file__).parent / "codes_data" / "clip_constants.json").read_text())
    return tuple(d["overshoot"]), d["entries"]


CLIP_OVERSHOOT_TABLE, _CLIP_ENTRIES = _load_clip_table()


def clip_key(band: str, headroom: float, ace: tuple = (), const: str = "") -> str:
    """codes_data/clip_constants.json's key. Plain clipping looks the same to
    every constellation; ACE depends on the constellation's regions."""
    k = f"{band}@{headroom:g}"
    return k + f"+ace{'-'.join(f'{v:g}' for v in ace)}:{const}" if ace else k


def clip_consts(band: str, headroom: float, ace: tuple = (), const: str = "") -> tuple:
    """(gain by frame count, default gain, clip-noise ratio) for a band at a
    headroom, from codes_data/clip_constants.json (stock overshoot)."""
    e = _CLIP_ENTRIES[clip_key(band, headroom, ace, const)]
    return {1: e["gain_1f"]}, e["gain"], 10 ** (-e["sdr_db"] / 10)


def clip_peak_db(band: str, headroom: float, ace: tuple = (), const: str = "") -> float:
    """Post-clip envelope peak-to-average of a burst, for PEP-fair scores."""
    return _CLIP_ENTRIES[clip_key(band, headroom, ace, const)]["peak_db"]


# Per band at its default headroom (what a submode without its own gets).
CLIP = {b: clip_consts(b, BANDS[b].clip_headroom_db) for b in BANDS}
assert CLIP_OVERSHOOT_TABLE == CLIP_OVERSHOOT, "clip table measured with another overshoot"


# The ladder, provisional (2026-09-23): the survivors of scripts/prune.py
# over runs/ladder.csv, minus the cuts agreed then (0.083 and 0.183
# polar: below the sync floor / within 1 dB of 0.167; learned-256: 8 dB
# worse on AWGN than learned-64 at the same rate). Indices are not frozen;
# plan step 9 freezes them with their constellations and interleavers.
# Payload bits/cu (CRC excluded) in the comments.
def _m(i, name, code, const, frames, k, band="w", headroom=None, ace=()):
    return name, SubmodeSpec(i, name, code, const, frames, k=k, band=band, clip_headroom_db=headroom, ace=ace)


# The pruned ladder (2026-09-23): scripts/prune.py runs/ladder_final.csv
# --sync runs/sync_floor.csv --keep polar-gray-qam4-f1-k48@h0
# n4-polar-gray-qam4-f2-k48@h0; thresholds
# in CANDIDATES.md. Indices by payload rate within each header index
# space (w; n10 with n4, which syncs on it; w48). Clip headroom per
# submode (scripts/pick_headroom.py). Provisional: the narrow ladder may
# be trimmed once there is on-air experience.
SUBMODES = dict([
    # 1200 Hz. ack-1f is kept for latency (432 ms), not by domination.
    _m(0, "ack-4f", "polar", "gray-qam4", 4, 56, headroom=0),  # 56 bps
    _m(1, "polar-k96-f8", "polar", "gray-qam4", 8, 104, headroom=0),  # 69 bps
    _m(2, "polar-k96-f4", "polar", "gray-qam4", 4, 104, headroom=0),  # 139 bps
    _m(3, "polar-k192-f8", "polar", "gray-qam4", 8, 200, headroom=0),  # 153 bps
    _m(4, "ack-1f", "polar", "gray-qam4", 1, 56, headroom=0),  # 222 bps
    _m(5, "qpsk-r1/5", "ldpc", "gray-qam4", 8, 384, headroom=0),  # 319 bps
    _m(6, "qpsk-r1/3", "ldpc", "gray-qam4", 8, 640, headroom=0),  # 528 bps
    _m(7, "qpsk-r1/2", "ldpc", "gray-qam4", 8, 960, headroom=0),  # 806 bps
    _m(8, "16qam-r1/3", "ldpc", "gray-qam16", 4, 640, headroom=0),  # 1056 bps
    _m(9, "qpsk-r3/4", "ldpc", "gray-qam4", 8, 1440, headroom=0),  # 1222 bps
    _m(10, "16qam-r1/2", "ldpc", "gray-qam16", 8, 1920, headroom=0),  # 1639 bps
    # <=500 Hz: n10 and n4 (n4 on n10's preamble and header)
    # n4-ack-2f is kept for latency (768 ms; the n4 LDPC modes are 3.9 s)
    _m(0, "n4-ack-8f", "polar", "gray-qam4", 8, 56, band="n4", headroom=0),  # 28 bps
    _m(1, "n4-qpsk-r1/5", "ldpc", "gray-qam4", 24, 192, band="n4", headroom=0),  # 51 bps
    _m(2, "n10-ack-4f", "polar", "gray-qam4", 4, 56, band="n10", headroom=0),  # 56 bps
    _m(3, "n4-qpsk-r1/3", "ldpc", "gray-qam4", 24, 320, band="n4", headroom=0),  # 88 bps
    _m(4, "n4-ack-2f", "polar", "gray-qam4", 2, 56, band="n4", headroom=0),  # 111 bps
    _m(5, "n10-qpsk-r1/5", "ldpc", "gray-qam4", 10, 200, band="n10", headroom=0),  # 128 bps
    _m(6, "n4-qpsk-r1/2", "ldpc", "gray-qam4", 24, 480, band="n4", headroom=0),  # 134 bps
    _m(7, "n4-qpsk-r2/3", "ldpc", "gray-qam4", 24, 640, band="n4", headroom=0),  # 176 bps
    _m(8, "n4-16qam-r1/3", "ldpc", "gray-qam16", 24, 640, band="n4", headroom=0),  # 176 bps
    _m(9, "n10-qpsk-r1/3", "ldpc", "gray-qam4", 10, 336, band="n10", headroom=0),  # 222 bps
    _m(10, "n10-qpsk-r1/2", "ldpc", "gray-qam4", 10, 496, band="n10", headroom=0),  # 333 bps
    _m(11, "n10-16qam-r1/3", "ldpc", "gray-qam16", 10, 664, band="n10", headroom=0),  # 439 bps
    _m(12, "n10-qpsk-r3/4", "ldpc", "gray-qam4", 10, 752, band="n10", headroom=0),  # 500 bps
    _m(13, "n10-16qam-r1/2", "ldpc", "gray-qam16", 10, 1000, band="n10", headroom=0),  # 672 bps
    _m(14, "n10-16qam-r2/3", "ldpc", "gray-qam16", 10, 1336, band="n10", headroom=2),  # 906 bps
    _m(15, "n10-16qam-r3/4", "ldpc", "gray-qam16", 10, 1504, band="n10", headroom=3),  # 1022 bps
    # past n10's first 16 indices (CW_BITS), toward VARA 500's ~10 kB/min at
    # 25 dB: learned sets, headroom by pick_headroom (runs/clip_n10_top.csv)
    _m(16, "n10-64l-r3/4", "ldpc", "c64-w48-r34", 10, 2256, band="n10", headroom=6),  # 1544 bps
    _m(17, "n10-256l-r3/4", "ldpc", "c256-w48-r58", 10, 3008, band="n10", headroom=8),  # 2067 bps
    # 2400 Hz, data only
    _m(0, "w48-qpsk-r1/5", "ldpc", "gray-qam4", 4, 384, band="w48", headroom=0),  # 639 bps
    _m(1, "w48-qpsk-r1/3", "ldpc", "gray-qam4", 4, 640, band="w48", headroom=0),  # 1056 bps
    _m(2, "w48-qpsk-r1/2", "ldpc", "gray-qam4", 4, 960, band="w48", headroom=0),  # 1611 bps
    _m(3, "w48-qpsk-r2/3", "ldpc", "gray-qam4", 4, 1280, band="w48", headroom=0),  # 2167 bps
    _m(4, "w48-16qam-r1/3", "ldpc", "gray-qam16", 4, 1280, band="w48", headroom=1),  # 2167 bps
    _m(5, "w48-qpsk-r3/4", "ldpc", "gray-qam4", 4, 1440, band="w48", headroom=0),  # 2444 bps
    _m(6, "w48-16qam-r1/2", "ldpc", "gray-qam16", 4, 1920, band="w48", headroom=1, ace=(1.0,)),  # 3278 bps
    _m(7, "w48-16qam-r2/3", "ldpc", "gray-qam16", 4, 2560, band="w48", headroom=1, ace=(1.0,)),  # 4389 bps
    _m(8, "w48-64l-r1/2", "ldpc", "c64-w48-r12", 2, 1440, band="w48", headroom=4),  # 4889 bps
    _m(9, "w48-16qam-r3/4", "ldpc", "gray-qam16", 4, 2880, band="w48", headroom=3, ace=(1.0,)),  # 4944 bps
    _m(10, "w48-16qam-r5/6", "ldpc", "gray-qam16", 4, 3200, band="w48", headroom=5),  # 5500 bps
    _m(11, "w48-64l-r7/12", "ldpc", "c64-w48-r712", 2, 1680, band="w48", headroom=5),  # 5722 bps
    _m(12, "w48-64l-r2/3", "ldpc", "c64-w48-r23", 2, 1920, band="w48", headroom=6),  # 6556 bps
    _m(13, "w48-64l-r3/4", "ldpc", "c64-w48-r34", 2, 2160, band="w48", headroom=5),  # 7389 bps
    _m(14, "w48-256l-r5/8", "ldpc", "c256-w48-r58", 2, 2400, band="w48", headroom=6),  # 8222 bps
])
