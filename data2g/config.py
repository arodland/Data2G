"""Waveform, burst and submode constants.

The waveform half is SSTVAE's (`sstvae/config.py`, PROTOCOL_VERSION 4),
copied rather than imported so this project owns its on-air format; the
measurements behind each number are recorded there. Two deliberate
differences: the beacon carrier is reclaimed for data (all 24 carriers
carry payload), and the preamble repeat count is ours to raise.

The single source of on-air constants. Every module-level UPPER_CASE
name here becomes a constexpr in native/core/generated/config.hpp (BANDS
and SUBMODES as structs with their derived fields; the generator's SKIP
lists the rest it emits in another form or not at all), so after a
change run tools/gen_native_tables.py, whose --check fails until then.

Native lookups on the generated tables (BANDS[name], SUBMODES[name],
BY_INDEX) are native/core/config/config.hpp.
"""

import os
from dataclasses import dataclass
from typing import NamedTuple

# --- profile ------------------------------------------------------------------
# Which radio service this process speaks: "hf" (SSB, the default) or "fm"
# (VHF/UHF FM; draft, nothing on it is frozen yet). Fixed at process start:
# every constant below may depend on it, and the native core is built once
# per profile (cmake -DDATA2G_PROFILE=fm; tools/gen_native_tables.py writes
# each profile's tables). HF and FM stations do not interoperate, so a
# mode index or a protocol version means different things on each.
PROFILE = os.environ.get("DATA2G_PROFILE", "hf")
PROFILES = ("hf", "fm")
if PROFILE not in PROFILES:
    raise ValueError(f"DATA2G_PROFILE={PROFILE!r}: one of {', '.join(PROFILES)}")
# Frozen data measured under this profile, where it differs from HF's
# (codes_data/<profile>/, else codes_data/).
DATA_DIR = "" if PROFILE == "hf" else PROFILE


def data_path(name: str):
    """codes_data/<profile>/name if this profile has its own, else codes_data/name."""
    from pathlib import Path

    base = Path(__file__).parent / "codes_data"
    own = base / DATA_DIR / name
    return own if DATA_DIR and own.exists() else base / name


# FM: 16 kHz, to carry a wide band to about 6 kHz over a 9600 port; carriers
# stay on HF's 50 Hz grid (M doubles), so narrow and wide are bands of one
# grid, as HF's 500 and 2300 Hz are. Provisional (FM plan, M4 and M5).
FS = {"hf": 8000, "fm": 16000}[PROFILE]  # audio sample rate, Hz

# --- OFDM waveform (identical to SSTVAE) ------------------------------------
# DATA2G_RS, DATA2G_NCP and DATA2G_CARRIER0 are for studies only (the FM
# plan's carrier spacing, Q19: scripts/fm_spacing_study.py); like
# DATA2G_PREAMBLE_REPEATS they need the matching receiver on the far end, and
# tools/gen_native_tables.py refuses them.
RS = int(os.environ.get("DATA2G_RS", 50))  # carrier spacing == symbol rate of one carrier, Hz
NC = 24  # carriers, all of them data
M = FS // RS  # useful symbol length, samples (160)
NCP = int(os.environ.get("DATA2G_NCP", 32))  # cyclic prefix, samples (4 ms)
NSYM = M + NCP  # 192 samples, 24 ms
CARRIER0 = int(os.environ.get("DATA2G_CARRIER0", 950))  # Hz; occupied ~925..2125 Hz
FCENTER = int(os.environ.get("DATA2G_FCENTER", 1500))  # Hz; baseband conversion frequency (env: studies only)
# The mixer's phase must repeat every symbol (whole cycles per NSYM), or each
# symbol reaches baseband with its own common phase, which the pilots cannot
# follow: a shorter cyclic prefix on FM (scripts/fm_cp_study.py) needs another FCENTER.
assert FCENTER * NSYM % FS == 0, "FCENTER * NSYM must be a multiple of FS"
# The demodulator's M-sample correlation nulls the heterodyne image at
# -(f + FCENTER) only if that lands on the grid (waveform.dsp.to_baseband).
assert (CARRIER0 + FCENTER) % RS == 0, "CARRIER0 + FCENTER must be a multiple of RS"

SYMS_PER_FRAME = int(os.environ.get("DATA2G_SYMS_PER_FRAME", 6))  # 1 pilot + 5 data (env: studies only, scripts/fm_pilot_study.py)
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
# FM (plan M3, scripts/fm_rx_study.py, 2026-11-15): 1.1 x the largest 1200 s
# noise peak over what an idle FM receiver hands the modem, at FM's +-12.5 Hz
# search, blanked: mic-path hiss with no carrier (open squelch) 26.1, idle
# carrier at 8 dB CNR 22.5 (mic) / 24.5 (flat), white 21.2. HF's 24.1 at
# +-150 Hz would have passed FM's mic hiss (33.8 there) every ~20 s. Loss is
# the same at any of these thresholds (FM's cliff: the preamble is found far
# below where data fails), so the threshold buys idle CPU, not range.
if PROFILE == "fm":
    PREAMBLE_THRESHOLDS = {8: 28.7}
# FM's f55 and f108 under it too (scripts/mf_detect_study.py noise, 600 s per source,
# blanked, 2026-11-17): largest peaks f55 17.7 white, 23.6 mic hiss, 21.5 / 24.1 idle
# carrier at 8 dB (mic / flat); f108 21.8, 25.8, 20.6, 21.3. 1.1 x 25.8 = 28.4.
# Frequency-offset search, +-Hz. SSTVAE searched +-625 Hz because it cost
# nothing there; here it cost ~5x the detector's CPU against +-150 Hz, and
# noise-only peaks ran 5-10% higher (2026-09-24). A mode up to 2400 Hz
# wide can't use the room anyway: +-150 Hz already pushes w48's edge
# carriers into the SSB filter's skirts.
# FM: +-12.5 Hz. FM audio has no frequency offset (an RF one comes out of
# the discriminator as DC); the sound cards' clocks move 1500 Hz by under
# 0.3 Hz. A physical bound, not a narrowing by expected modes: every band
# is still searched. Measured (scripts/fm_rx_study.py): the same loss as
# +-150 Hz within 0.05 dB everywhere, the detector's CPU 6.7x less.
# DATA2G_ACQUIRE_REACH_HZ: studies only, refused by the table generator.
ACQUIRE_REACH_HZ = float(os.environ.get("DATA2G_ACQUIRE_REACH_HZ", {"hf": 150.0, "fm": 12.5}[PROFILE]))
# Per-symbol noise in the demapper (modem.symbol_noise_scale): an FM click,
# or the gap the Blanker leaves of one, hits a whole OFDM symbol. Measured on
# FM (scripts/fm_rx_study.py, 30 paired trials): mobile paths 0.9 (QPSK),
# 1.8 (16-QAM), 2.25 dB (64-point) better, flutter 0.1-0.5, slow fading
# 0.2-0.35, never worse. Not tried on HF yet. DATA2G_SYMBOL_NOISE (0/1):
# studies only.
SYMBOL_NOISE = bool(int(os.environ.get("DATA2G_SYMBOL_NOISE", int(PROFILE == "fm"))))
# Interleaving window, frames (codes.spread_group): 0, every codeword over the whole
# burst. A window lets a codeword group complete, and decode, while the burst
# arrives (FM plan: decoding while the burst arrives).
# Channel estimate in windows of this many frames (modem.data_channel), each from
# its own pilots and one either side: what a receiver estimating as frames
# arrive sees. 0: the whole burst.
# FM: 8 and 8. scripts/fm_window_study.py (4 s bursts, 30 paired trials, every
# fmchannel condition): 8-frame windows, spread and estimate, at most +0.12 dB
# anywhere, better on mobile paths (0.3-0.4 dB) and slow fading (0.45-1.24 dB),
# most likely from the per-window noise level; 4 frames cost up to +0.35 dB.
# HF: 0 and 0 (issue #212, docs/hf-window-study.md): spread windows of 2.3-6.9 s cost 0.3-2.3 dB on
# fading (n4 up to 4.5), and a Pi 4 gains at most ~50 ms. DATA2G_SPREAD_WINDOW, DATA2G_EST_WINDOW: studies only.
SPREAD_WINDOW = int(os.environ.get("DATA2G_SPREAD_WINDOW", {"hf": 0, "fm": 8}[PROFILE]))
EST_WINDOW = int(os.environ.get("DATA2G_EST_WINDOW", {"hf": 0, "fm": 8}[PROFILE]))
# Decoding while the burst arrives (receiver.Receiver "partial" events, ModemRx.advance):
# FM on (its windows let codeword groups complete early); HF has no windows. DATA2G_DECODE_EARLY: studies.
DECODE_EARLY = bool(int(os.environ.get("DATA2G_DECODE_EARLY", {"hf": 0, "fm": 1}[PROFILE])))
# FM transmit, per station's radio path (modem.TX_PATH, "mic" or "flat"); HF ignores
# these. Flat (a 9600 port: no pre-emphasis, no limiter, f^2 discriminator noise):
# a fixed +3 dB/octave tilt on the whole burst (1.4-2.9 dB on f108, 0.9-1.6 dB on f55,
# scripts/fm_tilt_study.py) and no TX clipping from 16-QAM up (clipping buys no
# deviation on FM, the width being set by rms; 1.2-1.4 dB on 64/256-point,
# scripts/fm_clip_study.py). Mic (pre-emphasis and the radio's 5 kHz limiter): the
# top modes sent quieter so the limiter rarely acts (under 1%): gain per bits per
# cell, 64-point 0.70, 256-point 0.55 (fm_clip_study; more than doubles their BMI).
FM_TX_TILT_DB_OCT = 3.0
FM_FLAT_NOCLIP_BITS = 4
FM_MIC_GAIN = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.70, 1.0, 0.55)  # by bits per cell
# FM: every burst at one scale, the modem's unit-RMS audio over this peak (rms about -14 dBFS),
# the rare crest above it clipped. HF's per-burst envelope-peak normalisation (an SSB
# transmitter is PEP limited) would undo FM_MIC_GAIN and send unclipped flat-path bursts
# (peaks near 5) about 6 dB under QPSK's (peaks near 2.5); FM deviation, and so the 15 kHz
# width and every FM study, go by rms. 5.0: the largest crest of the unclipped flat modes.
FM_TX_PEAK = 5.0
# The Doppler spread the equalizer assumes for a burst too short to measure one
# (equalizer.measure_spread: under 8 pilots). HF: 2.0, what MPD measures. FM:
# the audio channel is the radios' fixed response, RF Doppler reaching it as
# clicks, not fading of the tones; 0.2 Hz measured 0.06-0.16 dB better on
# flutter and mobile paths for QPSK and 16-QAM, never worse (scripts/fm_pilot_study.py
# spf6p, 2026-11-16). arq.policy reads a spread equal to it as unmeasured.
DEFAULT_SPREAD_HZ = float(os.environ.get("DATA2G_DEFAULT_SPREAD_HZ", {"hf": 2.0, "fm": 0.2}[PROFILE]))
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
# The CRC is seeded with PROTOCOL_VERSION XOR PROFILE_TAG, so a receiver of
# another version, another profile (or SSTVAE) accepts a header only by
# 1-in-64 chance. FM needs the tag: its preamble on the same 50 Hz grid is
# the same analog waveform as HF's. Distinct seeds while HF's version < 32.
# Header symbols per band: BandSpec.header_syms.
PROFILE_TAG = {"hf": 0, "fm": 0x20}[PROFILE]
# FM counts its own versions from 1 (no FM format is frozen yet); 2: codewords
# interleaved in 8-frame windows (SPREAD_WINDOW); 3: FM's own bands f55 and f108 (new
# preambles, headers and submode indices; the FM plan's M4/M5 bands); 4: f55's and
# f108's measured ladders (14 modes each, runs/fm_ladder/final_proposal.md); 5: no n4 (its
# header indices on n10's sync band unused) and FM's ARQ band codes w, n10, f55, f108 (no w48,
# no CPM). FM's ARQ session VERSION is HF's: PROFILE_TAG and this keep the profiles apart.
PROTOCOL_VERSION = {"hf": 15, "fm": 5}[PROFILE]  # HF: 10: first frozen submode table; n4 on n10 sync (2026-09-23); 11: header copy;
# 12: polar codewords carry CRC-24 (k + 8, payloads kept); n10's header 5 | 5 (CW_BITS, 2026-10-06,
# in place: development, one operator); 13: a non-periodic c8r50 front (2026-10-24: the tiled
# Costas array matched one period off, sync 0.76, and its header read as another word); 14: the CPM
# modes' data codewords GF(M) LDPC on tone likelihoods (data2g.gfldpc, 2026-10-27: 0.7-1.5 dB on
# ideal channels over the binary LDPC on bit LLRs), same payloads and burst framing; 15: 20 ms lead-in
# and lead-out silence, not 100 ms (2026-10-27, scripts/burst_trim.py): every OFDM burst 160 ms shorter
# A second header copy, time-diverse, on the 4-symbol headers (w, w48): a
# frame of its own (pilot, the 4 header symbols, the first again) after
# data frame HEADER_COPY_AFTER, or after the last on a shorter burst. The
# receiver combines the copies' LLRs. scripts/header_diversity.py (2000
# bursts a cell): w48 MPP -1 dB header loss 8.5% -> 2.4%, MPD -1 14.5% ->
# 3.8%; w wrong headers accepted at MPD -3 dB 10.2% -> 4.0%. Spaced beat
# contiguous (an 8-symbol header: w48 MPP -1 4.8%); 1-4 frames later alike.
HEADER_COPY_BANDS = ("w", "w48") + (("f55", "f108") if PROFILE == "fm" else ())  # FM's bands: 4-symbol headers too
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

# Silence around a burst: room for the TX bandpass's tails (three passes of
# a 201-tap FIR, under -67 dB of the burst's mean power past 160 samples;
# -34 dB past 80). 100 ms each until protocol 15, SSTVAE's file padding;
# the radio's key-up time is the engine's PTT delay (--ptt-on-delay-ms)
# and its key-down the PTT off delay, so neither belongs in every burst.
LEADIN_SAMPLES = FS // 50  # 20 ms
LEADOUT_SAMPLES = FS // 50

# --- TX conditioning (SSTVAE values; plan step 8 revisits per submode) ------
CLIP_HEADROOM_DB = 1.0
CLIP_OVERSHOOT = (1.0, 1.5, 2.0)
# SSTVAE's wide-band filter, two carrier spacings past the band each side:
# (850, 2200) on HF; it moves with the grid (DATA2G_RS, DATA2G_CARRIER0).
TX_BANDPASS = (float(CARRIER0 - 2 * RS), float(CARRIER0 + RS * (NC + 1)))
# The post-clip band-pass FIR (waveform.dsp.tx_condition): 25 ms of taps, 201 at
# 8 kHz. In taps it would widen with FS (twice the transition at 16 kHz cost the
# edge carriers ~3 dB), so it scales; LEADIN_SAMPLES, in time, covers its tails.
TX_FILTER_TAPS = FS // 40 + 1
# (what the clipper does to the data, as a receiver sees it: CLIP, below BANDS)

# --- receiver -----------------------------------------------------------------
# Samples the data symbols' FFT window starts before the cyclic prefix's
# end (SSTVAE's value; the header, read before the delay profile is known,
# backs off further: modem.HEADER_BACKOFF).
DEMOD_BACKOFF = 6
# Noise bandwidth every reported SNR is referred to (Hz).
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
    def clip(self) -> "ClipConsts":
        """The clip constants of this submode's bursts (native modem::clip_consts(spec))."""
        return clip_consts(self.band, self.headroom, self.ace, self.constellation)

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
# FM's narrow and wide bands (FM plan M4, M5; FM profile only): 450..3150 Hz
# and 300..5650 Hz on the same 50 Hz grid. Band edges are the plan's, from
# VARA FM and VardeFM, until real radios are measured (M0, Q7). 4 header
# symbols each, no longer in time than w48's (the code is longer, its rate
# lower). Pilots: scripts/design_pilot.py.
if PROFILE == "fm":
    PILOT_F55 = (520, 990, 205, 328, 843, 26, 497, 464, 602, 400, 937, 155, 864, 270, 513, 550, 246, 421, 718, 656, 973, 727, 144, 460, 694, 547, 717, 342, 845, 108, 670, 823, 696, 928, 898, 290, 0, 661, 363, 73, 569, 917, 541, 544, 530, 557, 789, 535, 422, 332, 121, 875, 418, 162, 17)  # 0.98 dB
    PILOT_F108 = (162, 22, 839, 645, 743, 304, 220, 961, 911, 679, 404, 272, 796, 1019, 730, 913, 783, 992, 1, 16, 210, 1002, 204, 399, 392, 192, 528, 55, 474, 746, 859, 976, 178, 325, 664, 161, 871, 422, 895, 404, 151, 644, 35, 403, 324, 391, 686, 209, 552, 81, 756, 989, 405, 425, 555, 647, 562, 839, 55, 221, 614, 282, 88, 320, 803, 882, 1, 1023, 606, 488, 309, 697, 76, 193, 558, 68, 373, 663, 499, 185, 159, 666, 232, 918, 717, 718, 263, 730, 564, 426, 820, 207, 438, 698, 318, 964, 619, 309, 172, 551, 366, 232, 273, 44, 103, 301, 999, 1021)  # 1.45 dB: best of ~500 starts (1.45-1.51); 108 carriers sit higher than 55's 0.98
    BANDS["f55"] = BandSpec("f55", -10, 55, PILOT_F55, 4)
    BANDS["f108"] = BandSpec("f108", -13, 108, PILOT_F108, 4)


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
def _load_clip_table():
    import json

    d = json.loads(data_path("clip_constants.json").read_text())
    return tuple(d["overshoot"]), d["entries"]


CLIP_OVERSHOOT_TABLE, _CLIP_ENTRIES = _load_clip_table()


def clip_key(band: str, headroom: float, ace: tuple = (), const: str = "") -> str:
    """codes_data/clip_constants.json's key. Plain clipping looks the same to
    every constellation; ACE depends on the constellation's regions."""
    k = f"{band}@{headroom:g}"
    return k + f"+ace{'-'.join(f'{v:g}' for v in ace)}:{const}" if ace else k


class ClipConsts(NamedTuple):
    """What the clipper does to the data, as a receiver sees it (native
    modem::ClipConsts)."""

    gains: dict  # data over pilot amplitude, by burst length in frames where it differs from `gain`
    gain: float  # the same, any other length
    ratio: float  # clip-noise power over the data's

    def gain_for(self, n_frames: int) -> float:
        return self.gains.get(n_frames, self.gain)


def clip_consts(band: str, headroom: float, ace: tuple = (), const: str = "") -> ClipConsts:
    """A band's at a headroom (and ACE and constellation, if it uses ACE),
    from codes_data/clip_constants.json (stock overshoot)."""
    e = _CLIP_ENTRIES[clip_key(band, headroom, ace, const)]
    return ClipConsts({1: e["gain_1f"]}, e["gain"], 10 ** (-e["sdr_db"] / 10))


def clip_peak_db(band: str, headroom: float, ace: tuple = (), const: str = "") -> float:
    """Post-clip envelope peak-to-average of a burst, for PEP-fair scores."""
    return _CLIP_ENTRIES[clip_key(band, headroom, ace, const)]["peak_db"]


# Per band at its default headroom (what a submode without its own gets).
CLIP = {b: clip_consts(b, BANDS[b].clip_headroom_db) for b in BANDS}
assert CLIP_OVERSHOOT_TABLE == CLIP_OVERSHOOT, "clip table measured with another overshoot"


def _m(i, name, code, const, frames, k, band="w", headroom=None, ace=()):
    return name, SubmodeSpec(i, name, code, const, frames, k=k, band=band, clip_headroom_db=headroom, ace=ace)


# The pruned ladder (2026-09-23): scripts/prune.py runs/ladder_final.csv
# --sync runs/sync_floor.csv --keep polar-gray-qam4-f1-k48@h0
# n4-polar-gray-qam4-f2-k48@h0, minus the cuts agreed then (0.083 and
# 0.183 polar: below the sync floor / within 1 dB of 0.167; learned-256:
# 8 dB worse on AWGN than learned-64 at the same rate); thresholds in
# CANDIDATES.md. Indices by payload rate within each header index space
# (w; n10 with n4, which syncs on it; w48). Clip headroom per submode
# (scripts/pick_headroom.py). Provisional: the narrow ladder may be
# trimmed once there is on-air experience. Payload rate (CRC excluded)
# in the comments.
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
# FM's narrow (f55) and wide (f108) ladders (FM plan M4/M5), measured on FM
# channels: runs/fm_ladder/final_proposal.md (scripts/fm_ladder_study.py, 30
# paired trials, streamed receiver; 10% knee CNR, static: f55 mic / flat,
# f108 flat). One f55 ladder serves mic and flat radios; the transmit differs
# by station (modem.TX_PATH). Constellations: the existing learned sets, one
# per family (runs/fm_const/report.txt). Indices 14 and 15 free. Approved
# 2026-11-17, subject to on-air testing. Not in the ARQ's modes yet (M6).
if PROFILE == "fm":
    SUBMODES.update(dict([
        _m(0, "f55-qpsk-r21", "ldpc", "gray-qam4", 4, 456, band="f55", headroom=2),  # 730 bps; 1.25 / 0.25 dB
        _m(1, "f55-qpsk-r40", "ldpc", "gray-qam4", 4, 888, band="f55", headroom=2),  # 1421; 2.75 / 2.0
        _m(2, "f55-qpsk-r60", "ldpc", "gray-qam4", 4, 1320, band="f55", headroom=2),  # 2138; 4.0 / 3.25
        _m(3, "f55-16qam-r47", "ldpc", "gray-qam16", 2, 1040, band="f55", headroom=3),  # 3379; 5.25 / 4.5
        _m(4, "f55-64l-r50", "ldpc", "c64-w48-r12", 2, 1648, band="f55", headroom=6),  # 5417; 7.0 / 6.25
        _m(5, "f55-64l-r67", "ldpc", "c64-w48-r12", 2, 2216, band="f55", headroom=6),  # 7321; 8.5 / 7.5
        _m(6, "f55-64l-r75", "ldpc", "c64-w48-r23", 2, 2472, band="f55", headroom=6),  # 8179; 9.75 / 8.25
        _m(7, "f55-64l-r80", "ldpc", "c64-w48-r23", 2, 2640, band="f55", headroom=6),  # 8743; 10.5 / 9.0
        _m(8, "f55-64l-r83", "ldpc", "c64-w48-r23", 2, 2752, band="f55", headroom=6),  # 9118; 11.25 / 9.25
        _m(9, "f55-256l-r67", "ldpc", "c256-w48-r12", 1, 1480, band="f55", headroom=6),  # 9708; 14.25 / 10.0
        _m(10, "f55-256l-r72", "ldpc", "c256-w48-r12", 1, 1584, band="f55", headroom=6),  # 10405; 15.75 / 11.0
        _m(11, "f55-256l-r80", "ldpc", "c256-w48-r58", 1, 1760, band="f55", headroom=6),  # 11585; 19.25 / 13.25
        _m(12, "f55-256l-r83", "ldpc", "c256-w48-r58", 1, 1832, band="f55", headroom=6),  # 12068; 21.25 / 14.0
        _m(13, "f55-256l-r88", "ldpc", "c256-w48-r58", 1, 1928, band="f55", headroom=6),  # 12712; 27.25 / 15.75
        _m(0, "f108-qpsk-r21", "ldpc", "gray-qam4", 2, 456, band="f108", headroom=2),  # 1475; 2.25
        _m(1, "f108-qpsk-r41", "ldpc", "gray-qam4", 2, 888, band="f108", headroom=2),  # 2870; 4.0
        _m(2, "f108-qpsk-r60", "ldpc", "gray-qam4", 2, 1296, band="f108", headroom=2),  # 4237; 5.5
        _m(3, "f108-16qam-r41", "ldpc", "gray-qam16", 1, 888, band="f108", headroom=6),  # 5739; 6.5
        _m(4, "f108-64l-r40", "ldpc", "c64-w48-r12", 1, 1296, band="f108", headroom=6),  # 8474; 8.5
        _m(5, "f108-64l-r50", "ldpc", "c64-w48-r12", 1, 1616, band="f108", headroom=6),  # 10620; 10.0
        _m(6, "f108-64l-r59", "ldpc", "c64-w48-r12", 1, 1920, band="f108", headroom=6),  # 12658; 11.5
        _m(7, "f108-64l-r67", "ldpc", "c64-w48-r23", 1, 2176, band="f108", headroom=6),  # 14374; 13.0
        _m(8, "f108-64l-r75", "ldpc", "c64-w48-r23", 1, 2432, band="f108", headroom=6),  # 16091; 14.75
        _m(9, "f108-64l-r80", "ldpc", "c64-w48-r23", 1, 2592, band="f108", headroom=6),  # 17163; 15.75
        _m(10, "f108-256l-r67", "ldpc", "c256-w48-r12", 1, 2896, band="f108", headroom=6),  # 19202; 17.5
        _m(11, "f108-256l-r75", "ldpc", "c256-w48-r58", 1, 3240, band="f108", headroom=6),  # 21508; 19.75
        _m(12, "f108-256l-r80", "ldpc", "c256-w48-r58", 1, 3456, band="f108", headroom=6),  # 22956; 21.25
        _m(13, "f108-256l-r87", "ldpc", "c256-w48-r58", 1, 3776, band="f108", headroom=6),  # 25102; 23.75
    ]))
    # FM has no n4 (plan Q2): its header indices on n10's sync band stay unused,
    # n10's keep their numbers and its 5 | 5 header split (no format re-freeze)
    SUBMODES = {n: s for n, s in SUBMODES.items() if s.band != "n4"}
# (sync band, header index) -> submode: what a header's index means
# (native config::by_index)
BY_INDEX = {(s.sync_band, s.index): s for s in SUBMODES.values()}
