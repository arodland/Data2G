"""FM sounding probe: known multitones that follow a ping's identity burst
in the same over (arq.ping), measured without decoding.

One PERIOD (20 ms, 50 Hz bins) of tones on odd bins with fixed random phases,
repeated in two parts:
- the sounding: every tone from TONE_LO_HZ to TONE_HI_HZ at SOUND_DB (low
  enough to stay linear on a mic radio, whose pre-emphasis boosts the top
  tones into its limiter), SOUND_MEAS periods: the path's passband, noise
  spectrum and band edges at 50 Hz resolution (NARROW or WIDE by measurement:
  nothing has to decode);
- the staircase: the tones inside the data band (STAIR_BAND), conditioned as
  data are (mic: the top modes' clip; flat: the tilt, then a clip that keeps
  the loudest step under the sound card's full scale, config.FM_TX_PEAK), at each level of STEPS_DB relative to the
  ping's level (ping.tx_level): SINAD per step finds the drive that suits the
  path (a limiter's knee, an under-driven sender, a ceiling). The last step
  repeats the reference: S/N moving between the two says the link varied.
Per tone and step, across the measured periods: signal = |mean|^2, noise = the
variance. Odd-order products of tones on odd bins land on odd bins, so a fixed
quarter of the odd bins carry no tone (HOLES): their power over the tones'
noise is the radio's distortion. The even bins (GAPS): noise and even-order
products.

On the flat path both parts carry the TX tilt (modem.tx_tilt_fir's curve),
applied per period so they stay periodic."""

import numpy as np

from .. import config, modem
from ..config import FS

PERIOD = FS // 50  # samples: 50 Hz bins
BIN_HZ = FS / PERIOD
TONE_LO_HZ, TONE_HI_HZ = 100.0, 7000.0
SETTLE, MEAS = 2, 10  # periods per step: a limiter's / AGC's settling, then measured
GUARD = 1  # of the SETTLE periods, measured at the stretch's end instead: a sample clock slow by up to
# a period over the probe still keeps the next (louder) step out (-500 ppm reached it without)
SOUND_DB, SOUND_MEAS = -12.0, 20
STAIR_BAND = "f55"
STEPS_DB = (4.0, 0.0, -6.0, -12.0, -18.0, 0.0)
REF = 1  # the step at the ping's level; the last repeats it
GAP_S = 0.05  # silence between the identity burst's end and the probe


def _hash(k, salt: int) -> np.ndarray:
    """Bin numbers -> [0, 1), well mixed (murmur3's 32-bit finaliser; a
    linear hash would line the tones' phases up into an impulse); native
    reproduces it exactly."""
    m = np.uint64(0xFFFFFFFF)
    h = ((np.asarray(k, dtype=np.uint64) + np.uint64(salt)) * np.uint64(0x9E3779B1)) & m
    h ^= h >> np.uint64(16)
    h = (h * np.uint64(0x85EBCA6B)) & m
    h ^= h >> np.uint64(13)
    h = (h * np.uint64(0xC2B2AE35)) & m
    h ^= h >> np.uint64(16)
    return h.astype(np.float64) / 2.0 ** 32


_bins = np.arange(int(np.ceil(TONE_LO_HZ / BIN_HZ)), int(TONE_HI_HZ / BIN_HZ) + 1)
_odd = _bins[_bins % 2 == 1]
# odd-order products of tones on odd bins land on odd bins: a fixed quarter of
# them left empty (HOLES) catches a symmetric limiter's; the even bins (GAPS)
# its even-order ones and the noise
HOLES = _odd[_hash(_odd, 7919) < 0.25]
TONES = np.setdiff1d(_odd, HOLES)
GAPS = _bins[_bins % 2 == 0]


def tone_hz(tones=None) -> np.ndarray:
    return (TONES if tones is None else tones) * BIN_HZ


def _band_hz(band: str) -> tuple[float, float]:
    b = modem.BANDS[band]
    lo = config.CARRIER0 + modem.RS * b.k0
    return lo, lo + modem.RS * (b.nc - 1)


def stair_tones() -> np.ndarray:
    lo, hi = _band_hz(STAIR_BAND)
    f = TONES * BIN_HZ
    return TONES[(f >= lo) & (f <= hi)]


def stair_holes() -> np.ndarray:
    lo, hi = _band_hz(STAIR_BAND)
    f = HOLES * BIN_HZ
    return HOLES[(f >= lo) & (f <= hi)]


def _tilt_gain(bins: np.ndarray) -> np.ndarray:
    """modem.tx_tilt_fir's amplitude curve (f108's band) at these bins."""
    lo, hi = _band_hz("f108")
    f = np.clip(bins * BIN_HZ, lo, hi)
    return (f / np.sqrt(lo * hi)) ** (config.FM_TX_TILT_DB_OCT / (20 * np.log10(2)))


def _period(tones: np.ndarray, tilt: bool = False) -> np.ndarray:
    """One period of these tones, unit rms; `tilt`: the flat path's."""
    X = np.zeros(PERIOD // 2 + 1, complex)
    X[tones] = np.exp(2j * np.pi * _hash(tones, 104729)) * (_tilt_gain(tones) if tilt else 1.0)
    x = np.fft.irfft(X, PERIOD)
    return x / np.sqrt(np.mean(x ** 2))


def _stair_period(tx_path: str) -> np.ndarray:
    """The staircase's period, conditioned as the top modes are, unit rms: mic,
    their clip; flat (unclipped from 16-QAM up), the tilt and a clip at the
    sound card's full scale less the loudest step."""
    from ..waveform import dsp

    if tx_path == "flat":
        c = config.FM_TX_PEAK * 10 ** (-max(STEPS_DB) / 20)
        p = np.clip(_period(stair_tones(), tilt=True), -c, c)
    else:
        head = modem.SUBMODES[f"{STAIR_BAND}-256l-r88"].headroom
        x = dsp.tx_condition(np.tile(_period(stair_tones()), 8), head, bandpass=modem.BANDS[STAIR_BAND].tx_bandpass)
        p = x[4 * PERIOD:5 * PERIOD]  # steady state: periodic again
    return p / np.sqrt(np.mean(p ** 2))


def n_samples() -> int:
    return int(round(GAP_S * FS)) + (SETTLE + SOUND_MEAS + len(STEPS_DB) * (SETTLE + MEAS)) * PERIOD


def waveform(tx_path: str | None = None, level: float = 1.0) -> np.ndarray:
    """The probe (silence GAP_S, the sounding, the staircase), unit rms x
    `level` at STEPS_DB 0, through the TX path."""
    path = modem.TX_PATH if tx_path is None else tx_path
    snd = np.tile(_period(TONES, tilt=path == "flat"), SETTLE + SOUND_MEAS) * 10 ** (SOUND_DB / 20)
    st = _stair_period(path)
    x = np.concatenate([snd] + [np.tile(st, SETTLE + MEAS) * 10 ** (g / 20) for g in STEPS_DB])
    return np.concatenate([np.zeros(int(round(GAP_S * FS))), x * level])


def _periods(y: np.ndarray) -> np.ndarray:
    """Every period's spectrum from the probe's start (the gap's end), the sample
    clocks' offset taken out: between periods a slip of tau samples turns bin k
    by -2 pi k tau / PERIOD, which coherent averaging over 10-20 periods cannot
    survive (100 ppm, a cheap USB sound card's, flattened the staircase). tau is
    fitted on the sounding (every tone, weighted by its strength) and every
    period is turned back by its index."""
    off = int(round(GAP_S * FS))
    n = (len(STEPS_DB) * (SETTLE + MEAS) + SETTLE + SOUND_MEAS)
    seg = y[off:off + n * PERIOD]
    if len(seg) < n * PERIOD:
        seg = np.concatenate([seg, np.zeros(n * PERIOD - len(seg))])
    F = np.fft.rfft(seg.reshape(n, PERIOD), axis=1)
    S = F[SETTLE - GUARD:SETTLE - GUARD + SOUND_MEAS, TONES]
    c = np.sum(S[1:] * np.conj(S[:-1]), axis=0)
    w, th = np.abs(c), np.angle(c)
    tau = -np.sum(w * TONES * th) / max(2 * np.pi / PERIOD * np.sum(w * TONES.astype(float) ** 2), 1e-30)
    r = np.arange(n)[:, None]
    k = np.arange(PERIOD // 2 + 1)[None, :]
    return F * np.exp(2j * np.pi * k * tau * r / PERIOD)


def _stretch(F: np.ndarray, r0: int, n: int, tones: np.ndarray, holes: np.ndarray) -> dict:
    T = F[r0:r0 + n][:, tones]
    m = T.mean(axis=0)
    noise = np.sum(np.abs(T - m) ** 2, axis=0) / (n - 1)  # per period
    return dict(sig=np.maximum(np.abs(m) ** 2 - noise / n, 0.0),  # |mean|^2 holds noise / n
                noise=noise, hole=np.mean(np.abs(F[r0:r0 + n][:, holes]) ** 2, axis=0))


def analyse(y: np.ndarray) -> dict:
    """The probe as received, `y` starting where waveform() starts (a few
    samples either way is harmless: a delay only turns each tone's phase).
    -> {"sound": stretch, "steps": [stretch per STEPS_DB]}; a stretch: sig and
    noise per tone, hole per empty odd bin (powers in the received audio's
    units, per period)."""
    F = _periods(np.asarray(y, dtype=np.float64))
    steps, r = [], SETTLE + SOUND_MEAS
    for _ in STEPS_DB:
        steps.append(_stretch(F, r + SETTLE - GUARD, MEAS, stair_tones(), stair_holes()))
        r += SETTLE + MEAS
    return dict(sound=_stretch(F, SETTLE - GUARD, SOUND_MEAS, TONES, HOLES), steps=steps)


def sinad_db(st: dict) -> float:
    """A staircase step's signal over noise plus distortion (the holes' excess
    over the tones' noise, spread over every tone)."""
    s, n = st["sig"].sum(), st["noise"].sum()
    d = max(float(np.mean(st["hole"])) - float(np.mean(st["noise"])), 0.0) * len(st["sig"])
    return float(10 * np.log10(max(s, 1e-30) / max(n + d, 1e-30)))


# The verdict (calibrated on the FM simulator, tests/test_fm_probe.py; on air: runs/onair/):
# CAP_DB: the SINAD the top modes need; more buys nothing, so the best drive is
# the step whose SINAD, capped here, is highest, the reference winning ties
# within TIE_DB, then the louder step. EDGE_DB: a sounding tone this far under
# the data band's median S/N is outside the passband. WIDE_HZ: a passband
# reaching this high carries WIDE (f108's top). VARIED_DB: the reference and its
# repeat this far apart (SINAD, capped: an over-driven radio's S/N alone moves
# between them): the link itself moved during the probe.
CAP_DB = 28.0
TIE_DB = 1.5
EDGE_DB = 10.0
WIDE_HZ = 5400.0
VARIED_DB = 2.0
REPORT_BYTES = 21


def _db(s: float, n: float = 1.0) -> float:
    """s / n in dB, each floored: a stretch of exact zeros (a digital squelch,
    our own transmission's gap in the history) is 0 dB, not 0/0."""
    return float(10 * np.log10(max(float(s), 1e-30) / max(float(n), 1e-30)))


def summary(a: dict, audio_level: dict) -> dict:
    """analyse() -> the numbers a report carries."""
    sd = a["sound"]
    f = tone_hz()
    snr = 10 * np.log10(np.maximum(sd["sig"], 1e-30) / np.maximum(sd["noise"], 1e-30))
    lo, hi = _band_hz(STAIR_BAND)
    nb = (f >= lo) & (f <= hi)
    thr = float(np.median(snr[nb])) - EDGE_DB
    ok = np.convolve(snr >= thr, np.ones(3), "same") >= 2  # one odd tone either way does not cut the band
    i = int(np.argmin(np.abs(f - 1000.0)))
    a_, b_ = i, i
    while a_ > 0 and ok[a_ - 1]:
        a_ -= 1
    while b_ < len(f) - 1 and ok[b_ + 1]:
        b_ += 1
    w_lo, w_hi = _band_hz("f108")
    top = (f >= w_hi - (w_hi - w_lo) / 3) & (f <= w_hi)
    thirds = np.array_split(np.where(nb)[0], 3)
    return dict(lo_hz=float(f[a_]) if ok[i] else 0.0, hi_hz=float(f[b_]) if ok[i] else 0.0,
                snr_nb=_db(sd["sig"][nb].sum(), sd["noise"][nb].sum()),
                snr_top=_db(sd["sig"][top].sum(), sd["noise"][top].sum()),
                snr_tilt=_db(sd["sig"][thirds[0]].sum(), sd["noise"][thirds[0]].sum())
                - _db(sd["sig"][thirds[2]].sum(), sd["noise"][thirds[2]].sum()),
                noise_tilt=_db(sd["noise"][thirds[2]].mean(), sd["noise"][thirds[0]].mean()),
                snr=[_db(s["sig"].sum(), s["noise"].sum()) for s in a["steps"]],
                sinad=[sinad_db(s) for s in a["steps"]], **audio_level)


def _q(v: float, step: float, lo: int, hi: int) -> int:
    return int(np.clip(round(v / step), lo, hi)) & 0xFF


def pack_report(m: dict) -> bytes:
    """summary() -> REPORT_BYTES: band edges (50 Hz), the sounding's S/N in the
    data band and WIDE's top third, its S/N tilt across the data band and its
    noise's (0.5 dB), S/N and SINAD per step (0.5 dB), input rms and peak
    (0.5 dBFS), clipped samples (per mille)."""
    s = [_q(m["lo_hz"], BIN_HZ, 0, 255), _q(m["hi_hz"], BIN_HZ, 0, 255), _q(m["snr_nb"], 0.5, -128, 127),
         _q(m["snr_top"], 0.5, -128, 127), _q(m["snr_tilt"], 0.5, -128, 127), _q(m["noise_tilt"], 0.5, -128, 127),
         *(_q(v, 0.5, -128, 127) for v in m["snr"]), *(_q(v, 0.5, -128, 127) for v in m["sinad"]),
         _q(m["rms_dbfs"], 0.5, -128, 0), _q(m["peak_dbfs"], 0.5, -128, 0), _q(m["clip_permille"], 1, 0, 255)]
    assert len(s) == REPORT_BYTES
    return bytes(s)


def unpack_report(b: bytes) -> dict:
    if len(b) < REPORT_BYTES:
        raise ValueError("probe: short report")
    s = [v - 256 if v > 127 else v for v in b[:REPORT_BYTES]]
    n = len(STEPS_DB)
    return dict(lo_hz=b[0] * BIN_HZ, hi_hz=b[1] * BIN_HZ, snr_nb=s[2] * 0.5, snr_top=s[3] * 0.5, snr_tilt=s[4] * 0.5,
                noise_tilt=s[5] * 0.5, snr=[v * 0.5 for v in s[6:6 + n]], sinad=[v * 0.5 for v in s[6 + n:6 + 2 * n]],
                rms_dbfs=s[6 + 2 * n] * 0.5, peak_dbfs=s[7 + 2 * n] * 0.5, clip_permille=b[8 + 2 * n])


def fmt(m: dict) -> str:
    """A report as host words (key=value, dB)."""
    return (f"band={m['lo_hz']:.0f}-{m['hi_hz']:.0f} snr={m['snr_nb']:.1f} top={m['snr_top']:.1f} "
            f"tilt={m['snr_tilt']:.1f} noise_tilt={m['noise_tilt']:.1f} "
            f"steps={'/'.join(f'{g:+.0f}' for g in STEPS_DB)} "
            f"step_snr={'/'.join(f'{v:.1f}' for v in m['snr'])} sinad={'/'.join(f'{v:.1f}' for v in m['sinad'])} "
            f"rms={m['rms_dbfs']:.1f} peak={m['peak_dbfs']:.1f} clip={int(m['clip_permille'])}")


def best_step(m: dict) -> float:
    """The drive change (dB) the staircase says suits the path: 0 at the ping's level."""
    n = len(STEPS_DB) - 1  # the repeat is not a level of its own
    cap = [min(v, CAP_DB) for v in m["sinad"][:n]]
    top = max(cap)
    if cap[REF] >= top - TIE_DB:
        return 0.0
    return max(STEPS_DB[i] for i in range(n) if cap[i] >= top - TIE_DB)


def verdict(m: dict, tx_path: str) -> list[str]:
    """Plain advice for the pinging station from the far end's report of its probe."""
    out = []
    if m["clip_permille"] or m["peak_dbfs"] >= -0.1:
        out.append("the far end's sound card clipped: it should lower its input level")
    if m["hi_hz"] <= 0:
        return out + ["the probe was not heard: check the level and the path"]
    wide = m["hi_hz"] >= WIDE_HZ
    out.append(f"passband {m['lo_hz']:.0f}-{m['hi_hz']:.0f} Hz: " + ("WIDE works" if wide else "NARROW only"))
    if tx_path == "mic" and m["snr_tilt"] >= 1.0 and m["noise_tilt"] >= 1.5:
        out.append(f"top of the data band {m['snr_tilt']:.0f} dB under the bottom (a flat 9600 path's noise): "
                   "set tx_path flat if this is a data port")
    if abs(min(m["sinad"][REF], CAP_DB) - min(m["sinad"][-1], CAP_DB)) >= VARIED_DB:
        out.append("S/N varied during the probe: ping again before changing the level")
        return out
    d = best_step(m)
    if d < 0:
        out.append(f"your audio drives the radio's limiter: lower the TX level about {-d:.0f} dB"
                   + (", or set tx_path mic on a mic/speaker radio" if tx_path == "flat" else ""))
    elif d > 0:
        out.append(f"more level helps: raise the TX level about {d:.0f} dB if the radio's data input has headroom")
    else:
        out.append(f"level OK (SINAD {m['sinad'][REF]:.0f} dB)")
    return out
