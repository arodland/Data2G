"""FM ping / level check (FM plan M7): is a station's TX path setting and
audio level right, checked on air without a session (VARA FM's Autotune is
what users expect).

A ping and its reply are the same burst: a T_PING frame filling MAX_CTL
control codewords of the band's most robust OFDM mode (PING_MODES), so the
receiver has 4 x 1100 (f55) or 4 x 1080 (f108) known QPSK cells to measure
the sender's transmitter with. It goes out at the level the sender's top
modes do (min config.FM_MIC_GAIN on the mic path), where the radio's limiter
matters. Each end measures the other's burst (measure); the reply carries
the replier's measurement of the ping (pack_report), so the pinger learns
how its own transmitter arrives, and a verdict (verdict) says what to change.

Body: from call, to call (frames.pack_call, 8 bytes each), flags (bit 0:
a reply, which carries an 8-byte report), then zeros to fill the codewords.
FM only (frames.T_PING); HF never sends or reads it."""

import numpy as np

from .. import config, modem
from ..config import FS
from . import frames as F

PING_MODES = {1: "f55-qpsk-r21", 2: "f108-qpsk-r21"}  # cap (NARROW, WIDE) -> mode
REPLY_HOLDOFF_S = 10.0  # one reply per caller per this long (an unattended station is not a beacon)
REPORT_BYTES = 9

# Verdict thresholds, measured on the FM simulator (scripts/fm_ping_study.py,
# runs/fm_ping/: 5 station setups, static and flutter at 440, CNR 12-30):
# DIST_DB: the per-frame median of the decision-directed error over the
# pilot-based noise, dB, above which the sender's audio drives its radio's
# limiter. f55: correct setups read at most 1.38 (mic, flutter at 30 dB), the
# limiter at least 1.64; f108 (flat radios only, no limiter) reads up to 1.95
# from the flat radio's filters. TILT: a flat (9600) receiver's noise rises with
# f^2 (NOISE_TILT_DB: its top third over its bottom; mic receivers read <= 0,
# flat ones +1.4..+4.0); an untilted sender's S/N then falls across the band
# (TILT_DB, bottom over top: tx_path mic on a flat radio +0.9..+2.7, tx_path
# flat -2.1..-3.0). MIN_SNR_DB: under it distortion hides in the noise.
DIST_DB = {"f55": 1.5, "f108": 2.5}
TILT_DB = 1.0
NOISE_TILT_DB = 1.5
MIN_SNR_DB = 12.0
CLIP_DBFS = -0.1  # an input peak this close to full scale: the sound card clipped


def tx_level(tx_path: str) -> float:
    """A ping's level against its mode's own: the quietest top mode's on the
    mic path (config.FM_MIC_GAIN), where the radio's limiter is what matters."""
    return min(config.FM_MIC_GAIN) if tx_path == "mic" else 1.0


def fmt(m: dict) -> str:
    """A measurement as event / host words: key=value, dB to 0.1."""
    return (f"snr={m['snr_db']:.1f} rms={m['rms_dbfs']:.1f} peak={m['peak_dbfs']:.1f} clip={int(m['clip_permille'])} "
            f"thirds={'/'.join(f'{v:.1f}' for v in m['thirds_db'])} dist={m['dist_db']:.1f} "
            f"noise_tilt={m['noise_tilt_db']:.1f}")


def ping_body(frm: str, to: str, report: bytes | None, payload_bytes: int) -> bytes:
    """The T_PING extension body, zero-filled so the control takes MAX_CTL codewords."""
    head = F.pack_call(frm) + F.pack_call(to) + bytes([1 if report is not None else 0]) + (report or b"")
    fill = F.MAX_CTL * payload_bytes - 4 - 2 - len(head)  # core (4) and the extension's type/length (2)
    return head + bytes(min(255 - len(head), max(0, fill)))


def unpack_ping(body: bytes) -> tuple[str, str, bytes | None]:
    """-> (from, to, the report if a reply). ValueError if malformed."""
    if len(body) < 17:
        raise ValueError("ping: short")
    frm, to, flags = F.unpack_call(body[:8]), F.unpack_call(body[8:16]), body[16]
    if flags & 1:
        if len(body) < 17 + REPORT_BYTES:
            raise ValueError("ping: short report")
        return frm, to, body[17:17 + REPORT_BYTES]
    return frm, to, None


def _q(v: float, step: float, lo: int, hi: int) -> int:
    return int(np.clip(round(v / step), lo, hi))


def pack_report(m: dict) -> bytes:
    """A measurement (measure) -> 9 bytes: snr (0.25 dB), input rms and peak
    (0.5 dBFS), clipped samples (per mille), the band thirds' S/N (0.5 dB),
    distortion (0.1 dB), the noise's top third over its bottom (0.5 dB).
    Signed fields as two's complement bytes."""
    s = [_q(m["snr_db"], 0.25, -128, 127), _q(m["rms_dbfs"], 0.5, -128, 0), _q(m["peak_dbfs"], 0.5, -128, 0),
         _q(m["clip_permille"], 1, 0, 255), *(_q(v, 0.5, -128, 127) for v in m["thirds_db"]),
         _q(m["dist_db"], 0.1, 0, 255), _q(m["noise_tilt_db"], 0.5, -128, 127)]
    return bytes(v & 0xFF for v in s)


def unpack_report(b: bytes) -> dict:
    s = [v - 256 if v > 127 else v for v in b[:REPORT_BYTES]]
    return dict(snr_db=s[0] * 0.25, rms_dbfs=s[1] * 0.5, peak_dbfs=s[2] * 0.5, clip_permille=b[3],
                thirds_db=[v * 0.5 for v in s[4:7]], dist_db=b[7] * 0.1, noise_tilt_db=s[8] * 0.5)


def _dither(n: int) -> np.ndarray:
    """A tiny deterministic noise (the receiver needs a noise estimate) that
    native reproduces exactly: a multiplicative hash of the sample index."""
    k = np.arange(n, dtype=np.uint64)
    return 1e-4 * (((k * np.uint64(2654435761)) % np.uint64(1 << 32)).astype(np.float64) / 2.0**31 - 1.0)


def level(x: np.ndarray) -> dict:
    """Input audio level over a burst's samples (full scale 1.0)."""
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return dict(rms_dbfs=-64.0, peak_dbfs=-64.0, clip_permille=0)
    db = lambda p: float(10 * np.log10(max(p, 1e-7)))
    return dict(rms_dbfs=db(np.mean(x ** 2)), peak_dbfs=db(np.max(x ** 2)),
                clip_permille=int(round(1000 * np.mean(np.abs(x) >= 10 ** (CLIP_DBFS / 20)))))


def measure(r: dict, payloads: list[bytes], audio_level: dict) -> dict:
    """What a ping burst says about its sender's transmitter. `r`: its
    modem.receive result; `payloads`: its decoded control codewords;
    `audio_level`: level() of its samples as they came in.
    - snr_db: the receiver's snr_est (what the gear shifter reads);
    - thirds_db: per-carrier S/N averaged over the bottom, middle and top
      third of the band (pilot-based channel and noise): a flat path's f^2
      noise slopes it down unless the sender tilts (tx_path flat);
    - dist_db: the error of every data cell against the channel times the
      cell that was sent (the decoded burst rebuilt), over the pilot-based
      noise: the radio's limiter distorts the signal, not the pilots, so it
      shows here and nowhere else (fork K: it does not show in the pilots)."""
    from . import burst as B
    from .phy import measure as phy_measure
    from .phy import tx_audio

    spec, est = r["spec"], r["est"]
    sent = B.TxBurst(spec.name, [B.Slot(B.ctl_mask(0, i, 0), 0, p) for i, p in enumerate(payloads)], 0)
    pad = np.zeros(FS // 2)
    clean = np.concatenate([pad, tx_audio(sent, tx_path="mic", level=1.0), pad])
    rc = modem.receive(clean + _dither(len(clean)))  # acquired as any burst: native has no known-timing receive
    # each side over its own pilot-based channel: the clean burst is read at its
    # own timing, so its cells carry another phase slope across the carriers
    y, x = modem.data_symbols(r["raw"]), modem.data_symbols(rc["raw"]) / rc["est"]["h"]
    h = est["h"]
    a = np.sum(np.conj(h * x) * y) / np.sum(np.abs(h * x) ** 2)  # the one gain the estimate may miss
    n0 = np.broadcast_to(est.get("n0_k", est["n0"]), h.shape)
    var = n0 + est["mse"]
    # the median over frames: the limiter distorts every frame, while flutter's
    # clicks and fades put errors the pilots miss in a few (a mean read 3 dB on
    # a correct mic setup at 440 MHz; scripts/fm_ping_study.py)
    per_frame = np.mean(np.abs(y - a * h * x) ** 2, axis=(1, 2)) / np.mean(var, axis=(1, 2))
    dist = float(10 * np.log10(np.median(per_frame)))
    n0_c = np.mean(n0, axis=(0, 1))  # per carrier
    p = np.mean(np.abs(h) ** 2 * np.mean(np.abs(x) ** 2), axis=(0, 1))
    thirds = [float(10 * np.log10(np.mean(t))) for t in np.array_split(p / n0_c, 3)]
    nt = [float(10 * np.log10(np.mean(t))) for t in np.array_split(n0_c, 3)]
    return dict(snr_db=float(phy_measure(r)["snr_est"]), thirds_db=thirds, noise_tilt_db=nt[2] - nt[0],
                dist_db=max(0.0, dist), **audio_level)


def verdict(far: dict, tx_path: str, band: str = "f55") -> list[str]:
    """Plain advice for the pinging station from `far`, the far end's
    measurement of its burst (unpack_report) in `band`, given its own tx_path."""
    out = []
    tilt = far["thirds_db"][0] - far["thirds_db"][2]
    if far["snr_db"] < MIN_SNR_DB:
        out.append(f"S/N {far['snr_db']:.0f} dB: too low to judge distortion")
    elif far["dist_db"] > DIST_DB[band]:
        out.append("your audio drives the radio's limiter: lower the TX level"
                   + (", or set tx_path mic on a mic/speaker radio" if tx_path == "flat" else ""))
    if tx_path == "mic" and tilt >= TILT_DB and far["noise_tilt_db"] >= NOISE_TILT_DB:
        out.append(f"top of band {tilt:.0f} dB under the bottom (a flat 9600 path's noise): set tx_path flat if this is a data port")
    if far["clip_permille"] or far["peak_dbfs"] >= CLIP_DBFS:
        out.append("the far end's sound card clipped: it should lower its input level")
    return out or ["level OK"]
