"""FM ping / level check (FM plan M7): is a station's TX path setting and
audio level right, and which bandwidth does the path carry, checked on air
without a session (VARA FM's Autotune is what users expect).

A ping is one over: an identity burst (a T_PING frame in PING_MODE, f55, which
any FM path carries) and the sounding probe right behind it (arq.probe: known
multitones, measured without decoding). The station pinged measures the probe
and replies the same way: an identity burst carrying its report of the ping's
probe, and its own probe. One round trip gives the pinger how its transmitter
arrives there (the report: passband, S/N, the level staircase) and how the
other's arrives here (its own analysis of the reply's probe), and a verdict
(probe.verdict).

Body: from call, to call (frames.pack_call, 8 bytes each), flags (bit 0: a
reply, which carries a probe.REPORT_BYTES report; bit 1 reserved: a deviation
block after the report, from a receiver with an IF output; readers ignore what
follows the report). FM only (frames.T_PING);
HF never sends or reads it."""

import numpy as np

from .. import config
from . import frames as F
from . import probe as PR

PING_MODE = "f55-qpsk-r21"
REPLY_HOLDOFF_S = 10.0  # one reply per caller per this long (an unattended station is not a beacon)
CLIP_DBFS = -0.1  # an input peak this close to full scale: the sound card clipped


def tx_level(tx_path: str) -> float:
    """A probe's reference level against unit rms: the quietest top mode's on
    the mic path (config.FM_MIC_GAIN), where the radio's limiter is what matters."""
    return min(config.FM_MIC_GAIN) if tx_path == "mic" else 1.0


def ping_body(frm: str, to: str, report: bytes | None) -> bytes:
    """The T_PING extension body."""
    return F.pack_call(frm) + F.pack_call(to) + bytes([1 if report is not None else 0]) + (report or b"")


def unpack_ping(body: bytes) -> tuple[str, str, bytes | None]:
    """-> (from, to, the report if a reply). ValueError if malformed."""
    if len(body) < 17:
        raise ValueError("ping: short")
    frm, to, flags = F.unpack_call(body[:8]), F.unpack_call(body[8:16]), body[16]
    if flags & 1:
        if len(body) < 17 + PR.REPORT_BYTES:
            raise ValueError("ping: short report")
        return frm, to, body[17:17 + PR.REPORT_BYTES]
    return frm, to, None


def level(x: np.ndarray) -> dict:
    """Input audio level over the probe's samples (full scale 1.0)."""
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return dict(rms_dbfs=-64.0, peak_dbfs=-64.0, clip_permille=0)
    db = lambda p: float(10 * np.log10(max(p, 1e-7)))
    return dict(rms_dbfs=db(np.mean(x ** 2)), peak_dbfs=db(np.max(x ** 2)),
                clip_permille=int(round(1000 * np.mean(np.abs(x) >= 10 ** (CLIP_DBFS / 20)))))
