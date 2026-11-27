"""Every mode the ARQ can use: the frozen OFDM ladder (config.SUBMODES)
and the constant-envelope CPM modes (data2g.cpm), with what differs by
family: burst length, control codeword, how a burst goes on air; the
bandwidth each occupies, and the recommendation code that names a mode
on air (the core word's 6-bit recommend, docs/arq.md §4, §8).
Native: arq/modes.hpp."""

from .. import codes, cpm, modem
from ..config import PROFILE, SUBMODES
from .frames import BANDS_CODE, MAX_CTL

# Every mode compiled in (HF: those on a band with a recommendation code).
FM = PROFILE == "fm"
MODES = ({**SUBMODES, **cpm.SPECS} if FM else
         {**{n: s for n, s in SUBMODES.items() if s.sync_band in BANDS_CODE}, **cpm.SPECS})


def is_cpm(spec) -> bool:
    return getattr(spec, "family", "ofdm") == "cpm"


# What the ARQ may choose and name on air. FM (plan M6, Q2): its bands with a
# code (frames.BANDS_CODE: w, n10, f55, f108), indices under 16 (n10's past 15
# take CPM's code on HF); no w48, no CPM: compiled, never chosen.
SELECTABLE = frozenset(n for n, s in MODES.items() if not is_cpm(s) and s.sync_band in BANDS_CODE and s.index < 16
                       ) if FM else frozenset(MODES)


def burst_seconds(spec, n_cw: int, dup: bool = False) -> float:
    """On air, n_cw slots (control included; `dup`: a CPM burst's control twice)."""
    return cpm.burst_seconds(spec, n_cw, dup) if is_cpm(spec) else modem.burst_seconds(spec, n_cw)


def ctl_spec(spec):
    """The codeword a burst of `spec` carries its control in."""
    return cpm.CTL[spec.grid] if is_cpm(spec) else spec


def ctl_payload_bytes(spec) -> int:
    return codes.payload_bytes(ctl_spec(spec))


def max_ctl(spec) -> int:
    """Control codewords a burst may carry: CPM's header can say only one
    (twice when duplicated)."""
    return 1 if is_cpm(spec) else MAX_CTL


def min_cw(spec, data: bool) -> int:
    """Fewest slots a data burst gets: a CPM codeword is long (one data
    codeword needs 5.5-15.5 s with its control), so size classes are floors."""
    return 2 if (data and is_cpm(spec)) else 1


# --- bandwidth --------------------------------------------------------------------

# bandwidth cap codes (a session's, KISS's) and their names (VARA HF: BW2300).
# FM: 500 Hz (the n10 floor), narrow (w and f55: 450..3150 Hz) and wide
# (f108: 300..5650 Hz), named as VARA FM's NARROW / WIDE; the per-direction
# capability that replaces one cap per session is later M6 work
CAP_HZ = {0: 500, 1: 2700, 2: 5350} if FM else {0: 500, 1: 1200, 2: 2400}
BW_NAME = {0: "500", 1: "NARROW", 2: "WIDE"} if FM else {0: "500", 1: "1200", 2: "2300"}
WIDTH_HZ = {"n4": 200, "n10": 500, "w": 1200, "w48": 2400, "f55": 2700, "f108": 5350}


def width_hz(spec) -> float:
    return cpm.GRIDS[spec.band].bandwidth if is_cpm(spec) else WIDTH_HZ[spec.band]


def allowed(cap: int) -> list:
    """The modes within cap `cap`, MODES' order (policy.allowed: less the study's dropped ones)."""
    return [s for s in MODES.values() if width_hz(s) <= CAP_HZ[cap] and s.name in SELECTABLE]


# --- the recommendation code ---------------------------------------------------------

REC_EXT = 63  # sentinel in Core.recommend: read extended mode code from T_REC_EXT TLV

# Flat 0..62 recommendation table (docs/arq.md §4). Ordered on HF to maximize
# nods' +-2 step across the 500 Hz CPM floor, the wide CPM / OFDM crossover, and
# each band's data ladder, with polar and ACK-only modes at the top.
HF_RECOMMEND = (
    # 0..5: 500 Hz CPM floor (all caps; r1/3<->r1/2 = +-1, cross-grid same rate = +-2)
    "fsk8w50-r1/3", "fsk8w50-r1/2",
    "fsk16r25-r1/3", "fsk16r25-r1/2",
    "fsk8r50-r1/3", "fsk8r50-r1/2",
    # 6..8: 2300 Hz CPM (fsk8r50-r1/2 <-> fsk32r62-r1/2 = +-2; r1/2 <-> r2/3 = +-1)
    "fsk32r62-r1/3", "fsk32r62-r1/2", "fsk32r62-r2/3",
    # 9..19: w + low w48 interleaved by threshold (fsk32r62-r1/2 <-> qpsk-r1/5 = +-2;
    #        adjacent cap-1 (w) and cap-2 (w48) rungs are <= 2 steps apart)
    "qpsk-r1/5", "qpsk-r1/3", "w48-qpsk-r1/5", "qpsk-r1/2", "w48-qpsk-r1/3",
    "16qam-r1/3", "w48-qpsk-r1/2", "qpsk-r3/4", "w48-16qam-r1/3", "16qam-r1/2", "w48-qpsk-r2/3",
    # 20..29: high w48
    "w48-qpsk-r3/4", "w48-16qam-r1/2", "w48-16qam-r2/3", "w48-64l-r1/2",
    "w48-16qam-r3/4", "w48-64l-r7/12", "w48-16qam-r5/6", "w48-64l-r2/3",
    "w48-64l-r3/4", "w48-256l-r5/8",
    # 30..39: n10 (500 Hz OFDM, contiguous)
    "n10-qpsk-r1/5", "n10-qpsk-r1/3", "n10-qpsk-r1/2", "n10-16qam-r1/3",
    "n10-qpsk-r3/4", "n10-16qam-r1/2", "n10-16qam-r2/3", "n10-16qam-r3/4",
    "n10-64l-r3/4", "n10-256l-r3/4",
    # 40..44: n4 (200 Hz OFDM, contiguous)
    "n4-qpsk-r1/5", "n4-qpsk-r1/3", "n4-qpsk-r1/2", "n4-qpsk-r2/3", "n4-16qam-r1/3",
    # 45..52: polar & ACK-only (out of every data ladder)
    "polar-k96-f8", "polar-k96-f4", "polar-k192-f8",
    "ack-4f", "ack-1f", "n10-ack-4f", "n4-ack-8f", "n4-ack-2f",
)

FM_RECOMMEND = (
    # 0..7: n10 data
    "n10-qpsk-r1/5", "n10-qpsk-r1/3", "n10-qpsk-r1/2", "n10-16qam-r1/3",
    "n10-qpsk-r3/4", "n10-16qam-r1/2", "n10-16qam-r2/3", "n10-16qam-r3/4",
    # 8..13: w data
    "qpsk-r1/5", "qpsk-r1/3", "qpsk-r1/2", "16qam-r1/3", "qpsk-r3/4", "16qam-r1/2",
    # 14..27: f55 data
    "f55-qpsk-r21", "f55-qpsk-r40", "f55-qpsk-r60", "f55-16qam-r47",
    "f55-64l-r50", "f55-64l-r67", "f55-64l-r75", "f55-64l-r80", "f55-64l-r83",
    "f55-256l-r67", "f55-256l-r72", "f55-256l-r80", "f55-256l-r83", "f55-256l-r88",
    # 28..41: f108 data
    "f108-qpsk-r21", "f108-qpsk-r41", "f108-qpsk-r60", "f108-16qam-r41",
    "f108-64l-r40", "f108-64l-r50", "f108-64l-r59", "f108-64l-r67", "f108-64l-r75", "f108-64l-r80",
    "f108-256l-r67", "f108-256l-r75", "f108-256l-r80", "f108-256l-r87",
    # 42..47: polar & ACK-only
    "polar-k96-f8", "polar-k96-f4", "polar-k192-f8",
    "ack-4f", "ack-1f", "n10-ack-4f",
)

RECOMMEND_TABLE = FM_RECOMMEND if FM else HF_RECOMMEND
_REC_BY_NAME = {name: i for i, name in enumerate(RECOMMEND_TABLE)}
assert len(RECOMMEND_TABLE) < REC_EXT and len(_REC_BY_NAME) == len(RECOMMEND_TABLE)
assert set(RECOMMEND_TABLE) == SELECTABLE, set(RECOMMEND_TABLE) ^ SELECTABLE


def encode(submode: str) -> int:
    code = _REC_BY_NAME.get(submode)
    if code is None:
        if submode not in MODES:
            raise KeyError(submode)
        raise ValueError(f"{submode}: no recommendation code on {'FM' if FM else 'HF'}")
    return code


def decode(rec: int) -> str | None:
    if 0 <= rec < len(RECOMMEND_TABLE):
        return RECOMMEND_TABLE[rec]
    return None
