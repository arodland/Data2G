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

CPM_CODE = 3  # the recommendation's band code for CPM modes (index: data2g.cpm.SPECS' order)
# ... which leaves its indices 8-15 to n10's past 15 (16-23; CW_BITS gave n10
# 32): the recommendation is 6 bits, band (2) | index (4), every band code taken
N10_HIGH = 8
assert len(cpm.SPECS) <= N10_HIGH
BY_CODE = {code: band for band, code in BANDS_CODE.items()}


def encode(submode: str) -> int:
    s = MODES[submode]
    if FM:  # every code a band, every index under 16 (SELECTABLE)
        if submode not in SELECTABLE:
            raise ValueError(f"{submode}: no recommendation code on FM")
        return BANDS_CODE[s.sync_band] << 4 | s.index
    if is_cpm(s):
        return CPM_CODE << 4 | list(cpm.SPECS).index(submode)
    if s.sync_band == "n10" and s.index >= 16:
        assert s.index < 16 + 16 - N10_HIGH, f"{submode}: no recommendation code"
        return CPM_CODE << 4 | (N10_HIGH + s.index - 16)
    return BANDS_CODE[s.sync_band] << 4 | s.index


def decode(rec: int) -> str | None:
    if FM:
        s = modem.BY_INDEX.get((BY_CODE[rec >> 4], rec & 15))
        return s.name if s and s.name in SELECTABLE else None
    if rec >> 4 == CPM_CODE:
        i, names = rec & 15, list(cpm.SPECS)
        if i >= N10_HIGH:
            s = modem.BY_INDEX.get(("n10", 16 + i - N10_HIGH))
            return s.name if s else None
        return names[i] if i < len(names) else None
    band = BY_CODE.get(rec >> 4)
    s = modem.BY_INDEX.get((band, rec & 15)) if band else None
    return s.name if s else None
