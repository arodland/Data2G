"""Every mode the ARQ can use: the frozen OFDM ladder (config.SUBMODES)
and the constant-envelope CPM modes (data2g.cpm), with what differs by
family: burst length, control codeword, how a burst goes on air."""

from .. import codes, cpm, modem
from ..config import SUBMODES

MODES = {**SUBMODES, **cpm.SPECS}


def is_cpm(spec) -> bool:
    return getattr(spec, "family", "ofdm") == "cpm"


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
    return 1 if is_cpm(spec) else 4


def min_cw(spec, data: bool) -> int:
    """Fewest slots a data burst gets: a CPM codeword is long (one data
    codeword needs 5.5-15.5 s with its control), so size classes are floors."""
    return 2 if (data and is_cpm(spec)) else 1
