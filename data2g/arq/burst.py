"""What a burst is to the PHY (docs/arq.md §2, §3): its slots, each a
payload at an RV under a codeword identity (mask_id), and the received
side's interface. Shared by the ARQ (link, session, engine) and by KISS,
which has no link.Station; phy.py turns a mask_id into the CRC mask
value and a TxBurst into audio. Native: arq/burst.hpp."""

from dataclasses import dataclass
from typing import Protocol

from .frames import SEQ_MOD

# --- codeword identity: (session key, direction byte, 7-bit seq or SEQ_MOD + control index) ---


def ctl_mask(direction: int, i: int, key: int = 0) -> tuple:
    return (key, direction, SEQ_MOD + i)


COMPACT_CONNECT = ctl_mask(0, 4)  # a compact CONNECT's mask (frames.pack_connect); ctl_mask's i is 0-3
EPOCH_MOD = 64  # abandon epochs in a data codeword's identity: the direction byte's 6 spare bits


def data_mask(direction: int, seq: int, key: int = 0, comp: bool = False, epoch: int = 0) -> tuple:
    """A data codeword's identity. `comp`: deflated (T_COMP); `epoch`: the
    sender's abandon epoch (its slicing). Both are folded into the
    direction byte, so a codeword decoded under the wrong compression or
    slicing assumption fails its CRC (docs/arq.md §2, §9a)."""
    return (key, direction | 2 * comp | 4 * (epoch % EPOCH_MOD), seq % SEQ_MOD)


# --- bursts ------------------------------------------------------------------------

@dataclass
class Slot:
    mask_id: tuple  # (session key, direction, 7-bit seq or 128 + control index)
    rv: int
    payload: bytes


@dataclass
class TxBurst:
    submode: str
    slots: list[Slot]
    burst_seq: int  # absolute count of this station's bursts
    cap: int = 0  # bandwidth cap code the TX filter is for (cpm.TX_FILTERS; modes.CAP_HZ)


def dup_ctl(burst) -> bool:
    """Its control sent twice (ARQ_DUP): a control slot at RV 1."""
    return any(s.rv for s in burst.slots if s.mask_id[2] >= SEQ_MOD)


class RxBurst(Protocol):
    """A received burst (phy.ModemRx on air, fakes in tests)."""
    submode: str
    n_cw: int

    def decode(self, slot: int, mask_id: tuple, rv: int, key: tuple | None) -> bytes | None:
        """Payload if slot `slot` decodes with this mask at this RV
        (combined with soft bits stored under `key`, which a failed
        decode adds to), else None."""

    def forget(self, key: tuple) -> None:
        """Drop soft bits stored under `key`."""
