"""The ARQ layer on the real modem: link.TxBurst -> audio, a received
burst -> link.RxBurst (masked CRCs, docs/arq.md §2; soft-bit combining
across resends, §5), and the gear shifter's measurements.

Used by scripts/phy_session.py (phase G: whole sessions through the modem
and hfchannel) and the live stack (phase H).
"""

import zlib

import numpy as np

from .. import codes, modem
from ..config import BANDS, RS, SNR_REF_BW_HZ, SUBMODES
from . import predictor as P


def mask_value(mask_id: tuple) -> int:
    """A slot's CRC mask from (session key, direction, seq or 128 + control
    index). Key 0 (connect frames, before a session) is mask 0."""
    key, direction, s = mask_id
    if key == 0:
        return 0
    return zlib.crc32(bytes([key >> 8, key & 255, direction, s])) or 1


def tx_audio(burst) -> np.ndarray:
    """A link.TxBurst -> unit-RMS audio. Codewords scramble by their mask
    alone (index 0), so a resend in another slot combines."""
    spec = SUBMODES[burst.submode]
    bits = np.stack([codes.encode(spec, s.payload, s.rv, mask_value(s.mask_id)) for s in burst.slots])
    return modem.modulate_bits(codes.spread(bits, spec.bits_per_cu), spec)


def measure(r: dict) -> dict:
    """The gear shifter's inputs from modem.receive's result: effective MI
    against thermal noise and estimation error, not the transmitter's clip
    noise (it belongs to the submode sent, not to the channel)."""
    est = r["est"]
    h, var = est["h"], est["n0"] + est["mse"]
    nc = BANDS[r["band"]].nc
    d0, d1 = r["support"]
    out = dict(snr_est=10 * np.log10(est["p_sig"] / est["n0"] * nc * RS / SNR_REF_BW_HZ),
               spread_est=est["spread_hz"], delay_est_ms=(d1 - d0) / modem.FS * 1000,
               headroom=r["spec"].headroom, frames=r["n_cw"] * r["spec"].frames_per_cw)
    for c in P.CONSTS:
        out[f"mi_{c}"] = P.effective_mi(h, var, c)
    return out


def soft_bits(r: dict) -> np.ndarray:
    """(n_cw, coded_bits) soft bits in mapping order from modem.receive's result."""
    spec, est = r["spec"], r["est"]
    var = modem.noise_var(est["h"], est) + est["mse"]
    return np.asarray(codes.despread(modem.soft_bits(r["raw"], est["h"], var, spec), r["n_cw"], spec.bits_per_cu))


class ModemRx:
    """link.RxBurst over a received burst. `store`: this station's soft
    bits per codeword key, {key: (buffer, highest RV, submode, where)}, kept
    across bursts.

    ponytail: one decode per call (the link asks slot by slot, masks known
    only then); batch the data slots if latency on long polar bursts bites."""

    def __init__(self, r: dict, store: dict):
        self.spec, self.n_cw, self.submode = r["spec"], r["n_cw"], r["spec"].name
        self.soft, self.store = soft_bits(r), store

    def decode(self, slot: int, mask_id: tuple, rv: int, key: tuple | None) -> bytes | None:
        if slot >= self.n_cw:
            return None
        m = mask_value(mask_id)
        if key is None:
            payload, ok = codes.decode_many(self.spec, self.soft[slot:slot + 1], m, index=0)[0]
            return payload if ok else None
        buf, top, name, where = self.store.get(key, (None, 0, self.submode, None))
        if name != self.submode:
            raise AssertionError(f"soft bits of {key} stored in {name} ({where}), resent in {self.submode} "
                                 f"slot {slot} rv {rv}")
        buf = codes.combine(self.spec, None if buf is None else buf.copy(), self.soft[slot:slot + 1], rv)
        top = max(top, rv)
        payload, ok = codes.decode_buffer(self.spec, buf, top, m, index=0)[0]
        if ok:
            return payload
        self.store[key] = (buf, top, self.submode, (slot, rv, mask_id))
        return None

    def forget(self, key: tuple) -> None:
        self.store.pop(key, None)
