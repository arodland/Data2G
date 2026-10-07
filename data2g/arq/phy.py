"""The ARQ layer on the real modem: link.TxBurst -> audio, a received
burst -> link.RxBurst (masked CRCs, docs/arq.md §2; soft-bit combining
across resends, §5), and the gear shifter's measurements.

Used by scripts/phy_session.py (phase G: whole sessions through the modem
and hfchannel) and the live stack (phase H).
"""

import os
import time
import zlib
from functools import lru_cache

import numpy as np

from .. import codes, constellation, cpm, equalizer, modem
from ..config import BANDS, NSYM, RS, SNR_REF_BW_HZ, SUBMODES, SYMS_PER_FRAME
from ..waveform import ofdm
from . import predictor as P
from .frames import SEQ_MOD
from .modes import MODES, ctl_spec, is_cpm

# DATA2G_DD (default 1; 0 turns it off): decision-directed re-estimation. When a slot
# fails, its decoder's a-posteriori LLRs and every decoded codeword of the
# burst become soft pilots (equalizer.refine), and it is decoded again, up
# to DD_ITERS times.
DD = os.environ.get("DATA2G_DD", "1") == "1"
DD_ITERS = 2
# a live receiver's DD stops starting refines this long after the burst's
# decode began: the reply must start within REPLY_START_S of the sender's
# turn, and a 12 s 256l burst with every slot failing took 54 s of DD
DD_BUDGET_S = 1.0


def mask_value(mask_id: tuple) -> int:
    """A slot's CRC mask from (session key, direction, seq or 128 + control
    index). Key 0 (connect frames, before a session) is mask 0, except a
    compact CONNECT's (link.COMPACT_CONNECT)."""
    key, direction, s = mask_id
    if key == 0:
        return 0xC0C0C0 if s == SEQ_MOD + 4 else 0
    return zlib.crc32(bytes([key >> 8, key & 255, direction, s])) or 1


@lru_cache(maxsize=None)
def peak_db(submode: str) -> float:
    """A mode's burst envelope peak over its average power (dB), measured once
    on a fixed burst (a control and a data codeword, seeded payloads): the
    energy inputs report a burst's SNR against its peak (a peak-limited
    transmitter), with this per mode. Rounded to 0.001 dB: the FFT differs
    by an ulp across machines, and gen_native_tables must regenerate the same
    bytes everywhere."""
    from .. import hfchannel
    from .link import Slot, TxBurst, ctl_mask, data_mask

    spec = MODES[submode]
    rng = np.random.default_rng(0)
    rand = lambda sp: bytes(rng.integers(0, 256, codes.payload_bytes(sp), dtype=np.uint8))  # noqa: E731
    burst = TxBurst(submode, [Slot(ctl_mask(0, 0, 5), 0, rand(ctl_spec(spec))), Slot(data_mask(0, 1, 5), 0, rand(spec))], 0)
    x = tx_audio(burst)
    return round(float(10 * np.log10(np.max(np.abs(hfchannel._analytic(x)) ** 2) / 2 / hfchannel.active_power(x))), 3)


def tx_audio(burst) -> np.ndarray:
    """A link.TxBurst -> unit-RMS audio. Codewords scramble by burst
    position alone (codes.scramble_seed: no key on air); a resend in
    another slot still combines (codes.flip)."""
    spec = MODES[burst.submode]
    if is_cpm(spec):
        # control slots (masks >= 128) in the grid's short codeword; two of
        # them are one codeword at RV 0 and 1 (ARQ_DUP)
        n_ctl = sum(1 for s in burst.slots if s.mask_id[2] >= 128)
        coded = [codes.encode(ctl_spec(spec) if i < n_ctl else spec, s.payload, s.rv, mask_value(s.mask_id), i)
                 for i, s in enumerate(burst.slots)]
        return cpm.modulate(spec, coded, dup=n_ctl == 2)
    bits = np.stack([codes.encode(spec, s.payload, s.rv, mask_value(s.mask_id), i) for i, s in enumerate(burst.slots)])
    return modem.modulate_bits(codes.spread(bits, spec.bits_per_cu), spec)


def measure(r: dict) -> dict:
    if r.get("family") == "cpm":
        return cpm.measure(cpm.GRIDS[r["band"]], r["E"], len(r["E"]))
    return _measure_ofdm(r)


def _measure_ofdm(r: dict) -> dict:
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


def soft_bits(r: dict):
    """Per slot soft bits in mapping order ((n_cw, coded_bits) for OFDM; a
    list, control and data lengths differing, for CPM)."""
    if r.get("family") == "cpm":
        return r["soft"]
    return _soft_ofdm(r)


def _dd_estimate(r: dict, post: dict) -> dict:
    """r["est"] re-made by equalizer.refine with soft symbols as pilots:
    `post` {slot: a-posteriori LLRs of its coded bits, mapping order}
    (a decoded codeword's are its bits, as large LLRs)."""
    spec, est, raw = r["spec"], r["est"], r["raw"]
    pts = constellation.load(spec.constellation)
    m = constellation.bits_per_symbol(pts)
    llr = np.zeros((r["n_cw"], spec.coded_bits))
    have = np.zeros(llr.shape, dtype=np.uint8)
    for s, v in post.items():
        llr[s], have[s] = v, 1
    shape = raw[:, 1:].shape
    L = np.clip(codes.spread(llr, m).reshape(-1, m), -30, 30)
    labels = (np.arange(len(pts))[:, None] >> np.arange(m - 1, -1, -1)) & 1  # (M, m), modulate's order
    # log P(label) as matmuls: the (symbols, M, m) select-and-sum was 70% of a 256l refine
    lp = -np.logaddexp(0, L) @ labels.T - np.logaddexp(0, -L) @ (1 - labels).T
    prob = np.exp(lp - lp.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    x = (prob @ pts).reshape(shape)
    v = (prob @ np.abs(pts) ** 2).reshape(shape) - np.abs(x) ** 2
    k = codes.spread(have, m)[::m].reshape(shape).astype(bool) & (np.abs(x) > 1e-3)
    g, n_f, nc = est["gain"], shape[0], shape[2]
    p = est["p_sig"]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(k, raw[:, 1:] / (g * x), 0)
    w = np.where(k, g**2 * np.abs(x) ** 2 / (est.get("n0_k", est["n0"]) + est["clip_ratio"] * g**2 * p + g**2 * p * v), 0)
    kc = r["kc"]
    air = np.arange(n_f) + (0 if kc is None else (np.arange(n_f) >= kc))
    t_rows = (air[:, None] * SYMS_PER_FRAME + np.arange(1, SYMS_PER_FRAME)) * NSYM / modem.FS
    t_pilot = np.arange(len(r["hp"])) * SYMS_PER_FRAME * NSYM / modem.FS
    h, mse = equalizer.refine(r["hp"], t_pilot, z.reshape(-1, nc), w.reshape(-1, nc), t_rows, r["support"], est,
                              ofdm.band(spec.band).bb)
    out = dict(est)
    out["h"] = g * h.reshape(shape)
    out["mse"] = g**2 * mse.reshape(shape)
    return out


def _soft_ofdm(r: dict, est: dict | None = None) -> np.ndarray:
    """(n_cw, coded_bits) soft bits in mapping order from modem.receive's result."""
    spec, est = r["spec"], est or r["est"]
    var = modem.noise_var(est["h"], est) + est["mse"]
    return np.asarray(codes.despread(modem.soft_bits(r["raw"], est["h"], var, spec), r["n_cw"], spec.bits_per_cu))


def _soft_slot(r: dict, est: dict, slot: int) -> np.ndarray:
    """_soft_ofdm(r, est)[slot] alone: codes.spread deals codeword
    symbols round-robin, so the slot's are every n_cw-th of the burst's."""
    n = r["n_cw"]
    var = modem.noise_var(est["h"], est) + est["mse"]
    pick = lambda a: a.reshape(-1)[slot::n]
    return constellation.llr(pick(r["raw"][:, 1:]), pick(est["h"]), pick(var),
                             constellation.load(r["spec"].constellation))


def _decode_post(spec, buf, top: int, rv: int, soft) -> tuple:
    """One LDPC decode as codes.decode_raw (buf None: the slot's soft bits
    alone) or codes.decode_buffer (the combined buffer `buf`, RVs up to
    `top`, this slot sent at `rv`) would -> (info bits as decoded, still
    scrambled unless `buf`; converged; a-posteriori LLRs of the slot's
    coded bits in mapping order): DD needs the posterior of every failed
    decode, so it comes from the same pass."""
    perm = codes.interleaver(spec)
    if buf is None:
        d = np.empty(spec.coded_bits)
        d[perm] = soft
        dec, llr = codes._decoder(spec), d[None]
    else:
        extent = min(codes.buffer_len(spec), (min(top, codes.rv_cycle(spec) - 1) + 1) * spec.coded_bits)
        dec, llr = codes._ext_decoder(spec, extent), buf[:, :extent]
    out, ok, post = dec.decode(llr, iters=40, posterior=True)
    code = post[0]
    return codes._numpy(out)[0], bool(codes._numpy(ok)[0]), (code if buf is None else code[codes.rv_positions(spec, rv)])[perm]


class ModemRx:
    """link.RxBurst over a received burst. `store`: this station's soft
    bits per codeword key, {key: (buffer, highest RV, submode, where)}, kept
    across bursts.

    ponytail: one decode per call (the link asks slot by slot, masks known
    only then); batch the data slots if latency on long polar bursts bites."""

    def __init__(self, r: dict, store: dict, dd_budget: float | None = None):
        """`dd_budget`: seconds from here after which DD starts no more
        refines (None: no limit; studies, so results don't depend on CPU)."""
        self.dd_until = None if dd_budget is None else time.monotonic() + dd_budget
        self.spec, self.n_cw, self.submode = r["spec"], r["n_cw"], r["spec"].name
        # computed once per burst, shared by whoever decodes it (KISS, then ARQ)
        if "_soft" not in r:
            r["_soft"] = soft_bits(r)
        self.soft, self.store = r["_soft"], store
        self._memo = {}  # (slot, mask) -> a one-off decode's result: asked again, free
        # slot -> its one-off decode (codes.decode_raw): the scrambler is
        # unkeyed, so each slot decodes once and every mask asked is a CRC
        # check (a burst's slot 0 is asked under the session's key, KISS's, mask 0)
        self._raw = {}
        self.n_ctl_slots = r.get("n_ctl_slots", 0)  # CPM: slots in the control codeword's spec
        self.r = r
        self._post, self._blind = {}, False  # DD: slot -> LLRs of its coded bits
        # DD: the refined estimate in use (None: r["est"]) and the soft bits
        # made from it, per slot as asked: remaking the whole burst's on each
        # refine cost 0.53 s of a 12 s 256l burst's 0.73 s
        self._est, self._soft_dd = None, {}

    def _soft(self, slot: int):
        if self._est is None:
            return self.soft[slot]
        if slot not in self._soft_dd:
            self._soft_dd[slot] = _soft_slot(self.r, self._est, slot)
        return self._soft_dd[slot]

    def _spec(self, slot: int):
        return ctl_spec(self.spec) if slot < self.n_ctl_slots else self.spec

    def decode(self, slot: int, mask_id: tuple, rv: int, key: tuple | None) -> bytes | None:
        if slot >= self.n_cw:
            return None
        if self.n_ctl_slots and (mask_id[2] >= 128) != (slot < self.n_ctl_slots):
            # CPM: its header says which slots are control (their own short
            # codeword); a blind ARQ_DUP pair probe on a data slot is a miss
            return None
        m = mask_value(mask_id)
        spec = self._spec(slot)
        if key is None:
            if (slot, m) not in self._memo:
                self._memo[(slot, m)] = codes.check(spec, *self._decoded(slot, spec), m)
            return self._memo[(slot, m)]
        buf0, top, name, where = self.store.get(key, (None, 0, self.submode, None))
        if name != self.submode:
            raise AssertionError(f"soft bits of {key} stored in {name} ({where}), resent in {self.submode} "
                                 f"slot {slot} rv {rv}")
        top = max(top, rv)
        soft0 = (self._est, self._soft_dd)
        # the buffer holds unscrambled soft bits: each slot's flipped by its
        # own scrambling, so a resend in any slot combines
        fl = codes.flip(spec, slot, rv)
        for it in range(DD_ITERS + 1):
            buf = codes.combine(spec, None if buf0 is None else buf0.copy(), fl * self._soft(slot), rv)
            if not self._dd(spec):
                payload, ok = codes.decode_buffer(spec, buf, top, m, index=codes.PLAIN)[0]
                if ok:
                    return payload
                break
            bits, conv, post = _decode_post(spec, buf, top, rv, None)
            payload, ok = codes._payloads(spec, bits[None], [conv], [m], [codes.PLAIN])[0]
            if ok:
                self._learn(slot, spec, payload, rv, m)
                return payload
            if it == DD_ITERS or self._late():
                break
            self._refine(slot, spec, fl * post)  # back to the bits as sent
        if self._est is not soft0[0]:
            self._undo(slot, soft0)
            buf = codes.combine(spec, None if buf0 is None else buf0.copy(), fl * self._soft(slot), rv)
        self.store[key] = (buf, top, self.submode, (slot, rv, mask_id))
        return None

    def raw(self, slot: int) -> list[bytes]:
        """`slot`'s mask-free decode as payload bytes, CRC unchecked: polar's
        list best first, LDPC's one candidate if it converged. A broadcast
        control reads its group from these, then checks the CRC under that
        group's key (docs/broadcast.md §2)."""
        if slot >= self.n_cw:
            return []
        spec = self._spec(slot)
        cands, usable = self._decoded(slot, spec)
        nb = codes.payload_bytes(spec)
        return [np.packbits(c[:8 * nb]).tobytes() for c, u in zip(cands, usable) if u]

    def _decoded(self, slot: int, spec) -> tuple:
        """`slot` decoded alone, mask left open (codes.decode_raw's row), with
        DD while it fails to converge. A converged codeword is what was on
        air, whoever it was for: DD learns from it."""
        if slot not in self._raw:
            soft0 = (self._est, self._soft_dd)
            for it in range(DD_ITERS + 1):
                if not self._dd(spec):
                    cands, usable = codes.decode_raw(spec, np.asarray(self._soft(slot))[None], index=slot)
                    cands, usable = cands[0], usable[0]
                    break
                bits, conv, post = _decode_post(spec, None, 0, 0, self._soft(slot))
                cands, usable = codes.descramble(spec, bits, slot)[None], np.array([conv])
                if conv:
                    self._post[slot] = 30.0 * (1 - 2.0 * codes.encode_info(spec, bits[None], 0)[0])
                    break
                if it == DD_ITERS or self._late():
                    break
                self._refine(slot, spec, post)
            if not usable.any():
                self._undo(slot, soft0)
            self._raw[slot] = cands, usable
        return self._raw[slot]

    def _dd(self, spec) -> bool:
        return DD and "hp" in self.r and spec.code == "ldpc" and not self.n_ctl_slots

    def _late(self) -> bool:
        return self.dd_until is not None and time.monotonic() >= self.dd_until

    def _learn(self, slot: int, spec, payload: bytes, rv: int, m: int) -> None:
        if self._dd(spec):
            self._post[slot] = 30.0 * (1 - 2.0 * codes.encode(spec, payload, rv, m, index=slot))

    def _refine(self, slot: int, spec, post: np.ndarray) -> None:
        """DD after a failed decode of `slot`: its decoder's a-posteriori
        LLRs `post` join the other slots' as soft pilots, the channel is
        re-estimated and the soft bits remade."""
        self._post[slot] = post
        if not self._blind:
            # RV 0 slots the link has not asked about yet: parity checks
            # alone say they decoded (a resend at another RV simply fails)
            self._blind = True
            todo = [s for s in range(self.n_cw) if s not in self._post]
            if todo:
                info, ok = codes.decode_llrs(spec, np.asarray(self.soft)[todo])
                for s, i, o in zip(todo, info, ok):
                    if o:
                        self._post[s] = 30.0 * (1 - 2.0 * codes.encode_info(spec, i[None], 0)[0])
        self._est, self._soft_dd = _dd_estimate(self.r, self._post), {}

    def _undo(self, slot: int, soft0) -> None:
        """DD did not rescue `slot`: a failed decode's posterior can be
        confidently wrong (AWGN -1.5 dB: 20% sign errors, the channel's
        own rate, and it moved the estimate by 53%), so neither it nor the
        estimate made from it outlives the attempt; later slots and the
        soft bits stored for combining see the burst as before."""
        self._post.pop(slot, None)
        self._est, self._soft_dd = soft0

    def forget(self, key: tuple) -> None:
        self.store.pop(key, None)
