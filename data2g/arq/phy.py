"""The ARQ layer on the real modem: burst.TxBurst -> audio, a received
burst -> burst.RxBurst (masked CRCs, docs/arq.md §2; soft-bit combining
across resends, §5), and the gear shifter's measurements.

Used by the live stack (engine.py) and by scripts/phy_session.py (whole
sessions through the modem and hfchannel).

Native: arq/phy.hpp.
"""

import logging
import os
import time
import zlib
from functools import lru_cache

import numpy as np

from .. import codes, constellation, cpm, equalizer, modem
from ..config import FM_FLAT_NOCLIP_BITS, FM_TX_PEAK, PROFILE, RS, SNR_REF_BW_HZ
from ..waveform import dsp, ofdm
from . import predictor as P
from .burst import COMPACT_CONNECT
from .frames import MAX_CTL, SEQ_MOD
from .modes import MODES, ctl_spec, is_cpm

log = logging.getLogger(__name__)

# DATA2G_DD (default 1; 0 turns it off): decision-directed re-estimation. When a slot
# fails, its decoder's a-posteriori LLRs and every decoded codeword of the
# burst become soft pilots (equalizer.refine), and it is decoded again, up
# to DD_ITERS times.
DD = os.environ.get("DATA2G_DD", "1") == "1"
DD_ITERS = 2
# CPM data slots (GF(M)): DD makes the metric coherent (cpm.dd_metric: a
# channel estimate per row from the decoder's soft decisions), up to
# CPM_DD_ITERS re-decodes. DATA2G_CPM_DD=0 turns it off alone.
CPM_DD = os.environ.get("DATA2G_CPM_DD", "1") == "1"
CPM_DD_ITERS = 4
# a live receiver's DD stops starting refines this long after the burst's
# decode began (a 12 s 256l burst with every slot failing took 54 s of DD).
# In a session the engine also stops it at the turnaround deadline: my reply
# keyed within my t_turn of the peer's burst's end (engine._dd_budget), so
# DD gets what the plain decode leaves, which may be nothing. inf: no limit
# at all (paired studies, tools/with_native.py --dd-budget inf)
DD_BUDGET_S = 1.0
# A control is a few bytes (median 12, at most 28 in linksim sessions) zeroed
# to its codeword's payload (frames.Control.pack). A control slot that fails
# is decoded again with the control stream past each of these byte counts in
# turn known to be zero (codes.decode_raw_pinned): at code level, 10% of LDPC
# controls failed 1-6 dB lower, 4 dB the median over the modes
# (scripts/ctl_prior_study.py). The same for an ARQ_DUP pair once combined.
# Polar: CPM's control carries 9-16 B of its 20 (median 12) in real-modem
# sessions, so shorter cuts.
# LDPC: since T_INLINE (docs/arq.md §5a) a control's content can end
# anywhere up to 269 B, past any fixed cut. Instead, the slot's own soft
# bits say where its padding likely starts, and one decode leans each
# payload bit toward zero by how likely it is padding (codes.pad_prior):
# within 0.0 dB median, 0.6 at most, of pinning at the exact length, at
# every length, one decode (scripts/ctl_prior_study.py --boundary). DATA2G_PIN_EST=0:
# the fixed cuts, for A/B runs.
PIN_CUTS = (16, 32)
POLAR_PIN_CUTS = (12, 16)
PIN_EST = os.environ.get("DATA2G_PIN_EST", "1") == "1"
EST = 0  # an offset no cut gives: the estimated prior (codes.pad_prior) in place of a cut


# A primed control (docs/arq.md §3a) must also score this many nats over
# random bits (codes.score): noise passes with probability under e^-10 on
# top of the CRC.
PRIMED_SCORE = 10.0
# A primed decode CRC-checks only the first PRIMED_PATHS paths of the polar
# list: PRIME_TRIES (6) x 4 = 24 checks, as many as the blind decode's (plain
# and two cuts, 8 paths each), so a failed control's false-accept chance stays
# 24 x 2^-24. Its cost at code level: 0.03-0.24 dB (all 8: scripts/primed_study.py).
PRIMED_PATHS = 4


def _with_prior(spec, buf: np.ndarray) -> np.ndarray:
    """A control's combined buffer (1, L) leaning toward its estimated padding (codes.pad_prior)."""
    return codes.add_prior_buffer(spec, buf, codes.pad_prior(spec, codes.info_llrs_buffer(spec, buf[0])))


def _ctl_index(mask_id: tuple) -> int:
    """A control mask's codeword index in the control stream (a compact
    CONNECT's, burst.COMPACT_CONNECT, is its only codeword)."""
    i = mask_id[2] - SEQ_MOD
    return i if i < MAX_CTL else 0


def mask_value(mask_id: tuple) -> int:
    """A slot's CRC mask from its identity (burst.ctl_mask, data_mask).
    Key 0 (connect frames, before a session) is mask 0, except a compact
    CONNECT's (burst.COMPACT_CONNECT)."""
    key, direction, s = mask_id
    if key == 0:
        return 0xC0C0C0 if s == COMPACT_CONNECT[2] else 0
    return zlib.crc32(bytes([key >> 8, key & 255, direction, s])) or 1


def peak_db(submode: str, cap: int = 0) -> float:
    """A mode's burst envelope peak over its average power (dB), measured once
    on a fixed burst (a control and a data codeword, seeded payloads): the
    energy inputs report a burst's SNR against its peak (a peak-limited
    transmitter), with this per mode and bandwidth cap (a CPM grid's TX
    filter is the cap's: cpm.TX_FILTERS; OFDM's is not). Rounded to 0.001
    dB: the FFT differs by an ulp across machines, and gen_native_tables
    must regenerate the same bytes everywhere."""
    return _peak_db(submode, cap if is_cpm(MODES[submode]) else 0)


@lru_cache(maxsize=None)
def _peak_db(submode: str, cap: int) -> float:
    from .burst import Slot, TxBurst, ctl_mask, data_mask

    spec = MODES[submode]
    rng = np.random.default_rng(0)
    rand = lambda sp: bytes(rng.integers(0, 256, codes.payload_bytes(sp), dtype=np.uint8))  # noqa: E731
    burst = TxBurst(submode, [Slot(ctl_mask(0, 0, 5), 0, rand(ctl_spec(spec))), Slot(data_mask(0, 1, 5), 0, rand(spec))], 0, cap)
    x = tx_audio(burst)
    return round(float(10 * np.log10(np.max(dsp.envelope(x) ** 2) / 2 / dsp.active_power(x))), 3)


def full_scale(x: np.ndarray) -> np.ndarray:
    """A burst's unit-RMS audio for the sound card. HF: its envelope peak at
    full scale (PEP limited; arq.engine._start_tx's reasons). FM: one fixed
    scale for every burst, so modes keep their levels (config.FM_TX_PEAK)."""
    if PROFILE == "fm":
        return np.clip(x / FM_TX_PEAK, -1.0, 1.0)
    return x / np.max(dsp.envelope(x))


def tone_scale(x: np.ndarray) -> np.ndarray:
    """A constant-envelope signal built at peak 1.0 (a CW ID, a Hand) for the
    sound card. HF: as built, the bursts' peak. FM: a burst's RMS
    (full_scale), 11 dB under that: the level a ping set the radio's
    deviation by."""
    return x * (2 ** 0.5 / FM_TX_PEAK) if PROFILE == "fm" else x


def tx_audio(burst, tx_path: str | None = None, level: float | None = None) -> np.ndarray:
    """A link.TxBurst -> unit-RMS audio (a CPM burst filtered for its cap:
    cpm.TX_FILTERS), times burst.level (an FM ping: its top modes' level,
    arq.ping) or `level`, then the sounding probe if burst.probe (arq.probe). Codewords scramble by burst position alone
    (codes.scramble_seed: no key on air); a resend in another slot still
    combines (codes.flip). A nod (burst.nod): its keyed sequence. `tx_path`:
    FM's (modem.modulate's; default modem.TX_PATH)."""
    if getattr(burst, "nod", None) is not None:
        from . import nod

        meaning, fam = burst.nod
        return nod.waveform(nod.family(*fam)[meaning], burst.cap)
    spec = MODES[burst.submode]
    if is_cpm(spec):
        # control slots (masks >= 128) in the grid's short codeword; two of
        # them are one codeword at RV 0 and 1 (ARQ_DUP)
        n_ctl = sum(1 for s in burst.slots if s.mask_id[2] >= 128)
        coded = [codes.encode(ctl_spec(spec) if i < n_ctl else spec, s.payload, s.rv, mask_value(s.mask_id), i)
                 for i, s in enumerate(burst.slots)]
        return cpm.modulate(spec, coded, dup=n_ctl == 2, cap=burst.cap)
    bits = np.stack([codes.encode(spec, s.payload, s.rv, mask_value(s.mask_id), i) for i, s in enumerate(burst.slots)])
    x = modem.modulate_bits(codes.spread(bits, spec.bits_per_cu, codes.spread_group(spec)), spec, tx_path)
    g = getattr(burst, "level", 1.0) if level is None else level
    x = x if g == 1.0 else x * g
    if getattr(burst, "probe", False):
        from . import probe

        x = np.concatenate([x, probe.waveform(tx_path, g)])
    return x


def measure(r: dict) -> dict:
    """The gear shifter's inputs (arq/predictor's features) from a received
    burst (modem.receive's Received, or cpm.receive's dict, family "cpm"):
    the modem's measurements plus effective MI per constellation."""
    if r.get("family") == "cpm":
        out = cpm.measure(cpm.GRIDS[r["band"]], r["E"], len(r["E"]))
        snr = out.pop("snr")
        for c in P.CONSTS:
            out[f"mi_{c}"] = P.effective_mi(np.sqrt(snr), np.ones_like(snr), c)
        return out
    return _measure_ofdm(r)


def _measure_ofdm(r: modem.Received) -> dict:
    """The gear shifter's inputs from modem.receive's result: effective MI
    against thermal noise and estimation error, not the transmitter's clip
    noise (it belongs to the submode sent, not to the channel)."""
    est = r["est"]
    n0 = est.get("n0_k", est["n0"]) if PROFILE == "fm" and P.DATA_MER else est["n0"]
    h, var = est["h"], n0 + est["mse"]
    d0, d1 = r.get("seen_support", r["support"])
    out = dict(snr_est=modem.snr_db(est, r["band"]),
               spread_est=est["spread_hz"], delay_est_ms=(d1 - d0) / modem.FS * 1000,
               headroom=r["spec"].headroom, frames=r["n_cw"] * r["spec"].frames_per_cw,
               n0_spread_db=n0_spread_db(est))
    for c in P.CONSTS:
        out[f"mi_{c}"] = P.effective_mi(h, var, c)
    return out


# Post-decode data-aided MER (issue #290): repeating low-PAPR pilots absorb TX
# audio-path nonlinearity into h, while higher-PAPR data symbols take it as
# noise. Exact re-encoded symbols for decoded slots + strided top-K Gaussian-
# mixture EM for undecoded slots measure the excess data-cell error variance
# `delta * |h|^2` above `n0_k + mse + tx_clip * |h|^2`.
MER_MAX_EM_CELLS = 1024
MER_EM_TOP_K = 12
MER_EM_ITERS = 6
MER_DEADBAND_RATIO = 1.35


def data_mer(r: modem.Received, coded_bits: dict[int, np.ndarray], measured: dict) -> dict:
    """Refine `measured` with data-cell MER from `r` and decoded slots'
    `coded_bits` ({slot: coded bits in mapping order}). Always attaches
    `mer_db` (2500 Hz convention); lowers `snr_est` and recomputes `mi_*` when
    data-cell variance exceeds the pilot-predicted baseline past the deadband."""
    spec, est = r["spec"], r["est"]
    y, h = modem.data_symbols(r["raw"]), est["h"]
    N = y.size
    if N == 0:
        return dict(measured)
    pts = constellation.load(spec.constellation)
    yf, hf = y.reshape(-1), h.reshape(-1)
    h2 = np.abs(hf) ** 2
    sig_data = float(np.mean(h2))
    nc = h.shape[2]
    bw = 10.0 * np.log10(nc * RS / SNR_REF_BW_HZ)

    n0_k = np.broadcast_to(np.asarray(est.get("n0_k", est["n0"]), dtype=float), h.shape).reshape(-1)
    mse = np.asarray(est["mse"], dtype=float).reshape(-1)
    # FM f108 >= 16-QAM is a flat-only band transmitted unclipped (headroom 40);
    # subtracting spec.clip.ratio (-24.6 dB) would hide real distortion there.
    fm_noclip = PROFILE == "fm" and spec.band == "f108" and spec.bits_per_cu >= FM_FLAT_NOCLIP_BITS
    clip_r = 0.0 if fm_noclip else float(est["clip_ratio"])
    base = n0_k + mse + clip_r * h2
    base_mean = float(np.mean(base))

    ex_m = np.zeros(N, dtype=bool)
    ex_err = 0.0
    for s, cbits in coded_bits.items():
        if not 0 <= s < r["n_cw"]:
            continue
        cells = modem.slot_cells(r, s)
        if cells.size == 0 or int(cells.max()) >= N:
            continue
        xs = constellation.modulate(cbits, pts)
        fresh = ~ex_m[cells]
        if np.any(fresh):
            cf = cells[fresh]
            ex_err += float(np.sum(np.abs(yf[cf] - hf[cf] * xs[fresh]) ** 2))
            ex_m[cf] = True
    n_ex = int(ex_m.sum())
    n_u = N - n_ex

    if n_u == 0:
        v_data = ex_err / N
    else:
        u_idx = np.flatnonzero(~ex_m)
        step = max(1, (n_u + MER_MAX_EM_CELLS - 1) // MER_MAX_EM_CELLS)
        sel = u_idx[::step]
        yu, hu, bu, h2u = yf[sel], hf[sel], base[sel], h2[sel]
        good_h = h2u > 1e-30
        with np.errstate(divide="ignore", invalid="ignore"):
            zu = np.where(good_h, yu / hu, 0.0)
        d2 = h2u[:, None] * ((zu.real[:, None] - pts.real[None, :]) ** 2
                             + (zu.imag[:, None] - pts.imag[None, :]) ** 2)
        if not np.all(good_h):
            d2[~good_h] = (np.abs(yu[~good_h]) ** 2)[:, None]
        if d2.shape[1] > MER_EM_TOP_K:
            part = np.argpartition(d2, MER_EM_TOP_K, axis=1)[:, :MER_EM_TOP_K]
            d2k = np.take_along_axis(d2, part, axis=1)
        else:
            d2k = d2
        v_u_hard = float(np.mean(d2k.min(axis=1)))
        v_data = max((ex_err + v_u_hard * n_u) / N * 2.0, base_mean, sig_data * 1e-4)
        delta_em = max(0.0, (v_data - base_mean) / max(sig_data, 1e-30))
        for _ in range(MER_EM_ITERS):
            vc = np.maximum(bu + delta_em * h2u, 1e-30)
            lw = -d2k / vc[:, None]
            lw -= lw.max(axis=1, keepdims=True)
            w = np.exp(lw)
            w /= w.sum(axis=1, keepdims=True)
            v_u = float(np.mean(np.sum(w * d2k, axis=1)))
            v_data = (ex_err + v_u * n_u) / N
            delta_em = max(0.0, (v_data - base_mean) / max(sig_data, 1e-30))

    mer_db = float(10.0 * np.log10(max(sig_data, 1e-30) / max(v_data, 1e-30)) + bw)
    db_fac = max(MER_DEADBAND_RATIO, 1.0 + 3.0 / np.sqrt(N))
    excess = (v_data - base_mean) if v_data > db_fac * base_mean else 0.0
    delta = excess / max(sig_data, 1e-30)

    out = dict(measured, mer_db=mer_db)
    if delta > 0.0:
        p_sig = float(est["p_sig"])
        n0_snr = float(est.get("n0_snr", est["n0"]))
        snr_eff = float(10.0 * np.log10(1.0 / (n0_snr / max(p_sig, 1e-30) + delta)) + bw)
        off = P.FM_MER_OFFSET_DB if PROFILE == "fm" else 0.0
        out["snr_est"] = min(float(measured["snr_est"]), snr_eff + off)
        n0_mi = est.get("n0_k", est["n0"]) if PROFILE == "fm" else est["n0"]
        var_d = n0_mi + est["mse"] + delta * (np.abs(h) ** 2)
        for c in P.CONSTS:
            out[f"mi_{c}"] = P.effective_mi(h, var_d, c)
    return out


def n0_spread_db(est) -> float:
    """Standard deviation over frames of each frame's noise level, dB, from
    the per-window estimate (config.EST_WINDOW; 0 without one: HF). FM's
    regime input: a fading or shadowed path moves the noise between windows
    (median over the study's bursts: static 0.35 dB, flutter 0.27, slow fading
    1.04, mobile 1.77; scripts/fm_regime_study.py), where HF's Doppler spread
    reads about zero on FM audio."""
    n0 = np.asarray(est.get("n0_k", est["n0"]), float)
    if n0.ndim != 3 or len(n0) < 2:
        return 0.0
    return float(np.std(10 * np.log10(np.mean(n0, axis=(1, 2)))))


def soft_bits(r: dict):
    """Per slot soft bits in mapping order ((n_cw, coded_bits) for OFDM; a
    list, control and data lengths differing, for CPM)."""
    if r.get("family") == "cpm":
        return r["soft"]
    return modem.burst_soft(r)


def _dd_estimate(r: modem.Received, post: dict) -> modem.DataEstimate:
    """r["est"] re-made by equalizer.refine with soft symbols as pilots:
    `post` {slot: a-posteriori LLRs of its coded bits, mapping order}
    (a decoded codeword's are its bits, as large LLRs)."""
    spec, est, y = r["spec"], r["est"], modem.data_symbols(r["raw"])
    pts = constellation.load(spec.constellation)
    m = constellation.bits_per_symbol(pts)
    llr = np.zeros((r["n_cw"], spec.coded_bits))
    have = np.zeros(llr.shape, dtype=np.uint8)
    for s, v in post.items():
        llr[s], have[s] = v, 1
    shape = y.shape
    n_in = y.size  # burst order is time order: a partial burst's symbols are the first so many
    L = np.clip(codes.spread(llr, m, codes.spread_group(spec)).reshape(-1, m)[:n_in], -30, 30)
    labels = (np.arange(len(pts))[:, None] >> np.arange(m - 1, -1, -1)) & 1  # (M, m), modulate's order
    # log P(label) as matmuls: the (symbols, M, m) select-and-sum was 70% of a 256l refine
    lp = -np.logaddexp(0, L) @ labels.T - np.logaddexp(0, -L) @ (1 - labels).T
    prob = np.exp(lp - lp.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    x = (prob @ pts).reshape(shape)
    v = (prob @ np.abs(pts) ** 2).reshape(shape) - np.abs(x) ** 2
    k = codes.spread(have, m, codes.spread_group(spec))[::m][:n_in].reshape(shape).astype(bool) & (np.abs(x) > 1e-3)
    g, nc = est["gain"], shape[2]
    p = est["p_sig"]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(k, y / (g * x), 0)
    # modem.noise_var's with |h|^2 at its expectation g^2 p, plus the soft symbol's own variance
    w = np.where(k, g**2 * np.abs(x) ** 2 / (est.get("n0_k", est["n0"]) + est["clip_ratio"] * g**2 * p + g**2 * p * v), 0)
    t_rows, t_pilot = modem.symbol_times(r)
    h, mse = equalizer.refine(r["hp"], t_pilot, z.reshape(-1, nc), w.reshape(-1, nc), t_rows, r["support"], est,
                              ofdm.band(spec.band).bb)
    out = dict(est)
    out["h"] = g * h.reshape(shape)
    out["mse"] = g**2 * mse.reshape(shape)
    return out


def cpm_pass(r: dict, soft: list, post: dict, slot: int, late=lambda: False) -> tuple:
    """A CPM data slot's mask-free decode, as a streaming receiver makes it
    once the slot and cpm.COH_W rows more are in: the square law first, then
    while it fails up to CPM_DD_ITERS re-decodes on cpm.dd_metric, from the
    posteriors `post` of earlier slots that decoded and its own last one.
    The same whenever it runs, so ModemRx makes these in slot order and a
    receiver may make them as the burst arrives. -> (cands, usable, the
    decoded codeword's certain posterior or None); `late`: DD_BUDGET_S's."""
    spec = r["spec"]
    known = {s: v for s, v in post.items() if s < slot and v is not None}
    s_ = soft[slot]
    for it in range(CPM_DD_ITERS + 1):
        bits, conv, p = codes.decode_posterior(spec, s_)
        if conv or it == CPM_DD_ITERS or late():
            break
        s_ = cpm.dd_soft(r, cpm.dd_metric(r, {**known, slot: p}, slot), slot)
    cands, usable = codes.descramble(spec, bits, slot)[None], np.array([conv])
    return cands, usable, _sure(spec, codes.encode_info(spec, bits[None], 0)[0]) if conv else None


def _sure(spec, coded: np.ndarray) -> np.ndarray:
    """A decoded codeword's coded bits (mapping order, as sent) as DD's
    certain posterior: LLRs +-30, or GF(M) log-probabilities 0 / -30 per symbol."""
    if spec.code != "gf":
        return 30.0 * (1 - 2.0 * coded)
    p = spec.bits_per_cu
    v = np.asarray(coded).reshape(-1, p) @ (1 << np.arange(p)[::-1])
    out = np.full((len(v), 1 << p), -30.0)
    out[np.arange(len(v)), v] = 0.0
    return out.reshape(-1)


class _SlotSoft:
    """A partial burst's soft bits, a ready slot's at a time (modem.slot_soft
    on r's estimate), made when first asked."""

    def __init__(self, r: dict):
        self.r, self._made = r, {}

    def __getitem__(self, slot: int) -> np.ndarray:
        if slot not in self._made:
            self._made[slot] = modem.slot_soft(self.r, self.r["est"], slot)
        return self._made[slot]


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
        self.store = store
        self._adopt(r)
        self._memo = {}  # (slot, mask) -> a one-off decode's result: asked again, free
        # slot -> its one-off decode (codes.decode_raw): the scrambler is
        # unkeyed, so each slot decodes once and every mask asked is a CRC
        # check (a burst's slot 0 is asked under the session's key, KISS's, mask 0)
        self._raw = {}
        self._pin = {}  # (slot, cut) -> its decode with the payload past cut bytes pinned to zero
        self._pairs = {}  # (slot, cut or None) -> slots slot, slot + 1 combined (RV 0, 1), decoded mask open
        self.n_ctl_slots = r.get("n_ctl_slots", 0)  # CPM: slots in the control codeword's spec
        self.r = r
        self._post, self._blind = {}, False  # DD: slot -> LLRs of its coded bits
        self._pass_post = {}  # CPM: data slot -> its cpm_pass's posterior (None: it failed)
        # DD: the refined estimate in use (None: r["est"]) and the soft bits
        # made from it, per slot as asked: remaking the whole burst's on each
        # refine cost 0.53 s of a 12 s 256l burst's 0.73 s
        self._est, self._soft_dd = None, {}
        self.dd_total = dd_budget  # advance(): the whole burst's DD budget, shared out per window
        self._dd_spent = 0.0
        self._early = set()  # slots advance() decoded: a keyed read at RV 0 checks that decode first
        self._coded = {}  # OFDM: slot -> decoded coded bits (mapping order), for data_mer
        self._mer_cache = None

    def _adopt(self, r: dict) -> None:
        """r's soft bits: the whole burst's at once, or a partial burst's per
        ready slot as asked (modem.receive(partial=True))."""
        self.r = r
        n_f = 0 if r.get("family") == "cpm" else r["n_cw"] * r["spec"].frames_per_cw
        partial = r.get("frames_in", n_f) < n_f
        if partial:
            self.soft = _SlotSoft(r)
        else:
            if "_soft" not in r:
                r["_soft"] = soft_bits(r)
            self.soft = r["_soft"]

    def advance(self, r: dict, budget_s: float | None = None) -> list[int]:
        """Decoding while the burst arrives: adopt `r`, the same burst with
        more of it in (modem.receive(partial=True), or the whole), keeping
        every decode that was usable and the sure posteriors DD learned from
        them, and decode the slots newly ready (modem.ready_slots), DD
        refining at most `budget_s` seconds from now (None: the rest of the
        burst's budget). A failed slot is tried again on the next advance,
        with more of the burst's pilots. -> the slots decoded this time."""
        self._adopt(r)
        # the slots decoded already need no soft bits: the rest's as asked, a
        # whole burst's included (its burst-wide soft_bits was ~95 ms of a Pi's
        # end-of-burst on a 256l x39 burst, docs/pi-bench-fm.md)
        self.soft = _SlotSoft(r)
        self._memo, self._pin, self._pairs = {}, {}, {}
        self._raw = {s: v for s, v in self._raw.items() if v[1].any()}
        self._est, self._soft_dd = None, {}
        if self.dd_total is not None:
            left = max(0.0, self.dd_total - self._dd_spent)
            budget_s = left if budget_s is None else min(budget_s, left)
        t0 = time.monotonic()
        self.dd_until = None if budget_s is None else t0 + budget_s
        # a slot that failed mid-burst waits for the whole burst: its window's estimate
        # barely changes as later windows arrive, and retrying on each one was 110-128
        # LDPC tries for 39 slots of a failing 256l burst on a Pi (runs/pi-fmstream)
        whole = r.get("frames_in", -1) < 0 or r["frames_in"] >= r["n_cw"] * r["spec"].frames_per_cw
        new = [s for s in modem.ready_slots(r) if s not in self._raw and (whole or s not in self._early)]
        for s in new:
            self._decoded(s, self._spec(s))
        self._early.update(new)
        self._dd_spent += time.monotonic() - t0
        return new

    def _soft(self, slot: int):
        if self._est is None:
            return self.soft[slot]
        if slot not in self._soft_dd:
            self._soft_dd[slot] = (cpm.dd_soft(self.r, cpm.dd_metric(self.r, self._est, slot), slot) if self.n_ctl_slots
                                   else modem.slot_soft(self.r, self._est, slot))
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
                got = codes.check(spec, *self._decoded(slot, spec), m)
                if got is not None:
                    self._learn(slot, spec, got, 0, m)
                elif mask_id[2] >= SEQ_MOD:  # a control (burst.ctl_mask)
                    got = self._check_pinned(slot, spec, m, _ctl_index(mask_id))
                self._memo[(slot, m)] = got
            return self._memo[(slot, m)]
        if rv == 0 and slot in self._early and slot in self._raw:
            # decoded while the burst arrived (advance): its CRC under this mask, before decoding it again
            got = codes.check(spec, *self._raw[slot], m)
            if got is not None:
                self._learn(slot, spec, got, rv, m)
                return got
        buf0, top, name, where = self.store.get(key, (None, 0, self.submode, None))
        if name != self.submode:  # a resend in another mode: what was kept can't combine, so start over
            log.warning("soft bits of %s stored in %s (%s), resent in %s slot %d rv %d: dropped", key, name, where,
                        self.submode, slot, rv)
            buf0, top = None, 0
        top = max(top, rv)
        soft0 = (self._est, self._soft_dd)
        # the buffer holds unscrambled soft bits: each slot's flipped by its
        # own scrambling, so a resend in any slot combines
        for it in range(self._iters() + 1):
            plain = codes.plain(spec, self._soft(slot), slot, rv)
            buf = codes.combine(spec, None if buf0 is None else buf0.copy(), plain, rv)
            if not self._dd(spec):
                payload, ok = codes.decode_buffer(spec, buf, top, m, index=codes.PLAIN)[0]
                if ok:
                    self._learn(slot, spec, payload, rv, m)
                    return payload
                break
            bits, conv, post = codes.decode_posterior(spec, buf=buf, max_rv=top, rv=rv)
            payload, ok = codes.payloads(spec, bits[None], [conv], [m], [codes.PLAIN])[0]
            if ok:
                self._learn(slot, spec, payload, rv, m)
                return payload
            if it == self._iters() or self._late():
                break
            self._refine(slot, spec, codes.plain(spec, post, slot, rv))  # back to the codeword as sent
        if self._est is not soft0[0]:
            self._undo(slot, soft0)
            plain = codes.plain(spec, self._soft(slot), slot, rv)
            buf = codes.combine(spec, None if buf0 is None else buf0.copy(), plain, rv)
        if buf0 is not None and mask_id[2] >= SEQ_MOD:  # a control combined (ARQ_DUP): its padding pinned
            for off in self._offsets(spec, _ctl_index(mask_id)):
                if off == EST:
                    payload, ok = codes.decode_buffer(spec, _with_prior(spec, buf), top, m, index=codes.PLAIN)[0]
                else:
                    payload, ok = codes.decode_buffer_pinned(spec, buf, top, codes.zero_tail(spec, off), m)[0]
                if ok:
                    self._learn(slot, spec, payload, rv, m)
                    return payload
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
        out = [np.packbits(c[:8 * nb]).tobytes() for c, u in zip(cands, usable) if u]
        # pinned candidates after the plain ones: LDPC's when it did not
        # converge; polar's always, as its list is never empty. A broadcast
        # control is one codeword per slot (CPM: one, twice at most).
        for off in () if out and spec.code != "polar" else self._offsets(spec, 0 if self.n_ctl_slots else slot):
            c, u = self._pinned(slot, spec, off)
            out += [np.packbits(b[:8 * nb]).tobytes() for b, ok in zip(c, u) if ok]
        return list(dict.fromkeys(out))

    def converged(self, slot: int, pair: bool = False) -> bool:
        """`slot` decoded, mask open and nothing pinned, to a codeword (LDPC:
        every parity check satisfied): alone, or with `pair` combined with
        slot + 1 as raw_pair does. Polar always gives a list, so never says:
        False."""
        if slot >= self.n_cw or (pair and not self._pair_ok(slot)):
            return False
        spec = self._spec(slot)
        if spec.code == "polar":
            return False
        return bool((self._pair(slot, None) if pair else self._decoded(slot, spec))[1].any())

    def raw_pair(self, slot: int) -> list[bytes]:
        """raw() of a codeword sent twice: `slot` at RV 0 and `slot + 1` at
        RV 1, combined (a broadcast control under DUP, docs/broadcast.md
        §3). [] if the two aren't both there, or (CPM) not both control."""
        if not self._pair_ok(slot):
            return []
        spec = self._spec(slot)
        nb = codes.payload_bytes(spec)
        cands, usable = self._pair(slot, None)
        out = [np.packbits(c[:8 * nb]).tobytes() for c, u in zip(cands, usable) if u]
        for off in () if out and spec.code != "polar" else self._offsets(spec, slot // 2):
            c, u = self._pair(slot, off)
            out += [np.packbits(b[:8 * nb]).tobytes() for b, ok in zip(c, u) if ok]
        return list(dict.fromkeys(out))

    def decode_pair(self, slot: int, mask_id: tuple) -> bytes | None:
        """The pair raw_pair reads, CRC-checked under `mask_id` (a control's:
        its padding pinned if it fails plain), or None."""
        if not self._pair_ok(slot):
            return None
        spec, m = self._spec(slot), mask_value(mask_id)
        for off in (None, *self._offsets(spec, _ctl_index(mask_id))):
            got = codes.check(spec, *self._pair(slot, off), m)
            if got is not None:
                self._learn(slot, spec, got, 0, m)
                self._learn(slot + 1, spec, got, 1, m)
                return got
        return None

    def decode_known(self, slot: int, mask_id: tuple, template: bytes, tmask: bytes, pair: bool = False) -> bytes | None:
        """A primed retry (docs/arq.md §3a): control slot `slot` (with
        `pair`: slots slot, slot + 1 at RV 0 and 1, combined) decoded again
        with the payload bits `tmask` sets known to be `template`'s
        (codes.known_bits) frozen. Polar codewords only (CPM's control, the
        polar and ACK modes): an LDPC control gained 0-0.3 dB at code level
        over its padding estimate, under a 16-bit CRC. The payload if it
        passes its masked CRC and its codeword scores at least PRIMED_SCORE
        nats over random bits, else None."""
        if slot >= self.n_cw or (pair and not self._pair_ok(slot)):
            return None
        spec, m = self._spec(slot), mask_value(mask_id)
        if spec.code != "polar" or (self.n_ctl_slots and slot >= self.n_ctl_slots):
            return None
        known = codes.known_bits(spec, template, tmask)
        if pair:
            buf = codes.combine(spec, None, codes.plain(spec, self._soft(slot), slot, 0), 0)
            buf = codes.combine(spec, buf, codes.plain(spec, self._soft(slot + 1), slot + 1, 1), 1)
            cands, usable = codes.decode_buffer_raw(spec, buf, 1, known)
        else:
            cands, usable = codes.decode_raw_pinned(spec, np.asarray(self._soft(slot))[None], known, slot)
        usable = np.asarray(usable[0]).copy()
        usable[PRIMED_PATHS:] = False
        got = codes.check(spec, cands[0], usable, m)
        if got is None:
            return None
        # a pair's RV 1 is the same codeword: one more reception, at its own slot's scrambling
        sc = codes.score(spec, self._soft(slot), got, m, slot)
        if pair:
            sc += codes.score(spec, self._soft(slot + 1), got, m, slot + 1)
        if sc < PRIMED_SCORE:
            log.info("primed control in slot %d: CRC passed, score %.1f under %.0f nats, dropped", slot, sc, PRIMED_SCORE)
            return None
        self._learn(slot, spec, got, 0, m)
        if pair:
            self._learn(slot + 1, spec, got, 1, m)
        return got

    def decode_repeat(self, slot: int, mask_id: tuple, key: tuple) -> bytes | None:
        """Control slot `slot`, combined with the same slot of the bursts
        before it kept under `key` (their decodes failed; a failure here
        keeps this one too): bit-identical repeats (a CONNECT's tries, a
        repeat after a timeout, robust polls; docs/arq.md §3a). The payload
        must also score PRIMED_SCORE nats on this burst's own soft bits: the
        sum may decode to an earlier, different burst's codeword. What was
        kept in another mode is dropped: not this burst's repeat."""
        if key in self.store and self.store[key][2] != self.submode:
            del self.store[key]
        had = slot in self._coded
        got = self.decode(slot, mask_id, 0, key)
        if got is None:
            return None
        sc = codes.score(self._spec(slot), self._soft(slot), got, mask_value(mask_id), slot)
        if sc < PRIMED_SCORE:
            if not had:
                self._coded.pop(slot, None)
            log.info("repeated control in slot %d: CRC passed, score %.1f under %.0f nats, dropped", slot, sc, PRIMED_SCORE)
            return None
        return got

    def _pair_ok(self, slot: int) -> bool:
        if slot + 1 >= self.n_cw:
            return False
        return not self.n_ctl_slots or slot + 1 < self.n_ctl_slots

    def _pair(self, slot: int, off: int | None) -> tuple:
        """Slots `slot` (RV 0) and `slot + 1` (RV 1) combined and decoded
        once, mask open (codes.decode_buffer_raw's row); `off`: the payload
        past it known zero. No DD: a pair is a control, one short decode."""
        if (slot, off) not in self._pairs:
            spec = self._spec(slot)
            buf = codes.combine(spec, None, codes.plain(spec, self._soft(slot), slot, 0), 0)
            buf = codes.combine(spec, buf, codes.plain(spec, self._soft(slot + 1), slot + 1, 1), 1)
            if off == EST:
                cands, usable = codes.decode_buffer_raw(spec, _with_prior(spec, buf), 1)
            else:
                cands, usable = codes.decode_buffer_raw(spec, buf, 1, None if off is None else codes.zero_tail(spec, off))
            self._pairs[(slot, off)] = cands[0], usable[0]
        return self._pairs[(slot, off)]

    def _offsets(self, spec, i: int) -> tuple:
        """Control codeword i's payload offsets past which a cut (PIN_CUTS,
        POLAR_PIN_CUTS: bytes of the control stream) knows it zero; LDPC
        (PIN_EST): EST alone."""
        if spec.code == "gf":  # nothing to pin (codes.decode_raw_pinned)
            return ()
        if spec.code == "ldpc" and PIN_EST:
            return (EST,)
        pb = codes.payload_bytes(spec)
        cuts = PIN_CUTS if spec.code == "ldpc" else POLAR_PIN_CUTS
        return tuple(dict.fromkeys(c - i * pb for c in cuts if 0 < c - i * pb < pb))

    def _pinned(self, slot: int, spec, off: int) -> tuple:
        """`slot` decoded alone with its payload past `off` bytes known zero (EST:
        its estimated padding prior instead), as decode_raw's row."""
        if (slot, off) not in self._pin:
            soft = np.asarray(self._soft(slot))
            if off == EST:
                prior = codes.pad_prior(spec, codes.info_llrs(spec, soft, slot))
                cands, usable = codes.decode_raw(spec, codes.add_prior(spec, soft, prior, slot)[None], index=[slot])
            else:
                cands, usable = codes.decode_raw_pinned(spec, soft[None], codes.zero_tail(spec, off), slot)
            self._pin[(slot, off)] = cands[0], usable[0]
        return self._pin[(slot, off)]

    def _check_pinned(self, slot: int, spec, m: int, i: int) -> bytes | None:
        """Control codeword i that failed alone: decoded again with its
        padding known, the shortest cut first. Each try is CRC-checked under
        `m` like any decode."""
        for off in self._offsets(spec, i):
            got = codes.check(spec, *self._pinned(slot, spec, off), m)
            if got is not None:
                self._learn(slot, spec, got, 0, m)
                return got
        return None

    def _decoded(self, slot: int, spec) -> tuple:
        """`slot` decoded alone, mask left open (codes.decode_raw's row), with
        DD while it fails to converge. A converged codeword is what was on
        air, whoever it was for: DD learns from it."""
        if slot not in self._raw and self.n_ctl_slots and self._dd(spec):
            self._cpm_passes(slot)
        if slot not in self._raw:
            soft0 = (self._est, self._soft_dd)
            for it in range(self._iters() + 1):
                if not self._dd(spec):
                    cands, usable = codes.decode_raw(spec, np.asarray(self._soft(slot))[None], index=slot)
                    cands, usable = cands[0], usable[0]
                    break
                bits, conv, post = codes.decode_posterior(spec, self._soft(slot))
                cands, usable = codes.descramble(spec, bits, slot)[None], np.array([conv])
                if conv:
                    coded = codes.encode_info(spec, bits[None], 0)[0]
                    self._post[slot] = _sure(spec, coded)
                    if not self.n_ctl_slots and self.r.get("family") != "cpm":
                        self._coded[slot] = coded
                    break
                if it == self._iters() or self._late():
                    break
                self._refine(slot, spec, post)
            if not usable.any():
                self._undo(slot, soft0)
            self._raw[slot] = cands, usable
        return self._raw[slot]

    def _cpm_passes(self, upto: int) -> None:
        """cpm_pass for every data slot through `upto` not yet made, in slot
        order, each seeing only earlier passes' posteriors (not what keyed
        decodes learned since), so they are what a streaming receiver makes."""
        for s in range(self.n_ctl_slots, upto + 1):
            if s not in self._pass_post:
                cands, usable, p = cpm_pass(self.r, self.soft, self._pass_post, s, self._late)
                self._pass_post[s] = p
                self._raw.setdefault(s, (cands, usable))
                if p is not None:
                    self._post.setdefault(s, p)

    def _dd(self, spec) -> bool:
        if self.n_ctl_slots:  # CPM: data slots (GF(M)), on the burst's complex bins
            return DD and CPM_DD and "Z" in self.r and spec.code == "gf"
        return DD and "hp" in self.r and spec.code == "ldpc"

    def _iters(self) -> int:
        return CPM_DD_ITERS if self.n_ctl_slots else DD_ITERS

    def _late(self) -> bool:
        return self.dd_until is not None and time.monotonic() >= self.dd_until

    def _learn(self, slot: int, spec, payload: bytes, rv: int, m: int) -> None:
        is_ofdm = not self.n_ctl_slots and self.r.get("family") != "cpm"
        if is_ofdm or self._dd(spec):
            coded = codes.encode(spec, payload, rv, m, index=slot)
            if is_ofdm:
                self._coded[slot] = coded
            if self._dd(spec):
                self._post[slot] = _sure(spec, coded)

    def measure_data(self, measured: dict) -> dict | None:
        """Refine `measured` with post-decode data-cell MER (data_mer).
        Returns None for CPM bursts or when DATA2G_DATA_MER=0."""
        if not P.DATA_MER or self.n_ctl_slots or self.r.get("family") == "cpm":
            return None
        key = (len(self._coded), measured.get("snr_est"))
        if self._mer_cache is not None and self._mer_cache[0] == key:
            return self._mer_cache[1]
        out = data_mer(self.r, self._coded, measured)
        self._mer_cache = (key, out)
        return out

    def _refine(self, slot: int, spec, post: np.ndarray) -> None:
        """DD after a failed decode of `slot`: its decoder's a-posteriori
        LLRs `post` join the other slots' as soft pilots, the channel is
        re-estimated and the soft bits remade."""
        self._post[slot] = post
        if not self._blind:
            # RV 0 slots the link has not asked about yet: parity checks
            # alone say they decoded (a resend at another RV simply fails)
            self._blind = True
            todo = [s for s in range(self.n_ctl_slots, self.n_cw) if s not in self._post]
            if self.n_ctl_slots:  # CPM: every slot's pass (a streaming receiver has made them)
                self._cpm_passes(self.n_cw - 1)
            elif todo:
                if isinstance(self.soft, _SlotSoft):  # a partial burst: the slots in so far
                    todo = [s for s in todo if s in set(modem.ready_slots(self.r))]
                info, ok = codes.decode_llrs(spec, np.stack([self.soft[s] for s in todo])) if todo else ([], [])
                for s, i, o in zip(todo, info, ok):
                    if o:
                        coded = codes.encode_info(spec, i[None], 0)[0]
                        self._post[s] = _sure(spec, coded)
                        if not self.n_ctl_slots and self.r.get("family") != "cpm":
                            self._coded[s] = coded
        # CPM: the estimate is the posteriors themselves; each slot's metric is local (cpm.dd_metric), made as asked
        est = dict(self._post) if self.n_ctl_slots else _dd_estimate(self.r, self._post)
        self._est, self._soft_dd = est, {}

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
