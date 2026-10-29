"""Payload codecs: bytes <-> coded bits, and soft bits -> bytes. The codes
themselves (ldpc.py, polar.py) know nothing of submodes: this module keys
them by spec (CodeSpec) and frames what they carry.

TX: payload -> info_bits (payload, its CRC XORed with the ARQ's mask, zero
fill, PN9-scrambled by burst position) -> encode_info (LDPC, or polar with
the payload's bits reversed: polar_order; then the BICM interleaver) ->
spread (burst symbol order). RX undoes it:
despread -> a decode -> descramble and check the CRC.

  framing       crc_bits, payload_bytes, info_bits, payloads, crc_ok
  codes, format ldpc_code, polar_code, frozen (format/*.npz), fingerprint
  interleaving  interleaver, compute_interleaver (design time)
  burst order   spread, despread
  scrambling    scramble_seed, scrambler, PLAIN, flip, descramble
  encode        encode, encode_info
  IR (HARQ)     rv_cycle, rv_positions, buffer_len, ir_extent, combine,
                decode_buffer, decode_buffer_raw (a buffer, mask open)
  decode        decode_many: a burst, masks known (modem);
                decode_raw + check: decode once, try each CRC mask (ARQ);
                decode_posterior: with the LDPC posterior (ARQ's DD);
                decode_llrs: info bits and success, no framing
  studies only  decode_llrs(device=) (torch: decoders_torch),
                label_reliability; decode (one codeword) is for tests

Callers: modem.py and cpm.py (encode, spread, despread, combine,
decode_many), arq/phy.py (the IR and decode entry points), the ARQ, KISS
and the TNC (payload_bytes, rv_cycle).

Soft convention everywhere: positive means bit 0 (an LLR log P0/P1).
Every codeword's info bits end in a CRC over its payload: CRC-16/CCITT-
FALSE for short LDPC codewords, CRC-32 (zlib) once k >= 512, where
CRC-16's 2^-16 miss rate would dominate the 1e-6 undetected-error target;
CRC-24C on polar (crc_bits).
Coded bits leave here already interleaved (bit-interleaved coded
modulation): `encode` returns them in constellation-mapping order and the
decoders take soft bits in that order.

GF(M) (code "gf": the CPM modes' data, data2g.gfldpc): one code symbol
per M-FSK tone. Its soft values are q = M log-likelihoods per symbol
(symbol-major, the symbol's value v carried by tone gray(v)) where binary
codes have one LLR per bit: soft_len values a codeword, the interleaver
and RVs move whole symbols, and scrambling permutes each symbol's
likelihoods (plain) where it flips a bit's sign. Coded bits leave
encode as for any code: each symbol's p bits, MSB first.

Native: native/core/codes (codes::Spec: CodeSpec with the interleaver,
info set and CRC size looked up once). Parity:
tests/test_native_codes.py, and tests/native_subs/{base,codes}.py swap
these functions for C++ under pytest --native, private ones included
(_decoder, _with_crc).
"""

import binascii
from pathlib import Path
from functools import lru_cache
from typing import Protocol

import numpy as np

from . import constellation, gfldpc, ldpc, polar


class CodeSpec(Protocol):
    """What this module reads of a spec: config.SubmodeSpec and
    cpm.CpmSpec both have it. Specs key lru_caches, so they are frozen
    (hashable). Native: codes::Spec, built from config::SUBMODES and the
    CPM tables."""
    name: str
    code: str  # "ldpc" | "polar"
    k: int  # info bits incl. CRC
    coded_bits: int
    index: int  # interleaver seed and frozen-file name
    constellation: str  # LDPC interleaver: label reliabilities ("fsk*": none)
    band: str  # frozen-file name
    frames_per_cw: int  # fingerprint only
    headroom: float  # fingerprint only


INTERLEAVER_SEED = 2026


# --- framing: CRCs -------------------------------------------------------------------

def crc_bits(spec: CodeSpec) -> int:
    """CRC-24 on polar codewords: CRC-aided list decoding takes the first of
    its 8 paths whose CRC checks, so a failed decode passed CRC-16 8 in 65536
    times, and a random control word fails the session (scripts/crc_study.py;
    k + 8 kept the payloads). LDPC: CRC-32 from k 512, else CRC-16 (a decode
    must also converge: payloads)."""
    if spec.code == "polar":
        return 24
    return 32 if spec.k >= 512 else 16


def _crc24_table() -> list[int]:
    out = []
    for i in range(256):
        c = i << 16
        for _ in range(8):
            c = ((c << 1) ^ CRC24_POLY) & 0xFFFFFF if c & 0x800000 else (c << 1) & 0xFFFFFF
        out.append(c)
    return out


CRC24_POLY = 0xB2B117  # CRC-24C, for polar-coded control
_CRC24 = _crc24_table()


def crc24(data: bytes, crc: int = 0xFFFFFF) -> int:
    for b in data:
        crc = ((crc << 8) & 0xFFFFFF) ^ _CRC24[(crc >> 16) ^ b]
    return crc


def payload_bytes(spec: CodeSpec) -> int:
    return (spec.k - crc_bits(spec)) // 8


def _with_crc(payload: bytes, n_crc: int, mask: int = 0) -> bytes:
    """payload + CRC, XORed with `mask` (docs/arq.md §2: a session's
    per-codeword identity; 0 outside sessions, the frozen format)."""
    if n_crc == 16:
        return payload + (binascii.crc_hqx(payload, 0xFFFF) ^ mask & 0xFFFF).to_bytes(2, "big")
    if n_crc == 24:
        return payload + (crc24(payload) ^ mask & 0xFFFFFF).to_bytes(3, "big")
    return payload + (binascii.crc32(payload) ^ mask & 0xFFFFFFFF).to_bytes(4, "big")


# --- the codes per spec, and frozen formats ------------------------------------------

@lru_cache(maxsize=None)
def ldpc_code(spec: CodeSpec) -> ldpc.QCLDPC:
    return ldpc.qc_code(spec.k, spec.coded_bits)


@lru_cache(maxsize=None)
def gf_code(spec: CodeSpec) -> gfldpc.GFCode:
    """GF(M) specs: the mother code (codes_data/gf_codes.npz), its
    rate-compatible extension included; the codeword is its first n_sym
    symbols."""
    return gfldpc.load(spec.name)


def n_sym(spec: CodeSpec) -> int:
    """GF(M): code symbols a codeword."""
    return spec.coded_bits // spec.bits_per_cu


def soft_len(spec: CodeSpec) -> int:
    """Soft values a codeword: its coded bits, or GF(M) q per symbol."""
    return n_sym(spec) << spec.bits_per_cu if spec.code == "gf" else spec.coded_bits


# ponytail: GA-DE frozen set at a fixed design point until
# a per-submode search (planned; no script yet) designs one.
POLAR_DESIGN_SNR_DB = -3.0
POLAR_LIST = 8
# The modem's LDPC iteration cap (native codes::ldpc_iters, the app's "decoder effort"): the layered
# decoder at 20 matched the old flooding one at 40; 25 is about 0.1 dB better (scripts/ldpc_schedule).
LDPC_ITERS = 25
GF_ITERS = 50  # the GF(M) decoder's cap (flooding FFT-BP; scripts/gf_study.py)


FORMAT_DIR = Path(__file__).parent / "format"


def fingerprint(spec: CodeSpec) -> str:
    """Everything a frozen file depends on; a mismatch means it is stale.
    (The empty field was a protograph path, never set; kept so the frozen
    fingerprints still match.)"""
    return (f"{spec.band}|{spec.index}|{spec.code}|{spec.constellation}|{spec.frames_per_cw}|"
            f"{spec.k}||{spec.headroom:g}")


def frozen(spec: CodeSpec) -> dict | None:
    """The committed on-air data for a submode (tools/freeze_format.py),
    or None if there is none for exactly this spec (a candidate, or a
    submode whose parameters changed since it was frozen)."""
    path = FORMAT_DIR / f"{spec.band}_{spec.index:02d}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return dict(d) if str(d["fingerprint"]) == fingerprint(spec) else None


@lru_cache(maxsize=None)
def polar_order(spec: CodeSpec) -> np.ndarray:
    """(k,) the order info bits enter a polar code (its info set, increasing
    index): the payload's bits reversed, then the CRC and fill. A control's
    trailing zeros (frames.Control.pack) then sit on the least reliable
    positions, where knowing them buys most (decode_raw_pinned): CPM control
    with 12 B of content, 10% failure about -4.3 dB pinned against -3.0 with
    the payload in order (-2.8 plain). Its own inverse."""
    nb = 8 * payload_bytes(spec)
    return np.concatenate([np.arange(nb)[::-1], np.arange(nb, spec.k)])


@lru_cache(maxsize=None)
def polar_code(spec: CodeSpec) -> polar.PolarCode:
    f = frozen(spec)
    info = tuple(int(i) for i in f["info_pos"]) if f is not None and "info_pos" in f else None
    return polar.PolarCode(spec.k, spec.coded_bits, design_snr_db=POLAR_DESIGN_SNR_DB, frozen_override=info)


# --- interleaving (BICM) -------------------------------------------------------------

def label_reliability(points: np.ndarray, rate: float, n: int = 20000) -> np.ndarray:
    """Per-label-bit BMI on AWGN at the operating point: the Es/N0 where
    the constellation's total BMI equals rate * m. Fixed seed; frozen as
    data with the ladder (plan step 9)."""
    m = constellation.bits_per_symbol(points)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, (n, m))
    x = constellation.modulate(bits.reshape(-1), points)
    z = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)

    def per_label(esn0_db):
        var = np.full(n, 10 ** (-esn0_db / 10))
        l = constellation.llr(x + z * np.sqrt(var), np.ones(n), var, points).reshape(n, m)
        return 1 - np.mean(np.logaddexp(0, -(1 - 2 * bits) * l), axis=0) / np.log(2)

    lo, hi = -10.0, 40.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if per_label(mid).sum() >= rate * m else (mid, hi)
    return per_label(hi)


@lru_cache(maxsize=None)
def interleaver(spec: CodeSpec) -> np.ndarray:
    """The frozen permutation if this submode has one, else computed."""
    f = frozen(spec)
    return f["perm"] if f is not None else compute_interleaver(spec)


def _deinterleave(spec: CodeSpec, soft, dtype=None):
    """(B, coded_bits) soft bits in mapping order -> code order: out[:,
    perm[i]] = soft[:, i], undoing encode_info's coded[:, perm]. numpy (in
    `dtype`, default soft's), or torch on soft's device. GF(M): whole
    symbols' likelihoods."""
    perm = _soft_perm(spec)
    if isinstance(soft, np.ndarray):
        out = np.empty_like(soft, dtype=dtype)
        out[:, perm] = soft
        return out
    import torch

    out = torch.empty_like(soft)
    out[:, torch.as_tensor(perm, device=soft.device)] = soft
    return out


@lru_cache(maxsize=None)
def _soft_perm(spec: CodeSpec) -> np.ndarray:
    """The interleaver over soft values: GF(M)'s symbol permutation expanded to its q likelihoods each."""
    perm = np.asarray(interleaver(spec))
    if spec.code != "gf":
        return perm
    q = 1 << spec.bits_per_cu
    return (perm[:, None] * q + np.arange(q)).reshape(-1)


def compute_interleaver(spec: CodeSpec) -> np.ndarray:
    """perm[i] = which code bit goes to mapping position i (label bit
    i % m of symbol i // m).

    LDPC: degree-aware. Code bits sorted by variable-node degree go, in
    equal groups, to label bits sorted by reliability: the most connected
    bits on the most reliable labels; each group scattered randomly over
    its label's positions. Measured, learned 64-point, LDPC r1/2 n=2880:
    AWGN BER at 12 dB 1.0e-3 (random) -> 4.2e-5, mpd PER at 19 dB 0.16 ->
    0.088; the reverse assignment fails outright. Others: random."""
    rng = np.random.default_rng(INTERLEAVER_SEED + spec.index)
    if spec.code == "gf":
        return rng.permutation(n_sym(spec))  # whole symbols
    n = spec.coded_bits
    if spec.code != "ldpc" or spec.constellation.startswith("fsk"):
        return rng.permutation(n)  # (M-FSK: no labels to rank, data2g.cpm)
    deg = ldpc.sent_degrees(ldpc_code(spec))
    order = np.lexsort((rng.random(n), -deg))  # highest degree first
    pts = constellation.load(spec.constellation)
    m = constellation.bits_per_symbol(pts)
    labels = np.argsort(label_reliability(pts, spec.k / n))[::-1]  # most reliable first
    perm = np.empty(n, dtype=np.int64)
    per = n // m
    for g, lab in enumerate(labels):
        perm[rng.permutation(np.arange(lab, n, m))] = order[g * per : (g + 1) * per]
    return perm


# --- burst order, scrambling, encode -------------------------------------------------

def spread(coded, m: int):
    """(..., n_cw, N) coded bits -> (..., n_cw * N) in burst order: every
    codeword's constellation symbols (m bits each, kept together) dealt
    round-robin over the whole burst, so each codeword gets the burst's
    full time diversity. Measured, 16-QAM r1/2 at 14 dB: 32-frame bursts
    on mpp/mpd PER 3.2e-2 -> 0, 8-frame mpd 4.2e-2 -> 3.9e-3; neutral on
    mpg/mps. Must be symbol-, not bit-wise: dealing bits puts codeword j
    on label bit j mod m of every symbol (weak bits for half of them).
    Works on numpy arrays and torch tensors alike."""
    *lead, n_cw, n = coded.shape
    return coded.reshape(*lead, n_cw, n // m, m).swapaxes(-3, -2).reshape(*lead, n_cw * n)


def despread(x, n_cw: int, m: int):
    """Inverse of `spread`: (..., n_cw * N) -> (..., n_cw, N)."""
    *lead, total = x.shape
    n = total // n_cw
    return x.reshape(*lead, n // m, n_cw, m).swapaxes(-3, -2).reshape(*lead, n_cw, n)


PLAIN = -1  # burst position meaning "not scrambled" (soft bits already flipped: flip())


def scramble_seed(index: int = 0) -> int:
    """A codeword's scrambler seed (1-511, distinct for positions 0-510) from
    its position in the burst alone: public, so anyone can descramble what
    is on air without a session key (docs/arq.md §2). PLAIN: 0, all-zero
    PN9."""
    return 0 if index == PLAIN else 1 + (index * 0x9E3779B1) % 511


@lru_cache(maxsize=None)
def flip(spec: CodeSpec, index: int, rv: int = 0) -> np.ndarray:
    """(coded_bits,) +-1 in mapping order: soft bits of a codeword sent at
    burst position `index` and RV `rv`, times this, are those of the same
    codeword unscrambled (the code is linear: C(u ^ s) = C(u) ^ C(s)).
    Resends in other slots combine that way, and decode with index PLAIN."""
    return 1.0 - 2.0 * encode_info(spec, scrambler(spec.k, scramble_seed(index))[None], rv)[0]


def plain(spec: CodeSpec, soft: np.ndarray, index: int, rv: int = 0) -> np.ndarray:
    """(soft_len,) soft values of a codeword sent at burst position `index`
    and RV `rv` -> those of the same codeword unscrambled: flip's signs, or
    GF(M) each symbol's likelihoods permuted (L'[w] = L[w ^ s], s its
    scrambling: the code is GF(2)-linear too)."""
    if spec.code != "gf":
        return flip(spec, index, rv) * soft
    s = _gf_flip(spec, index, rv)
    q = 1 << spec.bits_per_cu
    L = np.asarray(soft).reshape(-1, q)
    return L[np.arange(len(L))[:, None], np.arange(q)[None] ^ s[:, None]].reshape(-1)


@lru_cache(maxsize=None)
def _gf_flip(spec: CodeSpec, index: int, rv: int) -> np.ndarray:
    """(n_sym,) the scrambling's symbols, mapping order."""
    x = encode_info(spec, scrambler(spec.k, scramble_seed(index))[None], rv)
    return gfldpc.pack(x, spec.bits_per_cu, n_sym(spec))[0]


@lru_cache(maxsize=None)
def scrambler(k: int, seed: int = 0x1FF) -> np.ndarray:
    """PN9 (x^9 + x^5 + 1, output = feedback bit) from `seed`, k bits. Info
    bits are XORed with it before encoding: an unscrambled zero-padded
    payload (control codewords, a stream's last codeword) codes to ~10%
    ones, piles OFDM symbols onto a few points, and the clipper wrecks them
    (in real-modem sessions, scripts/phy_session.py: 3-codeword bursts
    failed their control at 8 dB AWGN, every time). Seeds differ per
    codeword: identical payloads in one burst,
    scrambled alike, still decoded half the time at 8 dB."""
    reg, out = seed, np.empty(k, np.uint8)
    for i in range(k):
        bit = ((reg >> 8) ^ (reg >> 4)) & 1
        out[i] = bit
        reg = ((reg << 1) | bit) & 0x1FF
    return out


def info_bits(spec: CodeSpec, payload: bytes, crc_mask: int = 0, index: int = 0) -> np.ndarray:
    """payload -> the k info bits the encoder takes: payload, masked CRC,
    zero fill, scrambled."""
    if len(payload) != payload_bytes(spec):
        raise ValueError(f"{spec.name} carries {payload_bytes(spec)} bytes, got {len(payload)}")
    bits = np.unpackbits(np.frombuffer(_with_crc(payload, crc_bits(spec), crc_mask), np.uint8))
    return np.pad(bits, (0, spec.k - len(bits))) ^ scrambler(spec.k, scramble_seed(index))


def encode(spec: CodeSpec, payload: bytes, rv: int = 0, crc_mask: int = 0, index: int = 0) -> np.ndarray:
    """One codeword's payload -> (coded_bits,) array of 0/1, interleaved.
    `index`: its position in the burst (the scrambler seed)."""
    return encode_info(spec, info_bits(spec, payload, crc_mask, index)[None], rv)[0]


def encode_info(spec: CodeSpec, bits: np.ndarray, rv: int = 0) -> np.ndarray:
    """(B, k) info bits (CRC included) -> (B, coded_bits), interleaved.
    `rv`: redundancy version (rv_positions); 0 is the first transmission."""
    if spec.code == "gf":
        c, p = gf_code(spec), spec.bits_per_cu
        x = c.encode(gfldpc.pack(np.asarray(bits), p, c.k))[:, _rv_symbols(spec, rv)]
        return gfldpc.unpack(x[:, interleaver(spec)], p)
    if rv and spec.code == "ldpc":
        coded = ldpc_code(spec).mother().encode(bits)[:, rv_positions(spec, rv)]
    elif spec.code == "ldpc":
        coded = ldpc_code(spec).encode(bits)
    else:
        coded = polar_code(spec).encode(np.asarray(bits)[:, polar_order(spec)])
    return coded[:, interleaver(spec)]


# --- incremental redundancy (docs/arq.md §5) -----------------------------------------
# LDPC: RV r sends positions [r n, (r + 1) n) of the mother code's circular
# buffer (every base row: rate ~1/5), wrapping. RV 0 is the codeword as
# frozen; RV 1 onwards is fresh parity until the buffer runs out, then
# repeats (Chase). Polar: every RV is the same codeword (Chase). GF(M):
# the same over symbols, of a raptor-like mother (gfldpc.extend: rate 1/6;
# on two receptions of fsk8r50-r1/3, ideal channels, 0.3-0.5 dB better
# than Chase, even on fsk32r62: scripts/gf_study.py ir).
# Soft bits accumulate over the buffer in code order ("buffer": (B, L)).

def rv_cycle(spec: CodeSpec) -> int:
    """Distinct redundancy versions: resend r uses RV r mod this."""
    return 1 if spec.code == "polar" else 4  # LDPC, GF(M): mother codes (rate ~1/5, 1/6)


def buffer_len(spec: CodeSpec) -> int:
    if spec.code == "gf":
        return gf_code(spec).n << spec.bits_per_cu
    return ldpc_code(spec).mother().n if spec.code == "ldpc" else spec.coded_bits


def _rv_symbols(spec: CodeSpec, rv: int) -> np.ndarray:
    """GF(M): the mother code's symbols RV `rv` carries, as LDPC's bits (rv_positions)."""
    n = n_sym(spec)
    return (rv % rv_cycle(spec) * n + np.arange(n)) % gf_code(spec).n


def rv_positions(spec: CodeSpec, rv: int) -> np.ndarray:
    """Buffer positions RV `rv` carries, in code order."""
    n = spec.coded_bits
    if spec.code == "gf":
        q = 1 << spec.bits_per_cu
        return (_rv_symbols(spec, rv)[:, None] * q + np.arange(q)).reshape(-1)
    if spec.code != "ldpc":
        return np.arange(n)
    return (rv % rv_cycle(spec) * n + np.arange(n)) % buffer_len(spec)


def combine(spec: CodeSpec, buf: np.ndarray | None, soft: np.ndarray, rvs) -> np.ndarray:
    """Add (B, coded_bits) soft bits in mapping order, sent at RVs `rvs`,
    into the (B, L) buffer (None: a fresh one). -> the buffer."""
    soft = np.atleast_2d(soft)
    if buf is None:
        buf = np.zeros((len(soft), buffer_len(spec)))
    deint = _deinterleave(spec, soft, float)
    for b, rv in enumerate(np.broadcast_to(rvs, len(soft))):
        np.add.at(buf[b], rv_positions(spec, int(rv)), deint[b])
    return buf


def ir_extent(spec: CodeSpec, max_rv: int) -> int:
    """The buffer positions an LDPC decode reads with RVs up to `max_rv`
    received: the mother code is cut there (unreceived parity would only
    slow it)."""
    return min(buffer_len(spec), (min(max_rv, rv_cycle(spec) - 1) + 1) * soft_len(spec))


def decode_buffer(spec: CodeSpec, buf: np.ndarray, max_rv: int = 0, crc_mask=0, index=None) -> list[tuple[bytes, bool]]:
    """(B, L) combined buffers, with RVs up to `max_rv` received -> [(payload,
    crc_ok)]: LDPC decodes the mother code cut to ir_extent."""
    masks = np.broadcast_to(crc_mask, len(buf))
    idx = np.arange(len(buf)) if index is None else np.broadcast_to(index, len(buf))
    if spec.code == "polar":
        return payloads(spec, *_decode_code_order(spec, _decoder(spec), buf, crc_mask=masks, index=idx),
                        masks, idx)
    extent = ir_extent(spec, max_rv)
    return payloads(spec, *_decode_code_order(spec, _ext_decoder(spec, extent), buf[:, :extent]), masks, idx)


def decode_posterior(spec: CodeSpec, soft=None, buf=None, max_rv: int = 0, rv: int = 0) -> tuple:
    """One LDPC decode as decode_raw (`soft`: a slot's (coded_bits,) soft
    bits alone) or decode_buffer (`buf`: a (1, L) combined buffer, RVs up
    to `max_rv`, the slot sent at `rv`) would -> (info bits (k,) as decoded,
    still scrambled unless from `buf`; converged; a-posteriori LLRs of the
    slot's coded bits in mapping order). Decision-directed re-estimation
    (arq/phy.py, DD) needs the posterior of every failed decode, so it
    comes from the same pass. Native: codes::decode_posterior. LDPC only
    (DD runs on OFDM)."""
    if spec.code != "ldpc":
        raise ValueError(f"decode_posterior: {spec.name} is not LDPC")
    perm = interleaver(spec)
    if buf is None:
        dec, llr = _decoder(spec), _deinterleave(spec, np.atleast_2d(soft), float)
    else:
        extent = ir_extent(spec, max_rv)
        dec, llr = _ext_decoder(spec, extent), buf[:, :extent]
    out, ok, post = dec.decode(llr, iters=LDPC_ITERS, posterior=True)
    code = post[0]
    return _numpy(out)[0], bool(_numpy(ok)[0]), (code if buf is None else code[rv_positions(spec, rv)])[perm]


@lru_cache(maxsize=None)
def _ext_decoder(spec: CodeSpec, extent: int):
    if spec.code == "gf":
        return gfldpc.Decoder(gfldpc.cut(gf_code(spec), extent >> spec.bits_per_cu))
    return ldpc.Decoder(ldpc_code(spec).mother(extent))


# --- decode --------------------------------------------------------------------------

def payloads(spec: CodeSpec, bits: np.ndarray, converged, masks=None, index=None) -> list[tuple[bytes, bool]]:
    """Decoded (scrambled) info bits -> [(payload, ok)]; row i was sent
    with CRC mask masks[i] at burst position index[i] (default: i). ok:
    the CRC checks and the decoder converged (LDPC: every parity check
    satisfied). The CRC alone let a failed LDPC decode's guess through 1 in
    65536 (CRC16): the v6 session data delivered one corrupt codeword that
    way; requiring convergence rejected 41 of 46579 correct decodes
    (scripts/crc_study.py)."""
    n_crc = crc_bits(spec)
    out = []
    masks = np.zeros(len(bits), int) if masks is None else masks
    index = np.arange(len(bits)) if index is None else index
    for b, m, i, c in zip(bits, masks, index, converged):
        b = descramble(spec, b, i)
        data = np.packbits(b[: 8 * (payload_bytes(spec) + n_crc // 8)]).tobytes()
        payload = data[: -n_crc // 8]
        out.append((payload, bool(c) and _with_crc(payload, n_crc, int(m)) == data))
    return out


def decode(spec: CodeSpec, soft: np.ndarray) -> tuple[bytes, bool]:
    """(coded_bits,) soft values in mapping order -> (payload, crc_ok)."""
    return decode_many(spec, soft[None])[0]


def decode_many(spec: CodeSpec, soft: np.ndarray, crc_mask=0, index=None) -> list[tuple[bytes, bool]]:
    """(B, coded_bits) -> [(payload, crc_ok)] in one batched decode: 64
    polar codewords take 0.04-0.08 s batched, 1.6-3.3 s one at a time
    (scripts/decode_latency.py). `crc_mask`, `index`: per row, the CRC mask
    and burst position it was sent with (default 0 and the row number)."""
    masks = np.broadcast_to(crc_mask, len(soft))
    idx = np.arange(len(soft)) if index is None else np.broadcast_to(index, len(soft))
    return payloads(spec, *decode_llrs(spec, soft, crc_mask=masks, index=idx), masks, idx)


def decode_llrs(spec: CodeSpec, llr, iters: int = LDPC_ITERS, device=None, crc_mask=0, index=0):
    """Batched decode: (B, coded_bits) LLRs in mapping order (numpy, or
    torch with `device`) -> (info bits (B, k) numpy uint8, success (B,)
    numpy bool). LDPC: success = H satisfied. Polar: the first list path
    (best metric first) whose CRC checks; success = one did. `device`
    (studies only): decode on torch there (decoders_torch)."""
    if device is not None:
        import torch

        deint = _deinterleave(spec, torch.as_tensor(llr, dtype=torch.float32, device=device))
    else:
        deint = _deinterleave(spec, np.asarray(llr, dtype=np.float32))
    return _decode_code_order(spec, _decoder(spec, device), deint, iters, crc_mask, index)


def decode_raw(spec: CodeSpec, soft: np.ndarray, index=None) -> tuple[np.ndarray, np.ndarray]:
    """(B, coded_bits) soft bits in mapping order -> (candidates (B, L, k)
    uint8, descrambled; usable (B, L) bool), with the CRC mask left open:
    decode once, then check() each mask in question. LDPC: one candidate,
    usable when H is satisfied. Polar: the list, best metric first, all
    usable (the CRC picks). `index`: per row, the burst position (default
    the row number)."""
    soft = np.asarray(soft, dtype=np.float32)
    deint = _deinterleave(spec, soft)
    if spec.code != "polar":
        out, ok = _decode_code_order(spec, _decoder(spec), deint)
        cands, usable = out[:, None], ok[:, None]
    else:
        cands = _numpy(_decoder(spec).decode(deint)[0])[..., polar_order(spec)]
        usable = np.ones(cands.shape[:2], bool)
    idx = np.arange(len(soft)) if index is None else np.broadcast_to(index, len(soft))
    return np.stack([descramble(spec, c, i) for c, i in zip(cands, idx)]), usable


def decode_buffer_raw(spec: CodeSpec, buf: np.ndarray, max_rv: int = 0, known=None) -> tuple[np.ndarray, np.ndarray]:
    """decode_raw of (B, L) combined buffers (decode_buffer's input, RVs up
    to `max_rv` received): the CRC mask left open, candidates already plain
    (the buffer is unscrambled). `known`: info bits known as in
    decode_buffer_pinned. A broadcast control sent twice reads its group
    from these (kisslink, docs/broadcast.md §2)."""
    buf = np.asarray(buf)
    if spec.code != "polar":
        extent = ir_extent(spec, max_rv)
        b = buf if known is None else pin_buffer(spec, buf, known)
        out, ok = _decode_code_order(spec, _ext_decoder(spec, extent), np.asarray(b[:, :extent], np.float32))
        return out[:, None].astype(np.uint8), ok[:, None]
    deint = np.asarray(buf[:, : spec.coded_bits], np.float32)
    if known is None:
        cands = _numpy(_decoder(spec).decode(deint)[0])[..., polar_order(spec)].astype(np.uint8)
    else:
        cands = _pinned_paths(spec, deint, known, PLAIN)
    return cands, np.ones(cands.shape[:2], bool)


def descramble(spec: CodeSpec, bits: np.ndarray, index: int) -> np.ndarray:
    """(..., k) decoded info bits sent at burst position `index` -> unscrambled."""
    return bits.astype(np.uint8) ^ scrambler(spec.k, scramble_seed(int(index)))[: bits.shape[-1]]


def check(spec: CodeSpec, cands: np.ndarray, usable: np.ndarray, crc_mask: int) -> bytes | None:
    """One row of decode_raw -> the payload of its first usable candidate
    whose CRC passes under `crc_mask`, or None."""
    n_crc = crc_bits(spec)
    nbytes = payload_bytes(spec) + n_crc // 8
    for b, u in zip(cands, usable):
        if u:
            data = np.packbits(b[: 8 * nbytes]).tobytes()
            if _with_crc(data[: -n_crc // 8], n_crc, crc_mask) == data:
                return data[: -n_crc // 8]
    return None


# --- known info bits (source-aware decoding) -------------------------------------------
# A receiver often knows part of a codeword before decoding it: a control
# codeword is a few bytes of content and then zeros to its payload size
# (frames.Control.pack). LDPC is systematic, so a known info bit that is sent
# is pinned at the decoder's input, its soft bit set to the clamp; the code
# is then effectively shortened (the punctured first 2Z info columns have no
# soft bit to pin). Polar sends no info bits: known ones are frozen instead,
# in a code with the rest as its info set, after their own codeword is taken
# off the soft bits (the code is linear); polar_order puts the payload's tail
# on its weakest positions so that this pays. LDPC: nothing on air changes.
# decode_raw_pinned and decode_buffer_pinned do either.

@lru_cache(maxsize=None)
def _info_code_positions(spec: CodeSpec) -> np.ndarray:
    """(k,) where each info bit is sent at RV 0 in code order (its position
    in an IR buffer too: the mother code starts with the codeword), or -1."""
    out = np.full(spec.k, -1, dtype=np.int64)
    if spec.code != "ldpc":
        return out
    sent = np.asarray(ldpc_code(spec).sent)
    p = np.flatnonzero(sent < spec.k)  # sent positions (code order) carrying info bits
    out[sent[p]] = p
    return out


@lru_cache(maxsize=None)
def systematic_positions(spec: CodeSpec) -> np.ndarray:
    """(k,) where each info bit is sent at RV 0, in mapping order, or -1
    (punctured, filler; polar, GF(M): every one)."""
    if spec.code != "ldpc":
        return np.full(spec.k, -1, dtype=np.int64)
    code = _info_code_positions(spec)
    where = np.empty(spec.coded_bits, dtype=np.int64)
    where[np.asarray(interleaver(spec))] = np.arange(spec.coded_bits)  # code order -> mapping order
    return np.where(code >= 0, where[np.maximum(code, 0)], -1)


def zero_tail(spec: CodeSpec, nbytes: int) -> np.ndarray:
    """(k,) int8 known info bits, -1 unknown: payload bytes from `nbytes`
    on are zero (a control's padding), and the zero fill past the CRC."""
    known = np.full(spec.k, -1, dtype=np.int8)
    known[8 * nbytes : 8 * payload_bytes(spec)] = 0
    known[8 * payload_bytes(spec) + crc_bits(spec):] = 0
    return known


def known_bits(spec: CodeSpec, template: bytes, mask: bytes) -> np.ndarray:
    """(k,) int8 known info bits, -1 unknown: the payload bits `mask` sets
    are `template`'s (both payload_bytes long, MSB first), and the zero
    fill past the CRC. The CRC is never known: a primed decode still has
    to pass it (docs/arq.md §3a)."""
    pb = payload_bytes(spec)
    t = np.unpackbits(np.frombuffer(bytes(template[:pb]).ljust(pb, b"\0"), np.uint8))
    m = np.unpackbits(np.frombuffer(bytes(mask[:pb]).ljust(pb, b"\0"), np.uint8)).astype(bool)
    known = np.full(spec.k, -1, dtype=np.int8)
    known[: 8 * pb][m] = t[m]
    known[8 * pb + crc_bits(spec):] = 0
    return known


def score(spec: CodeSpec, soft: np.ndarray, payload: bytes, crc_mask: int, index: int) -> float:
    """Log-likelihood ratio (nats) of `payload`'s RV 0 codeword, sent at
    burst position `index`, against uniformly random bits, from its soft
    bits (mapping order). On noise, or anything that is not this codeword,
    P(score >= T) <= e^-T for calibrated LLRs (Markov: E[e^score] = 1)."""
    c = 1.0 - 2.0 * encode(spec, payload, 0, crc_mask, index)
    return float(np.sum(np.log(2.0) - np.logaddexp(0.0, -c * np.asarray(soft, float))))


def pin(spec: CodeSpec, soft: np.ndarray, known: np.ndarray, index: int) -> np.ndarray:
    """(coded_bits,) soft bits of an RV 0 codeword sent at burst position
    `index`, with every sent info bit that `known` gives ((k,) 0/1, -1
    unknown; unscrambled) pinned to the decoder's clamp. Unchanged for
    polar (no sent info bits)."""
    pos = systematic_positions(spec)
    sel = (np.asarray(known) >= 0) & (pos >= 0)
    out = np.array(soft, dtype=np.float32)
    bit = np.asarray(known)[sel].astype(np.uint8) ^ scrambler(spec.k, scramble_seed(index))[sel]
    out[pos[sel]] = ldpc.CH_CLAMP * (1.0 - 2.0 * bit)
    return out


def pin_buffer(spec: CodeSpec, buf: np.ndarray, known: np.ndarray) -> np.ndarray:
    """codes.pin for a combined IR buffer ((B, L), code order, unscrambled:
    decode_buffer's input). Unchanged for polar."""
    pos = _info_code_positions(spec)
    sel = (np.asarray(known) >= 0) & (pos >= 0)
    out = np.array(buf, dtype=float)
    out[:, pos[sel]] = ldpc.CH_CLAMP * (1.0 - 2.0 * np.asarray(known)[sel])
    return out


def info_llrs(spec: CodeSpec, soft: np.ndarray, index: int) -> np.ndarray:
    """(k,) the channel's LLR (log P0/P1) of each info bit as unscrambled, 0
    where it isn't sent: an RV 0 codeword's (coded_bits,) soft bits in
    mapping order, sent at burst position `index` (PLAIN: already plain)."""
    pos = systematic_positions(spec)
    sign = 1.0 - 2.0 * scrambler(spec.k, scramble_seed(index))
    return np.where(pos >= 0, np.asarray(soft, float)[np.maximum(pos, 0)] * sign, 0.0)


def info_llrs_buffer(spec: CodeSpec, buf: np.ndarray) -> np.ndarray:
    """info_llrs of one combined IR buffer row (code order, unscrambled)."""
    pos = _info_code_positions(spec)
    return np.where(pos >= 0, np.asarray(buf, float)[np.maximum(pos, 0)], 0.0)


def pad_prior(spec: CodeSpec, llr: np.ndarray) -> np.ndarray:
    """A control's padding, estimated from its info bits' LLRs `llr`
    (info_llrs) -> (k,) prior LLRs toward 0 per info bit, inf where known
    zero (the fill past the CRC). Where the padding starts: S(j), the sum
    over the bits of payload bytes j.. of log 2 P(bit = 0), gains about
    log 2 a sent bit of padding and loses on a content bit's 1s; softmax(S)
    over j = 1..payload_bytes is the start's distribution (bytes with no bit
    sent, LDPC's punctured first 2Z columns, add nothing and stay level). A
    payload bit is padding with P = P(start <= its byte), and with content
    bits uniform its prior is log(1 + P) / (1 - P)."""
    pb = payload_bytes(spec)
    t = np.log(2) - np.logaddexp(0.0, -np.asarray(llr, float)[: 8 * pb])  # 0 where llr is 0
    s = np.append(np.cumsum(t[::-1])[::-1][8::8], 0.0)  # S(j), j = 1..pb; one sequential sum, as native adds
    p = np.exp(s - s.max())
    pad = np.cumsum(p / np.sum(p))  # P(start <= j), j = 1..pb
    out = np.zeros(spec.k)
    out[8 : 8 * pb] = np.repeat(2 * np.arctanh(np.minimum(pad[:-1], 1 - 1e-7)), 8)  # byte 0: content
    out[8 * pb + crc_bits(spec):] = np.inf
    return out


def add_prior(spec: CodeSpec, soft: np.ndarray, prior: np.ndarray, index: int) -> np.ndarray:
    """(coded_bits,) soft bits of an RV 0 codeword sent at burst position
    `index` with each sent info bit's `prior` (pad_prior: toward 0,
    unscrambled) added, and set to the clamp where it is inf (codes.pin's).
    Unchanged for polar and GF(M) (no sent info bits)."""
    pos = systematic_positions(spec)
    sel = (np.asarray(prior) != 0) & (pos >= 0)
    sign = 1.0 - 2.0 * scrambler(spec.k, scramble_seed(index))[sel]
    pr = np.asarray(prior, float)[sel]
    out = np.array(soft, dtype=np.float32)
    out[pos[sel]] = np.where(np.isinf(pr), ldpc.CH_CLAMP * sign, out[pos[sel]] + pr * sign)
    return out


def add_prior_buffer(spec: CodeSpec, buf: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """add_prior for combined IR buffers ((B, L), code order, unscrambled)."""
    pos = _info_code_positions(spec)
    sel = (np.asarray(prior) != 0) & (pos >= 0)
    pr = np.asarray(prior, float)[sel]
    out = np.array(buf, dtype=float)
    out[:, pos[sel]] = np.where(np.isinf(pr), ldpc.CH_CLAMP, out[:, pos[sel]] + pr)
    return out


@lru_cache(maxsize=None)
def _pinned_polar(spec: CodeSpec, known_pos: tuple) -> tuple:
    """(the full code, the decoder of the code with info bits `known_pos`
    frozen, the info bits it decodes), numpy's."""
    info = np.asarray(polar_code(spec).info_pos)
    full = polar.PolarCode(spec.k, spec.coded_bits, POLAR_DESIGN_SNR_DB, tuple(int(i) for i in info))
    rest = np.setdiff1d(np.arange(spec.k), known_pos)
    code = polar.PolarCode(len(rest), spec.coded_bits, POLAR_DESIGN_SNR_DB, tuple(int(i) for i in info[rest]))
    return full, polar.SCLDecoder(code, POLAR_LIST), rest


def _pinned_paths(spec: CodeSpec, deint: np.ndarray, known: np.ndarray, index: int) -> np.ndarray:
    """Polar: (B, coded_bits) LLRs in code order, sent at burst position
    `index`, with the info bits `known` gives -> the list (B, L, k) of info
    bits as decoded (scrambled), best first."""
    order = polar_order(spec)
    known = np.asarray(known)
    u = np.where(known >= 0, known, 0).astype(np.uint8) ^ scrambler(spec.k, scramble_seed(int(index)))
    known, u = known[order], u[order]  # in the code's order
    kp = np.flatnonzero(known >= 0)
    full, dec, rest = _pinned_polar(spec, tuple(int(i) for i in kp))
    u[known < 0] = 0
    paths, _ = dec.decode(np.asarray(deint, np.float32) * (1.0 - 2.0 * full.encode(u[None])[0]))
    out = np.empty((*paths.shape[:2], spec.k), np.uint8)
    out[..., rest] = paths
    out[..., kp] = u[kp]
    return out[..., order]


def decode_raw_pinned(spec: CodeSpec, soft: np.ndarray, known: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    """decode_raw of (B, coded_bits) soft bits sent at burst position
    `index`, with the info bits `known` gives ((k,) 0/1, -1 unknown;
    unscrambled) known: LDPC pinned (pin), polar frozen; GF(M): nothing known (pin's no-op)."""
    if spec.code != "polar":
        return decode_raw(spec, np.stack([pin(spec, s, known, index) for s in np.atleast_2d(soft)]),
                          index=np.full(len(np.atleast_2d(soft)), index))
    cands = _pinned_paths(spec, _deinterleave(spec, np.atleast_2d(np.asarray(soft, np.float32))), known, index)
    return descramble(spec, cands, index), np.ones(cands.shape[:2], bool)


def decode_buffer_pinned(spec: CodeSpec, buf: np.ndarray, max_rv: int, known: np.ndarray, crc_mask=0) -> list[tuple[bytes, bool]]:
    """decode_buffer (index PLAIN) with the info bits `known` gives known:
    LDPC pinned (pin_buffer), polar frozen; GF(M): nothing known."""
    if spec.code != "polar":
        return decode_buffer(spec, pin_buffer(spec, buf, known), max_rv, crc_mask, index=PLAIN)
    paths = _pinned_paths(spec, np.asarray(buf)[:, : spec.coded_bits], known, PLAIN)
    masks = np.broadcast_to(crc_mask, len(paths))
    rep = lambda v: np.repeat(np.broadcast_to(v, len(paths)), paths.shape[1])  # noqa: E731
    crc = crc_ok(spec, paths.reshape(-1, spec.k), rep(masks), rep(PLAIN)).reshape(paths.shape[:2])
    pick = np.where(crc.any(axis=1), crc.argmax(axis=1), 0)
    idx = np.full(len(paths), PLAIN)
    return payloads(spec, paths[np.arange(len(paths)), pick], crc.any(axis=1), masks, idx)


def _numpy(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _decode_code_order(spec: CodeSpec, dec, deint, iters: int = LDPC_ITERS, crc_mask=0, index=0):
    """decode_llrs after deinterleaving: LLRs in the decoder's code order."""
    if spec.code == "gf":
        hard, ok, _ = dec.decode(np.asarray(deint).reshape(len(deint), -1, 1 << spec.bits_per_cu), GF_ITERS)
        return gfldpc.unpack(hard[:, dec.code.info], spec.bits_per_cu)[:, :spec.k], ok
    if spec.code == "ldpc":
        out, ok = dec.decode(deint, iters=iters)
        return _numpy(out), _numpy(ok)
    paths, _ = dec.decode(deint)
    paths = _numpy(paths)[..., polar_order(spec)]  # (B, L, k), best first
    rep = lambda v: np.repeat(np.broadcast_to(v, len(paths)), paths.shape[1])  # noqa: E731
    crc = crc_ok(spec, paths.reshape(-1, spec.k), rep(crc_mask), rep(index)).reshape(paths.shape[:2])
    pick = np.where(crc.any(axis=1), crc.argmax(axis=1), 0)
    return paths[np.arange(len(paths)), pick], crc.any(axis=1)


def crc_ok(spec: CodeSpec, bits: np.ndarray, crc_mask=0, index=0) -> np.ndarray:
    """(B, >= payload+CRC bits) decoded (scrambled) info bits -> (B,) CRC
    matches, row i sent with crc_mask[i] at burst position index[i]."""
    n_crc = crc_bits(spec)
    nbytes = payload_bytes(spec) + n_crc // 8
    out = []
    for r, m, i in zip(bits, np.broadcast_to(crc_mask, len(bits)), np.broadcast_to(index, len(bits))):
        d = np.packbits(r[: 8 * nbytes].astype(np.uint8) ^ scrambler(spec.k, scramble_seed(int(i)))[: 8 * nbytes])
        out.append(_with_crc(d[: -n_crc // 8].tobytes(), n_crc, int(m)) == d.tobytes())
    return np.array(out)


@lru_cache(maxsize=None)
def _decoder(spec: CodeSpec, device=None):
    """The numpy decoder, or with `device` the torch one there (studies)."""
    if device is not None:
        from . import decoders_torch

        if spec.code == "ldpc":
            return decoders_torch.Decoder(ldpc_code(spec), device=device)
        return decoders_torch.SCLDecoder(polar_code(spec), POLAR_LIST, device=device)
    if spec.code == "gf":
        return gfldpc.Decoder(gfldpc.cut(gf_code(spec), n_sym(spec)))
    if spec.code == "ldpc":
        return ldpc.Decoder(ldpc_code(spec))
    return polar.SCLDecoder(polar_code(spec), POLAR_LIST)
