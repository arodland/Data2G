"""Payload codecs: bytes <-> coded bits, and soft bits -> bytes.

Soft convention everywhere: positive means bit 0 (an LLR log P0/P1).
Every codeword's info bits end in a CRC over its payload: CRC-16/CCITT-
FALSE for short codewords, CRC-32 (zlib) once k >= 512, where CRC-16's
2^-16 miss rate would dominate the 1e-6 undetected-error target.

Coded bits leave here already interleaved (bit-interleaved coded
modulation): `encode` returns them in constellation-mapping order and
`decode` takes soft bits in that order.
"""

import binascii
from pathlib import Path
from functools import lru_cache

import numpy as np

from . import constellation, ldpc, polar
from .config import SubmodeSpec

INTERLEAVER_SEED = 2026


def crc_bits(spec: SubmodeSpec) -> int:
    """CRC-24 on polar codewords: CRC-aided list decoding takes the first of
    its 8 paths whose CRC checks, so a failed decode passed CRC-16 8 in 65536
    times, and a random control word fails the session (scripts/crc_study.py;
    k + 8 kept the payloads). LDPC: CRC-32 from k 512, else CRC-16 (a decode
    must also converge: _payloads)."""
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


def payload_bytes(spec: SubmodeSpec) -> int:
    return (spec.k - crc_bits(spec)) // 8


def _with_crc(payload: bytes, n_crc: int, mask: int = 0) -> bytes:
    """payload + CRC, XORed with `mask` (docs/arq.md §2: a session's
    per-codeword identity; 0 outside sessions, the frozen format)."""
    if n_crc == 16:
        return payload + (binascii.crc_hqx(payload, 0xFFFF) ^ mask & 0xFFFF).to_bytes(2, "big")
    if n_crc == 24:
        return payload + (crc24(payload) ^ mask & 0xFFFFFF).to_bytes(3, "big")
    return payload + (binascii.crc32(payload) ^ mask & 0xFFFFFFFF).to_bytes(4, "big")


@lru_cache(maxsize=None)
def ldpc_code(spec: SubmodeSpec) -> ldpc.QCLDPC:
    return ldpc.qc_code(spec.k, spec.coded_bits)


# ponytail: GA-DE frozen set at a fixed design point until
# scripts/design_polar.py searches one per submode.
POLAR_DESIGN_SNR_DB = -3.0
POLAR_LIST = 8


FORMAT_DIR = Path(__file__).parent / "format"


def _fingerprint(spec: SubmodeSpec) -> str:
    """Everything a frozen file depends on; a mismatch means it is stale.
    (The empty field was a protograph path, never set; kept so the frozen
    fingerprints still match.)"""
    return (f"{spec.band}|{spec.index}|{spec.code}|{spec.constellation}|{spec.frames_per_cw}|"
            f"{spec.k}||{spec.headroom:g}")


def frozen(spec: SubmodeSpec) -> dict | None:
    """The committed on-air data for a submode (tools/freeze_format.py),
    or None if there is none for exactly this spec (a candidate, or a
    submode whose parameters changed since it was frozen)."""
    path = FORMAT_DIR / f"{spec.band}_{spec.index:02d}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return dict(d) if str(d["fingerprint"]) == _fingerprint(spec) else None


@lru_cache(maxsize=None)
def polar_code(spec: SubmodeSpec) -> polar.PolarCode:
    f = frozen(spec)
    info = tuple(int(i) for i in f["info_pos"]) if f is not None and "info_pos" in f else None
    return polar.PolarCode(spec.k, spec.coded_bits, design_snr_db=POLAR_DESIGN_SNR_DB, frozen_override=info)


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
def interleaver(spec: SubmodeSpec) -> np.ndarray:
    """The frozen permutation if this submode has one, else computed."""
    f = frozen(spec)
    return f["perm"] if f is not None else compute_interleaver(spec)


def compute_interleaver(spec: SubmodeSpec) -> np.ndarray:
    """perm[i] = which code bit goes to mapping position i (label bit
    i % m of symbol i // m).

    LDPC: degree-aware. Code bits sorted by variable-node degree go, in
    equal groups, to label bits sorted by reliability: the most connected
    bits on the most reliable labels; each group scattered randomly over
    its label's positions. Measured, learned 64-point, LDPC r1/2 n=2880:
    AWGN BER at 12 dB 1.0e-3 (random) -> 4.2e-5, mpd PER at 19 dB 0.16 ->
    0.088; the reverse assignment fails outright. Others: random."""
    rng = np.random.default_rng(INTERLEAVER_SEED + spec.index)
    n = spec.coded_bits
    if spec.code != "ldpc" or spec.constellation.startswith("fsk"):
        return rng.permutation(n)  # (M-FSK: no labels to rank, data2g.cpm)
    code = ldpc_code(spec)
    deg = np.bincount(code.edges[1], minlength=code.n_cols)[code.sent]
    order = np.lexsort((rng.random(n), -deg))  # highest degree first
    pts = constellation.load(spec.constellation)
    m = constellation.bits_per_symbol(pts)
    labels = np.argsort(label_reliability(pts, spec.k / n))[::-1]  # most reliable first
    perm = np.empty(n, dtype=np.int64)
    per = n // m
    for g, lab in enumerate(labels):
        perm[rng.permutation(np.arange(lab, n, m))] = order[g * per : (g + 1) * per]
    return perm


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
def flip(spec: SubmodeSpec, index: int, rv: int = 0) -> np.ndarray:
    """(coded_bits,) +-1 in mapping order: soft bits of a codeword sent at
    burst position `index` and RV `rv`, times this, are those of the same
    codeword unscrambled (the code is linear: C(u ^ s) = C(u) ^ C(s)).
    Resends in other slots combine that way, and decode with index PLAIN."""
    return 1.0 - 2.0 * encode_info(spec, scrambler(spec.k, scramble_seed(index))[None], rv)[0]


@lru_cache(maxsize=None)
def scrambler(k: int, seed: int = 0x1FF) -> np.ndarray:
    """PN9 (x^9 + x^5 + 1, output = feedback bit) from `seed`, k bits. Info
    bits are XORed with it before encoding: an unscrambled zero-padded
    payload (control codewords, a stream's last codeword) codes to ~10%
    ones, piles OFDM symbols onto a few points, and the clipper wrecks them
    (phase G: 3-codeword bursts failed their control at 8 dB AWGN, every
    time). Seeds differ per codeword: identical payloads in one burst,
    scrambled alike, still decoded half the time at 8 dB."""
    reg, out = seed, np.empty(k, np.uint8)
    for i in range(k):
        bit = ((reg >> 8) ^ (reg >> 4)) & 1
        out[i] = bit
        reg = ((reg << 1) | bit) & 0x1FF
    return out


def info_bits(spec: SubmodeSpec, payload: bytes, crc_mask: int = 0, index: int = 0) -> np.ndarray:
    """payload -> the k info bits the encoder takes: payload, masked CRC,
    zero fill, scrambled."""
    if len(payload) != payload_bytes(spec):
        raise ValueError(f"{spec.name} carries {payload_bytes(spec)} bytes, got {len(payload)}")
    bits = np.unpackbits(np.frombuffer(_with_crc(payload, crc_bits(spec), crc_mask), np.uint8))
    return np.pad(bits, (0, spec.k - len(bits))) ^ scrambler(spec.k, scramble_seed(index))


def encode(spec: SubmodeSpec, payload: bytes, rv: int = 0, crc_mask: int = 0, index: int = 0) -> np.ndarray:
    """One codeword's payload -> (coded_bits,) array of 0/1, interleaved.
    `index`: its position in the burst (the scrambler seed)."""
    return encode_info(spec, info_bits(spec, payload, crc_mask, index)[None], rv)[0]


def encode_info(spec: SubmodeSpec, bits: np.ndarray, rv: int = 0) -> np.ndarray:
    """(B, k) info bits (CRC included) -> (B, coded_bits), interleaved.
    `rv`: redundancy version (rv_positions); 0 is the first transmission."""
    if rv and spec.code == "ldpc":
        coded = ldpc_code(spec).mother().encode(bits)[:, rv_positions(spec, rv)]
    else:
        code = ldpc_code(spec) if spec.code == "ldpc" else polar_code(spec)
        coded = code.encode(bits)
    return coded[:, interleaver(spec)]


# --- incremental redundancy (docs/arq.md §5) -----------------------------------------
# LDPC: RV r sends positions [r n, (r + 1) n) of the mother code's circular
# buffer (every base row: rate ~1/5), wrapping. RV 0 is the codeword as
# frozen; RV 1 onwards is fresh parity until the buffer runs out, then
# repeats (Chase). Polar: every RV is the same codeword (Chase).
# Soft bits accumulate over the buffer in code order ("buffer": (B, L)).

def rv_cycle(spec: SubmodeSpec) -> int:
    """Distinct redundancy versions: resend r uses RV r mod this."""
    return 4 if spec.code == "ldpc" else 1


def buffer_len(spec: SubmodeSpec) -> int:
    return ldpc_code(spec).mother().n if spec.code == "ldpc" else spec.coded_bits


def rv_positions(spec: SubmodeSpec, rv: int) -> np.ndarray:
    """Buffer positions RV `rv` carries, in code order."""
    n = spec.coded_bits
    if spec.code != "ldpc":
        return np.arange(n)
    return (rv % rv_cycle(spec) * n + np.arange(n)) % buffer_len(spec)


def combine(spec: SubmodeSpec, buf: np.ndarray | None, soft: np.ndarray, rvs) -> np.ndarray:
    """Add (B, coded_bits) soft bits in mapping order, sent at RVs `rvs`,
    into the (B, L) buffer (None: a fresh one). -> the buffer."""
    soft = np.atleast_2d(soft)
    if buf is None:
        buf = np.zeros((len(soft), buffer_len(spec)))
    deint = np.empty_like(soft, dtype=float)
    deint[:, interleaver(spec)] = soft
    for b, rv in enumerate(np.broadcast_to(rvs, len(soft))):
        np.add.at(buf[b], rv_positions(spec, int(rv)), deint[b])
    return buf


def decode_buffer(spec: SubmodeSpec, buf: np.ndarray, max_rv: int = 0, crc_mask=0, index=None) -> list[tuple[bytes, bool]]:
    """(B, L) combined buffers, with RVs up to `max_rv` received -> [(payload,
    crc_ok)]: LDPC decodes the mother code cut to the buffer's received
    extent (unreceived parity would only slow it)."""
    masks = np.broadcast_to(crc_mask, len(buf))
    idx = np.arange(len(buf)) if index is None else np.broadcast_to(index, len(buf))
    if spec.code != "ldpc":
        return _payloads(spec, *_decode_code_order(spec, _decoder(spec), buf, crc_mask=masks, index=idx),
                         masks, idx)
    extent = min(buffer_len(spec), (min(max_rv, rv_cycle(spec) - 1) + 1) * spec.coded_bits)
    return _payloads(spec, *_decode_code_order(spec, _ext_decoder(spec, extent), buf[:, :extent]), masks, idx)


@lru_cache(maxsize=None)
def _ext_decoder(spec: SubmodeSpec, extent: int):
    return ldpc.MinSumDecoder(ldpc_code(spec).mother(extent))


def _payloads(spec: SubmodeSpec, bits: np.ndarray, converged, masks=None, index=None) -> list[tuple[bytes, bool]]:
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


def decode(spec: SubmodeSpec, soft: np.ndarray) -> tuple[bytes, bool]:
    """(coded_bits,) soft values in mapping order -> (payload, crc_ok)."""
    return decode_many(spec, soft[None])[0]


def decode_many(spec: SubmodeSpec, soft: np.ndarray, crc_mask=0, index=None) -> list[tuple[bytes, bool]]:
    """(B, coded_bits) -> [(payload, crc_ok)] in one batched decode: 64
    polar codewords take 0.04-0.08 s batched, 1.6-3.3 s one at a time
    (scripts/decode_latency.py). `crc_mask`, `index`: per row, the CRC mask
    and burst position it was sent with (default 0 and the row number)."""
    masks = np.broadcast_to(crc_mask, len(soft))
    idx = np.arange(len(soft)) if index is None else np.broadcast_to(index, len(soft))
    return _payloads(spec, *decode_llrs(spec, soft, crc_mask=masks, index=idx), masks, idx)


def decode_llrs(spec: SubmodeSpec, llr, iters: int = 40, device=None, crc_mask=0, index=0):
    """Batched decode: (B, coded_bits) LLRs in mapping order (numpy, or
    torch with `device`) -> (info bits (B, k) numpy uint8, success (B,)
    numpy bool). LDPC: success = H satisfied. Polar: the first list path
    (best metric first) whose CRC checks; success = one did. `device`
    (studies only): decode on torch there (decoders_torch)."""
    if device is not None:
        import torch

        llr = torch.as_tensor(llr, dtype=torch.float32, device=device)
        deint = torch.empty_like(llr)
        deint[:, torch.as_tensor(interleaver(spec), device=device)] = llr
    else:
        llr = np.asarray(llr, dtype=np.float32)
        deint = np.empty_like(llr)
        deint[:, interleaver(spec)] = llr
    return _decode_code_order(spec, _decoder(spec, device), deint, iters, crc_mask, index)


def decode_raw(spec: SubmodeSpec, soft: np.ndarray, index=None) -> tuple[np.ndarray, np.ndarray]:
    """(B, coded_bits) soft bits in mapping order -> (candidates (B, L, k)
    uint8, descrambled; usable (B, L) bool), with the CRC mask left open:
    decode once, then check() each mask in question. LDPC: one candidate,
    usable when H is satisfied. Polar: the list, best metric first, all
    usable (the CRC picks). `index`: per row, the burst position (default
    the row number)."""
    soft = np.asarray(soft, dtype=np.float32)
    deint = np.empty_like(soft)
    deint[:, interleaver(spec)] = soft
    if spec.code == "ldpc":
        out, ok = _decoder(spec).decode(deint, iters=40)
        cands, usable = _numpy(out)[:, None], _numpy(ok)[:, None]
    else:
        cands = _numpy(_decoder(spec).decode(deint)[0])
        usable = np.ones(cands.shape[:2], bool)
    idx = np.arange(len(soft)) if index is None else np.broadcast_to(index, len(soft))
    return np.stack([descramble(spec, c, i) for c, i in zip(cands, idx)]), usable


def descramble(spec: SubmodeSpec, bits: np.ndarray, index: int) -> np.ndarray:
    """(..., k) decoded info bits sent at burst position `index` -> unscrambled."""
    return bits.astype(np.uint8) ^ scrambler(spec.k, scramble_seed(int(index)))[: bits.shape[-1]]


def check(spec: SubmodeSpec, cands: np.ndarray, usable: np.ndarray, crc_mask: int) -> bytes | None:
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


def _numpy(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _decode_code_order(spec: SubmodeSpec, dec, deint, iters: int = 40, crc_mask=0, index=0):
    """decode_llrs after deinterleaving: LLRs in the decoder's code order."""
    if spec.code == "ldpc":
        out, ok = dec.decode(deint, iters=iters)
        return _numpy(out), _numpy(ok)
    paths, _ = dec.decode(deint)
    paths = _numpy(paths)  # (B, L, k), best first
    rep = lambda v: np.repeat(np.broadcast_to(v, len(paths)), paths.shape[1])  # noqa: E731
    crc = crc_ok(spec, paths.reshape(-1, spec.k), rep(crc_mask), rep(index)).reshape(paths.shape[:2])
    pick = np.where(crc.any(axis=1), crc.argmax(axis=1), 0)
    return paths[np.arange(len(paths)), pick], crc.any(axis=1)


def crc_ok(spec: SubmodeSpec, bits: np.ndarray, crc_mask=0, index=0) -> np.ndarray:
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
def _decoder(spec: SubmodeSpec, device=None):
    """The numpy decoder, or with `device` the torch one there (studies)."""
    if device is not None:
        from . import decoders_torch

        if spec.code == "ldpc":
            return decoders_torch.MinSumDecoder(ldpc_code(spec), device=device)
        return decoders_torch.SCLDecoder(polar_code(spec), POLAR_LIST, device=device)
    if spec.code == "ldpc":
        return ldpc.MinSumDecoder(ldpc_code(spec))
    return polar.SCLDecoder(polar_code(spec), POLAR_LIST)
