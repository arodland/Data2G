"""ARQ frame codecs (docs/arq.md §4, §5, §7, §9a): the 32-bit core
control word, extension TLVs, callsign packing and stream records.

Native: arq/frames.hpp.
"""

import zlib
from dataclasses import dataclass, field
from pathlib import Path

from ..config import PROFILE

SEQ_BITS = 7
SEQ_MOD = 1 << SEQ_BITS
BITMAP_BITS = 64  # the bitmap extension's width (MAX_CODEWORDS)
# < SEQ_MOD / 2 as selective repeat needs: a cumulative ACK can land WINDOW
# past the sender's base, and at SEQ_MOD / 2 it aliases (base - 64 in 7
# bits: tests/test_arq.py test_full_window_bursts)
WINDOW = SEQ_MOD // 2 - 1
BURST_MOD = 8
MAX_CTL = 4  # control codewords a burst may have: the core's 2-bit n_ctl

# frame types (core word)
# ARQ_DUP: an ARQ burst whose control codewords each go twice, RV 0 then
# RV 1 (the receiver combines the pair if the first fails); the peer asks
# for it with T_DUPCTL when it predicts this burst's control is at risk
ARQ, SESSION, PROBE, ARQ_DUP = 0, 1, 2, 3
# extension types (slots 5, 19, 20 retired at wire 1.0: T_RESYNC, T_NODS, T_KEEP).
# Forward-compatibility split: 1..127 ignorable if unknown; 128..255 must-understand (drop control if unknown).
T_PAD, T_NEW, T_ABANDON, T_RV, T_BITMAP = range(5)
T_REPORT, T_SURVEY, T_SOUND, T_BUFFER = range(6, 10)
T_SESS = 10  # a session frame's body (data2g.arq.session)
T_REPLY = 11  # recommended submode for the peer's control-only bursts (kept unconditionally across bursts)
T_CHAT = 12  # empty: the sender's host has CHAT ON (latency over throughput, docs/arq.md §9)
CHAT_LINE_BYTES = 200  # CHAT ON plans for at least a chat line; T_BUFFER is sent only above it
T_DUPCTL = 14  # empty: "duplicate your control codewords" (ARQ_DUP), from the burst's receiver
# bit per data slot (resends, then new), MSB first, cut after the last set
# byte, sent when any is set: that codeword is raw deflate. A resend's bit
# is left 0 once the peer has its flag from the codeword's first burst
# (docs/arq.md §9a)
T_COMP = 15
HIST = 4096  # delivered stream bytes a compressed codeword's deflate is primed with
# 4 KB primed ahead of the history so short streams compress too: built from
# C4 web text by scripts/build_zdict.py to share as many frequent substrings
# with it as it can (session version 3)
ZDICT = (Path(__file__).parent / "zdict.bin").read_bytes()
MAX_INFLATE = 1 << 16  # bytes one compressed codeword may inflate to
T_CQ = 13  # packed callsign + bandwidth cap code: a CQ frame (VARA's CQFRAME), no session
# packed callsign + the session key it identifies for: an ID frame (docs/arq.md §7a), mask 0
T_ID = 16
T_DRAINED = 17  # empty: this burst's new codewords end the sender's queued data (a hint: the peer may hold its reply)
# a data codeword carried in the control's padding (docs/arq.md §5a): seq (7 bits) | comp << 7, its
# payload with trailing zeros cut (zero padded back to the data mode's payload), then inline_check.
# Only the last new codeword of a burst, and only where it fits the padding.
T_INLINE = 18
# FM only (arq.ping): a ping or its reply, a measurement burst for the TX path and level check (FM plan M7)
T_PING = 21
# Extended mode recommendation (1-2 B big-endian) when Core.recommend == modes.REC_EXT (63)
T_REC_EXT = 22
# Extended capability/constraint TLV reserved for CAP_EXT (bit 15 of caps)
T_CAPS = 23

TLV_CRITICAL = 128  # 1..127 ignorable if unknown; 128..255 must-understand (Control.unpack rejects unknown)
KNOWN_CRITICAL_TLVS: frozenset[int] = frozenset()

# Capability / constraint bitmasks exchanged in CONNECT and CONNECT_ACK (wire 1.0):
# caps (16 bits): permissive/ignorable abilities; req (8 bits): mandatory peer constraints.
CAP_NODS = 1 << 0   # this station listens for nods (docs/arq.md §3b; cleared by --no-nods)
CAP_EXT = 1 << 15   # escape hatch: extended capability/constraint TLV (T_CAPS) follows
DEFAULT_CAPS = CAP_NODS
SUPPORTED_REQ = 0   # mandatory peer-constraint bits this version knows how to honor


def inline_check(mask_id: tuple, payload: bytes) -> bytes:
    """T_INLINE's 2-byte check: CRC-32 of the codeword's identity (burst.data_mask:
    key, direction with compression and abandon epoch, seq) and its padded
    payload, low 16 bits. As a data slot's masked CRC (§2), a wrong seq,
    compression or slicing fails it, and so does a control that passed its
    own CRC corrupted."""
    key, d, s = mask_id
    return (zlib.crc32(bytes([key >> 8, key & 255, d, s]) + payload) & 0xFFFF).to_bytes(2, "big")
# session control subtypes (in a SESSION frame's first extension byte)
CONNECT, CONNECT_ACK, CONNECT_NAK, DISC, DISC_ACK = range(1, 6)

# sync bands, 2 bits. FM (plan M6, Q2): w and n10 as the floor and for ACKs,
# f55 and f108 its own ladders; no w48 (f55 replaces it), n4 or CPM, so
# every code names a band
BANDS_CODE = {"w": 0, "n10": 1, "f55": 2, "f108": 3} if PROFILE == "fm" else {"w": 0, "n10": 1, "w48": 2}


@dataclass
class Core:
    ftype: int = ARQ  # 2 bits
    n_ctl: int = 1  # 1-MAX_CTL control codewords
    burst_seq: int = 0  # 3 bits
    acted_on: int = 0  # 3 bits: peer burst seq whose ACK this burst acts on
    cum: int = 0  # 7 bits: next peer seq this station expects
    reply_lost: bool = False
    k: int = 0  # 6 bits: resends in this burst
    recommend: int = 0  # 6 bits: sync band (2) | index (4)
    size_hint: int = 1  # 2 bits: shrink / hold / grow / max

    def pack(self) -> bytes:
        assert 1 <= self.n_ctl <= MAX_CTL and 0 <= self.k < 64
        v = self.ftype
        for val, bits in ((self.n_ctl - 1, 2), (self.burst_seq, 3), (self.acted_on, 3), (self.cum, 7),
                          (int(self.reply_lost), 1), (self.k, 6), (self.recommend, 6), (self.size_hint, 2)):
            assert 0 <= val < (1 << bits), (val, bits)
            v = (v << bits) | val
        return v.to_bytes(4, "big")

    @classmethod
    def unpack(cls, b: bytes) -> "Core":
        v = int.from_bytes(b[:4], "big")
        fields = []
        for bits in (2, 6, 6, 1, 7, 3, 3, 2):  # from the least significant end
            fields.append(v & ((1 << bits) - 1))
            v >>= bits
        size_hint, recommend, k, reply_lost, cum, acted_on, burst_seq, n_ctl = fields
        return cls(ftype=v, n_ctl=n_ctl + 1, burst_seq=burst_seq, acted_on=acted_on, cum=cum,
                   reply_lost=bool(reply_lost), k=k, recommend=recommend, size_hint=size_hint)


@dataclass
class Control:
    """A burst's whole control content: core + extensions."""
    core: Core
    ext: dict = field(default_factory=dict)  # type -> bytes

    def pack(self, payload_bytes: int) -> list[bytes]:
        """-> control codeword payloads (the core's n_ctl set to fit)."""
        body = b"".join(bytes([t, len(v)]) + v for t, v in sorted(self.ext.items()))
        total = 4 + len(body)
        n = max(1, -(-total // payload_bytes))
        if n > MAX_CTL:
            raise ValueError(f"control needs {n} codewords of {payload_bytes} B (max {MAX_CTL})")
        self.core.n_ctl = n
        stream = self.core.pack() + body
        stream += bytes(n * payload_bytes - len(stream))
        return [stream[i * payload_bytes:(i + 1) * payload_bytes] for i in range(n)]

    @classmethod
    def unpack(cls, payloads: list[bytes]) -> "Control":
        stream = b"".join(payloads)
        core = Core.unpack(stream)
        ext, i = {}, 4
        while i + 2 <= len(stream):
            t, n = stream[i], stream[i + 1]
            if t == T_PAD:
                break
            if i + 2 + n > len(stream):
                raise ValueError("truncated extension")
            if t >= TLV_CRITICAL and t not in KNOWN_CRITICAL_TLVS:
                raise ValueError(f"unknown critical extension {t}")
            ext[t] = stream[i + 2:i + 2 + n]
            i += 2 + n
        return cls(core, ext)


# --- extension payloads -----------------------------------------------------

def pack_bitmap(received: set[int], cum: int) -> bytes:
    """Bit i (MSB first) = seq cum + 1 + i received (mod SEQ_MOD), cut
    after the last set byte (empty: nothing beyond cum received)."""
    v = 0
    for i in range(BITMAP_BITS):
        if (cum + 1 + i) % SEQ_MOD in received:
            v |= 1 << (BITMAP_BITS - 1 - i)
    return v.to_bytes(BITMAP_BITS // 8, "big").rstrip(b"\0")


def unpack_bitmap(b: bytes, cum: int) -> set[int]:
    v = int.from_bytes(b.ljust(BITMAP_BITS // 8, b"\0"), "big")
    return {(cum + 1 + i) % SEQ_MOD for i in range(BITMAP_BITS) if v >> (BITMAP_BITS - 1 - i) & 1}


def pack_rv(rvs: list[int]) -> bytes:
    v = 0
    for r in rvs:
        v = (v << 2) | (r & 3)
    n = -(-2 * len(rvs) // 8)
    return (v << (8 * n - 2 * len(rvs))).to_bytes(n, "big") if rvs else b""


def unpack_rv(b: bytes, k: int) -> list[int]:
    v = int.from_bytes(b, "big")
    total = 8 * len(b)
    return [(v >> (total - 2 * (i + 1))) & 3 for i in range(k)]


def pack_flags(flags: list[bool]) -> bytes:
    out = bytearray(-(-len(flags) // 8))
    for i, f in enumerate(flags):
        if f:
            out[i // 8] |= 0x80 >> i % 8
    return bytes(out).rstrip(b"\0")


def unpack_flags(b: bytes, n: int) -> list[bool]:
    return [i < 8 * len(b) and bool(b[i // 8] >> (7 - i % 8) & 1) for i in range(n)]


# --- compression (docs/arq.md §9a) ------------------------------------------------

def deflate(hist: bytes, data: bytes) -> bytes:
    c = zlib.compressobj(9, zlib.DEFLATED, -15, 9, zdict=ZDICT + hist)
    return c.compress(data) + c.flush()


def deflate_fit(hist: bytes, data: bytes, pb: int) -> tuple[int, bytes] | None:
    """The longest prefix of `data` past `pb` bytes whose deflate (primed
    with `hist`) fits `pb` bytes -> (its length, deflated), or None: a
    raw codeword carries as much. Binary search: deflated length is
    near enough monotone in the input's."""
    lo, hi = pb + 1, min(len(data), 16 * pb)
    if lo > hi or len(z := deflate(hist, data[:lo])) > pb:
        return None  # incompressible: one trial
    best, lo = (lo, z), lo + 1
    while lo <= hi:
        m = (lo + hi) // 2
        if len(z := deflate(hist, data[:m])) <= pb:
            best, lo = (m, z), m + 1
        else:
            hi = m - 1
    return best


def inflate(hist: bytes, payload: bytes) -> bytes:
    """A compressed codeword (zero padded) -> its stream bytes."""
    d = zlib.decompressobj(-15, zdict=ZDICT + hist)
    try:
        out = d.decompress(payload, MAX_INFLATE)
    except zlib.error as e:
        raise ValueError(f"inflate: {e}") from None
    if not d.eof:
        raise ValueError("inflate: truncated or over MAX_INFLATE")
    return out


# --- callsigns ----------------------------------------------------------------

# 6 bits; space pads at the end, so a name may hold spaces ("VARA KISS") but
# trailing ones don't survive the round trip
CALL_ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-"
CALL_CHARS = 10


def pack_call(call: str) -> bytes:
    call = call.upper()
    if len(call) > CALL_CHARS or any(c not in CALL_ALPHABET for c in call):
        raise ValueError(f"callsign {call!r}: up to {CALL_CHARS} of A-Z 0-9 / - space")
    v = 0
    for c in call.ljust(CALL_CHARS):
        v = (v << 6) | CALL_ALPHABET.index(c)
    return v.to_bytes(CALL_CHARS * 6 // 8 + 1, "big")


def unpack_call(b: bytes) -> str:
    v = int.from_bytes(b, "big")
    codes = [(v >> (6 * (CALL_CHARS - 1 - i))) & 63 for i in range(CALL_CHARS)]
    if max(codes) >= len(CALL_ALPHABET):
        raise ValueError(f"callsign code {max(codes)}")
    return "".join(CALL_ALPHABET[c] for c in codes).rstrip()


CALL_BASE = 40  # base-40 packs 10 chars of CALL_ALPHABET (39 chars) in 54 bits (40^10 < 2^54)
CALL_B40_MAX = CALL_BASE ** CALL_CHARS


def pack_call_b40(call: str) -> int:
    """Callsign (up to 10 chars of CALL_ALPHABET) -> 54-bit base-40 integer."""
    call = call.upper()
    if len(call) > CALL_CHARS or any(c not in CALL_ALPHABET for c in call):
        raise ValueError(f"callsign {call!r}: up to {CALL_CHARS} of A-Z 0-9 / - space")
    v = 0
    for c in call.ljust(CALL_CHARS):
        v = v * CALL_BASE + CALL_ALPHABET.index(c)
    return v


def unpack_call_b40(v: int) -> str:
    """54-bit base-40 integer -> callsign (ValueError if out of range or invalid digit)."""
    if v < 0 or v >= CALL_B40_MAX:
        raise ValueError(f"callsign code {v}")
    chars = []
    for _ in range(CALL_CHARS):
        v, d = divmod(v, CALL_BASE)
        if d >= len(CALL_ALPHABET):
            raise ValueError(f"callsign code {d}")
        chars.append(CALL_ALPHABET[d])
    return "".join(reversed(chars)).rstrip()


# --- session frame bodies (docs/arq.md §7, §7a) ------------------------------------
# CONNECT uses a single unified 20-byte bit-packed payload on all modes (OFDM
# and CPM) under burst.COMPACT_CONNECT: no Core or T_SESS wrapper.
# CONNECT_ACK, CONNECT_NAK, DISC, DISC_ACK are T_SESS bodies led by their
# subtype byte; DISC and DISC_ACK are the subtype alone. T_CQ and T_ID bodies
# have no subtype.

COMPACT_BYTES = 20
# Bit widths of the 160-bit (20 B) CONNECT wire payload, MSB first:
# version(4) | caller(54) | callee(54) | nonce(16) | cap(2) | t_turn10(6) | caps(16) | req(8)
_CONNECT_FIELDS = (("version", 4), ("caller", 54), ("callee", 54), ("nonce", 16),
                   ("cap", 2), ("t_turn10", 6), ("caps", 16), ("req", 8))


@dataclass(frozen=True)
class Connect:
    """Unified 20-byte (160-bit) CONNECT payload on all modes:
    version (4 b, wire major version), caller and callee (54 b base-40 each),
    nonce (16 b), bandwidth cap code (2 b), t_turn in tenths (6 b),
    caps (16 b, permissive), req (8 b, mandatory peer constraints)."""
    version: int
    caller: str
    callee: str
    nonce: int
    cap: int
    t_turn10: int
    caps: int = 0
    req: int = 0
    BYTES = COMPACT_BYTES

    def pack(self) -> bytes:
        vals = (
            (self.version, 4), (pack_call_b40(self.caller), 54), (pack_call_b40(self.callee), 54),
            (self.nonce, 16), (self.cap, 2), (self.t_turn10, 6), (self.caps, 16), (self.req, 8),
        )
        v = 0
        for x, bits in vals:
            if not 0 <= x < (1 << bits):
                raise ValueError(f"CONNECT field {x} over {bits} bits")
            v = (v << bits) | x
        return v.to_bytes(COMPACT_BYTES, "big")

    @classmethod
    def unpack(cls, p: bytes) -> "Connect":
        """A payload of at least BYTES (20 B); ValueError if too short or callsigns invalid."""
        if len(p) < cls.BYTES:
            raise ValueError(f"CONNECT of {len(p)} B")
        v = int.from_bytes(p[:cls.BYTES], "big")
        fields = []
        for _, bits in reversed(_CONNECT_FIELDS):
            fields.append(v & ((1 << bits) - 1))
            v >>= bits
        req, caps, t_turn10, cap, nonce, callee_i, caller_i, version = fields
        return cls(version, unpack_call_b40(caller_i), unpack_call_b40(callee_i),
                   nonce, cap, t_turn10, caps, req)

    @classmethod
    def calls(cls, p: bytes) -> tuple[str, str]:
        """(caller, callee) from a 20-byte CONNECT payload."""
        c = cls.unpack(p)
        return c.caller, c.callee

    @staticmethod
    def mask(fields: set[str] | frozenset[str]) -> bytes:
        """20-byte bitmask with 1-bits on the named CONNECT fields (for primed reception)."""
        v = 0
        for name, bits in _CONNECT_FIELDS:
            v = (v << bits) | (((1 << bits) - 1) if name in fields else 0)
        return v.to_bytes(COMPACT_BYTES, "big")


@dataclass(frozen=True)
class ConnectAck:
    """subtype, CONNECT's nonce (2 B), agreed cap code, callee's t_turn in tenths,
    caps (2 B), req (1 B)."""
    nonce: int
    cap: int
    t_turn10: int
    caps: int = 0
    req: int = 0
    BYTES = 8

    def pack(self) -> bytes:
        return (bytes([CONNECT_ACK]) + self.nonce.to_bytes(2, "big") + bytes([self.cap, self.t_turn10])
                + self.caps.to_bytes(2, "big") + bytes([self.req]))

    @classmethod
    def unpack(cls, body: bytes) -> "ConnectAck":
        if len(body) < cls.BYTES:
            raise ValueError(f"CONNECT_ACK of {len(body)} B")
        return cls(int.from_bytes(body[1:3], "big"), body[3], body[4],
                   int.from_bytes(body[5:7], "big"), body[7])


@dataclass(frozen=True)
class ConnectNak:
    """subtype, CONNECT's nonce (2 B), reason (session.NAK_*), detail (1 B:
    [min_major:4 | max_major:4] for NAK_VERSION, unsupported req mask for NAK_CAPS)."""
    nonce: int
    reason: int
    detail: int = 0
    BYTES = 5

    def pack(self) -> bytes:
        return bytes([CONNECT_NAK]) + self.nonce.to_bytes(2, "big") + bytes([self.reason, self.detail])

    @classmethod
    def unpack(cls, body: bytes) -> "ConnectNak":
        if len(body) < cls.BYTES:
            raise ValueError(f"CONNECT_NAK of {len(body)} B")
        return cls(int.from_bytes(body[1:3], "big"), body[3], body[4])


def pack_cq(call: str, cap: int) -> bytes:
    """A T_CQ body: the call and its bandwidth cap code."""
    return pack_call(call) + bytes([cap])


def unpack_cq(body: bytes) -> tuple[str, int]:
    if len(body) < 9:
        raise ValueError(f"CQ of {len(body)} B")
    return unpack_call(body[:8]), body[8]


def pack_id(call: str, key: int) -> bytes:
    """A T_ID body: the call and the session key it identifies for."""
    return pack_call(call) + key.to_bytes(2, "big")


def unpack_id(body: bytes) -> tuple[str, int]:
    if len(body) < 10:
        raise ValueError(f"ID of {len(body)} B")
    return unpack_call(body[:8]), int.from_bytes(body[8:10], "big")


# --- stream records -------------------------------------------------------------

def to_records(data: bytes) -> bytes:
    """Host bytes -> [len 1..255][bytes] records."""
    out = bytearray()
    for i in range(0, len(data), 255):
        chunk = data[i:i + 255]
        out += bytes([len(chunk)]) + chunk
    return bytes(out)


class RecordReader:
    """Delivered stream bytes (records and zero padding) -> host bytes."""

    def __init__(self):
        self.buf = bytearray()
        self.delivered = 0  # host bytes out, for the log's throughput

    def feed(self, b: bytes) -> bytes:
        self.buf += b
        out = bytearray()
        i = 0
        while i < len(self.buf):
            n = self.buf[i]
            if n == 0:
                i += 1
                continue
            if i + 1 + n > len(self.buf):
                break
            self.delivered += n
            out += self.buf[i + 1:i + 1 + n]
            i += 1 + n
        del self.buf[:i]
        return bytes(out)
