"""ARQ frame codecs (docs/arq.md §4, §5, §7, §9a): the 32-bit core
control word, extension TLVs, callsign packing and stream records."""

import zlib
from dataclasses import dataclass, field
from pathlib import Path

SEQ_BITS = 7
SEQ_MOD = 1 << SEQ_BITS
BITMAP_BITS = 64  # the bitmap extension's width (MAX_CODEWORDS)
# < SEQ_MOD / 2 as selective repeat needs: a cumulative ACK can land WINDOW
# past the sender's base, and at SEQ_MOD / 2 it aliases (base - 64 in 7
# bits: tests/test_arq.py test_full_window_bursts)
WINDOW = SEQ_MOD // 2 - 1
BURST_MOD = 8

# frame types (core word)
# ARQ_DUP: an ARQ burst whose control codewords each go twice, RV 0 then
# RV 1 (the receiver combines the pair if the first fails); the peer asks
# for it with T_DUPCTL when it predicts this burst's control is at risk
ARQ, SESSION, PROBE, ARQ_DUP = 0, 1, 2, 3
# extension types
T_PAD, T_NEW, T_ABANDON, T_RV, T_BITMAP, T_RESYNC, T_REPORT, T_SURVEY, T_SOUND, T_BUFFER = range(10)
T_REPLY = 11  # recommended submode for the peer's control-only bursts (10 is session.T_SESS)
T_CHAT = 12  # empty: the sender's host has CHAT ON (latency over throughput, docs/arq.md §9)
CHAT_LINE_BYTES = 200  # CHAT ON plans for at least a chat line; T_BUFFER is sent only above it
T_DUPCTL = 14  # empty: "duplicate your control codewords" (ARQ_DUP), from the burst's receiver
# bit per data slot (resends, then new), MSB first, cut after the last set
# byte, sent when any is set: that codeword is raw deflate. A resend's bit
# is left 0 once the peer has its flag from the codeword's first burst
# (docs/arq.md §9a)
T_COMP = 15
HIST = 4096  # delivered stream bytes a compressed codeword's deflate is primed with
# ~4 KB of common English words, commonest last (nearest), primed ahead of the
# history so short streams compress too (wordfreq top list, session version 3)
WORDS = (Path(__file__).parent / "words.txt").read_bytes()
MAX_INFLATE = 1 << 16  # bytes one compressed codeword may inflate to
T_CQ = 13  # packed callsign + bandwidth cap code: a CQ frame (VARA's CQFRAME), no session
# session control subtypes (in a SESSION frame's first extension byte)
CONNECT, CONNECT_ACK, CONNECT_NAK, DISC, DISC_ACK = range(1, 6)

BANDS_CODE = {"w": 0, "n10": 1, "w48": 2}  # sync bands, 2 bits


@dataclass
class Core:
    ftype: int = ARQ  # 2 bits
    n_ctl: int = 1  # 1-4 control codewords
    burst_seq: int = 0  # 3 bits
    acted_on: int = 0  # 3 bits: peer burst seq whose ACK this burst acts on
    cum: int = 0  # 7 bits: next peer seq this station expects
    reply_lost: bool = False
    k: int = 0  # 6 bits: resends in this burst
    recommend: int = 0  # 6 bits: sync band (2) | index (4)
    size_hint: int = 1  # 2 bits: shrink / hold / grow / max

    def pack(self) -> bytes:
        assert 1 <= self.n_ctl <= 4 and 0 <= self.k < 64
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
        if n > 4:
            raise ValueError(f"control needs {n} codewords of {payload_bytes} B (max 4)")
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
    c = zlib.compressobj(9, zlib.DEFLATED, -15, 9, zdict=WORDS + hist)
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
    d = zlib.decompressobj(-15, zdict=WORDS + hist)
    try:
        out = d.decompress(payload, MAX_INFLATE)
    except zlib.error as e:
        raise ValueError(f"inflate: {e}") from None
    if not d.eof:
        raise ValueError("inflate: truncated or over MAX_INFLATE")
    return out


# --- callsigns ----------------------------------------------------------------

CALL_ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-"  # 6 bits, space = end
CALL_CHARS = 10


def pack_call(call: str) -> bytes:
    call = call.upper()
    if len(call) > CALL_CHARS or any(c not in CALL_ALPHABET[1:] for c in call):
        raise ValueError(f"callsign {call!r}: up to {CALL_CHARS} of A-Z 0-9 / -")
    v = 0
    for c in call.ljust(CALL_CHARS):
        v = (v << 6) | CALL_ALPHABET.index(c)
    return v.to_bytes(CALL_CHARS * 6 // 8 + 1, "big")


def unpack_call(b: bytes) -> str:
    v = int.from_bytes(b, "big")
    chars = [CALL_ALPHABET[(v >> (6 * (CALL_CHARS - 1 - i))) & 63] for i in range(CALL_CHARS)]
    return "".join(chars).rstrip()


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
