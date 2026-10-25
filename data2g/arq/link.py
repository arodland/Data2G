"""ARQ link core (docs/arq.md §3-§6, §9a, §10): one station's data
transfer state for both directions, independent of the PHY.

Station.build() -> the next TxBurst; Station.handle(rx) takes a received
burst (burst.RxBurst); on_timeout() is the master's retry. The mode
choice comes from a Policy (declared here; policy.GearShifter on air).

Accounting rules, all from docs/arq.md §10 (the failure to design out is
two stations that hear each other fine and exchange bursts forever
because they disagree about state):

- Seqs are unbounded ints here; 7 bits on the wire, unwrapped against a
  known anchor (the receiver's cumulative, the sender's base).
- Every data codeword's CRC mask includes its direction, seq,
  compression and abandon epoch (burst.data_mask, the mask_id an
  RxBurst decodes under), so a slot mapped
  to the wrong seq, taken under the wrong T_COMP flag or from an older
  slicing fails its CRC: accounting errors can cost progress, never
  correctness.
- A control that passes its CRC but is malformed is dropped like one
  that failed (Station._check), before any state changes.
- The resend list is a pure function of one snapshot, the reply the
  sender acted on, which both sides hold (Station._snapshots).
- RVs and the new-codeword start seq are explicit (extensions).
- A burst whose control codewords do not all decode is discarded
  whole and not answered.
- Watchdog: turns with the peer's control decoded but no progress while
  this station has data outstanding lead to resync (abandon from the
  base, drop soft bits); resyncs without progress to FAILED.

Native: arq/link.hpp (Station, TxSide, RxSide).
"""

import logging
import os
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field

from . import frames as F
# the PHY abstraction; named here too, for link's callers
from .burst import COMPACT_CONNECT, EPOCH_MOD, RxBurst, Slot, TxBurst, ctl_mask, data_mask, dup_ctl  # noqa: F401
from .frames import SEQ_MOD, WINDOW

NO_PROGRESS_TURNS = 8
RESYNCS_BEFORE_FAIL = 3
REPEATS_BEFORE_SHRINK = 1
MAX_ESCALATION = 4
FLOOR_DECAY_TURNS = 4  # clean turns (no escalation) that lower the escalation floor by one
# extensions a short control codeword (CPM) sheds first when it can't hold
# everything, least useful first; none of them is state the peer must agree on
OPTIONAL_TLVS = (F.T_BUFFER, F.T_CHAT, F.T_REPLY, F.T_DUPCTL)
LINK_LOST_MISSES = 12  # consecutive timeouts; the session layer adds its 90 s bound
# DATA2G_COMPRESS=0: send every codeword raw (for testing); compressed ones are still received
COMPRESS = os.environ.get("DATA2G_COMPRESS", "1") != "0"

ACTIVE, FAILED = "active", "failed"

log = logging.getLogger("data2g.link")


def unwrap(s7: int, anchor: int) -> int:
    """The absolute seq = s7 (mod SEQ_MOD) nearest `anchor`, in
    [anchor - SEQ_MOD/2, anchor + SEQ_MOD/2)."""
    d = (s7 - anchor) % SEQ_MOD
    if d >= SEQ_MOD // 2:
        d -= SEQ_MOD
    return anchor + d


# --- the policy ----------------------------------------------------------------------

class Policy:
    """Everything Station, Session and Engine ask of the mode choice:
    policy.GearShifter on air; tests and scripts/linksim.py subclass it with
    simpler ones. choose, payload_bytes and rv_cycle are required (Session
    also needs connect_mode and airtime); the rest default to having no
    opinion. Native: arq::Policy (link.hpp), whose has_*() say which
    defaults a policy keeps."""

    # -- link.Station

    def choose(self, station: "Station", escalation: int) -> tuple[str, int]:
        """-> (submode name, max codewords) for this station's next burst."""
        raise NotImplementedError

    def payload_bytes(self, submode: str) -> int:
        raise NotImplementedError

    def ctl_payload_bytes(self, submode: str) -> int:
        """A control codeword's payload (CPM carries control in a short one)."""
        return self.payload_bytes(submode)

    def max_ctl(self, submode: str) -> int:
        return F.MAX_CTL

    def rv_cycle(self, submode: str) -> int:
        """Distinct RVs the submode's code has (4 LDPC, 1 polar)."""
        raise NotImplementedError

    def recommend(self, station: "Station") -> tuple[int, int, int | None]:
        """-> (data mode, size hint, reply mode or None): what the peer
        should send me next (§8), as the core word's codes."""
        return 0, 1, None

    want_dup = False  # ask the peer to duplicate its next data burst's control (T_DUPCTL)

    def outcome(self, submode: str, decoded: int, sent: int, usable: bool | None = None):
        """After each received burst: its first-transmission codewords
        decoded of those sent, `usable`: its control decoded (the gear
        shifter's online correction)."""

    # for the log
    measured: dict | None = None  # the peer's last burst, as measured

    def mode_name(self, rec: int) -> str:
        """A recommendation code's mode."""
        return str(rec)

    def airtime(self, submode: str, n_cw: int, dup: bool = False) -> float | None:
        """A burst's seconds on air (`dup`: its control twice); None:
        unknown, left out of the log. Session needs it."""
        return None

    # -- session.Session

    def connect_mode(self, cap: int, tries: int = 0) -> str:
        """The mode of session frames at bandwidth cap `cap`, `tries` retries in."""
        raise NotImplementedError

    def reply_hold(self, station: "Station", burst: "TxBurst") -> float | None:
        """Seconds past t_turn to wait for a reply to `burst` (None:
        session.REPLY_START_S alone)."""
        return None

    def next_capacity(self, station: "Station") -> int | None:
        """Payload bytes the next data burst will carry (None: unknown; the
        session's write coalescing and the host's BUFFER)."""
        return None

    # -- engine.Engine

    def observe(self, measured: dict, submode: str, now: float):
        """The receiver's measurements of a peer burst."""

    def observe_energy(self, snr_db: float, now: float):
        """A peer burst's in-band SNR from its power alone, heard or not."""

    def expected_reply(self) -> tuple[str, float] | None:
        """The burst I asked the peer for last: its mode and seconds on air."""
        return None


# --- the two directions -----------------------------------------------------------

@dataclass
class Codeword:
    seq: int
    start: int  # stream offset of its first byte
    length: int
    submode: str
    payload: bytes
    heard: int = 0  # sends in bursts the peer is known to have decoded (sets the RV)
    comp: bool = False  # payload is deflate of the stream bytes (T_COMP)
    cstart: int = 0  # offset in the delivered stream (hist) where its bytes begin
    first_bn: int = -1  # my burst number that sent it new
    comp_known: bool = False  # the peer acted on first_bn: it holds the flag, resends omit it


class TxSide:
    """This station's data going out."""

    def __init__(self):
        self.buf = bytearray()  # stream bytes from self.buf_off on
        self.buf_off = 0
        self.stream_end = 0  # next stream byte not yet in a codeword
        self.cws: dict[int, Codeword] = {}  # outstanding (sent, not acked)
        self.base = 0  # first unacked seq
        self.next = 0  # next new seq
        self.ack: tuple[int, frozenset] | None = None  # (cum, received) acted on
        self.acked = 0  # stream bytes acked (host bytes and their record length bytes)
        # acked payload bytes, as sent and uncompressed (a raw codeword's padding counts in both)
        self.acked_wire = self.acked_plain = 0
        # the stream as the peer will deliver it (raw codewords with their
        # padding): compression's history, from offset hist_off on
        self.hist = bytearray()
        self.hist_off = 0
        # stream offsets where records start, from the one buf_off falls in
        # (clean() cuts only there), and each one's host offset
        self.recs: list[int] = []
        self.rec_host: list[int] = []
        self.written = 0  # host bytes written: the next one's host offset
        self.mark = 0  # stream offset of the first byte written after the last clean()
        # clean(): host offsets [from, to) on air, not acked, and to's stream offset
        self.limbo: tuple[int, int, int] | None = None
        # the limbo settled, for the host: host offsets (from, split, to), delivered before split
        self.cleaned: tuple[int, int, int] | None = None

    def write(self, data: bytes):
        r = F.to_records(data)
        self.recs += range(self._end(), self._end() + len(r), 256)
        self.rec_host += range(self.written, self.written + len(data), 255)
        self.written += len(data)
        self.buf += r

    def pending(self) -> bool:
        return bool(self.cws) or self.unsent() > 0

    def unsent(self) -> int:
        """Stream bytes written but in no codeword yet."""
        return self.buf_off + len(self.buf) - self.stream_end

    def unacked(self) -> int:
        """Stream bytes the peer has not acked, sent or not (buf holds
        them from the first unacked codeword on)."""
        return len(self.buf)

    def on_ack(self, cum: int, received: frozenset) -> bool:
        """-> whether the base advanced. cum must lie in [base, next]."""
        if not self.base <= cum <= self.next:
            raise ProtocolError(f"peer cumulative {cum} outside [{self.base}, {self.next}]")
        advanced = cum > self.base
        for s in range(self.base, cum):
            if c := self.cws.pop(s, None):
                self.acked += c.length
                self.acked_wire += len(c.payload)
                self.acked_plain += c.length if c.comp else len(c.payload)
        self.base = cum
        self.ack = (cum, frozenset(s for s in received if cum < s < self.next))
        keep = min((c.start for c in self.cws.values()), default=self.stream_end)
        del self.buf[: keep - self.buf_off]
        self.buf_off = keep
        n = max(0, bisect_right(self.recs, keep) - 1)
        del self.recs[:n], self.rec_host[:n]
        keep = min((c.cstart for c in self.cws.values()), default=self.hist_off + len(self.hist)) - F.HIST
        if keep > self.hist_off:
            del self.hist[: keep - self.hist_off]
            self.hist_off = keep
        if self.limbo is not None and self._limbo_start(self.limbo[2]) == self.limbo[2]:
            self._settle(self.limbo[2])  # all of it acked
        return advanced

    def new_codeword(self, submode: str, pb: int, compress: bool, bn: int) -> Codeword:
        """The next `pb` bytes of stream as seq `next`, or (`compress`)
        more of it deflated into `pb` bytes when that carries more."""
        i = self.stream_end - self.buf_off
        n = min(pb, len(self.buf) - i)
        payload, comp = bytes(self.buf[i:i + n]) + bytes(pb - n), False
        if compress and (fit := F.deflate_fit(bytes(self.hist[-F.HIST:]), bytes(self.buf[i:i + 16 * pb]), pb)):
            n, z = fit
            payload, comp = z + bytes(pb - len(z)), True
        c = Codeword(self.next, self.stream_end, n, submode, payload, comp=comp,
                     cstart=self.hist_off + len(self.hist), first_bn=bn)
        self.hist += self.buf[i:i + n] if comp else payload
        self.cws[c.seq] = c
        self.next += 1
        self.stream_end += n
        return c

    def missing(self) -> list[int]:
        """Unacked seqs per the acted-on reply only, ascending (§4).
        None before any reply: there is no snapshot to map them by."""
        if self.ack is None:
            return []
        cum, received = self.ack
        return [s for s in range(cum, self.next) if s not in received]

    def abandon(self) -> int:
        """Re-slice from the base (§4 mode change, §10 resync). -> A.

        Forgets the acted-on reply: it describes the stream before the
        abandon, and resends mapped against it would disagree with the
        receiver's snapshot (caught by tests/test_arq.py: resends of the
        re-sliced codewords cited the pre-abandon reply). No resends until
        a reply to a post-abandon burst arrives, as before the first reply."""
        a = self.base
        lo = None if self.limbo is None else self._limbo_start(self.limbo[2])
        if a in self.cws:
            self.stream_end = self.cws[a].start
            del self.hist[self.cws[a].cstart - self.hist_off:]
        self.cws.clear()
        self.next = a
        self.ack = None
        if lo is not None:
            self._settle(lo)  # abandons are made against an exact ACK: nothing from the base on arrived
        return a

    def clean(self, upto: int | None = None) -> tuple[int, int, int, int] | None:
        """CLEANTXBUFFER: drop the bytes never on air, but finish a record
        begun on air (the peer would join the next to it); `upto`: only those
        before host offset upto. -> host offsets (dropped from, to, limbo
        from, to), or None (LATE: a byte at or past upto is on air or
        finishes a record that is). The limbo is on air, not acked: settled
        by the next ACK that covers it or the next abandon (Station abandons
        for it). Without one it is empty, at the drop."""
        cut = self._rec(self.stream_end)
        drop = self._hoff(cut)
        if upto is not None and upto < drop:
            return None
        drop_to = self.written if upto is None else min(upto, self.written)
        self._cut(cut, self._split(drop_to))
        if self.limbo is None and (lo := self._limbo_start(cut)) < cut:
            self.limbo = (self._hoff(lo), drop, cut)
        self.mark = cut
        lf, lt = self.limbo[:2] if self.limbo is not None else (drop, drop)
        return drop, drop_to, lf, lt

    def end_limbo(self) -> tuple[int, int, int] | None:
        """The session ended with the limbo unsettled. -> host offsets
        (from, split, to): delivered before split (records acked whole), the
        rest on air but unacked, fate unknown. None without a limbo."""
        if self.limbo is None:
            return None
        f, t, _ = self.limbo
        acked = self.cws[self.base].start if self.base in self.cws else self.stream_end
        self.limbo = None
        return f, max(f, self.rec_host[bisect_right(self.recs, acked) - 1]), t

    def queued(self, vara: bool = False) -> tuple[int, int]:
        """(unsent, unacked) host bytes, as unsent() and unacked() count stream
        bytes. `vara`: only those written after the last clean()."""
        lo = self.mark if vara else 0
        return self._host_from(max(self.stream_end, lo)), self._host_from(max(self.buf_off, lo))

    def _end(self) -> int:
        return self.buf_off + len(self.buf)

    def _host_from(self, x: int) -> int:
        """Host bytes from stream offset x to the end: the length bytes of records starting there left out."""
        return self._end() - x - (len(self.recs) - bisect_left(self.recs, x))

    def _rec(self, x: int) -> int:
        """The first record start at or after stream offset x (the stream's end if none)."""
        i = bisect_left(self.recs, x)
        return self.recs[i] if i < len(self.recs) else self._end()

    def _hoff(self, x: int) -> int:
        """The host offset of stream offset x, a record start or the stream's end."""
        i = bisect_left(self.recs, x)
        return self.rec_host[i] if i < len(self.recs) else self.written

    def _split(self, h: int) -> int:
        """The stream offset of host offset h, never on air: the record it
        falls in is split there (each part its own length byte)."""
        if h >= self.written:
            return self._end()
        i = bisect_right(self.rec_host, h) - 1
        if self.rec_host[i] == h:
            return self.recs[i]
        s, k = self.recs[i], h - self.rec_host[i]
        j = s - self.buf_off
        n = self.buf[j]
        self.buf[j:j + 1 + k] = bytes([k]) + self.buf[j + 1:j + 1 + k] + bytes([n - k])
        self.recs[i + 1:] = [s + 1 + k] + [r + 1 for r in self.recs[i + 1:]]
        self.rec_host.insert(i + 1, h)
        return s + 1 + k

    def _cut(self, lo: int, hi: int):
        """Delete whole records [lo, hi), all past stream_end (host offsets stay)."""
        del self.buf[lo - self.buf_off:hi - self.buf_off]
        i, j = bisect_left(self.recs, lo), bisect_left(self.recs, hi)
        self.recs[i:] = [r - (hi - lo) for r in self.recs[j:]]
        del self.rec_host[i:j]

    def _limbo_start(self, cut: int) -> int:
        """Where the unacked records before stream offset `cut` begin (cut: none)."""
        return min(self._rec(self.cws[self.base].start), cut) if self.base in self.cws else cut

    def _settle(self, lo: int):
        """The limbo from stream offset `lo` on never arrived: drop it."""
        f, t, cut = self.limbo
        split = self._hoff(lo) if lo < cut else t
        self._cut(lo, cut)
        self.mark -= cut - lo
        self.limbo, self.cleaned = None, (f, split, t)


class RxSide:
    """The peer's data coming in."""

    def __init__(self):
        self.cum = 0
        self.buf: dict[int, tuple[bytes, bool]] = {}  # held above cum: (payload, compressed)
        self.reader = F.RecordReader()
        self.out = bytearray()
        self.hist = b""  # the last HIST stream bytes delivered (compression's history)
        self.wire = self.plain = 0  # delivered payload bytes, as received and inflated
        # seqs flagged compressed by a control decoded here (its data slot may
        # have failed): a resend after the sender knows that omits the bit
        self.comp_seqs: set[int] = set()

    def accept(self, seq: int, payload: bytes, comp: bool = False) -> bool:
        """-> whether new bytes were delivered. `comp`: deflated (T_COMP),
        inflated in seq order, primed with the stream delivered before it."""
        if seq < self.cum or seq in self.buf or seq >= self.cum + WINDOW:
            return False
        self.buf[seq] = (payload, comp)
        delivered = False
        while self.cum in self.buf:
            p, z = self.buf.pop(self.cum)
            self.comp_seqs.discard(self.cum)
            self.wire += len(p)
            if z:
                try:
                    p = F.inflate(self.hist, p)
                except ValueError as e:
                    raise ProtocolError(f"seq {self.cum}: {e}") from None
            self.plain += len(p)
            self.hist = (self.hist + p)[-F.HIST:]
            self.out += self.reader.feed(p)
            self.cum += 1
            delivered = True
        return delivered

    def abandon(self, a: int) -> list[int]:
        """Drop what was buffered at or above max(a, cum). -> dropped seqs."""
        lo = max(a, self.cum)
        gone = [s for s in self.buf if s >= lo]
        for s in gone:
            del self.buf[s]
        self.comp_seqs = {s for s in self.comp_seqs if s < lo}
        return gone


class ProtocolError(Exception):
    pass


# --- one station ------------------------------------------------------------------

@dataclass
class Station:
    direction: int  # 0 caller, 1 callee: this station's data direction
    policy: Policy
    master: bool = False  # the caller: the only one that retries (§6)
    key: int = 0  # session key: every CRC mask includes it (§2)
    max_misses: int | None = LINK_LOST_MISSES  # None: the session's clock decides (90 s)
    cap: int = 2  # session bandwidth cap code (0: 500 Hz, 1: 1200, 2: none); for the policy
    last_rx_data: bool = False  # the peer's last handled burst carried data slots
    peer_recommend: int | None = None  # the peer's recommendation for my next burst (§8)
    peer_size_hint: int = 1
    chat: bool = False  # my host's CHAT ON
    peer_chat: bool = False  # the peer's, from its last burst
    peer_queued: int = 0  # bytes the peer had queued at its last burst (T_BUFFER, chat only)
    peer_drained: bool = False  # the peer's last burst ended its queued data (T_DRAINED)
    peer_wants_dup: bool = False  # the peer asked for duplicated control (T_DUPCTL)
    peer_reply_lost: bool = False  # the peer's last burst said its previous reply was lost
    peer_abandon: bool = False  # the peer's last burst carried T_ABANDON: it holds re-sliced data
    peer_reply_recommend: int | None = None  # ... for my control-only bursts
    tx: TxSide = field(default_factory=TxSide)
    stats: Counter = field(default_factory=Counter)  # for the log: cw_new, cw_comp, cw_resend, rx_ok, rx_lost, timeouts
    rx: RxSide = field(default_factory=RxSide)
    state: str = ACTIVE
    fail_reason: str = ""
    bursts_sent: int = 0
    last_sent: TxBurst | None = None
    peer_burst: int | None = None  # 3-bit seq of the last peer burst decoded
    reply_lost: bool = False
    misses: int = 0  # consecutive timeouts
    reply_escalation: int = 0  # consecutive peer repeats/polls: my replies are lost
    # the escalation the last recovery took: the next drop starts there
    # instead of climbing the whole ladder again (at MPP -8 each drop took
    # 4 polls until the session's clock ran out)
    esc_floor: int = 0
    _clean: int = 0  # clean turns since the floor last moved
    _sent_esc: int = 0  # the escalation my last built burst went at
    no_progress: int = 0
    resyncs: int = 0
    resync_due: bool = False
    _snapshots: dict = field(default_factory=dict)  # my burst number -> (cum, received) as conveyed
    _sent_seqs: dict = field(default_factory=dict)  # my burst number -> data seqs in it
    # peer burst seq whose ACK my tx side holds; starts at 7, which no peer
    # burst carries before one has been decoded (peers start at 0)
    _acted_on: int = F.BURST_MOD - 1
    # my bursts by absolute number: the latest built, and the latest the
    # peer acted on. -1: the virtual burst before the first (seq 7).
    _latest: int = -1
    _confirmed: int = -1
    _abandon: bytes | None = None  # my pending abandon TLV, until the peer applies it
    _abandon_bursts: set = field(default_factory=set)  # my burst seqs that carried it
    _abandon_epoch: int = 0
    _peer_epoch: int = 0  # the last peer abandon epoch applied
    _stale: bool = False  # the last burst handled repeated one already answered: its ACK may be stale
    _answered: bool = False  # answered() since the last burst handled: a repeat of it means my reply was lost

    @property
    def peer(self) -> int:
        return 1 - self.direction

    def write(self, data: bytes):
        self.tx.write(data)

    def read(self) -> bytes:
        out = bytes(self.rx.out)
        self.rx.out.clear()
        return out

    # -- transmit ------------------------------------------------------------------

    def build(self, fresh: bool = True) -> TxBurst:
        """The next burst: control, resends, new codewords.

        `fresh`: built as the direct reply to a peer burst just handled,
        so the ACK it acts on is the peer's current receive state (a
        station's receive state changes only when it handles the other's
        bursts). Only a fresh burst may abandon, re-slice or change the
        data submode: a stale ACK's cumulative can lag what the peer has
        already delivered, and re-slicing from it would re-send delivered
        bytes under different codeword boundaries (caught by
        tests/test_arq.py as corrupted delivery). A non-fresh burst (the
        master after a timeout) is a control-only poll, in whatever mode
        escalation picks; its reply makes the next burst fresh."""
        if self._latest - self._confirmed >= F.BURST_MOD - 1 and self.last_sent is not None:
            # 7 of my bursts unconfirmed: one more new seq could alias an old
            # one in the peer's 3-bit acted-on. Repeat the latest instead.
            log.info("TX b%d repeat: 7 bursts unconfirmed", self._latest % F.BURST_MOD)
            self.stats["cw_resend"] += len(self._sent_seqs.get(self._latest, []))
            return self.last_sent
        bn = self._latest + 1  # this burst's absolute number; wire seq bn mod 8
        escalation = self._sent_esc = self._escalation()
        submode, max_cw = self.policy.choose(self, escalation)
        if not self._data_allowed(fresh, submode):
            max_cw = 0  # control only
        rec, hint, reply = self.policy.recommend(self)  # what the peer should send me next (§8)
        ext = self._control_tlvs(bn, reply)
        core = F.Core(ftype=F.ARQ if fresh else F.PROBE, burst_seq=bn % F.BURST_MOD, acted_on=self._acted_on,
                      cum=self.rx.cum % SEQ_MOD, reply_lost=self.reply_lost, recommend=rec, size_hint=hint)
        self.reply_lost = False

        # control codewords: CPM carries control in a short codeword, one per burst
        cpb = self.policy.ctl_payload_bytes(submode)
        missing = self.tx.missing() if fresh else []
        has_new = fresh and self.new_available() > 0
        k, has_new, bitmap, n_ctl, dup = self._fit(ext, core.cum, missing, has_new, max_cw, submode, fresh)
        # what this burst's ACK conveys, exactly (the snapshot must match it)
        if bitmap:
            ext[F.T_BITMAP] = bitmap
        conveyed = F.unpack_bitmap(bitmap, core.cum) if bitmap else set()
        conveyed = frozenset(unwrap(x, self.rx.cum) for x in conveyed)

        resend = missing[:k]
        cycle = self.policy.rv_cycle(submode)
        rvs = [self.tx.cws[x].heard % cycle for x in resend]
        room = max(0, max_cw - dup * n_ctl - len(resend)) if has_new else 0
        # a new codeword is compressed only when its T_COMP bit fits the
        # control's padding: compression never costs a control codeword
        spare = n_ctl * cpb - self._ctl_bytes(ext, k, has_new, 0)
        new = self._new_codewords(submode, room, k, spare, bn)
        core.k = len(resend)
        if resend:
            ext[F.T_RV] = F.pack_rv(rvs)
        if new:
            ext[F.T_NEW] = bytes([new[0].seq % SEQ_MOD])
        if comp := F.pack_flags([self._comp_bit(x) for x in resend] + [c.comp for c in new]):
            ext[F.T_COMP] = comp
        if dup == 2 and not (resend or new):
            dup = 1  # nothing but control after all
        if dup == 2:
            core.ftype = F.ARQ_DUP
        if new and not self.new_available() and self._ctl_bytes(ext, 0, False, 0) + 2 <= n_ctl * cpb:
            # my host's data ends here: the peer may hold its reply for its
            # host's answer (session.REPLY_HOLD_S). Only in the padding, like T_COMP
            ext[F.T_DRAINED] = b""
        ctl = F.Control(core, ext).pack(cpb)
        assert core.n_ctl <= n_ctl, (core.n_ctl, n_ctl)  # T_COMP stayed in the padding

        burst = TxBurst(submode, self._slots(ctl, dup, resend, rvs, new), self.bursts_sent, self.cap)
        self._record(bn, burst, conveyed, resend, new)
        if log.isEnabledFor(logging.INFO):
            self._log_tx(bn, burst, fresh, core, ext, dup, resend, rvs, new, escalation, hint, reply)
        return burst

    def _escalation(self) -> int:
        """My own timeouts (master), or the peer telling me my replies are
        lost (it repeats or polls: §6), climbing from the last recovery's floor."""
        raw = max(self.misses, self.reply_escalation)
        return min(max(raw, self.esc_floor + raw - 1), MAX_ESCALATION) if raw else 0

    def _data_allowed(self, fresh: bool, submode: str) -> bool:
        """Whether this burst may carry data codewords. Abandons the
        outstanding ones first when a resync is due or they are in another
        submode (§4)."""
        stale, self._stale = self._stale, False
        if not fresh:
            return False
        reslice = any(c.submode != submode for c in self.tx.cws.values())
        # CLEANTXBUFFER's limbo: the exact ACK says the rest never arrived (TxSide.abandon drops it)
        limbo = self.tx.limbo is not None
        if stale and (self.resync_due or reslice or limbo):
            # answering a repeat of the peer burst I last answered: its ACK
            # may predate my last burst, so it may not abandon (§4). Control
            # only until a reply to this one; the resync waits for it too.
            return False
        if self.resync_due:
            self.resync_due = False
            self._start_abandon(1)
        elif reslice or limbo:
            self._start_abandon(0)  # a resend must keep its submode (§4)
        return True

    def settle_limbo(self):
        """Call after handle(): CLEANTXBUFFER's limbo settles against its ACK
        when that is exact (not a repeat's), so the host can report SETTLED
        before the reply is built; else the next fresh build settles it."""
        if self.tx.limbo is not None and not self._stale:
            self._start_abandon(0)

    def _start_abandon(self, reset: int):
        a = self.tx.abandon()
        self._abandon_epoch = (self._abandon_epoch + 1) % 128
        self._abandon = bytes([a % SEQ_MOD, reset | self._abandon_epoch << 1])
        self._abandon_bursts = set()

    def _control_tlvs(self, bn: int, reply: int | None) -> dict:
        """The TLVs that do not depend on what data the burst carries."""
        ext = {}
        if self._abandon is not None:
            # every burst carries it until the peer answers one that did: a
            # receiver that missed it would join old and new slicing
            ext[F.T_ABANDON] = self._abandon
            self._abandon_bursts.add(bn)
            if self._abandon[1] & 1:
                ext[F.T_RESYNC] = b""
        if reply is not None:
            ext[F.T_REPLY] = bytes([reply])
        if self.chat:
            ext[F.T_CHAT] = b""
        if self.policy.want_dup:
            ext[F.T_DUPCTL] = b""  # the peer's next data burst: duplicate its control (ARQ_DUP)
        if self.chat or self.peer_chat:
            # what I have queued, so the peer's CHAT objective plans for a file
            # rather than a chat line when there is one (informational only)
            queued = self.new_available()
            if queued > F.CHAT_LINE_BYTES:  # below it the peer plans for a line anyway
                ext[F.T_BUFFER] = min(queued, 65535).to_bytes(2, "big")
        return ext

    def _fit(self, ext: dict, cum: int, missing: list[int], has_new: bool, max_cw: int, submode: str,
             fresh: bool) -> tuple[int, bool, bytes, int, int]:
        """Shrink the burst until control and resends fit: resends first,
        then optional TLVs (deleted from `ext`), the bitmap, new data.
        -> (resends k, has_new, bitmap, control codewords, dup factor)."""
        cpb = self.policy.ctl_payload_bytes(submode)
        max_ctl = self.policy.max_ctl(submode)
        bitmap = F.pack_bitmap({s % SEQ_MOD for s in self.rx.buf}, cum)
        k = min(len(missing), max(0, max_cw - 1))
        while True:
            e = dict(ext)
            if bitmap:
                e[F.T_BITMAP] = bitmap
            # a compressed resend's T_COMP bit, if the peer may lack it, is
            # mandatory: reserved like T_RV
            zk = max((j + 1 for j, x in enumerate(missing[:k]) if self._comp_bit(x)), default=0)
            n_ctl = self._ctl_size(e, cpb, k, has_new, zk)
            # the peer asked for duplicated control: data bursts only
            dup = 2 if (fresh and self.peer_wants_dup and (k or has_new)) else 1
            if n_ctl <= max_ctl and dup * n_ctl + k <= max(max_cw, dup * n_ctl):
                return k, has_new, bitmap, n_ctl, dup
            if k:
                k -= 1
            elif optional := [t for t in OPTIONAL_TLVS if t in ext]:
                del ext[optional[0]]  # hints and requests: a later burst carries them
            elif bitmap:
                bitmap = b""  # costs extra resends, never wrong ones
            elif has_new:
                has_new = False
            else:
                raise ValueError(f"control does not fit {submode}")

    def _new_codewords(self, submode: str, room: int, k: int, spare: int, bn: int) -> list[Codeword]:
        """Up to `room` new codewords; `spare` control padding bytes decide compression."""
        pb = self.policy.payload_bytes(submode)
        new = []
        while len(new) < room and self.tx.next - self.tx.base < WINDOW and self.new_available() > 0:
            fits = COMPRESS and 2 + -(-(k + len(new) + 1) // 8) <= spare
            new.append(self.tx.new_codeword(submode, pb, fits, bn))
        return new

    def _slots(self, ctl: list[bytes], dup: int, resend: list[int], rvs: list[int], new: list[Codeword]) -> list[Slot]:
        slots = [Slot(ctl_mask(self.direction, i, self.key), rv, p) for i, p in enumerate(ctl) for rv in range(dup)]
        # every outstanding codeword was sliced in the current epoch (abandon clears them)
        ep = self._abandon_epoch
        slots += [Slot(data_mask(self.direction, x, self.key, self.tx.cws[x].comp, ep), rv, self.tx.cws[x].payload)
                  for x, rv in zip(resend, rvs)]
        slots += [Slot(data_mask(self.direction, c.seq, self.key, c.comp, ep), 0, c.payload) for c in new]
        return slots

    def _record(self, bn: int, burst: TxBurst, conveyed: frozenset, resend: list[int], new: list[Codeword]):
        """Remember what burst `bn` conveyed and carried, for the peer's ACK of it."""
        self._snapshots[bn] = (self.rx.cum, conveyed)
        self._sent_seqs[bn] = resend + [c.seq for c in new]
        self._latest = bn
        for old in [b for b in self._snapshots if b < self._confirmed]:
            self._snapshots.pop(old, None)
            self._sent_seqs.pop(old, None)
        self.bursts_sent += 1
        self.last_sent = burst
        self.stats["cw_new"] += len(new)
        self.stats["cw_comp"] += sum(c.comp for c in new)
        self.stats["cw_resend"] += len(resend)

    def _log_tx(self, bn, burst, fresh, core, ext, dup, resend, rvs, new, escalation, hint, reply):
        kind = "data" if resend or new else ("ack" if fresh else "poll")
        parts = [f"{kind} {self._burst_desc(burst.submode, len(burst.slots), dup == 2)}"]
        if resend:
            parts.append("resend " + " ".join(f"{x}/rv{rv}" for x, rv in zip(resend, rvs)))
        if new:
            parts.append(f"new {new[0].seq}" + (f"-{new[-1].seq}" if len(new) > 1 else "")
                         + f" {sum(c.length for c in new)} B"
                         + (f" ({sum(c.comp for c in new)} compressed)" if any(c.comp for c in new) else ""))
        if self.tx.pending():
            parts.append(f"unacked {self.tx.next - self.tx.base}, queued {self.new_available()} B")
        parts.append(f"ack cum {self.rx.cum}" + (f" +{len(self.rx.buf)} held" if self.rx.buf else ""))
        parts.append(f"ask data {self._mode(core.recommend)} size {hint}, reply {self._mode(reply)}")
        flags = [f"escalated {escalation}"] * bool(escalation) + [f"timeout {self.misses}"] * bool(self.misses)
        if F.T_ABANDON in ext:
            flags.append(("resync" if F.T_RESYNC in ext else "abandon") + f" at {self.tx.base}")
        flags += ["dup ctl"] * (dup == 2) + ["reply lost"] * core.reply_lost
        log.info("TX b%d %s", bn % F.BURST_MOD, " | ".join(parts + flags))

    def _mode(self, rec: int | None) -> str:
        return "-" if rec is None else self.policy.mode_name(rec)

    def _burst_desc(self, submode: str, n_cw: int, dup: bool = False) -> str:
        t = self.policy.airtime(submode, n_cw, dup)
        return f"{submode} x{n_cw}" + (f" {t:.1f}s" if t is not None else "")

    def _snr(self) -> str:
        m = self.policy.measured
        if not m or "snr_est" not in m:
            return ""
        mi = "/".join(f"{v:.2f}" for k, v in m.items() if k.startswith("mi_"))  # predictor.CONSTS' order
        return (f" | snr {m['snr_est']:.1f} dB, spread {m.get('spread_est', 0.0):.2f} Hz,"
                f" delay {m.get('delay_est_ms', 0.0):.1f} ms" + (f", mi {mi}" if mi else "")
                + (f", energy {self.policy.energies[-1]:.1f} dB" if getattr(self.policy, "energies", None) else ""))

    def _comp_bit(self, seq: int) -> bool:
        c = self.tx.cws[seq]
        return c.comp and not c.comp_known

    def new_available(self) -> int:
        """Bytes my next new codewords can take (TxSide.unsent)."""
        return self.tx.unsent()

    def data_in_flight(self) -> bool:
        """My latest burst carried data codewords and the peer has not acted
        on it yet (its reply would say what arrived)."""
        return bool(self._sent_seqs.get(self._latest))

    @staticmethod
    def _ctl_bytes(ext, k, has_new, zk) -> int:
        """Control bytes with T_RV, T_NEW and a T_COMP covering the first
        `zk` data slots, as they will be packed."""
        e = dict(ext)
        if k:
            e[F.T_RV] = bytes(-(-2 * k // 8))
        if has_new:
            e[F.T_NEW] = b"\0"
        if zk:
            e[F.T_COMP] = bytes(-(-zk // 8))
        return 4 + sum(2 + len(v) for v in e.values())

    @classmethod
    def _ctl_size(cls, ext, pb, k, has_new, zk=0) -> int:
        return max(1, -(-cls._ctl_bytes(ext, k, has_new, zk) // pb))

    def on_timeout(self, allow_repeat: bool = True) -> TxBurst | None:
        """Master only: no decodable reply in time. -> what to send.
        `allow_repeat` False skips the identical repeat for a poll: after a
        long burst, a lost reply is the likelier loss and cheaper to recover
        by asking (the session decides, it knows airtimes)."""
        assert self.master
        self.misses += 1
        self.stats["timeouts"] += 1
        if self.max_misses is not None and self.misses > self.max_misses:
            self._fail("link lost")
            return None
        if allow_repeat and not self.esc_floor and self.misses <= REPEATS_BEFORE_SHRINK and self.last_sent is not None:
            log.info("TX b%d repeat: timeout %d", self._latest % F.BURST_MOD, self.misses)
            self.stats["cw_resend"] += len(self._sent_seqs.get(self._latest, []))
            return self.last_sent  # identical, same burst seq (§6 step 1)
        return self.build(fresh=False)

    # -- receive -------------------------------------------------------------------

    def handle(self, rx: RxBurst) -> bool:
        """Process a received burst. -> whether its control decoded (then
        this station must answer). Discards the burst otherwise."""
        self.stats["rx_ok" if (ok := self._handle(rx)) else "rx_lost"] += 1
        if not ok:
            log.info("RX %s%s | control lost, discarded", self._burst_desc(rx.submode, rx.n_cw), self._snr())
        return ok

    def _handle(self, rx: RxBurst) -> bool:
        got = self._read_control(rx)
        if not got:
            return False
        ctl, dup = got
        core, ext = ctl.core, ctl.ext
        was, repeat = self._note_peer(core, ext)
        before = (self.tx.base, self.tx.next, self.rx.cum)
        try:
            # never guess on a protocol error: a blind resync can corrupt
            acted, progress = self._take_ack(core, ext)
            abandoned = self._take_abandon(rx, ext)
            slots, n_dec, n_old, delivered = self._take_data(rx, core, ext, dup, acted)
        except ProtocolError as e:
            self._fail(f"protocol: {e}")
            return True
        if log.isEnabledFor(logging.INFO):
            self._log_rx(rx, core, dup, slots, n_dec, n_old, before, was, repeat, abandoned)
        self.reply_lost = repeat
        self._stale = repeat
        self.peer_burst = core.burst_seq
        self._answered = False
        self._watchdog(progress or delivered)
        return True

    def _read_control(self, rx: RxBurst) -> tuple[F.Control, int] | None:
        """The burst's control and its dup factor; falsy if lost or malformed."""
        first = rx.decode(0, ctl_mask(self.peer, 0, self.key), 0, None)
        paired = False
        if first is None and rx.n_cw >= 2:
            # an ARQ_DUP burst's control pair, combined (a wrong guess just
            # fails the masked CRC)
            first = self._ctl_pair(rx, 0, 0)
            paired = first is not None
        if first is None:
            self.policy.outcome(rx.submode, 0, 0, usable=False)
            return None
        try:
            core = F.Core.unpack(first)
            dup = 2 if core.ftype == F.ARQ_DUP else 1
            if paired and dup == 1:
                return None  # combined as a pair, but not sent as one
            if dup * core.n_ctl > rx.n_cw:
                return self._malformed(f"{core.n_ctl} control codewords x{dup} in a burst of {rx.n_cw}")
            payloads = [first]
            for i in range(1, core.n_ctl):
                p = rx.decode(dup * i, ctl_mask(self.peer, i, self.key), 0, None)
                if p is None and dup == 2:
                    p = self._ctl_pair(rx, 2 * i, i)
                if p is None:
                    return None
                payloads.append(p)
            ctl = F.Control.unpack(payloads)
        except ValueError as e:
            return self._malformed(str(e))
        if why := self._check(ctl.core, ctl.ext, rx.n_cw - dup * ctl.core.n_ctl):
            return self._malformed(why)
        return ctl, dup

    def _note_peer(self, core, ext) -> tuple[tuple, bool]:
        """Take the peer's requests and flags, and move my escalation.
        -> (its previous recommendation and size hint, whether this burst repeats one I answered)."""
        escalated = bool(self.misses or self.reply_escalation)
        self.misses = 0
        was = (self.peer_recommend, self.peer_size_hint)
        self.peer_recommend, self.peer_size_hint = core.recommend, core.size_hint
        self.peer_reply_recommend = ext[F.T_REPLY][0] if F.T_REPLY in ext and ext[F.T_REPLY] else None
        self.peer_chat = F.T_CHAT in ext
        self.peer_drained = F.T_DRAINED in ext
        self.peer_wants_dup = F.T_DUPCTL in ext
        self.peer_reply_lost = core.reply_lost
        self.peer_abandon = F.T_ABANDON in ext
        self.peer_queued = int.from_bytes(ext[F.T_BUFFER], "big") if len(ext.get(F.T_BUFFER, b"")) == 2 else 0
        seen = self.peer_burst is not None and core.burst_seq == self.peer_burst
        repeat = seen and self._answered
        # a repeat or a poll means my last reply did not get through
        self.reply_escalation = self.reply_escalation + 1 if (repeat or core.ftype == F.PROBE) else 0
        if not self.reply_escalation:
            if escalated:
                # recovered: my last burst got through at this escalation
                self.esc_floor, self._clean = self._sent_esc, 0
            elif self.esc_floor:
                self._clean += 1
                if self._clean >= FLOOR_DECAY_TURNS:
                    self.esc_floor, self._clean = self.esc_floor - 1, 0
        return was, repeat

    def _take_ack(self, core, ext) -> tuple[int, bool]:
        """Their ACK of my data (their cum / bitmap), acted on in my next burst.
        -> (my burst it acted on, progress). Raises ProtocolError."""
        # which of my bursts the peer acted on: its 3-bit seq, resolved in
        # [confirmed, latest], at most 8 long (build keeps it so)
        acted = [k for k in range(self._confirmed, self._latest + 1) if k % F.BURST_MOD == core.acted_on]
        if len(acted) != 1:
            raise ProtocolError(f"acted-on {core.acted_on} outside my bursts {self._confirmed}..{self._latest}")
        acted = acted[0]
        self._confirmed = acted
        if self._abandon is not None and acted in self._abandon_bursts:
            self._abandon = None  # the peer has applied it
        for x in self._sent_seqs.pop(acted, []):
            if c := self.tx.cws.get(x):
                c.heard += 1  # the peer decoded that burst: it holds soft bits
                # and mapped its new slots by T_NEW: it holds their T_COMP bits
                c.comp_known |= c.first_bn == acted
        # (settle_limbo can start one before any burst carries it: every ACK is from before it)
        if (self._abandon is not None and acted < min(self._abandon_bursts, default=self._latest + 1)
                and unwrap(core.cum, self.tx.base) > self.tx.base):
            # an ACK of a burst from before my pending abandon describes the
            # old slicing: past the abandon point it can't be mapped (a late
            # burst delivered old codewords there). Never guess (§4).
            raise ProtocolError(f"ACK of pre-abandon burst {acted % F.BURST_MOD} past the abandon at {self.tx.base}")
        cum = unwrap(core.cum, self.tx.base)
        received = F.unpack_bitmap(ext[F.T_BITMAP], core.cum) if F.T_BITMAP in ext else set()
        received = frozenset(unwrap(x, cum) for x in received)
        progress = self.tx.on_ack(cum, received)
        self._acted_on = core.burst_seq
        return acted, progress

    def _take_abandon(self, rx: RxBurst, ext) -> bool:
        """Apply the peer's abandon if it is a new one. -> whether it was. Raises ProtocolError."""
        if not (F.T_ABANDON in ext and ext[F.T_ABANDON][1] >> 1 != self._peer_epoch):
            return False
        a = unwrap(ext[F.T_ABANDON][0], self.rx.cum)
        if a != self.rx.cum:
            raise ProtocolError(f"abandon at {a}, cumulative {self.rx.cum}")  # only sent against an exact ACK
        self._peer_epoch = ext[F.T_ABANDON][1] >> 1
        self.rx.abandon(a)
        # every seq from a on is re-sliced: soft bits held for any of
        # them (the failed ones, not only the decoded ones) belong to
        # the old codewords (scripts/phy_session.py found them combined into new ones)
        self._forget_all(rx)
        return True

    def _take_data(self, rx: RxBurst, core, ext, dup: int, acted: int) -> tuple[list, int, int, bool]:
        """Decode and deliver their data slots.
        -> (slots, decoded, already had, progress). Raises ProtocolError."""
        n_ctl_slots = dup * core.n_ctl
        slots = self._map(core, ext, rx.n_cw - n_ctl_slots + core.n_ctl, acted)
        bits = F.unpack_flags(ext.get(F.T_COMP, b""), len(slots))
        # new slots' bits are kept (the sender omits them from resends once I
        # act on this burst); a resend's bit is sent whenever it is needed
        self.rx.comp_seqs |= {seq for (seq, _), z in zip(slots[core.k:], bits[core.k:])
                              if z and seq is not None and seq >= self.rx.cum}
        comp = [z or seq in self.rx.comp_seqs for (seq, _), z in zip(slots, bits)]
        self.last_rx_data = bool(slots)
        progress = False
        n_ok = n_new = n_dec = n_old = 0
        for i, (seq, rv) in enumerate(slots, start=n_ctl_slots):
            if seq is None or seq < self.rx.cum:
                n_old += 1
                continue
            key = (self.peer, seq)
            p = rx.decode(i, data_mask(self.peer, seq, self.key, comp[i - n_ctl_slots], self._peer_epoch), rv, key)
            if i >= n_ctl_slots + core.k:
                n_new += 1
                n_ok += p is not None
            if p is not None:
                n_dec += 1
                rx.forget(key)
                progress |= self.rx.accept(seq, p, comp[i - n_ctl_slots])
        self.policy.outcome(rx.submode, n_ok, n_new, usable=True)  # first transmissions only (resends gain from IR)
        return slots, n_dec, n_old, progress

    def _log_rx(self, rx, core, dup, slots, n_dec, n_old, before, was, repeat, abandoned):
        base0, next0, cum0 = before
        kind = "data" if slots else ("poll" if core.ftype == F.PROBE else "ack")
        parts = [f"{kind} {self._burst_desc(rx.submode, rx.n_cw, dup == 2)}" + self._snr()]
        if slots:
            parts.append(f"resend {core.k} + new {len(slots) - core.k}, decoded {n_dec}/{len(slots) - n_old}"
                         + (f", {n_old} already had" if n_old else "")
                         + f", cum {cum0}->{self.rx.cum}" + (f" +{len(self.rx.buf)} held" if self.rx.buf else ""))
        if self.tx.base != base0:
            parts.append(f"acked {base0}->{self.tx.base}")
        elif next0 > base0:
            parts.append(f"acked none of {base0}-{next0 - 1}")
        now = (self.peer_recommend, self.peer_size_hint)
        ask = f"wants data {self._mode(now[0])} size {now[1]}"
        if was[0] is not None and was != now:
            ask += f" (was {self._mode(was[0])} size {was[1]})"
        parts.append(ask + f", reply {self._mode(self.peer_reply_recommend)}")
        flags = (["repeat (my reply lost)"] * repeat + ["abandon"] * abandoned + ["dup ctl"] * (dup == 2)
                 + ["wants dup ctl"] * self.peer_wants_dup)
        log.info("RX b%d %s", core.burst_seq, " | ".join(parts + flags))

    @staticmethod
    def _check(core, ext, n_data) -> str | None:
        """What is wrong with a CRC-valid control (a false CRC accept, a
        broken peer), or None. Checked before any state changes: a bad one
        is dropped like a failed control codeword, and the existing
        machinery (repeat, watchdog, bounded failure) recovers (§10)."""
        if core.k > n_data:
            return f"K {core.k} in {n_data} data slots"
        if len(ext.get(F.T_RV, b"")) < -(-2 * core.k // 8):
            return f"T_RV of {len(ext.get(F.T_RV, b''))} B for K {core.k}"
        if F.T_NEW in ext and not ext[F.T_NEW]:
            return "empty T_NEW"
        if F.T_ABANDON in ext and len(ext[F.T_ABANDON]) < 2:
            return f"T_ABANDON of {len(ext[F.T_ABANDON])} B"
        return None

    def _malformed(self, why: str) -> bool:
        log.warning("RX malformed control (%s): dropped", why)
        return False

    def _ctl_pair(self, rx: RxBurst, slot: int, i: int) -> bytes | None:
        """Control codeword i from slots `slot` (RV 0) and `slot + 1` (RV 1),
        combined under a key used for nothing else and dropped after."""
        key = ("ctl", id(rx), i)
        mask = ctl_mask(self.peer, i, self.key)
        rx.decode(slot, mask, 0, key)
        p = rx.decode(slot + 1, mask, 1, key)
        rx.forget(key)
        return p

    def answered(self):
        """Call after sending the reply to the burst just handled."""
        self._answered = True

    def _map(self, core, ext, n_cw, acted) -> list[tuple[int | None, int]]:
        """(seq, rv) per data slot, from the snapshot of my burst the peer
        acted on (resends) and the `new` extension (new codewords)."""
        out = []
        snap = self._snapshots.get(acted)
        rvs = F.unpack_rv(ext.get(F.T_RV, b""), core.k)
        if core.k:
            if snap is None:
                out += [(None, 0)] * core.k
            else:
                cum, received = snap
                seqs, s = [], cum
                while len(seqs) < core.k:
                    if s not in received:
                        seqs.append(s)
                    s += 1
                out += list(zip(seqs, rvs))
        n_new = n_cw - core.n_ctl - core.k
        if n_new > 0:
            if F.T_NEW not in ext:
                out += [(None, 0)] * n_new
            else:
                start = unwrap(ext[F.T_NEW][0], self.rx.cum)
                out += [(start + j, 0) for j in range(n_new)]
        return out

    def _forget_all(self, rx: RxBurst):
        for s in range(self.rx.cum, self.rx.cum + WINDOW):
            rx.forget((self.peer, s))

    def _fail(self, why: str):
        log.warning("link failed: %s", why)
        self.state = FAILED
        self.fail_reason = why

    def _watchdog(self, progress: bool):
        if progress:
            self.no_progress = 0
            self.resyncs = 0
            self.resync_due = False  # a deferred resync is moot once data moves
            return
        if not self.tx.pending():
            return
        self.no_progress += 1
        if self.no_progress >= NO_PROGRESS_TURNS:
            self.no_progress = 0
            self.resyncs += 1
            if self.resyncs > RESYNCS_BEFORE_FAIL:
                self._fail("no progress")
            else:
                self.resync_due = True
                log.info("watchdog: %d turns without progress, resync %d of %d",
                         NO_PROGRESS_TURNS, self.resyncs, RESYNCS_BEFORE_FAIL)
