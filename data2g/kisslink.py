"""The KISS TNC's link layer: mode shifting without a session.

KISS gives the TNC frames and nothing back, and a mode can only be chosen
from how the receiver hears us. So every burst carries reports: for each
station this TNC has heard lately, the mode (and burst size) that station
should use to reach us, as the ARQ shifter recommends it from what we
measured (data2g.arq.policy.GearShifter, one per heard station).

Stations are AX.25 callsigns, read from the frames themselves: a burst's
sender is its first frame's RF sender (the last digipeater that has
repeated it, else the source), and a frame goes to its RF next hop (the
first digipeater that hasn't, else the destination). AX.25 traffic in
connected mode runs both ways (I frames one way, RR the other), so
reports flow wherever it does.

A frame goes in a mode shifted for its next hop when it is connected-mode
AX.25 (I and S frames; U frames other than UI) and that station has
reported on us lately. Anything else (UI frames whatever their
destination, non-AX.25, stations without a fresh report) goes in the
cap's robust broadcast mode.

Burst: control codeword(s) (the ARQ's control masks, with KISS_KEY) then
data codewords (data masks by slot index). Control: [version | n_ctl - 1]
[sender hash, 2] [n reports] then per report [station hash, 2]
[mode code << 2 | size hint]. Data: [length, 2][frame] back to back (a
zero length ends it), as the single-mode TNC had. A failed codeword loses
the frames it touches, and the rest if it held a length. No resends: AX.25
retries at layer 2.
"""

import logging
import struct
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

from . import codes, cpm
from .config import MAX_CODEWORDS
from .arq import phy as PHY
from .arq import policy as G
from .arq.link import Slot, TxBurst, ctl_mask, data_mask
from .arq.modes import MODES, ctl_payload_bytes, is_cpm, max_ctl

log = logging.getLogger("data2g.kiss")
KISS_KEY = 0x4B53  # CRC masks of KISS bursts (an ARQ session's key is a hash: 1 in 65536 alike)
VERSION = 1
REPORT_MAX_S = 180.0  # a report older than this is not followed
HEARD_MAX_S = 600.0  # stations reported on: heard this recently
# robust broadcast mode per cap (data2g.arq.policy.CAP_HZ): what everyone hears
BROADCAST = {0: "n10-qpsk-r1/5", 1: "qpsk-r1/5", 2: "qpsk-r1/5"}
BROADCAST_S = G.SIZE_S[-1]
MIN_SUCCESS = 0.9  # recommended modes: predicted first-transmission codeword success at least this  # broadcast bursts: at most the longest size class


# --- AX.25 ------------------------------------------------------------------

def _call(b: bytes) -> str:
    call = bytes(c >> 1 for c in b[:6]).decode("ascii", "replace").strip()
    ssid = (b[6] >> 1) & 0x0F
    return f"{call}-{ssid}" if ssid else call


@dataclass(frozen=True)
class Ax25:
    dst: str
    src: str
    next_hop: str  # who hears it next on RF
    sender: str  # who puts it on RF
    connected: bool  # connected-mode frame (I, S, or a U frame other than UI)


def parse_ax25(frame: bytes) -> Ax25 | None:
    """AX.25 addresses and frame type, or None if `frame` isn't AX.25."""
    addrs, i = [], 0
    while i + 7 <= len(frame):
        addrs.append(frame[i:i + 7])
        i += 7
        if addrs[-1][6] & 1:  # address extension bit: the last address
            break
    else:
        return None
    if len(addrs) < 2 or len(addrs) > 10 or i >= len(frame):
        return None
    if any(not all(0x40 <= c <= 0xB4 and not c & 1 for c in a[:6]) for a in addrs):
        return None  # a shifted ASCII callsign on every address
    ctrl = frame[i]
    ui = ctrl & 0xEF == 0x03  # UI, with the P/F bit either way
    digis = addrs[2:]
    repeated = [d for d in digis if d[6] & 0x80]  # H bit: has repeated
    pending = [d for d in digis if not d[6] & 0x80]
    return Ax25(dst=_call(addrs[0]), src=_call(addrs[1]),
                next_hop=_call(pending[0]) if pending else _call(addrs[0]),
                sender=_call(repeated[-1]) if repeated else _call(addrs[1]), connected=not ui)


def station_hash(call: str) -> int:
    """Nonzero 16-bit id of a callsign (FNV-1a)."""
    h = 0x811C9DC5
    for b in call.upper().encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return (h & 0xFFFF) or 1


# --- the link ---------------------------------------------------------------

@dataclass
class Peer:
    shifter: G.GearShifter
    heard: float  # when we last heard it
    report: tuple | None = None  # (rec byte, time): how it wants us to send to it


@dataclass
class KissLink:
    cap: int = 2  # policy.CAP_HZ key: 2 = 2400 Hz, 0 = 500 Hz
    queue: list = field(default_factory=list)  # frames waiting
    peers: dict = field(default_factory=dict)  # station hash -> Peer
    me: set = field(default_factory=set)  # hashes this TNC has sent as
    clock: callable = time.monotonic
    n_sent: int = 0
    broadcast: str | None = None  # the robust broadcast mode (None: BROADCAST[cap])

    def __post_init__(self):
        self.broadcast = self.broadcast or BROADCAST[self.cap]
        s = MODES.get(self.broadcast)
        if s is None or s not in G.allowed(self.cap):
            raise ValueError(f"broadcast mode {self.broadcast!r}: not a mode within the bandwidth cap")
        self._stub = SimpleNamespace(cap=self.cap, rx=SimpleNamespace(buf={}), chat=False, peer_chat=False,
                                     peer_queued=0)

    # -- sending -------------------------------------------------------------

    def enqueue(self, frame: bytes):
        self.queue.append(frame)

    def _route(self, frame: bytes) -> tuple[str, int]:
        """-> (mode, size hint) for a frame."""
        ax = parse_ax25(frame)
        if ax is not None and ax.connected:
            p = self.peers.get(station_hash(ax.next_hop))
            if p is not None and p.report is not None and self.clock() - p.report[1] <= REPORT_MAX_S:
                mode = G.decode(p.report[0] >> 2)
                if mode is not None and MODES[mode] in G.allowed(self.cap):
                    return mode, p.report[0] & 3
        return self.broadcast, len(G.SIZE_S) - 1

    def _control(self, spec, sender: int) -> list[bytes]:
        """Control codeword payloads: header and as many fresh reports as fit."""
        pb, n_max = ctl_payload_bytes(spec), max_ctl(spec)
        now = self.clock()
        fresh = sorted(((h, p) for h, p in self.peers.items() if now - p.heard <= HEARD_MAX_S and p.shifter.measured),
                       key=lambda hp: -hp[1].heard)
        room = (n_max * pb - 4) // 3
        body = b""
        for h, p in fresh[:max(0, room)]:
            rec, hint, _ = p.shifter.recommend(self._stub)
            body += struct.pack(">HB", h, (rec << 2) | hint)
        n = len(body) // 3
        n_ctl = max(1, -(-(4 + len(body)) // pb))
        stream = bytes([VERSION << 4 | (n_ctl - 1)]) + struct.pack(">HB", sender, n) + body
        stream += bytes(n_ctl * pb - len(stream))
        return [stream[i * pb:(i + 1) * pb] for i in range(n_ctl)]

    def next_burst(self) -> TxBurst | None:
        """The next burst to send: the first queued frame's mode, with every
        queued frame that goes in the same mode, in queue order, as many as
        fit. None: nothing queued."""
        if not self.queue:
            return None
        mode, hint = self._route(self.queue[0])
        spec = MODES[mode]
        first = parse_ax25(self.queue[0])
        sender = station_hash(first.sender) if first is not None else 0
        if sender:
            self.me.add(sender)
        ctl = self._control(spec, sender)
        seconds = G.SIZE_S[hint] if mode != self.broadcast else BROADCAST_S
        pb = codes.payload_bytes(spec)
        # the size class is a preference: a burst grows to carry its first frame
        # (a 256-byte PACLEN I frame outgrew a 12 s qpsk-r1/5 burst), up to what
        # its header can say
        limit = 1 + 1 + cpm.MAX_DATA if is_cpm(spec) else MAX_CODEWORDS
        need = len(ctl) + -(-(2 + len(self.queue[0])) // pb)
        n_max = min(limit, max(G.slots_for(spec, seconds), need))
        room = (n_max - len(ctl)) * pb
        taken, rest, size = [], [], 0
        for f in self.queue:
            if self._route(f)[0] == mode and size + 2 + len(f) <= room:
                taken.append(f)
                size += 2 + len(f)
            else:
                rest.append(f)
        self.queue = rest
        if not taken:  # the first frame alone doesn't fit: it never will in this mode
            log.error("%d-byte frame dropped: a %s burst carries at most %d", len(rest[0]), mode, room - 2)
            self.queue = rest[1:]
            return self.next_burst()
        stream = b"".join(struct.pack(">H", len(f)) + f for f in taken)
        stream += bytes(-len(stream) % pb) if stream else bytes(pb)
        data = [stream[i:i + pb] for i in range(0, len(stream), pb)]
        slots = [Slot(ctl_mask(0, i, KISS_KEY), 0, p) for i, p in enumerate(ctl)]
        slots += [Slot(data_mask(0, len(ctl) + j, KISS_KEY), 0, p) for j, p in enumerate(data)]
        self.n_sent += 1
        return TxBurst(mode, slots, self.n_sent)

    # -- receiving -----------------------------------------------------------

    def on_burst(self, r: dict) -> list[bytes] | None:
        """A received burst (modem.receive's or cpm.receive's dict) -> the
        frames in it, or None if it isn't a KISS burst (no codeword decodes
        with KISS_KEY's masks: an ARQ burst, or one lost whole). Updates what
        we know of its sender."""
        rx = PHY.ModemRx(r, {})
        mode, n = r["spec"].name, r["n_cw"]
        now = self.clock()
        c0 = rx.decode(0, ctl_mask(0, 0, KISS_KEY), 0, None)
        sender, reports, start = 0, [], None
        if c0 is not None and c0[0] >> 4 == VERSION:
            n_ctl = (c0[0] & 3) + 1
            ctl = [c0] + [rx.decode(i, ctl_mask(0, i, KISS_KEY), 0, None) for i in range(1, n_ctl)]
            start = n_ctl
            sender, k = struct.unpack(">HB", c0[1:4])
            if all(c is not None for c in ctl):
                stream = b"".join(ctl)[4:]
                reports = [struct.unpack(">HB", stream[3 * j:3 * j + 3]) for j in range(min(k, len(stream) // 3))]
        payloads, ok = [], []
        first = start if start is not None else 1
        for i in range(first, n):
            p = rx.decode(i, data_mask(0, i, KISS_KEY), 0, None)
            if start is None:
                if p is None:
                    continue  # still control, or lost: the stream starts at the first data slot decoded
                start = i
            payloads.append(p or bytes(codes.payload_bytes(r["spec"])))
            ok.append(p is not None)
        from .tnc import unpack

        if c0 is None and not any(ok):
            return None
        frames, _ = unpack(payloads, ok) if payloads else ([], 0)
        if not sender and frames:
            ax = parse_ax25(frames[0])
            sender = station_hash(ax.sender) if ax is not None else 0
        if sender and sender not in self.me:
            peer = self.peers.setdefault(sender, Peer(G.GearShifter(use_cpm=True, min_success=MIN_SUCCESS), now))
            peer.heard = now
            peer.shifter.outcome(mode, sum(ok), len(ok))
            peer.shifter.observe(PHY.measure(r), mode, now)
            for h, rec in reports:
                if h in self.me:
                    peer.report = (rec, now)
        return frames
