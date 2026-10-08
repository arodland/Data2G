"""Broadcast over KISS (docs/broadcast.md): named groups, each a KISS port,
with per-station rate shifting on the ports that ask for it.

Groups. A port carries one group (up to 10 characters of the callsign
alphabet: "APRS", "CHAT"); port 0 is always open, on "KISS 0". Every
codeword of a group's bursts is CRC-masked with the group's key, a hash of
its packed name. The control carries the name (port 0's leaves it out), so
a receiver decodes the control without a mask, reads the name, and checks
the CRC under that name's key: one decode per burst, however many groups
are open, and a promiscuous listener needs no list of groups.

Burst: control codeword(s), then data codewords. Control: a 1-byte header
[version 4 bits | reserved 2 | n_ctl - 1 (2)], then TLVs [type][length]
[value] (zero padded): T_GROUP (the packed group), or T_GROUP_FROM (group
and the sender's callsign, 120 bits packed together), and T_REPORTS on a
port with rate shifting on. Data: [length, 2][frame] back to back (a zero
length ends it). A failed codeword loses the frames it touches, and the
rest if it held a length. No resends: AX.25 retries at layer 2.

Rate shifting (BCAST MODE n AUTO mode; off by default). KISS gives the
TNC frames and nothing back, and a mode can only be chosen from how the
receiver hears us. So a shifting port's bursts carry reports: for each
station this TNC has heard lately, the mode (and burst size) that station
should use to reach us, as the ARQ shifter recommends it from what we
measured (data2g.arq.policy.GearShifter, one per heard station). Stations
are AX.25 callsigns, read from the frames themselves: a burst's sender is
its first frame's RF sender (the last digipeater that has repeated it,
else the source), and a frame goes to its RF next hop (the first
digipeater that hasn't, else the destination). A frame goes in a mode
shifted for its next hop when it is connected-mode AX.25 (I and S frames;
U frames other than UI) and that station has reported on us lately;
anything else goes in the port's fallback mode. Off, frames are never
parsed as AX.25 and reports heard are ignored.

Statuses (KissLink.events, for the host): BCAST n HEARD [call], BCAST n
LOST k, BCAST * MISSED submode n_cw, BCAST n DROPPED k. KISS ACKMODE acks
(KissLink.acks): (port, ack) for each frame enqueued with an ack, once the
burst carrying it has gone (on_sent).
"""

import logging
import struct
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

from . import codes, cpm
from .config import max_codewords
from .arq import frames as F
from .arq import phy as PHY
from .arq import policy as G
from .arq.link import Slot, TxBurst, ctl_mask, data_mask
from .arq.modes import MODES, ctl_payload_bytes, is_cpm, max_ctl

log = logging.getLogger("data2g.kiss")
VERSION = 2
T_GROUP, T_GROUP_FROM, T_REPORTS = 1, 2, 3  # control TLVs (docs/broadcast.md §3)
PORT0_GROUP = "KISS 0"
N_PORTS = 16  # KISS port numbers 0-15
REPORT_MAX_S = 180.0  # a report older than this is not followed
HEARD_MAX_S = 600.0  # stations reported on: heard this recently
# robust broadcast mode per cap (data2g.arq.policy.CAP_HZ): what everyone hears
BROADCAST = {0: "n10-qpsk-r1/5", 2: "qpsk-r1/5"}
BROADCAST_S = G.SIZE_S[-1]  # bursts in a port's own mode: at most the longest size class
SLOT_S = 1.0  # the shortest p-persistence slot (KissLink.slot_s)
MIN_SUCCESS = 0.9  # recommended modes: predicted first-transmission codeword success at least this



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



# --- groups and control ------------------------------------------------------

def group_name(group: str) -> str:
    """A group as it reads back from the air (upper case, trailing spaces
    gone); ValueError if it won't pack."""
    return F.unpack_call(F.pack_call(group))


def group_key(group: str) -> int:
    """Nonzero 16-bit key of a group: FNV-1a of its packed name."""
    h = 0x811C9DC5
    for b in F.pack_call(group):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return (h & 0xFFFF) or 1


def pack_pair(group: str, call: str) -> bytes:
    """T_GROUP_FROM: group and callsign, 60 bits each, in 15 bytes."""
    v = int.from_bytes(F.pack_call(group), "big") << 60 | int.from_bytes(F.pack_call(call), "big")
    return v.to_bytes(15, "big")


def unpack_pair(b: bytes) -> tuple[str, str]:
    v = int.from_bytes(b, "big")
    return F.unpack_call((v >> 60).to_bytes(8, "big")), F.unpack_call((v & ((1 << 60) - 1)).to_bytes(8, "big"))


def parse_tlvs(b: bytes) -> dict:
    """[type][length][value]... up to a zero type (padding) -> {type: value};
    ValueError if one runs past the end."""
    out, i = {}, 0
    while i + 2 <= len(b) and b[i]:
        t, n = b[i], b[i + 1]
        if i + 2 + n > len(b):
            raise ValueError("truncated TLV")
        out[t] = b[i + 2:i + 2 + n]
        i += 2 + n
    return out


def read_control(tlvs: dict) -> tuple[str, str | None, bytes | None]:
    """Control TLVs -> (group, sender's call or None, reports or None);
    ValueError if malformed."""
    group, call = PORT0_GROUP, None
    if T_GROUP_FROM in tlvs:
        if len(tlvs[T_GROUP_FROM]) != 15:
            raise ValueError("T_GROUP_FROM length")
        group, call = unpack_pair(tlvs[T_GROUP_FROM])
    elif T_GROUP in tlvs:
        if len(tlvs[T_GROUP]) != 8:
            raise ValueError("T_GROUP length")
        group = F.unpack_call(tlvs[T_GROUP])
    rep = tlvs.get(T_REPORTS)
    if rep is not None and (len(rep) < 2 or (len(rep) - 2) % 3):
        raise ValueError("T_REPORTS length")
    return group, call, rep


def _tlv(t: int, v: bytes) -> bytes:
    return bytes([t, len(v)]) + v


# --- the link ---------------------------------------------------------------

@dataclass
class Peer:
    shifter: G.GearShifter
    heard: float  # when we last heard it
    report: tuple | None = None  # (rec byte, time): how it wants us to send to it


@dataclass
class Port:
    group: str  # as it reads back (group_name)
    mode: str  # the transmit mode; with auto, the fallback
    auto: bool = False  # rate shifting (BCAST MODE n AUTO mode)
    from_call: str | None = None  # sent in T_GROUP_FROM

    @property
    def key(self) -> int:
        return group_key(self.group)


@dataclass
class KissLink:
    cap: int = 2  # policy.CAP_HZ key: 2 = 2400 Hz, 0 = 500 Hz
    queue: list = field(default_factory=list)  # (port, frame, ack or None) waiting
    peers: dict = field(default_factory=dict)  # station hash -> Peer
    me: set = field(default_factory=set)  # hashes this TNC has sent as
    clock: callable = time.monotonic
    n_sent: int = 0
    broadcast: str | None = None  # a new port's transmit mode (None: BROADCAST[cap])
    # channel access (the engine applies them). A slot must outlast our
    # carrier sense: a burst reads as BUSY 0.44-0.79 s after it starts (10 dB)
    persist: int = 63  # KISS P: after BUSY, a slot is taken with probability (P + 1) / 256
    slot_s: float = SLOT_S  # KISS SLOTTIME
    busy_limit_s: float = 60.0  # BUSY held a burst this long: send anyway (modem73's csma)
    ports: dict = field(default_factory=dict)  # port number -> Port; 0 always
    events: list = field(default_factory=list)  # statuses for the host, oldest first
    acks: list = field(default_factory=list)  # (port, ack) of frames that went out
    _inflight: tuple | None = None  # (the last burst handed out, its frames' (port, ack))

    def __post_init__(self):
        self.broadcast = self.broadcast or BROADCAST[self.cap]
        self._check_mode(self.broadcast)
        self.ports.setdefault(0, Port(PORT0_GROUP, self.broadcast))
        self._stub = SimpleNamespace(cap=self.cap, rx=SimpleNamespace(buf={}), chat=False, peer_chat=False,
                                     peer_queued=0)

    def _check_mode(self, mode: str):
        s = MODES.get(mode)
        if s is None or s not in G.allowed(self.cap):
            raise ValueError(f"mode {mode!r}: not a mode within the bandwidth cap")

    def command(self, cmd: int, payload: bytes):
        """A KISS command from a client. P is kept. SLOTTIME (10 ms units)
        can lengthen the slot, not shorten it: clients' defaults (100 ms)
        suit VHF carrier detect, not ours. TXDELAY (--ptt-on-delay-ms is
        ours), TXTAIL and the rest are ignored."""
        if not payload or cmd not in (2, 3):
            log.debug("KISS command %d ignored", cmd)
            return
        if cmd == 2:
            self.persist = payload[0]
        else:
            self.slot_s = max(SLOT_S, payload[0] / 100)
        log.info("KISS %s = %d", "P" if cmd == 2 else "SLOTTIME", payload[0])

    # -- ports (the host's BCAST commands) -------------------------------------

    def open(self, group: str, from_call: str | None = None) -> int:
        """A port for `group` -> its number (1-15); ValueError if the group
        or call won't pack, or every port is taken."""
        p = Port(group_name(group), self.broadcast, from_call=group_name(from_call) if from_call else None)
        self._check_fits(p, p.mode, False, n=1)
        for n in range(1, N_PORTS):
            if n not in self.ports:
                self.ports[n] = p
                log.info("broadcast port %d open: %s%s", n, p.group, f" from {p.from_call}" if p.from_call else "")
                return n
        raise ValueError("every port is open")

    def close(self, n: int):
        """Close port n (1-15); frames queued for it are dropped (DROPPED)."""
        if not 1 <= n < N_PORTS or n not in self.ports:
            raise ValueError(f"port {n} is not open")
        del self.ports[n]
        self._drop(lambda q: q[0] == n, n)

    def set_mode(self, n: int, mode: str, auto: bool = False):
        """Port n's transmit mode for frames sent after it; with `auto`,
        rate shifting on and `mode` its fallback. ValueError if the port is
        closed, or the mode is outside the cap or too small for the control."""
        if n not in self.ports:
            raise ValueError(f"port {n} is not open")
        self._check_mode(mode)
        self._check_fits(self.ports[n], mode, auto, n)
        self.ports[n].mode, self.ports[n].auto = mode, auto

    def _check_fits(self, p: Port, mode: str, auto: bool, n: int):
        """The control a port needs (its group, and room for a report) must
        fit the mode's control codewords."""
        spec = MODES[mode]
        need = 1 + len(self._group_tlv(n, p)) + (len(_tlv(T_REPORTS, bytes(2))) if auto else 0)  # reports: the sender at least
        if need > min(4, max_ctl(spec)) * ctl_payload_bytes(spec):
            raise ValueError(f"mode {mode}: a {need}-byte control doesn't fit")

    def _drop(self, which, port: int):
        k = sum(1 for q in self.queue if which(q))
        if k:
            self.queue = [q for q in self.queue if not which(q)]
            self.events.append(f"BCAST {port} DROPPED {k}")

    # -- sending -------------------------------------------------------------

    def enqueue(self, frame: bytes, port: int = 0, ack=None):
        """A frame for `port`; `ack` (opaque, e.g. a KISS ACKMODE tag) comes
        back in `acks` once the frame has gone. To a closed port: DROPPED."""
        if port not in self.ports:
            self.events.append(f"BCAST {port} DROPPED 1")
            return
        if not frame:
            return  # a zero length is the burst's end marker (tnc.unpack)
        self.queue.append((port, frame, ack))

    def take_events(self) -> list[str]:
        """The statuses since the last call, oldest first."""
        out, self.events = self.events, []
        return out

    def take_acks(self) -> list:
        """(port, ack) of frames gone out since the last call."""
        out, self.acks = self.acks, []
        return out

    def on_sent(self, burst: TxBurst):
        """A burst finished transmitting: its frames' acks are due."""
        if self._inflight is not None and burst == self._inflight[0]:
            self.acks += self._inflight[1]
            self._inflight = None

    def missed(self, submode: str, n_cw: int):
        """A burst heard whose header decoded but nothing else: it can't be
        tied to a port (docs/broadcast.md §5)."""
        self.events.append(f"BCAST * MISSED {submode} {n_cw}")

    def _route(self, port: Port, frame: bytes) -> tuple[str, int]:
        """-> (mode, size hint) for a frame on `port`."""
        if port.auto:
            ax = parse_ax25(frame)
            if ax is not None and ax.connected:
                p = self.peers.get(station_hash(ax.next_hop))
                if p is not None and p.report is not None and self.clock() - p.report[1] <= REPORT_MAX_S:
                    mode = G.decode(p.report[0] >> 2)
                    if mode is not None and MODES[mode] in G.allowed(self.cap):
                        return mode, p.report[0] & 3
        return port.mode, len(G.SIZE_S) - 1

    @staticmethod
    def _group_tlv(n: int, p: Port) -> bytes:
        if p.from_call:
            return _tlv(T_GROUP_FROM, pack_pair(p.group, p.from_call))
        return b"" if n == 0 else _tlv(T_GROUP, F.pack_call(p.group))

    def _control(self, spec, n: int, port: Port, sender: int) -> list[bytes]:
        """Control codeword payloads: header, the group, and on a shifting
        port as many fresh reports as fit."""
        pb, n_max = ctl_payload_bytes(spec), min(4, max_ctl(spec))
        body = self._group_tlv(n, port)
        if port.auto:
            now = self.clock()
            fresh = sorted(((h, p) for h, p in self.peers.items()
                            if now - p.heard <= HEARD_MAX_S and p.shifter.measured), key=lambda hp: -hp[1].heard)
            room = min((n_max * pb - 1 - len(body) - 4) // 3, (255 - 2) // 3)
            rep = struct.pack(">H", sender)
            for h, p in fresh[:max(0, room)]:
                rec, hint, _ = p.shifter.recommend(self._stub)
                rep += struct.pack(">HB", h, (rec << 2) | hint)
            body += _tlv(T_REPORTS, rep)
        n_ctl = max(1, -(-(1 + len(body)) // pb))
        stream = bytes([VERSION << 4 | (n_ctl - 1)]) + body
        stream += bytes(n_ctl * pb - len(stream))
        return [stream[i * pb:(i + 1) * pb] for i in range(n_ctl)]

    def next_burst(self) -> TxBurst | None:
        """The next burst to send: the first queued frame's port and mode,
        with every queued frame of that port that goes in the same mode, in
        queue order, as many as fit. None: nothing queued."""
        if not self.queue:
            return None
        n, first, _ = self.queue[0]
        port = self.ports[n]
        mode, hint = self._route(port, first)
        spec = MODES[mode]
        sender = 0
        if port.auto and (ax := parse_ax25(first)) is not None:
            sender = station_hash(ax.sender)
            self.me.add(sender)
        ctl = self._control(spec, n, port, sender)
        seconds = G.SIZE_S[hint] if mode != port.mode else BROADCAST_S
        pb = codes.payload_bytes(spec)
        # the size class is a preference: a burst grows to carry its first frame
        # (a 256-byte PACLEN I frame outgrew a 12 s qpsk-r1/5 burst), up to what
        # its header can say
        limit = 1 + 1 + cpm.MAX_DATA if is_cpm(spec) else max_codewords(spec.sync_band)
        need = len(ctl) + -(-(2 + len(first)) // pb)
        n_max = min(limit, max(G.slots_for(spec, seconds), need))
        room = (n_max - len(ctl)) * pb
        taken, rest, size = [], [], 0
        for q in self.queue:
            if q[0] == n and self._route(port, q[1])[0] == mode and size + 2 + len(q[1]) <= room:
                taken.append(q)
                size += 2 + len(q[1])
            else:
                rest.append(q)
        if not taken:  # the first frame alone doesn't fit: it never will in this mode
            log.error("%d-byte frame dropped: a %s burst carries at most %d", len(first), mode, room - 2)
            self.queue = self.queue[1:]
            self.events.append(f"BCAST {n} DROPPED 1")
            return self.next_burst()
        self.queue = rest
        stream = b"".join(struct.pack(">H", len(q[1])) + q[1] for q in taken)
        stream += bytes(-len(stream) % pb) if stream else bytes(pb)
        data = [stream[i:i + pb] for i in range(0, len(stream), pb)]
        key = port.key
        slots = [Slot(ctl_mask(0, i, key), 0, p) for i, p in enumerate(ctl)]
        slots += [Slot(data_mask(0, len(ctl) + j, key), 0, p) for j, p in enumerate(data)]
        self.n_sent += 1
        burst = TxBurst(mode, slots, self.n_sent, self.cap)
        self._inflight = (burst, [(n, q[2]) for q in taken if q[2] is not None])
        return burst

    # -- receiving -----------------------------------------------------------

    def _read_control(self, rx, n_cw: int):
        """The burst's control, checked under the key of the group it names
        -> (group, sender's call, reports, n_ctl), or None."""
        for c0 in rx.raw(0):
            if c0[0] >> 4 != VERSION or (c0[0] & 3) + 1 > n_cw:
                continue
            n_ctl = (c0[0] & 3) + 1
            rest = [(rx.raw(i) or [bytes(len(c0))])[0] for i in range(1, n_ctl)]  # the best guess, to read
            try:
                group = read_control(parse_tlvs(b"".join([c0] + rest)[1:]))[0]
                key = group_key(group)
            except ValueError:
                continue
            ctl = [rx.decode(i, ctl_mask(0, i, key), 0, None) for i in range(n_ctl)]
            if None in ctl or ctl[0] != c0:
                continue
            try:
                return (*read_control(parse_tlvs(b"".join(ctl)[1:])), n_ctl)
            except ValueError as e:
                log.warning("RX malformed broadcast control (%s): dropped", e)
                return None
        return None

    def _data(self, rx, r: dict, start: int, key: int) -> tuple[list, list, int]:
        """Data slots from `start` under `key` -> (frames, per-slot ok, frames lost)."""
        from .tnc import unpack

        payloads, ok = [], []
        for i in range(start, r["n_cw"]):
            p = rx.decode(i, data_mask(0, i, key), 0, None)
            payloads.append(p or bytes(codes.payload_bytes(r["spec"])))
            ok.append(p is not None)
        frames, lost = unpack(payloads, ok) if payloads else ([], 0)
        return frames, ok, lost

    def on_burst(self, r: dict, rx=None) -> list[tuple[int, bytes]] | None:
        """A received burst (modem.receive's or cpm.receive's dict; `rx` its
        phy.ModemRx, shared with the engine's other checks) -> [(port,
        frame)] for the open ports it belongs to ([] for another group's), or
        None if it isn't a broadcast burst (an ARQ burst, or one lost whole).
        Statuses go to `events`; a shifting port learns from its sender."""
        rx = rx or PHY.ModemRx(r, {}, PHY.DD_BUDGET_S)  # never DD unbounded
        now = self.clock()
        ctl = self._read_control(rx, r["n_cw"])
        if ctl is not None:
            group, call, rep, n_ctl = ctl
            frames, ok, lost = self._data(rx, r, n_ctl, group_key(group))
            ports = sorted(i for i, p in self.ports.items() if p.group == group)
            if rep is not None and any(self.ports[i].auto for i in ports):
                self._learn(rep, frames, r, ok, now)
            for i in ports:
                self.events.append(f"BCAST {i} HEARD" + (f" {call}" if call else ""))
                if lost:
                    self.events.append(f"BCAST {i} LOST {lost}")
            return [(i, f) for i in ports for f in frames]
        # control lost: slot 1 passing as data under an open port's key says the
        # burst is that group's, with a one-codeword control (docs/broadcast.md §2)
        if r["n_cw"] < 2:
            return None
        keys = sorted({p.key for p in self.ports.values()})
        key = next((k for k in keys if rx.decode(1, data_mask(0, 1, k), 0, None) is not None), None)
        if key is None:
            return None
        frames, _, lost = self._data(rx, r, 1, key)
        ports = sorted(i for i, p in self.ports.items() if p.key == key)
        if len({self.ports[i].group for i in ports}) > 1:  # two groups, one key: whose it is can't be told
            log.warning("broadcast burst under a key %d open groups share: dropped", len(ports))
            for i in ports:
                self.events.append(f"BCAST {i} LOST {len(frames) + lost}")
            return []
        for i in ports:
            self.events.append(f"BCAST {i} HEARD")
            if lost:
                self.events.append(f"BCAST {i} LOST {lost}")
        return [(i, f) for i in ports for f in frames]

    def _learn(self, rep: bytes, frames: list, r: dict, ok: list, now: float):
        """A shifting port heard `rep` (T_REPORTS): what we know of its sender."""
        sender = struct.unpack(">H", rep[:2])[0]
        if not sender and frames and (ax := parse_ax25(frames[0])) is not None:
            sender = station_hash(ax.sender)
        if not sender or sender in self.me:
            return
        mode = r["spec"].name
        peer = self.peers.setdefault(sender, Peer(G.GearShifter(use_cpm=True, min_success=MIN_SUCCESS), now))
        peer.heard = now
        peer.shifter.outcome(mode, sum(ok), len(ok))
        peer.shifter.observe(PHY.measure(r), mode, now)
        for j in range(2, len(rep), 3):
            h, rec = struct.unpack(">HB", rep[j:j + 3])
            if h in self.me:
                peer.report = (rec, now)
