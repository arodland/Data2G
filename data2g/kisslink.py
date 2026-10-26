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
[version 4 bits | dup 1 | reserved 1 | n_ctl - 1 (2)], then TLVs [type]
[length][value] (zero padded): T_GROUP (the packed group), or T_GROUP_FROM
(group and the sender's callsign, 120 bits packed together), and T_REPORTS
on a port with rate shifting on. A DUP port (BCAST MODE n DUP mode) sends
each control codeword twice, RV 0 then RV 1, and a receiver combines the
two when the first fails alone. Data: each frame COBS-encoded and ended by
a zero byte, back to back, zero padded, with a zero byte more wherever a
frame would start a codeword (pack_frames, unpack_frames); each data
codeword's mask carries the slot the data starts at (data_start_mask). A
failed codeword loses only the frames it touches: the next zero byte in a
good codeword starts a frame. No resends: AX.25 retries at layer 2.

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

Statuses (KissLink.events, for the host): BCAST n HEARD [call] snr, BCAST n
LOST k, BCAST * MISSED submode n_cw [snr], BCAST n DROPPED k. KISS ACKMODE acks
(KissLink.acks): (port, ack) for each frame enqueued with an ack, once the
burst carrying it has gone (on_sent).

Native: native/core/kisslink. KissLink.read_burst_control and
read_burst_frames are also native/core/monitor's way into a broadcast
burst.
"""

import logging
import struct
import time
from dataclasses import dataclass, field

import numpy as np

from . import codes, cpm
from .config import max_codewords
from .arq import frames as F
from .arq import phy as PHY
from .arq import policy as G
from .arq.burst import Slot, TxBurst, ctl_mask, data_mask
from .arq.modes import MODES, ctl_payload_bytes, is_cpm, max_ctl

log = logging.getLogger("data2g.kiss")
VERSION = 3
DUP = 0x08  # the control header's dup bit: each control codeword sent twice, RV 0 then RV 1
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


def check_mode(mode: str, cap: int):
    """ValueError unless `mode` is a mode within bandwidth cap `cap`."""
    s = MODES.get(mode)
    if s is None or s not in G.allowed(cap):
        raise ValueError(f"mode {mode!r}: not a mode within the bandwidth cap")


def _snr(r: dict) -> str:
    """A heard burst's status suffix: its pilot SNR (snr_est, dB), as the monitor shows it."""
    return f" {PHY.measure(r)['snr_est']:.1f}"


def _tlv(t: int, v: bytes) -> bytes:
    return bytes([t, len(v)]) + v


# --- data -------------------------------------------------------------------

def framed_len(n: int) -> int:
    """The most bytes an n-byte frame takes in a burst's data: COBS adds a
    byte per 254 and one more, then the zero that ends it. n + 2 up to 253
    bytes, as the 2-byte length it replaced."""
    return n + 2 + n // 254


def cobs(frame: bytes) -> bytes:
    """Consistent Overhead Byte Stuffing: `frame` with no zero byte left,
    each run of up to 254 non-zero bytes led by its length + 1."""
    out, run = bytearray(), bytearray()
    for b in frame:
        if b:
            run.append(b)
            if len(run) < 254:
                continue
            out += bytes([255]) + run  # a full run: no zero implied after it
        else:
            out += bytes([len(run) + 1]) + run
        run.clear()
    return bytes(out + bytes([len(run) + 1]) + run)


def uncobs(b: bytes) -> bytes | None:
    """cobs() undone; None if `b` isn't one (a code byte runs past the end)."""
    out, i = bytearray(), 0
    while i < len(b):
        c = b[i]
        if not c or i + c > len(b):
            return None
        out += b[i + 1:i + c]
        i += c
        if c < 255 and i < len(b):
            out.append(0)
    return bytes(out)


def data_start_mask(slot: int, start: int, key: int) -> tuple:
    """A broadcast data codeword's identity: its slot, and the slot its
    burst's data starts at (the control slots before it), in the direction
    byte's spare bits. A receiver that lost the control finds both."""
    return data_mask(0, slot, key, epoch=start)


def _ctl_indices(slot: int, n_ctl_slots: int) -> list[int]:
    """The control codewords (after the first) an OFDM burst's `slot` may
    hold at RV 0: i in slot i, or under DUP in slot 2i. CPM: none (its
    header says which slots are control)."""
    if n_ctl_slots:
        return []
    return sorted({i for i in (slot, slot // 2 if slot % 2 == 0 else 0) if 1 <= i < F.MAX_CTL})


# first data slots a burst may have: 1-4 control codewords, each sent once or twice (DUP)
DATA_STARTS = sorted({f * n for f in (1, 2) for n in range(1, F.MAX_CTL + 1)})
PROBE_SLOTS = 8  # control lost: data looked for in slots 1-8


def _framed(frame: bytes, at: int, pb: int) -> bytes:
    """`frame` as it goes at byte `at` of a burst's data: COBS, its ending
    zero, and a zero before it if it would start a codeword. Without that,
    the zero telling a reader where it starts would be the last byte of the
    codeword before, and the frame lost with that codeword."""
    return (b"" if at % pb else b"\0") + cobs(frame) + b"\0"


def pack_frames(frames: list[bytes], pb: int) -> list[bytes]:
    """Frames -> data codeword payloads of `pb` bytes: each frame COBS-encoded
    and ended by a zero, back to back, zero-padded (empty frames to a
    reader); a frame that would start a codeword gets a zero before it."""
    stream = b""
    for f in frames:
        stream += _framed(f, len(stream), pb)
    stream += bytes(-len(stream) % pb)
    return [stream[i:i + pb] for i in range(0, len(stream), pb)]


def unpack_frames(payloads: list[bytes], ok: list[bool]) -> tuple[list[bytes], int]:
    """Data codeword payloads and their CRC results -> (whole frames,
    frames lost). A zero byte in a good codeword always ends a frame, so
    parsing picks up again after a failed codeword. Lost counts each run of
    frames a failed codeword touched as one: at least that many."""
    stream = b"".join(payloads)
    good = np.repeat(np.array(ok, dtype=bool), [len(x) for x in payloads])
    out, lost, start, damaged = [], 0, 0, False
    for i in range(len(stream)):
        if not good[i]:
            damaged = True
        elif stream[i] == 0:
            if damaged:
                lost += 1
            elif i > start:
                f = uncobs(stream[start:i])
                if f is None:
                    lost += 1
                else:
                    out.append(f)
            start, damaged = i + 1, False
    return out, lost + damaged


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
    dup: bool = False  # each control codeword sent twice (BCAST MODE n DUP mode)
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
        check_mode(self.broadcast, self.cap)
        self.ports.setdefault(0, Port(PORT0_GROUP, self.broadcast))

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

    def set_mode(self, n: int, mode: str, auto: bool = False, dup: bool = False):
        """Port n's transmit mode for frames sent after it; with `auto`,
        rate shifting on and `mode` its fallback; with `dup`, its control
        codewords sent twice. ValueError if the port is closed, or the mode
        is outside the cap or too small for the control."""
        if n not in self.ports:
            raise ValueError(f"port {n} is not open")
        check_mode(mode, self.cap)
        self._check_fits(self.ports[n], mode, auto, n)
        self.ports[n].mode, self.ports[n].auto, self.ports[n].dup = mode, auto, dup

    def reset(self):
        """Ports as a new link has them: 1-15 closed (frames queued for them
        DROPPED, in port order), port 0 back in the broadcast mode."""
        for n in [n for n in self.ports if n]:
            self.close(n)
        self.set_mode(0, self.broadcast)

    def _check_fits(self, p: Port, mode: str, auto: bool, n: int):
        """The control a port needs (its group, and room for a report) must
        fit the mode's control codewords."""
        spec = MODES[mode]
        need = 1 + len(self._group_tlv(n, p)) + (len(_tlv(T_REPORTS, bytes(2))) if auto else 0)  # reports: the sender at least
        if need > min(F.MAX_CTL, max_ctl(spec)) * ctl_payload_bytes(spec):
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
            return  # nothing to send (and an empty frame reads as padding: unpack_frames)
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

    def missed(self, submode: str, n_cw: int, snr: float | None = None):
        """A burst heard whose header decoded but nothing else: it can't be
        tied to a port (docs/broadcast.md §5). `snr`: its in-band energy
        reading (dB, snr_est's scale), if one was made."""
        self.events.append(f"BCAST * MISSED {submode} {n_cw}" + (f" {snr:.1f}" if snr is not None else ""))

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
        pb, n_max = ctl_payload_bytes(spec), min(F.MAX_CTL, max_ctl(spec))
        body = self._group_tlv(n, port)
        if port.auto:
            now = self.clock()
            fresh = sorted(((h, p) for h, p in self.peers.items()
                            if now - p.heard <= HEARD_MAX_S and p.shifter.measured), key=lambda hp: -hp[1].heard)
            room = min((n_max * pb - 1 - len(body) - 4) // 3, (255 - 2) // 3)
            rep = struct.pack(">H", sender)
            for h, p in fresh[:max(0, room)]:
                rec, hint, _ = p.shifter.recommend(G.StationView(cap=self.cap))  # no session: an idle one
                rep += struct.pack(">HB", h, (rec << 2) | hint)
            body += _tlv(T_REPORTS, rep)
        n_ctl = max(1, -(-(1 + len(body)) // pb))
        stream = bytes([VERSION << 4 | (DUP if port.dup else 0) | (n_ctl - 1)]) + body
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
        start = (1 + port.dup) * len(ctl)  # control slots: the first data slot
        seconds = G.SIZE_S[hint] if mode != port.mode else BROADCAST_S
        pb = codes.payload_bytes(spec)
        # the size class is a preference: a burst grows to carry its first frame
        # (a 256-byte PACLEN I frame outgrew a 12 s qpsk-r1/5 burst), up to what
        # its header can say
        limit = start + cpm.MAX_DATA if is_cpm(spec) else max_codewords(spec.sync_band)
        need = start + -(-len(_framed(first, 0, pb)) // pb)
        n_max = min(limit, max(G.slots_for(spec, seconds), need))
        room = (n_max - start) * pb
        taken, rest, size = [], [], 0
        for q in self.queue:
            k = len(_framed(q[1], size, pb))
            if q[0] == n and self._route(port, q[1])[0] == mode and size + k <= room:
                taken.append(q)
                size += k
            else:
                rest.append(q)
        if not taken:  # the first frame alone doesn't fit: it never will in this mode
            log.error("%d-byte frame dropped: a %s burst carries at most %d", len(first), mode, room - 3)
            self.queue = self.queue[1:]
            self.events.append(f"BCAST {n} DROPPED 1")
            return self.next_burst()
        self.queue = rest
        data = pack_frames([q[1] for q in taken], pb)
        key = port.key
        slots = [Slot(ctl_mask(0, i, key), rv, p) for i, p in enumerate(ctl) for rv in range(1 + port.dup)]
        slots += [Slot(data_start_mask(start + j, start, key), 0, p) for j, p in enumerate(data)]
        self.n_sent += 1
        burst = TxBurst(mode, slots, self.n_sent, self.cap)
        self._inflight = (burst, [(n, q[2]) for q in taken if q[2] is not None])
        return burst

    # -- receiving -----------------------------------------------------------

    @staticmethod
    def _control_reads(rx):
        """Codeword 0's readings, each with whether it took the RV 1 copy
        too: slot 0 alone, then slots 0 and 1 combined (DUP) when slot 0
        alone didn't decode (OFDM: no LDPC candidate converged; CPM: its
        header says two control slots), so other bursts cost no decode more."""
        alone = rx.raw(0)
        yield from ((c, False) for c in alone)
        if rx.n_ctl_slots == 2 or (not rx.n_ctl_slots and not alone):
            yield from ((c, True) for c in rx.raw_pair(0))

    @staticmethod
    def read_burst_control(rx, n_cw: int):
        """The burst's control, checked under the key of the group it names
        -> (group, sender's call, reports, the first data slot), or None."""
        for c0, paired in KissLink._control_reads(rx):
            if c0[0] >> 4 != VERSION:
                continue
            f = 2 if c0[0] & DUP else 1
            n_ctl = (c0[0] & 3) + 1
            if paired and f == 1:
                continue  # combined as a pair, but not sent as one
            if f * n_ctl > n_cw or (rx.n_ctl_slots and rx.n_ctl_slots != f * n_ctl):
                continue
            rest = [(rx.raw(f * i) or [bytes(len(c0))])[0] for i in range(1, n_ctl)]  # the best guess, to read
            try:
                group = read_control(parse_tlvs(b"".join([c0] + rest)[1:]))[0]
                key = group_key(group)
            except ValueError:
                continue
            ctl = []
            for i in range(n_ctl):
                p = rx.decode(f * i, ctl_mask(0, i, key), 0, None)
                if p is None and f == 2:
                    p = rx.decode_pair(f * i, ctl_mask(0, i, key))
                ctl.append(p)
            if None in ctl or ctl[0] != c0:
                continue
            try:
                return (*read_control(parse_tlvs(b"".join(ctl)[1:])), f * n_ctl)
            except ValueError as e:
                log.warning("RX malformed broadcast control (%s): dropped", e)
                return None
        return None

    @staticmethod
    def find_data(rx, keys: list[int]) -> tuple[int, int] | None:
        """A burst whose control was lost: the first of slots 1-8 that
        passes as data under one of `keys`, its mask naming where the data
        starts -> (key, first data slot), or None. Each slot decodes once;
        every (start, key) is a CRC check, and a converged codeword of
        another burst passes one 1 time in 65536 (CRC-16). So a slot alone
        is believed only where one check per key was all it took before:
        slot 1 starting the data, a burst's only data slot, or the start
        CPM's header gives; elsewhere a later slot must pass under the same
        key and start too."""
        starts = [rx.n_ctl_slots] if rx.n_ctl_slots else DATA_STARTS
        for j in range(1, min(rx.n_cw, PROBE_SLOTS + 1)):
            for start in (s for s in starts if s <= j):
                for k in keys:
                    if rx.decode(j, data_start_mask(j, start, k), 0, None) is None:
                        continue
                    alone = j == start and (start == 1 or start == rx.n_cw - 1 or rx.n_ctl_slots)
                    if alone or any(rx.decode(i, data_start_mask(i, start, k), 0, None) is not None
                                    for i in range(j + 1, rx.n_cw)):
                        return k, start
            # a codeword that decoded but is neither an open port's data nor the
            # rest of its control (OFDM: control codeword 1-3 at RV 0, in slot i
            # or under DUP slot 2i): another burst, so no slot further on is looked at
            if rx.converged(j) and not any(rx.decode(j, ctl_mask(0, i, k), 0, None) is not None
                                           for i in _ctl_indices(j, rx.n_ctl_slots) for k in keys):
                return None
        return None

    @staticmethod
    def read_burst_frames(rx, start: int, key: int) -> tuple[list, list, int]:
        """Data slots from `start` under `key` -> (frames, per-slot ok, frames lost)."""
        payloads, ok = [], []
        for i in range(start, rx.n_cw):
            p = rx.decode(i, data_start_mask(i, start, key), 0, None)
            payloads.append(p or bytes(codes.payload_bytes(rx.spec)))
            ok.append(p is not None)
        frames, lost = unpack_frames(payloads, ok) if payloads else ([], 0)
        return frames, ok, lost

    def on_burst(self, r: dict, rx=None) -> list[tuple[int, bytes]] | None:
        """A received burst (modem.receive's or cpm.receive's dict; `rx` its
        phy.ModemRx, shared with the engine's other checks) -> [(port,
        frame)] for the open ports it belongs to ([] for another group's), or
        None if it isn't a broadcast burst (an ARQ burst, or one lost whole).
        Statuses go to `events`; a shifting port learns from its sender."""
        rx = rx or PHY.ModemRx(r, {}, PHY.DD_BUDGET_S)  # never DD unbounded
        now = self.clock()
        ctl = self.read_burst_control(rx, r["n_cw"])
        if ctl is not None:
            group, call, rep, start = ctl
            frames, ok, lost = self.read_burst_frames(rx, start, group_key(group))
            ports = sorted(i for i, p in self.ports.items() if p.group == group)
            if rep is not None and any(self.ports[i].auto for i in ports):
                self._learn(rep, frames, r, ok, now)
            snr = _snr(r) if ports else ""
            for i in ports:
                self.events.append(f"BCAST {i} HEARD" + (f" {call}" if call else "") + snr)
                if lost:
                    self.events.append(f"BCAST {i} LOST {lost}")
            return [(i, f) for i in ports for f in frames]
        # control lost: a data slot passing under an open port's key says the
        # burst is that group's, and its mask where the data starts (docs/broadcast.md §2).
        # OFDM: a slot 0 that decoded (alone or paired) but wasn't a broadcast
        # control is another kind of burst (ARQ, CQ), not one whose control was lost
        if not rx.n_ctl_slots and (rx.converged(0) or rx.converged(0, pair=True)):
            return None
        found = self.find_data(rx, sorted({p.key for p in self.ports.values()}))
        if found is None:
            return None
        key, start = found
        frames, _, lost = self.read_burst_frames(rx, start, key)
        ports = sorted(i for i, p in self.ports.items() if p.key == key)
        if len({self.ports[i].group for i in ports}) > 1:  # two groups, one key: whose it is can't be told
            log.warning("broadcast burst under a key %d open groups share: dropped", len(ports))
            for i in ports:
                self.events.append(f"BCAST {i} LOST {len(frames) + lost}")
            return []
        snr = _snr(r) if ports else ""
        for i in ports:
            self.events.append(f"BCAST {i} HEARD{snr}")
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
