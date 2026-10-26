"""The Data2G server: one radio, one integrated server, every port always on.

ARQ sessions (data2g.arq) on two TCP ports, as VARA: commands
(8300, CR-terminated lines) and data (8301, the session's byte stream).

Broadcast (docs/broadcast.md): frames on KISS port 8100 (VARA HF's KISS
port), each KISS port a named group (data2g.kisslink); port 0 is always
open, on "KISS 0". Broadcast bursts go out only between ARQ sessions and
while the channel is free; each burst heard goes to whichever it belongs
to. The station itself is an arq.engine.Engine clocked by the sound card;
PTT through rigctld; every burst heard and sent recorded for offline
replay (scripts/replay.py).

Commands: MYCALL call, LISTEN ON|OFF, ACCEPT [host]:port (Data2G: LISTEN ON,
but each session's data goes over a connection Data2G makes to host:port,
default the command client's own, instead of the data port; a connect is refused if that
fails, and the session disconnects when the far end closes it; LISTEN ON|OFF
ends it), CONNECT from to, DISCONNECT, ABORT,
CQFRAME call bw (a CQ frame at 500 | 1200 | 2300 | 2750 Hz: no session;
one heard is notified the same way, the sender's call and bandwidth),
BW500 | BW1200 | BW2300 | BW2750 (session bandwidth cap: BW500 keeps a
session inside a 500 Hz band-plan segment; BW2300/BW2750 are the full
2400 Hz; BW1200 is a Data2G extension), CHAT ON|OFF, VERSION, SN (-> SN
snr: the pilot SNR, dB, of the session peer's last burst decoded; WRONG
before one), CLEANTXBUFFER (Host._clean), DATA2G ON|OFF (a client that
knows Data2G: BUFFER is the plain VARA count, CLEANTXBUFFER reports host
byte offsets, data written before CONNECTED is dropped; off again when the
client goes), TXPOWER [dB] (DATA2G ON: -> TXPOWER db, the output volume;
with dB, -60 to 20, sets it -> OK); broadcast:
BCAST OPEN group [FROM call] (-> BCAST PORT n), BCAST CLOSE n, BCAST MODE n
[AUTO] [DUP] mode, MODES (MODE name bandwidth bytes-per-codeword max-codewords
seconds-at-1 seconds-at-max lines). Replies OK / WRONG (MYCALL's OK followed by
REGISTERED call for each call). Notifications: CONNECTED src dst bw, DISCONNECTED
(DATA2G ON: DISCONNECTED reason [detail], docs/arq.md §9),
PTT ON|OFF, BUSY ON|OFF, BUFFER n, IAMALIVE (every 60 s), and (Data2G)
MODE submode; broadcast statuses (BCAST n HEARD [call] snr, BCAST n LOST k,
BCAST * MISSED submode n_cw [snr], BCAST n DROPPED k) once the client has sent a
BCAST command, so VARA clients never see them.

    data2g-host --mycall W1AW --input-device USB --output-device USB --rigctld-port 4532

Native: the Host (command semantics) is native/core/host; serve() is
native/app/station.cpp and main() native/apps/data2g_host.cpp. Under
`pytest --native` tests/native_subs/host.py substitutes Host.
"""

from . import __version__, threads

threads.cap_env(1)  # before numpy loads: receiver.py has the reasoning

import argparse  # noqa: E402
import itertools  # noqa: E402
import logging  # noqa: E402
import queue  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from . import audio  # noqa: E402
from .arq import modes  # noqa: E402
from .arq import session as S  # noqa: E402
from .arq.engine import Engine  # noqa: E402
from .config import FS  # noqa: E402
from .tnc import KissServer, Rigctld  # noqa: E402

log = logging.getLogger("data2g.host")
VERSION = f"Data2G {__version__}"
# caps are policy.CAP_HZ's keys (0: 500 Hz, 1: 1200, 2: 2400)
BW = {"BW500": 0, "BW1200": 1, "BW2300": 2, "BW2750": 2}
BW_NAME = modes.BW_NAME
KISS_BW_CAP = {2400: 2, 500: 0}  # --kiss-bw -> the KISS link's cap
BLOCK = FS // 10  # audio block: 0.1 s
VOLUME_DB = (-60.0, 20.0)  # TXPOWER's range: the native GUI's output volume setting
DIAL_TIMEOUT_S = 1.0  # ACCEPT's connection must be up by then (the caller waits ~3.5 s for an answer)
ALIVE_S = 60.0  # IAMALIVE on the command port this often, as VARA does
BUFFER_REPEAT_S = 30.0  # a nonzero BUFFER is repeated this often
# VARA settings a client may send that have no Data2G meaning (yet): OK, logged.
# ponytail: from memory of VARA clients, not a checked list; check the log on first contact with Pat
IGNORED = {"COMPRESSION", "PUBLIC", "CWID", "P2P", "WINLINK", "REGISTERED", "ENCRYPTION", "IGNOREKISSDCD"}


def mode_lines(cap: int) -> list[str]:
    """MODES: one line per mode within the cap, narrowest first: name,
    bandwidth (Hz), bytes per codeword, max codewords (what a receiver
    takes: engine.receive_accept), and airtime (s) at 1 and at max codewords."""
    from . import codes, cpm
    from .arq import policy as G
    from .arq.engine import receive_accept
    from .arq.modes import burst_seconds, is_cpm

    rx = dict(receive_accept().max_cw)
    out = []
    for s in sorted(G.allowed(cap), key=lambda s: (G.width_hz(s), s.name)):
        n = 1 + cpm.MAX_DATA if is_cpm(s) else rx[s.name]
        out.append(f"MODE {s.name} {G.width_hz(s):.0f} {codes.payload_bytes(s)} {n} "
                   f"{burst_seconds(s, 1):.2f} {burst_seconds(s, n):.2f}")
    return out


class Host:
    """VARA command semantics over an Engine: no sockets, no audio (those
    are serve()'s). command() and data_in() take the client's side;
    after_step() collects what goes back after each Engine.step."""

    def __init__(self, engine: Engine, buffer_credit: int | None = None):
        """`buffer_credit`: queued bytes BUFFER may leave out, at most (None:
        the next burst's whole capacity; 0: plain VARA, all of them)."""
        self.engine, self.cap, self.listening = engine, 2, False
        self.accept_addr: tuple[str, int] | None = None  # ACCEPT host:port
        self.dial = None  # serve(): (host, port) -> True once a connection there carries the session's data
        engine.set_accept_connect(self._accept_connect)
        self.buffer_credit = buffer_credit
        self._bcast = False  # the client has sent BCAST: broadcast statuses go to it
        self.data2g = False  # DATA2G ON
        self.output_volume = 0.0  # dB, TXPOWER: serve() applies it
        self._up = False  # CONNECTED sent, DISCONNECTED not yet (DATA2G ON: data before it is dropped)
        self.out_cmd: list[str] = []
        self.out_data = bytearray()
        self._ptt = self._busy = False
        self._buffer = 0
        self._buffer_t = 0.0  # engine time of the last BUFFER line
        self._mode = None
        self._alive = 0.0  # engine time of the last IAMALIVE

    def command(self, line: str, peer: str = "127.0.0.1"):
        """`peer`: the address the command came from (ACCEPT's default host)."""
        words = line.strip().split()
        if not words:
            return
        cmd, args = words[0].upper(), words[1:]
        e = self.engine
        ok = True
        if cmd == "MYCALL" and len(args) >= 1:
            try:
                e.set_call(*args)  # VARA takes several: all answer connects
            except ValueError:
                ok = False
            else:  # VARA tells a registered station so, per call; to us, every station is
                self.out_cmd += ["OK"] + [f"REGISTERED {c}" for c in (e.call, *e.aliases)]
                return
        elif cmd == "LISTEN" and args and args[0].upper() == "CQ":
            pass  # CQ frames are always reported
        elif cmd == "LISTEN" and args and args[0].upper() in ("ON", "OFF"):
            self.listening, self.accept_addr = args[0].upper() == "ON", None
            e.listen(self.listening)
        elif cmd == "ACCEPT" and len(args) == 1 and (addr := _addr(args[0], peer)):
            self.listening, self.accept_addr = True, addr
            e.listen()
        elif cmd == "CONNECT" and len(args) == 2:
            try:
                e.set_call(args[0], *e.aliases)
                e.connect(args[1], self.cap)
            except (RuntimeError, ValueError):
                ok = False
        elif cmd == "DISCONNECT":
            e.disconnect()
        elif cmd == "ABORT":
            self._session_over()
            e.abort()
            self._up = False
            self.out_cmd.append("DISCONNECTED ABORT" if self.data2g else "DISCONNECTED")
            if self.listening:
                e.listen()
        elif cmd == "CQFRAME" and len(args) == 2 and "BW" + args[1] in BW:
            try:
                e.send_cq(args[0], BW["BW" + args[1]])
            except (RuntimeError, ValueError):  # a session under way; a callsign that won't pack
                ok = False
        elif cmd in BW:
            self.cap = BW[cmd]
        elif cmd == "CHAT" and args and args[0].upper() in ("ON", "OFF"):
            e.set_chat(args[0].upper() == "ON")
        elif cmd == "DATA2G" and args and args[0].upper() in ("ON", "OFF"):
            self.data2g = args[0].upper() == "ON"
        elif cmd == "TXPOWER" and self.data2g and not args:
            self.out_cmd.append(f"TXPOWER {self.output_volume:.1f}")
            return
        elif cmd == "TXPOWER" and self.data2g and len(args) == 1:
            try:
                v = float(args[0])
            except ValueError:
                v = float("nan")
            ok = VOLUME_DB[0] <= v <= VOLUME_DB[1]
            if ok:
                self.output_volume = v
        elif cmd == "CLEANTXBUFFER" and (not args or (self.data2g and len(args) == 1 and args[0].isdigit()
                                                      and len(args[0]) <= 18)):
            self.out_cmd.append(self._clean(int(args[0]) if args else None))
            return
        elif cmd == "BCAST" and e.kiss is not None:
            self._bcast = True
            reply = self._bcast_command(args)
            self.out_cmd.append(reply or "WRONG")
            return
        elif cmd == "MODES" and e.kiss is not None:
            self.out_cmd += mode_lines(e.kiss.cap)
        elif cmd in IGNORED:
            log.info("accepted, not implemented: %s", line.strip())
        elif cmd == "VERSION":
            self.out_cmd.append(f"VERSION {VERSION}")
            return
        elif cmd == "SN" and e.session_snr is not None:
            self.out_cmd.append(f"SN {e.session_snr:.1f}")
            return
        else:
            ok = False
        self.out_cmd.append("OK" if ok else "WRONG")

    def _bcast_command(self, args: list[str]) -> str | None:
        """BCAST OPEN group [FROM call] | CLOSE n | MODE n [AUTO] [DUP] mode ->
        the reply, or None (WRONG). A group may be several words."""
        link, kw = self.engine.kiss, [w.upper() for w in args]
        try:
            if kw[0] == "OPEN" and len(args) >= 2:
                words, call = args[1:], None
                if len(words) >= 3 and kw[-2] == "FROM":
                    words, call = words[:-2], words[-1]
                return f"BCAST PORT {link.open(' '.join(words), call)}"
            if kw[0] == "CLOSE" and len(args) == 2:
                link.close(int(args[1]))
                return "OK"
            flags = kw[2:-1]
            if kw[0] == "MODE" and len(args) >= 3 and set(flags) <= {"AUTO", "DUP"} and len(set(flags)) == len(flags):
                link.set_mode(int(args[1]), args[-1].lower(), "AUTO" in flags, "DUP" in flags)
                return "OK"
        except (ValueError, IndexError):
            pass
        return None

    def _clean(self, upto: int | None = None) -> str:
        """CLEANTXBUFFER [n] -> the reply (docs/arq.md §9): VARA's
        BUFFEREMPTY, FAILED (its trailing space too) or OK; DATA2G ON, OK
        with host byte offsets, or LATE for an n already on air. Bytes on air
        but unacked stay in limbo until an ACK settles them (SETTLED, from
        after_step or _session_over); VARA clients see them gone at once."""
        s = self.engine.session
        if not s.queued_bytes(not self.data2g)[1]:
            return "CLEANTXBUFFER BUFFEREMPTY"
        if s.state != S.CONNECTED or s.station is None:
            return "CLEANTXBUFFER FAILED "
        r = s.station.tx.clean(upto)
        if r is None:
            return "CLEANTXBUFFER LATE"
        return "CLEANTXBUFFER OK " + " ".join(map(str, r)) if self.data2g else "CLEANTXBUFFER OK"

    def _session_over(self):
        """The session is ending (the engine replaces it next): a limbo still
        unsettled goes as CLEANTXBUFFER SETTLED from split to UNKNOWN."""
        st = self.engine.session.station
        if st is not None and (r := st.tx.end_limbo()) is not None and self.data2g:
            self.out_cmd.append(f"CLEANTXBUFFER SETTLED {r[0]} {r[1]} {r[2]} UNKNOWN")

    def client_gone(self):
        """The command client's TCP connection closed: it owned the session.
        A session under way is disconnected gracefully (the peer is told, and
        what's queued still goes); a call being placed is dropped; listening
        stops, so no connect is accepted that nobody will serve (clients send
        LISTEN ON again when they reconnect)."""
        e = self.engine
        log.info("command client gone")
        if e.session.state in (S.CONNECTED, S.DISCONNECTING):
            e.disconnect()
        elif e.session.state == S.CONNECTING:
            e.abort()
        self.listening, self.accept_addr, self.data2g = False, None, False
        if e.session.state in (S.LISTEN, S.CLOSED):
            e.listen(False)
        if e.kiss is not None:  # its ports go, and port 0 back to its defaults
            e.kiss.reset()
            self._bcast = False

    def _accept_connect(self, caller: str, callee: str) -> bool:
        if self.accept_addr is None:
            return True
        log.info("ACCEPT: %s calling %s, connecting to %s:%d", caller, callee, *self.accept_addr)
        return self.dial is not None and self.dial(self.accept_addr)

    def data_in(self, data: bytes):
        if self.data2g and not self._up:
            # VARA queues it for the session; a Data2G client's offsets start at CONNECTED
            log.info("DATA2G: %d B before CONNECTED dropped", len(data))
        else:
            self.engine.session.write(data)
        # always answer data with a BUFFER line, changed or not: Pat counts
        # what it wrote until one arrives, and blocks once that count passes
        # 7x its next write (it waited forever on a 6-byte B2F line)
        self._buffer = None

    def after_step(self, ptt: bool):
        e = self.engine
        if e.now - self._alive >= ALIVE_S:  # some VARA clients count on it
            self._alive = e.now
            self.out_cmd.append("IAMALIVE")
        st = e.session.station
        if st is not None and (c := st.tx.cleaned) is not None:  # before a DISCONNECTED below replaces the session
            if self.data2g:
                self.out_cmd.append(f"CLEANTXBUFFER SETTLED {c[0]} {c[1]} {c[2]}")
            st.tx.cleaned = None
        for ev in e.events():
            if ev.startswith("CONNECTED"):
                peer = ev.split()[1]
                me = e.session.call  # the call dialed, when MYCALL gave several
                src, dst = (me, peer) if e.session.master else (peer, me)
                self.out_cmd.append(f"CONNECTED {src} {dst} {BW_NAME[e.session.cap]}")
                self._up = True
            elif ev.startswith("CQFRAME"):
                _, call, cap = ev.split()
                self.out_cmd.append(f"CQFRAME {call} {BW_NAME[int(cap)]}")
            elif ev.startswith("DISCONNECTED"):
                log.info("%s", ev)
                self._session_over()
                self._up = False
                self.out_cmd.append(ev if self.data2g else "DISCONNECTED")
        if e.session.state == S.CLOSED and self.listening:
            e.listen()
        self.out_data += e.session.read()
        if e.kiss is not None:
            events = e.kiss.take_events()
            if self._bcast:
                self.out_cmd += events
        if ptt != self._ptt:
            self._ptt = ptt
            self.out_cmd.append("PTT ON" if ptt else "PTT OFF")
            if ptt and e.tx_submode is not None and e.tx_submode != self._mode:
                self._mode = e.tx_submode
                self.out_cmd.append(f"MODE {self._mode}")
        if e.channel_busy != self._busy:
            self._busy = e.channel_busy
            self.out_cmd.append("BUSY ON" if self._busy else "BUSY OFF")
        # BUFFER, as VARA defines it: bytes the peer hasn't acknowledged yet,
        # sent or not. Pat blocks writes while BUFFER >= 7x its write (<= 250
        # B B2F blocks), tuned to VARA's short frames, and its Flush (after
        # every message) waits for BUFFER 0. Counting everything kept Data2G
        # bursts (up to ~12 KB) short: a 20 KB Pat transfer took 5x as long
        # as the same bytes written at once. So while more than the next
        # burst waits, BUFFER is the unsent bytes past it (the credit); once
        # it has all gone, 1 until the peer acks it all, so Flush still waits
        # for delivery but writes never block. --buffer-credit 0 reports the
        # plain VARA figure. CHAT ON and DATA2G ON also report it: the client
        # watches BUFFER fall to see which of its messages were ACKed. Host
        # bytes, as VARA's; the credit is payload bytes, a length byte per
        # record off, which no client can see.
        st = e.session.station
        unsent, unacked = e.session.queued_bytes(not self.data2g)
        buffered = unacked
        if (st and self.buffer_credit != 0 and not e.session.chat and not self.data2g
                and (credit := e.session.policy.next_capacity(st)) is not None):
            if self.buffer_credit is not None:
                credit = min(credit, self.buffer_credit)
            buffered = unsent - credit if unsent > credit else min(unacked, 1)
        # Pat's write and Flush give up after a minute without a BUFFER line
        if buffered != self._buffer or (buffered and e.now - self._buffer_t >= BUFFER_REPEAT_S):
            self._buffer = buffered
            self._buffer_t = e.now
            self.out_cmd.append(f"BUFFER {buffered}")


def _addr(s: str, default_host: str) -> tuple[str, int] | None:
    """ACCEPT's [host]:port, or None."""
    host, sep, port = s.rpartition(":")
    if not sep or not port.isdigit() or not 0 < int(port) < 65536:
        return None
    return host.strip("[]") or default_host, int(port)


# --- sockets ------------------------------------------------------------------------

def _shut(c):
    """Close a socket another thread may be blocked reading: close() alone would hold the connection open."""
    try:
        c.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    c.close()


class _Port:
    """One TCP listener; lines or bytes go to `on_input(item, client_id)`. With `multi` any number of clients may be
    connected and `on_close` fires when the last leaves; otherwise a new client replaces the old. send() goes to
    every client, or to one by id (nowhere if it has gone)."""

    def __init__(self, addr, on_input, lines: bool, on_close=None, multi: bool = False):
        self.on_input, self.lines, self.on_close, self.multi = on_input, lines, on_close, multi
        self.clients: dict[int, socket.socket] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.srv = socket.create_server(addr, reuse_port=False)
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                c, peer = self.srv.accept()
            except OSError:
                return
            log.info("client %s:%d on port %d", *peer, self.srv.getsockname()[1])
            cid = next(self._ids)
            with self._lock:
                old = [] if self.multi else list(self.clients.values())  # the new client first: the old one's close isn't a loss
                self.clients = {**self.clients, cid: c} if self.multi else {cid: c}
            for o in old:
                _shut(o)
            threading.Thread(target=self._read, args=(cid, c), daemon=True).start()

    def _read(self, cid, c):
        buf = b""
        while True:
            try:
                d = c.recv(4096)
            except OSError:
                d = b""
            if not d:
                self._drop(cid, c)
                return
            if not self.lines:
                self.on_input(d, cid)
                continue
            buf += d
            if len(buf) > 65536 and b"\r" not in buf and b"\n" not in buf:
                buf = b""  # a client that never ends its line
            while b"\r" in buf or b"\n" in buf:
                i = min(x for x in (buf.find(b"\r"), buf.find(b"\n")) if x >= 0)
                line, buf = buf[:i], buf[i + 1:]
                if line.strip():
                    self.on_input(line.decode("ascii", "replace"), cid)

    def peer(self, cid: int) -> str:
        """Client `cid`'s address (127.0.0.1 if it has gone)."""
        try:
            return self.clients[cid].getpeername()[0]
        except (KeyError, OSError):
            return "127.0.0.1"

    def send(self, data: bytes, to: int | None = None):
        if not data:
            return
        for cid, c in list(self.clients.items()):
            if to is not None and cid != to:
                continue
            try:
                c.sendall(data)
            except OSError:
                self._drop(cid, c)

    def _drop(self, cid, c):
        """Forget a client and tell the host if it was the last, once, whichever of recv or send saw it go."""
        with self._lock:
            if self.clients.get(cid) is not c:
                return
            self.clients = {k: v for k, v in self.clients.items() if k != cid}
            last = not self.clients
        c.close()
        if last and self.on_close:
            self.on_close()

    def close(self):
        self.srv.close()
        for c in self.clients.values():
            _shut(c)


class _Dialer:
    """ACCEPT: the session's own connection, made when a call is answered.
    What it reads goes to `inbox` as ("dialed", (sock, bytes)), b"" when it closed."""

    def __init__(self, inbox: queue.Queue):
        self.inbox, self.sock = inbox, None

    def __call__(self, addr: tuple[str, int]) -> bool:
        self.close()
        try:
            # ponytail: blocks the audio loop up to DIAL_TIMEOUT_S; a connect thread if a slow far end matters
            c = socket.create_connection(addr, timeout=DIAL_TIMEOUT_S)
        except OSError as ex:
            log.warning("ACCEPT: can't connect to %s:%d: %s", *addr, ex)
            return False
        c.settimeout(None)
        self.sock = c
        threading.Thread(target=self._read, args=(c,), daemon=True).start()
        return True

    def _read(self, c):
        while True:
            try:
                d = c.recv(4096)
            except OSError:
                d = b""
            self.inbox.put(("dialed", (c, d)))
            if not d:
                return

    def send(self, data: bytes):
        try:
            self.sock.sendall(data)
        except OSError:
            pass  # its reader sees it go

    def close(self):
        if self.sock is not None:
            _shut(self.sock)
            self.sock = None


def _send_cmd(port, host, asker=None):
    """Send the lines the host has queued. Those a command from client `asker` produced are its reply and go to it
    alone, except DISCONNECTED (ABORT's), which is news for every client; with no asker everything is for all."""
    for line in host.out_cmd:
        port.send(line.encode() + b"\r", None if line.startswith("DISCONNECTED") else asker)
    host.out_cmd.clear()


def serve(a, pa, stop: threading.Event | None = None):
    """Run until SIGINT/SIGTERM (or `stop` is set)."""
    from .kisslink import KissLink

    link = KissLink(cap=KISS_BW_CAP[a.kiss_bw], broadcast=a.broadcast_mode, busy_limit_s=a.kiss_busy_limit)
    engine = Engine(a.mycall or "NOCALL", ptt_delay_s=a.ptt_on_delay_ms / 1000, record_dir=a.record_dir,
                    min_header_score=a.min_header_score, kiss=link, stats_interval_s=a.stats_interval)
    host = Host(engine, None if a.buffer_credit < 0 else a.buffer_credit)
    inbox: queue.Queue = queue.Queue()
    cmd = _Port((a.host, a.command_port), lambda line, cid: inbox.put(("cmd", (line, cid))), lines=True,
                on_close=lambda: inbox.put(("gone", None)), multi=True)
    data = _Port((a.host, a.command_port + 1), lambda d, cid: inbox.put(("data", d)), lines=False)
    host.dial = dialer = _Dialer(inbox)
    kiss = KissServer((a.kiss_address, a.kiss_port), lambda *f: inbox.put(("kiss", f)), link.command)
    threading.Thread(target=kiss.serve_forever, name="kiss", daemon=True).start()
    rig = Rigctld(a.rigctld_host, a.rigctld_port)
    dec, interp = audio.Decimator(a.sample_rate), audio.Interpolator(a.sample_rate)
    per = BLOCK * (a.sample_rate // FS)
    # Stereo where the device has it (left channel in, the same signal on both out): PortAudio's ALSA
    # host API corrupts the heap reading or writing a mono stream on a PipeWire device that has more
    # channels (glibc aborts in the first read, e.g. "malloc(): invalid size (unsorted)").
    nin, nout = audio.channels(pa, a.input_device, "input"), audio.channels(pa, a.output_device, "output")
    inp = audio.Capture(pa, nin, a.sample_rate, a.input_device)
    out = audio.Player(pa, nout, a.sample_rate, a.output_device, a.tx_lead_ms / 1000)
    host.output_volume = a.output_volume
    stop = stop or threading.Event()
    if threading.current_thread() is threading.main_thread():
        # SIGHUP: a closed terminal; the finally below must still unkey
        for sig in (signal.SIGINT, signal.SIGTERM) + ((signal.SIGHUP,) if hasattr(signal, "SIGHUP") else ()):
            signal.signal(sig, lambda *_: stop.set())
    log.info("commands on %s:%d, data on %d", a.host, a.command_port, a.command_port + 1)
    log.info("KISS on %s:%d: %d Hz cap, broadcasts in %s", a.kiss_address, a.kiss_port, a.kiss_bw, link.broadcast)
    log.info("recording to %s", a.record_dir or "(off)")
    keyed, off_pending, slow = False, False, 0
    try:
        while not stop.is_set():
            while not inbox.empty():
                kind, v = inbox.get()
                if kind == "cmd":
                    line, cid = v
                    log.info("command: %s", line)
                    _send_cmd(cmd, host)  # what was already pending goes first, to everyone
                    host.command(line, cmd.peer(cid))
                    _send_cmd(cmd, host, cid)  # the reply goes to the client that asked
                elif kind == "gone":
                    host.client_gone()
                elif kind == "dialed":
                    c, d = v
                    if c is not dialer.sock:
                        continue  # a closed session's
                    if d:
                        host.data_in(d)
                    else:
                        log.info("ACCEPT: the far end closed: disconnecting")
                        engine.disconnect()
                elif kind == "kiss":
                    port, frame, ack = v
                    link.enqueue(frame, port, ack)
                else:
                    host.data_in(v)
            _send_cmd(cmd, host)  # replies at once, not after the audio step (clients time out at ~2 s)
            x = inp.read(per, stop)
            if x is None:
                break
            t0 = time.perf_counter()
            y, ptt = engine.step(dec(x) if not keyed else np.zeros(BLOCK))
            if time.perf_counter() - t0 > BLOCK / FS:
                slow += 1
                log.debug("step took %.2f s (%d slow)", time.perf_counter() - t0, slow)
            if ptt and not keyed:
                rig.ptt(True)
                off_pending = False
                out.start()
                keyed = True
            if keyed:
                out.write(np.clip(interp(y) * 10 ** (host.output_volume / 20), -1, 1))
            if keyed and not ptt:
                out.drain()
                time.sleep(a.ptt_off_delay_ms / 1000)
                keyed = False
                off_pending = not rig.ptt(False)
            elif off_pending and not keyed:
                off_pending = not rig.ptt(False)  # a PTT off that failed: until it lands
            host.after_step(ptt)
            for line in host.out_cmd:
                cmd.send(line.encode() + b"\r")
            host.out_cmd.clear()
            if dialer.sock is not None:
                dialer.send(bytes(host.out_data))
                if engine.session.state not in (S.CONNECTED, S.DISCONNECTING):
                    dialer.close()  # the session is over
            else:
                data.send(bytes(host.out_data))
            host.out_data.clear()
            for port, f in engine.take_kiss_rx():
                kiss.broadcast(f, port)
            for port, ack in link.take_acks():
                kiss.send_ack(ack, port)
    finally:
        log.info("shutting down")
        rig.release()
        cmd.close()
        data.close()
        dialer.close()
        kiss.shutdown()
        kiss.close_clients()
        kiss.server_close()
        inp.close()
        out.close()


def main():
    ap = argparse.ArgumentParser(description="Data2G server: VARA-style ARQ sessions and KISS broadcast on one radio")
    ap.add_argument("--kiss-port", type=int, default=8100, help="as VARA HF's")
    ap.add_argument("--kiss-address", default="127.0.0.1")
    ap.add_argument("--kiss-busy-limit", type=float, default=60.0, metavar="S",
                    help="a KISS burst held this long by BUSY is sent anyway")
    ap.add_argument("--kiss-bw", type=int, choices=(2400, 500), default=2400, help="KISS bandwidth cap, Hz")
    ap.add_argument("--broadcast-mode", metavar="MODE",
                    help="a broadcast port's transmit mode until BCAST MODE sets one "
                         "(default: qpsk-r1/5, n10-qpsk-r1/5 with --kiss-bw 500)")
    ap.add_argument("--mycall")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--command-port", type=int, default=8300, help="data is on the next port")
    ap.add_argument("--list-audio-devices", action="store_true")
    ap.add_argument("--input-device")
    ap.add_argument("--output-device")
    ap.add_argument("--sample-rate", type=int, default=48000, help="a multiple of 8000")
    ap.add_argument("--output-volume", type=float, default=0.0, help="dB; 0 puts a burst's peak at full scale")
    ap.add_argument("--rigctld-host", default="localhost")
    ap.add_argument("--rigctld-port", type=int, default=4532, help="0: no PTT")
    ap.add_argument("--ptt-on-delay-ms", type=int, default=100)
    ap.add_argument("--ptt-off-delay-ms", type=int, default=50)
    ap.add_argument("--tx-lead-ms", type=int, default=100,
                    help="TX audio queued ahead of the sound card: slack for a late audio step")
    ap.add_argument("--min-header-score", type=float, default=0.0)
    ap.add_argument("--buffer-credit", type=int, default=-1,
                    help="bytes queued for the next burst that BUFFER leaves out, at most, so VARA clients "
                         "that throttle on it (Pat) keep a whole burst queued; -1: the next burst's full "
                         "capacity, 0: report every unacked byte (plain VARA)")
    ap.add_argument("--record-dir", default=f"recordings/{time.strftime('%Y%m%d-%H%M%S')}",
                    help="where every burst heard and sent is logged ('' turns it off)")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--stats-interval", type=float, default=60.0, metavar="S",
                    help="log a connection's throughput this often (0: only at disconnect)")
    ap.add_argument("--list-modes", action="store_true", help="modes within --kiss-bw, narrowest first")
    a = ap.parse_args()
    if a.list_modes:
        print("\n".join(mode_lines(KISS_BW_CAP[a.kiss_bw])))
        return
    if a.broadcast_mode is not None:
        from .kisslink import check_mode

        try:
            check_mode(a.broadcast_mode, KISS_BW_CAP[a.kiss_bw])
        except ValueError:
            ap.error(f"--broadcast-mode {a.broadcast_mode}: not a mode within {a.kiss_bw} Hz")
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s")
    a.record_dir = Path(a.record_dir) if a.record_dir else None

    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        if a.list_audio_devices:
            audio.list_devices(pa)
            return
        if a.sample_rate % FS:
            ap.error(f"--sample-rate must be a multiple of {FS}")
        a.input_device = audio.find_device(pa, a.input_device, "input")
        a.output_device = audio.find_device(pa, a.output_device, "output")
        serve(a, pa)
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
