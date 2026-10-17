"""The TNC's links to other programs: the KISS byte codec and its TCP
server (after xssfox/freedvtnc2), and PTT through rigctld ("T 1" / "T 0").
data2g-host (data2g.host) serves them; what the KISS frames carry over the
air is data2g.kisslink's.

Native: native/core/tnc/tnc.hpp (the KISS codec); the server and PTT are
the app's (native/app/tcp_servers.hpp, core/rig/).
"""

import logging
import socket
import socketserver
import threading

log = logging.getLogger("data2g.tnc")

# --- KISS -------------------------------------------------------------------

FEND, FESC, TFEND, TFESC = 0xC0, 0xDB, 0xDC, 0xDD
DATA, ACKMODE = 0x00, 0x0C  # KISS commands (low nibble; the port is the high one)


def kiss_encode(data: bytes, port: int = 0, cmd: int = DATA) -> bytes:
    """A KISS frame: a data frame (command 0) for `data` on `port` by default."""
    body = bytes([port << 4 | cmd]) + data
    body = body.replace(bytes([FESC]), bytes([FESC, TFESC])).replace(bytes([FEND]), bytes([FESC, TFEND]))
    return bytes([FEND]) + body + bytes([FEND])


KISS_MAX_FRAME = 1 << 16  # a longer frame is junk (a client that never sends FEND)


class KissDecoder:
    """Bytes in, whole frames (command byte, data) out, across reads."""

    def __init__(self):
        self.buf, self.esc, self.skip = bytearray(), False, False

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        out = []
        for b in data:
            if b == FEND:
                if self.buf and not self.skip:
                    out.append((self.buf[0], bytes(self.buf[1:])))
                self.buf, self.esc, self.skip = bytearray(), False, False
            elif self.skip:
                pass
            elif len(self.buf) >= KISS_MAX_FRAME:  # no FEND in sight: discard to the next one
                self.buf, self.skip = bytearray(), True
            elif self.esc:
                self.buf.append({TFEND: FEND, TFESC: FESC}.get(b, b))
                self.esc = False
            elif b == FESC:
                self.esc = True
            else:
                self.buf.append(b)
        return out


# --- PTT --------------------------------------------------------------------

class Rigctld:
    """PTT through a rigctld TCP connection; port 0 disables it."""

    def __init__(self, host: str, port: int):
        self.addr, self.sock = (host, port), None
        self.keyed = False  # a PTT on reached a connected rigctld

    def ptt(self, on: bool) -> bool:
        """True if rigctld took it (or there is no rigctld); False: try again later."""
        if not self.addr[1]:
            return True
        for _ in range(2):  # one reconnect
            try:
                if self.sock is None:
                    self.sock = socket.create_connection(self.addr, timeout=2)
                self.sock.sendall(b"T 1\n" if on else b"T 0\n")
                self.keyed |= on
                reply = self.sock.recv(64)
                if not reply:  # rigctld closed on us
                    raise ConnectionError("connection closed")
                if not reply.startswith(b"RPRT 0"):
                    log.warning("rigctld answered %r to PTT %s", reply, "on" if on else "off")
                return True
            except OSError as e:
                log.warning("rigctld %s:%d: %s", *self.addr, e)
                self.close()
        log.error("PTT %s failed", "on" if on else "off")
        return False

    def release(self):
        """At exit: PTT off only if we keyed the radio. Never connected, or
        never keyed: not ours to touch, so no attempt, no wait, no warning."""
        if self.keyed:
            self.ptt(False)
        self.close()

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None


# --- KISS server ------------------------------------------------------------

class KissServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, on_packet, on_command=None):
        """`on_packet(port, frame, ack)`: a data frame, or an ACKMODE one
        (`ack` (client, tag) for send_ack, else None). `on_command(cmd,
        payload)`: KISS commands other than those."""
        self.clients, self.lock, self.on_packet, self.on_command = set(), threading.Lock(), on_packet, on_command
        super().__init__(addr, _KissHandler)

    def broadcast(self, data: bytes, port: int = 0):
        """A frame heard on `port`, to every client."""
        frame = kiss_encode(data, port)
        with self.lock:
            for c in list(self.clients):
                try:
                    c.sendall(frame)
                except OSError:
                    self.clients.discard(c)

    def send_ack(self, ack, port: int):
        """An ACKMODE frame went out: its tag back to the client that sent it."""
        client, tag = ack
        with self.lock:
            if client in self.clients:
                try:
                    client.sendall(kiss_encode(tag, port, ACKMODE))
                except OSError:
                    self.clients.discard(client)

    def close_clients(self):
        with self.lock:
            for c in self.clients:
                try:
                    c.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


class _KissHandler(socketserver.BaseRequestHandler):
    def handle(self):
        srv, peer = self.server, "%s:%d" % self.client_address[:2]
        with srv.lock:
            srv.clients.add(self.request)
        log.info("KISS client %s connected", peer)
        dec = KissDecoder()
        try:
            while data := self.request.recv(4096):
                for cmd, payload in dec.feed(data):
                    if cmd & 0x0F == DATA:
                        srv.on_packet(cmd >> 4, payload, None)
                    elif cmd & 0x0F == ACKMODE and len(payload) >= 2:  # [tag, 2][frame]: ack it once sent
                        srv.on_packet(cmd >> 4, payload[2:], (self.request, payload[:2]))
                    elif srv.on_command is not None:
                        srv.on_command(cmd & 0x0F, payload)
        except OSError:
            pass
        finally:
            with srv.lock:
                srv.clients.discard(self.request)
            log.info("KISS client %s disconnected", peer)

