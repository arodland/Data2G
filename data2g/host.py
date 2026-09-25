"""VARA-style TNC host for Data2G ARQ sessions (gear-shifter phase H).

Two TCP ports, as VARA: commands (8300, CR-terminated lines) and data
(8301, the session's byte stream). The station itself is an
arq.engine.Engine clocked by the sound card; PTT through rigctld; every
burst heard and sent recorded for offline replay (scripts/replay.py).

Commands: MYCALL call, LISTEN ON|OFF, CONNECT from to, DISCONNECT, ABORT,
CQFRAME call bw (a CQ frame at 500 | 1200 | 2300 | 2750 Hz: no session;
one heard is notified the same way, the sender's call and bandwidth),
BW500 | BW1200 | BW2300 | BW2750 (session bandwidth cap: BW500 keeps a
session inside a 500 Hz band-plan segment; BW2300/BW2750 are the full
2400 Hz; BW1200 is a Data2G extension), CHAT ON|OFF, VERSION.
Replies OK / WRONG. Notifications: CONNECTED src dst bw, DISCONNECTED,
PTT ON|OFF, BUSY ON|OFF, BUFFER n, IAMALIVE (every 60 s), and (Data2G)
MODE submode.

    data2g-host --mycall W1AW --input-device USB --output-device USB --rigctld-port 4532
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import queue  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import signal as sps  # noqa: E402

from .arq import session as S  # noqa: E402
from .arq.engine import Engine  # noqa: E402
from .config import FS  # noqa: E402
from .tnc import Decimator, Rigctld, _device, _pa_float  # noqa: E402

log = logging.getLogger("data2g.host")
VERSION = "Data2G 0.1"
BW = {"BW500": 0, "BW1200": 1, "BW2300": 2, "BW2750": 2}
BW_NAME = {0: "500", 1: "1200", 2: "2300"}
BLOCK = FS // 10  # audio block: 0.1 s
ALIVE_S = 60.0  # IAMALIVE on the command port this often, as VARA does
# VARA settings a client may send that have no Data2G meaning (yet): OK, logged.
# ponytail: from memory of VARA clients, not a checked list; check the log on first contact with Pat
IGNORED = {"COMPRESSION", "PUBLIC", "CWID", "P2P", "WINLINK", "REGISTERED", "ENCRYPTION", "IGNOREKISSDCD"}


class Host:
    """VARA command semantics over an Engine: no sockets, no audio (those
    are serve()'s). command() and data_in() take the client's side;
    after_step() collects what goes back after each Engine.step."""

    def __init__(self, engine: Engine, buffer_credit: int | None = None):
        """`buffer_credit`: queued bytes BUFFER may leave out, at most (None:
        the next burst's whole capacity; 0: plain VARA, all of them)."""
        self.engine, self.cap, self.listening = engine, 2, False
        self.buffer_credit = buffer_credit
        self.out_cmd: list[str] = []
        self.out_data = bytearray()
        self._ptt = self._busy = False
        self._buffer = 0
        self._mode = None
        self._alive = 0.0  # engine time of the last IAMALIVE

    def command(self, line: str):
        words = line.strip().split()
        if not words:
            return
        cmd, args = words[0].upper(), words[1:]
        e = self.engine
        ok = True
        if cmd == "MYCALL" and len(args) >= 1:
            e.set_call(*args)  # VARA takes several: all answer connects
        elif cmd == "LISTEN" and args and args[0].upper() == "CQ":
            pass  # CQ frames are always reported
        elif cmd == "LISTEN" and args and args[0].upper() in ("ON", "OFF"):
            self.listening = args[0].upper() == "ON"
            e.listen(self.listening)
        elif cmd == "CONNECT" and len(args) == 2:
            try:
                e.set_call(args[0], *e.aliases)
                e.connect(args[1], self.cap)
            except RuntimeError:
                ok = False
        elif cmd == "DISCONNECT":
            e.session.disconnect()
        elif cmd == "ABORT":
            e.abort()
            self.out_cmd.append("DISCONNECTED")
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
        elif cmd in IGNORED:
            log.info("accepted, not implemented: %s", line.strip())
        elif cmd == "VERSION":
            self.out_cmd.append(f"VERSION {VERSION}")
            return
        else:
            ok = False
        self.out_cmd.append("OK" if ok else "WRONG")

    def client_gone(self):
        """The command client's TCP connection closed: it owned the session.
        A session under way is disconnected gracefully (the peer is told, and
        what's queued still goes); a call being placed is dropped; listening
        stops, so no connect is accepted that nobody will serve (clients send
        LISTEN ON again when they reconnect)."""
        e = self.engine
        log.info("command client gone")
        if e.session.state in (S.CONNECTED, S.DISCONNECTING):
            e.session.disconnect()
        elif e.session.state == S.CONNECTING:
            e.abort()
        self.listening = False
        if e.session.state in (S.LISTEN, S.CLOSED):
            e.listen(False)

    def data_in(self, data: bytes):
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
        for ev in e.events():
            if ev.startswith("CONNECTED"):
                peer = ev.split()[1]
                me = e.session.call  # the call dialed, when MYCALL gave several
                src, dst = (me, peer) if e.session._master else (peer, me)
                self.out_cmd.append(f"CONNECTED {src} {dst} {BW_NAME[e.session.cap]}")
            elif ev.startswith("CQFRAME"):
                _, call, cap = ev.split()
                self.out_cmd.append(f"CQFRAME {call} {BW_NAME[int(cap)]}")
            elif ev.startswith("DISCONNECTED"):
                log.info("%s", ev)
                self.out_cmd.append("DISCONNECTED")
        if e.session.state == S.CLOSED and self.listening:
            e.listen()
        self.out_data += e.session.read()
        if ptt != self._ptt:
            self._ptt = ptt
            self.out_cmd.append("PTT ON" if ptt else "PTT OFF")
            if ptt and e.tx and e.tx[0].submode != self._mode:
                self._mode = e.tx[0].submode
                self.out_cmd.append(f"MODE {self._mode}")
        if e.receiver.busy != self._busy:
            self._busy = e.receiver.busy
            self.out_cmd.append("BUSY ON" if self._busy else "BUSY OFF")
        # BUFFER: bytes still waiting after the next burst goes. VARA
        # clients throttle on it: Pat blocks writes while BUFFER >= 7x its
        # write (<= 250 B B2F blocks), tuned to VARA's short frames. Counting
        # everything queued kept ~1-2 KB here, and every Data2G burst (up to
        # ~12 KB) went out short: a 20 KB Pat transfer took 5x as long as the
        # same bytes written at once. A DISCONNECT still sends and gets
        # acknowledged all of it before the session closes.
        st = e.session.station
        unsent = len(e.session._pending_write)
        if st:
            unsent += st.tx.buf_off + len(st.tx.buf) - st.tx.stream_end
            if hasattr(e.session.policy, "next_capacity") and self.buffer_credit != 0:
                credit = e.session.policy.next_capacity(st)
                if self.buffer_credit is not None:
                    credit = min(credit, self.buffer_credit)
                unsent = max(0, unsent - credit)
        buffered = unsent
        if buffered != self._buffer:
            self._buffer = buffered
            self.out_cmd.append(f"BUFFER {buffered}")


class Interpolator:
    """FS -> device rate, keeping filter state across blocks (per-block
    resampling would click at every block edge and splatter)."""

    def __init__(self, rate: int):
        self.u = rate // FS
        self.taps = sps.firwin(32 * self.u + 1, 0.9 * FS / 2, fs=rate) * self.u if self.u > 1 else np.ones(1)
        self.zi = np.zeros(len(self.taps) - 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        up = np.zeros(len(x) * self.u)
        up[:: self.u] = x
        y, self.zi = sps.lfilter(self.taps, 1.0, up, zi=self.zi)
        return y


# --- sockets and audio ---------------------------------------------------------------

class _Port:
    """One TCP listener holding at most one client; lines or bytes go to `on_input`."""

    def __init__(self, addr, on_input, lines: bool, on_close=None):
        self.on_input, self.lines, self.client, self.on_close = on_input, lines, None, on_close
        self.srv = socket.create_server(addr, reuse_port=False)
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                c, peer = self.srv.accept()
            except OSError:
                return
            log.info("client %s:%d on port %d", *peer, self.srv.getsockname()[1])
            old, self.client = self.client, c  # the new client first: the old one's close isn't a loss
            if old is not None:
                old.close()
            threading.Thread(target=self._read, args=(c,), daemon=True).start()

    def _read(self, c):
        buf = b""
        while True:
            try:
                d = c.recv(4096)
            except OSError:
                d = b""
            if not d:
                if self.client is c:
                    self.client = None
                    if self.on_close:
                        self.on_close()
                return
            if not self.lines:
                self.on_input(d)
                continue
            buf += d
            while b"\r" in buf or b"\n" in buf:
                i = min(x for x in (buf.find(b"\r"), buf.find(b"\n")) if x >= 0)
                line, buf = buf[:i], buf[i + 1:]
                if line.strip():
                    self.on_input(line.decode("ascii", "replace"))

    def send(self, data: bytes):
        c = self.client
        if c is not None and data:
            try:
                c.sendall(data)
            except OSError:
                self.client = None

    def close(self):
        self.srv.close()
        if self.client is not None:
            self.client.close()


def serve(a, pa, stop: threading.Event | None = None):
    """Run until SIGINT/SIGTERM (or `stop` is set)."""
    engine = Engine(a.mycall or "NOCALL", ptt_delay_s=a.ptt_on_delay_ms / 1000, record_dir=a.record_dir,
                    min_header_score=a.min_header_score)
    host = Host(engine, None if a.buffer_credit < 0 else a.buffer_credit)
    inbox: queue.Queue = queue.Queue()
    cmd = _Port((a.host, a.command_port), lambda line: inbox.put(("cmd", line)), lines=True,
                on_close=lambda: inbox.put(("gone", None)))
    data = _Port((a.host, a.command_port + 1), lambda d: inbox.put(("data", d)), lines=False)
    rig = Rigctld(a.rigctld_host, a.rigctld_port)
    dec, interp = Decimator(a.sample_rate), Interpolator(a.sample_rate)
    per = BLOCK * (a.sample_rate // FS)
    inp = pa.open(format=_pa_float(), channels=1, rate=a.sample_rate, input=True, input_device_index=a.input_device,
                  frames_per_buffer=per)
    out = pa.open(format=_pa_float(), channels=1, rate=a.sample_rate, output=True, output_device_index=a.output_device,
                  frames_per_buffer=per)
    gain = 10 ** (a.output_volume / 20)
    stop = stop or threading.Event()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())
    log.info("commands on %s:%d, data on %d; recording to %s", a.host, a.command_port, a.command_port + 1,
             a.record_dir or "(off)")
    keyed, slow = False, 0
    try:
        while not stop.is_set():
            while not inbox.empty():
                kind, v = inbox.get()
                if kind == "cmd":
                    log.info("command: %s", v)
                    host.command(v)
                elif kind == "gone":
                    host.client_gone()
                else:
                    host.data_in(v)
            x = np.frombuffer(inp.read(per, exception_on_overflow=False), dtype=np.float32).astype(np.float64)
            t0 = time.perf_counter()
            y, ptt = engine.step(dec(x) if not keyed else np.zeros(BLOCK))
            if time.perf_counter() - t0 > BLOCK / FS:
                slow += 1
                log.debug("step took %.2f s (%d slow)", time.perf_counter() - t0, slow)
            if ptt and not keyed:
                rig.ptt(True)
                keyed = True
            if keyed:
                out.write(np.clip(interp(y) * gain, -1, 1).astype(np.float32).tobytes())
            if keyed and not ptt:
                time.sleep(a.ptt_off_delay_ms / 1000)
                rig.ptt(False)
                keyed = False
            host.after_step(ptt)
            for line in host.out_cmd:
                cmd.send(line.encode() + b"\r")
            host.out_cmd.clear()
            data.send(bytes(host.out_data))
            host.out_data.clear()
    finally:
        log.info("shutting down")
        rig.ptt(False)
        rig.close()
        cmd.close()
        data.close()
        inp.close()
        out.close()


def main():
    ap = argparse.ArgumentParser(description="Data2G ARQ host, VARA-style TCP")
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
    ap.add_argument("--min-header-score", type=float, default=0.0)
    ap.add_argument("--buffer-credit", type=int, default=-1,
                    help="bytes queued for the next burst that BUFFER leaves out, at most, so VARA clients "
                         "that throttle on it (Pat) keep a whole burst queued; -1: the next burst's full "
                         "capacity, 0: report everything queued (plain VARA)")
    ap.add_argument("--record-dir", default=f"recordings/{time.strftime('%Y%m%d-%H%M%S')}",
                    help="where every burst heard and sent is logged ('' turns it off)")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s")
    a.record_dir = Path(a.record_dir) if a.record_dir else None

    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        if a.list_audio_devices:
            for i in range(pa.get_device_count()):
                d = pa.get_device_info_by_index(i)
                print(f"{i:3d}  in {d['maxInputChannels']:2d}  out {d['maxOutputChannels']:2d}  "
                      f"{int(d['defaultSampleRate']):6d} Hz  {d['name']}")
            return
        if a.sample_rate % FS:
            ap.error(f"--sample-rate must be a multiple of {FS}")
        a.input_device = _device(pa, a.input_device, "input")
        a.output_device = _device(pa, a.output_device, "output")
        serve(a, pa)
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
