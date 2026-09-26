"""KISS TNC for Data2G, after xssfox/freedvtnc2, with mode shifting.

Frames from KISS clients (TCP) go out as Data2G bursts; every burst heard
comes back to every client as KISS frames. The mode is chosen per frame
(data2g.kisslink): connected-mode AX.25 to a station that has reported on
us goes in the mode it asked for; UI frames, non-AX.25 and unreported
stations go in the cap's robust broadcast mode. PTT through rigctld
("T 1" / "T 0"). Also here: the streaming Receiver the ARQ host uses.

    data2g-tnc --list-audio-devices
    data2g-tnc --bw 2400 --input-device USB --output-device USB \\
        --rigctld-port 4532 --ptt-on-delay-ms 100

Framing of the frames: [length, 2 bytes big-endian][frame] back to back
across a burst's data codewords, zero-padded (a zero length ends it).
"""

import os

# One thread per math library, set before numpy loads. OpenBLAS (numpy)
# and OpenMP (torch) otherwise keep a worker per core spinning between
# the receiver's many small operations: listening took 300-400% CPU on
# 1200 Hz and 600-700% on 2400 Hz, against ~10% of one core with one
# thread each (2026-09-24). Decoding single-threaded stays well inside
# real time (a 32 s burst in ~4 s). An explicit setting still wins.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import queue  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import socketserver  # noqa: E402
import struct  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402

import numpy as np  # noqa: E402
from scipy import signal as sps  # noqa: E402

from . import codes, modem  # noqa: E402
from .config import BANDS, FS, LEADIN_SAMPLES, MAX_CODEWORDS, NSYM  # noqa: E402

log = logging.getLogger("data2g.tnc")

# --- KISS -------------------------------------------------------------------

FEND, FESC, TFEND, TFESC = 0xC0, 0xDB, 0xDC, 0xDD


def kiss_encode(data: bytes, port: int = 0) -> bytes:
    """A KISS data frame (command 0) for `data`."""
    body = bytes([port << 4]) + data
    body = body.replace(bytes([FESC]), bytes([FESC, TFESC])).replace(bytes([FEND]), bytes([FESC, TFEND]))
    return bytes([FEND]) + body + bytes([FEND])


class KissDecoder:
    """Bytes in, whole frames (command byte, data) out, across reads."""

    def __init__(self):
        self.buf, self.esc = bytearray(), False

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        out = []
        for b in data:
            if b == FEND:
                if self.buf:
                    out.append((self.buf[0], bytes(self.buf[1:])))
                self.buf, self.esc = bytearray(), False
            elif self.esc:
                self.buf.append({TFEND: FEND, TFESC: FESC}.get(b, b))
                self.esc = False
            elif b == FESC:
                self.esc = True
            else:
                self.buf.append(b)
        return out


# --- framing ----------------------------------------------------------------

def capacity(spec, max_cw: int = MAX_CODEWORDS) -> int:
    """Bytes a burst of at most `max_cw` codewords carries, length fields included."""
    return max_cw * codes.payload_bytes(spec)


def pack(packets: list[bytes], spec) -> list[bytes]:
    """Packets -> codeword payloads for one burst."""
    p = codes.payload_bytes(spec)
    stream = b"".join(struct.pack(">H", len(x)) + x for x in packets)
    if len(stream) > capacity(spec):
        raise ValueError(f"{len(stream)} bytes exceed a burst's {capacity(spec)}")
    stream += bytes(-len(stream) % p)
    return [stream[i : i + p] for i in range(0, len(stream), p)]


def unpack(payloads: list[bytes], ok: list[bool]) -> tuple[list[bytes], int]:
    """Codeword payloads and their CRC results -> (whole packets, packets
    lost). Parsing stops at a length it can not trust."""
    stream = b"".join(payloads)
    good = np.repeat(np.array(ok, dtype=bool), [len(x) for x in payloads])
    out, lost, pos = [], 0, 0
    while pos + 2 <= len(stream):
        if not good[pos : pos + 2].all():
            lost += 1  # at least this one; the rest can not be found
            break
        n = struct.unpack(">H", stream[pos : pos + 2])[0]
        if n == 0 or pos + 2 + n > len(stream):
            break
        if good[pos + 2 : pos + 2 + n].all():
            out.append(stream[pos + 2 : pos + 2 + n])
        else:
            lost += 1
        pos += 2 + n
    return out, lost


# --- receive ----------------------------------------------------------------

SILENCE_RMS = 1e-4  # below this (-80 dBFS) the receiver's input is taken as silence: no search


def search_span(bands, cpm_grids=()) -> int:
    """Samples holding a whole preamble and header (every copy) of any of these."""
    from . import cpm

    n = max((modem.head_samples(b) for b in bands), default=0)
    for g in cpm_grids:
        lay = cpm.layout(g, cpm.stream_symbols(g, 0, False))
        n = max(n, (lay.hdr_rows[0][-1] + 2) * cpm.GRIDS[g].T)
    return n


def receive_any(y: np.ndarray, lead: int = 0, cpm_grids=None) -> dict | None:
    """One burst starting within `lead` samples of y's start, whichever
    family (OFDM first, as Receiver tries them) -> modem.receive's or
    cpm.receive's dict, or None. Offline use (sessions through the real
    modem); searches only the head, as the streaming receiver does."""
    from . import cpm

    grids = tuple(cpm.GRIDS) if cpm_grids is None else tuple(cpm_grids)
    head = lead + search_span(tuple(BANDS), grids)
    try:
        r = modem.receive(y, head=min(head, len(y)))
        if r["score"] >= Receiver.SUSPECT_SCORE:
            return r
    except modem.SyncError:
        r = None
    for g in grids:
        lock = cpm.find(cpm.GRIDS[g], y[:head])
        if lock is not None:
            return cpm.receive(y, lock)  # a CPM lock beats a suspect OFDM header
    return r


class Receiver:
    """Streaming receiver at FS: feed() audio as it arrives, get events
    back. Looks for a preamble and header in a short rolling buffer
    (find_burst); once one decodes, waits for the rest of the burst it
    claims, then receives it, searching only the start of the segment for
    the preamble it already found (modem.receive's `head`), so little work
    is left when the burst ends."""

    def __init__(self, accept: modem.Accept, cpm_grids=()):
        """`cpm_grids`: data2g.cpm grids to listen for too (early lock:
        front sync block and first header copy)."""
        from . import cpm

        self.accept, self.bands = accept, accept.bands
        self.grids = [cpm.GRIDS[g] for g in cpm_grids]
        # the least audio worth searching (a whole preamble and header), and
        # what a trim keeps (so one still arriving survives it)
        self.min_search = search_span(self.bands, cpm_grids)
        self.keep = self.min_search + FS
        self.reset()

    # new audio between preamble searches (one is ~40 ms of CPU at +-150 Hz):
    # a reply's header must be seen before the sender's deadline (the audio
    # loopback missed replies at 0.5 s and a 1.5 s first search)
    HOP = FS // 4

    def reset(self):
        self.buf = np.zeros(0)
        self.off = 0  # stream sample index of buf[0]
        self.pending = None
        self.pilots_ok = False
        self.powers = deque(maxlen=self.FLOOR_BLOCKS)  # in-band power of 0.1 s blocks (on_air)
        self._ps, self._pn = 0.0, 0  # the block being summed
        self.fresh = self.HOP  # samples fed since the last preamble search (search at once)
        self.last_start = -1  # stream start of the last burst handled (not to be found again)

    @property
    def busy(self) -> bool:
        """A burst is being received (our own replies wait for it)."""
        return self.pending is not None

    @property
    def channel_busy(self) -> bool:
        """What the host reports as BUSY: a burst is being received and its
        frame pilots say it is really there. A false lock (a weak header off
        noise or a missed burst's data) held BUSY for the whole length it
        claimed, up to 12 s, and a VARA client doesn't transmit under BUSY."""
        return self.pending is not None and (self.pilots_ok or self.on_air)

    # BUSY by energy too: in-band power over the recent noise floor (the 5th
    # percentile of 0.1 s block powers over two minutes, so a long burst does
    # not lift it). A false lock whose pilots are absent keeps BUSY only while
    # something is on air: a real burst whose preamble was missed (its frame
    # grid is not the false lock's), or another station.
    ON_AIR_DB = 3.0
    FLOOR_BLOCKS = 1200  # 0.1 s blocks: two minutes

    @property
    def on_air(self) -> bool:
        if len(self.powers) < 50:
            return False
        floor = np.percentile(self.powers, 5)
        return float(np.mean(list(self.powers)[-3:])) > floor * 10 ** (self.ON_AIR_DB / 10)

    def _check_pilots(self):
        """pilots_ok from the newest PILOT_PAIRS frame pilot pairs: under the
        band's noise level, nothing is there (any more)."""
        p = self.pending
        if p.get("family") == "cpm" or "p0" not in p:
            return
        c = modem.pilot_coherence(self.buf, dict(p, p0=p["p0"] - self.off), modem.PILOT_PAIRS, latest=True)
        if len(c) >= modem.PILOT_PAIRS:
            ok = float(np.mean(c)) > modem.PILOT_NOISE[p["spec"].band]
            if ok != self.pilots_ok:
                log.info("%s burst: pilots %s (coherence %.2f)", p["spec"].name, "back" if ok else "gone", np.mean(c))
            self.pilots_ok = ok

    def _trim(self, n: int):
        self.off += len(self.buf) - n if n < len(self.buf) else 0
        self.buf = self.buf[-n:] if n < len(self.buf) else self.buf

    SUPERSEDE_MARGIN = 0.05  # header score a later header needs over the pending one
    SUSPECT_SCORE = 0.36  # below it a header may be a false lock (false ones score 0.19-0.34)

    def _supersede(self, out: list, whole: bool = False) -> bool:
        """While committed to a burst, keep looking in the newer audio: a
        header past the pending one's, scoring clearly better, replaces it.
        A false lock (data whose preamble was missed reads as a header half
        the time at a 0.25 score floor) then costs a short BUSY, not the
        real burst that follows; the margin keeps a real burst's own data
        from replacing it (false headers score 0.19-0.34, real ones mostly
        0.4-0.6). `whole`: search everything after the pending header (as it
        completes), not just the newest audio."""
        self.fresh = 0
        p = self.pending
        if p.get("family") == "cpm":
            return False  # scores aren't comparable across families; CPM bursts are long and certain
        hdr_end = p["start"] + BANDS[p["band"]].preamble_samples + modem.header_samples(p["band"]) - self.off
        w0 = hdr_end if whole else max(hdr_end, len(self.buf) - self.keep)
        if len(self.buf) - w0 < self.keep // 2:
            return False
        try:
            q = modem.find_burst(self.buf[w0:], self.bands, self.accept)
        except modem.SyncError:
            return False
        if q["score"] < p["score"] + self.SUPERSEDE_MARGIN:
            return False
        q = dict(q, start=q["start"] + w0 + self.off, end=q["end"] + w0 + self.off, p0=q["p0"] + w0 + self.off)
        log.info("%s header (score %.2f) superseded by %s (score %.2f)", p["spec"].name, p["score"],
                 q["spec"].name, q["score"])
        if not whole:  # completing: the caller has already handled p
            out.append(("burst", {"header": p, "rx": None,
                                  "audio": self.buf[max(0, p["start"] - self.off):q["start"] - self.off]}))
        self.pending = q
        self.pilots_ok = q["score"] >= self.SUSPECT_SCORE
        out.append(("header", q))
        return True

    def _find_cpm(self) -> dict | None:
        """An early lock on any listened-for CPM grid, as find_burst's dict."""
        from . import cpm

        for g in self.grids:
            lock = cpm.find(g, self.buf)
            if lock is not None:
                return dict(lock, n_cw=1 + lock["dup"] + lock["n_data"])
        return None

    def feed(self, x: np.ndarray) -> list[tuple[str, dict]]:
        """-> events in order: ("header", find_burst's dict) as a burst
        starts, then ("burst", {"header": that dict, "rx": modem.receive's
        dict or None if it was lost, "audio": the segment received}). The dicts' start/end are stream
        sample indices (samples fed since reset)."""
        # 0.1 s block powers for the noise floor (feeds may be any size)
        sq = np.asarray(x, dtype=np.float64) ** 2
        while len(sq):
            take = min(len(sq), FS // 10 - self._pn)
            self._ps, self._pn, sq = self._ps + float(np.sum(sq[:take])), self._pn + take, sq[take:]
            if self._pn == FS // 10:
                self.powers.append(self._ps / self._pn)
                self._ps = self._pn = 0
        self.buf = np.concatenate([self.buf, x])
        self.fresh += len(x)
        out = []
        while True:
            if self.pending is None:
                # a burst's end is checked on every call (the reply waits on
                # it); new preambles only every HOP samples
                if len(self.buf) < self.min_search or self.fresh < self.HOP:
                    break
                self.fresh = 0
                # digital silence (a muted or looped-back input) has no noise
                # to normalize by, and filter ringing then reads as preambles
                # (a burst's own tail read as a header at 1e-6 noise). A
                # radio's noise at the sound card is far above -80 dBFS.
                if np.sqrt(np.mean(self.buf[-self.min_search :] ** 2)) < SILENCE_RMS:
                    self._trim(self.keep)
                    break
                try:
                    p = modem.find_burst(self.buf, self.bands, self.accept)
                    if p["score"] < self.SUSPECT_SCORE:
                        p = self._find_cpm() or p  # a strong CPM burst read as a weak OFDM header
                except modem.SyncError:
                    p = self._find_cpm()
                    if p is None:
                        self._trim(self.keep)
                        break
                if p["start"] + self.off <= self.last_start:
                    break  # the suspect burst just handled, still in the kept window
                self.pending = dict(p, start=p["start"] + self.off, end=p["end"] + self.off)
                if "p0" in p:
                    self.pending["p0"] = p["p0"] + self.off
                # BUSY at once on a clear header; a suspect one waits for its pilots
                self.pilots_ok = p.get("family") == "cpm" or p["score"] >= self.SUSPECT_SCORE
                log.info("receiving %s burst: %d codeword(s), %.1f s, %s score %.2f", p["spec"].name,
                         p["n_cw"], (p["end"] - p["start"]) / FS, "sync" if p.get("family") == "cpm" else "header",
                         p["score"])
                out.append(("header", self.pending))
            p = self.pending
            if self.off + len(self.buf) < p["end"] + LEADIN_SAMPLES:
                if self.fresh >= self.HOP:
                    self._check_pilots()
                    if self._supersede(out):
                        continue
                break
            if p.get("family") == "cpm":
                from . import cpm

                r = cpm.receive(self.buf, dict(p, start=p["start"] - self.off))
                out.append(("burst", {"header": p, "rx": r, "audio": self.buf[max(0, p["start"] - self.off):
                                                                               p["end"] - self.off]}))
                self.last_start = p["start"]
                self._trim(len(self.buf) - (p["end"] - self.off))
                self.pending = None
                continue
            s0 = max(0, p["start"] - LEADIN_SAMPLES - self.off)
            seg = self.buf[s0 : p["end"] + LEADIN_SAMPLES - self.off]
            head = p["start"] - self.off - s0 + modem.head_samples(p["band"]) + NSYM + 3 * modem.M
            try:
                r = modem.receive(seg, [p["band"]], self.accept, head=min(head, len(seg)))
            except modem.SyncError as e:
                log.warning("burst lost: %s", e)
                r = None
            out.append(("burst", {"header": p, "rx": r, "audio": seg}))
            self.last_start = p["start"]
            # a better header inside this burst's span (it was a false lock,
            # and the real one began before it ended) is next, not trimmed away
            if self._supersede(out, whole=True):
                self._trim(len(self.buf) - max(0, self.pending["start"] - LEADIN_SAMPLES - self.off))
                continue
            # a suspect (possibly false) burst may have claimed an end past a
            # real preamble that began inside it: keep the last search window
            # (a confident one keeps nothing of itself: its own data would
            # false-lock and hold BUSY after every real burst)
            cut = p["end"] - self.off
            if p["score"] < self.SUSPECT_SCORE:
                cut = min(cut, len(self.buf) - self.keep)
            self._trim(len(self.buf) - max(0, cut))
            self.pending = None
        return out


class Decimator:
    """Device rate (a multiple of FS) -> FS, keeping filter state across
    chunks so chunk edges add nothing."""

    def __init__(self, rate: int):
        self.d = rate // FS
        self.taps = sps.firwin(32 * self.d + 1, 0.9 * FS / 2, fs=rate) if self.d > 1 else np.ones(1)
        self.zi = np.zeros(len(self.taps) - 1)
        self.phase = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = sps.lfilter(self.taps, 1.0, x, zi=self.zi)
        out = y[self.phase :: self.d]
        self.phase = (self.phase - len(x)) % self.d
        return out


# --- PTT --------------------------------------------------------------------

class Rigctld:
    """PTT through a rigctld TCP connection; port 0 disables it."""

    def __init__(self, host: str, port: int):
        self.addr, self.sock = (host, port), None

    def ptt(self, on: bool):
        if not self.addr[1]:
            return
        for _ in range(2):  # one reconnect
            try:
                if self.sock is None:
                    self.sock = socket.create_connection(self.addr, timeout=2)
                self.sock.sendall(b"T 1\n" if on else b"T 0\n")
                reply = self.sock.recv(64)
                if not reply.startswith(b"RPRT 0"):
                    log.warning("rigctld answered %r to PTT %s", reply, "on" if on else "off")
                return
            except OSError as e:
                log.warning("rigctld %s:%d: %s", *self.addr, e)
                self.close()
        log.error("PTT %s failed", "on" if on else "off")

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None


# --- KISS server ------------------------------------------------------------

class KissServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, on_packet):
        self.clients, self.lock, self.on_packet = set(), threading.Lock(), on_packet
        super().__init__(addr, _KissHandler)

    def broadcast(self, data: bytes):
        frame = kiss_encode(data)
        with self.lock:
            for c in list(self.clients):
                try:
                    c.sendall(frame)
                except OSError:
                    self.clients.discard(c)

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
                    if cmd & 0x0F == 0:
                        srv.on_packet(payload)
                    else:
                        log.debug("KISS command 0x%02x ignored", cmd)
        except OSError:
            pass
        finally:
            with srv.lock:
                srv.clients.discard(self.request)
            log.info("KISS client %s disconnected", peer)


# --- the TNC ----------------------------------------------------------------

class TNC:
    """KISS TNC with mode shifting (data2g.kisslink): frames from KISS
    clients go out in the mode each next hop reported for us, or the cap's
    robust broadcast mode; every burst heard comes back as KISS frames."""

    def __init__(self, a, pa):
        from . import cpm
        from .kisslink import KissLink

        self.rate, self.a, self.pa = a.sample_rate, a, pa
        self.link, self.lock = KissLink(cap=a.cap), threading.Lock()
        self.stop = threading.Event()
        self.transmitting = threading.Event()
        self.rx_reset = threading.Event()  # set by TX, acted on by the RX thread
        self.txcv = threading.Condition()
        self.rxq = queue.Queue()
        self.accept = modem.Accept.of(None, None, a.min_header_score)
        self.receiver = Receiver(self.accept, cpm_grids=tuple(cpm.GRIDS))
        self.rig = Rigctld(a.rigctld_host, a.rigctld_port)
        self.kiss = KissServer((a.kiss_tcp_address, a.kiss_tcp_port), self.queue_packet)
        self.gain = 10 ** (a.output_volume / 20)
        self.inp = pa.open(format=_pa_float(), channels=1, rate=self.rate, input=True,
                           input_device_index=a.input_device, frames_per_buffer=self.rate // 10,
                           stream_callback=self._captured)
        self.out = pa.open(format=_pa_float(), channels=1, rate=self.rate, output=True,
                           output_device_index=a.output_device)

    def _captured(self, data, frames, time_info, status):
        import pyaudio

        if not self.transmitting.is_set():  # half duplex: no hearing ourselves
            self.rxq.put(data)
        return None, pyaudio.paContinue

    def queue_packet(self, data: bytes):
        with self.lock:
            self.link.enqueue(data)
        with self.txcv:
            self.txcv.notify()
        log.debug("queued %d-byte frame", len(data))

    def rx_loop(self):
        dec, acc, n = Decimator(self.rate), [], 0
        while not self.stop.is_set():
            try:
                chunk = self.rxq.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.rx_reset.is_set():  # a transmission happened: start afresh
                self.rx_reset.clear()
                self.receiver.reset()
                acc, n = [], 0
            x = dec(np.frombuffer(chunk, dtype=np.float32).astype(np.float64))
            acc.append(x)
            n += len(x)
            if n < FS // 2:
                continue
            x, acc, n = np.concatenate(acc), [], 0
            for kind, ev in self.receiver.feed(x):
                if kind != "burst" or ev["rx"] is None:
                    continue
                r = ev["rx"]
                with self.lock:
                    frames = self.link.on_burst(r)
                log.info("RX %s: %d codeword(s), %d frame(s)", r["spec"].name, r["n_cw"], len(frames))
                for f in frames:
                    self.kiss.broadcast(f)

    def tx_loop(self):
        while not self.stop.is_set():
            with self.lock:
                burst = self.link.next_burst()
            if burst is None:
                with self.txcv:
                    self.txcv.wait(0.2)
                continue
            if self.receiver.channel_busy:
                log.info("channel busy, holding a %s burst", burst.submode)
                while self.receiver.channel_busy and not self.stop.is_set():
                    time.sleep(0.1)
            self.transmit(burst)

    def transmit(self, burst):
        from .arq import phy as PHY

        x = PHY.tx_audio(burst)
        x = sps.resample_poly(x, self.rate // FS, 1) if self.rate != FS else x
        x = np.clip(x / np.max(np.abs(x)) * self.gain, -1, 1).astype(np.float32)
        log.info("TX %s: %d codeword(s), %.1f s", burst.submode, len(burst.slots), len(x) / self.rate)
        self.transmitting.set()
        try:
            self.rig.ptt(True)
            time.sleep(self.a.ptt_on_delay_ms / 1000)
            self.out.write(x.tobytes())
            time.sleep(self.a.ptt_off_delay_ms / 1000)
        finally:
            self.rig.ptt(False)
            self.rx_reset.set()
            self.transmitting.clear()

    def run(self):
        threads = [threading.Thread(target=f, name=f.__name__, daemon=True)
                   for f in (self.rx_loop, self.tx_loop, self.kiss.serve_forever)]
        for t in threads:
            t.start()
        self.inp.start_stream()
        log.info("KISS on %s:%d; bandwidth cap %d Hz", self.a.kiss_tcp_address, self.a.kiss_tcp_port,
                 {0: 500, 2: 2400}[self.a.cap])
        while not self.stop.wait(0.5):
            pass
        log.info("shutting down")
        self.kiss.shutdown()
        self.kiss.close_clients()
        self.kiss.server_close()
        for t in threads[:2]:
            t.join(timeout=30)  # a transmission in progress finishes
        self.inp.stop_stream()
        self.inp.close()
        self.out.close()
        self.rig.ptt(False)
        self.rig.close()


def _pa_float():
    import pyaudio

    return pyaudio.paFloat32


def _device(pa, want: str | None, kind: str) -> int | None:
    """Index for a device given by index or name substring; None: default."""
    if want is None:
        return None
    if want.isdigit():
        return int(want)
    key = "maxInputChannels" if kind == "input" else "maxOutputChannels"
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d[key] > 0 and want.lower() in d["name"].lower():
            return i
    raise SystemExit(f"no {kind} device matching {want!r} (see --list-audio-devices)")


def main():
    ap = argparse.ArgumentParser(description="KISS TNC for Data2G, shifting modes per station")
    ap.add_argument("--list-audio-devices", action="store_true")
    ap.add_argument("--log-level", default="INFO",
                    choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    ap.add_argument("--bw", type=int, choices=(2400, 500), default=2400, help="bandwidth cap, Hz")
    ap.add_argument("--min-header-score", type=float, default=0.0,
                    help="header match floor over the modem's own, 0..1")
    ap.add_argument("--input-device", help="index or name substring (default: system default)")
    ap.add_argument("--output-device", help="index or name substring (default: system default)")
    ap.add_argument("--sample-rate", type=int, default=48000, help="audio device rate, a multiple of 8000")
    ap.add_argument("--output-volume", type=float, default=0.0,
                    help="dB; 0 puts a burst's peak at digital full scale, negative is quieter")
    ap.add_argument("--kiss-tcp-port", type=int, default=8001)
    ap.add_argument("--kiss-tcp-address", default="0.0.0.0")
    ap.add_argument("--rigctld-host", default="localhost")
    ap.add_argument("--rigctld-port", type=int, default=4532, help="0: no PTT")
    ap.add_argument("--ptt-on-delay-ms", type=int, default=0, help="after keying, before audio")
    ap.add_argument("--ptt-off-delay-ms", type=int, default=0, help="after audio, before unkeying")
    a = ap.parse_args()
    a.cap = {2400: 2, 500: 0}[a.bw]
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s")

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
        if a.output_volume > 0:
            log.warning("--output-volume %+g dB clips the burst's peaks", a.output_volume)
        a.input_device = _device(pa, a.input_device, "input")
        a.output_device = _device(pa, a.output_device, "output")
        tnc = TNC(a, pa)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: tnc.stop.set())
        tnc.run()
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
