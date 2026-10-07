"""data2g-bulk: one-way transfer of a text over the modem, no ARQ (docs/bulk.md).

Send: the text is cut into blocks, one codeword each, every block its own
raw-deflate stream (zdict, no history: a lost codeword loses only its own
text). Burst b of a pass carries a control codeword, blocks [b h, (b+1) h)
at RV 2p and burst b-1's blocks again at RV 2p+1 (p: the pass), so every
block goes twice, a burst apart, and combines as incremental redundancy.
A pass ends with a burst of copies only. Bursts go back to back, so burst
g starts a fixed time after burst 0: the receiver, once it has one control,
knows where every burst is, and receives those whose header (or control)
was lost at their known position.

Receive: modem audio (WAV, or raw float32 mono at 8 kHz on stdin) -> the
text, with each run of lost blocks marked.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import zlib  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from functools import cached_property  # noqa: E402

import numpy as np  # noqa: E402
from scipy.io import wavfile  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from . import codes, cpm, modem  # noqa: E402
from .arq import frames as F  # noqa: E402
from .arq import phy as PHY  # noqa: E402
from .arq.link import Slot, TxBurst  # noqa: E402
from .arq.modes import MODES, ctl_payload_bytes, ctl_spec, is_cpm  # noqa: E402
from .config import FS, MAX_CODEWORDS  # noqa: E402
from .tnc import Receiver  # noqa: E402

log = logging.getLogger("data2g.bulk")
VERSION = 2
CTL = struct.Struct(">BHHHBB")  # version, stream, burst in pass, blocks, blocks per burst (h), pass
CTL_KEY = 0xB17C  # the control's mask key: any receiver reads any stream's control
CTL_MASK = (CTL_KEY, 0, 128)  # last byte >= 128: a control slot's (CPM's receiver checks)
MAX_BLOCKS = 32767  # data masks keep their last byte under 128
# the control in this many slots, at RVs 0, 1, ...: it must outlast a block's
# two sends combined, or the receiver never learns where the copies are (one
# control slot at the data's rate: qpsk-r1/2 on mpp heard no stream at -2 dB,
# where copies would have delivered)
CTL_SLOTS = 4
# CPM: its header allows one control codeword or two (polar: RV 1 is a plain
# repeat), and cpm.MAX_DATA data codewords
CPM_CTL_SLOTS = 2
MAX_H = (MAX_CODEWORDS - CTL_SLOTS) // 2
CPM_MAX_H = cpm.MAX_DATA // 2
TOL = FS // 5  # a heard burst within this of where burst g is due is burst g
PRE_S = 180
FLUSH = 6 * FS  # silence fed after the input ends, so the last burst completes


def data_mask(stream: int, i: int) -> tuple:
    """Block i's CRC mask (phy.mask_value's tuple): its identity, no payload bytes spent."""
    return (stream, i >> 7, i & 127)


def n_ctl(spec) -> int:
    return CPM_CTL_SLOTS if is_cpm(spec) else CTL_SLOTS


def max_h(spec) -> int:
    return CPM_MAX_H if is_cpm(spec) else MAX_H


def airtime(spec, n_cw: int) -> int:
    """A burst's samples as phy.tx_audio makes them (cpm.burst_seconds adds
    the ramps on top of the symbols: 160 samples a burst too many)."""
    if is_cpm(spec):
        k = n_ctl(spec)
        return cpm.layout(spec.grid, cpm.stream_symbols(spec.grid, n_cw - k, k == 2)).n * cpm.GRIDS[spec.grid].T
    return round(modem.burst_seconds(spec, n_cw) * FS)


# --- blocks -------------------------------------------------------------------

def stored(data: bytes) -> bytes:
    """A final raw-deflate stored block: inflate gives `data` back (5 bytes of header)."""
    return bytes([1]) + struct.pack("<HH", len(data), len(data) ^ 0xFFFF) + data


def pack(text: bytes, pb: int) -> list[bytes]:
    """Text -> pb-byte block payloads, each a raw-deflate stream of its own:
    deflated against the zdict alone, else stored."""
    out, i = [], 0
    while i < len(text):
        f = F.deflate_fit(b"", text[i:], pb)
        n, z = f if f else (min(pb - 5, len(text) - i), None)
        out.append((z or stored(text[i:i + n])).ljust(pb, b"\0"))
        i += n
    if len(out) > MAX_BLOCKS:
        raise SystemExit(f"{len(out)} blocks: at most {MAX_BLOCKS} (a faster mode carries more per block)")
    return out


def unpack(blocks: dict, n: int) -> tuple[bytes, list]:
    """{index: payload} -> (the text, lost runs [(first, last)]), each run
    marked in the text."""
    out, lost = [], []
    for i in range(n):
        try:
            out.append(F.inflate(b"", blocks[i]))
            continue
        except (KeyError, ValueError):
            pass
        if lost and lost[-1][1] == i - 1:
            lost[-1] = (lost[-1][0], i)
        else:
            lost.append((i, i))
            out.append(None)
    marks = iter(lost)
    text = b"".join(b if b is not None else ("\n[... blocks %d-%d of %d lost ...]\n" % (*next(marks), n)).encode()
                    for b in out)
    return text, lost


def stream_id(blocks: list[bytes]) -> int:
    """From the content: a resent text keeps its id, so its codewords combine with the first sending's."""
    s = zlib.crc32(b"".join(blocks)) & 0xFFFF
    return s if s not in (0, CTL_KEY) else 1


# --- layout -------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    spec: object
    n: int  # blocks
    h: int  # new blocks per burst

    @property
    def per_pass(self) -> int:
        return -(-self.n // self.h) + 1  # the last burst: copies only

    def slots(self, g: int) -> tuple[int, int, list]:
        """Burst g (counted over passes) -> (pass, burst in pass, [(block, rv)])."""
        p, b = divmod(g, self.per_pass)
        new = [(i, 2 * p % 4) for i in range(b * self.h, min(self.n, (b + 1) * self.h))]
        old = [(i, (2 * p + 1) % 4) for i in range(max(0, (b - 1) * self.h), min(self.n, b * self.h))]
        return p, b, new + old

    def n_cw(self, g: int) -> int:
        return n_ctl(self.spec) + len(self.slots(g)[2])

    def length(self, g: int) -> int:
        return airtime(self.spec, self.n_cw(g))

    def offset(self, g: int) -> int:
        """Samples from burst 0's start to burst g's (bursts back to back)."""
        p, b = divmod(g, self.per_pass)
        return p * self._cum[-1] + self._cum[b]

    @cached_property
    def _cum(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum([self.length(j) for j in range(self.per_pass)])])


# --- send -----------------------------------------------------------------------

def bursts(text: bytes, mode: str, h: int, passes: int) -> list[TxBurst]:
    spec = MODES[mode]
    pb = codes.payload_bytes(spec)
    blocks = pack(text, pb)
    lay, stream = Layout(spec, len(blocks), h), stream_id(blocks)
    out = []
    for g in range(passes * lay.per_pass):
        p, b, slots = lay.slots(g)
        ctl = CTL.pack(VERSION, stream, b, lay.n, h, p).ljust(ctl_payload_bytes(spec), b"\0")
        out.append(TxBurst(mode, [Slot(CTL_MASK, rv, ctl) for rv in range(n_ctl(spec))] +
                           [Slot(data_mask(stream, i), rv, blocks[i]) for i, rv in slots], g))
    log.info("%d B of text -> %d blocks of %d B (%.2fx), stream %04x, %d bursts", len(text), lay.n, pb,
             len(text) / (lay.n * pb), stream, len(out))
    return out


def cpm_audio(b: TxBurst) -> np.ndarray:
    """A CPM burst as phy.tx_audio makes it, but the data codewords' tones
    dealt round-robin over the burst's data (codes.spread): a CPM codeword
    is otherwise a contiguous 3.1 s (fsk32r62), and a slow fade took it
    whole (fsk32r62-r1/2, 1% points: mpp -10.3 dB, mps -7.8, mpg -7.4).
    The control slots stay as they are, so sync and header don't change."""
    spec = MODES[b.submode]
    k = n_ctl(spec)
    coded = [codes.encode(ctl_spec(spec) if i < k else spec, s.payload, s.rv, PHY.mask_value(s.mask_id), i)
             for i, s in enumerate(b.slots)]
    if len(coded) > k:
        data = codes.spread(np.stack(coded[k:]), cpm.GRIDS[spec.grid].bits)
        coded = coded[:k] + list(data.reshape(len(coded) - k, -1))
    return cpm.modulate(spec, coded, dup=k == 2)


def undeal(r: dict) -> dict:
    """A received CPM burst's data soft bits back in codeword order (cpm_audio's deal undone)."""
    if r.get("family") == "cpm" and not r.get("_undealt"):
        k, soft = r["n_ctl_slots"], r["soft"]
        if len(soft) > k:
            data = codes.despread(np.concatenate(soft[k:]), len(soft) - k, cpm.GRIDS[r["band"]].bits)
            r["soft"] = list(soft[:k]) + list(data)
        r["_undealt"] = True
    return r


def tx_audio(bs: list[TxBurst]) -> np.ndarray:
    """Bursts back to back, each at full-scale peak (as the engine sends them)."""
    xs = [cpm_audio(b) if is_cpm(MODES[b.submode]) else PHY.tx_audio(b) for b in bs]
    return np.concatenate([x / np.max(np.abs(x)) for x in xs])


def check_mode(mode: str, h: int):
    spec = MODES.get(mode)
    if spec is None:
        raise SystemExit(f"mode {mode!r}: one of {', '.join(sorted(MODES))}")
    if min(codes.payload_bytes(spec), ctl_payload_bytes(spec)) < max(CTL.size, 6):
        raise SystemExit(f"mode {mode}: {codes.payload_bytes(spec)} B codewords can't hold the control")
    if not 1 <= h <= max_h(spec):
        raise SystemExit(f"blocks per burst in {mode}: 1..{max_h(spec)}")


# --- receive --------------------------------------------------------------------

@dataclass
class Stream:
    lay: Layout
    stream: int
    fit: list = field(default_factory=list)  # (offset of burst g, preamble start heard) per heard burst
    cfo: float = 0.0
    blocks: dict = field(default_factory=dict)  # index -> payload
    done: set = field(default_factory=set)  # bursts handled
    heard: set = field(default_factory=set)  # bursts whose header was heard
    stats: dict = field(default_factory=lambda: dict(heard=0, headerless=0, control_lost=0))

    def due(self, g: int) -> float:
        """Where burst g's preamble starts, from the heard bursts' timing (a sound card's ppm included)."""
        t, s = np.array(self.fit[-32:], dtype=float).T
        if len(t) < 2 or np.ptp(t) == 0:
            return s[-1] + self.lay.offset(g) - t[-1]
        a, c = np.polyfit(t - t[-1], s, 1)
        return c + a * (self.lay.offset(g) - t[-1])

    def nearest(self, start: int) -> int | None:
        """The burst due at `start`, if one is."""
        # walk from the last burst heard: bursts are seconds long, so this is a few steps
        g = max(self.heard) if self.heard else 0
        while g > 0 and self.due(g) > start + TOL:
            g -= 1
        while self.due(g + 1) <= start + TOL:
            g += 1
        return g if abs(self.due(g) - start) <= TOL else None


class Rx:
    """Bursts in, blocks out. `store`: soft bits of blocks not yet decoded,
    kept across bursts and passes (phy.ModemRx)."""

    def __init__(self, dd_cap: float | None = None):
        """`dd_cap`: DD (phy.ModemRx's decision-directed re-estimation) starts
        no refine past this share of a burst's airtime from the burst's
        decode start, so a live receiver keeps up (a 12 s 256l burst with
        every slot failing took 54 s of DD). None: no cap (files, studies)."""
        self.store, self.streams, self.cur, self.dd_cap = {}, {}, None, dd_cap
        self.waiting = []  # (start, r) heard before any control placed them

    def _rx(self, r: dict):
        budget = None if self.dd_cap is None else self.dd_cap * airtime(r["spec"], r["n_cw"]) / FS
        return PHY.ModemRx(undeal(r), self.store, budget)

    def control(self, rx, r: dict) -> tuple | None:
        c = rx.decode(0, CTL_MASK, 0, None)
        key = ("ctl", id(rx))
        for i in range(min(n_ctl(r["spec"]), r["n_cw"])):
            if c is not None:
                break
            c = rx.decode(i, CTL_MASK, i, key)  # combined with the slots before
        rx.forget(key)
        if c is None:
            return None
        ver, stream, b, n, h, p = CTL.unpack(c[:CTL.size])
        if ver != VERSION or not 1 <= h <= max_h(r["spec"]) or not 1 <= n <= MAX_BLOCKS:
            return None
        return stream, b, n, h, p

    def heard(self, start: int, r: dict | None):
        """A burst the streaming receiver found, its preamble at `start`
        (r: modem.receive's dict, None if it failed past the header)."""
        rx = None if r is None else self._rx(r)
        ctl = None if rx is None else self.control(rx, r)
        if ctl is not None:
            stream, b, n, h, p = ctl
            spec = r["spec"]
            lay = Layout(spec, n, h)
            if b >= lay.per_pass or r["n_cw"] != lay.n_cw(b):
                log.warning("control of stream %04x doesn't fit its burst: ignored", stream)
                return
            st = self.streams.get(stream)
            if st is None or st.lay != lay:
                st = self.streams[stream] = Stream(lay, stream)
                log.info("stream %04x: %d blocks, %s, %d per burst", stream, n, spec.name, h)
            g = p * st.lay.per_pass + b
            if st.fit and abs(st.due(g) - start) > TOL:
                log.warning("stream %04x: burst %d heard %.2f s off its timing: restarted", stream, g,
                            (start - st.due(g)) / FS)
                st.fit.clear()
            self.cur = st
            st.fit.append((st.lay.offset(g), start))
            st.cfo = r["cfo"]
            self._burst(st, g, rx, r)
            for s, w in self.waiting:  # heard earlier, control lost: placed now
                self.heard(s, w)
            self.waiting = []
            return
        st = self.cur
        g = None if st is None else st.nearest(start)
        if g is None or r is None or r["spec"] != st.lay.spec or r["n_cw"] != st.lay.n_cw(g):
            if st is None and r is not None:
                self.waiting = (self.waiting + [(start, r)])[-16:]
            return
        st.stats["control_lost"] += 1
        st.fit.append((st.lay.offset(g), start))
        st.cfo = r["cfo"]
        self._burst(st, g, rx, r)

    def _burst(self, st: Stream, g: int, rx, r: dict, headerless: bool = False):
        if g in st.done:
            return
        st.done.add(g)
        if not headerless:
            st.heard.add(g)
        st.stats["headerless" if headerless else "heard"] += 1
        for slot, (i, rv) in enumerate(st.lay.slots(g)[2], n_ctl(st.lay.spec)):
            key = (st.stream, i)
            if i in st.blocks:
                continue
            p = rx.decode(slot, data_mask(st.stream, i), rv, key)
            if p is not None:
                st.blocks[i] = p
                st.stats[f"rv{rv}"] = st.stats.get(f"rv{rv}", 0) + 1  # rv1: the copy was needed
                rx.forget(key)

    def missed(self, buf: np.ndarray, off: int):
        """Bursts of the current stream not heard but due inside `buf`
        (stream index `off`), received at their known position: from the
        stream's first burst (a fade on burst 0's header lost its blocks in
        modes whose copies can't decode alone) to the last burst heard. Not
        past that: after a transfer's end the slots are noise, whose soft bits
        would only dilute the blocks' stored ones."""
        st = self.cur
        if st is None or not st.heard:
            return
        last = max(st.heard)
        for g in range(last):
            if g in st.done:
                continue
            s = round(st.due(g))
            n_cw, spec = st.lay.n_cw(g), st.lay.spec
            lo, hi = s - modem.LEADIN_SAMPLES, s + st.lay.length(g) + FS // 2
            if lo < off:
                st.done.add(g)  # trimmed away already
                continue
            if hi > off + len(buf):
                break
            seg = buf[lo - off:hi - off]
            try:
                if is_cpm(spec):  # a lock as cpm.find makes one, from the timing
                    k = n_ctl(spec)
                    r = cpm.receive(seg, dict(band=spec.grid, spec=spec, start=s - lo, cfo=st.cfo,
                                              n_data=n_cw - k, dup=k == 2))
                else:
                    r = modem.receive(seg, known=dict(start=s - lo, cfo=st.cfo, spec=spec, n_cw=n_cw))
            except modem.SyncError as e:
                log.warning("burst %d (header lost) not received: %s", g, e)
                st.done.add(g)
                continue
            log.info("burst %d: header lost, received at its known position", g)
            self._burst(st, g, self._rx(r), r, headerless=True)


def listener(mode: str | None = None) -> Receiver:
    """The streaming receiver for every mode, or for `mode` alone: listening
    for OFDM too, a fsk32r62 burst read as an n10 header at 10 dB on mpp and
    the false lock hid the next burst (4 of 7 missed; timing recovered them)."""
    if mode is None:
        return Receiver(modem.Accept.of(None), cpm_grids=tuple(cpm.GRIDS))
    if is_cpm(MODES[mode]):
        return Receiver(modem.Accept(()), cpm_grids=(MODES[mode].grid,))
    return Receiver(modem.Accept.of([mode]))


def receive(chunks, mode: str | None = None, rx: Rx | None = None, on_chunk=None) -> Rx:
    """Audio chunks at FS -> the Rx with everything heard (`mode`: listen
    for that mode only; `on_chunk(rx)` after each chunk is handled)."""
    rx, rcv = rx or Rx(), listener(mode)
    buf, off = np.zeros(0), 0
    for x in chunks:
        buf = np.concatenate([buf, x])
        for kind, ev in rcv.feed(x):
            if kind == "burst":
                rx.heard(ev["header"]["start"], ev["rx"])
        rx.missed(buf, off)
        # keep what a burst still due may need (before any control: PRE_S, two of the longest bursts)
        keep = len(buf) - PRE_S * FS
        if rx.cur is not None and rx.cur.heard:
            st = rx.cur
            g = 0
            while g in st.done:
                g += 1
            keep = min(keep, round(st.due(g)) - modem.LEADIN_SAMPLES - off)
        if keep > 0:
            buf, off = buf[keep:], off + keep
        if on_chunk:
            on_chunk(rx)
    for kind, ev in rcv.feed(np.zeros(FLUSH)):
        if kind == "burst":
            rx.heard(ev["header"]["start"], ev["rx"])
    rx.missed(np.concatenate([buf, np.zeros(FLUSH)]), off)
    return rx


def read_audio(path: str, chunk_s: float = 1.0):
    """Chunks at FS from a WAV file, or raw float32 mono at FS from stdin ("-")."""
    n = int(chunk_s * FS)
    if path == "-":
        while b := sys.stdin.buffer.read(4 * n):
            yield np.frombuffer(b[:len(b) // 4 * 4], dtype=np.float32).astype(np.float64)
        return
    rate, x = wavfile.read(path)
    x = x[:, 0] if x.ndim > 1 else x
    x = x / 32768.0 if x.dtype == np.int16 else x.astype(np.float64)
    if rate != FS:
        g = np.gcd(rate, FS)
        x = resample_poly(x, FS // g, rate // g)
    for i in range(0, len(x), n):
        yield x[i:i + n]


class Printer:
    """The current stream's text to `out` in block order, each block once it
    is final: decoded, or lost when the burst carrying its copy has been
    handled. A later pass can still fill a lost block in: `path`, if given,
    is rewritten with the whole text as it stands whenever a block arrives
    (the first stream's; a later one's at path.<id>)."""

    def __init__(self, out, path: str | None = None):
        self.out, self.path, self.st, self.k, self.n_got, self.first = out, path, None, 0, -1, None

    def __call__(self, rx: Rx):
        st = rx.cur
        if st is None:
            return
        if st is not self.st:
            self.st, self.k, self.n_got = st, 0, -1
            self.first = self.first or st.stream
            print(f"\n== stream {st.stream:04x}: {st.lay.n} blocks, {st.lay.spec.name}, "
                  f"{st.lay.h} per burst ==", file=sys.stderr, flush=True)
        handled = {g % st.lay.per_pass for g in st.done}
        start = self.k
        while self.k < st.lay.n:
            k = self.k
            if k in st.blocks:
                try:
                    b = F.inflate(b"", st.blocks[k])
                except ValueError:
                    b = b"[block %d unreadable]" % k
            elif k // st.lay.h + 1 in handled:
                b = b"[block %d lost]" % k
            else:
                break
            self.out.write(b)
            self.k += 1
        self.out.flush()
        if self.k == st.lay.n and start < st.lay.n:
            print(f"\n== stream {st.stream:04x}: {len(st.blocks)}/{st.lay.n} blocks ==", file=sys.stderr, flush=True)
        if self.path and len(st.blocks) != self.n_got:
            self.n_got = len(st.blocks)
            path = self.path if st.stream == self.first else f"{self.path}.{st.stream:04x}"
            open(path, "wb").write(unpack(st.blocks, st.lay.n)[0])


def read_device(device: str | None, rate: int, chunk_s: float = 0.5):
    """Chunks at FS from a PortAudio input (data2g-host's capture and decimator)."""
    import threading

    import pyaudio

    from .host import Capture, _channels
    from .tnc import Decimator, _device

    pa = pyaudio.PyAudio()
    try:
        dev = _device(pa, device, "input")
        inp, dec, never = Capture(pa, _channels(pa, dev, "input"), rate, dev), Decimator(rate), threading.Event()
        try:
            while (x := inp.read(int(chunk_s * rate), never)) is not None:
                yield dec(x)
        finally:
            inp.close()
    finally:
        pa.terminate()


def list_devices():
    import pyaudio

    pa = pyaudio.PyAudio()
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d["maxInputChannels"]:
            print(f"{i:3d}  in {d['maxInputChannels']:2d}  {int(d['defaultSampleRate']):6d} Hz  {d['name']}")
    pa.terminate()


# --- command line -------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="data2g-bulk", description=__doc__.split("\n\n")[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send", help="text -> WAV")
    s.add_argument("text", help="input text file ('-': stdin)")
    s.add_argument("-o", "--output", required=True, help="WAV file to write (8 kHz, 16 bit)")
    s.add_argument("-m", "--mode", default="qpsk-r1/2", help="OFDM or CPM mode (default %(default)s)")
    s.add_argument("--blocks-per-burst", type=int, help=f"new blocks per burst: OFDM 1-{MAX_H} (default 15), "
                   f"CPM 1-{CPM_MAX_H} (default {CPM_MAX_H})")
    s.add_argument("--passes", type=int, default=1, help="times the whole text is sent, RVs rotating (default 1)")
    s.add_argument("--lead-ms", type=float, default=500, help="silence before the first burst, for PTT/VOX")
    r = sub.add_parser("recv", help="WAV (or raw float32 at 8 kHz on stdin) -> text")
    r.add_argument("audio", help="WAV file, or '-' for raw float32 mono at 8 kHz on stdin")
    r.add_argument("-o", "--output", required=True, help="text file to write (several streams: .<id> appended)")
    r.add_argument("--dd-cap", type=float, help="DD stops starting refines past this share of a burst's airtime "
                   "(default: no cap)")
    r.add_argument("-m", "--mode", help="listen for this mode only (default: every mode)")
    li = sub.add_parser("listen", help="an audio device -> text on stdout as it arrives")
    li.add_argument("--input-device", help="index or name substring (default: the system's)")
    li.add_argument("--sample-rate", type=int, default=48000, help="a multiple of 8000 (default %(default)s)")
    li.add_argument("--list-audio-devices", action="store_true")
    li.add_argument("-o", "--output", help="also keep the whole text here, rewritten as blocks arrive")
    li.add_argument("--dd-cap", type=float, default=0.25, help="DD stops starting refines past this share of a "
                    "burst's airtime, so decoding keeps up with the audio (default %(default)s)")
    li.add_argument("-m", "--mode", help="listen for this mode only (default: every mode)")
    a = ap.parse_args(argv)
    if a.cmd in ("recv", "listen") and a.mode and a.mode not in MODES:
        ap.error(f"mode {a.mode!r}: one of {', '.join(sorted(MODES))}")
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")
    if a.cmd == "send":
        h = a.blocks_per_burst or (CPM_MAX_H if a.mode in cpm.SPECS else 15)
        check_mode(a.mode, h)
        text = sys.stdin.buffer.read() if a.text == "-" else open(a.text, "rb").read()
        bs = bursts(text, a.mode, h, a.passes)
        x = np.concatenate([np.zeros(int(a.lead_ms * FS / 1000)), tx_audio(bs), np.zeros(FS // 2)])
        wavfile.write(a.output, FS, np.round(x * 32000).astype(np.int16))
        print(f"{len(bs)} bursts, {len(x) / FS:.1f} s -> {a.output}", file=sys.stderr)
        return
    if a.cmd == "listen":
        if a.list_audio_devices:
            return list_devices()
        if a.sample_rate % FS:
            ap.error(f"--sample-rate must be a multiple of {FS}")
        print("listening (Ctrl-C to stop)", file=sys.stderr)
        rx = Rx(a.dd_cap)
        try:
            receive(read_device(a.input_device, a.sample_rate), a.mode, rx=rx,
                    on_chunk=Printer(sys.stdout.buffer, a.output))
        except KeyboardInterrupt:
            pass
        for stream, st in rx.streams.items():
            print(f"\nstream {stream:04x}: {len(st.blocks)}/{st.lay.n} blocks", file=sys.stderr)
        return
    rx = receive(read_audio(a.audio), a.mode, rx=Rx(a.dd_cap))
    if not rx.streams:
        raise SystemExit("no stream heard")
    for stream, st in rx.streams.items():
        text, lost = unpack(st.blocks, st.lay.n)
        path = a.output if len(rx.streams) == 1 else f"{a.output}.{stream:04x}"
        open(path, "wb").write(text)
        s = st.stats
        print(f"stream {stream:04x}: {st.lay.n - sum(b - a + 1 for a, b in lost)}/{st.lay.n} blocks, "
              f"{s.get('rv1', 0)} of them only with their copy ({s['heard']} bursts heard, "
              f"{s['headerless']} received with the header lost, {s['control_lost']} placed by timing "
              f"with the control lost) -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
