"""The radio side shared by data2g-host's personalities: the streaming
Receiver, KISS framing and its TCP server (after xssfox/freedvtnc2),
rigctld PTT ("T 1" / "T 0"), resampling and audio devices. The KISS
link layer is data2g.kisslink; the server is data2g.host.

Frame framing in a KISS burst: [length, 2 bytes big-endian][frame] back
to back across the data codewords, zero-padded (a zero length ends it).
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
from .config import BANDS, FS, LEADIN_SAMPLES, NSYM, max_codewords  # noqa: E402

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

def capacity(spec, max_cw: int | None = None) -> int:
    """Bytes a burst of at most `max_cw` codewords (default: what its header
    can announce) carries, length fields included."""
    return (max_cw or max_codewords(spec.sync_band)) * codes.payload_bytes(spec)


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
    if r is None:
        # no preamble: the header copy of a burst whose head faded (as Receiver._find_copy)
        locks = [c for c in (modem.find_copy(y, b) for b in modem.HEADER_COPY_BANDS) if c is not None]
        locks = [c for c in locks if c["start"] <= lead and c["end"] <= len(y)]
        if locks:
            try:
                return modem.receive(y, copy=max(locks, key=lambda c: c["score"]))
            except modem.SyncError:
                pass
    return r


class Receiver:
    """Streaming receiver at FS: feed() audio as it arrives, get events
    back. Looks for a preamble and header in a short rolling buffer
    (find_burst); once one decodes, waits for the rest of the burst it
    claims, then receives it, searching only the start of the segment for
    the preamble it already found (modem.receive's `head`), so little work
    is left when the burst ends."""

    def __init__(self, accept: modem.Accept, cpm_grids=(), blank: bool = True):
        """`cpm_grids`: data2g.cpm grids to listen for too (early lock:
        front sync block and first header copy). `blank`: impulse-blank
        the input (Blanker)."""
        from . import cpm

        self.accept, self.bands = accept, accept.bands
        self.grids = [cpm.GRIDS[g] for g in cpm_grids]
        # the least audio worth searching (a whole preamble and header), and
        # what a trim keeps (so one still arriving survives it)
        self.min_search = search_span(self.bands, cpm_grids)
        self.keep = self.min_search + FS
        from .waveform import ofdm, sync

        # each band's detection statistic, computed once per sample (not per search)
        self.detectors = {b: sync.StreamDetector(ofdm.band(b)) for b in self.bands}
        self.blanker = Blanker() if blank else None
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
        self.confirmed = False  # the pending burst's header is clear and its pilots are there
        self.powers = deque(maxlen=self.FLOOR_BLOCKS)  # in-band power of 0.1 s blocks (on_air)
        self._ps, self._pn = 0.0, 0  # the block being summed
        self.fresh = self.HOP  # samples fed since the last preamble search (search at once)
        self.last_start = -1  # stream start of the last burst handled (not to be found again)
        # per band / CPM grid: starts before this stream index have been
        # searched with all the audio their header needs (not searched again)
        self.decided: dict = {}
        for d in getattr(self, "detectors", {}).values():
            d.reset()

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
            # a copy lock is one header copy: confirmed (no further search) at
            # the single-copy commit score
            self.confirmed = ok and p["score"] >= (modem.COPY_COMMIT_SCORE if "copy" in p else self.SUSPECT_SCORE)
            if ok != self.pilots_ok:
                log.info("%s burst: pilots %s (coherence %.2f)", p["spec"].name, "back" if ok else "gone", np.mean(c))
            self.pilots_ok = ok

    def _trim(self, n: int):
        self.off += len(self.buf) - n if n < len(self.buf) else 0
        self.buf = self.buf[-n:] if n < len(self.buf) else self.buf
        for d in self.detectors.values():
            d.trim(self.off)

    def _stats(self, w0: int = 0) -> dict:
        """Per band, the detection statistic for buf[w0:]'s starts (new audio
        fed first), starts already decided masked out (-1)."""
        out = {}
        for b, d in self.detectors.items():
            if d.fed < self.off:  # its audio was trimmed away: start over
                d.reset()
                d.fed = self.off
            d.feed(modem.to_baseband(self.buf[d.fed - self.off:], d.fed))
            # no search looks further back than `keep` (the statistic grew
            # with a long burst in the buffer, and was copied every hop)
            d.trim(self.off + len(self.buf) - self.keep - d.span)
            n = len(self.buf) - w0 - d.span + 1
            if n <= 0:
                continue
            lo = self.off + w0
            S = d.stat(lo, lo + n)
            S[:, :max(0, self.decided.get(b, 0) - lo)] = -1.0
            out[b] = S
        return out

    # a decided start stays searchable this much longer: the old whole-buffer
    # search met each start ~4 times, which rescued weak preambles now and
    # then (n10-ack-4f AWGN -9 dB: 64/80 found, 60-62 once each)
    REVISIT = 2 * HOP

    def _searched(self):
        """Every start whose whole head (both header copies) is in the buffer
        is decided: a later search of it would read the same."""
        from . import cpm

        end = self.off + len(self.buf)
        for b in self.detectors:
            self.decided[b] = max(self.decided.get(b, 0),
                                  end - modem.head_samples(b) - BANDS[b].preamble_samples - self.REVISIT)
        for g in self.grids:
            self.decided[g.name] = max(self.decided.get(g.name, 0), end - search_span((), (g.name,)) - g.T)

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
        q = None
        try:
            q = modem.find_burst(self.buf[w0:], self.bands, self.accept, stats=self._stats(w0))
            q = dict(q, start=q["start"] + w0 + self.off, end=q["end"] + w0 + self.off, p0=q["p0"] + w0 + self.off)
        except modem.SyncError:
            pass
        finally:
            self._searched()
        if "copy" in p and (q is None or q["score"] < p["score"] + self.SUPERSEDE_MARGIN):
            # a copy lock can be a copy read off the wrong frame, taken before
            # the burst's own copy arrived (at the header floor, 0.26-0.33):
            # the true one, later, replaces it
            c = self._find_copy()
            if c is not None:
                q = dict(c, start=c["start"] + self.off, end=c["end"] + self.off, p0=c["p0"] + self.off,
                         copy=dict(c["copy"], pc=c["copy"]["pc"] + self.off))
        if q is None or q["score"] < p["score"] + self.SUPERSEDE_MARGIN or q["start"] == p["start"]:
            return False
        log.info("%s header (score %.2f) superseded by %s (score %.2f)", p["spec"].name, p["score"],
                 q["spec"].name, q["score"])
        if not whole:  # completing: the caller has already handled p
            out.append(("burst", {"header": p, "rx": None,
                                  "audio": self.buf[max(0, p["start"] - self.off):q["start"] - self.off]}))
        self.pending = q
        self.pilots_ok = "copy" in q or q["score"] >= self.SUSPECT_SCORE
        self.confirmed = False
        out.append(("header", q))
        return True

    def _find_copy(self) -> dict | None:
        """A burst whose preamble and header faded, found from its frame
        pilots and header copy (modem.find_copy) on the bands that carry a
        copy, from the detectors' kept matched filter outputs: the best
        lock, as find_burst's dict (buffer indices), or None."""
        best = None
        for band in modem.HEADER_COPY_BANDS:
            d = self.detectors.get(band)
            level = d.level() if d is not None else None
            if level is None:
                continue
            a = max(d.c0, self.off)  # the stream index both the buffer and C cover from
            C = d.C[:, a - d.c0:]
            x = self.buf[a - self.off:]
            C = C[:, :max(0, len(x) - modem.M + 1)]
            lock = modem.find_copy(x, band, self.accept, C, level)
            if lock is not None and (best is None or lock["score"] > best[0]["score"]):
                best = (lock, a - self.off)
        if best is None:
            return None
        lock, k = best
        return dict(lock, start=lock["start"] + k, end=lock["end"] + k, p0=lock["p0"] + k,
                    copy=dict(lock["copy"], pc=lock["copy"]["pc"] + k))

    def _find_cpm(self) -> dict | None:
        """An early lock on any listened-for CPM grid, as find_burst's dict."""
        from . import cpm

        for g in self.grids:
            lock = cpm.find(g, self.buf, lo=max(0, self.decided.get(g.name, 0) - self.off),
                            hi=len(self.buf) - search_span((), (g.name,)) + 2 * g.T)
            if lock is not None:
                return dict(lock, n_cw=1 + lock["dup"] + lock["n_data"])
        return None

    def feed(self, x: np.ndarray) -> list[tuple[str, dict]]:
        """-> events in order: ("header", find_burst's dict) as a burst
        starts, then ("burst", {"header": that dict, "rx": modem.receive's
        dict or None if it was lost, "audio": the segment received}). The dicts' start/end are stream
        sample indices (samples fed since reset)."""
        if self.blanker is not None:
            x = self.blanker(x)
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
                    for d in self.detectors.values():  # no noise level in silence: skip it
                        d.reset()
                        d.fed = self.off + len(self.buf)
                    break
                try:
                    try:
                        p = modem.find_burst(self.buf, self.bands, self.accept, stats=self._stats())
                        if p["score"] < self.SUSPECT_SCORE:
                            p = self._find_cpm() or p  # a strong CPM burst read as a weak OFDM header
                    except modem.SyncError:
                        p = self._find_cpm()
                finally:
                    self._searched()
                if p is None:
                    p = self._find_copy()  # the preamble faded: the frame pilots and header copy
                if p is None:
                    self._trim(self.keep)
                    break
                if p["start"] + self.off <= self.last_start:
                    break  # the suspect burst just handled, still in the kept window
                self.pending = dict(p, start=p["start"] + self.off, end=p["end"] + self.off)
                if "p0" in p:
                    self.pending["p0"] = p["p0"] + self.off
                if "copy" in p:
                    self.pending["copy"] = dict(p["copy"], pc=p["copy"]["pc"] + self.off)
                # BUSY at once on a clear header (or a copy lock: its pilots
                # passed already); a suspect one waits for its pilots
                self.pilots_ok = p.get("family") == "cpm" or "copy" in p or p["score"] >= self.SUSPECT_SCORE
                self.confirmed = False
                log.debug("receiving %s burst: %d codeword(s), %.1f s, %s score %.2f", p["spec"].name,
                         p["n_cw"], (p["end"] - p["start"]) / FS,
                         "sync" if p.get("family") == "cpm" else "header copy" if "copy" in p else "header", p["score"])
                out.append(("header", self.pending))
            p = self.pending
            if self.off + len(self.buf) < p["end"] + LEADIN_SAMPLES:
                if self.fresh >= self.HOP:
                    self._check_pilots()
                    # the later-header search is for false locks: a confirmed
                    # burst skips it (it was 30% of a Pat exchange's CPU)
                    if self.confirmed:
                        self.fresh = 0
                    elif self._supersede(out):
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
                if "copy" in p:
                    at = self.off + s0  # seg's stream index
                    lock = dict(p, start=p["start"] - at, end=p["end"] - at, p0=p["p0"] - at,
                                copy=dict(p["copy"], pc=p["copy"]["pc"] - at))
                    r = modem.receive(seg, accept=self.accept, copy=lock)
                else:
                    r = modem.receive(seg, [p["band"]], self.accept, head=min(head, len(seg)))
            except modem.SyncError as e:
                log.warning("burst lost: %s", e)
                r = None
            out.append(("burst", {"header": p, "rx": r, "audio": seg}))
            self.last_start = p["start"]
            # a better header inside this burst's span (it was a false lock,
            # and the real one began before it ended) is next, not trimmed away
            if not self.confirmed and self._supersede(out, whole=True):
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


class Blanker:
    """Impulse blanker, after modem73's (RFnexus/modem73, public domain):
    against a slow envelope of |x|, zero samples above ZERO x it and limit
    those above LIMIT x it to that. The envelope follows min(|x|, 3 x env)
    with an 85 ms time constant, so clicks barely lift it. In 10 ms blocks,
    not per sample: a block whose median |x| is over RESYNC x the envelope
    is a level step (a strong station keying up), not a click (those are
    under half a block), and the envelope jumps to it before the block is
    blanked, so a burst's first samples are kept."""

    BLOCK = FS // 100
    ZERO, LIMIT, RESYNC = 8.0, 6.0, 2.0
    TAU = 683  # samples (modem73: 4096 at 48 kHz)
    GUARD = 8  # samples zeroed each side of a zeroed one (1 ms: beat 0 and 16, scripts/blanker_study.py)

    def __init__(self):
        self.env = 0.0
        self.n_blanked = 0  # samples zeroed or limited

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = np.array(x, dtype=np.float64)
        for i in range(0, len(y), self.BLOCK):
            b = y[i:i + self.BLOCK]  # a view: blanked in place
            m = np.abs(b)
            med = float(np.median(m))
            if med > self.RESYNC * self.env:
                self.env = float(np.mean(np.minimum(m, 3 * med)))
            if self.env <= 0:  # digital silence
                continue
            hit = m > self.LIMIT * self.env
            if hit.any():
                zero = m > self.ZERO * self.env
                if self.GUARD and zero.any():
                    zero = np.convolve(zero, np.ones(2 * self.GUARD + 1))[self.GUARD:self.GUARD + len(b)] > 0
                    hit |= zero
                b[hit] *= np.where(zero[hit], 0.0, self.LIMIT * self.env / np.maximum(m[hit], 1e-30))
                self.n_blanked += int(hit.sum())
            self.env += (float(np.mean(np.minimum(m, 3 * self.env))) - self.env) * min(1.0, len(b) / self.TAU)
        return y


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

    def __init__(self, addr, on_packet, on_command=None):
        """`on_command(cmd, payload)`: KISS commands other than data."""
        self.clients, self.lock, self.on_packet, self.on_command = set(), threading.Lock(), on_packet, on_command
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
                    elif srv.on_command is not None:
                        srv.on_command(cmd & 0x0F, payload)
        except OSError:
            pass
        finally:
            with srv.lock:
                srv.clients.discard(self.request)
            log.info("KISS client %s disconnected", peer)


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
