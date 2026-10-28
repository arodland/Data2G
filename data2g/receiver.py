"""The live receive front end: audio in, bursts out.

Receiver is what arq.engine.Engine feeds every block it hears. It searches
a short rolling buffer for a preamble and header (modem.find_burst over
sync.StreamDetector's statistics), a CPM grid's early lock (cpm.find; a
front whose first header copy faded waits for its second, cpm.late_lock), or
a burst whose head faded (modem.find_copy, from its header copy); holds
BUSY while the burst arrives; replaces a false lock by a better header
(supersede); and hands the whole burst's audio to modem.receive /
cpm.receive once it has ended. Its input is impulse-blanked (audio.Blanker).

Here:
- search_span, receive_any: the same search offline, over a whole
  recording (replay, studies).
- NoiseProfile: the passband's noise between bursts (the engine records it
  with each burst heard; the outcome model's noise inputs).
- Receiver: the streaming receiver.

Lock dicts (modem.Lock, cpm's) carry sample positions: start, end, p0 and
copy's pc. Inside Receiver they are stream indices (samples fed since
reset); _shift moves a lock between origins.

Native: native/core/receiver/receiver.hpp (Receiver, receive_any, and
decode: a burst's receive, split out so arq::Engine can run it on its
worker) and receiver/noise_profile.hpp. Under `pytest --native`,
tests/native_subs/receiver.py substitutes Receiver, receive_any and
search_span (and engine.Receiver) by name.
"""

from . import threads

# One thread per math library, set before numpy loads. OpenBLAS (numpy)
# and OpenMP (torch) otherwise keep a worker per core spinning between
# the receiver's many small operations: listening took 300-400% CPU on
# 1200 Hz and 600-700% on 2400 Hz, against ~10% of one core with one
# thread each (2026-09-24). Decoding single-threaded stays well inside
# real time (a 32 s burst in ~4 s). An explicit setting still wins.
threads.cap_env(1)

import logging  # noqa: E402
from collections import deque  # noqa: E402

import numpy as np  # noqa: E402

from . import audio, modem  # noqa: E402
from .config import BANDS, FS, SNR_REF_BW_HZ  # noqa: E402
from .waveform import dsp  # noqa: E402

log = logging.getLogger("data2g.receiver")


def _shift(lock: dict, k: int) -> dict:
    """`lock` with every sample position moved by k: start, end, and p0 and
    copy's pc where it has them (native Pending::shift)."""
    q = dict(lock, start=lock["start"] + k, end=lock["end"] + k)
    if "p0" in q:
        q["p0"] += k
    if "copy" in q:
        q["copy"] = dict(q["copy"], pc=q["copy"]["pc"] + k)
    return q


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
        lock = cpm.find(cpm.GRIDS[g], y[:head], wait=True)
        if lock is not None and lock.get("wait"):  # its first header copy faded: both, as Receiver waits for
            lock = cpm.late_lock(cpm.GRIDS[g], y, lock) if lock["header_end"] <= len(y) else None
        if lock is not None:
            return cpm.receive(y, lock)  # a CPM lock beats a suspect OFDM header
    for g in grids:  # the front missed: the mid block and both header copies (as Receiver._find_cpm)
        gr = cpm.GRIDS[g]
        lock = cpm.find_mid(gr, y[:lead + cpm.hdr_span(g, 2) + gr.T])
        if lock is not None and lock["start"] <= lead:
            return cpm.receive(y, lock)
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


class NoiseProfile:
    """Noise across the passband from the audio heard between bursts: per
    sub-band (BANDS_HZ, split at the modes' band edges), the median 0.1 s
    block power over the last WINDOW blocks of idle audio, and its 90th
    percentile over that median (intermittent interference: a 2.39 kHz
    signal on air, recordings/20261002-232711, was +11.6 dB there with
    the rest of the band at +2). A burst's own measurements see only its
    band; this sees the rest (the outcome model's planned inputs).

    The caller feeds what it hears (not while it transmits) and marks
    spans that are not noise: bursts heard, the radio's recovery after its
    own transmission. A block is kept once it is COMMIT_S old (a header is
    heard within ~2 s of its burst's start) and outside every marked span.

    Impulses too: per block, its 10 ms pieces whose peak is over IMPULSE_X
    times the block's median 10 ms RMS (on air, recordings 2026-09/10:
    impulses ~22 dB over the noise, 1-3 ms, clustered; Gaussian peaks over
    80 samples stay under 5x). Counted on the raw input, before the
    blanker: what is on air."""

    BANDS_HZ = ((350, 950), (950, 1300), (1300, 1750), (1750, 2100), (2100, 2700))
    BLOCK = FS // 10
    WINDOW = 600  # blocks: 60 s of idle audio
    COMMIT_S = 3.0
    MIN_BLOCKS = 20
    RECOVER_S = 0.6  # after our own transmission (the radio's receiver comes back: -86 dB on air)
    PIECE = FS // 100  # impulse detection: 10 ms pieces
    IMPULSE_X = 10.0  # 20 dB
    RECENT = 400  # blocks (40 s) of spectra kept for span energy (span_snr_db)

    def __init__(self):
        f = np.fft.rfftfreq(self.BLOCK, 1 / FS)
        self._bins = [(f >= lo) & (f < hi) for lo, hi in self.BANDS_HZ]
        self._win = np.hanning(self.BLOCK)
        # times as sample indices (round(t * FS)): exact, in Python and C++
        self._buf, self._s0 = np.zeros(0), 0  # unblocked audio and its first sample's index
        self._pending = deque()  # (start, end, band powers, impulses) not yet COMMIT_S old
        self._busy = []  # marked (start, end)
        self.kept = deque(maxlen=self.WINDOW)  # band powers
        self.kept_impulses = deque(maxlen=self.WINDOW)  # impulses per kept block
        self.kept_spec = deque(maxlen=self.WINDOW)  # each kept block's power spectrum (span_snr_db's floor)
        self.recent = deque(maxlen=self.RECENT)  # (start, end, power spectrum) of every block fed

    def feed(self, x: np.ndarray, t_start: float):
        """Audio heard from t_start (s). A gap since the last feed (we
        transmitted) starts a new block."""
        s = round(t_start * FS)
        if len(self._buf) and self._s0 + len(self._buf) == s:
            self._buf = np.concatenate([self._buf, x])
        else:
            self._buf, self._s0 = np.asarray(x, dtype=np.float64), s
        while len(self._buf) >= self.BLOCK:
            blk = self._buf[:self.BLOCK]
            spec = self.spectrum(blk)
            self.recent.append((self._s0, self._s0 + self.BLOCK, spec))
            pieces = blk.reshape(-1, self.PIECE)
            ref = np.median(np.sqrt(np.mean(pieces ** 2, axis=1)))
            imp = int(np.sum(np.max(np.abs(pieces), axis=1) > self.IMPULSE_X * ref)) if ref > 0 else 0
            self._pending.append((self._s0, self._s0 + self.BLOCK, [float(spec[b].mean()) for b in self._bins], imp, spec))
            self._buf, self._s0 = self._buf[self.BLOCK:], self._s0 + self.BLOCK
        now = s + len(x)
        commit = round(self.COMMIT_S * FS)
        while self._pending and self._pending[0][1] <= now - commit:
            a, b, p, imp, spec = self._pending.popleft()
            if not any(s0 < b and a < e for s0, e in self._busy):
                self.kept.append(p)
                self.kept_impulses.append(imp)
                self.kept_spec.append(spec)
        self._busy = [(s0, e) for s0, e in self._busy if e > now - 2 * commit]

    def spectrum(self, blk: np.ndarray) -> np.ndarray:
        """A BLOCK's power spectrum (Hann window), as every block is measured."""
        return np.abs(np.fft.rfft(blk * self._win)) ** 2

    def _in_band(self, lo: float, hi: float) -> np.ndarray:
        f = np.fft.rfftfreq(self.BLOCK, 1 / FS)
        return (f >= lo) & (f < hi)

    def span_snr_db(self, spectra, lo: float, hi: float, floor_spectra=None) -> float | None:
        """A burst's in-band SNR from power alone (energy inputs, predictor.N_ENERGY):
        its blocks' mean power in lo..hi Hz over the noise floor there (the
        median of the kept blocks'), excess over 1, in SNR_REF_BW_HZ; dB.
        None without a floor or blocks. `spectra`: the burst's block power
        spectra (spectrum(); span_spectra() for audio already fed).
        `floor_spectra`: the noise to measure against instead of the kept
        blocks (the sim's noise is set per burst, against its peak)."""
        ref = self.kept_spec if floor_spectra is None else floor_spectra
        if (floor_spectra is None and len(ref) < self.MIN_BLOCKS) or not len(ref) or not len(spectra):
            return None
        band = self._in_band(lo, hi)
        floor = float(np.median([s[band].mean() for s in ref]))
        power = float(np.mean([s[band].mean() for s in spectra]))
        if floor <= 0:
            return None
        return float(10 * np.log10(max(power / floor - 1, 1e-3) * (hi - lo) / SNR_REF_BW_HZ))

    def span_z(self, spectra, lo: float, hi: float) -> float | None:
        """How far `spectra`'s mean power in lo..hi Hz is over the noise
        floor, in the kept blocks' own spread (median, MAD) of one block
        scaled to the span: a carrier sense that is the same test in any
        band and any span (span_snr_db's reading scales with both: a 3.5 s
        2.4 kHz window of noise read over -18 dB 44% of the time). None
        without a floor or blocks."""
        if len(self.kept_spec) < self.MIN_BLOCKS or not len(spectra):
            return None
        band = self._in_band(lo, hi)
        ref = np.array([s[band].mean() for s in self.kept_spec])
        floor = float(np.median(ref))
        spread = 1.4826 * float(np.median(np.abs(ref - floor)))
        if spread <= 0:
            return None
        power = float(np.mean([s[band].mean() for s in spectra]))
        return (power - floor) / spread * np.sqrt(len(spectra))

    def span_spectra(self, start: float, end: float) -> list:
        """The fed blocks wholly inside start..end (s)."""
        a, b = round(start * FS), round(end * FS)
        return [s for s0, e0, s in self.recent if s0 >= a and e0 <= b]

    def mark(self, start: float, end: float):
        """Not noise from start to end (s)."""
        self._busy.append((round(start * FS), round(end * FS)))

    def snapshot(self) -> dict | None:
        """{"noise_db": per band, 10 log10 of the median block power;
        "noise_tail_db": per band, the 90th percentile over the median (dB);
        "impulses_per_min": impulses over IMPULSE_X per minute of it;
        "noise_blocks": blocks it is over}, or None with under MIN_BLOCKS."""
        if len(self.kept) < self.MIN_BLOCKS:
            return None
        P = np.array(self.kept)
        med, p90 = np.median(P, axis=0), np.percentile(P, 90, axis=0)
        med = np.maximum(med, 1e-30)  # digital silence
        return {"noise_db": [float(10 * np.log10(m)) for m in med],
                "noise_tail_db": [float(10 * np.log10(q / m)) for q, m in zip(p90, med)],
                "impulses_per_min": 60.0 * sum(self.kept_impulses) / (len(P) * self.BLOCK / FS),
                "noise_blocks": len(P)}


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
        # what a trim with nothing found holds: a CPM burst's start until its
        # mid block and second header copy are in (find_mid)
        self.hold = max([self.keep] + [cpm.hdr_span(g, 2) + FS for g in cpm_grids])
        from .waveform import ofdm, sync

        # each band's detection statistic, computed once per sample (not per search)
        self.detectors = {b: sync.StreamDetector(ofdm.band(b)) for b in self.bands}
        self.blanker = audio.Blanker() if blank else None
        self.reset()

    # new audio between preamble searches (one is ~40 ms of CPU at +-150 Hz):
    # a reply's header must be seen before the sender's deadline (the audio
    # loopback missed replies at 0.5 s and a 1.5 s first search)
    HOP = FS // 4
    # audio kept either side of an OFDM burst's span for its receive (a slip
    # moves its end up to modem.SLIP_MAX later); a burst decodes this long
    # after its closing pilot
    MARGIN = 320

    def reset(self):
        self.buf = np.zeros(0)
        self.off = 0  # stream sample index of buf[0]
        self.pending = None
        self.pilots_ok = False
        self.confirmed = False  # the pending burst's header is clear and its pilots are there
        self.powers = deque(maxlen=self.FLOOR_BLOCKS)  # in-band power of 0.1 s blocks (on_air)
        self._ps, self._pn = 0.0, 0  # the block being summed
        # samples fed since the last preamble search (search at once), less
        # whole hops only: searches stay on the stream's HOP grid. Reset to 0
        # (at the first search, once min_search was in, or as a burst
        # completed: both at a chunk boundary) moved every later search with
        # the chunking
        self.fresh = self.HOP
        self.last_start = -1  # stream start of the last burst handled (not to be found again)
        # per band / CPM grid: starts before this stream index have been
        # searched with all the audio their header needs (not searched again)
        self.decided: dict = {}
        # CPM fronts whose first header copy fell short (cpm.find's candidates,
        # stream indices): read again with both copies once the second is in
        self.cpm_wait: list = []
        for d in getattr(self, "detectors", {}).values():
            d.reset()

    def drop(self):
        """Forget the pending burst: its header was no burst's (arq.engine:
        a CW ID was keyed over its preamble). The search goes on past it."""
        self.last_start = self.pending["start"]
        self.pending = None
        self.pilots_ok = self.confirmed = False

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
        # only p0 (the pilots' grid) is read: the native port shifts just it too
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
        if n < len(self.buf):
            self.off += len(self.buf) - n
            self.buf = self.buf[len(self.buf) - n:]  # not [-n:], which for n = 0 is everything
        for d in self.detectors.values():
            d.trim(self.off)

    def _stats(self, w0: int = 0) -> dict:
        """Per band, the detection statistic for buf[w0:]'s starts (new audio
        fed first), starts already decided masked out (-1)."""
        out = {}
        for b, d in self.detectors.items():
            if d.fed < self.off:  # its audio was trimmed away: start over
                d.restart_at(self.off)
            d.feed(dsp.to_baseband(self.buf[d.fed - self.off:], d.fed))
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
            # a mid block is decided once its second header copy is in
            self.decided[g.name + "/mid"] = max(self.decided.get(g.name + "/mid", 0),
                                                end - (len(cpm.mid_block(g)) + cpm.hdr_len(g)) * g.T - g.T)

    CPM_TAIL = FS // 20  # kept before a CPM burst's end (Receiver.feed): 440 ppm over a 90 s burst
    SUPERSEDE_MARGIN = 0.05  # header score a later header needs over the pending one
    SUSPECT_SCORE = 0.36  # below it a header may be a false lock (false ones score 0.19-0.34)

    def _supersede(self, out: list, whole: bool = False) -> bool:
        """While committed to a burst, keep looking in the newer audio: a
        header past the pending one's, scoring clearly better, replaces it.
        A false lock (data whose preamble was missed reads as a header half
        the time at a 0.25 score floor) then costs a short BUSY, not the
        real burst that follows; the margin keeps a real burst's own data
        from replacing it (false headers score 0.19-0.34, real ones mostly
        0.4-0.6). Under a suspect header, any CPM lock replaces it: a false
        n4 header claiming 52 s hid three FSK bursts (recordings/20261024-190009,
        19:27:36). `whole`: search everything after the pending header (as it
        completes), not just the newest audio."""
        self.fresh %= self.HOP
        p = self.pending
        if p.get("family") == "cpm":
            return False  # scores aren't comparable across families; CPM bursts are long and certain
        q = None
        if "copy" not in p and p["score"] < self.SUSPECT_SCORE:
            q = self._find_cpm()
            if q is not None:  # (else the OFDM search below marks both searched)
                self._searched()
                q = _shift(q, self.off)
        if q is None:
            hdr_end = p["start"] + BANDS[p["band"]].preamble_samples + modem.header_samples(p["band"]) - self.off
            w0 = hdr_end if whole else max(hdr_end, len(self.buf) - self.keep)
            if len(self.buf) - w0 < self.keep // 2:
                return False
            try:
                q = modem.find_burst(self.buf[w0:], self.bands, self.accept, stats=self._stats(w0))
                q = _shift(q, w0 + self.off)
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
                    q = _shift(c, self.off)
            if q is None or q["score"] < p["score"] + self.SUPERSEDE_MARGIN or q["start"] == p["start"]:
                return False
        log.info("%s header (score %.2f) superseded by %s (score %.2f)", p["spec"].name, p["score"],
                 q["spec"].name, q["score"])
        if not whole:  # completing: the caller has already handled p
            out.append(("burst", {"header": p, "rx": None,
                                  "audio": self.buf[max(0, p["start"] - self.off):max(0, q["start"] - self.off)]}))
        self.pending = q
        self.pilots_ok = q.get("family") == "cpm" or "copy" in q or q["score"] >= self.SUSPECT_SCORE
        self.confirmed = False
        out.append(("header", dict(q, stream_end=self.off + len(self.buf))))
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
            a, C = d.outputs_from(self.off)  # from the stream index both the buffer and C cover
            x = self.buf[a - self.off:]
            C = C[:, :max(0, len(x) - modem.M + 1)]
            lock = modem.find_copy(x, band, self.accept, C, level)
            if lock is not None and (best is None or lock["score"] > best[0]["score"]):
                best = (lock, a - self.off)
        if best is None:
            return None
        lock, k = best
        return _shift(lock, k)

    def _find_cpm(self) -> dict | None:
        """A lock on any listened-for CPM grid, as find_burst's dict: a
        waiting candidate's (both header copies) once its second copy is in,
        else an early lock. A front whose first copy fell short waits."""
        from . import cpm

        end = self.off + len(self.buf)
        self.cpm_wait = [c for c in self.cpm_wait if c["first"] >= self.off]  # trimmed with a burst handled
        for c in [c for c in self.cpm_wait if c["header_end"] <= end]:
            self.cpm_wait.remove(c)
            g = cpm.GRIDS[c["band"]]
            lock = cpm.late_lock(g, self.buf, dict(c, start=c["start"] - self.off))
            if lock is not None:
                return dict(lock, n_cw=1 + lock["dup"] + lock["n_data"])
        for g in self.grids:
            lock = cpm.find(g, self.buf, lo=max(0, self.decided.get(g.name, 0) - self.off),
                            hi=len(self.buf) - search_span((), (g.name,)) + 2 * g.T, wait=True)
            if lock is None:
                continue
            if not lock.get("wait"):
                return dict(lock, n_cw=1 + lock["dup"] + lock["n_data"])
            c = dict(lock, start=lock["start"] + self.off, header_end=lock["header_end"] + self.off)
            c["first"] = c["header_end"] - cpm.hdr_span(g.name, 2)  # its earliest tile start: kept until read
            # the next search may meet the same front again before it is decided
            if not any(w["band"] == g.name and abs(w["start"] - c["start"]) < cpm.hdr_span(g.name, 1)
                       for w in self.cpm_wait):
                self.cpm_wait.append(c)
        for g in self.grids:  # the front missed: the mid block and both header copies
            lock = cpm.find_mid(g, self.buf, lo=max(0, self.decided.get(g.name + "/mid", 0) - self.off),
                                hi=len(self.buf) - (len(cpm.mid_block(g)) + cpm.hdr_len(g) - 1) * g.T)
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
                self.fresh %= self.HOP
                # digital silence (a muted or looped-back input) has no noise
                # to normalize by, and filter ringing then reads as preambles
                # (a burst's own tail read as a header at 1e-6 noise). A
                # radio's noise at the sound card is far above -80 dBFS.
                if np.sqrt(np.mean(self.buf[-self.min_search :] ** 2)) < SILENCE_RMS:
                    self._trim(self.keep)
                    for d in self.detectors.values():  # no noise level in silence: skip it
                        d.restart_at(self.off + len(self.buf))
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
                    self._trim(self.hold)  # (a waiting candidate's start is within it)
                    break
                if p["start"] + self.off <= self.last_start:
                    break  # the suspect burst just handled, still in the kept window
                self.pending = _shift(p, self.off)
                # BUSY at once on a clear header (or a copy lock: its pilots
                # passed already); a suspect one waits for its pilots
                self.pilots_ok = p.get("family") == "cpm" or "copy" in p or p["score"] >= self.SUSPECT_SCORE
                self.confirmed = False
                log.debug("receiving %s burst: %d codeword(s), %.1f s, %s score %.2f", p["spec"].name,
                         p["n_cw"], (p["end"] - p["start"]) / FS,
                         "sync" if p.get("family") == "cpm" else "header copy" if "copy" in p else "header", p["score"])
                # stream_end: the stream index of the audio fed so far (the caller's now)
                out.append(("header", dict(self.pending, stream_end=self.off + len(self.buf))))
            p = self.pending
            if self.off + len(self.buf) < p["end"] + self.MARGIN:
                if self.fresh >= self.HOP:
                    self._check_pilots()
                    # the later-header search is for false locks: a confirmed
                    # burst skips it (it was 30% of a Pat exchange's CPU)
                    if self.confirmed:
                        self.fresh %= self.HOP
                    elif self._supersede(out):
                        continue
                break
            if p.get("family") == "cpm":
                from . import cpm

                r = cpm.receive(self.buf, dict(p, start=p["start"] - self.off))
                out.append(("burst", {"header": p, "rx": r, "audio": self.buf[max(0, p["start"] - self.off):
                                                                               p["end"] - self.off]}))
                self.last_start = p["start"]
                # keep a little before its end: a CPM burst has no lead-out silence,
                # and with the sender's clock fast (30 ppm) the next back-to-back
                # burst's front began 4 samples before this one's computed end; cut
                # there, every second burst was missed (data2g-bulk)
                self._trim(len(self.buf) - (p["end"] - self.CPM_TAIL - self.off))
                self.pending = None
                continue
            s0 = max(0, p["start"] - self.MARGIN - self.off)
            seg = self.buf[s0 : p["end"] + self.MARGIN - self.off]
            head = p["start"] - self.off - s0 + modem.receive_head(p["band"])
            try:
                if "copy" in p:
                    at = self.off + s0  # seg's stream index
                    r = modem.receive(seg, accept=self.accept, copy=_shift(p, -at))
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
                self._trim(len(self.buf) - max(0, self.pending["start"] - self.MARGIN - self.off))
                continue
            # a suspect (possibly false) burst may have claimed an end past a
            # real preamble that began inside it: keep the last search window
            # (a confident one keeps nothing of itself: its own data would
            # false-lock and hold BUSY after every real burst), and every
            # CPM front waiting inside it for its second header copy
            cut = p["end"] - self.off
            if p["score"] < self.SUSPECT_SCORE:
                cut = min([cut, len(self.buf) - self.keep] + [c["first"] - self.off for c in self.cpm_wait])
            self._trim(len(self.buf) - max(0, cut))
            self.pending = None
        return out

