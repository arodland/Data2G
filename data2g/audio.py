"""Between the sound card and the modem: rate conversion, impulse
blanking, and data2g-host's PortAudio streams.

- Decimator (device rate -> FS) and Interpolator (FS -> device rate):
  integer-ratio FIRs whose state carries across blocks.
- Blanker: impulse blanking of what is heard (receiver.Receiver runs its
  input through one).
- Player, Capture: our own FIFOs behind PortAudio callbacks; find_device,
  channels and list_devices pick and describe the cards (pyaudio is
  loaded only by these).

Native: native/core/audio/filters.hpp (Decimator, Interpolator, Blanker),
fifo.hpp (Capture's and Player's FIFOs), audio.hpp (device selection).
Under `pytest --native` tests/native_subs/audio.py substitutes the three
filters.
"""

import logging
import threading
import time

import numpy as np
from scipy import signal as sps

from .config import FS

log = logging.getLogger("data2g.audio")


# --- rate conversion and conditioning ---------------------------------------

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


# --- sound cards (pyaudio) --------------------------------------------------

def _pa_float():
    import pyaudio

    return pyaudio.paFloat32


def find_device(pa, want: str | None, kind: str) -> int | None:
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


def channels(pa, index: int | None, kind: str) -> int:
    """2 if the device (None: the default) has two or more channels of this kind, else 1."""
    d = pa.get_device_info_by_index(index) if index is not None else (
        pa.get_default_input_device_info() if kind == "input" else pa.get_default_output_device_info())
    return 2 if d["maxInputChannels" if kind == "input" else "maxOutputChannels"] >= 2 else 1


def list_devices(pa):
    """--list-audio-devices: index, channels each way, default rate, name."""
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        print(f"{i:3d}  in {d['maxInputChannels']:2d}  out {d['maxOutputChannels']:2d}  "
              f"{int(d['defaultSampleRate']):6d} Hz  {d['name']}")


class Player:
    """Output from our own FIFO through a PortAudio callback. The loop runs on the input clock and hands
    over 0.1 s at a time, which a blocking write can't bridge: with 4800-frame buffers PipeWire padded
    64 zeros after every block (a 1.3 ms hole each 0.1 s, on air too), and with its default buffers
    (~11 ms) any late step underruns. Callback periods are a multiple of 256 frames (2048 at 48 kHz: a
    main thread holding the GIL made 1024 late), and the FIFO starts `lead` ahead of the card."""

    def __init__(self, pa, nout: int, rate: int, device: int | None, lead_s: float):
        self.nout, self.rate, self.lead = nout, rate, int(lead_s * rate)
        self.fifo, self.lock, self.active, self.underruns = bytearray(), threading.Lock(), False, 0
        self.s = pa.open(format=_pa_float(), channels=nout, rate=rate, output=True, output_device_index=device,
                         frames_per_buffer=256 * max(1, rate // 6000), stream_callback=self._fill)

    def _fill(self, _in, n, _t, _status):
        need = n * self.nout * 4
        with self.lock:
            b = bytes(self.fifo[:need])
            del self.fifo[:need]
        if len(b) < need and self.active:
            self.underruns += 1
            log.warning("TX audio underrun (%d)", self.underruns)
        return b + bytes(need - len(b)), 0  # 0: paContinue

    def start(self):
        """Key-up: queue the lead, so a late step doesn't leave the device empty."""
        self.write(np.zeros(self.lead))
        self.active = True

    def write(self, y: np.ndarray):
        with self.lock:
            self.fifo += np.repeat(y.astype(np.float32), self.nout).tobytes()

    def drain(self):
        """Wait until everything queued has left the sound card (then PTT can drop without clipping)."""
        self.active = False  # the last callback coming up short is the end, not an underrun
        while True:
            with self.lock:
                left = len(self.fifo)
            if not left:
                break
            time.sleep(left / (self.nout * 4 * self.rate) + 0.002)
        time.sleep(self.s.get_output_latency())

    def close(self):
        self.s.close()


class Capture:
    """Input through a PortAudio callback into our own FIFO, read a block at a time. A blocking read
    dropped whatever the card's small buffer couldn't hold while the loop was busy (a slow step, the
    drain after TX): ~50 s of a 14 min session, holes of 60-290 ms inside bursts, each one fatal."""

    def __init__(self, pa, nin: int, rate: int, device: int | None):
        self.nin, self.rate = nin, rate
        self.fifo, self.cond, self.overflows, self._late = bytearray(), threading.Condition(), 0, False
        self.s = pa.open(format=_pa_float(), channels=nin, rate=rate, input=True, input_device_index=device,
                         frames_per_buffer=256 * max(1, rate // 6000), stream_callback=self._take)

    def _take(self, data, _n, _t, status):
        if status & 2:  # paInputOverflow: the callback itself was late (GIL), samples are gone
            self.overflows += 1
            log.warning("RX audio overflow (%d)", self.overflows)
        with self.cond:
            self.fifo += data
            self.cond.notify()
        return None, 0  # 0: paContinue

    def read(self, frames: int, stop: threading.Event) -> np.ndarray | None:
        """The next `frames` frames, left channel (None once `stop` is set)."""
        need = frames * self.nin * 4
        with self.cond:
            while len(self.fifo) < need:
                if stop.is_set():
                    return None
                self.cond.wait(0.5)
            b = bytes(self.fifo[:need])
            del self.fifo[:need]
            behind = len(self.fifo) / (self.nin * 4 * self.rate)
        if behind > 1.0 and not self._late:
            log.warning("RX audio %.1f s behind the card", behind)
        self._late = behind > 1.0
        return np.frombuffer(b, dtype=np.float32)[::self.nin].astype(np.float64)

    def close(self):
        self.s.close()
