"""CW (Morse) station ID: the keyed tone (`audio`) and `CwSense`, which
hears one on air so a waiting station holds instead of keying over it
(docs/arq.md §7b).

Adapted from ion-hf (crates/ion-dsp/src/cw.rs: PARIS timing, raised-cosine
key edges; crates/ion/src/link/node.rs: CwSense), whose author relicensed
that code for Data2G. The sense here differs: Ion's three tone bins against
six reference bins 300-700 Hz off fire on Data2G's narrow modes (n4, n10)
and on c16r25's 40 ms tones, so here a key-down window has one tone near
the centre and nothing 100 Hz or more from it, a CW ID needs key-up windows
(nothing above the noise anywhere) between its tones, and its hits must all
be the same tone.

Native: cwid/cwid.hpp.
"""

from dataclasses import dataclass, field

import numpy as np

from .config import FCENTER, FS

WPM = 20  # default speed; PARIS timing: a dot is 1.2 / wpm s
MIN_WPM, MAX_WPM = 15, 40  # CwSense's rule (key-up windows within WIN_N) holds down to 15 wpm
FREQ_HZ = float(FCENTER)  # every band and CPM grid holds 1500 Hz (narrowest: n4, 1450-1600 Hz)
RAMP = int(0.005 * FS)  # raised-cosine key edge, samples (5 ms): no clicks, ~200 Hz wide at -40 dB

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.", "H": "....",
    "I": "..", "J": ".---", "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---", "P": ".--.",
    "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..", "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.", "/": "-..-.",
}


def id_text(call: str) -> str:
    """What a station keys: DE and its call less any SSID (W1AW-7 -> DE W1AW)."""
    return "DE " + call.upper().split("-")[0]


def keying(text: str) -> list[tuple[bool, int]]:
    """(key down, dots) runs: elements one dot apart, characters three,
    words seven. Characters without a Morse code are dropped."""
    out: list[tuple[bool, int]] = []
    for word in text.upper().split():
        if out:
            out.append((False, 7))
        first = True
        for c in word:
            p = MORSE.get(c)
            if p is None:
                continue
            if not first:
                out.append((False, 3))
            first = False
            for i, e in enumerate(p):
                if i:
                    out.append((False, 1))
                out.append((True, 3 if e == "-" else 1))
    return out


def unit_samples(wpm: float) -> int:
    return int(round(FS * 1.2 / wpm))


def audio(text: str, wpm: float = WPM, freq: float = FREQ_HZ) -> np.ndarray:
    """`text` keyed at `wpm` on `freq`: a constant-envelope tone, peak 1.0
    (the bursts' peak, engine._start_tx), each edge a raised cosine RAMP
    samples long from the nominal key time. Starts at the first key-down
    and ends RAMP samples after the last key-up."""
    if not MIN_WPM <= wpm <= MAX_WPM:
        raise ValueError(f"CW ID speed {wpm} wpm outside {MIN_WPM}-{MAX_WPM}")
    unit = unit_samples(wpm)
    key = np.concatenate([np.full(d * unit, float(on)) for on, d in keying(text)] + [np.zeros(RAMP)])
    # a step through this kernel rises as 0.5 - 0.5 cos(pi k / RAMP); the kernel's sum is 1
    w = np.sin(np.pi * (np.arange(RAMP) + 0.5) / RAMP)
    env = np.convolve(key, w / w.sum())[:len(key)]
    return env * np.sin(2 * np.pi * freq * np.arange(len(key)) / FS)


def seconds(text: str, wpm: float = WPM) -> float:
    return (sum(d for _, d in keying(text)) * unit_samples(wpm) + RAMP) / FS


@dataclass
class CwBurst:
    """A CW ID among the bursts one PTT sends (arq.engine): no slots, no
    session; its audio is `audio(text, wpm)`."""
    text: str
    wpm: float = WPM
    submode: str = "cw"
    slots: list = field(default_factory=list)
    burst_seq: int = 0
    cap: int = 0


class CwSense:
    """Hears a CW ID about FREQ_HZ. Per WIN samples (20 ms, Hann), the power
    in 25 Hz steps over 1200-1800 Hz against a noise floor (the mean of the
    bins 150 Hz and more off the centre, tracked down fast and up slowly):
    - elsewhere: a bin 100 Hz or more from the tone's over RATIO times the
      floor and LEAK times the strongest tone of the last WIN_N windows (a
      burst: OFDM fills them, n4's other carriers, a CPM tone off the centre);
    - else hit: the tone (the strongest bin within +-50 Hz) over RATIO
      times the floor (a key-down);
    - else quiet (a key-up gap).
    `active` over the last WIN_N windows: at least MIN_HITS hits, MIN_QUIET
    quiet ones and at most MAX_ELSEWHERE elsewhere. A steady carrier on the
    centre never goes quiet. A strong steady carrier 100-300 Hz off the
    centre hides an ID (every window reads elsewhere): then the peer's turn
    may cost a timeout. Inside a burst it may read a CW ID now and then
    (narrow ones near their floor); holding then is harmless, the burst
    holds anyway. `hits_from`: where the HITS_FROM-th latest hit window
    while active started, samples since reset (the receiver's stream clock
    in the engine); -1 before there are that many: a header whose preamble
    started at or before it had a CW ID keyed through it."""

    WIN = FS // 50
    STEP_HZ = 25.0
    K = 12  # bins at centre + STEP_HZ k, |k| <= K: 1200-1800 Hz
    TONE_K = 2  # the tone: the strongest within +-50 Hz (mistuning)
    APART_K = 4  # elsewhere: a bin this far from the tone's (100 Hz; Hann leaks -31 dB there) ...
    RATIO = 10.0  # ... over RATIO times the floor and LEAK times the tone
    LEAK = 10 ** -1.5  # (a key edge splatters to -20 dB 100 Hz off within a window)
    FLOOR_K = 6  # the floor: the mean of bins 150 Hz and more off the centre, tracked
    FLOOR_DOWN, FLOOR_UP = 0.1, 0.004  # an up step's time constant ~5 s: a 16 s burst still reads elsewhere at its end
    WIN_N = 50  # 1 s: past a word gap at 15 wpm. 25 windows (6, 2, 3) heard IDs 2 dB lower, but read n4 at 0 dB as CW
    MIN_HITS, MIN_QUIET, MAX_ELSEWHERE = 12, 4, 6
    HITS_FROM = 3  # hits_from: the HITS_FROM-th latest hit

    def __init__(self, center: float = FREQ_HZ):
        self.center = center
        n = np.arange(self.WIN)
        self._k = np.arange(-self.K, self.K + 1)
        f = center + self.STEP_HZ * self._k
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * (n + 0.5) / self.WIN)
        self._basis = hann * np.exp(-2j * np.pi * np.outer(f, n) / FS)  # one DFT bin per frequency
        self.reset()

    def reset(self):
        self._buf = np.zeros(0)
        self._hist: list[int] = []  # per window: 1 + the tone's step + TONE_K for a hit, 0 quiet, -1 elsewhere
        self._floor = 0.0  # 0: none yet
        self._tones: list[float] = []  # the latest WIN_N windows' tone powers
        self.active = False
        self.n = 0  # samples classified since reset
        self._hits: list[int] = []  # the latest HITS_FROM hit windows' starts while active
        self.hits_from = -1

    def classify(self, x: np.ndarray) -> int:
        p = np.abs(self._basis @ x) ** 2
        k = self._k
        near = np.abs(k) <= self.TONE_K
        kt = k[near][np.argmax(p[near])]
        tone = p[near].max()
        m = max(float(p[np.abs(k) >= self.FLOOR_K].mean()), 1e-30)
        if not self._floor:
            self._floor = m
        thr = self.RATIO * self._floor
        a = self.FLOOR_DOWN if m < self._floor else self.FLOOR_UP
        self._floor *= (m / self._floor) ** a
        # LEAK against the key-down level (the strongest tone of WIN_N windows):
        # a key edge, in a window with part of the tone, splatters to -15 dB of it
        self._tones = (self._tones + [tone])[-self.WIN_N:]
        if p[np.abs(k - kt) >= self.APART_K].max() > max(thr, self.LEAK * max(self._tones)):
            return -1
        return 1 + int(kt) + self.TONE_K if tone > thr else 0

    def feed(self, x: np.ndarray) -> bool:
        """Audio in (any block size); whether a CW ID is on air now."""
        buf = np.concatenate([self._buf, x])
        n = len(buf) // self.WIN
        for i in range(n):
            c = self.classify(buf[i * self.WIN:(i + 1) * self.WIN])
            self._hist.append(c)
            h = self._hist = self._hist[-self.WIN_N:]
            hits = [v for v in h if v > 0]
            # one tone: all hits but one within a step of each other (n4's
            # carriers and c16r25's tones near the centre take turns)
            steady = max(sum(abs(v - u) <= 1 for v in hits) for u in range(1, 2 * self.TONE_K + 2))
            self.active = (len(hits) >= self.MIN_HITS and steady >= len(hits) - 1
                           and h.count(0) >= self.MIN_QUIET and h.count(-1) <= self.MAX_ELSEWHERE)
            if c > 0 and self.active:
                self._hits = (self._hits + [self.n])[-self.HITS_FROM:]
                self.hits_from = self._hits[0] if len(self._hits) == self.HITS_FROM else -1
            self.n += self.WIN
        self._buf = buf[n * self.WIN:]
        return self.active
