"""One station's live stack, clocked by audio samples (gear-shifter phase H).

`Engine.step(audio_in) -> (audio_out, ptt)` is called once per audio block
at FS: it feeds what was heard to the streaming receiver (tnc.Receiver),
hands headers and bursts to the ARQ session and the gear shifter, and
plays out whatever the session sends, half duplex. Its clock is the
sample count, so the same code runs behind a sound card (data2g/host.py)
or back to back with another Engine through a simulated channel, faster
than real time (tests/test_engine.py).

Recording (`record_dir`): every burst heard is saved as audio with what
the receiver made of it, and every burst sent as its slots, so a session
recorded at both ends can be replayed offline with exact knowledge of
what was on air (scripts/replay.py).
"""

import json
import logging
import random
import time
from pathlib import Path

import numpy as np

from .. import cpm, modem
from ..config import FS
from . import frames as F
from . import link as L
from ..tnc import Receiver
from . import phy as PHY
from . import session as S
from .policy import GearShifter

log = logging.getLogger("data2g.engine")

MAX_BURST_S = 16.0  # longest burst accepted from a header (the shifter's largest is 12 s)
# ID frames (docs/arq.md §7a): in a session, one goes ahead of this station's
# turn at least this often (FCC 97.119: every 10 minutes), and one more once
# the session is over: after its last burst (a DISC_ACK), else ID_GUARD_S
# after it closed, when the peer's own ID (after its DISC_ACK) has been heard
ID_INTERVAL_S = 600.0
ID_GUARD_S = S.REPLY_START_S


class Recorder:
    """events.jsonl (one JSON object per line), rx_NNNNN.npz per burst
    heard, and audio_in.f16: everything the receiver was fed, float16 at
    FS, sample i at engine time i / FS (zeros while transmitting), so a
    burst the receiver missed can still be looked for offline."""

    def __init__(self, path: str | Path, call: str):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = open(self.dir / "events.jsonl", "a", buffering=1)
        self.audio_file = open(self.dir / "audio_in.f16", "ab")
        self.n = 0
        self.event("start", call=call, wall=time.time(), fs=FS)

    def audio(self, x: np.ndarray):
        self.audio_file.write(x.astype(np.float16).tobytes())

    def event(self, kind: str, **kw):
        self.log.write(json.dumps(dict(kind=kind, **kw), default=_jsonable) + "\n")

    def rx(self, t: float, audio: np.ndarray, header: dict, r: dict | None, meas: dict | None) -> str:
        name = f"rx_{self.n:05d}.npz"
        self.n += 1
        np.savez_compressed(self.dir / name, audio=audio.astype(np.float32))
        self.event("rx", t=t, file=name, submode=header["spec"].name, n_cw=header["n_cw"],
                   score=header["score"], lost=r is None, meas=meas)
        return name


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, bytes):
        return v.hex()
    return str(v)


class Engine:
    def __init__(self, call: str, policy=None, ptt_delay_s: float = 0.1, record_dir=None, seed: int | None = None,
                 min_header_score: float = 0.0, kiss=None, stats_interval_s: float = 60.0):
        """`kiss`: a data2g.kisslink.KissLink to serve too (the KISS
        personality): its bursts are peeled off what's heard, and it sends
        when no ARQ session is under way and the channel is free."""
        self.call = call.upper()
        self.aliases = ()
        self.policy_factory = policy or GearShifter
        self.rng = random.Random(seed)
        self.accept = modem.Accept.of(None, MAX_BURST_S, min_header_score)
        self.receiver = Receiver(self.accept, cpm_grids=tuple(cpm.GRIDS))
        self.ptt_delay = int(ptt_delay_s * FS)
        self.rec = Recorder(record_dir, self.call) if record_dir else None
        self.n = 0  # samples processed
        self.tx = None  # [burst, audio, position]
        self.session = None
        self.store = {}  # soft bits per codeword key, across bursts (phy.ModemRx)
        self.chat = False
        self._extra: list = []  # bursts outside any session (CQ frames), sent when the channel is free
        self._events: list[str] = []  # host notifications from outside the session (CQFRAME)
        self.kiss = kiss
        self.stats_interval_s = stats_interval_s
        self.kiss_rx: list[bytes] = []  # frames heard for KISS clients
        self._kiss_busy = 0  # samples of unbroken BUSY a queued KISS burst has waited
        self._kiss_deferred = False  # the queued KISS burst has waited on BUSY
        self._kiss_slot = 0  # next p-persistence slot, samples
        self.id_interval_s = ID_INTERVAL_S
        self._id_for = None  # the session ID frames are being sent for
        self._id_due: float | None = None  # its next ID (None: closed, its last ID pending or sent)
        self._id_pending: list | None = None  # [session, earliest time]: its last ID, after it closed
        self._hold = 0.0  # nothing new goes before this (the peer's last ID may follow its DISC_ACK)
        self._new_session()

    # -- host side ---------------------------------------------------------------------

    @property
    def now(self) -> float:
        return self.n / FS

    @property
    def tx_samples(self) -> int:
        """The burst on air's length in samples (its audio, PTT delay included), 0 if none."""
        return len(self.tx[1]) if self.tx else 0

    def _new_session(self):
        self.session = S.Session(self.call, self.policy_factory(), rng=random.Random(self.rng.random()),
                                 aliases=self.aliases, stats_interval_s=self.stats_interval_s)
        self.session.set_chat(self.chat)
        self.store = {}

    def listen(self, on: bool = True):
        """Answer calls to this station (on), or stop (an idle session)."""
        if self.session.state in (S.CLOSED, S.LISTEN, S.IDLE):
            self._new_session()
            if on:
                self.session.listen()

    def set_call(self, call: str, *aliases: str):
        """Takes effect now if no session is under way, else for the next.
        `aliases`: more calls to answer connects to."""
        self.call = call.upper()
        self.aliases = tuple(a.upper() for a in aliases)
        if self.session.state in (S.IDLE, S.LISTEN, S.CLOSED):
            listening = self.session.state == S.LISTEN
            self.listen(listening)

    def abort(self):
        """Drop the session and anything on air, at once (no DISC)."""
        self.tx = None
        self.receiver.reset()
        self._new_session()

    def connect(self, peer: str, cap: int):
        if self.session.state not in (S.IDLE, S.LISTEN, S.CLOSED):
            raise RuntimeError(f"session {self.session.state}")
        self._new_session()
        self.session.connect(peer, cap, self.now)

    def set_chat(self, on: bool):
        self.chat = on
        self.session.set_chat(on)

    def events(self) -> list[str]:
        """Host notifications since the last call (CONNECTED ..., DISCONNECTED ...,
        CQFRAME call cap)."""
        ev, self.session.events[:] = self._events + list(self.session.events), []
        self._events = []
        return ev

    def send_cq(self, call: str, cap: int):
        """A CQ frame (VARA's CQFRAME): call and bandwidth, heard by anyone,
        no session. Not while one is under way."""
        if self.session.state not in (S.IDLE, S.LISTEN, S.CLOSED):
            raise RuntimeError(f"session {self.session.state}")
        self._extra.append(self._open_frame(cap, F.T_CQ, F.pack_call(call) + bytes([cap])))

    def _open_frame(self, cap: int, ext: int, body: bytes) -> L.TxBurst:
        """A control-only burst anyone can read (frame type SESSION, mask 0),
        in the cap's connect mode."""
        policy = self.session.policy
        mode = policy.connect_mode(cap)
        payloads = F.Control(F.Core(ftype=F.SESSION), {ext: body}).pack(policy.payload_bytes(mode))
        return L.TxBurst(mode, [L.Slot(L.ctl_mask(0, i, 0), 0, p) for i, p in enumerate(payloads)], 0)

    def _id_frame(self, s: S.Session) -> L.TxBurst:
        log.info("TX ID %s", s.call)
        return self._open_frame(s.cap, F.T_ID, F.pack_call(s.call) + s.station.key.to_bytes(2, "big"))

    def _id_check(self, t: float):
        """Track the session ID frames are owed for; once it closes, its last ID is pending."""
        s = self.session
        if s.station is None:
            return
        if s is not self._id_for:
            self._id_for, self._id_due = s, t + self.id_interval_s
        if s.state == S.CLOSED and self._id_due is not None:
            self._id_due = None
            self._id_pending = [s, t + ID_GUARD_S]
            self._hold = t + ID_GUARD_S

    def _with_id(self, burst: L.TxBurst, t: float) -> list:
        """A session's burst -> the bursts to send back to back: an ID first
        when one is due, or the pending last ID after a closed session's
        final burst (its DISC_ACK)."""
        s = self.session
        if s.state == S.CLOSED and self._id_pending is not None:
            self._id_pending = None
            return [burst, self._id_frame(s)]
        if s is self._id_for and self._id_due is not None and t >= self._id_due:
            self._id_due = t + self.id_interval_s
            return [self._id_frame(s), burst]
        return [burst]

    # -- audio side --------------------------------------------------------------------

    def step(self, x: np.ndarray) -> tuple[np.ndarray, bool]:
        """One block heard (ignored while transmitting) -> the block to play
        (a burst's peak at 1.0), PTT."""
        k = len(x)
        t = self.now + k / FS  # the block's end: when anything in it is known
        out = np.zeros(k)
        if self.rec:
            self.rec.audio(x if self.tx is None else np.zeros(k))
        if self.tx is None:
            self._hear(x, t)
            self._id_check(t)
            bursts = main = None
            held = t < self._hold and self.session.state != S.CLOSED  # a closed session's DISC_ACK goes
            if not self.receiver.busy and not held:  # a burst still arriving holds any reply (half duplex)
                burst = self.session.poll(t)
                self._id_check(t)
                if burst is not None:
                    bursts, main = self._with_id(burst, t), burst
                elif self._id_pending is not None and t >= self._id_pending[1]:
                    bursts, self._id_pending = [self._id_frame(self._id_pending[0])], None
                elif self._extra and t >= self._hold:
                    bursts = [self._extra.pop(0)]
            if (bursts is None and self.kiss is not None and t >= self._hold
                    and self.session.state in (S.IDLE, S.LISTEN, S.CLOSED)):
                burst = self._kiss_burst(k)  # KISS only between ARQ sessions
                bursts = [burst] if burst is not None else None
            else:
                self._kiss_busy = 0
            if bursts is not None:
                self._start_tx(bursts, t, main or bursts[0])
        if self.tx is not None:
            burst, audio, pos = self.tx  # burst: the session's, when an ID rides with it
            n = min(k, len(audio) - pos)
            out[:n] = audio[pos:pos + n]
            self.tx[2] += n
            if self.tx[2] >= len(audio):
                self.tx = None
                self.receiver.reset()  # our own transmission was not heard
                self.session.on_tx_end(burst, self.now + n / FS)
        self.n += k
        return out, self.tx is not None

    def next_event(self) -> float | None:
        return self.session.next_event()

    def _hear(self, x: np.ndarray, t: float):
        for kind, ev in self.receiver.feed(x):
            if kind == "header":
                self.session.on_header(ev["spec"].name, ev["n_cw"], t)
                continue
            r, h = ev["rx"], ev["header"]
            meas = PHY.measure(r) if r is not None else None
            if self.rec:
                self.rec.rx(t, ev["audio"], h, r, meas)
            if r is None:
                log.info("RX %s x%d: header heard (score %.2f), burst lost", h["spec"].name, h["n_cw"], h["score"])
                continue
            rx = PHY.ModemRx(r, self.store, PHY.DD_BUDGET_S)
            # in a session, its peer's bursts are the likely ones: a control
            # codeword under the session's key (the station's first decode,
            # remembered) claims the burst before KISS tries its keys on it
            # (two failed decodes per ARQ burst: ~10% of a Pat exchange's CPU)
            st = self.session.station
            ours = (st is not None and self.session.state in (S.CONNECTED, S.DISCONNECTING)
                    and rx.decode(0, L.ctl_mask(st.peer, 0, st.key), 0, None) is not None)
            if self.kiss is not None and not ours:
                frames = self.kiss.on_burst(r)
                if frames is not None:  # a KISS burst: not the session's
                    self.kiss_rx += frames
                    continue
            if self._cq(rx):
                continue
            self.session.policy.observe(meas, r["spec"].name, t)
            self.session.on_rx(rx, t)

    def _kiss_burst(self, k: int):
        """The next KISS burst if it may go now. A burst that waited on BUSY
        is p-persistent after it: each slot (SLOTTIME) is taken with
        probability (P + 1) / 256, so stations that queued under the same
        burst don't all key up as it ends. One queued on a free channel (a
        reply) goes at once. BUSY holds a burst, but not past busy_limit_s
        of it unbroken (a stuck BUSY). `k`: this block's samples."""
        link = self.kiss
        if not link.queue:
            self._kiss_busy, self._kiss_deferred = 0, False
            return None
        if self.receiver.busy:
            self._kiss_busy += k
            self._kiss_deferred = True
            if self._kiss_busy < link.busy_limit_s * FS:
                return None
            log.warning("KISS: BUSY for %.0f s, sending anyway", self._kiss_busy / FS)
        else:
            self._kiss_busy = 0
            if self._kiss_deferred:
                if self.n < self._kiss_slot:
                    return None
                if self.rng.random() >= (link.persist + 1) / 256:
                    self._kiss_slot = self.n + int(link.slot_s * FS)
                    return None
        self._kiss_busy, self._kiss_deferred = 0, False
        return link.next_burst()

    def _cq(self, rx) -> bool:
        """A CQ frame (notified: CQFRAME call cap) or an ID frame (ID call key)?
        Then nothing else to do."""
        first = rx.decode(0, L.ctl_mask(0, 0, 0), 0, None)
        if first is None:
            return False
        core = F.Core.unpack(first)
        if core.ftype != F.SESSION or core.n_ctl > rx.n_cw:
            return False
        payloads = [first] + [rx.decode(i, L.ctl_mask(0, i, 0), 0, None) for i in range(1, core.n_ctl)]
        if None in payloads:
            return False
        # a malformed one (docs/arq.md §4) is dropped as if it had not decoded
        try:
            ext = F.Control.unpack(payloads).ext
            if F.T_ID in ext:
                body = ext[F.T_ID]
                if len(body) < 10:
                    raise ValueError(f"ID of {len(body)} B")
                call, key = F.unpack_call(body[:8]), int.from_bytes(body[8:10], "big")
                log.info("RX ID %s (session %04x)", call, key)
                self._events.append(f"ID {call} {key}")
                return True
            if F.T_CQ not in ext:
                return False
            body = ext[F.T_CQ]
            if len(body) < 9:
                raise ValueError(f"CQ of {len(body)} B")
            self._events.append(f"CQFRAME {F.unpack_call(body[:8])} {body[8]}")
            return True
        except ValueError as e:
            log.warning("RX malformed CQ/ID frame (%s): dropped", e)
            return False

    def _start_tx(self, bursts: list, t: float, main: L.TxBurst):
        """Bursts back to back on one PTT (an ID frame with a session's
        burst). `main`: the one the session is told has gone, when all have."""
        xs = [PHY.tx_audio(b) for b in bursts]
        # peak at full scale: the modem's unit-RMS audio peaks at 2-3, and a
        # sound card clips at 1 (the audio loopback found 64-QAM bursts wrecked)
        audio = np.concatenate([np.zeros(self.ptt_delay)] + [x / np.max(np.abs(x)) for x in xs])
        self.tx = [main, audio, 0]
        if self.rec:
            # one event per burst, at its own start: the first's seconds include
            # the PTT delay, as a lone burst's always did (scripts/replay.py)
            for i, (b, x) in enumerate(zip(bursts, xs)):
                seconds = (len(x) + (self.ptt_delay if i == 0 else 0)) / FS
                self.rec.event("tx", t=t, submode=b.submode, burst_seq=b.burst_seq,
                               slots=[dict(mask=list(s.mask_id), rv=s.rv, payload=s.payload) for s in b.slots],
                               seconds=seconds)
                t += seconds

