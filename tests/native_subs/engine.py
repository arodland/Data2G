"""`pytest --native` substitutions for data2g.arq.engine.Engine. See __init__.py."""


def substitutions(native):
    """data2g.arq.engine.Engine: the C++ Engine in its deterministic (sync)
    mode behind engine.Engine's interface (native/bindings/bind_engine.cpp).
    The policy factory, random.Random(seed), the session's Python view and
    phy.tx_audio stay the Python objects, called as engine.py calls them; a
    KissLink that isn't the C++ one falls back to the Python Engine. host.py
    from-imports Engine, so it is listed there too."""
    import random

    import numpy as np

    from data2g import host, modem
    from data2g.arq import engine as E
    from data2g.arq import phy as PHY

    PyEngine, N, KL = E.Engine, native.engine, native.kisslink.KissLink

    class HandRefs:
        """engine.py's `hand_refs` dict (burst id -> (end s, cfo Hz, snr dB)),
        over the C++ engine's map: a test that reads refs or invents one for
        HAND SEND to answer sees the same object either way."""

        def __init__(self, n):
            self._n = n

        def __getitem__(self, k):
            for i, end, cfo, snr in self._n.hand_refs():
                if i == k:
                    return (end, cfo, snr)
            raise KeyError(k)

        def __setitem__(self, k, v):
            self._n.hand_ref_set(int(k), float(v[0]), float(v[1]), float(v[2]))

        def __iter__(self):
            return iter([i for i, *_ in self._n.hand_refs()])

        def __len__(self):
            return len(self._n.hand_refs())

        def __contains__(self, k):
            return any(i == k for i, *_ in self._n.hand_refs())

        def __bool__(self):
            return bool(self._n.hand_refs())

    class Engine:
        __doc__ = PyEngine.__doc__

        def __new__(cls, call, policy=None, ptt_delay_s=0.1, record_dir=None, seed=None, min_header_score=0.0,
                    kiss=None, stats_interval_s=60.0):
            if kiss is not None and not isinstance(kiss, KL):
                return PyEngine(call, policy, ptt_delay_s, record_dir, seed, min_header_score, kiss, stats_interval_s)
            return super().__new__(cls)

        def __init__(self, call, policy=None, ptt_delay_s=0.1, record_dir=None, seed=None, min_header_score=0.0,
                     kiss=None, stats_interval_s=60.0):
            self._n = N.Engine(call, policy or E.GearShifter, ptt_delay_s, record_dir, None, min_header_score, kiss,
                               stats_interval_s, rng=random.Random(seed), tx_audio=lambda b: PHY.tx_audio(b),
                               dd=bool(PHY.DD))
            self.accept = E.receive_accept(min_header_score)
            self._kiss_rx = []

        def __getattr__(self, name):  # step, session, connect, events, tx, _extra, ...
            return getattr(self._n, name)

        def step(self, x):
            return self._n.step(np.asarray(x, dtype=np.float64))

        n = property(lambda s: s._n.n, lambda s, v: setattr(s._n, "n", v))
        hand_refs = property(lambda s: HandRefs(s._n))
        # a test's burst on air: engine.py's tx list [burst, audio, mode], which
        # tests/test_hand.py sets to anchor a Hand window, onto the C++ Tx
        tx = property(lambda s: s._n.tx,
                      lambda s, t: s._n.test_tx(t[0].submode, np.asarray(t[1], dtype=np.float64)))
        _tx_kiss = property(lambda s: s._n.tx_kiss, lambda s, v: setattr(s._n, "tx_kiss", v))
        _test_acks = property(lambda s: s._n.test_acks, lambda s, v: setattr(s._n, "test_acks", v))
        _cw_until = property(lambda s: s._n.cw_until)

        @property
        def _hand_tx(self):  # engine.py's [first audio sample, audio, channel]
            t = self._n.hand_tx()  # the C++ engine's (first sample, channel)
            return None if t is None else [t[0], None, t[1]]

        id_interval_s = property(lambda s: s._n.id_interval_s, lambda s, v: setattr(s._n, "id_interval_s", v))
        cw_id = property(lambda s: s._n.cw_id, lambda s, v: setattr(s._n, "cw_id", v))
        pending = property(lambda s: s._n.pending, lambda s, v: setattr(s._n, "pending", v))
        hand_round = property(lambda s: s._n.hand_round, lambda s, v: setattr(s._n, "hand_round", v))
        cw_wpm = property(lambda s: s._n.cw_wpm, lambda s, v: setattr(s._n, "cw_wpm", v))

        @property
        def kiss_rx(self):
            self._kiss_rx += self._n.take_kiss_rx()
            return self._kiss_rx

        def take_kiss_rx(self):
            out, self._kiss_rx = self.kiss_rx, []
            return out

        # None while the C++ receiver is the engine's own; a test may set a Python one
        receiver = property(lambda s: s._n.receiver, lambda s, r: setattr(s._n, "receiver", r))

    return {(E, "Engine"): Engine, (host, "Engine"): Engine}
