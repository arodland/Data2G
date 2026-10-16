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
        id_interval_s = property(lambda s: s._n.id_interval_s, lambda s, v: setattr(s._n, "id_interval_s", v))

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
