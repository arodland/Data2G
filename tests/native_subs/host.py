"""`pytest --native` substitutions for data2g.host.Host. See __init__.py."""


def substitutions(native):
    """data2g.host.Host: the C++ Host (native/bindings/bind_host.cpp) when
    its engine is the C++ one (the engine provider's Engine); over a Python
    Engine it stays Python. out_cmd and out_data are this wrapper's list and
    bytearray, filled after each call, so tests read and clear them as
    host.py's."""
    from data2g import host

    from . import originals

    PyHost, N = originals.get((host, "Host"), host.Host), native.host

    class Host:
        __doc__ = PyHost.__doc__

        def __new__(cls, engine, buffer_credit=None):
            if not isinstance(getattr(engine, "_n", None), native.engine.Engine):
                return PyHost(engine, buffer_credit)
            return super().__new__(cls)

        def __init__(self, engine, buffer_credit=None):
            self.engine, self._n = engine, N.Host(engine._n, buffer_credit)
            self.out_cmd, self.out_data = [], bytearray()

        def _take(self, result=None):
            self.out_cmd += self._n.take_cmd()
            self.out_data += self._n.take_data()
            return result

        def command(self, line, peer="127.0.0.1"):
            return self._take(self._n.command(line, peer))

        def client_gone(self):
            return self._take(self._n.client_gone())

        def data_in(self, data):
            return self._take(self._n.data_in(bytes(data)))

        def after_step(self, ptt):
            return self._take(self._n.after_step(bool(ptt)))

        cap = property(lambda s: s._n.cap, lambda s, v: setattr(s._n, "cap", v))
        listening = property(lambda s: s._n.listening, lambda s, v: setattr(s._n, "listening", v))
        buffer_credit = property(lambda s: s._n.buffer_credit, lambda s, v: setattr(s._n, "buffer_credit", v))
        accept_addr = property(lambda s: s._n.accept_addr)
        dial = property(lambda s: None, lambda s, v: setattr(s._n, "dial", v))

    return {(host, "Host"): Host}
