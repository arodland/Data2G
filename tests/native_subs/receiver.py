"""`pytest --native` substitutions for data2g.receiver. See __init__.py."""


def substitutions(native):
    """data2g.receiver: search_span, receive_any and the streaming Receiver.
    Receiver is the C++ one behind receiver.Receiver's interface (engine
    from-imports it, so it is listed there too); it is the Python one when a
    test patches its search (_stats, _searched) or modem's compiled-in
    constants. Specs come back by name and are restored here."""
    import numpy as np

    from data2g import config, cpm, modem, receiver
    from data2g.arq import engine
    from data2g.waveform import sync

    T = native.receiver
    PyReceiver = receiver.Receiver
    py_receive_any = receiver.receive_any
    min_score = dict(modem.HEADER_MIN_SCORE)

    def same_config():
        return modem.PROTOCOL_VERSION == config.PROTOCOL_VERSION and modem.HEADER_MIN_SCORE == min_score

    def header(d):
        return dict(d, spec=(cpm.SPECS if d.get("family") == "cpm" else config.SUBMODES)[d["spec"]])

    def rx(d):
        if d is None:
            return None
        if d.get("family") == "cpm":
            return dict(d, spec=cpm.SPECS[d["spec"]])
        start, f, metric, alts = d["acq"]
        return dict(d, spec=config.SUBMODES[d["spec"]],
                    acq=sync.Acquisition(preamble_start=start, freq_offset=f, metric=metric, alternatives=alts))

    def event(kind, d):
        return (kind, header(d)) if kind == "header" else (kind, dict(d, header=header(d["header"]), rx=rx(d["rx"])))

    class Receiver(PyReceiver):
        __doc__ = PyReceiver.__doc__

        def __init__(self, accept, cpm_grids=(), blank=True):
            cls = type(self)
            self._n = None
            if cls._stats is PyReceiver._stats and cls._searched is PyReceiver._searched and same_config():
                self.accept, self.bands = accept, accept.bands
                self._n = T.Receiver(accept, list(cpm_grids), blank)
            else:
                super().__init__(accept, cpm_grids, blank)

        def reset(self):
            return super().reset() if self._n is None else self._n.reset()

        def expect(self, grids, lo, hi, cfo, until=None):
            if self._n is None:
                return super().expect(grids, lo, hi, cfo, until)
            return self._n.expect(list(grids), lo, hi, cfo, until)

        def drop(self):
            return super().drop() if self._n is None else self._n.drop()

        @property
        def pending(self):
            if self._n is None:
                return self.__dict__.get("_pending")
            p = self._n.pending
            return None if p is None else header(p)

        @pending.setter
        def pending(self, v):
            self.__dict__["_pending"] = v

        def feed(self, x):
            if self._n is None:
                return super().feed(x)
            return [event(k, d) for k, d in self._n.feed(np.asarray(x, dtype=np.float64))]

        busy = property(lambda s: PyReceiver.busy.fget(s) if s._n is None else s._n.busy)
        channel_busy = property(lambda s: PyReceiver.channel_busy.fget(s) if s._n is None else s._n.channel_busy)
        n_expected = property(lambda s: PyReceiver.n_expected.fget(s) if s._n is None else s._n.n_expected)
        expect_waits = property(lambda s: PyReceiver.expect_waits.fget(s) if s._n is None else s._n.expect_waits)
        on_air = property(lambda s: PyReceiver.on_air.fget(s) if s._n is None else s._n.on_air)

    def receive_any(y, lead=0, cpm_grids=None):
        if not same_config():
            return py_receive_any(y, lead, cpm_grids)
        return rx(T.receive_any(np.asarray(y, dtype=np.float64), int(lead),
                                None if cpm_grids is None else list(cpm_grids)))

    return {
        (receiver, "search_span"): lambda bands, cpm_grids=(): T.search_span(list(bands), list(cpm_grids)),
        (receiver, "receive_any"): receive_any,
        (receiver, "Receiver"): Receiver,
        (engine, "Receiver"): Receiver,
    }
