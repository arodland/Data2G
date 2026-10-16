"""`pytest --native` substitutions for data2g.arq.phy and data2g.kisslink. See __init__.py."""


def substitutions(native):
    """data2g.arq.phy (TX audio, ModemRx with DD, measure) and
    data2g.kisslink (KissLink, AX.25 parsing, data framing) in C++. ModemRx stays Python
    for a mode that isn't the configured one, or while a test patches a
    codec function it calls (C++ never calls back into them); it reads
    phy.DD when made, and its store stays the caller's dict."""
    import time

    from data2g import codes, kisslink
    from data2g.arq import modes
    from data2g.arq import phy as PHY

    P, K = native.phy, native.kisslink
    py_rx, py_tx, py_soft, py_measure = PHY.ModemRx, PHY.tx_audio, PHY.soft_bits, PHY.measure
    watched = ("decode_raw", "decode_buffer", "decode_posterior", "decode_llrs", "payloads", "check", "combine", "flip",
               "encode", "encode_info", "descramble")

    def own(spec):
        return modes.MODES.get(spec.name) == spec

    def substituted(f):  # the codes provider's, not a test's patch
        return getattr(f, "__module__", None) == f"{__package__}.codes"

    class ModemRx:
        """A class, as phy.ModemRx is, so a study can wrap its methods
        (scripts/crc_exposure.py patches __init__ and decode); the decoding
        is the C++ object's."""
        __doc__ = py_rx.__doc__

        def __new__(cls, r, store, dd_budget=None):
            if not own(r["spec"]) or not all(substituted(getattr(codes, f)) for f in watched):
                return py_rx(r, store, dd_budget)
            return super().__new__(cls)

        def __init__(self, r, store, dd_budget=None):
            self._n = P.ModemRx(r, store, dd_budget, bool(PHY.DD))
            self.spec, self.n_cw, self.submode = r["spec"], r["n_cw"], r["spec"].name
            self.n_ctl_slots, self.store, self.r = r.get("n_ctl_slots", 0), store, r

        _spec = py_rx._spec

        def decode(self, slot, mask_id, rv, key):
            return self._n.decode(slot, mask_id, rv, key)

        def raw(self, slot):
            return self._n.raw(slot)

        def forget(self, key):
            return self._n.forget(key)

    def tx_audio(burst):
        return P.tx_audio(burst) if burst.submode in modes.MODES else py_tx(burst)

    def soft_bits(r):
        return P.soft_bits(r) if own(r["spec"]) else py_soft(r)

    def measure(r):
        return P.measure(r) if own(r["spec"]) else py_measure(r)

    def parse_ax25(frame):
        t = K.parse_ax25(bytes(frame))
        return None if t is None else kisslink.Ax25(*t)

    def KissLink(cap=2, queue=None, peers=None, me=None, clock=time.monotonic, n_sent=0, broadcast=None, **kw):
        if queue or peers or me:
            raise NotImplementedError("--native: a KissLink starts with no queue, peers or me")
        return K.KissLink(cap, clock=None if clock is time.monotonic else clock, n_sent=n_sent, broadcast=broadcast,
                          **kw)

    return {
        (PHY, "mask_value"): P.mask_value,
        (PHY, "tx_audio"): tx_audio,
        (PHY, "soft_bits"): soft_bits,
        (PHY, "measure"): measure,
        (PHY, "ModemRx"): ModemRx,
        (kisslink, "parse_ax25"): parse_ax25,
        (kisslink, "station_hash"): K.station_hash,
        (kisslink, "pack_frames"): lambda frames, pb: K.pack_frames([bytes(f) for f in frames], int(pb)),
        (kisslink, "unpack_frames"): lambda payloads, ok: tuple(K.unpack_frames([bytes(p) for p in payloads],
                                                                                [bool(o) for o in ok])),
        (kisslink, "KissLink"): KissLink,
    }
