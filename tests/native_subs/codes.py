"""`pytest --native` substitutions for data2g.codes' codec: framing, encode,
IR combining and the decode entry points. See __init__.py."""


def substitutions(native):
    """data2g.codes' codec (encode, combine, decode) in C++. Specs go by
    name: a frozen submode, a CPM mode or a CPM control codeword; any other
    spec, torch (decode_llrs' device) and dtypes the binding lacks stay in
    Python. codes.decode routes through decode_many; _decode_code_order
    takes a decoder object and is C++-internal."""
    import functools

    import numpy as np

    from data2g import codes, config, cpm

    n = native.codes
    py = {k: getattr(codes, k) for k in (
        "crc_bits", "payload_bytes", "rv_cycle", "buffer_len", "rv_positions", "info_bits", "encode", "encode_info",
        "flip", "spread", "despread", "combine", "decode_buffer", "payloads", "decode_many", "decode_llrs",
        "decode_raw", "descramble", "check", "crc_ok", "interleaver", "ir_extent", "decode_posterior",
        "systematic_positions", "zero_tail", "pin", "pin_buffer", "polar_order", "decode_raw_pinned",
        "decode_buffer_pinned")}

    @functools.lru_cache(maxsize=None)
    def own(spec):
        return (config.SUBMODES.get(spec.name) == spec or cpm.SPECS.get(spec.name) == spec
                or cpm.CTL.get(getattr(spec, "grid", None)) == spec)

    def by_name(attr):
        f, p = getattr(n, attr), py[attr]
        return lambda spec, *a, **k: f(spec.name, *a, **k) if own(spec) else p(spec, *a, **k)

    @functools.lru_cache(maxsize=None)
    def flip(spec, index, rv=0):
        out = n.flip(spec.name, index, rv) if own(spec) else py["flip"](spec, index, rv)
        out.setflags(write=False)
        return out

    @functools.lru_cache(maxsize=None)
    def interleaver(spec):
        if not own(spec):
            return py["interleaver"](spec)
        out = n.interleaver(spec.name)
        out.setflags(write=False)
        return out

    def native_array(x, ndim):
        return isinstance(x, np.ndarray) and x.ndim >= ndim and x.dtype in (np.uint8, np.float64)

    def spread(coded, m):
        if not native_array(coded, 2):
            return py["spread"](coded, m)
        *lead, n_cw, N = coded.shape
        return n.spread(coded.reshape(-1, n_cw * N), n_cw, m).reshape(*lead, n_cw * N)

    def despread(x, n_cw, m):
        if not native_array(x, 1):
            return py["despread"](x, n_cw, m)
        *lead, total = x.shape
        return n.despread(x.reshape(-1, total), n_cw, m).reshape(*lead, n_cw, total // n_cw)

    def info_bits(spec, payload, crc_mask=0, index=0):
        if not own(spec):
            return py["info_bits"](spec, payload, crc_mask, index)
        return n.info_bits(spec.name, bytes(payload), crc_mask, index)

    def encode(spec, payload, rv=0, crc_mask=0, index=0):
        if not own(spec):
            return py["encode"](spec, payload, rv, crc_mask, index)
        return n.encode(spec.name, bytes(payload), rv, crc_mask, index)

    def combine(spec, buf, soft, rvs):
        if not own(spec):
            return py["combine"](spec, buf, soft, rvs)
        return n.combine(spec.name, buf, np.atleast_2d(soft), rvs)

    def payloads(spec, bits, converged, masks=None, index=None):
        if not own(spec):
            return py["payloads"](spec, bits, converged, masks, index)
        return n.payloads(spec.name, np.asarray(bits), np.asarray(converged, np.uint8), masks, index)

    def decode_llrs(spec, llr, iters=codes.LDPC_ITERS, device=None, crc_mask=0, index=0):
        if device is not None or not own(spec):
            return py["decode_llrs"](spec, llr, iters, device, crc_mask, index)
        return n.decode_llrs(spec.name, llr, iters, crc_mask, index)

    def descramble(spec, bits, index):
        if not own(spec):
            return py["descramble"](spec, bits, index)
        return n.descramble(spec.name, np.asarray(bits), int(index))

    @functools.lru_cache(maxsize=None)
    def systematic_positions(spec):
        if not own(spec):
            return py["systematic_positions"](spec)
        out = n.systematic_positions(spec.name)
        out.setflags(write=False)
        return out

    def pin(spec, soft, known, index):
        if not own(spec):
            return py["pin"](spec, soft, known, index)
        return n.pin(spec.name, np.asarray(soft, np.float64), np.asarray(known, np.int64), int(index))

    def pin_buffer(spec, buf, known):
        if not own(spec):
            return py["pin_buffer"](spec, buf, known)
        return n.pin_buffer(spec.name, np.atleast_2d(np.asarray(buf, np.float64)), np.asarray(known, np.int64))

    @functools.lru_cache(maxsize=None)
    def polar_order(spec):
        if not own(spec):
            return py["polar_order"](spec)
        out = n.polar_order(spec.name)
        out.setflags(write=False)
        return out

    def decode_raw_pinned(spec, soft, known, index):
        if not own(spec):
            return py["decode_raw_pinned"](spec, soft, known, index)
        return n.decode_raw_pinned(spec.name, np.atleast_2d(np.asarray(soft, np.float32)), np.asarray(known, np.int64),
                                   int(index))

    def decode_buffer_pinned(spec, buf, max_rv, known, crc_mask=0):
        if not own(spec):
            return py["decode_buffer_pinned"](spec, buf, max_rv, known, crc_mask)
        return n.decode_buffer_pinned(spec.name, np.atleast_2d(np.asarray(buf, np.float64)), max_rv,
                                      np.asarray(known, np.int64), int(crc_mask))

    return {
        **{(codes, a): by_name(a) for a in ("crc_bits", "payload_bytes", "rv_cycle", "buffer_len", "rv_positions",
                                             "encode_info", "decode_buffer", "decode_many", "decode_raw", "check",
                                             "crc_ok", "ir_extent", "decode_posterior", "zero_tail")},
        (codes, "flip"): flip,
        (codes, "systematic_positions"): systematic_positions,
        (codes, "pin"): pin,
        (codes, "pin_buffer"): pin_buffer,
        (codes, "polar_order"): polar_order,
        (codes, "decode_raw_pinned"): decode_raw_pinned,
        (codes, "decode_buffer_pinned"): decode_buffer_pinned,
        (codes, "interleaver"): interleaver,
        (codes, "spread"): spread,
        (codes, "despread"): despread,
        (codes, "info_bits"): info_bits,
        (codes, "encode"): encode,
        (codes, "combine"): combine,
        (codes, "payloads"): payloads,
        (codes, "decode_llrs"): decode_llrs,
        (codes, "descramble"): descramble,
    }
