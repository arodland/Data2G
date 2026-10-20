"""`pytest --native` substitutions for data2g.codes' primitives (CRC, scrambler,
frozen formats, the polar code and its decoder), data2g.constellation,
data2g.equalizer and data2g.ldpc. Applied first. See __init__.py."""

# Every data2g.equalizer function but _support_basis (its cache is internal
# to the C++ one).
EQUALIZER_NATIVE = ("residual_cfo", "delay_profile", "delay_support", "window_shift", "freq_smooth",
                    "support_slack", "preamble_noise", "preamble_noise_k", "per_carrier_noise", "_doppler_corr", "measure_spread",
                    "estimate", "time_shift_phase", "refine")


def substitutions(native):
    """(module, attribute) -> the native replacement. Specs are passed to C++
    by name, so a spec that isn't exactly the frozen one stays in Python."""
    import functools

    import numpy as np

    from data2g import codes, config, constellation, equalizer, ldpc

    py_frozen = codes.frozen
    nc = native.constellation
    py = {f: getattr(constellation, f) for f in ("load", "ace_dirs", "modulate", "llr", "ace_project")}
    names = set(nc.names())
    by_bytes = {nc.points(n).tobytes(): n for n in names}

    def frozen_array(a):
        a.setflags(write=False)
        return a

    @functools.lru_cache(maxsize=None)
    def load(name):
        return frozen_array(nc.points(name)) if name in names else py["load"](name)

    @functools.lru_cache(maxsize=None)
    def ace_dirs(name):
        return frozen_array(nc.ace_dirs(name)) if name in names else py["ace_dirs"](name)

    # Points cross to C++ by name: an array equal to a frozen set bit for bit
    # is that set (modem.QPSK is gray-qam4); any other stays in Python.
    def name_of(points):
        return by_bytes.get(points.tobytes()) if isinstance(points, np.ndarray) and points.dtype == complex else None

    def modulate(bits, points):
        name = name_of(points)
        return py["modulate"](bits, points) if name is None else nc.modulate(np.asarray(bits).reshape(-1), name)

    def llr(y, h, var, points):
        name = name_of(points)
        return py["llr"](y, h, var, points) if name is None else nc.llr(*np.broadcast_arrays(y, h, var), name)

    def ace_project(got, want, dirs):  # torch (channel_torch) stays in Python
        if not all(isinstance(a, np.ndarray) for a in (got, want, dirs)):
            return py["ace_project"](got, want, dirs)
        got, want = np.broadcast_arrays(got, want)
        return nc.ace_project(got, want, np.broadcast_to(dirs, got.shape + (2,))).reshape(got.shape)

    def frozen(spec):
        if config.SUBMODES.get(spec.name) != spec:
            return py_frozen(spec)
        d = {"fingerprint": codes.fingerprint(spec), "perm": native.codes.interleaver(spec.name)}
        if spec.code == "polar":
            d["info_pos"] = native.codes.info_pos(spec.name)
        return d

    py_polar_code, py_decoder = codes.polar_code, codes._decoder

    @functools.lru_cache(maxsize=None)
    def polar_code(spec):
        if config.SUBMODES.get(spec.name) == spec:
            return native.polar.polar_code(spec.name)
        py = py_polar_code(spec)  # e.g. the CPM control codeword: GA-designed
        if py.frozen_override is None and py.design_snr_db == codes.POLAR_DESIGN_SNR_DB:
            try:
                return native.polar.PolarCode(spec.k, spec.coded_bits)
            except IndexError:  # no frozen GA design for this (k, e)
                pass
        return py

    @functools.lru_cache(maxsize=None)
    def decoder(spec, device=None):
        code = codes.polar_code(spec) if spec.code == "polar" and device is None else None
        if isinstance(code, native.polar.PolarCode):
            return native.polar.SCLDecoder(code, codes.POLAR_LIST)
        return py_decoder(spec, device)

    return {
        (codes, "polar_code"): polar_code,
        (codes, "_decoder"): decoder,
        (codes, "crc24"): native.codes.crc24,
        (codes, "_with_crc"): native.codes.with_crc,
        (codes, "scramble_seed"): native.codes.scramble_seed,
        (codes, "scrambler"): native.codes.scrambler,
        (codes, "frozen"): frozen,
        (constellation, "load"): load,
        (constellation, "ace_dirs"): ace_dirs,
        (constellation, "modulate"): modulate,
        (constellation, "llr"): llr,
        (constellation, "ace_project"): ace_project,
        **{(equalizer, name): getattr(native.equalizer, name) for name in EQUALIZER_NATIVE},
        # every LDPC code (codes.ldpc_code, mother(), IR extents) and decoder
        # (codes._decoder, _ext_decoder, decode_posterior) is then C++
        (ldpc, "qc_code"): native.ldpc.qc_code,
        (ldpc, "Decoder"): native.ldpc.Decoder,
        (ldpc, "MinSumDecoder"): native.ldpc.Decoder,  # the old name, for scripts/
    }
