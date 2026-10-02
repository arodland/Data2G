"""`pytest --native` runs this suite with C++ functions (native/bindings)
substituted into the reference modules, so the whole suite is the native
port's acceptance test (docs/native-port-plan.md). Build the module with
tools/build_native.sh. A skip is not a pass: with --native, a missing or
stale module is an error.
"""

import functools
import sys
from pathlib import Path

import pytest

NATIVE_MODULE_DIR = Path(__file__).resolve().parent.parent / "native" / "build" / "python"
NATIVE_ABI = 1
_import_error = None
_originals = {}  # (module, attribute) -> the Python function --native replaced


def import_native():
    global _import_error
    if str(NATIVE_MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(NATIVE_MODULE_DIR))
    try:
        import data2g_native
    except ImportError as e:  # not built, or built for another interpreter
        _import_error = f"{e} (looked in {NATIVE_MODULE_DIR})"
        return None
    if getattr(data2g_native, "__abi__", None) != NATIVE_ABI:
        _import_error = f"{data2g_native.__file__} has ABI {getattr(data2g_native, '__abi__', '?')}, want {NATIVE_ABI}"
        return None
    return data2g_native


# Every data2g.equalizer function but _support_basis (its cache is internal
# to the C++ one).
EQUALIZER_NATIVE = ("residual_cfo", "delay_profile", "delay_support", "window_shift", "_freq_smooth",
                    "preamble_noise", "preamble_noise_k", "per_carrier_noise", "_doppler_corr", "measure_spread",
                    "estimate", "time_shift_phase", "refine")


def _substitutions(native):
    """(module, attribute) -> the native replacement. Specs are passed to C++
    by name, so a spec that isn't exactly the frozen one stays in Python."""
    import functools

    import numpy as np

    from data2g import codes, config, constellation, equalizer

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
        d = {"fingerprint": codes._fingerprint(spec), "perm": native.codes.interleaver(spec.name)}
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
        **_cpm_substitutions(native.cpm),
    }


def _cpm_substitutions(n):
    """data2g.cpm. Grids and specs go by name; one that isn't the frozen one
    stays in Python. The TX bandpass (dsp) and the MI features (predictor)
    are still Python's, applied to the native results."""
    import functools

    import numpy as np

    from data2g import cpm
    from data2g.arq import predictor as P

    py = {k: getattr(cpm, k) for k in ("header_symbols", "layout", "burst_seconds", "modulate", "detect",
                                       "read_header", "soft", "find", "measure")}

    def own_grid(g):
        return cpm.GRIDS.get(g.name) == g

    def own_spec(s):
        return cpm.SPECS.get(s.name, cpm.CTL.get(s.grid)) == s

    def header_symbols(grid, word):
        return n.header_symbols(grid, word >> 6) if word < 1 << 16 else py["header_symbols"](grid, word)

    @functools.lru_cache(maxsize=None)
    def layout(grid, n_sym):
        d = n.layout(grid, n_sym)
        return cpm.Layout(d["n"], d["sync_rows"], d["sync_tones"], d["hdr_rows"], d["data_rows"], d["front"])

    def burst_seconds(spec, n_cw, dup=False):
        return n.burst_seconds(spec.name, n_cw, dup) if own_spec(spec) else py["burst_seconds"](spec, n_cw, dup)

    def modulate(spec, coded, dup):
        if not own_spec(spec):
            return py["modulate"](spec, coded, dup)
        return cpm.bandpass(cpm.GRIDS[spec.grid], n.modulate(spec.name, list(coded), dup))

    def detect(g, x, reach_hz=150.0, fine=True, front_only=False, n_sym=0, floor=-1.0):
        if not own_grid(g):
            return py["detect"](g, x, reach_hz, fine, front_only, n_sym, floor)
        return n.detect(g.name, x, reach_hz, fine, front_only, n_sym, floor)

    def read_header(g, x, s0, cfo, copies=cpm.HDR_COPIES):
        if not own_grid(g):
            return py["read_header"](g, x, s0, cfo, copies)
        name, nd, d, score, h2 = n.read_header(g.name, x, s0, cfo, copies)
        return cpm.SPECS[name], nd, d, score, h2

    def soft(g, spec, x, s0, cfo, n_data, dup):
        if not own_grid(g):
            return py["soft"](g, spec, x, s0, cfo, n_data, dup)
        slots, E = n.soft(g.name, x, s0, cfo, n_data, dup)
        return list(slots), E

    def find(g, x, threshold=None, reach_hz=150.0, front_only=True, lo=0, hi=None):
        if not own_grid(g):
            return py["find"](g, x, threshold, reach_hz, front_only, lo, hi)
        d = n.find(g.name, x, threshold, reach_hz, front_only, lo, hi)
        if d is None:
            return None
        d["spec"] = cpm.SPECS[d["spec"]]
        return {**d, "band": g.name, "family": "cpm"}

    def measure(g, E, n_sym):
        if not own_grid(g):
            return py["measure"](g, E, n_sym)
        m = n.measure(g.name, E, n_sym)
        snr = m.pop("snr")
        out = dict(snr_est=m["snr_est"], spread_est=m["spread_est"], delay_est_ms=0.0, headroom=0.0, frames=m["frames"])
        for c in P.CONSTS:
            out[f"mi_{c}"] = P.effective_mi(np.sqrt(snr), np.ones_like(snr), c)
        return out

    def by_grid(native_fn, py_fn):
        return lambda g, *a, **k: native_fn(g.name, *a, **k) if own_grid(g) else py_fn(g, *a, **k)

    return {
        (cpm, "header_symbols"): header_symbols,
        (cpm, "layout"): layout,
        (cpm, "stream_symbols"): n.stream_symbols,
        (cpm, "burst_seconds"): burst_seconds,
        (cpm, "to_tones"): by_grid(n.to_tones, cpm.to_tones),
        (cpm, "tones"): by_grid(n.tones, cpm.tones),
        (cpm, "modulate"): modulate,
        (cpm, "_energies"): by_grid(n.energies, cpm._energies),
        (cpm, "_shares"): n.shares,
        (cpm, "detect"): detect,
        (cpm, "llrs"): by_grid(n.llrs, cpm.llrs),
        (cpm, "read_header"): read_header,
        (cpm, "soft"): soft,
        (cpm, "_peak_ratio"): by_grid(n.peak_ratio, cpm._peak_ratio),
        (cpm, "find"): find,
        (cpm, "measure"): measure,
    }


def pytest_addoption(parser):
    parser.addoption("--native", action="store_true", default=False,
                     help="run the suite against the C++ core (tools/build_native.sh builds it)")


def pytest_configure(config):
    if not config.getoption("--native"):
        return
    native = import_native()
    if native is None:
        raise pytest.UsageError(f"--native: cannot import data2g_native: {_import_error}")
    subs = _substitutions(native)
    for (module, attr), fn in subs.items():
        _originals[module, attr] = getattr(module, attr)
        setattr(module, attr, fn)
    config._native_count = len(subs)


def pytest_report_header(config):
    if config.getoption("--native"):
        return f"native: {config._native_count} substitutions from {NATIVE_MODULE_DIR}"


@pytest.fixture(scope="session")
def native():
    """The module for direct parity tests, with or without --native."""
    module = import_native()
    if module is None:
        pytest.skip(f"data2g_native not built: {_import_error}")
    return module


@pytest.fixture(scope="session")
def reference():
    """reference(module, "name"): the Python function, substituted or not."""
    return lambda module, attr: _originals.get((module, attr), getattr(module, attr))
