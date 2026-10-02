"""`pytest --native` runs this suite with C++ functions (native/bindings)
substituted into the reference modules, so the whole suite is the native
port's acceptance test (docs/native-port-plan.md). Build the module with
tools/build_native.sh. A skip is not a pass: with --native, a missing or
stale module is an error.
"""

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


def _substitutions(native):
    """(module, attribute) -> the native replacement. Specs are passed to C++
    by name, so a spec that isn't exactly the frozen one stays in Python."""
    import functools

    import numpy as np

    from data2g import codes, config, constellation

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

    return {
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
