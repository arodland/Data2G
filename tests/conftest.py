"""The suite tests data2g/, the normative Python reference. Layout:

- test_<module>.py: a data2g module's own tests (test_kiss is
  kisslink's; test_dd, test_preamble and test_receive_head are features of
  arq.phy, sync and modem).
- test_native_<area>.py: the C++ port (native/) against the Python, side
  by side; each skips if data2g_native isn't built. native/README.md maps
  every area to its core directory, binding and ctest.
- test_engine, test_host, test_native_host_e2e, test_speedtest: whole
  stations, back to back or over named pipes.

`pytest --native` runs the whole suite again with C++ substituted into the
reference modules (tests/native_subs/, the registry), so every Python test
is also the port's acceptance test. Build the module with
tools/build_native.sh. A skip is not a pass: with --native, a missing or
stale module is an error.

Fixtures: `native` (the data2g_native module, with or without --native)
and `reference` (reference(module, "name"): the Python one, substituted or
not). Tests that need every substitution undone monkeypatch
native_subs.originals back.

Parallel: pyproject's addopts run the suite on pytest-xdist workers
(`-n auto --dist loadgroup`), min(8, cores / 2) of them (the machine is
shared; DATA2G_TEST_WORKERS sets the count, `-n 0` runs in one process),
one BLAS thread each. LONG_FIRST's tests are handed out first, each to
its own worker, so the longest don't start last.
"""

import os

# before anything imports numpy: one BLAS thread per worker, as CI runs
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# and on xdist workers, the C++ pool at 1 (results are the same for any
# size): 8 workers' default pools of 4 starved the real-time host test,
# 71 s against its 56 s alone. Inherited by the hosts it starts, unchanged.
if os.environ.get("PYTEST_XDIST_WORKER"):
    os.environ.setdefault("DATA2G_THREADS", "1")

import native_subs  # noqa: E402
import pytest  # noqa: E402

# Tests that take tens of seconds each (real time over named pipes, 70 s):
# first in the collection, so loadgroup's per-test scheduling starts each
# on its own worker at once.
LONG_FIRST = ("tests/test_native_host_e2e.py::",)


def pytest_xdist_auto_num_workers(config):
    n = os.environ.get("DATA2G_TEST_WORKERS")
    return int(n) if n else min(8, max(1, (os.cpu_count() or 2) // 2))


def pytest_collection_modifyitems(config, items):
    items.sort(key=lambda it: not it.nodeid.startswith(LONG_FIRST))  # stable: the rest keep their order


def pytest_addoption(parser):
    parser.addoption("--native", action="store_true", default=False,
                     help="run the suite against the C++ core (tools/build_native.sh builds it)")


def pytest_configure(config):
    if not config.getoption("--native"):
        return
    native, why = native_subs.import_native()
    if native is None:
        raise pytest.UsageError(f"--native: cannot import data2g_native: {why}")
    subs = native_subs.build(native)
    native_subs.apply(subs)
    config._native_count = len(subs)


def pytest_report_header(config):
    if config.getoption("--native"):
        return f"native: {config._native_count} substitutions from {native_subs.MODULE_DIR}"


@pytest.fixture(scope="session")
def native():
    """The module for direct parity tests, with or without --native."""
    module, why = native_subs.import_native()
    if module is None:
        pytest.skip(f"data2g_native not built: {why}")
    return module


@pytest.fixture(scope="session")
def reference():
    """reference(module, "name"): the Python function, substituted or not."""
    return native_subs.reference


def _clear_caches():
    """Every functools cache in data2g's loaded modules: they hold objects of
    whichever implementation built them (a Python QCLDPC handed to the C++
    decoder fails)."""
    import sys

    for name, mod in list(sys.modules.items()):
        if name == "data2g" or name.startswith("data2g."):
            for v in list(vars(mod).values()):
                if callable(getattr(v, "cache_clear", None)):
                    v.cache_clear()


@pytest.fixture
def pure(monkeypatch):
    """Every --native substitution undone for the test: the reference is all
    Python (its callees included), the native side called directly. The
    caches are cleared on the way in and out, so neither implementation's
    objects outlive the swap (tests run in any order on xdist workers)."""
    _clear_caches()
    for (module, attr), fn in native_subs.originals.items():
        monkeypatch.setattr(module, attr, fn)
    yield
    monkeypatch.undo()
    _clear_caches()


@pytest.fixture
def protocol(monkeypatch):
    """protocol(v): audio recorded under another PROTOCOL_VERSION (its
    headers' CRC-6 seed), with the header tables built from it cleared,
    --native's and the Python ones behind them; put back after the test."""
    from data2g import cpm, modem

    names = ((modem, "_valid_words"), (modem, "_valid_signs"), (cpm, "valid_words"), (cpm, "header_rows"),
             (cpm, "mid_rows"))
    caches = {f for m, n in names for f in (getattr(m, n), native_subs.reference(m, n)) if hasattr(f, "cache_clear")}

    def use(v):
        monkeypatch.setattr(modem, "PROTOCOL_VERSION", v)
        for c in caches:
            c.cache_clear()

    yield use
    monkeypatch.undo()
    for c in caches:
        c.cache_clear()
