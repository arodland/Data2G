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
"""

import native_subs
import pytest


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
