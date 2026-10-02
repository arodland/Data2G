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
    from data2g import codes, config

    py_frozen = codes.frozen

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
        **_waveform_substitutions(native),
    }


def _waveform_substitutions(native):
    """data2g.waveform. Bands cross by name; a band whose spec isn't the
    configured one stays in Python. Modules that from-imported a name hold
    their own reference, so modem is listed too."""
    import functools

    import numpy as np

    from data2g import config, modem
    from data2g.waveform import dsp, ofdm, sync

    W = native.waveform
    py_band, py_detection_stat, py_raw_stat = ofdm.band, sync.detection_stat, sync._raw_stat
    py_refine, py_acquire = sync._refine, sync.acquire

    def name_of(band):
        if band is None:
            return "w"
        return band.spec.name if config.BANDS.get(band.spec.name) == band.spec else None

    class NativeBand(ofdm.Band):
        def modulate_symbols(self, symbols):
            return self._n.modulate_symbols(np.atleast_2d(symbols))

        def demod_window(self, z, start, backoff=0):
            return self._n.demod_window(z, int(start), int(backoff))

        def preamble_waveform(self):
            return self._n.preamble_waveform()

        def preamble_template(self):
            return self._template

    @functools.lru_cache(maxsize=None)
    def band(name="w"):
        if config.BANDS.get(name) is None:
            return py_band(name)
        n = W.band(name)
        b = NativeBand(spec=config.BANDS[name], freqs=n.freqs, bb=n.bb, mod=n.mod, demod=n.demod, pilot=n.pilot)
        t = n.preamble_template
        t.flags.writeable = False
        object.__setattr__(b, "_n", n)
        object.__setattr__(b, "_template", t)
        return b

    w = band("w")

    def detection_stat(z, band=None, reach=config.ACQUIRE_REACH_HZ, repeats=None):
        if (n := name_of(band)) is None:
            return py_detection_stat(z, band, reach, repeats)
        return W.detection_stat(z, n, reach, repeats or 0)

    def _raw_stat(z, band=None, reach=config.ACQUIRE_REACH_HZ, repeats=None, levels_from=None, outs=None):
        if (n := name_of(band)) is None:
            return py_raw_stat(z, band, reach, repeats, levels_from, outs)
        S, q, freqs, c = W.raw_stat(z, n, reach, repeats or 0, levels_from, outs is not None)
        if outs is not None:
            outs.extend(c)
        return S, q, freqs

    def _refine(z, band, n, f):
        if (name := name_of(band)) is None:
            return py_refine(z, band, n, f)
        return W.refine(z, name, int(n), float(f))

    def acquire(z, threshold=None, reach=config.ACQUIRE_REACH_HZ, search=None, band=None, S=None):
        if (n := name_of(band)) is None:
            return py_acquire(z, threshold, reach, search, band, S)
        try:
            start, f, metric, alts = W.acquire(z, n, threshold, reach,
                                               None if search is None else (int(search[0]), int(search[1])), S)
        except W.SyncError as e:
            raise sync.SyncError(str(e)) from None
        return sync.Acquisition(preamble_start=start, freq_offset=f, metric=metric, alternatives=alts)

    class StreamDetector(W.StreamDetector):
        def __init__(self, band, reach=config.ACQUIRE_REACH_HZ):
            assert name_of(band), "StreamDetector: native bands only"
            super().__init__(band.spec.name, reach)
            self.band = band

    def to_baseband(x, n0=0):
        return W.to_baseband(x, int(n0))

    def freq_correct(z, f_hz):
        return W.freq_correct(z, float(f_hz))

    def tx_condition(x, clip_headroom_db, overshoot=config.CLIP_OVERSHOOT, active=slice(None),
                     bandpass=config.TX_BANDPASS, project=None, closing=()):
        lo, hi, step = active.indices(len(x))
        assert step == 1
        return W.tx_condition(x, float(clip_headroom_db), list(overshoot), lo, hi, tuple(map(float, bandpass)),
                              project, list(closing))

    return {
        (ofdm, "band"): band,
        (ofdm, "_W"): w,
        (ofdm, "MOD_MATRIX"): w.mod,
        (ofdm, "DEMOD_MATRIX"): w.demod,
        (ofdm, "modulate_symbols"): w.modulate_symbols,
        (ofdm, "demod_window"): w.demod_window,
        (ofdm, "preamble_waveform"): w.preamble_waveform,
        (ofdm, "preamble_template"): w.preamble_template,
        (ofdm, "pilot_sequence"): lambda: w.pilot,
        (modem, "PILOT"): w.pilot,
        (dsp, "to_baseband"): to_baseband,
        (dsp, "freq_correct"): freq_correct,
        (dsp, "tx_condition"): tx_condition,
        (dsp, "papr_db"): W.papr_db,
        (modem, "to_baseband"): to_baseband,
        (modem, "freq_correct"): freq_correct,
        (modem, "tx_condition"): tx_condition,
        (modem, "acquire"): acquire,
        (sync, "detection_stat"): detection_stat,
        (sync, "_raw_stat"): _raw_stat,
        (sync, "_repeat_corr"): W.repeat_corr,
        (sync, "_repeat_corrs"): lambda z, t, freqs: W.repeat_corrs(z, t, list(freqs)),
        (sync, "first_path"): lambda power, peak, search=config.FIRST_PATH_SEARCH, frac=config.FIRST_PATH_FRAC,
        cyclic=False: W.first_path(power, int(peak), search, frac, cyclic),
        (sync, "_refine"): _refine,
        (sync, "_crossings"): lambda D, threshold, span, limit: W.crossings(D, threshold, span, limit),
        (sync, "acquire"): acquire,
        (sync, "StreamDetector"): StreamDetector,
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
