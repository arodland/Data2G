"""The `--native` substitution registry: which data2g functions and classes
the C++ core (native/bindings, built as data2g_native) stands in for, so the
Python test suite, or a study script, runs on the port.

One module per ported area, each with `substitutions(native) -> {(module,
attribute): replacement}`. A replacement takes the Python signature and
falls back to the Python original for anything C++ doesn't have (a spec
that isn't the frozen one, a patched constant, torch input): see each
area's docstring. Names are looked up when called, so substituting
`data2g.codes.encode` reaches every caller that writes `codes.encode`; a
module that from-imported a name is listed beside it.

Keys name private functions too (sync._raw_stat, codes._decoder,
modem._demod_frames, ...). Renaming one in data2g/ means renaming its key
here and, usually, its binding; plain `pytest` won't notice, only
`pytest --native` does.

Public API (tests/conftest.py and tools/with_native.py):

    import_native()       -> (data2g_native, None) or (None, why not)
    PROVIDERS             area name -> substitutions, in application order;
                          the names are tools/with_native.py's --skip names
    build(native, skip)   every provider's substitutions but skip's, merged
    apply(subs)           set them, recording the originals
    originals             (module, attribute) -> the Python one apply replaced
    reference(module, a)  the Python one, substituted or not

Every provider is called before any is applied, so each captures the Python
originals.
"""

import sys
from pathlib import Path

from data2g import config

from . import arq, audio, base, codes, cpm, engine, gear, host, modem, phy, receiver, tnc, waveform

# tools/build_native.sh: native/build for HF, native/build-<profile> for another
MODULE_DIR = (Path(__file__).resolve().parents[2] / "native"
              / ("build" if config.PROFILE == "hf" else f"build-{config.PROFILE}") / "python")
# native/bindings/module.cpp's __abi__: bumped with the Python-facing signatures
ABI = 9

PROVIDERS = {m.__name__.rpartition(".")[2]: m.substitutions
             for m in (base, waveform, cpm, codes, gear, audio, arq, modem, tnc, receiver, phy, engine, host)}

originals = {}


def import_native():
    if str(MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(MODULE_DIR))
    try:
        import data2g_native
    except ImportError as e:  # not built, or built for another interpreter
        return None, f"{e} (looked in {MODULE_DIR})"
    if getattr(data2g_native, "__abi__", None) != ABI:
        return None, f"{data2g_native.__file__} has ABI {getattr(data2g_native, '__abi__', '?')}, want {ABI}"
    if getattr(data2g_native, "__profile__", "hf") != config.PROFILE:
        return None, f"{data2g_native.__file__} is the {data2g_native.__profile__} build, want {config.PROFILE}"
    return data2g_native, None


# The study switches here (fast-fading-integration) are native too, read from the same variables with this
# branch's defaults: the wide grids and c8w50 on air (cpm::grids, DATA2G_CPM_ONAIR, the connect ladder), the leak
# model on every grid (cpm::study), fsk32r62-r2/3 (tables::cpm_r23) and DATA2G_PILOTS (native/core/modem/
# scattered.cpp). What native has not:
# - the scattered pilots' study-only switches: with any of them in the environment (and the flag on) every
#   provider that builds, receives or decodes an OFDM frame stays Python (the modem, the receiver, phy's ModemRx
#   and the engine, host and TNC that run them), and so do the burst timing and the shifter (gear, arq:
#   DATA2G_SC_HDR_PILOT=0 makes a flagged burst 24 ms shorter); codes, CPM and audio stay native. (Native refuses
#   them: config::sc_layout throws.)
# - r2/3's zero-shot outcome model (predictor.R23_MODEL: DATA2G_OUTCOME_R23, on with r2/3 here): C++ compiles the
#   installed model, so the predictor and shifter stay Python (gear's own guard), and so do the C++ Station and
#   Session that run a shifter of their own (arq), and the engine and host around them.
PILOTS_ENV = "DATA2G_PILOTS"
SC_STUDY_ENV = ("DATA2G_SC_BOOST_DB", "DATA2G_SC_RESTORE_PLAIN", "DATA2G_SC_RESTORE_QPSK", "DATA2G_SC_SHORT_FRAMES",
                "DATA2G_SC_SPREAD_ALL", "DATA2G_SC_COMB", "DATA2G_SC_SUPPORT_PAD", "DATA2G_SC_SPREAD_X",
                "DATA2G_SC_HDR_PILOT", "DATA2G_SC_MODES", "DATA2G_SC_SEP_HZ", "DATA2G_SC_SEP_SEARCH",
                "DATA2G_SC_SEARCH_MIN", "DATA2G_SC_SLEPIAN")
PILOTS_NATIVE = {"base", "waveform", "cpm", "codes", "audio"}
R23_MODEL_PY = {"gear", "arq", "engine", "host"}


def study_skip() -> set:
    """Providers the study toggles in effect keep in Python."""
    import os

    from data2g.arq import predictor

    skip = set()
    on = bool(config.SC_BANDS)
    if on and any(v in os.environ for v in SC_STUDY_ENV):
        skip |= set(PROVIDERS) - PILOTS_NATIVE
    if predictor.R23_MODEL:
        skip |= R23_MODEL_PY
    return skip


# modem's scattered-pilot globals (and the constants its estimate reads): native has them as the process started.
# A test or study that patches one (test_scattered's layouts, a study's SC_STRIDE or SC_PILOT_FIX) keeps that call
# in Python; compared by identity, so a patch with an equal value falls back too.
SC_NAMES = ("SC_BANDS", "SC_LAYOUT", "SC_STRIDE", "SC_STRIDES", "SC_PILOT_FIX", "SC_SLACK", "SC_HDR_PILOT",
            "SC_BOOST", "SC_RESTORE_ACE", "SC_RESTORE_PLAIN", "SC_RESTORE_QPSK", "SC_MODES", "SC_SHORT_FRAMES",
            "SC_SPREAD_ALL", "SC_SPREADS_SLOW", "SC_SPREAD_LOO", "SC_LOO_TOL", "SC_COPY_COHERENT", "SC_COPY_SUBBINS",
            "SC_COPY_ROWS", "SC_COMB_SPREAD", "SC_FAST_HZ", "SC_SPREADS", "SC_SLOW_RHO", "SC_WIDEN", "SC_FRAMES",
            "SC_SEP_HZ", "SC_SEP_REPORT_HZ", "SC_JOINT_REACH2_HZ", "SC_SEP_SEARCH", "SC_SEARCH_MIN", "SC_HOT_PASSES",
            "SC_HOT_EXCESS", "SC_HOT_POWER", "SC_SLEPIAN")
_sc0 = {}


def sc_same() -> bool:
    """Whether modem's scattered-pilot globals are what native has (see SC_NAMES)."""
    from data2g import modem

    if not _sc0:
        _sc0.update({n: getattr(modem, n) for n in SC_NAMES})
    return all(getattr(modem, n) is v for n, v in _sc0.items())


def build(native, skip=()):
    unknown = set(skip) - PROVIDERS.keys()
    if unknown:
        raise KeyError(f"no provider {', '.join(sorted(unknown))}")
    skip = set(skip) | study_skip()
    sc_same()  # the unpatched globals, before any test runs
    subs = {}
    for name, p in PROVIDERS.items():
        if name not in skip:
            subs.update(p(native))
    return subs


def apply(subs):
    for (module, attr), fn in subs.items():
        originals.setdefault((module, attr), getattr(module, attr))
        setattr(module, attr, fn)


def reference(module, attr):
    return originals.get((module, attr), getattr(module, attr))
