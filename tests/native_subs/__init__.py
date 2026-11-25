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


# The study toggles (Python only; native has master's defaults), merged from the three branches. On
# fast-fading-integration most are ON by default, so this reads what is in effect (the modules' settings), not
# the environment:
# - c8w50 on air, a connect sequence with W, every wide mode (DATA2G_CPM_ONAIR, DATA2G_CONNECT_SEQ,
#   DATA2G_CPM_WIDE, DATA2G_ROBUST_*), or fsk32r62-r2/3 offered (DATA2G_CPM_R23): a grid, modes, header words,
#   a recommendation code and a retry ladder the C++ tables and session do not have: everything but the OFDM
#   and audio providers stays Python (the shifter too).
# - the leak metric or the spill weights (DATA2G_CPM_LEAK, _LEAK_K, _LEAK_Z, _CTL_SQ, _SPILL): CPM soft values
#   C++ makes with master's metric: cpm, the receiver, phy's ModemRx, the engine and host, and arq (the C++
#   Station and Session).
# - scattered pilots (DATA2G_PILOTS): every provider that builds, receives or decodes an OFDM frame, plus burst
#   timing and the shifter (a flagged burst is 24 ms longer).
# (Each provider also routes a wide grid, L > 1, to Python by itself: the toolbox grids without a toggle.)
WIDE_NATIVE = {"base", "waveform", "modem", "audio"}
CPM_SOFT_PY = {"cpm", "receiver", "phy", "engine", "host", "arq"}
PILOTS_NATIVE = {"base", "waveform", "cpm", "codes", "audio"}


def study_skip() -> set:
    """Providers the study toggles in effect keep in Python."""
    import os

    from data2g import config, cpm

    skip = set()
    if (cpm.ON_AIR_WIDE or cpm.WIDE_ON or cpm.R23
            or any(os.environ.get(v, "") not in ("", "0", "off")
                   for v in ("DATA2G_CONNECT_SEQ", "DATA2G_ROBUST_RULE", "DATA2G_ROBUST_FAST"))):
        skip |= set(PROVIDERS) - WIDE_NATIVE
    if cpm.LEAK or any(v in os.environ for v in ("DATA2G_CPM_LEAK_K", "DATA2G_CPM_LEAK_Z", "DATA2G_CPM_CTL_SQ",
                                                    "DATA2G_CPM_SPILL")):
        skip |= CPM_SOFT_PY
    if getattr(config, "SC_BANDS", ()):
        skip |= set(PROVIDERS) - PILOTS_NATIVE
    return skip


def build(native, skip=()):
    unknown = set(skip) - PROVIDERS.keys()
    if unknown:
        raise KeyError(f"no provider {', '.join(sorted(unknown))}")
    skip = set(skip) | study_skip()
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
