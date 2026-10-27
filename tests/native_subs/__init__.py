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

from . import arq, audio, base, codes, cpm, engine, gear, host, modem, phy, receiver, tnc, waveform

MODULE_DIR = Path(__file__).resolve().parents[2] / "native" / "build" / "python"
# native/bindings/module.cpp's __abi__: bumped with the Python-facing signatures
ABI = 4

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
    return data2g_native, None


def build(native, skip=()):
    unknown = set(skip) - PROVIDERS.keys()
    if unknown:
        raise KeyError(f"no provider {', '.join(sorted(unknown))}")
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
