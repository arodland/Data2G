"""Run a study script with the C++ core substituted, as `pytest --native` does.

    python tools/with_native.py [--python] [--threads N] [--dd-budget S] scripts/loss_study.py --out runs/x.csv ...

--python applies nothing (the same wrapper, so both sides of a paired run
take the same path). --dd-budget overrides arq.phy.DD_BUDGET_S on both
sides: the budget is wall-clock, so a faster decoder finishes more DD
passes; `inf` makes paired runs comparable. Pool workers inherit the
substitutions through fork. --threads N also sizes the C++ pool (DATA2G_THREADS,
default 1 here, so forked workers do not oversubscribe).
"""

import argparse
import os
import runpy
import sys
from pathlib import Path

# The caps must be in place before numpy loads (data2g/threads.py), and this
# wrapper loads it before the script does. --threads N overrides 1.
if "--threads" in sys.argv:
    i = sys.argv.index("--threads")
    _n = sys.argv.pop(i + 1)
    sys.argv.pop(i)
else:
    _n = "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "DATA2G_THREADS"):
    os.environ.setdefault(_v, _n)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", action="store_true", help="substitute nothing")
    ap.add_argument("--dd-budget", type=float, default=None)
    ap.add_argument("--skip", default="", help="conftest providers to leave in Python, e.g. arq,gear "
                    "(studies that reach into Session/GearShifter internals)")
    ap.add_argument("script")
    a, rest = ap.parse_known_args()

    if a.dd_budget is not None:
        from data2g.arq import phy

        phy.DD_BUDGET_S = a.dd_budget
    if not a.python:
        import conftest

        native = conftest.import_native()
        if native is None:
            sys.exit(f"data2g_native: {conftest._import_error} (tools/build_native.sh builds it)")
        subs = conftest._substitutions(native)
        skip = {f"_{s}_substitutions" for s in a.skip.split(",") if s}
        unknown = skip - {p.__name__ for p in conftest._PROVIDERS}
        if unknown:
            sys.exit(f"--skip: no provider {', '.join(sorted(unknown))}")
        for p in conftest._PROVIDERS:
            if p.__name__ not in skip:
                subs.update(p(native))
        for (module, attr), fn in subs.items():
            setattr(module, attr, fn)
        print(f"with_native: {len(subs)} substitutions", file=sys.stderr)
    sys.argv = [a.script, *rest]
    sys.path.insert(0, str(Path(a.script).resolve().parent))
    runpy.run_path(a.script, run_name="__main__")


if __name__ == "__main__":
    main()
