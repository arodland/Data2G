"""data2g.threads: caps set before numpy loads, or a loud failure after."""

import os
import subprocess
import sys

from data2g import threads


def run(code, **env):
    base = {k: v for k, v in os.environ.items() if k not in threads.VARS}
    return subprocess.run([sys.executable, "-c", code], env=base | env, capture_output=True, text=True)


def test_sets_caps_before_numpy():
    r = run("from data2g import threads; threads.limit(2); import numpy, os; print(os.environ['OPENBLAS_NUM_THREADS'])")
    assert r.returncode == 0 and r.stdout.strip() == "2"


def test_numpy_first_without_caps_raises():
    r = run("import numpy; from data2g import threads; threads.limit(1)")
    assert r.returncode != 0 and "imported before the thread caps" in r.stderr


def test_numpy_first_with_exported_caps_is_fine():
    r = run("import numpy; from data2g import threads; threads.limit(1)", **{v: "1" for v in threads.VARS})
    assert r.returncode == 0, r.stderr
