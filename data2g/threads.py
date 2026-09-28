"""BLAS/OpenMP thread caps for study scripts (the machine is shared).

numpy's OpenBLAS and torch size their thread pools when they are imported,
so the environment must be set before that. A wrapper that imported
data2g's numpy users first (`python -c "import ...; loss_study.main()"`)
made a loss study's 8 workers run 25 threads each, 50x slower. Here that
fails loudly instead."""

import os
import sys

VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def limit(n: int = 1):
    """Cap threads at n per process, before numpy or torch loads. Already
    loaded: fine only if the caller exported the caps; otherwise raise."""
    loaded = [m for m in ("numpy", "torch") if m in sys.modules]
    if not loaded:
        for v in VARS:
            os.environ.setdefault(v, str(n))
        return
    unset = [v for v in VARS if v not in os.environ]
    if unset:
        raise RuntimeError(f"{', '.join(loaded)} was imported before the thread caps were set, so each worker "
                           f"would use a thread per core. Export them first: "
                           + " ".join(f"{v}={n}" for v in VARS))
