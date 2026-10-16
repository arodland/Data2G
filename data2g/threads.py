"""BLAS/OpenMP thread caps, set before numpy or torch loads.

numpy's OpenBLAS and torch size their thread pools when they are imported,
so the environment must be set before that. Study scripts call limit() (the
machine is shared): a wrapper that imported data2g's numpy users first
(`python -c "import ...; loss_study.main()"`) made a loss study's 8 workers
run 25 threads each, 50x slower, and limit() fails loudly instead.
data2g-host and the receiver (data2g.receiver has why) call cap_env(1) as
they are imported."""

import os
import sys

VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def cap_env(n: int = 1):
    """Cap threads at n per process unless the environment already says:
    no effect on a numpy or torch already loaded."""
    for v in VARS:
        os.environ.setdefault(v, str(n))


def limit(n: int = 1):
    """Cap threads at n per process, before numpy or torch loads. Already
    loaded: fine only if the caller exported the caps; otherwise raise."""
    loaded = [m for m in ("numpy", "torch") if m in sys.modules]
    if not loaded:
        cap_env(n)
        return
    unset = [v for v in VARS if v not in os.environ]
    if unset:
        raise RuntimeError(f"{', '.join(loaded)} was imported before the thread caps were set, so each worker "
                           f"would use a thread per core. Export them first: "
                           + " ".join(f"{v}={n}" for v in VARS))
