"""FM plan M4/Q8: how short can the cyclic prefix be? FM audio has no
multipath; the prefix covers the radios' filter group delay and the
receiver's timing error. Today 32 samples at 16 kHz (2 ms), symbol 20 ms.
Variants (DATA2G_NCP, samples; DATA2G_FCENTER 2000 but cp32): cp32 (today), cp32f (today at 2000 Hz), cp24 (1.5 ms, +2.3% rate),
cp16 (1 ms, +4.5%), cp8 (0.5 ms, +6.8%). w48 QPSK r1/2, 16-QAM r1/2,
64-point r3/4; mic and flat radios, every condition, streamed receiver,
FM profile defaults, 30 paired trials. Q19 found a 1 ms prefix failed
outright at 100 Hz spacing (acquisition timing is good to +-1 ms).

    python scripts/fm_cp_study.py run [--trials 30] [--workers 8]
    python scripts/fm_cp_study.py report
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_spacing_study as S  # noqa: E402

OUT = S.ROOT / "runs" / "fm_cp"
_BASE = {"FM_RX_STREAM": "1"}
# the mixer at 2000 Hz: whole cycles per symbol for every prefix here (config.FCENTER's
# assertion; 1500 Hz fits only 32); cp32 at 1500 is today, cp32f the same at 2000
VARIANTS = {"cp32": {**_BASE}, **{f"cp{n}" + ("f" if n == 32 else ""): {**_BASE, "DATA2G_NCP": str(n), "DATA2G_FCENTER": "2000"}
                                  for n in (32, 24, 16, 8)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "report", "variant"])
    ap.add_argument("--variant")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    if a.what == "variant":
        return S.run_variant(OUT, a.variant, a.trials, a.workers)
    if a.what == "run":
        for v in VARIANTS:  # one process each: the harness reads its switches at import
            subprocess.run([sys.executable, __file__, "variant", "--variant", v, "--trials", str(a.trials),
                            "--workers", str(a.workers)], env={**os.environ, **VARIANTS[v], "DATA2G_PROFILE": "fm"},
                           check=True)
    S.report_db(OUT, list(VARIANTS))


if __name__ == "__main__":
    main()
