"""FM plan: decoding while a burst arrives. What does a shorter interleaving
window cost? Today every codeword is dealt over the whole burst
(codes.spread), so none completes before the burst ends; a window of W
frames lets each group of codewords complete, and decode, W frames in.

    python scripts/fm_window_study.py run [--trials 20] [--workers 10] [--only V ...]
    python scripts/fm_window_study.py report

Through data2g.fmchannel on the FM profile (its defaults: +-12.5 Hz,
threshold 28.7, per-symbol noise), the harness of scripts/fm_spacing_study.py
(w48 QPSK r1/2, 16-QAM r1/2, 64-point r3/4; mic and flat; every condition,
146 and 440 MHz; paired seeds), with 4 s bursts (FM_BURST_FRAMES 28: 7
codewords of QPSK / 16-QAM, 14 of the 64-point). The receiver is
modem.demodulate: the window changes the data path, not receiver state.
Each variant against `all` in dB (fm_spacing_study.report_db).

Variants (DATA2G_SPREAD_WINDOW, frames; a frame is 144 ms):
    all   the whole burst (today)
    w16   16 frames, 2.3 s: QPSK / 16-QAM in groups of 4, the 64-point of 8
    w8    8 frames, 1.15 s: groups of 2 and 4
    w4    4 frames, 0.58 s: groups of 1 and 2
    e8    w8, and the channel estimated per 8 frames from their own pilots and
          one either side (DATA2G_EST_WINDOW): what a receiver that
          estimates and decodes as frames arrive sees
    e4    w4, likewise per 4 frames
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_spacing_study as S  # noqa: E402

OUT = S.ROOT / "runs" / "fm_window"
_BASE = {"FM_BURST_FRAMES": "28"}
VARIANTS = {
    "all": {**_BASE, "DATA2G_SPREAD_WINDOW": "0"},
    "w16": {**_BASE, "DATA2G_SPREAD_WINDOW": "16"},
    "w8": {**_BASE, "DATA2G_SPREAD_WINDOW": "8"},
    "w4": {**_BASE, "DATA2G_SPREAD_WINDOW": "4"},
    "e8": {**_BASE, "DATA2G_SPREAD_WINDOW": "8", "DATA2G_EST_WINDOW": "8"},
    "e4": {**_BASE, "DATA2G_SPREAD_WINDOW": "4", "DATA2G_EST_WINDOW": "4"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "report", "variant"])
    ap.add_argument("--variant")
    ap.add_argument("--only", nargs="+", help="run: these variants only")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    if a.what == "variant":
        return S.run_variant(OUT, a.variant, a.trials, a.workers)
    if a.what == "run":
        for v in a.only or VARIANTS:  # one process each: config reads its overrides at import
            subprocess.run([sys.executable, __file__, "variant", "--variant", v, "--trials", str(a.trials),
                            "--workers", str(a.workers)], env={**os.environ, **VARIANTS[v], "DATA2G_PROFILE": "fm"},
                           check=True)
    S.report_db(OUT, list(VARIANTS))


if __name__ == "__main__":
    main()
