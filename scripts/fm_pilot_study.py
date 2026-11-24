"""FM plan M4/Q8: pilot thinning. Today one symbol in six is a pilot
(DATA2G_SYMS_PER_FRAME 6: 16.7%, a pilot every 132 ms), sized for HF's
2 Hz Doppler. One in eleven (9.1%, every 242 ms) keeps every test mode's
codeword in whole frames (half the frames per codeword) and buys 9.1% rate.

FM audio's channel is the radios' fixed audio response: RF Doppler reaches
it as clicks and fades, not as fading of the audio tones. The equalizer's
prior for bursts too short to measure a spread (equalizer.DEFAULT_SPREAD_HZ,
2 Hz) is HF's; at one pilot in eleven it all but discards the pilots (clean
64-point bursts of 2-5 frames failed). So the prior is an arm of its own:

    spf6     today: one in six, prior 2 Hz, 8-frame windows
    spf6p    one in six, prior 0.2 Hz (DATA2G_DEFAULT_SPREAD_HZ)
    spf11    one in eleven, prior 2 Hz, 4-frame windows (about the same 1 s)
    spf11p   one in eleven, prior 0.2 Hz, 4-frame windows

w48 QPSK r1/2, 16-QAM r1/2, 64-point r3/4; mic and flat radios, every
condition, streamed receiver, FM profile defaults, 30 paired trials. Bursts
of one in eleven are shorter, so fading draws differ on mobile and slow
paths: those cells are not truly paired (see the prefix study's intervals).

    python scripts/fm_pilot_study.py run [--trials 30] [--workers 8]
    python scripts/fm_pilot_study.py report
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_spacing_study as S  # noqa: E402

OUT = S.ROOT / "runs" / "fm_pilot"
_BASE = {"FM_RX_STREAM": "1"}
_P = {"DATA2G_DEFAULT_SPREAD_HZ": "0.2"}
_11 = {"DATA2G_SYMS_PER_FRAME": "11", "DATA2G_SPREAD_WINDOW": "4", "DATA2G_EST_WINDOW": "4"}
VARIANTS = {"spf6": {**_BASE}, "spf6p": {**_BASE, **_P}, "spf11": {**_BASE, **_11}, "spf11p": {**_BASE, **_11, **_P}}


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
