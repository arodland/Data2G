"""FM plan M3: receiver changes for FM, each measured in dB of CNR against
today's receiver, through data2g.fmchannel on the FM profile.

    python scripts/fm_rx_study.py run [--trials 30] [--workers 10] [--only V ...]
    python scripts/fm_rx_study.py report

The harness is scripts/fm_spacing_study.py's (its 50 Hz variant: w48 modes
QPSK r1/2, 16-QAM r1/2, 64-point r3/4; mic and flat radios; every condition
at 146 and 440 MHz; 30 paired trials), each variant its own process with
its environment, through the live receiver (FM_RX_STREAM: receiver.Receiver
fed 0.25 s at a time after 1.8 s of open-squelch hiss, the carrier keying
up 0.2 s before the burst). Writes runs/fm_rx/<variant>.csv; `report` gives
each variant against the first in dB (fm_spacing_study.report_db), and the
false alarms on the hiss.

modem.demodulate on a buffer known to hold the burst (the spacing study's
receiver) cannot see these changes: every variant below, blanker and
thresholds included, came out identical through it (2026-11-15). It tries
the best peaks whatever their score; only the streaming receiver gates
on the threshold, and blanks.

Variants (FM_SPACING_THRESHOLD: the preamble threshold):
    base     today's receiver: Blanker, +-150 Hz search, 26.7 (white-noise
             peaks under FM, fm_spacing_study round 2)
    noblank  base without the Blanker
    fm150    base at the threshold FM noise needs at +-150 Hz: 37.2
    fm12     +-12.5 Hz (FM audio has no frequency offset), its threshold 28.7
    sym      fm12, and per-symbol noise in the demapper (modem.SYMBOL_NOISE)

Thresholds: 1.1 x the largest of 1200 s noise peaks (scripts/mf_detect_study.py
noise --blank, w48, 8 repeats, 2026-11-15) over what an idle FM receiver hands
the modem. +-150 Hz: white 22.7, mic hiss with no carrier (open squelch)
33.8, flat 21.1, idle carrier at 8 dB CNR 27.3 (mic) / 30.2 (flat). +-12.5 Hz:
21.2, 26.1, 18.4, 22.5, 24.5. Without the blanker an idle carrier's clicks
read 35-38 within 40 s.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_spacing_study as S  # noqa: E402

OUT = S.ROOT / "runs" / "fm_rx"
_STREAM = {"FM_RX_STREAM": "1"}
_FM12 = {**_STREAM, "FM_SPACING_THRESHOLD": "28.7", "DATA2G_ACQUIRE_REACH_HZ": "12.5"}
VARIANTS = {
    "base": {**_STREAM, "FM_SPACING_THRESHOLD": "26.7"},
    "noblank": {**_STREAM, "FM_SPACING_THRESHOLD": "26.7", "FM_RX_NO_BLANK": "1"},
    "fm150": {**_STREAM, "FM_SPACING_THRESHOLD": "37.2"},
    "fm12": _FM12,
    "sym": {**_FM12, "DATA2G_SYMBOL_NOISE": "1"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "report", "variant"])
    ap.add_argument("--variant")
    ap.add_argument("--only", nargs="+", help="run: these variants only")
    ap.add_argument("--trials", type=int, default=30)
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
