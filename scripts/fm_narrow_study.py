"""FM plan Q16: is a 12.5 kHz channel just a weaker SNR for the narrow
ladder? The same w48 modes on the mic path, three arms, streamed receiver
(scripts/fm_spacing_study.py's harness, FM profile defaults, 30 paired trials):

    wide     a 25 kHz channel radio (15 kHz IF, 5 kHz limiter), driven to 15 kHz (26 dB down)
    lowdev   the 25 kHz radio driven to 12.5 kHz: the bandwidth target alone
    narrow   a 12.5 kHz channel radio (11 kHz IF, 2.5 kHz limiter) at half wide's
             drive: its limiter clips as often as wide's (2.9%). Driven by bandwidth
             it never reaches 12.5 kHz (the limiter holds it near 10 kHz), and the
             bandwidth search drove it into 30-40% clipping
    narrow-hot  the same radio at 0.7 of wide's drive (about 14% clipped)
    half     the 25 kHz radio at half drive: narrow's deviation without its radio

narrow - half is what the narrow radio itself (IF filter, limiter) adds;
half - wide (about 6 dB above threshold) is the deviation cut as plain SNR.

    python scripts/fm_narrow_study.py run [--trials 30] [--workers 10]
    python scripts/fm_narrow_study.py report
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_spacing_study as S  # noqa: E402

OUT = S.ROOT / "runs" / "fm_narrow"
_BASE = {"FM_RX_STREAM": "1", "FM_RADIOS": "mic"}
VARIANTS = {
    "wide": {**_BASE},
    "lowdev": {**_BASE, "FM_OBW_HZ": "12500"},
    "narrow": {**_BASE, "FM_MIC_RADIO": "mic-narrow", "FM_DRIVE_RADIO": "mic", "FM_DRIVE_SCALE": "0.5"},
    "narrow-hot": {**_BASE, "FM_MIC_RADIO": "mic-narrow", "FM_DRIVE_RADIO": "mic", "FM_DRIVE_SCALE": "0.7"},
    "half": {**_BASE, "FM_DRIVE_SCALE": "0.5"},
}


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
