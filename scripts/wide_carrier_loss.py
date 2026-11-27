"""scripts/loss_study.py with an in-grid carrier preset, "incar" (scripts/wide_fsk_study.py's): cells chan:snr:incar."""
import sys

from scripts import wide_fsk_study  # noqa: F401 (adds the preset; pool workers inherit it)
from scripts import loss_study

if __name__ == "__main__":
    loss_study.main()
