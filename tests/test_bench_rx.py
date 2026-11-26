"""native data2g-bench-rx (native/apps): a burst it can't receive is reported lost, not a crash. Skips if the
build has no bench (tools/build_native.sh builds it)."""

import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).parent.parent / "native" / "build" / "data2g-bench-rx"


@pytest.mark.skipif(not BENCH.exists(), reason="data2g-bench-rx not built")
def test_a_burst_whose_header_cannot_be_read_is_lost(tmp_path):
    """A burst file holding noise alone: modem::receive throws SyncError, which aborted the whole run
    (std::terminate) where receiver::Receiver reports the burst lost (a mix qpsk-r1/2 hilat-m burst, 2026-11-25)."""
    audio = np.random.default_rng(0).normal(0, 0.1, 3 * 8000)
    path = tmp_path / "noise.d2gb"
    path.write_bytes(b"D2GB" + struct.pack("<qqii8s", len(audio), 8000, 2, 10, b"w") + audio.astype("<f8").tobytes()
                     + bytes(20))
    r = subprocess.run([str(BENCH), "--threads", "1", "--reps", "1", str(path)], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    assert "0/2" in r.stdout and r.stdout.rstrip().endswith("lost")


@pytest.mark.skipif(not BENCH.exists(), reason="data2g-bench-rx not built")
def test_a_listen_run_no_longer_than_its_warm_up_is_refused():
    """--listen leaves its first 5 s of feeds out as warm-up: a run of 5 s left none, and its quantiles read
    before the empty list's start (a segfault, 2026-11-26). The Receiver itself had fed all 250 feeds."""
    r = subprocess.run([str(BENCH), "--listen", "5"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "warm-up" in r.stderr
