"""scripts/cpu_profile/speedtest.py end to end: a transfer that completes
reports its rate; one on a dead channel fails at connect, exit 1."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts/cpu_profile/speedtest.py"


def speedtest(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=600)


def test_bulk_transfer_reports_rate():
    r = speedtest("awgn", "20", "--bytes", "2000")
    assert r.returncode == 0, r.stderr
    assert "2000 B in" in r.stdout and "bit/s" in r.stdout and "B/min" in r.stdout


def test_dead_channel_fails_during_connect():
    r = speedtest("awgn", "-20", "--bytes", "2000")
    assert r.returncode == 1 and "FAILED during connect" in r.stdout
