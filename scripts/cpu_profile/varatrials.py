"""vara_ref.sh over seeds `--seed`, `--seed` + 1, ... until 5 transfers
complete or 10 fail, reported as speedtrials.py reports ours (trials.py):
payload time only, raw.py's write of the whole message until the far end
has it all. Real time: a run lasts as long as VARA takes.

    python scripts/cpu_profile/varatrials.py <channel> <snr_db> [--bytes 20000] [--bw 2300] [--seed 0]

A run where VARA opened no audio is the harness's failure, not VARA's: it
stops the trials (exit 2). Each run's files: <out>/<channel>_<snr>dB_<bw>_<bytes>B/seed<n>/.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import trials

W = Path(__file__).resolve().parent


def stop(why):
    """The harness failed, not VARA: exit 2 (trials.run exits 1 when nothing completed)."""
    print(why, file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("channel", help="awgn | mpg | mpp | mpd | mps | doppler_hz:delay_ms")
    ap.add_argument("snr", type=float, help="dB, PEP over noise in 3000 Hz")
    ap.add_argument("--bytes", type=int, default=20000)
    ap.add_argument("--bw", choices=["500", "2300"], default="2300")
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--timeout", type=float, default=1800.0, help="one transfer's limit, s")
    ap.add_argument("--out", default="runs/varatrials")
    a = ap.parse_args()
    base = Path(a.out) / f"{a.channel}_{a.snr:g}dB_{a.bw}_{a.bytes}B"

    def trial(seed):
        env = dict(os.environ, SEED=str(seed), NOISE_SNR=str(a.snr), PY=sys.executable)
        if a.channel != "awgn":
            env["CHANNEL"] = a.channel
        p = subprocess.run([str(W / "vara_ref.sh"), a.bw, str(a.bytes), str(a.timeout), str(base / f"seed{seed}")],
                           env=env, capture_output=True, text=True, timeout=a.timeout + 300)
        out = p.stdout + p.stderr
        if "opened no audio" in out:
            stop(f"seed {seed}: {out.strip().splitlines()[-1]} (the harness, not VARA: stopping)")
        if m := re.search(r"done after ([\d.]+) s: a->b (\d+)/(\d+) B .* exact (\w+)", out):
            if m[4] != "True":
                stop(f"seed {seed}: data corrupted: {m[0]}")
            s, n = float(m[1]), int(m[3])
            return dict(phase="done", seconds=s, bps=8 * n / s, bpm=60 * n / s)
        if m := re.search(r"timeout after ([\d.]+) s", out):
            return dict(phase="bulk", t=float(m[1]))
        if "no connect" in out:
            return dict(phase="connect")
        stop(f"seed {seed}: unexpected output:\n{out}")

    trials.run(trial, a.seed, f"VARA {a.channel} {a.snr:g} dB, {a.bytes} B, BW{a.bw}")


if __name__ == "__main__":
    main()
