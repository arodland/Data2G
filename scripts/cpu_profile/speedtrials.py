"""speedtest.py over seeds `--seed`, `--seed` + 1, ... until 5 transfers
complete or 10 fail. Reports the success rate and the mean speed: of the
middle three with 5 successes (high and low dropped), else of all of them.

    python scripts/cpu_profile/speedtrials.py <channel> <snr_db> [--bytes 20000] [--bw 2300] [--seed 0]
"""
import argparse
import sys

import speedtest as T  # first: it sets the thread caps before numpy loads

import numpy as np  # noqa: E402

WINS, LOSSES = 5, 10


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("channel", help="awgn | mpg | mpp | mpd | mps | doppler_hz:delay_ms")
    ap.add_argument("snr", type=float, help="dB, PEP over noise in 3000 Hz")
    ap.add_argument("--bytes", type=int, default=20000)
    ap.add_argument("--bw", choices=["500", "1200", "2300", "2750"], default="2300")
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--latency", type=float, default=T.LATENCY_S)
    a = ap.parse_args()
    ok, failed = [], {}
    seed = a.seed
    while len(ok) < WINS and sum(failed.values()) < LOSSES:
        r = T.run(a.channel, a.snr, a.bytes, seed, cap=T.BW["BW" + a.bw], latency=a.latency)
        if r["phase"] == "done":
            ok.append(r)
            print(f"seed {seed}: {r['seconds']:.1f} s = {r['bps']:.0f} bit/s = {r['bpm']:.0f} B/min; {T.top2(r['airtime'])}", flush=True)
        else:
            failed[r["phase"]] = failed.get(r["phase"], 0) + 1
            print(f"seed {seed}: FAILED during {r['phase']} at {r['t']:.1f} s", flush=True)
        seed += 1
    n = len(ok) + sum(failed.values())
    why = ", ".join(f"{v} {k}" for k, v in failed.items())
    print(f"{a.channel} {a.snr:g} dB, {a.bytes} B, BW{a.bw}: {len(ok)}/{n} succeeded"
          + (f", failed: {why}" if failed else ""))
    if not ok:
        sys.exit(1)
    ok.sort(key=lambda r: r["bps"])
    used = ok[1:-1] if len(ok) == WINS else ok
    what = "middle 3 of 5" if len(ok) == WINS else f"all {len(ok)}"
    print(f"mean of {what}: {np.mean([r['seconds'] for r in used]):.1f} s = "
          f"{np.mean([r['bps'] for r in used]):.0f} bit/s = {np.mean([r['bpm'] for r in used]):.0f} B/min; "
          f"{T.top2({k: sum(r['airtime'].get(k, 0.0) for r in used) for q in used for k in q['airtime']})}")


if __name__ == "__main__":
    main()
