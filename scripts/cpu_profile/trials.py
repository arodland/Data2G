"""Seeds first, first + 1, ... until 5 transfers complete or 10 fail, then the
success rate and the mean speed: of the middle three with 5 successes (high
and low dropped), else of all of them. speedtrials.py and varatrials.py."""

import statistics
import sys

WINS, LOSSES = 5, 10


def run(trial, first, head, extra=lambda results: ""):
    """trial(seed) -> dict: phase ('done' or where it failed); when done,
    seconds, bps and bpm; when failed, optionally t (seconds). extra(results)
    -> text appended to a seed's line ([result]) and to the mean (those used).
    Exits 1 if nothing completed."""
    ok, failed = [], {}
    seed = first
    while len(ok) < WINS and sum(failed.values()) < LOSSES:
        r = trial(seed)
        if r["phase"] == "done":
            ok.append(r)
            print(f"seed {seed}: {r['seconds']:.1f} s = {r['bps']:.0f} bit/s = {r['bpm']:.0f} B/min{extra([r])}", flush=True)
        else:
            failed[r["phase"]] = failed.get(r["phase"], 0) + 1
            at = f" at {r['t']:.1f} s" if "t" in r else ""
            print(f"seed {seed}: FAILED during {r['phase']}{at}", flush=True)
        seed += 1
    n = len(ok) + sum(failed.values())
    why = ", ".join(f"{v} {k}" for k, v in failed.items())
    print(f"{head}: {len(ok)}/{n} succeeded" + (f", failed: {why}" if failed else ""))
    if not ok:
        sys.exit(1)
    ok.sort(key=lambda r: r["bps"])
    used = ok[1:-1] if len(ok) == WINS else ok
    what = "middle 3 of 5" if len(ok) == WINS else f"all {len(ok)}"
    mean = {k: statistics.mean(r[k] for r in used) for k in ("seconds", "bps", "bpm")}
    print(f"mean of {what}: {mean['seconds']:.1f} s = {mean['bps']:.0f} bit/s = {mean['bpm']:.0f} B/min{extra(used)}")
