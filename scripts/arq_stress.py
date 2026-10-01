"""ARQ link-core stress: many lockstep runs (tests/test_arq.py's harness)
across a loss grid; counts outcomes and failure reasons. Any
"protocol:" failure or accounting mismatch is a bug.

    uv run python scripts/arq_stress.py --seeds 2000
"""

import argparse
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
import test_arq as T  # noqa: E402

GRID = [(pb, pc) for pb in (0.0, 0.1, 0.3, 0.5) for pc in (0.0, 0.1, 0.3, 0.5)]


def one(args):
    seed, pb, pc = args
    rng = random.Random(seed)

    try:
        big = rng.random() < 0.3  # full-window bursts (64 codewords)
        result, stats = T.run(seed, pb, pc, rng.randrange(0, 30000 if big else 6000), rng.randrange(0, 3000),
                              max_turns=20000, change=rng.choice([0.0, 0.1, 0.4]), max_cw=64 if big else 20)
    except AssertionError as e:
        return (pb, pc, "CORRUPT", str(e)[:80], 0)
    return (pb, pc, result, T.LAST_REASON[0], stats["mismatch"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--sessions", action="store_true", help="whole sessions on a clock instead of the link core")
    a = ap.parse_args()
    if a.sessions:
        return sessions(a.seeds, a.jobs)
    jobs = [(1000 + i, *GRID[i % len(GRID)]) for i in range(a.seeds)]
    with Pool(a.jobs) as p:
        res = p.map(one, jobs, chunksize=16)
    by = Counter((pb, pc, r, why.split(":")[0] if why else "") for pb, pc, r, why, _ in res)
    for k in sorted(by):
        print(k, by[k])
    bad = [r for r in res if r[2] == "CORRUPT" or r[3].startswith("protocol") or r[4]]
    print(f"\n{len(res)} runs, {len(bad)} bugs (corruption, protocol fail-safe, or mismatch)")
    for r in bad[:10]:
        print("  ", r)



# --- sessions (tests/test_arq_session.py harness) ---------------------------------------

WALL_S = 120  # a run past this is a hang, not a slow run: the simulated clock has a horizon


class Hang(Exception):
    pass


def _alarm(signum, frame):
    raise Hang()


def one_session(args):
    import signal
    import traceback

    import test_arq_session as TS
    seed, pb, pc = args
    rng = random.Random(seed)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(WALL_S)
    try:
        late = rng.uniform(20.0, 400.0) if rng.random() < 0.5 else None  # the callee writes then: its wakes
        r = TS.run(seed, pb, pc, n_a=rng.randrange(0, 6000), n_b=rng.randrange(0, 3000), horizon=5000.0,
                   b_write_at=late, chat=rng.random() < 0.3)
    except AssertionError as e:
        return (pb, pc, "CORRUPT", f"seed {seed}: {str(e)[:60]}", 0, 0)
    except Hang:
        where = " <- ".join(f"{f.name}:{f.lineno}" for f in traceback.extract_stack(sys.exc_info()[2].tb_frame)[-1:]
                            ) + " | " + " <- ".join(f"{f.name}:{f.lineno}" for f in traceback.extract_tb(sys.exc_info()[2])[-4:])
        return (pb, pc, "HANG", f"seed {seed}: {where}", 0, 0)
    finally:
        signal.alarm(0)
    a, b = r["a"], r["b"]
    delivered = r["got_b"] == r["data_a"] and r["got_a"] == r["data_b"]
    if a.close_reason == "no answer" and b.state in ("listen", "closed"):
        out = "closed: no answer"  # the callee never heard a CONNECT, or timed its session out
    elif a.state != "closed" or b.state != "closed":
        out = "stuck"
    elif delivered:
        out = "done"
    else:
        out = "closed: " + a.close_reason.split(":")[0]
    return (pb, pc, out, a.close_reason, r["mismatch"], r["collisions"])


def sessions(n, jobs):
    jobs_ = [(5000 + i, *GRID[i % len(GRID)]) for i in range(n)]
    res = []
    with Pool(jobs) as p:
        for r in p.imap_unordered(one_session, jobs_, chunksize=4):
            res.append(r)
            if r[2] in ("CORRUPT", "stuck", "HANG") or r[4] or r[5]:
                print("BUG", r, flush=True)
            if len(res) % 100 == 0:
                print(len(res), "done", flush=True)
    by = Counter((pb, pc, out) for pb, pc, out, *_ in res)
    for k in sorted(by):
        print(k, by[k])
    bad = [r for r in res if r[2] in ("CORRUPT", "stuck", "HANG") or r[4] or r[5]]
    print(f"\n{len(res)} sessions, {len(bad)} bugs (corruption, hang, stuck, mismatch or collision)")
    for r in bad[:10]:
        print("  ", r)


if __name__ == "__main__":
    main()
