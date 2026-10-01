"""Build deflate's priming dictionary (frames.ZDICT, data2g/arq/zdict.bin): a 4 KB string that
shares as many frequent substrings as possible with a text corpus.

Greedy from a random byte: grow one string, each step appending a byte at
the right or prepending one at the left, whichever adds distinct 3..L-byte
substrings of the largest total corpus count (ties at random). Restarts for
--budget seconds across a pool; the KEEP that cover most are kept, every
rotation of each is scored on validation text, and the one that deflates
it smallest wins.

    python scripts/build_zdict.py OUT --train TEXT... --val TEXT... --test TEXT...
"""

import argparse
import random
import re
import resource
import time
import zlib
import multiprocessing
from collections import Counter
from pathlib import Path

L = 12  # longest substring counted
MINC = 4  # corpus count below which a substring is ignored
SIZE = 4096
KEEP = 10  # restarts kept for the rotation search
WORKERS = 8
ALPHA = bytes(range(32, 127)) + b"\n"
ALPHA_B = [bytes([c]) for c in ALPHA]
FREQ: dict[bytes, int] = {}  # pool workers inherit these (fork)
NXT: dict = {}
PRV: dict = {}
VAL: list[bytes] = []


def clean(b: bytes) -> bytes:
    """ASCII, LF line ends; a Gutenberg text -> its body, wraps joined."""
    t = b.decode("utf-8", "ignore").replace("\r\n", "\n")
    if (m := re.search(r"\*\*\* START OF .*?\*\*\*(.*)\*\*\* END OF", t, re.S)):
        t = re.sub(r"(?<!\n)\n(?!\n)", " ", m.group(1))  # hard wraps -> spaces, paragraphs kept
    return t.encode("ascii", "ignore")


def counts(corpus: bytes) -> dict[bytes, int]:
    freq = {}
    for n in range(3, L + 1):
        c = Counter(corpus[i:i + n] for i in range(len(corpus) - n + 1))
        freq.update((s, k) for s, k in c.items() if k >= MINC)
        print(f"  {n}-grams kept: {sum(k >= MINC for k in c.values())}", flush=True)
    return freq


def tables(freq):
    """3-gram continuations: 2-byte context -> bytes that may follow / precede."""
    nxt, prv = {}, {}
    for s in freq:
        if len(s) == 3:
            nxt.setdefault(s[:2], []).append(s[2:])
            prv.setdefault(s[1:], []).append(s[:1])
    return nxt, prv


def grow(seed: int) -> tuple[int, bytes]:
    """Greedy from a random byte, ties broken at random: each step adds the
    byte, at either end, whose new distinct substrings have the largest
    total corpus count. -> (total count covered, dictionary)."""
    rng = random.Random(seed)
    d = bytearray([rng.choice(ALPHA)])
    seen, total = set(), 0
    while len(d) < SIZE:
        cands = []
        for c in NXT.get(bytes(d[-2:])) or ALPHA_B:
            t = bytes(d[-(L - 1):]) + c
            cands.append((sum(FREQ.get(x, 0) for x in {t[i:] for i in range(len(t) - 2)} if x not in seen), 1, c))
        for c in PRV.get(bytes(d[:2])) or ALPHA_B:
            t = c + bytes(d[:L - 1])
            cands.append((sum(FREQ.get(x, 0) for x in {t[:j] for j in range(3, len(t) + 1)} if x not in seen), 0, c))
        top = max(g for g, _, _ in cands)
        g, right, c = rng.choice([x for x in cands if x[0] == top])
        total += g
        if right:
            d += c
            t = bytes(d[-L:])
            seen |= {t[i:] for i in range(len(t) - 2)}
        else:
            d[:0] = c
            t = bytes(d[:L])
            seen |= {t[:j] for j in range(3, len(t) + 1)}
    return total, bytes(d)


def restarts(args) -> tuple[int, list[tuple[int, bytes]]]:
    """Seeds seed0, seed0 + step, ... until the deadline -> (runs, the KEEP best)."""
    seed, step, deadline = args
    out = []
    while time.time() < deadline:
        out.append(grow(seed))
        seed += step
    return len(out), sorted(out, reverse=True)[:KEEP]


def rot_scores(args) -> list[int]:
    d, rs = args
    return [score(d[r:] + d[:r], VAL) for r in rs]


def score(zdict: bytes, tests: list[bytes]) -> int:
    """Total deflated bytes of 1 KB held-out pieces, each primed with zdict
    alone (a stream's first KB: where the dictionary matters most)."""
    total = 0
    for t in tests:
        for i in range(0, len(t) - 1000, 997 * 7):
            c = zlib.compressobj(9, zlib.DEFLATED, -15, 9, zdict=zdict)
            total += len(c.compress(t[i:i + 1000]) + c.flush())
    return total


def main():
    global FREQ, NXT, PRV, VAL
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True, help="picks the restart and the rotation")
    ap.add_argument("--test", nargs="+", required=True, help="reported only")
    ap.add_argument("--per-book", type=int, default=200_000)
    ap.add_argument("--budget", type=float, default=30.0, help="seconds of restarts")
    ap.add_argument("--baseline", nargs="*", default=[], help="dictionaries to score alongside")
    a = ap.parse_args()
    books = [clean(Path(f).read_bytes()) for f in a.train]
    corpus = b"\n\n".join(b[len(b) // 2 - a.per_book // 2:][:a.per_book] for b in books)
    VAL = [clean(Path(f).read_bytes()) for f in a.val]
    tests = [clean(Path(f).read_bytes()) for f in a.test]
    print(f"corpus {len(corpus)} B", flush=True)
    FREQ = counts(corpus)
    NXT, PRV = tables(FREQ)
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    print("deflated bytes, val / test")
    for f in a.baseline:
        print(f"baseline {f}: {score(Path(f).read_bytes(), VAL)} / {score(Path(f).read_bytes(), tests)}")
    print(f"no dictionary: {score(b'', VAL)} / {score(b'', tests)}", flush=True)
    with multiprocessing.get_context("fork").Pool(WORKERS) as pool:  # workers inherit the globals
        deadline = time.time() + a.budget
        res = pool.map(restarts, [(i, WORKERS, deadline) for i in range(WORKERS)])
        top = sorted((r for _, rs in res for r in rs), reverse=True)[:KEEP]
        print(f"{sum(n for n, _ in res)} restarts in {time.time() - deadline + a.budget:.0f} s; coverage"
              f" kept {top[-1][0]}..{top[0][0]}", flush=True)
        chunks = [range(i, min(i + 256, SIZE)) for i in range(0, SIZE, 256)]
        best = None
        for k, (cov, d) in enumerate(top):
            rot = [x for xs in pool.map(rot_scores, [(d, c) for c in chunks]) for x in xs]
            r = min(range(len(d)), key=rot.__getitem__)
            dr = d[r:] + d[:r]
            print(f"#{k} coverage {cov}: val unrotated {rot[0]}, median {sorted(rot)[len(rot) // 2]},"
                  f" best rotation {r} {rot[r]} | test {score(dr, tests)}", flush=True)
            if best is None or rot[r] < best[0]:
                best = (rot[r], dr)
    Path(a.out).write_bytes(best[1])
    print(f"best val {best[0]} -> {a.out}")


if __name__ == "__main__":
    main()
