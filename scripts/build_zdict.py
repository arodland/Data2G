"""Build deflate's priming dictionary (frames.ZDICT): a 4 KB string that
shares as many frequent substrings as possible with a text corpus.

Iterated epsilon-greedy: grow one string, each step appending a byte at
the right, prepending one at the left, or appending a frequent word,
scored by the corpus count of the distinct 3..L-byte substrings it adds
(per byte added). Several restarts; the one that deflates held-out text
smallest wins.

    python scripts/build_zdict.py OUT --train BOOK... --test TEXT...
"""

import argparse
import random
import re
import zlib
from collections import Counter
from pathlib import Path

L = 12  # longest substring counted
MINC = 4  # corpus count below which a substring is ignored
SIZE = 4096


def clean(b: bytes) -> bytes:
    """A Gutenberg text -> its body, LF line ends, wraps joined."""
    t = b.decode("utf-8", "ignore")
    if (m := re.search(r"\*\*\* START OF .*?\*\*\*(.*)\*\*\* END OF", t, re.S)):
        t = m.group(1)
    t = t.replace("\r\n", "\n")
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)  # hard wraps -> spaces, paragraphs kept
    return t.encode("ascii", "ignore")


def counts(corpus: bytes) -> dict[bytes, int]:
    freq = {}
    for n in range(3, L + 1):
        c = Counter(corpus[i:i + n] for i in range(len(corpus) - n + 1))
        freq.update((s, k) for s, k in c.items() if k >= MINC)
        print(f"  {n}-grams kept: {sum(k >= MINC for k in c.values())}", flush=True)
    return freq


def grow(freq, words, eps, rng) -> bytes:
    nxt, prv = {}, {}
    for s in freq:
        if len(s) == 3:
            nxt.setdefault(s[:2], []).append(s[2:])
            prv.setdefault(s[1:], []).append(s[:1])
    d = bytearray(words[0])
    seen = {bytes(d[i:j]) for i in range(len(d)) for j in range(i + 3, min(len(d), i + L) + 1)}

    def gain(new):
        return sum(freq.get(s, 0) for s in new if s not in seen)

    while len(d) < SIZE:
        cands = []
        for c in nxt.get(bytes(d[-2:]), ()):
            t = bytes(d[-(L - 1):]) + c
            cands.append((gain({t[i:] for i in range(len(t) - 2)}), "R", c))
        for c in prv.get(bytes(d[:2]), ()):
            t = c + bytes(d[:L - 1])
            cands.append((gain({t[:j] for j in range(3, len(t) + 1)}), "L", c))
        for w in words[:100]:
            t = bytes(d[-(L - 1):]) + w
            k = len(t) - len(w)  # new substrings end past k
            new = {t[i:j] for i in range(len(t)) for j in range(max(i + 3, k + 1), min(len(t), i + L) + 1)}
            cands.append((gain(new) / len(w), "R", w))
        cands.sort(key=lambda x: -x[0])
        g, side, s = rng.choice(cands[:5]) if rng.random() < eps else cands[0]
        if side == "R":
            d += s
            t = bytes(d[-(L + len(s) - 1):])
            seen |= {t[i:j] for i in range(len(t)) for j in range(i + 3, min(len(t), i + L) + 1)}
        else:
            d[:0] = s
            t = bytes(d[:L])
            seen |= {t[:j] for j in range(3, len(t) + 1)}
    return bytes(d[-SIZE:])


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
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--per-book", type=int, default=200_000)
    ap.add_argument("--baseline", nargs="*", default=[], help="dictionaries to score alongside")
    a = ap.parse_args()
    books = [clean(Path(f).read_bytes()) for f in a.train]
    corpus = b"\n\n".join(b[len(b) // 2 - a.per_book // 2:][:a.per_book] for b in books)
    tests = [clean(Path(f).read_bytes()) for f in a.test]
    print(f"corpus {len(corpus)} B", flush=True)
    freq = counts(corpus)
    wc = Counter(re.findall(rb" [A-Za-z']+", corpus))
    words = [w + b" " for w, _ in sorted(wc.items(), key=lambda x: -x[1] * len(x[0]))]
    for f in a.baseline:
        print(f"baseline {f}: {score(Path(f).read_bytes(), tests)}")
    print(f"no dictionary: {score(b'', tests)}")
    best = None
    for eps in (0.0, 0.05, 0.1, 0.2):
        for seed in range(1 if eps == 0 else 2):
            d = grow(freq, words, eps, random.Random(seed))
            s = score(d, tests)
            print(f"eps {eps} seed {seed}: {s}", flush=True)
            if best is None or s < best[0]:
                best = (s, d)
    Path(a.out).write_bytes(best[1])
    print(f"best {best[0]} -> {a.out}")


if __name__ == "__main__":
    main()
