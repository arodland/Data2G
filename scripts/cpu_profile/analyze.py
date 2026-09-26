"""py-spy raw (collapsed) profiles -> inclusive and self time per function,
and the main loop's split. Usage: analyze.py prof.txt [prof.txt ...]"""
import re
import sys
from collections import Counter


def load(paths):
    stacks = Counter()
    for p in paths:
        for line in open(p):
            line = line.rstrip()
            if not line:
                continue
            s, _, n = line.rpartition(" ")
            stacks[s] += int(n)
    return stacks


def fn(frame):
    # "func (path/file.py:line)" -> "file.py:func"
    m = re.match(r"(.*?) \((.*?):\d+\)", frame)
    if not m:
        return frame
    name, path = m.groups()
    return f"{path.split('/')[-1]}:{name}"


def main():
    stacks = load(sys.argv[1:])
    total = sum(stacks.values())
    incl, self_ = Counter(), Counter()
    for s, n in stacks.items():
        frames = [fn(f) for f in s.split(";") if f]
        for f in set(frames):
            incl[f] += n
        if frames:
            self_[frames[-1]] += n
    print(f"samples {total}")
    print("\n-- inclusive (% of CPU samples)")
    for f, n in incl.most_common(45):
        print(f"{100 * n / total:6.1f}  {f}")
    print("\n-- self")
    for f, n in self_.most_common(30):
        print(f"{100 * n / total:6.1f}  {f}")


if __name__ == "__main__":
    main()
