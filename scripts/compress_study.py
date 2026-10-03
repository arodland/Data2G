"""Codeword compression candidates (docs/arq.md §9a): how many codewords a
stream takes at each payload size, and CPU per codeword including the
binary search for the prefix that fits.

    python scripts/compress_study.py [FILE...]  # deflate, zstd (stdlib, 3.14); FILEs: more texts
    uv run --no-project --python 3.14 --with brotli python scripts/compress_study.py
"""

import gzip
import sys
import time
import zlib
from pathlib import Path

from compression import zstd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data2g.arq.frames import ZDICT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
P = zstd.CompressionParameter


def deflate(h, x):
    c = zlib.compressobj(9, zlib.DEFLATED, -15, 9, **({"zdict": h} if h else {}))
    return c.compress(x) + c.flush()


def zs(level):
    opts = {P.compression_level: level, P.checksum_flag: 0, P.content_size_flag: 0, P.dict_id_flag: 0}

    def f(h, x):
        return zstd.compress(x, options=opts, zstd_dict=zstd.ZstdDict(h, is_raw=True) if h else None)[4:]  # no magic
    return f


def run(data, pb, comp, hist):
    i = n = 0
    t = time.perf_counter()
    while i < len(data):
        h = data[max(0, i - hist):i] if hist else b""
        lo, hi, best = pb + 1, min(len(data) - i, 16 * pb), 0
        while lo <= hi:
            m = (lo + hi) // 2
            if len(comp(h, data[i:i + m])) <= pb:
                best, lo = m, m + 1
            else:
                hi = m - 1
        i += max(best, pb)
        n += 1
    return -(-len(data) // pb) / n, (time.perf_counter() - t) / n * 1e3


def main():
    srcs = {"text docs/arq.md": (ROOT / "docs/arq.md").read_bytes(),
            "code link.py": (ROOT / "data2g/arq/link.py").read_bytes()}
    srcs["gzipped text"] = gzip.compress(srcs["text docs/arq.md"])
    for f in sys.argv[1:]:  # past a Gutenberg header
        srcs[Path(f).name] = Path(f).read_bytes()[50000:]
    algs = [("deflate", deflate, 0), ("deflate+4K", deflate, 4096), ("deflate+32K", deflate, 32768),
            ("zdict+4K", lambda h, x: deflate(ZDICT + h, x), 4096),
            ("zstd3", zs(3), 0), ("zstd19", zs(19), 0), ("zstd3+4K", zs(3), 4096), ("zstd19+4K", zs(19), 4096)]
    try:
        import brotli  # no dictionary in the binding: no history
        algs.append(("brotli11", lambda h, x: brotli.compress(x, quality=11, lgwin=10), 0))
    except ImportError:
        pass
    sizes = (38, 76, 176, 396)
    for (name, d), n in ((s, n) for s in srcs.items() for n in (1000, 20000)):
        d = d[:n]
        print(f"\n{name} ({len(d)} B): fewer codewords x, ms per codeword")
        print(f"{'':12s}" + "".join(f"{pb:>16d}" for pb in sizes))
        for an, f, h in algs:
            print(f"{an:12s}" + "".join("%10.2fx %4.1f" % run(d, pb, f, h) for pb in sizes), flush=True)


if __name__ == "__main__":
    main()
