"""data2g.tnc: the KISS codec in C++ (native/core/tnc) against the Python
reference. Skips if the module isn't built; `pytest --native` errors
instead."""

import numpy as np

from data2g import tnc


def test_kiss_matches(native, reference):
    T = native.tnc
    rng = np.random.default_rng(0)
    frames = [bytes(rng.integers(0, 256, int(n), dtype=np.uint8)) for n in rng.integers(0, 300, 20)]
    frames += [bytes([0xC0, 0xDB, 0xDC, 0xDD] * 5)]
    wire = b"".join(T.kiss_encode(f, p % 16) for p, f in enumerate(frames))
    assert wire == b"".join(reference(tnc, "kiss_encode")(f, p % 16) for p, f in enumerate(frames))
    d, pd = T.KissDecoder(), reference(tnc, "KissDecoder")()
    for i in range(0, len(wire), 7):
        assert d.feed(wire[i:i + 7]) == pd.feed(wire[i:i + 7])
