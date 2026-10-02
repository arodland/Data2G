"""C++ (native/) against the Python reference, side by side. Skips if the
module isn't built; `pytest --native` errors instead (conftest.py)."""

import binascii

import numpy as np
import pytest

from data2g import codes, config

SPECS = list(config.SUBMODES.values())


def test_crcs(native, reference):
    rng = np.random.default_rng(1)
    crc24 = reference(codes, "crc24")
    for n in (0, 1, 7, 56, 300):
        data = rng.bytes(n)
        assert native.codes.crc16(data) == binascii.crc_hqx(data, 0xFFFF)
        assert native.codes.crc32(data) == binascii.crc32(data)
        assert native.codes.crc24(data) == crc24(data)


def test_with_crc(native, reference):
    rng = np.random.default_rng(2)
    with_crc = reference(codes, "_with_crc")
    for n_crc in (16, 24, 32):
        for mask in (0, 1, 0xABCDEF, 0xFFFFFFFF, int(rng.integers(0, 2**32))):
            data = rng.bytes(int(rng.integers(0, 400)))
            assert native.codes.with_crc(data, n_crc, mask) == with_crc(data, n_crc, mask)


def test_scrambler(native, reference):
    scramble_seed, scrambler = reference(codes, "scramble_seed"), reference(codes, "scrambler")
    for index in [codes.PLAIN, *range(600)]:
        assert native.codes.scramble_seed(index) == scramble_seed(index)
    for seed in range(512):
        np.testing.assert_array_equal(native.codes.scrambler(3200, seed), scrambler(3200, seed))


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_frozen_format(native, reference, spec):
    f = reference(codes, "frozen")(spec)
    np.testing.assert_array_equal(native.codes.interleaver(spec.name), f["perm"])
    if spec.code == "polar":
        np.testing.assert_array_equal(native.codes.info_pos(spec.name), f["info_pos"])


def test_submode_table(native):
    rows = native.config.submodes()
    assert [r["name"] for r in rows] == [s.name for s in SPECS]
    for r, s in zip(rows, SPECS):
        assert r == {"index": s.index, "name": s.name, "code": s.code, "constellation": s.constellation,
                     "band": s.band, "frames_per_cw": s.frames_per_cw, "k": s.k,
                     "coded_bits": s.coded_bits, "headroom": float(s.headroom)}


def test_generated_tables_current():
    import subprocess
    import sys
    from pathlib import Path

    tool = Path(__file__).resolve().parent.parent / "tools" / "gen_native_tables.py"
    r = subprocess.run([sys.executable, str(tool), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
