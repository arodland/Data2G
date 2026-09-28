import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data2g import constellation, modem  # noqa: E402
from data2g.channel_torch import CHANNELS, BurstChannel, bmi, llr  # noqa: E402
from data2g.config import DATA_SYMS_PER_FRAME, NC, SubmodeSpec  # noqa: E402

SPEC = SubmodeSpec(0, "t", "ldpc", "gray-qam16", 1)


def test_transmit_matches_numpy_modulator():
    rng = np.random.default_rng(0)
    n_f = 4
    bits = rng.integers(0, 2, n_f * DATA_SYMS_PER_FRAME * NC * 4)
    ref = modem.modulate_bits(bits, SPEC)
    data = constellation.modulate(bits, constellation.load(SPEC.constellation))
    ch = BurstChannel(SPEC, n_f, dtype=torch.float64)
    got = ch.transmit(torch.tensor(data.reshape(1, n_f, DATA_SYMS_PER_FRAME, NC)))[0].numpy()
    np.testing.assert_allclose(got, ref, atol=1e-9)


def test_ace_transmit_matches_numpy_and_keeps_points_in_their_regions():
    """ACE in both clippers alike; after it, each 16-QAM cell's error has
    no inward part left along its outer axes, and less clip error overall."""
    import dataclasses

    rng = np.random.default_rng(2)
    n_f = 4
    spec = dataclasses.replace(SubmodeSpec(0, "t", "ldpc", "gray-qam16", 1), clip_headroom_db=1.0,
                               ace=(1.0, 1.5, 2.0))
    bits = rng.integers(0, 2, n_f * DATA_SYMS_PER_FRAME * NC * 4)
    ref = modem.modulate_bits(bits, spec)
    data = constellation.modulate(bits, constellation.load(spec.constellation))
    ch = BurstChannel(spec, n_f, dtype=torch.float64, clip_consts=({}, 1.0, 0.0))
    got = ch.transmit(torch.tensor(data.reshape(1, n_f, DATA_SYMS_PER_FRAME, NC)))[0].numpy()
    np.testing.assert_allclose(got, ref, atol=1e-9)
    plain = modem.modulate_bits(bits, dataclasses.replace(spec, ace=()))
    win, _ = modem.ace_cells(spec, n_f)
    dem = modem.ofdm.band("w").mod[modem.NCP:].conj()

    dirs = constellation.ace_dirs(spec.constellation)[
        bits.reshape(-1, 4) @ np.array([8, 4, 2, 1])]

    def harm(x):
        """Mean square of the error that isn't outward along a point's axes."""
        cells = ((2.0 / modem.M) * (x[win] @ dem)).reshape(-1)
        cells = cells * np.vdot(cells, data) / np.vdot(cells, cells)  # the clip's gain out
        return np.mean(np.abs(cells - constellation.ace_project(cells, data, dirs)) ** 2)

    assert harm(ref) < 0.5 * harm(plain)


def test_llr_matches_numpy():
    rng = np.random.default_rng(1)
    p = constellation.gray_qam(4)
    y, h = (rng.normal(size=(2, 30)) + 1j * rng.normal(size=(2, 30)))
    var = rng.uniform(0.5, 2, 30)
    got = llr(torch.tensor(y), torch.tensor(h), torch.tensor(var), torch.tensor(p)).numpy().reshape(-1)
    np.testing.assert_allclose(got, constellation.llr(y, h, var, p), rtol=1e-9)


def test_bmi_agrees_with_the_full_modem():
    """QPSK at mpp 10 dB: the numpy modem (scripts/eq_floor.py) measured
    0.90-0.92 BMI over 5-10 bursts. The torch path skips acquisition but
    has the same TX, fading, noise convention and estimator."""
    spec = SubmodeSpec(0, "t", "ldpc", "gray-qam4", 1)
    ch = BurstChannel(spec, 20)
    p = torch.tensor(constellation.gray_qam(2), dtype=torch.complex64)
    vals = []
    for seed in range(3):  # one seed spans 0.856-0.898 (16 bursts of fading draws)
        g = torch.Generator().manual_seed(seed)
        bits = torch.randint(0, 2, (16, 20, DATA_SYMS_PER_FRAME, NC, 2), generator=g)
        data = p[bits[..., 0] * 2 + bits[..., 1]]
        y, h, var = ch.receive(ch.channel(ch.transmit(data), CHANNELS["mpp"], 10.0, g), CHANNELS["mpp"])
        vals.append(bmi(llr(y, h, var, p), bits.float()).item())
    assert 0.88 < np.mean(vals) < 0.94, vals
