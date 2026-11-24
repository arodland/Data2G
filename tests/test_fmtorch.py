"""data2g.fmtorch (FM constellation training's channel) against the numpy
FM chain (data2g.fmchannel) and modem it stands in for."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data2g import config, constellation, fmchannel, modem  # noqa: E402
from data2g.channel_torch import bmi, llr  # noqa: E402
from data2g.config import DATA_SYMS_PER_FRAME, SubmodeSpec  # noqa: E402
from data2g.fmtorch import FmBurstChannel  # noqa: E402

SPEC = SubmodeSpec(0, "t", "ldpc", "gray-qam16", 1, band="w48")
PTS = constellation.load(SPEC.constellation)
DRIVE = {"mic": 1475.0, "flat": 2395.0}  # about the 15 kHz drives (default_drive), fixed: the test is the chain


def _cells(b, n_f, seed):
    nc = config.BANDS[SPEC.band].nc
    bits = np.random.default_rng(seed).integers(0, 2, (b, n_f * DATA_SYMS_PER_FRAME * nc * 4))
    cells = constellation.modulate(bits.ravel(), PTS).reshape(b, n_f, DATA_SYMS_PER_FRAME, nc)
    return bits, torch.tensor(cells)


@pytest.mark.parametrize("radio,cond", [("mic", "static"), ("flat", "mobile-urban")])
def test_chain_matches_numpy_fmchannel(radio, cond):
    """The same burst, seed and CNR: torch's output is numpy's to rounding."""
    ch = FmBurstChannel(SPEC, 2, radio, cond, drive_dev_hz=DRIVE[radio], dtype=torch.float64)
    x = ch.transmit(_cells(1, 2, 0)[1])
    y = ch.channel(x, 12.0, seeds=[5])[0].numpy()
    ref = fmchannel.apply_channel(x[0].numpy(), 12.0, cond, radio, 5, drive_dev_hz=DRIVE[radio])
    assert np.max(np.abs(y - ref)) < 1e-8 * np.sqrt(np.mean(ref**2))


def test_seeded_and_differentiable():
    ch = FmBurstChannel(SPEC, 2, "mic", drive_dev_hz=DRIVE["mic"])
    cells = _cells(2, 2, 1)[1].to(torch.complex64).requires_grad_()
    a = ch.channel(ch.transmit(cells), 10.0, torch.Generator().manual_seed(3))
    b = ch.channel(ch.transmit(cells), 10.0, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)
    a.pow(2).sum().backward()
    assert cells.grad is not None and float(cells.grad.abs().sum()) > 0


def test_bmi_agrees_with_the_numpy_receiver():
    """16-QAM, mic radio, static path, CNR 13 (scripts/fm_bmi_check.py: within
    0.01 bit per cell wherever numpy acquired every burst)."""
    n_f, cnr, pad = 4, 13.0, config.FS // 2
    bits, cells = _cells(3, n_f, 2)
    ch = FmBurstChannel(SPEC, n_f, "mic", drive_dev_hz=DRIVE["mic"], dtype=torch.float64)
    with torch.no_grad():
        x = torch.nn.functional.pad(ch.transmit(cells), (pad, pad))
        y, h, var = ch.receive(ch.channel(x, cnr, seeds=[0, 1, 2])[:, pad:-pad], points=PTS)
        lam = llr(y, h, var, torch.tensor(PTS))
        got = bmi(lam, torch.tensor(bits.reshape(lam.shape), dtype=torch.float64)).item()
    vals = []
    for i in range(3):
        xi = np.concatenate([np.zeros(pad), modem.modulate_bits(bits[i], SPEC), np.zeros(pad)])
        r = modem.receive(fmchannel.apply_channel(xi, cnr, "static", "mic", i, drive_dev_hz=DRIVE["mic"]),
                          known=dict(start=pad + config.LEADIN_SAMPLES, cfo=0.0, spec=SPEC, n_cw=n_f))
        lam_np = constellation.llr(modem.data_symbols(r["raw"]), r["est"]["h"], modem._data_var(r, r["est"]), PTS)
        vals.append(1 - np.mean(np.logaddexp(0, -(1 - 2 * bits[i].reshape(lam_np.shape)) * lam_np)) / np.log(2))
    assert abs(got - np.mean(vals)) < 0.005, (got, vals)
