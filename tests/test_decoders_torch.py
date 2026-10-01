import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data2g import decoders_torch, ldpc, polar  # noqa: E402


@pytest.mark.parametrize("alpha", [None, 0.8])
def test_ldpc_numpy_matches_torch(alpha):
    code = ldpc.qc_code(500, 1000)
    rng = np.random.default_rng(1)
    x = 1.0 - 2.0 * code.encode(rng.integers(0, 2, (16, 500)))
    llr = (2 * (x + rng.normal(scale=0.9, size=x.shape)) / 0.81).astype(np.float32)
    out, ok, post = ldpc.MinSumDecoder(code).decode(llr, iters=40, alpha=alpha, posterior=True)
    t_out, t_ok, t_post = decoders_torch.MinSumDecoder(code).decode(torch.tensor(llr), iters=40, alpha=alpha,
                                                                    posterior=True)
    assert 0 < ok.sum() < len(ok)  # a mix of converged and failed decodes
    assert np.array_equal(ok, t_ok.numpy())
    assert np.array_equal(out[ok], t_out.numpy()[ok])
    # failed decodes oscillate, so float32 summation order alone sends them
    # apart: compare the converged ones
    post, t_post = post[ok], t_post.numpy()[ok]
    assert np.array_equal(np.sign(post), np.sign(t_post))
    # past ~20 phi runs on float32 tanh's last ULPs, where numpy's and
    # torch's libm differ by up to ~1.4 (and the value no longer matters)
    small = np.abs(t_post) < 20
    np.testing.assert_allclose(post[small], t_post[small], atol=1e-2)


def test_scl_numpy_matches_torch():
    code = polar.PolarCode(48, 480, design_snr_db=-4.0)
    rng = np.random.default_rng(2)
    x = 1.0 - 2.0 * code.encode(rng.integers(0, 2, (32, 48)))
    llr = (2 * (x + rng.normal(scale=1.0, size=x.shape))).astype(np.float32)
    u, pm = polar.SCLDecoder(code).decode(llr)
    t_u, t_pm = decoders_torch.SCLDecoder(code).decode(torch.tensor(llr))
    assert np.array_equal(u[:, 0], t_u.numpy()[:, 0])
    np.testing.assert_allclose(pm, t_pm.numpy(), rtol=1e-4, atol=1e-4)
