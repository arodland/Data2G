"""Gear-shifter phase C3: train the link predictor's MLP (data2g/arq/predictor.py)
on runs/predictor_data.csv (scripts/predictor_data.py), Gaussian
negative log-likelihood per (band, constellation) target, missing targets
(the next burst failed sync) masked. Reports held-out NLL and MAE against
two baselines (persistence: the next MI is this burst's, same band only;
a global mean), and the calibration of P(decode) through the link
abstraction. Writes codes_data/link_predictor.npz; --onnx also exports and
checks the ONNX graph against the numpy forward pass.

    uv run python scripts/train_predictor.py runs/predictor_data.csv --onnx runs/link_predictor.onnx
"""

import argparse
import csv
import json

import numpy as np
import torch

from data2g.arq import predictor as P

HIDDEN = 32


def load(path):
    rows = list(csv.DictReader(open(path)))
    keys = ("snr_est", "spread", "delay_est", "mi_", "frames", "headroom")

    def prev(r):
        if not r.get("prev_band"):
            return None
        return ({k: float(r["prev_" + k]) for k in (k[5:] for k in r if k.startswith("prev_")) if k.startswith(keys)},
                r["prev_band"], float(r["prev_age"]))

    x = np.array([P.inputs({k: float(r[k]) for k in r if k.startswith(keys)},
                           r["band1"], float(r["gap"]), float(r["window"]), prev(r)) for r in rows])
    y = np.array([[float(r[t]) if r[t] not in ("", "nan") else np.nan for t in P.TARGETS] for r in rows])
    return x, y, rows


class Net(torch.nn.Module):
    def __init__(self, n_in, n_out, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.l1 = torch.nn.Linear(n_in, HIDDEN)
        self.l2 = torch.nn.Linear(HIDDEN, HIDDEN)
        self.l3 = torch.nn.Linear(HIDDEN, 2 * n_out)
        self.n_out = n_out

    def forward(self, x):
        h = torch.tanh(self.l1((x - self.mean) / self.std))
        h = torch.tanh(self.l2(h))
        h = self.l3(h)
        # persistence skip: this burst's MI for the target's constellation,
        # corrected in logit space (zero correction = persistence)
        base = torch.logit(x[..., P.MI_COLS].clamp(1e-3, 1 - 1e-3))
        mu = torch.sigmoid(base + h[..., : self.n_out])
        sigma = torch.exp(torch.clamp(h[..., self.n_out:], -6, 1))
        return mu, sigma


def nll(mu, sigma, y):
    m = ~torch.isnan(y)
    yy = torch.where(m, y, mu.detach())
    v = 0.5 * ((yy - mu) / sigma) ** 2 + torch.log(sigma)
    return (v * m).sum() / m.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--out", default=str(P.DATA / "link_predictor.npz"))
    ap.add_argument("--onnx")
    a = ap.parse_args()
    torch.manual_seed(0)
    x, y, rows = load(a.data)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(x))
    n_test = len(x) // 5
    te, tr = idx[:n_test], idx[n_test:]
    mean, std = x[tr].mean(0), x[tr].std(0) + 1e-6
    net = Net(x.shape[1], y.shape[1], mean, std)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    xt, yt = torch.tensor(x[tr], dtype=torch.float32), torch.tensor(y[tr], dtype=torch.float32)
    xv, yv = torch.tensor(x[te], dtype=torch.float32), torch.tensor(y[te], dtype=torch.float32)
    for ep in range(a.epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            loss = nll(*net(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 50 == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                print(f"epoch {ep:4d} train {loss.item():.3f} held-out NLL {nll(*net(xv), yv).item():.3f}", flush=True)

    with torch.no_grad():
        mu, sigma = (t.numpy() for t in net(xv))
    yte = y[te]
    m = ~np.isnan(yte)
    mae = np.abs(mu - yte)[m].mean()
    # baselines: persistence (same band: this burst's MI), global target means
    pers = np.full_like(yte, np.nan)
    for i, j in enumerate(te):
        b1 = rows[j]["band1"]
        for k, t in enumerate(P.TARGETS):
            _, b, c = t.split("_", 2)
            if b == b1:
                pers[i, k] = float(rows[j][f"mi_{c}"])
    mp = m & ~np.isnan(pers)
    print(f"held-out MAE: model {mae:.4f}, same-band subset model {np.abs(mu - yte)[mp].mean():.4f} "
          f"vs persistence {np.abs(pers - yte)[mp].mean():.4f}; global mean {np.abs(np.nanmean(y[tr], 0) - yte)[m].mean():.4f}")
    kinds = np.array([rows[j]["kind"] for j in te])
    for k in sorted(set(kinds)):
        sel = (kinds == k)[:, None] & mp
        print(f"   {k:6s} same-band MAE model {np.abs(mu - yte)[sel].mean():.4f} vs persistence {np.abs(pers - yte)[sel].mean():.4f}")

    # calibration of P(decode): prediction, persistence and truth all through
    # predictor.mode_mi (the candidate's clip noise and the receiver's
    # estimation loss) and the submode's abstraction curve
    ab = json.load(open(P.DATA / "link_abstraction.json"))
    dop = np.array([float(rows[j]["doppler"]) for j in te])
    sp = np.array([float(rows[j]["spread_est"]) for j in te])
    pred, truth, ppers, tpers, pmod = [], [], [], [], []
    for s in P.SUBMODES.values():
        if s.name not in ab:
            continue
        k = P.TARGETS.index(f"next_{s.band}_{P.const_family(s.constellation)}")
        c = ab[s.name]
        sig = lambda v: 1 / (1 + np.exp(-c["slope"] * (v - c["mi50"])))  # noqa: E731
        ok = m[:, k]
        for i in np.flatnonzero(ok):
            lo, mid, hi = P.mode_mi([mu[i, k] - sigma[i, k], mu[i, k], mu[i, k] + sigma[i, k]], s, sp[i])
            pr = P.p_decode(float(mid), float(max(hi - lo, 1e-6) / 2), c["slope"], c["mi50"])
            tr = float(sig(P.mode_mi(yte[i, k], s, dop[i])))
            pred.append(pr)
            truth.append(tr)
            if not np.isnan(pers[i, k]):
                ppers.append(float(sig(P.mode_mi(pers[i, k], s, sp[i]))))
                tpers.append(tr)
                pmod.append(pr)
    pred, truth = np.array(pred), np.array(truth)
    ppers, tpers, pmod = np.array(ppers), np.array(tpers), np.array(pmod)
    print("calibration (predicted P bin: mean predicted, mean actual, n):")
    for lo in np.arange(0, 1, 0.1):
        sel = (pred >= lo) & (pred < lo + 0.1)
        if sel.any():
            print(f"  {lo:.1f}-{lo + 0.1:.1f}: {pred[sel].mean():.3f} {truth[sel].mean():.3f} {sel.sum()}")
    print(f"Brier {np.mean((pred - truth) ** 2):.4f} (all bands); same-band pairs: model "
          f"{np.mean((pmod - tpers) ** 2):.4f} vs persistence {np.mean((ppers - tpers) ** 2):.4f}")

    layers = [net.l1, net.l2, net.l3]
    np.savez(a.out, mean=mean, std=std, **{f"W{i}": l.weight.detach().numpy().T.astype(np.float64) for i, l in enumerate(layers)},
             **{f"b{i}": l.bias.detach().numpy().astype(np.float64) for i, l in enumerate(layers)})
    P.model.cache_clear()
    mu_np, _ = P.model(a.out)(x[te])
    print(f"numpy runtime vs torch: max |d mu| {np.abs(mu_np - mu).max():.2e}")
    if a.onnx:
        import onnxruntime as ort

        torch.onnx.export(net, torch.zeros(1, x.shape[1]), a.onnx, input_names=["x"], output_names=["mu", "sigma"],
                          dynamic_axes={"x": {0: "n"}})
        sess = ort.InferenceSession(a.onnx)
        mu_ox, _ = sess.run(None, {"x": x[te].astype(np.float32)})
        print(f"ONNX {a.onnx} vs torch: max |d mu| {np.abs(mu_ox - mu).max():.2e}")


if __name__ == "__main__":
    main()
