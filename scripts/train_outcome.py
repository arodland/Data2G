"""Train the outcome link predictor (data2g.arq.predictor.predict_outcome)
on scripts/outcome_data.py's rows: per submode, a logit for P(burst usable)
and one for P(codeword decodes | usable), from the receiver's measurements.
Loss: cross-entropy on the candidate each row tried (the burst's outcome;
its data codewords, binomial, only where the burst was usable). A tanh MLP
in numpy at runtime. Reports held-out reliability by channel and SNR.

    uv run python scripts/train_outcome.py runs/outcome_data.csv
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
import torch

from data2g.arq import predictor as P

HIDDEN = 64
MEAS = ("snr_est", "spread_est", "delay_est_ms", "headroom", "frames") + tuple(f"mi_{c}" for c in P.CONSTS)


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r["cand"] in P.OUTCOME_MODES]
    x, mode, bok, dsent, dok = [], [], [], [], []
    for r in rows:
        m = {k: float(r[k]) for k in MEAS}
        prev = None
        if r["prev_band"]:
            prev = ({k: float(r["prev_" + k]) for k in MEAS}, r["prev_band"], float(r["prev_age"]))
        x.append(P.outcome_inputs(m, r["band"], float(r["gap"]), float(r["cand_seconds"]), prev))
        mode.append(P.OUTCOME_MODES.index(r["cand"]))
        bok.append(float(r["burst_ok"]))
        dsent.append(float(r["data_sent"]))
        dok.append(float(r["data_ok"]))
    return (np.array(x), np.array(mode), np.array(bok), np.array(dsent), np.array(dok), rows)


class Net(torch.nn.Module):
    def __init__(self, n_in, n_modes, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.l1 = torch.nn.Linear(n_in, HIDDEN)
        self.l2 = torch.nn.Linear(HIDDEN, HIDDEN)
        self.l3 = torch.nn.Linear(HIDDEN, 2 * n_modes)

    def forward(self, x):
        h = torch.tanh(self.l1((x - self.mean) / self.std))
        return self.l3(torch.tanh(self.l2(h)))


def loss_fn(z, mode, bok, dsent, dok, n):
    zb = z.gather(1, mode[:, None])[:, 0]
    zc = z.gather(1, (mode + n)[:, None])[:, 0]
    lb = torch.nn.functional.binary_cross_entropy_with_logits(zb, bok, reduction="sum")
    # codewords given a usable burst: binomial log-likelihood, dok of dsent
    w = bok * (dsent > 0)
    lc = -(w * (dok * torch.nn.functional.logsigmoid(zc) + (dsent - dok) * torch.nn.functional.logsigmoid(-zc))).sum()
    return (lb + lc) / len(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", default=str(P.DATA / "outcome_predictor.npz"))
    a = ap.parse_args()
    torch.manual_seed(0)
    x, mode, bok, dsent, dok, rows = load(a.data)
    n = len(P.OUTCOME_MODES)
    # held out by sample (a sample's candidates share its measurements)
    seeds = np.array([int(r["seed"]) for r in rows])
    us = np.unique(seeds)
    test_seeds = set(np.random.default_rng(0).choice(us, len(us) // 5, replace=False))
    te = np.array([s in test_seeds for s in seeds])
    # early stopping on a validation split of the training samples (the
    # model overfits: held-out loss bottomed near epoch 50 of 300)
    rest = np.setdiff1d(us, list(test_seeds))
    val_seeds = set(np.random.default_rng(1).choice(rest, len(rest) // 8, replace=False))
    va = np.array([s in val_seeds for s in seeds])
    trn = ~te & ~va
    mean, std = x[trn].mean(0), x[trn].std(0) + 1e-6
    net = Net(x.shape[1], n, mean, std)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    T = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt)  # noqa: E731
    tr_t = [T(x[trn]), T(mode[trn], torch.long), T(bok[trn]), T(dsent[trn]), T(dok[trn])]
    va_t = [T(x[va]), T(mode[va], torch.long), T(bok[va]), T(dsent[va]), T(dok[va])]
    best_val, best_state = float("inf"), None
    te_t = [T(x[te]), T(mode[te], torch.long), T(bok[te]), T(dsent[te]), T(dok[te])]
    for ep in range(a.epochs):
        perm = torch.randperm(len(tr_t[0]))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = loss_fn(net(tr_t[0][b]), *(t[b] for t in tr_t[1:]), n)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                v = loss_fn(net(va_t[0]), *va_t[1:], n).item()
            if v < best_val:
                best_val, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
            if ep % 25 == 0:
                print(f"epoch {ep:4d} train {loss.item():.4f} validation {v:.4f} (best {best_val:.4f})", flush=True)
    net.load_state_dict(best_state)
    with torch.no_grad():
        print(f"best validation {best_val:.4f}; held-out test {loss_fn(net(te_t[0]), *te_t[1:], n).item():.4f}")
    with torch.no_grad():
        z = net(te_t[0]).numpy()
    idx = np.arange(len(z))
    pb = 1 / (1 + np.exp(-z[idx, mode[te]]))
    pc = 1 / (1 + np.exp(-z[idx, mode[te] + n]))
    # reliability of the joint per-codeword P (usable x codeword) against decoded fraction
    pj = pb * pc
    frac = np.where(dsent[te] > 0, dok[te] / np.maximum(dsent[te], 1), bok[te])
    print("\nheld-out reliability, P(burst usable) vs usable, and P(codeword) vs decoded fraction:")
    for lo in np.arange(0, 1, 0.1):
        s = (pb >= lo) & (pb < lo + 0.1)
        s2 = (pj >= lo) & (pj < lo + 0.1)
        print(f"  {lo:.1f}-{lo + 0.1:.1f}: burst {pb[s].mean() if s.any() else 0:.3f} vs {bok[te][s].mean() if s.any() else 0:.3f}"
              f" (n {s.sum():5d})   codeword {pj[s2].mean() if s2.any() else 0:.3f} vs {frac[s2].mean() if s2.any() else 0:.3f} (n {s2.sum():5d})")
    tr_rows = [rows[i] for i in np.flatnonzero(te)]
    g = defaultdict(list)
    for i, r in enumerate(tr_rows):
        snr = float(r["snr"])
        g[(r["kind"], "<0" if snr < 0 else "0-10" if snr < 10 else ">=10")].append(i)
    print("\nby channel and SNR: mean P(codeword) vs decoded fraction, Brier")
    for k in sorted(g):
        ii = np.array(g[k])
        print(f"  {k[0]:6s} {k[1]:5s} n {len(ii):5d}: {pj[ii].mean():.3f} vs {frac[ii].mean():.3f}   "
              f"Brier {np.mean((pj[ii] - frac[ii]) ** 2):.4f}")
    layers = [net.l1, net.l2, net.l3]
    np.savez(a.out, mean=mean, std=std, **{f"W{i}": l.weight.detach().numpy().T.astype(np.float64) for i, l in enumerate(layers)},
             **{f"b{i}": l.bias.detach().numpy().astype(np.float64) for i, l in enumerate(layers)})
    P.outcome_model.cache_clear()
    zn = P.outcome_model(a.out)(x[te])
    print(f"numpy runtime vs torch: max |d logit| {np.abs(zn - z).max():.2e}")


if __name__ == "__main__":
    main()
