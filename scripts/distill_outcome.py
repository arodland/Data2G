"""Can one outcome model do what the three-model router does?

Teacher: the shipped router (policy.GearShifter.gate): the gate's model where
spread_est < GATE_SPREAD_HZ and snr_est < GATE_SNR_DB, the fast region's
where spread_est >= FAST_SPREAD_HZ and snr_est in [FAST_SNR_LO_DB,
FAST_SNR_HI_DB), the installed one elsewhere, LOGIT_OFFSETS left out (the
judge adds them). The router reads the median spread_est of the peer's last
GATE_HIST bursts; rows hold one burst's, so the teacher routes on the row's
own here. Targets: every mode's P(burst usable) and P(codeword | usable) on
every row's inputs (dense soft targets, not one tried mode per row).

Student: train_outcome.Net (tanh MLP, energy inputs) on those targets, by
width and depth. Held out (20% of samples): the gap to the teacher per
region, as mean |dP| of the joint P(codeword) and as how often the student's
best mode by expected rate (P x payload rate) differs from the teacher's.
A small net that fits the teacher everywhere says the single-model retrains
failed on their labels and data, not on capacity.

    python -m scripts.distill_outcome runs/session_data_gt.csv,... --hidden 64 128 256 --depth 2 3 \\
        --out runs/distill
"""

from . import session_data as SD  # noqa: I001 (first: it sets the thread caps before numpy loads)

import argparse

import numpy as np
import torch

from data2g.arq import predictor as P

from . import train_outcome as T


def teacher(x_e, x_n, spread, snr):
    """(teacher logits (rows, 2n) in T.MODES' order, region per row: 0 installed, 1 gate, 2 fast)."""
    models = [P.outcome_model(str(p)) for p in (P.DATA / "outcome_predictor.npz", P.GATE_MODEL, P.FAST_MODEL)]
    for m in models:
        assert tuple(m.modes) == T.MODES and tuple(m.bands) == T.BANDS
    region = np.zeros(len(snr), dtype=int)
    region[(spread < P.GATE_SPREAD_HZ) & (snr < P.GATE_SNR_DB)] = 1
    region[(spread >= P.FAST_SPREAD_HZ) & (snr >= P.FAST_SNR_LO_DB) & (snr < P.FAST_SNR_HI_DB)] = 2
    z = np.empty((len(snr), 2 * len(T.MODES)))
    for k, m in enumerate(models):
        s = region == k
        if s.any():
            z[s] = m(x_e[s] if m.energy else x_n[s])
    return z, region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="csv, or several comma-separated")
    ap.add_argument("--hidden", type=int, nargs="+", default=[64])
    ap.add_argument("--depth", type=int, nargs="+", default=[2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="prefix: <out>_h<hidden>d<depth>.npz")
    ap.add_argument("--compare", nargs="*", default=[], help="model npz files scored as students (whole replacements)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    x_e, *_, rows, _ = T.load(a.data, energy=True)
    x_n = T.load(a.data)[0]
    spread = np.array([float(r["spread_est"]) for r in rows])
    snr = np.array([float(r["snr_est"]) for r in rows])
    z_t, region = teacher(x_e, x_n, spread, snr)
    p_t = 1 / (1 + np.exp(-np.clip(z_t, -40, 40)))
    n = len(T.MODES)
    rate = np.array([SD.rate(m) for m in T.MODES])
    print(f"{len(rows)} rows; regions installed/gate/fast: {np.bincount(region, minlength=3)}", flush=True)

    seeds = np.array([int(r["seed"]) for r in rows])
    us = np.unique(seeds)
    te = np.isin(seeds, np.random.default_rng(0).choice(us, len(us) // 5, replace=False))
    trn = ~te
    mean, std = x_e[trn].mean(0), x_e[trn].std(0) + 1e-6
    dev = torch.device(a.device)
    X = torch.tensor(x_e[trn], dtype=torch.float32, device=dev)
    Y = torch.tensor(p_t[trn], dtype=torch.float32, device=dev)
    Xte = torch.tensor(x_e[te], dtype=torch.float32, device=dev)
    pj_t = p_t[te, :n] * p_t[te, n:]
    best_t = np.argmax(pj_t * rate, axis=1)
    print("\nhidden depth | per region (installed, gate, fast, all): mean |dP(codeword)|, best mode differs")

    def report(tag, z_s):
        p_s = 1 / (1 + np.exp(-np.clip(z_s, -40, 40)))
        pj_s = p_s[:, :n] * p_s[:, n:]
        dp = np.abs(pj_s - pj_t).mean(axis=1)
        differs = np.argmax(pj_s * rate, axis=1) != best_t
        cells = []
        for k in (0, 1, 2, None):
            s = region[te] == k if k is not None else np.ones(te.sum(), bool)
            cells.append(f"{dp[s].mean():.4f} {differs[s].mean():5.1%}")
        print(f"{tag:>12s} | " + " | ".join(cells), flush=True)

    for pth in a.compare:  # whole-model replacements, for scale
        m = P.outcome_model(pth)
        report(pth.split("/")[-1][:12], m(x_e[te] if m.energy else x_n[te]))
    for h in a.hidden:
        for d in a.depth:
            torch.manual_seed(a.seed)
            net = T.Net(x_e.shape[1], n, mean, std, h, d).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=2e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
            for ep in range(a.epochs):
                perm = torch.randperm(len(X), device=dev)
                for i in range(0, len(perm), 1024):
                    b = perm[i:i + 1024]
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X[b]), Y[b])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                sched.step()
            with torch.no_grad():
                report(f"h{h} d{d}", net(Xte).cpu().numpy())
            layers = [layer.cpu() for layer in net.layers]
            out = f"{a.out}_h{h}d{d}.npz"
            np.savez(out, mean=mean, std=std, modes=np.array(T.MODES), bands=np.array(T.BANDS),
                     noise_inputs=np.array(False), link_inputs=np.array(False), energy_inputs=np.array(True),
                     **{f"W{i}": l.weight.detach().numpy().T.astype(np.float64) for i, l in enumerate(layers)},
                     **{f"b{i}": l.bias.detach().numpy().astype(np.float64) for i, l in enumerate(layers)})


if __name__ == "__main__":
    main()
