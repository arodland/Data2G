"""A conditional-neural-process outcome model: an encoder of the measured
conditions, run once, and a query head that takes each mode's descriptor
(what the mode is, and its measured ladder thresholds) and the burst's
length. A mode added or tweaked needs only its descriptor row.

Trained by distillation from a teacher outcome model (default the installed
one): every mode's P(burst usable) and P(codeword | usable) on every row's
inputs, at that row's burst length. `--hold` modes are left out of training
entirely: their held-out fit says whether the descriptors carry a new mode
without a retrain.

    python -m scripts.cnp_outcome <csvs> --hold n10-16qam-r2/3,w48-64l-r2/3 --out runs/cnp.npz
"""

from . import session_data as SD  # noqa: I001 (first: it sets the thread caps before numpy loads)

import argparse

import numpy as np
import torch

from data2g import codes, constellation
from data2g.arq import policy as G
from data2g.arq import predictor as P
from data2g.config import BANDS, SUBMODES

from . import train_outcome as T

NEVER_DB = 40.0  # a threshold the ladder never reached, as a number


def descriptor(name: str) -> list:
    """What the query head knows of a mode."""
    s = SD.MODES[name]
    th = G.MODE_THRESHOLDS[name]
    d = [min(t, NEVER_DB) / 10 for t in th] + [float(t >= 90) for t in th]
    is_cpm = name not in SUBMODES
    if is_cpm:
        tones, rate = int(s.grid[1:].split("r")[0]), float(s.grid.split("r")[1])
        bits, nc = np.log2(tones), 0.0
        band = s.grid
    else:
        bits, nc = constellation.bits_per_symbol(constellation.load(s.constellation)), BANDS[s.band].nc
        tones, rate, band = 0, 0.0, s.band
    d += [s.k / s.coded_bits, bits / 8, np.log2(codes.payload_bytes(s)), s.frames_per_cw / 16, nc / 48, float(is_cpm),
          np.log2(1 + tones) / 6, rate / 100, float("ack" in name or "polar" in name)]
    d += [float(band == b) for b in T.BANDS]
    return d


class CNP(torch.nn.Module):
    def __init__(self, n_in, n_desc, mean, std, enc=512, z=256, head=256):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.enc = torch.nn.Sequential(torch.nn.Linear(n_in, enc), torch.nn.Tanh(), torch.nn.Linear(enc, enc),
                                       torch.nn.Tanh(), torch.nn.Linear(enc, z), torch.nn.Tanh())
        self.head = torch.nn.Sequential(torch.nn.Linear(z + n_desc + 1, head), torch.nn.Tanh(),
                                        torch.nn.Linear(head, head), torch.nn.Tanh(), torch.nn.Linear(head, 2))

    def forward(self, x, secs, desc):
        """x (B, n_in) without the burst length, secs (B,) log2 seconds, desc (M, n_desc) -> (B, M, 2) logits."""
        z = self.enc((x - self.mean) / self.std)
        B, M = len(x), len(desc)
        q = torch.cat([z[:, None].expand(B, M, -1), desc[None].expand(B, M, -1), secs[:, None, None].expand(B, M, 1)], -1)
        return self.head(q)


SECS_COL = 27  # outcome_inputs' log2(seconds) column (after gap); the encoder gets the rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--teacher", default=str(P.DATA / "outcome_predictor.npz"))
    ap.add_argument("--hold", default="", help="modes left out of training (zero-shot test)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--enc", type=int, default=512)
    ap.add_argument("--z", type=int, default=256)
    ap.add_argument("--head", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    x, *_, rows, _ = T.load(a.data, energy=True)
    teach = P.outcome_model(a.teacher)
    assert tuple(teach.modes) == T.MODES
    z_t = teach(x)
    n = len(T.MODES)
    assert np.allclose(x[:200, SECS_COL], [np.log2(float(r["cand_seconds"])) for r in rows[:200]])
    secs = x[:, SECS_COL].copy()
    xe = np.delete(x, SECS_COL, axis=1)
    hold = [m for m in a.hold.split(",") if m]
    keep = np.array([m not in hold for m in T.MODES])
    desc = np.array([descriptor(m) for m in T.MODES], dtype=np.float64)
    dmean, dstd = desc.mean(0), desc.std(0) + 1e-6
    descn = (desc - dmean) / dstd
    seeds = np.array([int(r["seed"]) for r in rows])
    us = np.unique(seeds)
    te = np.isin(seeds, np.random.default_rng(0).choice(us, len(us) // 5, replace=False))
    trn = ~te
    mean, std = xe[trn].mean(0), xe[trn].std(0) + 1e-6
    dev = torch.device(a.device)
    Tt = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)  # noqa: E731
    X, S = Tt(xe[trn]), Tt(secs[trn])
    # targets (B, M, 2): P(usable), P(codeword) per mode
    Y = Tt(np.stack([1 / (1 + np.exp(-z_t[trn, :n])), 1 / (1 + np.exp(-z_t[trn, n:]))], -1))
    Dk = Tt(descn[keep])
    Yk = Y[:, torch.tensor(np.flatnonzero(keep), device=dev)]
    torch.manual_seed(1)
    net = CNP(xe.shape[1], desc.shape[1], mean, std, a.enc, a.z, a.head).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    for ep in range(a.epochs):
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X[b], S[b], Dk), Yk[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep:3d} loss {loss.item():.4f}", flush=True)
    with torch.no_grad():
        zs = torch.cat([net(Tt(xe[te][i:i + 4096]), Tt(secs[te][i:i + 4096]), Tt(descn)) for i in range(0, te.sum(), 4096)]).cpu().numpy()
    pt = 1 / (1 + np.exp(-z_t[te]))
    ps = 1 / (1 + np.exp(-zs))
    pj_t, pj_s = pt[:, :n] * pt[:, n:], ps[..., 0] * ps[..., 1]
    err = np.abs(pj_s - pj_t).mean(0)
    print(f"held-out rows: mean |dP(codeword)| against the teacher, trained modes {err[keep].mean():.4f}")
    for m in hold:
        k = T.MODES.index(m)
        print(f"  held-out mode {m:16s} {err[k]:.4f}  (teacher mean P {pj_t[:, k].mean():.3f}, cnp {pj_s[:, k].mean():.3f})")
    worst = np.argsort(err)[::-1][:5]
    print("  worst trained:", ", ".join(f"{T.MODES[k]} {err[k]:.4f}" for k in worst if keep[k]))
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    np.savez(a.out, kind=np.array("cnp"), modes=np.array(T.MODES), bands=np.array(T.BANDS), desc=desc, dmean=dmean,
             dstd=dstd, secs_col=np.array(SECS_COL), energy_inputs=np.array(True), hold=np.array(hold),
             **{f"p_{k}": v for k, v in sd.items()})


if __name__ == "__main__":
    main()
