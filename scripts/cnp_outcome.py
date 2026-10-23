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


def train_labels(a):
    """The CNP on the outcomes themselves: each row queries the mode it tried
    (train_outcome's loss: the burst's usability, its codewords binomial),
    early stopping on held-out sessions."""
    x, mode, bok, dsent, dok, rows, bmask = T.load(a.data, energy=True)
    secs = x[:, SECS_COL].copy()
    xe = np.delete(x, SECS_COL, axis=1)
    desc = np.array([descriptor(m) for m in T.MODES], dtype=np.float64)
    dmean, dstd = desc.mean(0), desc.std(0) + 1e-6
    hold = [m for m in a.hold.split(",") if m]
    seeds = np.array([int(r["seed"]) for r in rows])
    us = np.unique(seeds)
    te = np.isin(seeds, np.random.default_rng(0).choice(us, len(us) // 5, replace=False))
    rest = np.setdiff1d(us, seeds[te])
    va = np.isin(seeds, np.random.default_rng(1).choice(rest, len(rest) // 8, replace=False))
    trn = ~te & ~va & ~np.isin(mode, [T.MODES.index(m) for m in hold])
    mean, std = xe[trn].mean(0), xe[trn].std(0) + 1e-6
    dev = torch.device(a.device)
    Tt = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt, device=dev)  # noqa: E731
    D = Tt((desc - dmean) / dstd)
    torch.manual_seed(a.seed)
    net = CNP(xe.shape[1], desc.shape[1], mean, std, a.enc, a.z, a.head).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    def loss(sel_t):
        X, S, M, B, DS, DO = sel_t
        z = net.head(torch.cat([net.enc((X - net.mean) / net.std), D[M], S[:, None]], -1))
        lb = torch.nn.functional.binary_cross_entropy_with_logits(z[:, 0], B)
        w = B * (DS > 0)
        lc = -(w * (DO * torch.nn.functional.logsigmoid(z[:, 1]) + (DS - DO) * torch.nn.functional.logsigmoid(-z[:, 1]))).mean()
        return lb + lc

    pack = lambda s: [Tt(xe[s]), Tt(secs[s]), Tt(mode[s], torch.long), Tt(bok[s]), Tt(dsent[s]), Tt(dok[s])]  # noqa: E731
    tr, vl, ts = pack(trn), pack(va), pack(te)
    best, state = float("inf"), None
    for ep in range(a.epochs):
        perm = torch.randperm(len(tr[0]), device=dev)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            l = loss([t[b] for t in tr])
            opt.zero_grad()
            l.backward()
            opt.step()
        with torch.no_grad():
            v = loss(vl).item()
        if v < best:
            best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
        if ep % 10 == 0:
            print(f"epoch {ep:3d} train {l.item():.4f} validation {v:.4f} (best {best:.4f})", flush=True)
    net.load_state_dict(state)
    with torch.no_grad():
        print(f"best validation {best:.4f}; held-out test {loss(ts).item():.4f}")
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    np.savez(a.out, kind=np.array("cnp"), modes=np.array(T.MODES), bands=np.array(T.BANDS), desc=desc, dmean=dmean,
             dstd=dstd, secs_col=np.array(SECS_COL), energy_inputs=np.array(True), hold=np.array(hold),
             **{f"p_{k}": v for k, v in sd.items()})


def redescribe(src: str, out: str):
    """`src`'s CNP with every mode's descriptor row recomputed (descriptor:
    MODE_THRESHOLDS as they are now), weights and normalization kept: a
    mode's re-measured ladder reaches the model without a retrain."""
    d = dict(np.load(src))
    assert str(d["kind"]) == "cnp", src
    modes = [str(m) for m in d["modes"]]
    new = np.array([descriptor(m) for m in modes], dtype=d["desc"].dtype)
    print("changed:", [m for m, a, b in zip(modes, new, d["desc"]) if not np.array_equal(a, b)])
    np.savez(out, **dict(d, desc=new))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?")
    ap.add_argument("--redescribe", metavar="NPZ", help="rewrite NPZ's descriptors (no training) into --out")
    ap.add_argument("--teacher", default=str(P.DATA / "outcome_predictor.npz"))
    ap.add_argument("--hold", default="", help="modes left out of training (zero-shot test)")
    ap.add_argument("--cond", default="", help="policy.COND_OFFSETS' spec, applied to the teacher's targets")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--enc", type=int, default=512)
    ap.add_argument("--z", type=int, default=256)
    ap.add_argument("--head", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--labels", action="store_true", help="train on the rows' outcomes (the tried mode's), not a teacher")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.redescribe:
        return redescribe(a.redescribe, a.out)
    if a.labels:
        return train_labels(a)
    x, *_, rows, _ = T.load(a.data, energy=True)
    teach = P.outcome_model(a.teacher)
    assert tuple(teach.modes) == T.MODES
    z_t = np.concatenate([teach(x[i:i + 4096]) for i in range(0, len(x), 4096)])  # (DATA2G_OUTCOME_LCB applies)
    n = len(T.MODES)
    if a.cond:  # the shifter's conditional offsets (policy.COND_OFFSETS), on each row's own spread_est
        spread = np.array([float(r["spread_est"]) for r in rows])
        snr_est = np.array([float(r["snr_est"]) for r in rows])
        for e in a.cond.split(";"):
            lo, hi, m, off = e.split(":")
            sel = (spread >= float(lo)) & (snr_est < float(hi))
            z_t[sel, T.MODES.index(m)] += float(off)
            print(f"cond {e}: {sel.mean():.1%} of rows")
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
