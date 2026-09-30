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

from data2g import cpm
from data2g.config import SUBMODES
from data2g.arq import policy as G
from data2g.arq import predictor as P

HIDDEN = 64
MODES = tuple(m for m in P.OUTCOME_MODES + tuple(cpm.SPECS) if m not in G.DROP)  # DATA2G_DROP_MODES: a pruned set
SUBMODES_BAND = {m: s.sync_band for m, s in SUBMODES.items()}  # the model's outputs, saved with it
BANDS = P.BANDS + tuple(cpm.GRIDS)
MEAS = ("snr_est", "spread_est", "delay_est_ms", "headroom", "frames") + tuple(f"mi_{c}" for c in P.CONSTS)


def load(path, stale_header=()):
    """`stale_header`: data from before the header copy (PROTOCOL_VERSION
    11): its w/w48 candidates' burst labels are not today's (their codeword
    labels, given a usable burst, still are)."""
    rows = []
    for pth in path.split(","):
        stale = pth in stale_header
        rows += [dict(r, _bmask=0.0 if stale and SUBMODES_BAND.get(r["cand"]) in ("w", "w48") else 1.0)
                 for r in csv.DictReader(open(pth)) if r["cand"] in MODES]
    x, mode, bok, dsent, dok, bmask = [], [], [], [], [], []
    for r in rows:
        m = {k: float(r[k]) for k in MEAS}
        prev = None
        if r["prev_band"]:
            prev = ({k: float(r["prev_" + k]) for k in MEAS}, r["prev_band"], float(r["prev_age"]))
        x.append(P.outcome_inputs(m, r["band"], float(r["gap"]), float(r["cand_seconds"]), prev, BANDS))
        mode.append(MODES.index(r["cand"]))
        bok.append(float(r["burst_ok"]))
        dsent.append(float(r["data_sent"]))
        dok.append(float(r["data_ok"]))
        bmask.append(r["_bmask"])
    return (np.array(x), np.array(mode), np.array(bok), np.array(dsent), np.array(dok), rows, np.array(bmask))


class Net(torch.nn.Module):
    def __init__(self, n_in, n_modes, mean, std, hidden=HIDDEN, depth=2):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        sizes = [n_in] + [hidden] * depth + [2 * n_modes]
        self.layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))

    def forward(self, x):
        h = (x - self.mean) / self.std
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.layers[-1](h)


def loss_fn(z, mode, bok, dsent, dok, bmask, n):
    zb = z.gather(1, mode[:, None])[:, 0]
    zc = z.gather(1, (mode + n)[:, None])[:, 0]
    lb = (bmask * torch.nn.functional.binary_cross_entropy_with_logits(zb, bok, reduction="none")).sum()
    # codewords given a usable burst: binomial log-likelihood, dok of dsent
    w = bok * (dsent > 0)
    lc = -(w * (dok * torch.nn.functional.logsigmoid(zc) + (dsent - dok) * torch.nn.functional.logsigmoid(-zc))).sum()
    return (lb + lc) / len(mode)


def combine(paths, out):
    """Members' npz files -> one file (keys m<i>_<key>), which
    predictor.outcome_model loads as an ensemble."""
    arrays = {}
    for i, pth in enumerate(paths):
        d = np.load(pth)
        arrays.update({f"m{i}_{k}": d[k] for k in d.files})
    np.savez(out, **arrays)
    P.outcome_model.cache_clear()
    print(f"{out}: {len(paths)} members")


def extend(base: str, data: str, new: list, out: str, epochs: int = 300):
    """Output units for `new` modes added to an installed model (each
    ensemble member), trained on `data`'s rows for those modes; every
    other weight frozen, so the modes it knew predict exactly as before.
    Each member is fitted on its own bootstrap of the samples."""
    d = np.load(base)
    tags = sorted({k.split("_", 1)[0] for k in d.files}) if "m0_mean" in d.files else [""]
    x, mode, bok, dsent, dok, rows, bmask = load(data)
    keep = np.isin([MODES[i] for i in mode], new)
    x, bok, dsent, dok, bmask = x[keep], bok[keep], dsent[keep], dok[keep], bmask[keep]
    mode = np.array([new.index(MODES[i]) for i in mode[keep]])
    seeds = np.array([int(r["seed"]) for r, k in zip(rows, keep) if k])
    us = np.unique(seeds)
    te = np.isin(seeds, np.random.default_rng(0).choice(us, len(us) // 5, replace=False))
    rest = np.setdiff1d(us, seeds[te])
    va = np.isin(seeds, np.random.default_rng(1).choice(rest, len(rest) // 8, replace=False))
    trn = ~te & ~va
    k = len(new)
    T = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt)  # noqa: E731
    arrays, z_te = {}, []
    for i, tag in enumerate(tags):
        pre = f"{tag}_" if tag else ""
        md = {key[len(pre):]: d[key] for key in d.files if key.startswith(pre)}
        modes, n = tuple(str(m) for m in md["modes"]), len(md["modes"])
        assert tuple(str(b) for b in md["bands"]) == BANDS and not set(new) & set(modes)
        depth = sum(1 for key in md if key.startswith("W"))
        h = (x - md["mean"]) / md["std"]
        for j in range(depth - 1):
            h = np.tanh(h @ md[f"W{j}"] + md[f"b{j}"])
        tr_s = np.unique(seeds[trn])
        count = dict(zip(*np.unique(np.random.default_rng(i + 1).choice(tr_s, len(tr_s)), return_counts=True)))
        tr = np.repeat(np.flatnonzero(trn), [count.get(s, 0) for s in seeds[trn]])
        torch.manual_seed(i + 1)
        head = torch.nn.Linear(h.shape[1], 2 * k)
        opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
        tr_t = [T(h[tr]), T(mode[tr], torch.long), T(bok[tr]), T(dsent[tr]), T(dok[tr]), T(bmask[tr])]
        va_t = [T(h[va]), T(mode[va], torch.long), T(bok[va]), T(dsent[va]), T(dok[va]), T(bmask[va])]
        best, state = float("inf"), None
        for ep in range(epochs):
            perm = torch.randperm(len(tr_t[0]))
            for s in range(0, len(perm), 512):
                b = perm[s:s + 512]
                loss = loss_fn(head(tr_t[0][b]), *(t[b] for t in tr_t[1:]), k)
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                v = loss_fn(head(va_t[0]), *va_t[1:], k).item()
            if v < best:
                best, state = v, {key: t.clone() for key, t in head.state_dict().items()}
        head.load_state_dict(state)
        print(f"member {tag or 0}: {len(tr)} training rows, best validation {best:.4f}", flush=True)
        w, b = head.weight.detach().numpy().T.astype(np.float64), head.bias.detach().numpy().astype(np.float64)
        wl, bl = md[f"W{depth - 1}"], md[f"b{depth - 1}"]
        md[f"W{depth - 1}"] = np.concatenate([wl[:, :n], w[:, :k], wl[:, n:], w[:, k:]], axis=1)
        md[f"b{depth - 1}"] = np.concatenate([bl[:n], b[:k], bl[n:], b[k:]])
        md["modes"] = np.array(modes + tuple(new))
        arrays.update({pre + key: v for key, v in md.items()})
        with torch.no_grad():
            z_te.append(head(T(h[te])).numpy())
    np.savez(out, **arrays)
    # held out: the members' mean P, as the ensemble gives it
    idx = np.arange(te.sum())
    pb = np.mean([1 / (1 + np.exp(-z[idx, mode[te]])) for z in z_te], axis=0)
    pc = np.mean([1 / (1 + np.exp(-z[idx, mode[te] + k])) for z in z_te], axis=0)
    frac = np.where(dsent[te] > 0, dok[te] / np.maximum(dsent[te], 1), bok[te])
    snr = np.array([float(r["snr"]) for r, kk in zip(rows, keep) if kk])[te]
    kind = np.array([r["kind"] for r, kk in zip(rows, keep) if kk])[te]
    print(f"{out}: {len(tags)} members, {k} modes added; held out, P(codeword) vs decoded fraction:")
    for j, m in enumerate(new):
        for kd in ("awgn", "mpg", "mpp", "mpd", "random"):
            for lo in range(0, 40, 8):
                s = (mode[te] == j) & (kind == kd) & (snr >= lo) & (snr < lo + 8)
                if s.any():
                    print(f"  {m:15s} {kd:6s} {lo:2d}-{lo + 8:2d} dB n {s.sum():4d}: "
                          f"{(pb * pc)[s].mean():.3f} vs {frac[s].mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="csv, or several comma-separated")
    ap.add_argument("--stale-header", default="", help="comma-separated csvs from before the header copy")
    ap.add_argument("--seed", type=int, default=0, help="nonzero: an ensemble member (seeded, bootstrapped)")
    ap.add_argument("--ensemble", default="", help="comma-separated member npz files: combine them into --out")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=HIDDEN, help="hidden layer width")
    ap.add_argument("--depth", type=int, default=2, help="hidden layers")
    ap.add_argument("--input-noise", type=float, default=0.0,
                    help="augmentation: Gaussian noise on the continuous inputs, in standard deviations, per batch")
    ap.add_argument("--out", default=str(P.DATA / "outcome_predictor.npz"))
    ap.add_argument("--extend", metavar="BASE", help="add --new modes' outputs to this model, all else frozen")
    ap.add_argument("--new", default="", help="comma-separated modes, with --extend")
    a = ap.parse_args()
    if a.extend:
        return extend(a.extend, a.data, a.new.split(","), a.out, a.epochs)
    if a.ensemble:
        return combine(a.ensemble.split(","), a.out)
    torch.manual_seed(a.seed)
    x, mode, bok, dsent, dok, rows, bmask = load(a.data, set(filter(None, a.stale_header.split(","))))
    n = len(MODES)
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
    if a.seed:
        # an ensemble member: a bootstrap of the training samples (a sample's
        # candidates, or a session's bursts, stay together)
        tr_seeds = np.unique(seeds[trn])
        draw = np.random.default_rng(a.seed).choice(tr_seeds, len(tr_seeds))
        count = dict(zip(*np.unique(draw, return_counts=True)))
        trn = np.repeat(np.flatnonzero(trn), [count.get(s, 0) for s in seeds[trn]])
    mean, std = x[trn].mean(0), x[trn].std(0) + 1e-6
    # the noise leaves one-hots and flags (columns of only 0 and 1) alone
    cont = ~np.all(np.isin(x[trn], (0.0, 1.0)), axis=0)
    noise_scale = torch.tensor(a.input_noise * std * cont, dtype=torch.float32)
    net = Net(x.shape[1], n, mean, std, a.hidden, a.depth)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    T = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt)  # noqa: E731
    tr_t = [T(x[trn]), T(mode[trn], torch.long), T(bok[trn]), T(dsent[trn]), T(dok[trn]), T(bmask[trn])]
    va_t = [T(x[va]), T(mode[va], torch.long), T(bok[va]), T(dsent[va]), T(dok[va]), T(bmask[va])]
    best_val, best_state = float("inf"), None
    te_t = [T(x[te]), T(mode[te], torch.long), T(bok[te]), T(dsent[te]), T(dok[te]), T(bmask[te])]
    for ep in range(a.epochs):
        perm = torch.randperm(len(tr_t[0]))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            xb = tr_t[0][b]
            if a.input_noise:
                xb = xb + torch.randn_like(xb) * noise_scale
            loss = loss_fn(net(xb), *(t[b] for t in tr_t[1:]), n)
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
    layers = list(net.layers)
    np.savez(a.out, mean=mean, std=std, modes=np.array(MODES), bands=np.array(BANDS), **{f"W{i}": l.weight.detach().numpy().T.astype(np.float64) for i, l in enumerate(layers)},
             **{f"b{i}": l.bias.detach().numpy().astype(np.float64) for i, l in enumerate(layers)})
    P.outcome_model.cache_clear()
    zn = P.outcome_model(a.out)(x[te])
    print(f"numpy runtime vs torch: max |d logit| {np.abs(zn - z).max():.2e}")


if __name__ == "__main__":
    main()
