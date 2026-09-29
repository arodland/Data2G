"""Per-mode logit offsets for the outcome model from on-policy session rows
(data2g.arq.predictor.LOGIT_OFFSETS for P(burst usable), CW_LOGIT_OFFSETS
for P(codeword | usable)).

Per mode and head, the logit gap actual - predicted in each cell of channel
kind (awgn, mpg, mpp, mpd) x SNR window (snr_next). Probabilities are
clipped to [0.02, 0.98] first, so a saturated 0.998 vs 1.000 is no gap. A
mode gets an offset only if every cell with enough evidence (MIN_ROWS
bursts; MIN_CW codewords) agrees in sign, in at least MIN_CELLS cells, and
the weighted mean gap is at least MIN_GAP: then that mean, bounded at
MAX_OFF (offsets apply everywhere). Rows come from sessions the model
itself steered (session_data.py with it installed), not its training data.

    uv run python scripts/fit_offsets.py runs/session_data_v10onpol.csv runs/outcome_predictor_v10.npz
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from data2g.arq import predictor as P

sys.path.insert(0, str(Path(__file__).parent))
import train_outcome as T  # noqa: E402

KINDS = ("awgn", "mpg", "mpp", "mpd")
WINDOWS = ((-10, -4), (-4, 0), (0, 4), (4, 8), (8, 14), (14, 22), (22, 40))
MIN_ROWS, MIN_CW, MIN_CELLS, MIN_GAP, MAX_OFF, CLIP = 40, 200, 2, 0.3, 1.0, 0.02


def logit(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return float(np.log(p / (1 - p)))


def fit(gaps):
    """[(gap, weight)] per cell -> the offset, or None."""
    if len(gaps) < MIN_CELLS:
        return None
    g = np.array([x for x, _ in gaps])
    w = np.array([y for _, y in gaps], float)
    if not (np.all(g > 0) or np.all(g < 0)):
        return None
    m = float((g * w).sum() / w.sum())
    return round(float(np.clip(m, -MAX_OFF, MAX_OFF)), 2) if abs(m) >= MIN_GAP else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("model")
    a = ap.parse_args()
    x, mode, bok, dsent, dok, rows, _ = T.load(a.data)
    model = P.outcome_model(a.model)
    assert tuple(model.modes) == tuple(T.MODES)
    z = np.concatenate([model(x[i:i + 20000]) for i in range(0, len(x), 20000)])
    n = len(model.modes)
    pu_all = 1 / (1 + np.exp(-z[np.arange(len(z)), mode]))
    pc_all = 1 / (1 + np.exp(-z[np.arange(len(z)), n + mode]))
    kind = np.array([r["kind"] for r in rows])
    snr = np.array([float(r["snr_next"]) for r in rows])
    gu, gc = defaultdict(list), defaultdict(list)
    for k in KINDS:
        for lo, hi in WINDOWS:
            cell = (kind == k) & (snr >= lo) & (snr < hi)
            for m in np.unique(mode[cell]):
                s = cell & (mode == m)
                if s.sum() >= MIN_ROWS:
                    gu[m].append((logit(bok[s].mean()) - logit(pu_all[s].mean()), s.sum()))
                u = s & (bok > 0.5) & (dsent > 0)
                if dsent[u].sum() >= MIN_CW:
                    pc = (pc_all[u] * dsent[u]).sum() / dsent[u].sum()
                    gc[m].append((logit(dok[u].sum() / dsent[u].sum()) - logit(pc), dsent[u].sum()))
    for title, g in (("LOGIT_OFFSETS (P usable)", gu), ("CW_LOGIT_OFFSETS (P codeword | usable)", gc)):
        out = {T.MODES[m]: o for m in sorted(g) if (o := fit(g[m])) is not None}
        print(f"{title}: {out}")
        for m in sorted(g):
            print(f"  {T.MODES[m]:16s} cells {len(g[m])}: " + " ".join(f"{x:+.2f}" for x, _ in g[m]))


if __name__ == "__main__":
    main()
