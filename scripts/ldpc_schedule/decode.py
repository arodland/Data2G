"""Flooding vs layered sum-product (flood_record below; ldpc.Decoder, layered) on gen_llrs.py's points: per codeword,
the iteration it converged at (41: never, cap 40) and whether its decisions there were right, so one decode
answers for every cap. Same numpy phi on both sides.

    python scripts/ldpc_schedule/decode.py LLRDIR OUTDIR [WORKERS] [MAX_CW]
"""
import os, sys
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from multiprocessing import Pool
import numpy as np
from data2g import ldpc, codes
from data2g.config import SUBMODES

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(exist_ok=True)
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
MAX_CW = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
ITERS, CHUNK = 40, 500
NAMES = {m.replace("/", "_"): m for m in SUBMODES}


def flood_record(d, llr):
    """The flooding sum-product ldpc.Decoder ran before the layered schedule, recording each codeword's
    convergence."""
    ch = d.channel_llrs(np.clip(np.asarray(llr, np.float32), -ldpc.CH_CLAMP, ldpc.CH_CLAMP))
    b = ch.shape[0]
    c2v = np.zeros((b, d.n_edges), np.float32)
    v2c = np.full((b, d.n_edges + 1), d.BIG, np.float32)
    conv = np.full(b, ITERS + 1)
    first = np.zeros((b, d.code.k), np.uint8)
    for it in range(ITERS):
        tot = ch + (d.gather @ c2v.T).T
        v2c[:, :-1] = tot[:, d.var] - c2v
        m = v2c[:, d.chk]
        neg = m < 0
        sgn = np.where(neg, np.float32(-1), np.float32(1))
        sprod = np.where(np.logical_xor.reduce(neg, axis=-1, keepdims=True), np.float32(-1), np.float32(1))
        ph = ldpc._phi(np.abs(m))
        c2v = (sprod * sgn * ldpc._phi(ph.sum(axis=-1, keepdims=True) - ph))[:, ~d.pad]
        tot = ch + (d.gather @ c2v.T).T
        hard = (tot < 0).astype(np.uint8)
        ok = d.syndrome(hard)
        fresh = ok & (conv > ITERS)
        conv[fresh] = it + 1
        first[fresh] = hard[fresh, : d.code.k]
        if ok.all():
            break
    return conv, np.where((conv <= ITERS)[:, None], first, hard[:, : d.code.k])


def work(f):
    out = OUT / f.name
    if out.exists():
        return f.name, "skip"
    s = SUBMODES[NAMES[f.name.split("__")[0]]]
    code = codes.ldpc_code(s)
    fl = la = ldpc.Decoder(code)
    z = np.load(f)
    llr, info = z["llr"][:MAX_CW], z["info"][:MAX_CW]
    res = {}
    for name in ("flood", "layer"):
        conv, right = [], []
        for i in range(0, len(llr), CHUNK):
            l, inf = llr[i:i + CHUNK], info[i:i + CHUNK]
            if name == "flood":
                c, bits = flood_record(fl, l)
            else:
                c, bits = la.decode(l, iters=ITERS, record=True)[2:]
            conv.append(c)
            right.append((bits == inf).all(1))
        res[name + "_conv"], res[name + "_right"] = np.concatenate(conv), np.concatenate(right)
    np.savez(out, **res)
    return f.name, len(llr)


if __name__ == "__main__":
    files = sorted(DATA.glob("*.npz"), key=lambda p: p.stat().st_size)
    with Pool(WORKERS) as pool:
        for name, n in pool.imap_unordered(work, files):
            print(name, n, flush=True)
