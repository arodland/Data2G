"""#344: CPM tone metrics under an echo past a symbol (room reverb in acoustic
coupling, a delayed path). Prototype metrics against the shipped receiver.

    python -m scripts.cpm_echo_study --modes fsk32r62-r1/3 --chans awgn mpp e20 --trials 100

Bursts (control + one data codeword) through phy_session's ContinuousChannel,
average-power SNR in 2500 Hz (as cpm_floor_study e2e). Trial k has the same
seed at every SNR and in every arm. Echo channels eD: two equal Rayleigh paths
D ms apart at 0.05 Hz (path powers still change several-fold over a 10 s
burst). Per trial, from the shipped lock:
  rx   cpm.receive and ModemRx as shipped (control on coh_metric, data DD)
  nc   first pass, noncoherent: control and data on tone_metric, no DD
  rk   nc's rows read by a two-sided noncoherent rake (rake_metric): path
       strengths s_l, l = -RAKE_LAGS..RAKE_LAGS (0: the path the lock found),
       per RAKE_BLOCK rows from the decided tones; tone k at row i scored on
       every row i + l's energy on k against that row's predicted floor
       (every other path's tone there) less this hypothesis; RAKE_PASSES
       passes, each on the last's probabilities (sync and header rows known)
  rkp  rk, then up to FEEDBACK_PASSES re-decodes of the data slot on the
       rake fed the last failed decode's GF posteriors (cpm_pass's DD loop)
  rkg  rk with the sent tones (genie): what better decisions could give
Columns <arm>c (control slot) and <arm>d (data slot), lock (right word found).

Found on the way (2026-11-27, 100 trials, seeds 50000+):
- An echo floor alone (the earlier path's tone counted as noise, backward
  lags only) gained 0-0.5 dB, even with the sent tones: it throws away the
  echo's power, and the lock takes whichever path is stronger at the burst's
  start, so half the time the other path is a pre-echo (on the next row's
  tone). When the two swap mid-burst, every later symbol reads one row off
  (seed 50002, c8r50 e20 10 dB: 87% of rows' strongest tone is the next
  row's).
- 10% points (dB, mean power in 2500 Hz), runs/cpm_echo_r4.csv, data / control:
    rkg - nc: c32r62 e10 -2.17 / -0.60, e20 nc never / -4.75, e40 -1.50 / -0.42;
      c8r50 e20 -6.33 / -4.50 and e40 -6.75 / -4.40 where every other arm
      never reaches 10%; c16r25 e40 -9.11 / -7.67 likewise.
    rkp - nc: at best -0.90 / -0.25 (c32r62 e10), -1.50 control at e20;
      nothing at a whole-symbol delay (c8r50 e20 e40, c16r25 e40). The
      shipped receiver (rx: coherent control, data DD) is ahead of rkp on
      every echo cell's data (c32r62 e10: -4.75 against -3.40).
    harm: hilat, where the prototype drops leak_metric's neighbour terms:
      c16r25 never reaches 10%, c8r50 +1.05 / +0.66. AWGN, mpp, mpd within
      +-0.25 (control) and +-0.11 (data).
  The information is there (the genie), the practical arms do not find it:
  the decided tones and the strengths learnt from them stay one row off
  after a swap, the known rows are all in the burst's first ~40% (two sync
  blocks and header copies), and a first pass at 45-87% SER leaves the
  code's posteriors nothing to fix.
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpm_floor_study as S  # noqa: E402
import outcome_data as O  # noqa: E402
import phy_session as PS  # noqa: E402

from data2g import codes, cpm  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402

RAKE_LAGS = 4  # rows either side: c32r62 (16 ms) to 64 ms
RAKE_BLOCK = 40  # rows per path-strength estimate: the paths change within a burst
RAKE_PASSES = 2
FEEDBACK_PASSES = 4  # rkp: decodes feeding back, as CPM_DD_ITERS
RAKE_MIN = 0.1  # an echo path below this share of the strongest is left out (no echo: tone_metric as shipped)
ROOM_HZ = 0.05
CHANS = dict(S.CHANS, **{f"e{d}": (ROOM_HZ, float(d)) for d in (5, 10, 20, 40)})
ARMS = ("rx", "nc", "rk", "rkp", "rkg")
LAGS = np.arange(-RAKE_LAGS, RAKE_LAGS + 1)


def _softmax(G):
    P = np.exp(G - G.max(axis=1, keepdims=True))
    return P / P.sum(axis=1, keepdims=True)


def sent_rows(g, spec, b):
    """The burst's sent tone per row (cpm.modulate's) and its layout."""
    coded = [codes.encode(PHY.ctl_spec(spec) if i == 0 else spec, s.payload, s.rv, PHY.mask_value(s.mask_id), i)
             for i, s in enumerate(b.slots)]  # tx_audio's: slot 0 the control (no dup here)
    stream = np.concatenate([cpm.to_tones(g, c) for c in coded])
    lay = cpm.layout(g.name, len(stream))
    sym = np.empty(lay.n, int)
    sym[lay.sync_rows] = lay.sync_tones
    for rows in lay.hdr_rows:
        sym[rows] = cpm.header_symbols(g.name, cpm.header_word(spec.index, len(coded) - 1, False))
    sym[lay.data_rows] = stream
    return sym, lay


def strengths(E, P, n0):
    """(n, 2 L + 1) per row, each path's energy over n0 (s_l), from blocks of
    RAKE_BLOCK rows: the mean excess on the tone sent l rows before, over
    rows where that tone is more than one bin from every other path's tone
    here (lag 0: this row's)."""
    n = len(E)
    t = P.argmax(axis=1)
    S_ = np.zeros((n, len(LAGS)))
    for a in range(0, n, RAKE_BLOCK):
        rw = np.arange(a, min(n, a + RAKE_BLOCK))
        for j, lag in enumerate(LAGS):
            src = rw - lag
            ok = (src >= 0) & (src < n)
            if lag:
                ok &= np.abs(t[np.clip(src, 0, n - 1)] - t[rw]) > 1
            if ok.sum() >= 4:
                S_[rw, j] = max(float(np.mean(E[rw[ok], t[src[ok]]])) - n0, 0.0)
    return S_


def rake_metric(E, P, n0):
    """(n, m) tone log-likelihoods of every row under the paths' strengths:
    row j, tone k ~ Exp(T[j, k]), T = n0 + sum_l s_l(j) P[j - l, k]; tone k
    sent at i: each row i + l's observation less its other paths' share,
    E (1/N' - 1/(N' + s_l)) - ln(1 + s_l / N'), N' = T - s_l P[i, k]."""
    n, m = E.shape
    S_ = strengths(E, P, n0)
    top = S_.max(axis=1, keepdims=True)
    S_ = np.where(S_ >= RAKE_MIN * np.maximum(top, 1e-12), S_, 0.0)
    T = np.full((n, m), n0)
    for j, lag in enumerate(LAGS):
        src = np.arange(n) - lag
        ok = (src >= 0) & (src < n)
        T[ok] += S_[ok, j, None] * P[src[ok]]
    M = np.zeros((n, m))
    for j, lag in enumerate(LAGS):
        obs = np.arange(n) + lag  # where row i's tone arrives on this path
        ok = (obs >= 0) & (obs < n)
        s = S_[obs[ok], j][:, None]
        Np = np.maximum(T[obs[ok]] - s * P[ok], n0)
        M[ok] += E[obs[ok]] * (s / (Np + s) / Np) - np.log1p(s / Np)
    return M


def rake_soft(g, r, P0, slot, known=None, passes=RAKE_PASSES):
    """A slot's soft values from rake_metric over the whole burst's rows
    (time order), starting from P0; `known` (rows, tones) kept one-hot."""
    E = cpm._sq(r["Z"])
    n0 = cpm._n0(E)
    P = P0
    for _ in range(passes):
        M = rake_metric(E, P, n0)
        P = _softmax(M)
        if known is not None:
            P[known[0]] = 0.0
            P[known[0], known[1]] = 1.0
    rows = cpm.slot_rows(g, r["n_cw"] - r["n_ctl_slots"], r["dup"])[slot]
    M = M[rows]
    if slot < r["n_ctl_slots"]:
        return np.clip(cpm.bit_llrs(g, M), -30, 30)
    M = np.clip(M - M.max(axis=1, keepdims=True), -60, 0)
    return M[:, cpm._gray(g.m)].reshape(-1)


def trial(args):
    mode, chan, snr, seed = args
    spec = S.MODES[mode]
    g = cpm.GRIDS[spec.grid]
    rng = np.random.default_rng(seed)
    b = replace(O.burst(mode, 2, rng), cap=2)
    dd = CHANS[chan]
    ch = PS.ContinuousChannel("awgn", snr, seed, 120.0, *(dd or (None, None)))
    y = ch.apply(PHY.tx_audio(b), float(rng.uniform(5, 100)))
    out = {"lock": 0}
    r = S.receive_traced(y, mode)
    if r is None or r["spec"].name != mode or r["n_cw"] != 2 or r["dup"]:
        return mode, chan, snr, seed, dict(out, **{a + s: 0 for a in ARMS for s in "cd"})
    out["lock"] = 1
    lock = dict(spec=r["spec"], n_data=1, dup=False, start=r["preamble_start"], cfo=r["cfo"], band=g.name)

    def dec(rr, soft=None):
        rr = dict(rr)
        rr.pop("_soft", None)
        if soft is not None:
            rr["soft"] = soft
        return (int(PHY.ModemRx(dict(rr), {}).decode(0, b.slots[0].mask_id, 0, None) == b.slots[0].payload),
                int(PHY.ModemRx(dict(rr), {}).decode(1, b.slots[1].mask_id, 0, None) == b.slots[1].payload))

    out["rxc"], out["rxd"] = dec(cpm.receive(y, lock))
    coh, dd_it = cpm.COH_CTL, PHY.CPM_DD_ITERS
    cpm.COH_CTL, PHY.CPM_DD_ITERS = False, 0  # first pass, noncoherent: the metric alone
    try:
        rr = cpm.receive(y, lock)
        out["ncc"], out["ncd"] = dec(rr)
        sym, lay = sent_rows(g, spec, b)
        E_all = cpm._sq(rr["Z"])
        known = np.concatenate([lay.sync_rows] + list(lay.hdr_rows))
        kn = (known, sym[known])
        P_dec = np.full(E_all.shape, 1.0 / g.m)
        for rws in cpm.slot_rows(g, 1, False):
            P_dec[rws] = _softmax(cpm.tone_metric(g, E_all[rws], "data"))
        P_dec[known] = 0.0
        P_dec[known, sym[known]] = 1.0
        P_gen = np.zeros(E_all.shape)
        P_gen[np.arange(len(sym)), sym] = 1.0
        out["rkc"], out["rkd"] = dec(rr, [rake_soft(g, rr, P_dec, s, kn) for s in (0, 1)])
        out["rkgc"], out["rkgd"] = dec(rr, [rake_soft(g, rr, P_gen, s, None, 1) for s in (0, 1)])
        # rkp: the data slot's failed decodes feed their GF posteriors back (cpm_pass's DD loop)
        s0, P, drows = rake_soft(g, rr, P_dec, 0, kn), P_dec.copy(), cpm.slot_rows(g, 1, False)[1]
        out["rkpc"], out["rkpd"] = out["rkc"], 0
        for _ in range(FEEDBACK_PASSES + 1):
            sd = rake_soft(g, rr, P, 1, kn)
            if dec(rr, [s0, sd])[1]:
                out["rkpd"] = 1
                break
            _, conv, lp = codes.decode_posterior(spec, sd)
            lp = np.asarray(lp).reshape(-1, g.m)
            q = np.exp(lp - lp.max(axis=1, keepdims=True))
            P[drows] = 0.0
            P[drows[:, None], cpm._gray(g.m)[None]] = q / q.sum(axis=1, keepdims=True)
    finally:
        cpm.COH_CTL, PHY.CPM_DD_ITERS = coh, dd_it
    return mode, chan, snr, seed, out


def grid(mode, chan):
    lo = {"fsk32r62": -12.0, "fsk8r50": -14.0, "fsk16r25": -16.0}[mode.split("-")[0]] + (chan != "awgn") * 2.0
    return np.arange(lo, lo + (9.0 if chan == "awgn" else 16.0), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["fsk32r62-r1/3", "fsk16r25-r1/3", "fsk8r50-r1/3"])
    ap.add_argument("--chans", nargs="+", default=["awgn", "mpp", "mpd", "hilat", "e10", "e20", "e40"])
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="runs/cpm_echo.csv")
    a = ap.parse_args()
    jobs = [(m, c, float(s), 50000 + i) for m in a.modes for c in a.chans for s in grid(m, c) for i in range(a.trials)]
    rows = []
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Pool(a.jobs) as p, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "channel", "snr", "seed", "col", "ok"])
        for m, c, s, seed, out in p.imap_unordered(trial, jobs, chunksize=2):
            for col, ok in out.items():
                w.writerow([m, c, s, seed, col, ok])
                rows.append((m, c, s, col, ok))
    print("SNR (dB, mean power in 2500 Hz) at 10% failure (1% after /); arm - nc at 10% in brackets")
    for m in a.modes:
        for c in a.chans:
            snrs = sorted({r[2] for r in rows if r[0] == m and r[1] == c})
            pts = {}
            for col in ["lock"] + [x + s for s in "cd" for x in ARMS]:
                fail = [1 - np.mean([r[4] for r in rows if r[:2] == (m, c) and r[2] == s and r[3] == col]) for s in snrs]
                pts[col] = (S.crossing(snrs, fail, 0.1), S.crossing(snrs, fail, 0.01))
            cells = "  ".join(f"{k} {v[0]:6.2f}/{v[1]:6.2f}" for k, v in pts.items())
            d = "  ".join(f"{x} c {pts[x + 'c'][0] - pts['ncc'][0]:+.2f} d {pts[x + 'd'][0] - pts['ncd'][0]:+.2f}"
                          for x in ("rk", "rkp", "rkg"))
            print(f"  {m:15s} {c:6s} {cells}  ({d})")


if __name__ == "__main__":
    main()
