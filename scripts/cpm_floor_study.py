"""CPM floor vs ion-hf's MFSK floor (2026-10-26): bounds, the code on
ideal channels, and the real receiver end to end.

    python -m scripts.cpm_floor_study bound
    python -m scripts.cpm_floor_study ideal [--modes ...] [--n 200]
    python -m scripts.cpm_floor_study e2e --modes fsk32r62-r1/3 --chans awgn mpp --arms base genie [--trials 100]

`bound`: Es/N0 (per tone) where noncoherent orthogonal M-FSK reaches each
CPM mode's info rate per tone, coded modulation (symbol-wise, what a GF(M)
code can use) and BICM (bit LLRs, what our binary LDPC sees), on AWGN and
on ideally interleaved Rayleigh. Monte Carlo, exact metrics.

`ideal`: each mode's real LDPC code and cpm.llrs on synthetic tone
energies (AWGN, Rayleigh per tone), 10% codeword failure Es/N0: the code's
gap to the BICM bound with no receiver losses.

`e2e`: bursts (control + one data codeword) through phy_session's
ContinuousChannel, average-power SNR in 2500 Hz (no DATA2G_PEP_REF_DB:
Ion quotes mean power; CPM's envelope is within 0.1-0.6 dB of it). Trial k
has the same seed at every SNR and in every arm. Arms:
  base   the shipped receiver (find, late/mid lock, cpm.llrs)
  genie  known start, CFO and header; cpm.llrs
  ric    known timing as base finds it, Rician tone metric (ln I0 with a
         per-codeword fitted specular/diffuse split) in place of square-law
  nbr    +-2 neighbour bins, shares fitted on the known symbols
  loc    per-symbol signal level (+-LOC_W symbols) in the square law
  win    analysis window (1, 7/8, 3/4, 5/8 T, end-aligned) chosen per
         burst by the known symbols' MI (Ion's), with nbr's metric
  id     BICM-ID, ID_PASSES x LDPC_ITERS; it125 its no-feedback control
  gt gs gh diag path  the acquisition split (stages(): known timing; known
         timing and CFO; known header; the base lock's diagnosis and path)
Every arm decodes through cpm.receive, which refines the timing and CFO it
is given (cpm.refine, since cpm-acq-gap).
Channels: awgn mpg mpp mpd, flutter (10 Hz, 0.5 ms), nvis (1 Hz, 7 ms),
hilat (30 Hz, 7 ms); doppler is two-sided (2 sigma), as Ion's.

Found (2026-10-26, 200 trials a point, seeds 50000+; bursts at cap 2):
- bound, Es/N0 CM | BICM: 32-FSK r1/3 3.98 | 5.78 AWGN, 5.17 | 6.76
  Rayleigh; 8-FSK r1/3 3.10 | 4.11, 4.35 | 5.24; 16-FSK r1/3 3.56 | 5.00,
  4.82 | 6.06 (Ion's r1/4 32-FSK: 3.04 | 5.05, as its docs say).
- ideal, our LDPC: 0.75-0.9 dB over BICM on AWGN, 1.2-1.3 on Rayleigh;
  the exact ln I0 metric gains 0.1-0.2 dB on AWGN only.
- e2e 10% points (dB, 2500 Hz mean power), base / genie:
    fsk32r62-r1/3 awgn -9.1/-9.1 mpg -3.9/-4.3 mpp -5.4/-6.3 mpd -5.4/-6.1
      flutter -7.3/-7.5 nvis -4.1/-4.4 hilat -4.7/-4.6
    fsk8r50-r1/3 awgn -11.3/-12.0 mpg -3.5/-5.2 mpp -7.8/-9.5 mpd -8.3/-9.1
      flutter -9.6/-10.1 nvis -7.0/-8.0 hilat -6.4/-6.8
    fsk16r25-r1/3 awgn -13.5/-14.1 mpg -6.4/-8.3 mpp -11.0/-12.1
      mpd -11.2/-12.0 flutter -11.4/-12.0 nvis -10.1/-11.2
  nbr, loc, win: within +-0.15 dB everywhere but hilat (nbr -0.4 c32r62,
  -0.7 c8r50). id: -0.1 to -0.5 (-0.0 to -0.37 over it125).

Acquisition split (cpm-acq-gap, 2026-10-26, r1/3 modes, 200 trials, the
shipped receiver of then; runs/acq_split.csv): genie - gs (header given
sync) 0.0 on awgn/mpp/mpd for c8r50 and c16r25, 0.6-0.9 on mpg; gh (the
receiver's sync, known header) left c32r62 within 0.02 of the genie but
0.4-0.5 dB AWGN timing loss for c16r25 (decoding is that sensitive: T/16
or 4 Hz off cost 0.75 dB). Base failures the genie decoded were mostly no
lock at all: front placements right but under SYNC_THRESHOLD (coarse grid)
or PEAK_RATIO, and for c32r62 both header copies right but under
HEADER2_THRESHOLD. The fix (data2g.cpm, LOCK_LLR's notes): tone_llr gates
and cpm.refine.
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import i0e  # noqa: E402

from data2g import codes, cpm  # noqa: E402
from data2g.arq.modes import MODES  # noqa: E402
from data2g.config import FS  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

CPM_MODES = [m for m in MODES if m.startswith("fsk")]
CHANS = {"awgn": None, "mpg": (0.1, 0.5), "mpp": (1.0, 2.0), "mpd": (2.0, 4.0), "flutter": (10.0, 0.5),
         "nvis": (1.0, 7.0), "hilat": (30.0, 7.0)}


def lni0(x):
    return np.log(i0e(x)) + x


# --- bounds ------------------------------------------------------------------------------

def tone_metrics(m, esn0_db, n, rng, fading):
    """(n, m) symbol log-likelihoods (up to a per-row constant) of tone 0
    sent, N0 = 1, and the energies."""
    es = 10 ** (esn0_db / 10)
    h = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2) if fading else np.ones(n)
    y = (rng.normal(size=(n, m)) + 1j * rng.normal(size=(n, m))) / np.sqrt(2)
    y[:, 0] += np.sqrt(es) * h
    E = np.abs(y) ** 2
    L = E * es / (1 + es) if fading else lni0(2 * np.sqrt(es * E))
    return L, E


def cm_bicm(m, esn0_db, fading, n=40000, seed=1):
    rng = np.random.default_rng(seed)
    L, _ = tone_metrics(m, esn0_db, n, rng, fading)
    cm = np.log2(m) - np.mean(np.log2(np.exp(L - L[:, :1]).sum(axis=1)))
    b = int(np.log2(m))
    lab = np.arange(m)
    bicm = 0.0
    for k in range(b):
        bit = (lab >> k) & 1  # tone 0 sent: every bit is 0
        l0 = np.logaddexp.reduce(L[:, bit == 0], axis=1)
        l1 = np.logaddexp.reduce(L[:, bit == 1], axis=1)
        bicm += 1 - np.mean(np.logaddexp(0, l1 - l0)) / np.log(2)
    return cm, bicm


def solve(m, rate_bits, fading, which):
    lo, hi = -5.0, 20.0
    for _ in range(18):
        mid = (lo + hi) / 2
        v = cm_bicm(m, mid, fading)[which]
        lo, hi = (mid, hi) if v < rate_bits else (lo, mid)
    return (lo + hi) / 2


def bound(a):
    print("Es/N0 per tone (dB) where noncoherent M-FSK reaches the rate: CM | BICM (gap)")
    rows = [(32, 1.25, "ion r1/4 (32-FSK)")] + [(cpm.GRIDS[MODES[n].grid].m, MODES[n].k / cpm.DATA_N
                                                 * cpm.GRIDS[MODES[n].grid].bits, n) for n in CPM_MODES]
    for m, r, name in rows:
        out = []
        for fading in (False, True):
            c, b = solve(m, r, fading, 0), solve(m, r, fading, 1)
            out.append(f"{'ray' if fading else 'awgn'} {c:5.2f} | {b:5.2f} ({b - c:.2f})")
        print(f"  {name:18s} {r:.3f} b/tone  " + "   ".join(out))


# --- ideal channels ----------------------------------------------------------------------

def bit_llrs(g, L):
    """(n, m) tone log-likelihoods -> bit LLRs in mapping order, as cpm.llrs labels them."""
    label = np.argsort(cpm._gray(g.m))
    out = np.empty((len(L), g.bits))
    for b in range(g.bits):
        bit = (label >> (g.bits - 1 - b)) & 1
        out[:, b] = np.logaddexp.reduce(L[:, bit == 0], axis=1) - np.logaddexp.reduce(L[:, bit == 1], axis=1)
    return out.reshape(-1)


def ideal_point(args):
    name, fading, esn0, n, seed, metric = args
    spec = MODES[name]
    g = cpm.GRIDS[spec.grid]
    rng = np.random.default_rng(seed)
    ok = 0
    for i in range(n):
        payload = bytes(rng.integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8))
        bits = codes.encode(spec, payload, index=1)
        t = cpm.to_tones(g, bits)
        es = 10 ** (esn0 / 10)
        h = (rng.normal(size=len(t)) + 1j * rng.normal(size=len(t))) / np.sqrt(2) if fading else np.ones(len(t))
        y = (rng.normal(size=(len(t), g.m)) + 1j * rng.normal(size=(len(t), g.m))) / np.sqrt(2)
        y[np.arange(len(t)), t] += np.sqrt(es) * h
        E = np.abs(y) ** 2
        if metric == "exact":  # genie Es: ln I0 (AWGN) or the Rayleigh square law
            llr = bit_llrs(g, E * es / (1 + es) if fading else lni0(2 * np.sqrt(es * E)))
        else:
            llr = cpm.llrs(g, E)
        out = codes.decode_many(spec, llr[None], index=[1])[0]
        ok += out[1] and out[0] == payload
    return name, fading, esn0, ok / n


def crossing(xs, fail, target):
    for i in range(len(xs) - 1):
        if fail[i] >= target > fail[i + 1]:
            return xs[i] + (fail[i] - target) / (fail[i] - fail[i + 1]) * (xs[i + 1] - xs[i])
    return float("nan")


def ideal(a):
    grid = np.arange(3.0, 12.01, 0.25)
    jobs = [(m, f, float(e), a.n, 7, a.metric) for m in a.modes for f in (False, True) for e in grid]
    with Pool(a.jobs) as p:
        res = p.map(ideal_point, jobs)
    print("10% codeword failure Es/N0 per tone (dB), our LDPC + cpm.llrs, ideal channel")
    for m in a.modes:
        s = []
        for f in (False, True):
            fail = [1 - r[3] for r in res if r[0] == m and r[1] == f]
            s.append(f"{'ray' if f else 'awgn'} {crossing(list(grid), fail, 0.1):5.2f}")
        print(f"  {m:16s} " + "  ".join(s))


def fsk_demap(g, L, la=None):
    """(n, m) tone log-likelihoods -> (n, bits) extrinsic bit LLRs given a
    priori LLRs `la` (n, bits) of each tone's label (None: bit_llrs)."""
    if la is None:
        return bit_llrs(g, L).reshape(len(L), g.bits)
    label = np.argsort(cpm._gray(g.m))
    lb = ((label[:, None] >> (g.bits - 1 - np.arange(g.bits))) & 1).astype(float)  # (m tones, bits)
    d = L - la @ lb.T
    out = np.empty((len(L), g.bits))
    for i in range(g.bits):
        t = d + lb[:, i] * la[:, i:i + 1]  # leave out bit i's own prior
        out[:, i] = (np.logaddexp.reduce(np.where(lb[:, i] == 0, t, -np.inf), axis=1)
                     - np.logaddexp.reduce(np.where(lb[:, i] == 1, t, -np.inf), axis=1))
    return out


def id_point(args):
    """BICM-ID on ideal channels: (control: one decode of inner * (outer + 1)
    iterations; id: outer re-demaps of inner each), same codewords and noise."""
    from data2g import ldpc

    name, fading, esn0, n, outer, inner = args
    spec = MODES[name]
    g = cpm.GRIDS[spec.grid]
    dec, perm = codes._decoder(spec), codes.interleaver(spec)
    rng = np.random.default_rng(11)
    info = rng.integers(0, 2, (n, spec.k)).astype(np.uint8)
    t = np.stack([cpm.to_tones(g, c) for c in codes.encode_info(spec, info)])  # (n, sym)
    es = 10 ** (esn0 / 10)
    shape = t.shape + (g.m,)
    h = (rng.normal(size=t.shape) + 1j * rng.normal(size=t.shape)) / np.sqrt(2) if fading else np.ones(t.shape)
    y = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
    np.put_along_axis(y, t[..., None], np.take_along_axis(y, t[..., None], -1) + (np.sqrt(es) * h)[..., None], -1)
    E = np.abs(y) ** 2
    L = (E * es / (1 + es) if fading else lni0(2 * np.sqrt(es * E))).reshape(-1, g.m)
    ch = fsk_demap(g, L).reshape(n, -1)
    out, ok = dec.decode(codes._deinterleave(spec, ch.astype(np.float32)), iters=inner * (outer + 1))
    ctl = np.mean(ok & (out == info).all(axis=1))
    soft, done = ch, np.zeros(n, bool)
    for o in range(outer + 1):
        deint = codes._deinterleave(spec, soft.astype(np.float32))
        out, ok, post = dec.decode(deint, iters=inner, posterior=True)
        done |= ok & (out == info).all(axis=1)
        if done.all() or o == outer:
            break
        ext = np.clip(post - np.clip(deint, -ldpc.CH_CLAMP, ldpc.CH_CLAMP), -20, 20)[:, perm]
        soft = fsk_demap(g, L, ext.reshape(-1, g.bits)).reshape(n, -1)
    return name, fading, esn0, float(ctl), float(done.mean())


def bicm_id(a):
    grid = np.arange(3.0, 11.01, 0.25)
    jobs = [(m, f, float(e), a.n, a.outer, codes.LDPC_ITERS) for m in a.modes for f in (False, True) for e in grid]
    with Pool(a.jobs) as p:
        res = p.map(id_point, jobs)
    print(f"10% codeword failure Es/N0 (dB), exact metric: control ({codes.LDPC_ITERS * (a.outer + 1)} its) | "
          f"BICM-ID ({a.outer + 1} x {codes.LDPC_ITERS})")
    for m in a.modes:
        s = []
        for f in (False, True):
            r = [x for x in res if x[0] == m and x[1] == f]
            c, d = (crossing(list(grid), [1 - x[k] for x in r], 0.1) for k in (3, 4))
            s.append(f"{'ray' if f else 'awgn'} {c:5.2f} | {d:5.2f} ({d - c:+.2f})")
        print(f"  {m:16s} " + "   ".join(s))


# --- end to end --------------------------------------------------------------------------

ARMS = ("base", "genie", "nbr", "loc", "nbrloc", "win", "winloc")
WINDOWS = (1.0, 0.875, 0.75, 0.625)  # analysis window / T, ending at the symbol's end (win arms)
LOC_W = 8  # local signal level: +- this many symbols (loc arms)
ID_PASSES = 5  # BICM-ID arms ("id", "id-<arm>"): decodes of LDPC_ITERS each; "it125" its control


def win_energies(g, y, s0, cfo, rows, L, off, X):
    """(len(rows), m + 2X) energies of the window [off, off + L) of each
    symbol row, at the tone frequencies (X extra each side), CFO removed."""
    T = g.T
    idx = s0 + np.asarray(rows)[:, None] * T + off + np.arange(L)
    seg = np.where((idx >= 0) & (idx < len(y)), y[np.clip(idx, 0, len(y) - 1)], 0.0)
    f = g.f0 + (np.arange(g.m + 2 * X) - X) * g.rate + cfo
    W = np.exp(-2j * np.pi * np.outer(np.arange(L), f) / FS)
    return np.abs(seg @ W) ** 2 * (T / L)  # a full-window tone's scale


def fit(Ek, Ed, known_t, X, m):
    """Noise level (data symbols), per-offset signal shares p_j (j = -X..X) from known symbols."""
    n0 = max(float(np.median(np.sort(Ed[:, X:X + m], axis=1)[:, :-1])) / np.log(2), 1e-30)
    s = np.array([np.mean(Ek[np.arange(len(Ek)), known_t + X + j]) - n0 for j in range(-X, X + 1)])
    s = np.maximum(s, 0.0)
    s[X] = max(s[X], 1e-3 * n0)
    return n0, s / s.sum()


def level(E, n0, w):
    """Per-symbol signal energy: whole-row energy less noise, global (w None) or +-w symbols."""
    tot = E.sum(axis=1) - E.shape[1] * n0
    if w is None:
        return np.full(len(E), max(float(np.mean(tot)), 1e-3 * n0))
    k = np.ones(2 * w + 1)
    v = np.convolve(tot, k, "same") / np.convolve(np.ones(len(tot)), k, "same")
    return np.maximum(v, 1e-2 * n0)


def tone_ll(E, n0, p, es, X, m):
    """(n, m) log-likelihoods: each bin k + j of hypothesis tone k a
    Rayleigh cell of energy p_j es (square law, per-symbol es)."""
    L = np.zeros((len(E), m))
    for i, j in enumerate(range(-X, X + 1)):
        sj = p[i] * es[:, None]
        L += E[:, X + j:X + j + m] * (sj / (n0 * (n0 + sj)))
    return L


def symbol_mi(L, t):
    L = L - L[np.arange(len(L)), t][:, None]
    return np.log2(L.shape[1]) - float(np.mean(np.log2(np.exp(np.minimum(L, 50)).sum(axis=1))))


def arm_soft(g, y, lock, X, w, windows):
    """Slot 1's bit LLRs from tone_ll, the window (of `windows`) whose
    model gives the known symbols (sync, header) the most MI."""
    spec = lock["spec"]
    lay = cpm.layout(g.name, cpm.stream_symbols(g.name, lock["n_data"], lock["dup"]))
    h = cpm.header_symbols(g.name, cpm.header_word(spec.index, lock["n_data"], lock["dup"]))
    krows = np.concatenate([lay.sync_rows] + list(lay.hdr_rows))
    kt = np.concatenate([lay.sync_tones] + [h] * len(lay.hdr_rows))
    a = cpm.CTL[g.name].n_sym * (1 + lock["dup"])
    drows = lay.data_rows[a:a + cpm.DATA_N // g.bits]
    best = None
    for fr in windows:
        Lw = int(round(fr * g.T))
        Ek = win_energies(g, y, lock["start"], lock["cfo"], krows, Lw, g.T - Lw, X)
        Ed = win_energies(g, y, lock["start"], lock["cfo"], drows, Lw, g.T - Lw, X)
        n0, p = fit(Ek, Ed, kt, X, g.m)
        mi = symbol_mi(tone_ll(Ek, n0, p, level(Ek, n0, None), X, g.m), kt) if len(windows) > 1 else 0.0
        if best is None or mi > best[0]:
            best = (mi, Ed, n0, p)
    _, Ed, n0, p = best
    return tone_ll(Ed, n0, p, level(Ed, n0, w), X, g.m)


def base_ll(g, E):
    """cpm.llrs's tone log-likelihoods (square law, the codeword's es and N0)."""
    n0 = max(np.median(np.sort(E, axis=1)[:, :-1]) / np.log(2), 1e-12 * np.mean(E.max(axis=1)) + 1e-300)
    es = max(np.mean(E.max(axis=1)) - n0, 1e-3 * n0)
    return E * (es / (n0 + es) / n0)


def id_decode(g, spec, L, mask, index, passes):
    """BICM-ID over tone log-likelihoods L: -> (payload, ok)."""
    from data2g import ldpc

    soft = np.clip(bit_llrs(g, L), -30, 30)
    for o in range(passes):
        bits, conv, post = codes.decode_posterior(spec, soft=soft)
        got = codes.payloads(spec, bits[None], [conv], [mask], [index])[0]
        if got[1] or o == passes - 1:
            return got
        ext = np.clip(post - np.clip(soft, -ldpc.CH_CLAMP, ldpc.CH_CLAMP), -20, 20)
        soft = np.clip(fsk_demap(g, L, ext.reshape(-1, g.bits)).reshape(-1), -30, 30)


ARM_ARGS = {"nbr": (2, None, (1.0,)), "loc": (0, LOC_W, (1.0,)), "nbrloc": (2, LOC_W, (1.0,)),
            "win": (2, None, WINDOWS), "winloc": (2, LOC_W, WINDOWS)}


def e2e_trial(args):
    import outcome_data as O
    import phy_session as PS
    from dataclasses import replace
    from data2g.arq import phy as PHY

    mode, chan, snr, seed, arms = args
    spec = MODES[mode]
    g = cpm.GRIDS[spec.grid]
    rng = np.random.default_rng(seed)
    b = replace(O.burst(mode, 2, rng), cap=2)
    dd = CHANS[chan]
    ch = PS.ContinuousChannel("awgn", snr, seed, 120.0, *(dd or (None, None)))
    y = ch.apply(PHY.tx_audio(b), float(rng.uniform(5, 100)))
    want = b.slots[1].payload
    mask = PHY.mask_value(b.slots[1].mask_id)

    def dec(r, soft1=None):
        r = dict(r)
        r.pop("_soft", None)
        if soft1 is not None:
            r["soft"] = [r["soft"][0], soft1]
        return int(PHY.ModemRx(r, {}).decode(1, b.slots[1].mask_id, 0, None) == want)

    out = {}
    r = receive_traced(y, mode)
    good = r is not None and r["spec"].name == mode and r["n_cw"] == 2
    lock = None
    if good:
        lock = dict(spec=r["spec"], n_data=1, dup=r["dup"], start=r["preamble_start"], cfo=r["cfo"])
    st = stages(g, y, spec, r, lock) if any(a in STAGE_ARMS for a in arms) else None
    for arm in arms:
        if arm in STAGE_ARMS:
            out[arm] = st[arm](dec) if callable(st[arm]) else st[arm]
        elif arm == "genie":
            gl = dict(spec=spec, n_data=1, dup=False, start=int(PS.PAD_S * FS), cfo=PS.CFO_HZ, band=g.name)
            out[arm] = dec(cpm.receive(y, gl))
        elif not good:
            out[arm] = 0
        elif arm == "base":
            out[arm] = dec(r)
        elif arm == "it125":  # ID's control: as many iterations, no feedback
            it, codes.LDPC_ITERS = codes.LDPC_ITERS, ID_PASSES * codes.LDPC_ITERS
            out[arm] = int(id_decode(g, spec, base_ll(g, slot_E(g, r)), mask, 1, 1) == (want, True))
            codes.LDPC_ITERS = it
        elif arm.startswith("id"):
            L = base_ll(g, slot_E(g, r)) if arm == "id" else arm_soft(g, y, lock, *ARM_ARGS[arm[3:]])
            out[arm] = int(id_decode(g, spec, L, mask, 1, ID_PASSES) == (want, True))
        else:
            out[arm] = dec(r, np.clip(bit_llrs(g, arm_soft(g, y, lock, *ARM_ARGS[arm])), -30, 30))
    return mode, chan, snr, seed, out


STAGE_ARMS = ("gt", "gs", "gh", "diag", "path")


def cfo_at(g, y, s0, reach_hz=150.0):
    """CFO estimate at a known start: the front pattern's share, quarter-bin
    steps over +-reach_hz, then R/16 around the best (detect's grid)."""
    f = cpm.preamble_pattern(g)
    n = len(f)

    def sc(c):
        E = cpm._energies(g, y, s0, n, c)
        return E[np.arange(n), f].sum() / E.sum()
    c = max(np.arange(-reach_hz, reach_hz + 1e-9, g.rate / 4), key=sc)
    return float(max(c + np.arange(-0.125, 0.126, 0.0625) * g.rate, key=sc))


def rx_header(g, y, s0, cfo):
    """The receiver's header stage at a given sync (find's and late_lock's
    gates): copy 1, else both copies; (spec, n, dup) or None. Before
    cpm-acq-gap this was the share thresholds (HEADER_THRESHOLD, HEADER2_THRESHOLD)."""
    h = cpm.read_header(g, y, s0, cfo, copies=1)
    if h[3] >= cpm.LOCK_LLR[g.name][0] and h[3] - h[4] >= cpm.HDR_MARGIN[0]:
        return h[:3]
    h = cpm.read_header(g, y, s0, cfo, copies=cpm.HDR_COPIES)
    return h[:3] if h[3] >= cpm.LOCK_LLR[g.name][1] and h[3] - h[4] >= cpm.HDR_MARGIN[1] else None


def stages(g, y, spec, r, lock):
    """Per-stage genie arms (callables of dec) and the base lock's diagnosis:
      gt    known start; CFO (cfo_at) and header (rx_header) by the receiver
      gs    known start and CFO; header by the receiver
      gh    the receiver's sync (its lock's, else the front's best placement
            at any score), known header
      diag  0 no lock, 1 lock off (|dt| > T/8 or |df| > R/4), 2 header wrong, 3 right
      path  0 none, 1 find (copy 1), 2 late_lock (both), 3 find_mid"""
    import phy_session as PS

    s_true, f_true = int(PS.PAD_S * FS), PS.CFO_HZ
    truth = (spec, 1, False)

    def at(s0, cfo, h):
        if h is None:
            return lambda dec: 0
        return lambda dec: dec(cpm.receive(y, dict(spec=h[0], n_data=h[1], dup=h[2], start=s0, cfo=cfo,
                                                   band=g.name)))
    out = dict(gs=at(s_true, f_true, rx_header(g, y, s_true, f_true)))
    c = cfo_at(g, y, s_true)
    out["gt"] = at(s_true, c, rx_header(g, y, s_true, c))
    if lock is not None:
        s0, cfo = lock["start"], lock["cfo"]
    else:
        head = y[:s_true + FS // 2 + search_span(g.name)]
        _, s0, cfo = cpm.detect(g, head, front_only=True)
    out["gh"] = at(s0, cfo, truth)
    if r is None:
        out["diag"] = 0
    elif abs(r["preamble_start"] - s_true) > g.T / 8 or abs(r["cfo"] - f_true) > g.rate / 4:
        out["diag"] = 1
    else:
        out["diag"] = 2 if (r["spec"], r["n_cw"] - 1 - r["dup"], r["dup"]) != truth else 3
    out["path"] = 0 if r is None else r.get("_path", 0)
    return out


def search_span(grid):
    from data2g.receiver import search_span as ss
    return ss((), (grid,))


def receive_traced(y, mode):
    """outcome_data.receive with the lock path in r["_path"]."""
    import phy_session as PS

    g = cpm.GRIDS[MODES[mode].grid]
    lock = cpm.find(g, y[:int(PS.PAD_S * FS) + FS // 2 + search_span(g.name)], wait=True)
    path = 1
    if lock is not None and lock.get("wait"):
        lock, path = (cpm.late_lock(g, y, lock) if lock["header_end"] <= len(y) else None), 2
    elif lock is not None and lock["header_end"] - lock["start"] > cpm.hdr_span(g.name, 1):
        path = 2  # find ran late_lock itself
    if lock is None:
        lead = int(PS.PAD_S * FS) + FS // 2
        lock = cpm.find_mid(g, y[:lead + cpm.hdr_span(g.name, 2) + g.T])
        lock, path = (lock if lock is not None and lock["start"] <= lead else None), 3
    if lock is None:
        return None
    r = cpm.receive(y, lock)
    r["_path"] = path
    return r


def slot_E(g, r):
    """Data slot 1's tone energies from cpm.receive's stream energies."""
    a = cpm.CTL[g.name].n_sym * (1 + r["dup"])
    return r["E"][a:a + cpm.DATA_N // g.bits]


def e2e_grid(mode, chan):
    lo = {"fsk32r62": -12.0, "fsk8r50": -14.0, "fsk16r25": -16.0}[mode.split("-")[0]]
    if chan != "awgn":
        lo += 2.0
    return np.arange(lo, lo + (9.0 if chan == "awgn" else 14.0), 1.0)


def e2e(a):
    jobs = [(m, c, float(s), 50000 + i, a.arms) for m in a.modes for c in a.chans for s in e2e_grid(m, c)
            for i in range(a.trials)]
    rows = []
    with Pool(a.jobs) as p, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "channel", "snr", "seed", "arm", "ok"])
        for m, c, s, seed, out in p.imap_unordered(e2e_trial, jobs, chunksize=4):
            for arm, ok in out.items():
                w.writerow([m, c, s, seed, arm, ok])
                rows.append((m, c, s, arm, ok))
    summarize(rows, a.arms)


def summarize(rows, arms):
    print("SNR (dB, mean power in 2500 Hz) at 10% / 1% codeword failure, end to end; arm - base in brackets")
    for m in sorted({r[0] for r in rows}):
        for c in [c for c in CHANS if any(r[1] == c and r[0] == m for r in rows)]:
            snrs = sorted({r[2] for r in rows if r[0] == m and r[1] == c})
            pts = {}
            for arm in [a for a in arms if a not in ("diag", "path")]:
                fail = [1 - np.mean([r[4] for r in rows if r[0] == m and r[1] == c and r[2] == s and r[3] == arm])
                        for s in snrs]
                pts[arm] = (crossing(snrs, fail, 0.1), crossing(snrs, fail, 0.01))
            b10 = pts.get("base", (np.nan,))[0]
            cells = "  ".join(f"{arm} {v[0]:6.2f}/{v[1]:6.2f}" + ("" if arm == "base" else f" ({v[0] - b10:+.2f})")
                              for arm, v in pts.items())
            print(f"  {m:15s} {c:8s} {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bound", "ideal", "e2e", "id"])
    ap.add_argument("--outer", type=int, default=4, help="id: re-demap passes after the first decode")
    ap.add_argument("--modes", nargs="+", default=CPM_MODES)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--metric", default="cpm", choices=["cpm", "exact"], help="ideal: cpm.llrs or the genie metric")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--chans", nargs="+", default=list(CHANS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--out", default="runs/cpm_floor_e2e.csv")
    ap.add_argument("--probe", action="store_true", help="e2e: one trial per mode, timed, no pool")
    a = ap.parse_args()
    if a.probe:
        return probe(a)
    {"bound": bound, "ideal": ideal, "e2e": e2e, "id": bicm_id}[a.cmd](a)


def probe(a):
    import resource
    import time

    for m in a.modes:
        for c, s in (("awgn", 30.0), ("nvis", -2.0)):
            t = time.time()
            print(e2e_trial((m, c, s, 5, a.arms)), round(time.time() - t, 2), "s")
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3, "MB peak RSS")


if __name__ == "__main__":
    main()
