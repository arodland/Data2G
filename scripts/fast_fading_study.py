"""Achievable-rate screen of waveform candidates on fast, long-delay fading
(the ITU-R F.1487 severe presets: hilat-d is 30 Hz / 7 ms), 2026-11-23.

Not the modem: each candidate is a steady stream of data symbols through
hfchannel's two-path fading, received with genie symbol timing and no CFO,
no sync, header or control. What it measures is the rate a code could carry
(mutual information of the receiver's soft values), per second of air:

- OFDM (QPSK): bits/s = GMI of the bit LLRs x data bits/s. Numerology
  (useful symbol, CP), pilot pattern (every D-th symbol, every K-th carrier
  of it, staggered: K = 1 is a pilot row, D = 1 a comb), and the mapping:
  plain; pair (X, -X on adjacent carriers: ICI self-cancellation); idle (X,
  0); dtime / dfreq (differential QPSK along time / frequency, no pilots).
  Channel estimate: 2-D Wiener filter of the pilots (the channel's true
  Doppler spread, delay support 0..delay + 0.5 ms, true noise + interference
  variance); `dd` (the same filter over every other cell's symbol as if
  known: decision-directed estimation once every decision is right); and
  `perfect` (the true channel at each symbol's centre).
- M-FSK (continuous phase, noncoherent): bits/s = symbol MI x symbols/s, as
  the GF(M) code sees it. Tone spacing S, symbol = L sub-windows of 1/S:
  `full` one FFT over the symbol (L = 1: today's CPM grids), `sub` the L
  sub-windows' energies summed (the tone's bin is S wide, the symbol L/S
  long), `subg` the same without the first sub-window (delay guard).

SNR is PEP-referenced as DATA2G_PEP_REF_DB=5 loss-study cells (noise against
the waveform's envelope peak - 5 dB, in 2500 Hz), so a lower peak-to-average
is worth its dB. Each soft value's scale is fitted (one scalar per run): a
receiver that knows its average SINR. OFDM rows also carry the unscaled GMI.

Rates here have no sync/header/control/ARQ in them and assume a capacity-
achieving code: compare candidates with each other and with the fsk rows of
today's grids (fsk16-s25-L1, fsk8-s50-L1, fsk32-s62.5-L1), not with session bps.

    python -m scripts.fast_fading_study --out runs/fast_fading.csv
    python -m scripts.fast_fading_study --check
"""

from data2g import threads

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from data2g import constellation, hfchannel  # noqa: E402
from data2g.config import CLIP_OVERSHOOT, FS, SNR_REF_BW_HZ  # noqa: E402
from data2g.waveform.dsp import tx_condition  # noqa: E402

PEP_REF_DB = 5.0
CHANNELS = {**hfchannel.FADING_PRESETS, **hfchannel.ITU_PRESETS}
BLOCK_S = 1.0  # the 10th-percentile rate is over blocks this long (about a codeword)
PAD = 400  # silence each side: the TX filter's tails
BACKOFF = 6  # OFDM window start before the CP's end, samples (config.DEMOD_BACKOFF)
LN2 = np.log(2)
FIELDS = ["family", "name", "chan", "snr", "arm", "bps", "p10_bps", "frac", "max_bps", "bps_unscaled", "sir_db",
          "peak_db"]


def fade(x: np.ndarray, p: hfchannel.FadingPreset, seed: int):
    """hfchannel.fading, with its taps: (y, g1, g2, delay samples)."""
    rng = np.random.default_rng(seed)
    z = signal.hilbert(x)
    d = int(round(p.delay_ms * 1e-3 * FS))
    g1 = hfchannel._gaussian_taps(len(z), p.doppler_hz, rng)
    g2 = hfchannel._gaussian_taps(len(z), p.doppler_hz, rng)
    z2 = np.concatenate([np.zeros(d, dtype=complex), z[: len(z) - d]])
    return np.real((z * g1 + z2 * g2) / np.sqrt(2)), g1, g2, d


def peak_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.max(np.abs(signal.hilbert(x)) ** 2) / 2 / np.mean(x[PAD:-PAD] ** 2)))


def sigma(x: np.ndarray, snr_db: float) -> float:
    """Noise per sample for a PEP-referenced cell (phy_session.ContinuousChannel's)."""
    s_power = np.max(np.abs(signal.hilbert(x)) ** 2) / 2 / 10 ** (PEP_REF_DB / 10)
    return float(np.sqrt(s_power * (FS / 2) / SNR_REF_BW_HZ / 10 ** (snr_db / 10)))


def cnoise(rng, shape, var):
    return np.sqrt(var / 2) * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


class Tally:
    """Information carried, summed over seeds: total and per block."""

    def __init__(self):
        self.info = self.info1 = self.air = 0.0
        self.blocks = []

    def add(self, info: np.ndarray, block: np.ndarray, air_per: float, n_air: np.ndarray, info1=None):
        """info: bits carried by each soft value; block: its block; air_per:
        seconds of air per counted symbol; n_air: symbols counted in each block."""
        self.info += info.sum()
        self.info1 += info.sum() if info1 is None else info1.sum()
        self.air += n_air.sum() * air_per
        per = np.bincount(block, weights=info, minlength=len(n_air))
        ok = n_air >= 0.5 * n_air.max()
        self.blocks.append(np.maximum(per[ok], 0) / (n_air[ok] * air_per))

    def row(self, max_bps: float) -> dict:
        bps = max(self.info, 0) / self.air
        return dict(bps=round(bps, 1), p10_bps=round(float(np.percentile(np.concatenate(self.blocks), 10)), 1),
                    frac=round(bps / max_bps, 3), max_bps=round(max_bps, 1),
                    bps_unscaled=round(max(self.info1, 0) / self.air, 1))


def bit_info(llr: np.ndarray, bits: np.ndarray, scales) -> tuple:
    """Per bit, 1 - log2(1 + exp(-s L (1 - 2b))) at the scale s carrying the
    most (GMI), and at s = 1."""
    sg = llr * (1 - 2 * bits)
    best = None
    for s in scales:
        info = 1 - np.logaddexp(0, -s * sg) / LN2
        if best is None or info.sum() > best.sum():
            best = info
    return best, 1 - np.logaddexp(0, -sg) / LN2


def sym_info(E: np.ndarray, sym: np.ndarray, scales) -> np.ndarray:
    """Per symbol, log2 M - log2 sum_j exp(s (E_j - E_sent)) at the best scale:
    the square-law metric is linear in the energies (cpm.tone_metric)."""
    r = np.arange(len(E))
    best = None
    for s in scales:
        ll = s * E
        info = np.log2(E.shape[1]) - (logsumexp(ll, axis=1) - ll[r, sym]) / LN2
        if best is None or info.sum() > best.sum():
            best = info
    return best


# --- OFDM ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Ofdm:
    tu_ms: float  # useful symbol
    cp_ms: float
    K: int  # pilots on every K-th carrier of a pilot symbol, one carrier on each time
    D: int  # every D-th symbol carries pilots
    mapping: str = "plain"  # plain | pair | idle | dtime | dfreq
    family = "ofdm"

    @property
    def name(self) -> str:
        pil = "" if self.mapping in ("dtime", "dfreq") else f"-k{self.K}d{self.D}"
        return f"{self.mapping}-tu{self.tu_ms:g}-cp{self.cp_ms:g}{pil}"

    @property
    def freqs(self) -> np.ndarray:
        """Carriers in 875..2175 Hz (the w band's TX bandpass), an even number."""
        s = 1000 / self.tu_ms
        n = int((1300 - 2 * max(75, s / 2)) / s + 1e-9) + 1
        n -= n % 2
        return 1525 + s * (np.arange(n) - (n - 1) / 2)

    @property
    def weights(self) -> np.ndarray:
        return {"pair": np.array([1.0, -1.0]), "idle": np.array([1.0, 0.0])}.get(self.mapping, np.array([1.0]))


def pilot_mask(cfg: Ofdm, ns: int, ncl: int) -> np.ndarray:
    s = np.arange(ns)[:, None]
    return (s % cfg.D == 0) & ((np.arange(ncl)[None, :] - s // cfg.D) % cfg.K == 0)


def delay_corr(df: np.ndarray, tau: float) -> np.ndarray:
    """E[H(f + df) H*(f)] for power uniform over delays 0..tau."""
    return np.exp(-1j * np.pi * df * tau) * np.sinc(df * tau)


def wiener(Z, X, cfg: Ofdm, Ts: float, spread: float, tau: float, gain2: float, nvar: float, loo: bool = False):
    """2-D LMMSE estimate of the logical carriers' channel from the pilots
    within the channel's correlation time: (estimate, its error variance,
    half-window in symbols). Rows nearer an end than the window are not
    estimated. `loo`: from every other cell's symbol instead, as if known
    (cfg's K = D = 1): what decision-directed estimation gets once every
    decision is right."""
    ns, ncl = Z.shape
    w = cfg.weights
    u = w ** 2 / np.sum(w ** 2)
    f = cfg.freqs.reshape(ncl, len(w))
    df = f[:, None, :, None] - f[None, :, None, :]  # (a, b, i, j)
    Rf = np.einsum("abij,i,j->ab", delay_corr(df, tau), u, u)
    period = cfg.D * cfg.K
    wt = max(4 * cfg.D, int(np.ceil(0.7 / spread / Ts)))
    while wt > 2 * cfg.D and (2 * wt + 1) / cfg.D * ncl / cfg.K > 500:
        wt -= 1
    hls = Z * np.conj(X)
    est = np.zeros_like(Z)
    mse = np.zeros(Z.shape)
    for p in range(period):
        # the pilots of rows s - wt .. s + wt, s = p (mod period): the mask repeats every period rows
        start = (p - wt) % period
        ds, a = np.nonzero(pilot_mask(cfg, start + 2 * wt + 1, ncl)[start:])
        dt = (ds - wt) * Ts
        A = gain2 * np.exp(-(np.pi * spread * (dt[:, None] - dt[None, :])) ** 2 / 2) * Rf[a[:, None], a[None, :]]
        A[np.diag_indices_from(A)] += nvar
        B = gain2 * np.exp(-(np.pi * spread * dt[None, :]) ** 2 / 2) * Rf[:, a]  # (ncl, pilots)
        Wm = np.linalg.solve(A, B.conj().T).conj().T  # B A^-1 (A Hermitian)
        m = gain2 * Rf.diagonal().real - np.real(np.sum(Wm * B.conj(), axis=1))
        rows = np.arange(p, ns, period)
        rows = rows[(rows >= wt) & (rows < ns - wt)]
        V = hls[rows[:, None] + (ds - wt)[None, :], a[None, :]]
        est[rows] = V @ Wm.T
        mse[rows] = m
        if loo:  # leave each cell's own symbol out: (est - w_ii y_i) / (1 - w_ii), error n w_ii / (1 - w_ii)
            own = np.flatnonzero(ds == wt)
            wii = np.real(Wm[a[own], own])
            est[rows] = (est[rows] - wii * hls[rows]) / (1 - wii)
            mse[rows] = nvar * wii / (1 - wii)
    return est, mse, wt


def run_ofdm(cfg: Ofdm, chan: str, seeds, snrs, dur: float) -> list:
    pre = CHANNELS[chan]
    pts = constellation.load("gray-qam4")
    N, ncp = round(cfg.tu_ms * FS / 1000), round(cfg.cp_ms * FS / 1000)
    T = N + ncp
    f = cfg.freqs
    w = cfg.weights
    nc, g = len(f), len(w)
    ncl = nc // g
    ns = int(dur * FS / T)
    diff = cfg.mapping in ("dtime", "dfreq")
    arms = ("diff",) if diff else ("wiener", "dd", "perfect")
    every = Ofdm(cfg.tu_ms, cfg.cp_ms, 1, 1, cfg.mapping)
    tally = {(snr, arm): Tally() for snr in snrs for arm in arms}
    sir, peak = [], []
    for seed in seeds:
        rng = np.random.default_rng([seed, 7])
        bits = rng.integers(2, size=(ns, ncl, 2))
        X = constellation.modulate(bits.reshape(-1), pts).reshape(ns, ncl)
        if cfg.mapping == "dtime":
            X = np.cumprod(X, axis=0)
        elif cfg.mapping == "dfreq":
            X = np.cumprod(X, axis=1)
        X = X / np.abs(X)
        Xp = (X[:, :, None] * w).reshape(ns, nc)
        n = np.arange(T) - ncp
        x = np.real(np.exp(2j * np.pi * np.outer(n, f) / FS) @ Xp.T).T.reshape(-1)
        x = np.concatenate([np.zeros(PAD), x, np.zeros(PAD)])
        m = max(75.0, 500 / cfg.tu_ms)
        x = tx_condition(x, 0.0, bandpass=(f[0] - m, f[-1] + m), active=slice(PAD, -PAD))
        peak.append(peak_db(x))
        y, g1, g2, d = fade(x, pre, seed)
        starts = PAD + np.arange(ns) * T + ncp - BACKOFF
        dem = np.exp(-2j * np.pi * np.outer(np.arange(N) - BACKOFF, f) / FS)
        Y = (2.0 / N) * (y[starts[:, None] + np.arange(N)[None, :]] @ dem)
        tc = starts + N // 2
        H = (g1[tc][:, None] + g2[tc][:, None] * np.exp(-2j * np.pi * f * d / FS)[None, :]) / np.sqrt(2)
        ref = H * Xp
        alpha = np.vdot(ref, Y) / np.vdot(ref, ref)
        Z = (Y.reshape(ns, ncl, g) * (w / np.sum(w ** 2))).sum(-1)
        Hl = alpha * (H.reshape(ns, ncl, g) * (w ** 2 / np.sum(w ** 2))).sum(-1)
        var_i = float(np.mean(np.abs(Z - Hl * X) ** 2))
        sir.append(10 * np.log10(np.mean(np.abs(Hl) ** 2) / var_i))
        block = (starts // int(BLOCK_S * FS))
        nb = int(block.max()) + 1
        for snr in snrs:
            nv = 4 * sigma(x, snr) ** 2 / N / np.sum(w ** 2)
            Zn = Z + cnoise(np.random.default_rng([seed, 11, int(snr * 10) + 1000]), Z.shape, nv)
            if diff:
                if cfg.mapping == "dtime":
                    uu, b, blk = Zn[1:] * np.conj(Zn[:-1]), bits[1:], np.repeat(block[1:], ncl * 2)
                    n_air, air = np.bincount(block[1:], minlength=nb), T / FS
                else:
                    uu, b, blk = Zn[:, 1:] * np.conj(Zn[:, :-1]), bits[:, 1:], np.repeat(block, (ncl - 1) * 2)
                    n_air, air = np.bincount(block, minlength=nb), T / FS
                uu = uu / np.mean(np.abs(uu))
                one = np.ones(uu.shape)
                info, _ = bit_info(constellation.llr(uu, one + 0j, one, pts), b.reshape(-1), np.geomspace(0.05, 30, 30))
                tally[snr, "diff"].add(info, blk, air, n_air)
                continue
            pil = pilot_mask(cfg, ns, ncl)
            stats = (T / FS, pre.doppler_hz, (pre.delay_ms + 0.5) / 1000, abs(alpha) ** 2, nv + var_i)
            est, mse, wt = wiener(Zn, X, cfg, *stats)
            est_dd, mse_dd, wt_dd = wiener(Zn, X, every, *stats, loo=True)
            for arm, h, var, wt in (("wiener", est, nv + var_i + mse, wt), ("dd", est_dd, nv + var_i + mse_dd, wt_dd),
                                    ("perfect", Hl, np.full(Z.shape, nv + var_i), wt)):
                rows = np.zeros(ns, bool)
                rows[wt:ns - wt] = True
                data = ~pil & rows[:, None]
                blk = np.repeat(np.broadcast_to(block[:, None], data.shape)[data], 2)
                n_air = np.bincount(block[rows], minlength=nb)
                llr = constellation.llr(Zn[data], h[data], var[data], pts)
                info, info1 = bit_info(llr, bits[data].reshape(-1), np.geomspace(0.02, 2, 20))
                tally[snr, arm].add(info, blk, T / FS, n_air, info1)
    n_data = ncl * 2 * (1 if diff else 1 - 1 / (cfg.K * cfg.D))
    base = dict(family="ofdm", name=cfg.name, chan=chan, sir_db=round(float(np.mean(sir)), 1),
                peak_db=round(float(np.mean(peak)), 2))
    return [dict(base, snr=snr, arm=arm, **t.row(n_data / (T / FS))) for (snr, arm), t in tally.items()]


# --- M-FSK ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fsk:
    m: int  # tones
    spacing: float  # Hz
    L: int = 1  # symbol = L / spacing
    family = "fsk"

    @property
    def name(self) -> str:
        return f"fsk{self.m}-s{self.spacing:g}-L{self.L}"


def run_fsk(cfg: Fsk, chan: str, seeds, snrs, dur: float) -> list:
    pre = CHANNELS[chan]
    S, m, L = cfg.spacing, cfg.m, cfg.L
    nsub = round(FS / S)
    T = nsub * L
    f0 = round((1500 - (m - 1) * S / 2) / S) * S
    f = f0 + S * np.arange(m)
    ns = int(dur * FS / T)
    d = int(round(pre.delay_ms * 1e-3 * FS))
    offsets = sorted({0, min(d // 2, nsub // 2)})  # timing: on the first path, or between the two
    arms = ("full",) if L == 1 else ("full", "sub", "subg")
    tally = {(snr, arm, off): Tally() for snr in snrs for arm in arms for off in offsets}
    peak = []
    for seed in seeds:
        rng = np.random.default_rng([seed, 7])
        sym = rng.integers(m, size=ns)
        a = np.repeat(sym.astype(float), T)
        ng = int(0.1 * T)
        if ng > 1:  # cpm.tones' glide
            win = np.hanning(ng + 2)[1:-1]
            a = np.convolve(np.pad(a, ng, mode="edge"), win / win.sum(), mode="same")[ng:-ng]
        x = np.sqrt(2) * np.cos(2 * np.pi * np.cumsum(f0 + a * S) / FS)
        x = np.concatenate([np.zeros(PAD), x, np.zeros(PAD)])
        x = tx_condition(x, 0.0, CLIP_OVERSHOOT + CLIP_OVERSHOOT[-1:] * 3, bandpass=(f0 - 150, f[-1] + 150),
                         active=slice(PAD, -PAD))
        peak.append(peak_db(x))
        y, _, _, _ = fade(x, pre, seed)
        dem = np.exp(-2j * np.pi * np.outer(np.arange(nsub), f) / FS)
        for off in offsets:
            use = ns - 1  # the last symbol's window may run off the end
            starts = PAD + off + np.arange(use) * T
            idx = starts[:, None, None] + (np.arange(L) * nsub)[None, :, None] + np.arange(nsub)[None, None, :]
            Y = (2.0 / nsub) * (y[idx] @ dem)  # (use, L, m); a tone's phase steps by whole turns between sub-windows
            block = starts // int(BLOCK_S * FS)
            n_air = np.bincount(block)
            for snr in snrs:
                nv = 4 * sigma(x, snr) ** 2 / nsub
                Yn = Y + cnoise(np.random.default_rng([seed, 11, int(snr * 10) + 1000, off]), Y.shape, nv)
                for arm in arms:
                    if arm == "full":
                        E = np.abs(Yn.mean(axis=1)) ** 2 / (nv / L)
                    elif arm == "sub":
                        E = (np.abs(Yn) ** 2).sum(axis=1) / nv
                    else:
                        E = (np.abs(Yn[:, 1:]) ** 2).sum(axis=1) / nv
                    tally[snr, arm, off].add(sym_info(E, sym[:use], np.geomspace(1e-5, 1.0, 40)), block, T / FS, n_air)
    base = dict(family="fsk", name=cfg.name, chan=chan, sir_db="", peak_db=round(float(np.mean(peak)), 2))
    out = []
    for snr in snrs:
        for arm in arms:
            best = max((tally[snr, arm, off].row(np.log2(m) * FS / T) for off in offsets), key=lambda r: r["bps"])
            out.append(dict(base, snr=snr, arm=arm, **best))
    return out


# --- the grid ------------------------------------------------------------------------

PILOTS = [(1, 6), (1, 3), (1, 2), (2, 1), (3, 1), (4, 1), (6, 1), (2, 2), (3, 2)]


def candidates() -> list:
    out = []
    for tu in (5, 10, 15, 20, 40):
        for cp in (4, 8):
            for mapping in ("plain", "pair", "idle"):
                for K, D in PILOTS:
                    c = Ofdm(tu, cp, K, D, mapping)
                    if len(c.freqs) // len(c.weights) >= max(K, 2):
                        out.append(c)
            out += [Ofdm(tu, cp, 1, 1, "dtime"), Ofdm(tu, cp, 1, 1, "dfreq")]
    for m in (2, 4, 8, 16, 32):
        for S in (25, 50, 62.5, 100, 125, 200, 250):
            for L in (1, 2, 4, 8):
                if m * S <= 2000 and L / S <= 0.0801:
                    out.append(Fsk(m, S, L))
    return out


def job(args) -> list:
    cfg, chan, seeds, snrs, dur = args
    return (run_ofdm if cfg.family == "ofdm" else run_fsk)(cfg, chan, seeds, snrs, dur)


def check():
    x = np.random.default_rng(0).normal(size=4000)
    p = CHANNELS["hilat-d"]
    assert np.array_equal(fade(x, p, 3)[0], hfchannel.fading(x, p, seed=3)), "fade() is not hfchannel.fading"
    # the bins' noise: 4 sigma^2 / N per bin of (2 / N) sum y exp(-j...)
    nz = np.random.default_rng(1).normal(size=(4000, 160))
    v = np.var((2 / 160) * nz @ np.exp(-2j * np.pi * 1000 * np.arange(160) / FS))
    assert abs(v / (4 / 160) - 1) < 0.1, v
    flat = hfchannel.FadingPreset("slow", 0.05, 0.0)
    CHANNELS["slow"] = flat
    for cfg in (Ofdm(20, 4, 1, 6), Ofdm(20, 4, 2, 1, "pair"), Ofdm(20, 4, 2, 1, "idle"), Ofdm(10, 4, 1, 1, "dtime"),
                Ofdm(40, 4, 1, 1, "dfreq")):
        rows = {(r["arm"], r["snr"]): r for r in run_ofdm(cfg, "slow", [1], [0.0, 30.0], 10.0)}
        for arm in {a for a, _ in rows}:
            hi, lo = rows[arm, 30.0], rows[arm, 0.0]
            print(cfg.name, arm, "0 dB", lo["bps"], "30 dB", hi["bps"], "of", hi["max_bps"], "sir", hi["sir_db"],
                  "peak", hi["peak_db"])
            assert hi["frac"] > 0.8 and lo["bps"] < hi["bps"], (cfg.name, arm)
        if ("wiener", 0.0) in rows:
            assert rows["wiener", 0.0]["bps"] <= rows["perfect", 0.0]["bps"] * 1.02
    for cfg in (Fsk(16, 25), Fsk(8, 100, 4)):
        for r in run_fsk(cfg, "slow", [1], [-5.0, 30.0], 20.0):
            print(cfg.name, r["arm"], r["snr"], r["bps"], "of", r["max_bps"], "peak", r["peak_db"])
            assert r["snr"] < 0 or r["frac"] > 0.9, (cfg.name, r)
    print("ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/fast_fading.csv")
    ap.add_argument("--chans", default="hilat-d,hilat-m,lowlat-d,nvis-d,mpd")
    ap.add_argument("--snrs", default="-5,0,5,10,15,20,25")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--dur", type=float, default=30.0, help="seconds of air per seed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="", help="substring of family or name")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    snrs = [float(s) for s in a.snrs.split(",")]
    cands = [c for c in candidates() if a.only in f"{c.family} {c.name}"]
    jobs = [(c, ch, list(range(1, a.seeds + 1)), snrs, a.dur) for c in cands for ch in a.chans.split(",")]
    print(f"{len(cands)} candidates, {len(jobs)} jobs", file=sys.stderr)
    with open(a.out, "w", newline="") as fh, Pool(a.workers) as pool:
        wr = csv.DictWriter(fh, FIELDS)
        wr.writeheader()
        for i, rows in enumerate(pool.imap_unordered(job, jobs)):
            wr.writerows(rows)
            fh.flush()
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
