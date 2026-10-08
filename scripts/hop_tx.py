"""Hopped fsk8r50 (docs/hopping-fsk.md): the transmitter, its variants, and
the channels the study judges them on. Imported by hop_study.py,
hop_tx_shape.py and hop_report.py; a study, not a mode.

One tone at a time (constant envelope, fsk8r50's rate and code), each
symbol in one of K copies of the 8-tone grid by a fixed sequence:

    k1         one copy: fsk8r50 as is
    k4         four copies, interleaved tone by tone (copy c on tones 4j + c)
    k4b        four copies side by side
    g2-<Hz>    two copies, the second <Hz> above the first

TX parameters (a variant is mode[/key=val...], hop_study.py): bp (bandpass
margin past the outer tones, Hz), clip (clip headroom, dB), passes
(clip-and-filter passes, overshoot as config.CLIP_OVERSHOOT with its last
factor repeated), glide (the frequency trajectory smoothed over this
fraction of a symbol), dwell (symbols per hop), split (g2: one passband
per copy). DEFAULT is the filter fsk8r50 shipped before cpm.TX_FILTERS.
"""

import numpy as np
from scipy import signal

from data2g import cpm, codes, hfchannel
from data2g.config import CLIP_OVERSHOOT, FS
from data2g.waveform.dsp import tx_condition

G = cpm.GRIDS["c8r50"]
SPEC = cpm.SPECS["fsk8r50-r1/2"]
DEFAULT = dict(bp=50.0, clip=0.0, passes=3, glide=0.0, dwell=1, split=0)


def layout(mode):
    """-> (copies, each copy's lowest bin (block layouts), "block" | "comb")"""
    if mode == "k1":
        return 1, np.array([0]), "block"
    if mode == "k4":
        return 4, None, "comb"
    if mode == "k4b":
        return 4, np.arange(4) * G.m, "block"
    if mode.startswith("g2-"):
        return 2, np.array([0, int(float(mode[3:]) / G.rate)]), "block"
    raise ValueError(mode)


def fidx(mode, s, c):
    """Tone s of copy c -> its bin above the lowest."""
    K, off, lay = layout(mode)
    return K * s + c if lay == "comb" else off[c] + s


def overshoot(n):
    return tuple(CLIP_OVERSHOOT) + tuple(CLIP_OVERSHOOT[-1:]) * (n - len(CLIP_OVERSHOOT))


def tx(mode, p, seed):
    """One r1/2 codeword (seeded payload) -> (audio, its active slice, meta)."""
    p = {**DEFAULT, **p}
    rng = np.random.default_rng(seed)
    payload = rng.bytes(codes.payload_bytes(SPEC))
    sym = cpm.to_tones(G, codes.encode(SPEC, payload))
    L = len(sym)
    K, off, lay = layout(mode)
    c = (np.arange(L) // int(p["dwell"])) % K
    fi = fidx(mode, sym, c)
    span = int(fidx(mode, np.array([G.m - 1]), np.array([K - 1]))[0]) + 1
    T = G.T
    f0 = round((G.center - (span - 1) * G.rate / 2) / G.rate) * G.rate
    a = np.repeat(fi.astype(float), T)
    n = int(p["glide"] * T)
    if n > 1:
        w = np.hanning(n + 2)[1:-1]
        a = np.convolve(np.pad(a, n, mode="edge"), w / w.sum(), mode="same")[n:-n]
    x = np.sqrt(2) * np.cos(2 * np.pi * np.cumsum(f0 + a * G.rate) / FS)
    pad = 400
    x = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    act = slice(pad, pad + L * T)
    if p["split"] and lay == "block":
        bands = [(f0 + o * G.rate - p["bp"], f0 + (o + G.m - 1) * G.rate + p["bp"]) for o in off]
        x = _clip_filter_multi(x, p["clip"], overshoot(int(p["passes"])), act, bands)
    else:
        x = tx_condition(x, p["clip"], overshoot=overshoot(int(p["passes"])), active=act,
                         bandpass=(f0 - p["bp"], f0 + (span - 1) * G.rate + p["bp"]))
    return x, act, dict(payload=payload, L=L, c=c, f0=f0, span=span)


def _clip_filter_multi(x, clip_db, ov, act, bands):
    """dsp.tx_condition with the sum of band filters (one per copy)."""
    thresh = np.sqrt(2 * np.mean(x[act] ** 2)) * 10 ** (clip_db / 20)
    taps = sum(signal.firwin(201, b, fs=FS, pass_zero=False) for b in bands)
    for k in ov:
        z = signal.hilbert(x)
        scale = np.minimum(1.0, thresh / np.maximum(np.abs(z), 1e-12)) ** k
        x = np.convolve(np.real(z * scale), taps, mode="same")
    return x / np.sqrt(np.mean(x[act] ** 2))


def papr(x, act):
    """-> (envelope peak over average, dB; the peak power, a unit-RMS sinusoid's = 1)"""
    e = np.abs(signal.hilbert(x))[act] ** 2
    return 10 * np.log10(e.max() / 2 / np.mean(x[act] ** 2)), e.max() / 2


# --- channels ------------------------------------------------------------------------

def draw_paths(rng):
    """The ensemble: 2 paths (70%) or 3; extra delays log-uniform 0.1-5 ms,
    extra path powers uniform -12..0 dB, Doppler log-uniform 0.05-2 Hz (all
    paths). -> ([(delay ms, power dB)] with the first path (0, 0), Doppler)"""
    n = 2 if rng.random() < 0.7 else 3
    d = np.sort(np.exp(rng.uniform(np.log(0.1), np.log(5.0), n - 1)))
    p = rng.uniform(-12, 0, n - 1)
    dop = float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))
    return [(0.0, 0.0)] + list(zip(d, p)), dop


def multipath(x, paths, dop, rng):
    """Independent Rayleigh paths (hfchannel's gaussian Doppler), unit power in expectation."""
    z = hfchannel._analytic(x)
    y = np.zeros_like(z)
    tot = 0.0
    for tau, pdb in paths:
        d = int(round(tau * 1e-3 * FS))
        a = 10 ** (pdb / 20)
        g = hfchannel._gaussian_taps(len(z), dop, rng)
        y += a * g * np.concatenate([np.zeros(d, complex), z[:len(z) - d]])
        tot += a * a
    return np.real(y / np.sqrt(tot))


def channel(x, ch, seed):
    """'awgn' | 'grid:<delay ms>:<Doppler Hz>' (two equal paths) | 'ens'
    (draw_paths from the seed: the same channel for every variant)."""
    rng = np.random.default_rng(seed + 1)
    if ch == "awgn":
        return x
    if ch.startswith("grid:"):
        tau, dop = map(float, ch.split(":")[1:])
        return multipath(x, [(0.0, 0.0), (tau, 0.0)], dop, rng)
    paths, dop = draw_paths(rng)
    return multipath(x, paths, dop, rng)
