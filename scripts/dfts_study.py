"""DFT-spread OFDM (SC-FDMA) on the existing frame: is it worth a mode family?

The question (2026-10-19): OFDM with BICM gets frequency diversity only
through the code, so high-rate codes on few carriers run out of it (every
n10 mode above 16qam-r1/2 never reaches its 10% point on mpp or mpd), and
the clip noise that fades with each carrier caps SINR on fading. A unitary
DFT across each data symbol's carriers before the IFFT makes the data a
single carrier: lower PAPR before the clipper, and an MMSE equalizer that
collects the band's diversity before decoding. Everything else is today's
burst: preamble, header (and its copy), pilots, CP, the channel estimate.

TX: the data symbols of each OFDM symbol, s (nc,), go on the carriers as
X = F s / sqrt(nc) (localized, contiguous carriers), then burst_waveform
and tx_condition as usual (no ACE: its regions are per carrier).

RX: modem.receive (sync, header, pilots, equalizer.estimate) as for OFDM,
then per data symbol MMSE-FDE W_k = h_k* / (|h_k|^2 + v_k) with v_k the
thermal + clip + estimate noise noise_var uses, s_hat = F^H (W Y) sqrt(nc),
and the unbiased model s_hat = mu s + e, mu = mean(W h), var(e) =
mean(|W h|^2) - mu^2 + mean(|W|^2 v), into constellation.llr. No DD on
either arm (both would gain from it; this compares the waveforms).

Clip constants (Bussgang gain, SDR) and the PEP peak are measured here for
every arm the same way (noiseless loopback through modem.receive), since a
DFT-spread burst is not Gaussian on the carriers and the shipped table is
OFDM's. PEP-fair threshold = average-power threshold + peak (clip_constants
.py's peak: mean over bursts of the burst's max envelope power over its
mean, active part).

Threshold: the average SNR (2500 Hz) where FAIL of the codewords fail,
bursts of about 16 frames (codewords spread over them, as CANDIDATES.md),
a burst whose header fails counting all its codewords failed. Common
random numbers: trial k has the same seed at every SNR and on every arm.

Found (2026-10-19/20; runs/dfts_round1.sh, dfts_round2.sh; PEP-fair 10%
points, no DD, so a few tenths off mode_thresholds.json):

- The n10 fading floor is the channel estimate, not clipping or diversity.
  Noiseless and unclipped on mpp, the pilots' estimate held n10 to a median
  16.5 dB SINR (p10 10.4): freq_smooth's tight support drops a 2 ms second
  path 10 carriers can't resolve. Padding it (SUPPORT_SLACK 20) gave 26.9
  (p10 24.8). mpd's floor (median 19.6, p10 9.7) did not move: its 4 ms
  path sits on the CP.
- Padding everywhere costs low SNR (+0.2 to +0.75 dB on some awgn/mpg/mpp
  cells, as the rejected narrow pad did). /sauto (auto_estimate: the padded
  order when the energy it adds per dimension tops AUTO_K x the noise; 1.5
  to 3 alike) cost nothing in any of 56 cells (14 modes x 4 channels) and
  gave, on mpp: n10-16qam-r1/2 12.08 -> 8.89, n10-qpsk-r3/4 9.80 -> 7.55,
  n10-16qam-r2/3 never -> 12.71, r3/4 never -> 15.15, n10-64l-r3/4 never ->
  21.73, n10-256l-r3/4 never -> 30.45; on mpg 64l -0.94, 256l -1.13. n4 and
  the 1200 Hz modes unchanged (n4's rank cap; w resolves the paths).
- The same off the presets' round delays (runs/support_delays.sh: the
  second path 0.5-1.5x the preset's, per trial; no DD), tight-only /s4
  against shipped support_slack, 10% points: mpp~ n10-qpsk-r1/2 1.56 ->
  1.00, 16qam-r1/2 never -> 6.13, r2/3 never -> 9.75, 64l never -> 15.63;
  mpd~ qpsk-r1/2 4.56 -> 4.19, 16qam-r1/2 never -> 11.69; mpg~ 64l 20.06 ->
  19.13; n10-qpsk-r1/5 and every other cell unchanged. Not an artefact of
  mpp's 2 ms (n10's resolution exactly), and it reaches mpd once its path
  is off the CP's 4 ms.
- What estimation costs the bottom and middle (runs/genie_gap.sh: 10% points
  on the estimate against the true channel, delays drawn per trial, no DD):
  0-0.75 dB on awgn and mpg~ everywhere; on mpp~/mpd~ the low-rate modes
  lose most: qpsk-r1/5 1.69/1.88, w48-qpsk-r1/5 1.69/1.88, n4-qpsk-r1/3
  1.50/1.69, n10-qpsk-r1/5 1.12/0.75, qpsk-r1/3 1.50/0.56; rate 1/2 and up
  0-0.75; the polar ACKs 0-0.75. DD (/dd) recovers 0-0.75 of it; left
  with DD on mpp~/mpd~: qpsk-r1/5 1.50/1.12, w48-qpsk-r1/5 1.50/1.50,
  n4-qpsk-r1/3 1.12/1.69, n4-qpsk-r1/2 1.50/never.
- Most of that was the delay support reading noise peaks as paths: within
  -15 dB of the strongest, but near 0 dB per carrier the profile's noise
  floor is ~10 dB under a lone path (w, awgn -5.5 dB: support ~100
  samples, rank 20 of 24). delay_support's noise gate (SUPPORT_NOISE_Z,
  runs/support_gate.sh, with DD, paired): qpsk-r1/5 -0.56/-0.56/-1.12/
  -0.75 dB (awgn/mpg~/mpp~/mpd~), w48-qpsk-r1/5 -0.56/-0.75/-1.31/-1.12,
  qpsk-r1/3 mpg~ -0.94, n10 and n4 up to -0.75; one cell +0.19 (n10-16qam
  -r1/3 mpg~, one step); rate 1/2 and up on w/w48 unchanged. Left against
  the genie (no DD there): n4-qpsk-r1/3 mpp~/mpd~ 1.12/1.69, n4-qpsk-r1/2
  mpp~ 0.94, the rest 0.75 or less. TIME_TAPS 8 (/t8) on top: nothing.
- DFT-spread against the better OFDM arm: awgn -0.6 to -1.35 dB PEP-fair
  (2.4-5 dB less clip noise at a headroom, so it clips harder), n10 mpg
  -0.2 to -0.8; but mpp 0 to +2 dB worse, w48 mpg +0.2 to +3, mpd worse or
  never (linear MMSE-FDE's noise enhancement in notches; BICM over long
  spread codewords does better here). Not pursued: a flat-channel niche at
  best, unless turbo equalization wins the fading back.

    uv run python scripts/dfts_study.py consts --modes n10-16qam-r2/3 --headroom 0 2 4
    uv run python scripts/dfts_study.py thresholds --modes n10-16qam-r2/3 \\
        --arms ofdm dfts@2 --channels awgn mpp --out runs/dfts.csv
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
import dataclasses
import os
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

from data2g import codes, constellation, equalizer, hfchannel, modem
from data2g.config import BANDS, FS, LEADIN_SAMPLES, LEADOUT_SAMPLES, NSYM, SUBMODES, SYMS_PER_FRAME
from data2g.waveform import dsp, ofdm

CHANNELS = ("awgn", "mpg", "mpp", "mpd")
# "<preset>~": the preset's Doppler, its second path's delay drawn per trial
# uniformly over DELAY_SPREAD x the preset's, so no result hangs on a round
# delay (mpp's 2 ms is exactly n10's delay resolution, 16 samples).
DELAY_SPREAD = (0.5, 1.5)


def fading(chan: str, rng):
    """apply_channel's fading_preset for `chan`."""
    if chan == "awgn":
        return None
    if not chan.endswith("~"):
        return chan
    p = hfchannel.FADING_PRESETS[chan[:-1]]
    return hfchannel.FadingPreset(chan, p.doppler_hz, p.delay_ms * float(rng.uniform(*DELAY_SPREAD)))
FRAMES = 16  # burst length the codewords spread over, about
TRIALS = 100  # bursts per SNR point
CONST_BURSTS = 24


# --- arms -----------------------------------------------------------------------

def parse_arm(arm: str, spec) -> tuple[bool, float]:
    """'ofdm' (the shipped mode), 'ofdm@H' (its headroom H, ACE as shipped),
    'dfts' (the mode's headroom) or 'dfts@H' -> (dft-spread, headroom). A
    '/sN' suffix (receiver only) is use_slack's; '/genie' decodes on the
    true channel (true_channel), the sync and the noise estimate as shipped;
    '/dd' (ofdm only, last) decodes through arq.phy.ModemRx, DD unbudgeted;
    '/q0' turns delay_support's noise gate off, '/tN' sets TIME_TAPS N."""
    kind, _, h = arm.split("/")[0].partition("@")
    return kind == "dfts", float(h) if h else spec.headroom


def use_slack(arm: str) -> None:
    """The arm's support order: '/sN' a fixed order N (/s4: the receiver
    before support_slack); '/sauto' this study's own per-burst pick
    (auto_estimate, as support_slack shipped); none: the shipped receiver."""
    _, _, s = arm.partition("/s")
    equalizer.estimate = auto_estimate if s == "auto" else SHIPPED_ESTIMATE
    fixed = int(s) if s and s != "auto" else None
    tight, padded = (fixed, fixed) if fixed is not None else (SHIPPED_SLACK, SHIPPED_PADDED)
    if (equalizer.SUPPORT_SLACK, equalizer.PADDED_SLACK) != (tight, padded):
        equalizer.SUPPORT_SLACK, equalizer.PADDED_SLACK = tight, padded
        equalizer._support_basis.cache_clear()


def use_rx(arm: str) -> None:
    """The arm's other receiver knobs: '/q0' (SUPPORT_NOISE_Z off) and '/tN' (TIME_TAPS)."""
    opts = arm.split("/")[1:]
    equalizer.SUPPORT_NOISE_Z = 0.0 if "q0" in opts else SHIPPED_Z
    equalizer.TIME_TAPS = next((int(o[1:]) for o in opts if o[:1] == "t" and o[1:].isdigit()), SHIPPED_TAPS)


SHIPPED_Z = equalizer.SUPPORT_NOISE_Z
SHIPPED_TAPS = equalizer.TIME_TAPS
SHIPPED_SLACK = equalizer.SUPPORT_SLACK
SHIPPED_PADDED = equalizer.PADDED_SLACK
SHIPPED_ESTIMATE = equalizer.estimate
AUTO_SLACK = 20  # the padded support auto_estimate may pick
AUTO_K = 2.0  # pick it when the energy it adds per dimension exceeds AUTO_K x the noise


def _basis(bb: np.ndarray, support: tuple[int, int], slack: int) -> np.ndarray:
    """equalizer._support_basis at another slack."""
    d = np.arange(support[0] - slack, support[1] + slack + 1)
    U, s, _ = np.linalg.svd(np.exp(-2j * np.pi * np.outer(bb, d) / FS), full_matrices=False)
    return U[:, :min(int(np.sum(s > s[0] * 1e-2)), len(bb) - 2)]


def auto_estimate(h_pilot, support, bb=equalizer.BB_FREQS, n0_pre=np.inf, n0_pre_k=None):
    """equalizer.estimate on the shipped support, or the padded one when the
    pilots hold more energy in the dimensions padding adds than its noise
    would cost there (Mallows' Cp between the two orders): per added
    dimension, the pilots' mean energy outside the tight subspace against
    the thermal noise (the preamble's, else the tight residual's)."""
    Ut, Up = _basis(bb, support, SHIPPED_SLACK), _basis(bb, support, AUTO_SLACK)
    extra = Up.shape[1] - Ut.shape[1]
    pick = SHIPPED_SLACK
    if extra > 0:
        hp = (h_pilot @ np.conj(Up)) @ Up.T
        d = hp - (hp @ np.conj(Ut)) @ Ut.T
        n0 = n0_pre
        if not np.isfinite(n0):
            ht = (h_pilot @ np.conj(Ut)) @ Ut.T
            n0 = float(np.mean(np.abs(h_pilot - ht) ** 2) * len(bb) / (len(bb) - Ut.shape[1]))
        if np.sum(np.abs(d) ** 2) / len(h_pilot) / extra > AUTO_K * n0:
            pick = AUTO_SLACK
    equalizer.SUPPORT_SLACK = pick
    equalizer._support_basis.cache_clear()
    try:
        return SHIPPED_ESTIMATE(h_pilot, support, bb, n0_pre, n0_pre_k)
    finally:
        equalizer.SUPPORT_SLACK = SHIPPED_SLACK
        equalizer._support_basis.cache_clear()


def arm_spec(spec, headroom: float):
    return dataclasses.replace(spec, clip_headroom_db=headroom)


def n_codewords(spec) -> int:
    return max(1, round(FRAMES / spec.frames_per_cw))


# --- transmit -------------------------------------------------------------------

def precode(data: np.ndarray) -> np.ndarray:
    """(..., nc) data symbols -> the carriers' values, unitary."""
    return np.fft.fft(data, axis=-1) / np.sqrt(data.shape[-1])


def tx(bits: np.ndarray, spec, dfts: bool, headroom: float) -> tuple[np.ndarray, np.ndarray]:
    """Coded bits in burst order -> (unit-RMS audio, the carriers' data values (n_f, 5, nc))."""
    s = arm_spec(spec, headroom)
    b = ofdm.band(spec.band)
    data = constellation.modulate(bits, constellation.load(spec.constellation)).reshape(-1, 5, b.nc)
    if not dfts:
        return modem.modulate_bits(bits, s), data
    X = precode(data)
    x = modem.burst_waveform(X, s)
    y = dsp.tx_condition(x, headroom, b.spec.clip_overshoot, active=slice(LEADIN_SAMPLES, len(x) - LEADOUT_SAMPLES),
                         bandpass=BANDS[spec.sync_band].tx_bandpass)
    return y, X


def make_burst(spec, n_cw: int, rng):
    pb = codes.payload_bytes(spec)
    payloads = [rng.bytes(pb) for _ in range(n_cw)]
    bits = np.stack([codes.encode(spec, p, 0, index=i) for i, p in enumerate(payloads)])
    return payloads, codes.spread(bits, spec.bits_per_cu)


# --- clip constants, measured the same way for every arm -------------------------

@lru_cache(maxsize=None)
def consts(name: str, dfts: bool, headroom: float) -> tuple[float, float, float]:
    """(gain over the pilots' channel, clip noise ratio, peak dB) of this arm,
    from noiseless loopback bursts of n_codewords codewords."""
    spec = SUBMODES[name]
    rng = np.random.default_rng(12345)
    n_cw = n_codewords(spec)
    ys, hxs, peaks = [], [], []
    for _ in range(CONST_BURSTS):
        _, bits = make_burst(spec, n_cw, rng)
        x, X = tx(bits, spec, dfts, headroom)
        a = dsp.envelope(x[LEADIN_SAMPLES:len(x) - LEADOUT_SAMPLES]) ** 2
        peaks.append(a.max() / a.mean())
        r = modem.receive(np.concatenate([np.zeros(FS // 4), x, np.zeros(FS // 4)]), bands=[spec.sync_band])
        ys.append(modem.data_symbols(r["raw"]).ravel())
        hxs.append((r["est"]["h"] / r["est"]["gain"] * X).ravel())
    Y, hx = np.concatenate(ys), np.concatenate(hxs)
    g = np.vdot(hx, Y) / np.vdot(hx, hx).real
    ratio = np.sum(np.abs(Y - g * hx) ** 2) / np.sum(np.abs(g * hx) ** 2)
    return float(abs(g)), float(ratio), float(10 * np.log10(np.mean(peaks)))


# --- receive --------------------------------------------------------------------

def true_channel(r, spec, x, lead: int, kw: dict) -> np.ndarray:
    """The channel the data saw at every data cell, (n_f, 5, nc), in r's
    units (pilot-referenced, before the clip gain): a pilot-only twin of the
    burst (eq_floor.py's) through the same seeded channel without noise,
    demodulated on r's timing, scaled per carrier to the real burst's
    noiseless pilots (the twin skips the clipper and the TX bandpass)."""
    b, sb = ofdm.band(spec.band), ofdm.band(spec.sync_band)
    n_f = len(r["raw"])
    n_air = modem.frames_on_air(spec, r["n_cw"])
    syms = np.tile(b.pilot, (n_air * SYMS_PER_FRAME + 1, 1))
    hdr = np.tile(sb.pilot, (modem.header_samples(spec.sync_band) // NSYM, 1))
    twin = np.concatenate([np.zeros(LEADIN_SAMPLES), sb.preamble_waveform(), sb.modulate_symbols(hdr),
                           b.modulate_symbols(syms), np.zeros(LEADOUT_SAMPLES)])
    pad = lambda v: np.concatenate([np.zeros(lead), v, np.zeros(FS // 2 + len(x) - len(v))])  # noqa: E731

    def cells(v):
        z = dsp.freq_correct(dsp.to_baseband(hfchannel.apply_channel(pad(v), **kw)), r["cfo"])
        raw = modem._demod_frames(z, r["p0"], n_air, r["shift"], r["phi_ref"], r["steps"], band=spec.band,
                                  slips_in=r["slips"])[0]
        return np.delete(raw, r["kc"], axis=0) if n_air > n_f else raw

    ht, clean = cells(twin) / b.pilot, cells(x)
    hp_t, hp_c = ht[:, 0], clean[:, 0] / b.pilot
    g = np.sum(hp_c * np.conj(hp_t), axis=0) / np.sum(np.abs(hp_t) ** 2, axis=0)
    return ht[:, 1:] * g


def soft(r, spec, dfts: bool, gain: float, ratio: float, h_true=None) -> np.ndarray:
    """(n_cw, coded_bits) soft bits on this arm's clip constants; on
    `h_true` (true_channel's) with no estimation error when given."""
    est = r["est"]
    g0 = est["gain"]
    h = (est["h"] / g0 if h_true is None else h_true) * gain
    mse = 0.0 if h_true is not None else est["mse"] / g0**2 * gain**2
    var = est.get("n0_k", est["n0"]) + ratio * np.abs(h) ** 2 + mse
    Y = modem.data_symbols(r["raw"])
    pts = constellation.load(spec.constellation)
    if not dfts:
        llr = constellation.llr(Y, h, var, pts)
    else:
        nc = Y.shape[-1]
        W = np.conj(h) / (np.abs(h) ** 2 + var)
        s = np.fft.ifft(W * Y, axis=-1) * np.sqrt(nc)
        wh = (W * h).real
        mu = wh.mean(axis=-1, keepdims=True)
        ev = (wh**2).mean(axis=-1, keepdims=True) - mu**2 + (np.abs(W) ** 2 * var).mean(axis=-1, keepdims=True)
        llr = constellation.llr(s, np.broadcast_to(mu, s.shape).astype(complex), np.broadcast_to(np.maximum(ev, 1e-12), s.shape), pts)
    return np.asarray(codes.despread(llr, r["n_cw"], spec.bits_per_cu))


def trial(args) -> int:
    """Codewords that failed in one burst."""
    name, arm, chan, snr, seed = args
    spec = SUBMODES[name]
    dfts, hr = parse_arm(arm, spec)
    use_slack("")  # clip constants on the shipped estimator, as clip_constants.py
    use_rx("")
    gain, ratio, _ = consts(name, dfts, hr)
    use_slack(arm)
    use_rx(arm)
    n_cw = n_codewords(spec)
    rng = np.random.default_rng(seed)
    payloads, bits = make_burst(spec, n_cw, rng)
    x, _ = tx(bits, spec, dfts, hr)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    cfo = float(rng.uniform(-50, 50))
    kw = dict(freq_offset_hz=cfo, ppm=10, fading_preset=fading(chan, rng), seed=seed)
    y = hfchannel.apply_channel(np.concatenate([np.zeros(lead), x, np.zeros(FS // 2)]), snr_db=snr, **kw)
    try:
        r = modem.receive(y, bands=[spec.sync_band], head=FS + FS // 4 + modem.receive_head(spec.sync_band))
    except Exception:  # noqa: BLE001 (no lock, or a header read past the audio)
        return n_cw
    if r["spec"].name != name or r["n_cw"] != n_cw:
        return n_cw
    if arm.endswith("/dd"):  # the ARQ's receiver: DD unbudgeted, its own clip constants
        from data2g.arq import phy

        rx = phy.ModemRx(r, {})
        return sum(p not in rx.raw(i) for i, p in enumerate(payloads))
    h_true = true_channel(r, spec, x, lead, kw) if arm.endswith("/genie") else None
    got = codes.decode_many(spec, soft(r, spec, dfts, gain, ratio, h_true))
    return sum(not (ok and p == q) for (p, ok), q in zip(got, payloads))


def per(pool, name, arm, chan, snr, seed0) -> float:
    n_cw = n_codewords(SUBMODES[name])
    fails = pool.map(trial, [(name, arm, chan, snr, seed0 + k) for k in range(TRIALS)])
    return sum(fails) / (TRIALS * n_cw)


def threshold(pool, name, arm, chan, fail, start, seed0=0, cache=None) -> float:
    """Lowest SNR (to 0.25 dB) with PER <= fail; nan past 40 dB."""
    cache = {} if cache is None else cache

    def ok(snr):
        if snr not in cache:
            cache[snr] = per(pool, name, arm, chan, snr, seed0)
        return cache[snr] <= fail

    hi = start
    while not ok(hi):
        hi += 3
        if hi > 40:
            return float("nan")
    lo = hi - 3
    while ok(lo):
        hi, lo = lo, lo - 3
    while hi - lo > 0.25:
        mid = (hi + lo) / 2
        hi, lo = (mid, lo) if ok(mid) else (hi, mid)
    return hi


def _consts_job(a):
    name, dfts, hr = a
    return a, consts(name, dfts, hr)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("consts", help="clip gain, SDR and peak per arm")
    c.add_argument("--modes", nargs="+", required=True)
    c.add_argument("--headroom", type=float, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    c.add_argument("--jobs", type=int, default=4)
    t = sub.add_parser("thresholds")
    t.add_argument("--modes", nargs="+", required=True)
    t.add_argument("--arms", nargs="+", default=["ofdm", "dfts"])
    t.add_argument("--channels", nargs="+", default=list(CHANNELS))
    t.add_argument("--fail", type=float, default=0.1)
    t.add_argument("--start", type=float, default=None, help="search start (dB); default the shipped threshold - 2")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--jobs", type=int, default=4)
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    with Pool(a.jobs) as pool:
        if a.cmd == "consts":
            jobs = [(m, d, h) for m in a.modes for h in a.headroom for d in (False, True)]
            for (m, d, h), (g, ratio, peak) in pool.imap(_consts_job, jobs):
                print(f"{m:16s} {'dfts' if d else 'ofdm'} h{h:g}: gain {g:.3f}  SDR {-10 * np.log10(ratio):5.2f} dB  "
                      f"peak {peak:5.2f} dB", flush=True)
            return
        import json
        from pathlib import Path

        shipped = json.loads((Path(__file__).parents[1] / "data2g/codes_data/mode_thresholds.json").read_text())
        done = {(r["mode"], r["arm"], r["channel"]) for r in csv.DictReader(open(a.out))} if os.path.exists(a.out) else set()
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="") as f:
            w = csv.DictWriter(f, ["mode", "arm", "channel", "avg_db", "peak_db", "pep_db"])
            if new:
                w.writeheader()
            for name in a.modes:
                spec = SUBMODES[name]
                for arm in a.arms:
                    dfts, hr = parse_arm(arm, spec)
                    _, _, peak = consts(name, dfts, hr)
                    for ch in a.channels:
                        if (name, arm, ch) in done:
                            continue
                        ref = shipped["modes"].get(name, [0] * 4)[CHANNELS.index(ch.rstrip("~"))]
                        start = a.start if a.start is not None else (ref - 2 if ref < 90 else 15.0)
                        th = threshold(pool, name, arm, ch, a.fail, round(start * 4) / 4, a.seed)
                        row = dict(mode=name, arm=arm, channel=ch, avg_db=th, peak_db=round(peak, 2),
                                   pep_db=round(th + peak, 2))
                        w.writerow(row)
                        f.flush()
                        print(row, flush=True)


if __name__ == "__main__":
    main()
