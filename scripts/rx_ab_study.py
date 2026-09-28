"""Receiver-only A/B from modem73 and aicodix/modem, paired on the same
audio: 8-codeword bursts near each mode's 10% point, codeword failures
per variant.

- base: today's receiver.
- gate (modem73 robust_modem.hh:2182-2195): per carrier and pilot pair,
  r = |h_p + h_p+1|^2 / (2 (|h_p|^2 + |h_p+1|^2)), over its median across
  carriers; the frame's LLRs on that carrier x 1 if r/med >= 0.7, else
  max(2 (r/med)^2, 0.05). Catches the channel slewing between pilots,
  which the burst-wide LMMSE error (one Doppler for the burst) doesn't.
- median (aicodix decode.cc:245-309, Theil-Sen): the phase-slope
  estimators (bin phase step for clock tracking, residual CFO) as circular
  medians of the per-pair phase differences, not power-weighted phasor
  sums. Changes receive() itself (patched for that pass).

    uv run --no-sync python scripts/rx_ab_study.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import codes, equalizer, hfchannel, modem
from data2g.arq import phy as PHY
from data2g.config import FS
from data2g.equalizer import FRAME_S
from data2g.tnc import receive_any
from data2g.waveform.dsp import freq_correct, to_baseband

sys.path.insert(0, str(Path(__file__).parent))
import outcome_data as O  # noqa: E402

NAMES = ("qpsk-r1/5", "qpsk-r1/2", "n10-qpsk-r1/5", "n10-qpsk-r1/2", "w48-qpsk-r1/2", "16qam-r1/2")
CHANNELS = ("mpp", "mpd", "mpg", "awgn")
OFFSETS = (-1.0, 0.0)  # dB from the smallest burst's 10% point
N_CW = 8
VARIANTS = ("base", "gate", "median")


def _cmedian(a: np.ndarray, w: np.ndarray | None = None) -> float:
    """Circular median of angles `a` around their (weighted) mean direction."""
    m0 = np.angle(np.sum(np.exp(1j * a) if w is None else w * np.exp(1j * a)))
    return float(m0 + np.median(np.angle(np.exp(1j * (a - m0)))))


def median_bin_phase_step(h):
    return _cmedian(np.angle(h[1:] * np.conj(h[:-1])))


def median_residual_cfo(h_pilot):
    return _cmedian(np.angle(h_pilot[1:] * np.conj(h_pilot[:-1])).ravel()) / (2 * np.pi * FRAME_S)


def pilots(y, r):
    """The air frame pilots (n_air + 1, nc) as receive() last demodulated them."""
    spec = r["spec"]
    n_f = r["n_cw"] * spec.frames_per_cw
    n_air = n_f + (modem.copy_frame(spec.sync_band, n_f) is not None)
    z = freq_correct(to_baseband(y), r["cfo"])
    _, hp, _ = modem._demod_frames(z, r["p0"], n_air, r["shift"], r["phi_ref"], steps_in=r["steps"], band=spec.band)
    return hp


def gate_soft(y, r):
    """Soft bits with modem73's local decorrelation gate on the LLRs."""
    spec, est = r["spec"], r["est"]
    hp = pilots(y, r)
    rr = np.abs(hp[:-1] + hp[1:]) ** 2 / (2 * (np.abs(hp[:-1]) ** 2 + np.abs(hp[1:]) ** 2) + 1e-30)
    rel = rr / (np.median(rr, axis=1, keepdims=True) + 1e-30)
    w = np.where(rel >= 0.7, 1.0, np.maximum(2 * rel**2, 0.05))  # (n_air, nc): frame f between pilots f, f+1
    n_f = r["n_cw"] * spec.frames_per_cw
    kc = modem.copy_frame(spec.sync_band, n_f)
    if kc is not None:
        w = np.delete(w, kc, axis=0)
    var = (modem.noise_var(est["h"], est) + est["mse"]) / w[:, None, :]
    return np.asarray(codes.despread(modem.soft_bits(r["raw"], est["h"], var, spec), r["n_cw"], spec.bits_per_cu))


def fails(r, soft, b) -> int:
    if r is None or r["spec"].name != b.submode or r["n_cw"] != N_CW:
        return N_CW
    masks = [PHY.mask_value(s.mask_id) for s in b.slots]
    dec = codes.decode_many(r["spec"], soft, masks, index=0)
    return sum(not (ok and p == s.payload) for (p, ok), s in zip(dec, b.slots))


def rx(y):
    try:
        return receive_any(y, lead=FS)
    except Exception:  # noqa: BLE001 (a header read past the audio: a miss)
        return None


def trial(args):
    """-> codeword failures per variant."""
    name, chan, snr, seed = args
    rng = np.random.default_rng(seed)
    b = O.burst(name, N_CW, rng)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    x = np.concatenate([np.zeros(lead), PHY.tx_audio(b), np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=float(rng.uniform(-50, 50)), ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    r = rx(y)
    ok_r = r is not None and r.get("family") != "cpm" and r["spec"].name == name and r["n_cw"] == N_CW
    out = {"base": fails(r, PHY.soft_bits(r), b) if ok_r else N_CW,
           "gate": fails(r, gate_soft(y, r), b) if ok_r else N_CW}
    real = (modem._bin_phase_step, equalizer.residual_cfo)
    modem._bin_phase_step, equalizer.residual_cfo = median_bin_phase_step, median_residual_cfo
    try:
        rm = rx(y)
    finally:
        modem._bin_phase_step, equalizer.residual_cfo = real
    ok_m = rm is not None and rm.get("family") != "cpm" and rm["spec"].name == name and rm["n_cw"] == N_CW
    out["median"] = fails(rm, PHY.soft_bits(rm), b) if ok_m else N_CW
    out["received"] = int(ok_r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="runs/ladder_10pct.csv")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default="runs/rx_ab.csv")
    ap.add_argument("--names", nargs="+", default=list(NAMES))
    a = ap.parse_args()
    p10 = {r["name"]: r for r in csv.DictReader(open(a.ladder))}
    rows = []
    with Pool(a.jobs) as pool:
        print("mode channel snr | bursts received | codeword failures: base gate median | "
              "paired bursts better/worse vs base: gate, median", flush=True)
        for name in a.names:
            for chan in CHANNELS:
                if p10[name][chan] in ("", "nan"):
                    continue
                for off in OFFSETS:
                    snr = float(p10[name][chan]) + off
                    res = pool.map(trial, [(name, chan, snr, 104729 * k + 11) for k in range(a.trials)])
                    tot = {v: sum(t[v] for t in res) for v in VARIANTS}
                    bw = {v: (sum(t[v] < t["base"] for t in res), sum(t[v] > t["base"] for t in res))
                          for v in VARIANTS[1:]}
                    rows.append(dict(name=name, channel=chan, snr=round(snr, 3), trials=a.trials,
                                     received=sum(t["received"] for t in res), **tot,
                                     **{f"{v}_better": bw[v][0] for v in bw}, **{f"{v}_worse": bw[v][1] for v in bw}))
                    print(f"{name} {chan} {snr:.2f} | {rows[-1]['received']} | "
                          + " ".join(str(tot[v]) for v in VARIANTS) + " | "
                          + ", ".join(f"{bw[v][0]}/{bw[v][1]}" for v in bw), flush=True)
                    with open(a.out, "w", newline="") as f:
                        w = csv.DictWriter(f, list(rows[0]))
                        w.writeheader()
                        w.writerows(rows)


if __name__ == "__main__":
    main()
