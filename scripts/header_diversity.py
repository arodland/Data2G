"""Would a second header copy help, and where should it go? The 1200 and
2400 Hz bands' header is 4 symbols (96 ms): on MPP most missed bursts
fail there after a good preamble. Emulated without touching the modem:
the header's QPSK symbols also ride data symbols 1-4 of frame k of a real
burst; the receiver demodulates that frame with its own pilots and adds
its LLRs to the first copy's before the ML decode.

  base    the first copy alone (today)
  f0      + a copy in frame 0 (right after the header: ~ one 8-symbol header)
  f1..f4  + a copy in frame k (k x 144 ms later)

Per trial, the modem's acquisition and its start/CFO hypotheses (as
modem._best_header), the best read over HEADER_MIN_SCORE wins:
ok, wrong (another valid word accepted) or fail (none).

    uv run python scripts/header_diversity.py --out runs/header_diversity.csv
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from data2g import codes, constellation, hfchannel, modem
from data2g.config import BANDS, DATA_SYMS_PER_FRAME, FRAME_SAMPLES, FS, LEADIN_SAMPLES, LEADOUT_SAMPLES, M, NCP, NSYM, SUBMODES
from data2g.waveform import ofdm
from data2g.waveform.dsp import freq_correct, to_baseband
from data2g.waveform.sync import SyncError, acquire

MODE = {"w": "qpsk-r1/2", "w48": "w48-qpsk-r1/2"}
VARIANTS = ("base", "f0", "f1", "f2", "f4")
FRAME = {"f0": 0, "f1": 1, "f2": 2, "f4": 4}


def burst(spec, rng, n_frames=8):
    """A burst of random data with the header's symbols also in frames 0, 1,
    2 and 4 (every variant reads the copy it needs; the others are data to it)."""
    n_cw = -(-n_frames // spec.frames_per_cw)
    bits = rng.integers(0, 2, (n_cw, spec.coded_bits))
    b = ofdm.band(spec.band)
    data = constellation.modulate(codes.spread(bits, spec.bits_per_cu), constellation.load(spec.constellation))
    data = data.reshape(-1, DATA_SYMS_PER_FRAME, b.nc)
    hdr = constellation.modulate(modem.header_bits(spec.index, n_cw, spec.band), modem.QPSK).reshape(-1, b.nc)
    for k in FRAME.values():
        data[k, :len(hdr)] = hdr
    x = modem.burst_waveform(data, spec)
    x = modem.tx_condition(x, spec.headroom, b.spec.clip_overshoot, active=slice(LEADIN_SAMPLES, len(x) - LEADOUT_SAMPLES),
                           bandpass=BANDS[spec.sync_band].tx_bandpass)
    return x, n_cw, len(hdr)


def copy_llr(z, start, band, k, n_hdr):
    """LLRs of the header copy in frame k, its channel interpolated
    between frame k's pilot and frame k + 1's."""
    b = ofdm.band(band)
    p = start + b.spec.preamble_samples + modem.header_samples(band) + k * FRAME_SAMPLES
    if p + FRAME_SAMPLES + NSYM > len(z):
        return None
    win = lambda s: b.demod_window(z, p + s * NSYM + NCP, modem.HEADER_BACKOFF)  # noqa: E731
    h0, h1 = win(0) / b.pilot, win(6) / b.pilot
    ys = np.array([win(s) for s in range(1, 1 + n_hdr)])
    a = (np.arange(1, 1 + n_hdr) / 6.0)[:, None]
    hs = (1 - a) * h0 + a * h1
    return constellation.llr(ys, hs, np.ones(ys.shape), modem.QPSK)


def trial(args):
    band, chan, snr, seed = args
    spec = SUBMODES[MODE[band]]
    rng = np.random.default_rng(seed)
    x, n_cw, n_hdr = burst(spec, rng)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    x = np.concatenate([np.zeros(lead), x, np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=float(rng.uniform(-50, 50)), ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    want = (spec.index << 6) | (n_cw - 1)
    z0 = to_baseband(y)
    try:
        acq = acquire(z0, band=ofdm.band(band))
    except SyncError:
        return {v: "nosync" for v in VARIANTS}
    thr = modem.HEADER_MIN_SCORE[band]
    best = {v: (-1.0, None) for v in VARIANTS}
    hyps = [(acq.preamble_start, acq.freq_offset)] + acq.alternatives
    captured = []
    orig = modem.decode_header

    def grab(soft, band_, accept=None):
        captured.append(soft)
        return orig(soft, band_, accept)
    modem.decode_header = grab
    try:
        for h, (start, f) in enumerate(hyps):
            zb = freq_correct(z0, f)
            for k in (0, -1, 1, -2, 2):
                s = start + k * M
                if s < 0:
                    continue
                captured.clear()
                try:
                    modem._read_header(zb, s, band)
                except (SyncError, IndexError, ValueError):
                    continue
                llr1 = captured[0]
                pen = modem.ALT_PENALTY if h or k else 0.0
                for v in VARIANTS:
                    soft = llr1
                    if v != "base":
                        l2 = copy_llr(zb, s, band, FRAME[v], n_hdr)
                        if l2 is None:
                            continue
                        soft = llr1 + l2.reshape(llr1.shape)
                    word, _, score = orig(soft, band)
                    if score >= thr and score - pen > best[v][0]:
                        best[v] = (score - pen, word >> 6)
    finally:
        modem.decode_header = orig
    return {v: ("fail" if w is None else "ok" if w == want else "wrong") for v, (_, w) in best.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/header_diversity.csv")
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    cells = [(band, chan, snr) for band in ("w", "w48") for chan, snrs in
             (("mpp", (-6, -3, 0, 3, 6)), ("mpg", (-6, -3, 0, 3)), ("mpd", (-3, 0, 3, 6)), ("awgn", (-9, -7, -5)))
             for snr in (snrs if band == "w" else [s + 2 for s in snrs])]
    rows = []
    with Pool(a.jobs) as pool:
        for band, chan, snr in cells:
            res = pool.map(trial, [(band, chan, snr, 1000 * s + 17) for s in range(a.trials)])
            row = dict(band=band, channel=chan, snr=snr)
            for v in VARIANTS:
                c = defaultdict(int)
                for r in res:
                    c[r[v]] += 1
                row.update({f"{v}_fail": (c["fail"] + c["nosync"]) / len(res), f"{v}_wrong": c["wrong"] / len(res)})
            rows.append(row)
            print(band, chan, snr, "  ".join(f"{v} {row[v + '_fail']:.1%}/{row[v + '_wrong']:.1%}" for v in VARIANTS),
                  flush=True)
            with open(a.out, "w", newline="") as fh:
                w = csv.DictWriter(fh, list(rows[0]))
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
