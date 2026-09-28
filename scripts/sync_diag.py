"""Where sync fails: per band and channel, at a given SNR, classify each
burst the way scripts/sync_floor.py counts it, plus what the own-band
detector and a genie header read (true start and CFO) did:

  metric  own-band detector: nothing crossed the threshold
  mislock acquired, but > 2 NCP samples or > 20 Hz from the truth
  locked  acquired within those

  genie   header decodes at the true start and CFO (header-only floor)
  e2e     receive() result: ok / nopre / hdrfail / wrong

    uv run python scripts/sync_diag.py --at w:awgn:-6.75 n4:mpd:3 --trials 800
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
from collections import Counter
from multiprocessing import Pool

import numpy as np

from data2g import codes, hfchannel, modem
from data2g.config import BANDS, FS, LEADIN_SAMPLES, NCP, SUBMODES
from data2g.waveform import ofdm
from data2g.waveform.dsp import freq_correct, to_baseband
from data2g.waveform.sync import SyncError, acquire

BURST = {"w": "ack-1f", "n10": "n10-ack-4f", "n4": "n4-ack-2f", "w48": "w48-qpsk-r1/2"}


def trial(args):
    band, chan, snr, seed = args
    spec = SUBMODES[BURST[band]]
    rng = np.random.default_rng(seed)
    x = modem.modulate([rng.bytes(codes.payload_bytes(spec))], spec)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    f0 = rng.uniform(-50, 50)
    x = np.concatenate([np.zeros(lead), x, np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=f0, ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    start = lead + LEADIN_SAMPLES
    z0 = to_baseband(y)
    try:
        a = acquire(z0, band=ofdm.band(BANDS[band].sync_band))
        det = "locked" if abs(a.preamble_start - start) <= 2 * NCP and abs(a.freq_offset - f0) <= 20 else "mislock"
    except SyncError:
        det = "metric"
    g = modem._read_header(freq_correct(z0, f0), start, BANDS[band].sync_band)["hdr"]
    genie = g is not None and g[0] == spec and g[1] == 1
    try:
        r = modem.receive(y)
        ok = r["spec"] == spec and r["n_cw"] == 1 and abs(r["preamble_start"] - start) <= 2 * NCP
        e2e = "ok" if ok else "wrong"
    except SyncError as e:
        e2e = "nopre" if "no preamble" in str(e) else "hdrfail"
    return det, genie, e2e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", nargs="+", required=True, help="band:channel:snr")
    ap.add_argument("--trials", type=int, default=800)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    with Pool(a.jobs) as pool:
        for at in a.at:
            band, chan, snr = at.split(":")
            res = pool.map(trial, [(band, chan, float(snr), 10_000 + i) for i in range(a.trials)], chunksize=8)
            det = Counter(r[0] for r in res)
            e2e = Counter(r[2] for r in res)
            cross = Counter((r[0], r[1]) for r in res if r[2] != "ok")
            n = len(res)
            print(f"{band:4s} {chan:4s} {float(snr):6.2f} dB  n={n}  e2e fail {1 - e2e['ok'] / n:.4f} {dict(e2e)}"
                  f"  detector {dict(det)}  genie header fail {sum(not r[1] for r in res) / n:.4f}"
                  f"  e2e failures by (detector, genie ok): {dict(cross)}", flush=True)


if __name__ == "__main__":
    main()
