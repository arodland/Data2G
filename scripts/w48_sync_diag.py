"""Why w48's sync floor sits ~3 dB above w's on AWGN (-3.75 vs -7.0 dB):
detection, header and whole receive separated, on the bursts
scripts/sync_floor.py measures with.

    uv run python scripts/w48_sync_diag.py --snr -6 -5 -4
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import codes, hfchannel, modem
from data2g.config import BANDS, LEADIN_SAMPLES, NCP, SUBMODES
from data2g.waveform import ofdm, sync
from data2g.waveform.dsp import freq_correct, to_baseband

sys.path.insert(0, str(Path(__file__).parent))
from sync_floor import BURST  # noqa: E402

FS = 8000


def trial(args):
    band, snr, seed = args
    spec = SUBMODES[BURST[band]]
    rng = np.random.default_rng(seed)
    x = modem.modulate([rng.bytes(codes.payload_bytes(spec))], spec)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    cfo = rng.uniform(-50, 50)
    x = np.concatenate([np.zeros(lead), x, np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=cfo, ppm=10, seed=seed)
    z = to_baseband(y)
    truth = lead + LEADIN_SAMPLES
    out = {}
    # detection alone: this band's detector, the statistic at the truth
    S, freqs = sync.detection_stat(z, ofdm.band(band))
    i = int(np.argmin(np.abs(freqs - cfo)))
    out["stat_truth"] = float(S[max(0, i - 1):i + 2, truth - 2:truth + 3].max())
    out["stat_noise_max"] = float(np.delete(S.max(axis=0), range(truth - 200, truth + 200)).max())
    try:
        acq = sync.acquire(z, band=ofdm.band(band))
        out["detected"] = abs(acq.preamble_start - truth) <= 2 * NCP
    except sync.SyncError:
        out["detected"] = False
    # header with genie timing and CFO
    r = modem._read_header(freq_correct(z, cfo), truth, band)
    out["hdr_genie"] = r["hdr"] is not None and r["hdr"][0].name == spec.name and r["hdr"][1] == 1
    out["hdr_score"] = float(r["score"])
    # the whole receiver, every band's detector (as sync_floor)
    try:
        rr = modem.receive(y)
        out["receive"] = rr["spec"] == spec and rr["n_cw"] == 1 and abs(rr["preamble_start"] - truth) <= 2 * NCP
    except modem.SyncError:
        out["receive"] = False
    return band, snr, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, nargs="+", default=[-6.0, -5.0, -4.0])
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    jobs = [(b, s, seed) for b in ("w", "w48") for s in a.snr for seed in range(a.trials)]
    with Pool(a.jobs) as pool:
        res = pool.map(trial, jobs, chunksize=8)
    thr = BANDS["w"].preamble_threshold
    print(f"threshold {thr}")
    for b in ("w", "w48"):
        for s in a.snr:
            rs = [o for bb, ss, o in res if bb == b and ss == s]
            f = lambda k: np.mean([o[k] for o in rs])  # noqa: E731
            print(f"{b:4s} {s:+5.1f} dB: fail detect {1 - f('detected'):.3f}  genie header {1 - f('hdr_genie'):.3f}"
                  f"  receive {1 - f('receive'):.3f} | stat at truth median {np.median([o['stat_truth'] for o in rs]):6.1f}"
                  f" 1% {np.quantile([o['stat_truth'] for o in rs], 0.01):6.1f} | header score median "
                  f"{np.median([o['hdr_score'] for o in rs]):.2f} 5% {np.quantile([o['hdr_score'] for o in rs], 0.05):.2f}")


if __name__ == "__main__":
    main()
