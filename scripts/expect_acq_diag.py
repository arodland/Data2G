"""Where the real receiver's header reads land on bursts it loses at the
header but the genie reads: acquisition's start and CFO
against the truth, its alternatives, and the header score at each.

    uv run --no-sync python scripts/expect_acq_diag.py n4-ack-8f mpd -8
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import sys
from multiprocessing import Pool

import numpy as np

from data2g import codes, hfchannel, modem
from data2g.arq.engine import receive_accept
from data2g.config import FS, LEADIN_SAMPLES, M, SUBMODES
from data2g.waveform import ofdm
from data2g.waveform.dsp import freq_correct, to_baseband

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from expect_header_study import word_of  # noqa: E402


def trial(args):
    mode, chan, snr, seed = args
    spec = SUBMODES[mode]
    band = spec.sync_band
    rng = np.random.default_rng(seed)
    x = modem.modulate([rng.bytes(codes.payload_bytes(spec))], spec)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    f0 = rng.uniform(-50, 50)
    x = np.concatenate([np.zeros(lead), x, np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=f0, ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    truth, start, full = word_of(spec, 1), lead + LEADIN_SAMPLES, receive_accept()
    z0 = to_baseband(y)
    g = modem._read_header(freq_correct(z0, f0), start, band, full)
    if g["hdr"] is None or g["word"] != truth:
        return None
    try:
        r = modem.receive(y, accept=full)
        if (r["spec"].name, r["n_cw"]) == (mode, 1):
            return None
    except modem.SyncError:
        pass
    try:
        acq = modem._sync.acquire(z0, band=ofdm.band(band))
    except modem.SyncError:
        return ("missed",)
    hyps = [(acq.preamble_start, acq.freq_offset)] + list(acq.alternatives)
    out = []
    for s, f in hyps:
        out.append((s - start, round(f - f0, 2)))
    # best score near the truth at acquisition's CFO, and at the true CFO near acq's start
    za = freq_correct(z0, acq.freq_offset)
    near = max((modem._read_header(za, start + d, band, full)["score"], d) for d in range(-48, 49, 8))
    zt = freq_correct(z0, f0)
    at_acq = modem._read_header(zt, acq.preamble_start, band, full)["score"]
    return ("lost", out[:4], round(g["score"], 3), round(near[0], 3), near[1], round(at_acq, 3))


if __name__ == "__main__":
    mode, chan, snr = sys.argv[1], sys.argv[2], float(sys.argv[3])
    with Pool(2) as p:
        rows = [r for r in p.map(trial, [(mode, chan, snr, 40_000 + i) for i in range(150)]) if r is not None]
    for r in rows:
        print(r)
