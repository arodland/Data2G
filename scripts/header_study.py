"""Header decoding at the true start and CFO (genie sync), with variants
of the header's channel reference and decoder, to find what the header
can gain without air time:

  base    modem._read_header as it is
  fs      pilots and preamble reference projected onto a CP-long delay
          support (equalizer._freq_smooth) before interpolation
  all     reference from every preamble repeat instead of the last 4
  valid   ML over valid words only (CRC-6 right and a defined submode),
          instead of ML over all 65536 then the CRC check
  dd      after a decode, the decoded symbols join the pilots, the
          channel is re-estimated and the header decoded again

    uv run python scripts/header_study.py --at w:awgn:-6 n4:mpd:2 --variants base fs fs+valid
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

from data2g import codes, constellation, equalizer, hfchannel, modem
from data2g.config import FS, LEADIN_SAMPLES, M, NCP, NSYM, PREAMBLE_CP, SUBMODES
from data2g.modem import HEADER_BACKOFF, QPSK, _crc6, header_code, header_layout, header_samples
from data2g.waveform import ofdm
from data2g.waveform.dsp import freq_correct, to_baseband

BURST = {"w": "ack-1f", "n10": "n10-ack-4f", "n4": "n4-ack-2f", "w48": "w48-qpsk-r1/2"}


@lru_cache(maxsize=None)
def valid_mask(band):
    w = np.arange(2**16)
    v, crc = w >> 6, w & 0x3F
    ok = np.array([_crc6(int(x)) for x in v]) == crc
    idx = {s.index for s in SUBMODES.values() if s.band == band}
    return ok & np.isin(v >> 6, list(idx))


def decode(soft, band, valid):
    corr = modem._header_signs(band) @ soft.astype(np.float32)
    if valid:
        corr = np.where(valid_mask(band), corr, -np.inf)
    word = int(np.argmax(corr))
    score = float(corr[word] / (np.sqrt(np.sum(soft**2) * len(soft)) + 1e-12))
    return word, score


def read(z, start, band, opts):
    b = ofdm.band(band)
    PREAMBLE_REPEATS, PREAMBLE_SAMPLES = b.spec.preamble_repeats, b.spec.preamble_samples
    ref = PREAMBLE_REPEATS if "all" in opts else min(PREAMBLE_REPEATS, modem.REF_REPEATS)
    h_reps = np.array([b.demod_window(z, start + PREAMBLE_CP + r * M, HEADER_BACKOFF)
                       for r in range(PREAMBLE_REPEATS)]) / b.pilot
    u0 = start + PREAMBLE_CP + (PREAMBLE_REPEATS - ref) * M
    h_pre = h_reps[PREAMBLE_REPEATS - ref:].mean(axis=0)
    h0 = start + PREAMBLE_SAMPLES
    p0 = h0 + header_samples(band)
    h_first = b.demod_window(z, p0 + NCP, HEADER_BACKOFF) / b.pilot
    layout = header_layout(band)
    t_sym = h0 + np.arange(len(layout)) * NSYM + NCP + M / 2
    y_all = np.array([b.demod_window(z, int(t - M / 2), HEADER_BACKOFF) for t in t_sym])
    support = (HEADER_BACKOFF, HEADER_BACKOFF + NCP)  # delays as seen from the backed-off window

    def solve(t_p, h_p, t_d, y_d):
        if "fs" in opts:
            h_p = equalizer._freq_smooth(h_p, support, b.bb)[0]
        j = np.clip(np.searchsorted(t_p, t_d) - 1, 0, len(t_p) - 2)
        a = ((t_d - t_p[j]) / (t_p[j + 1] - t_p[j]))[:, None]
        hs = (1 - a) * h_p[j] + a * h_p[j + 1]
        return decode(constellation.llr(y_d, hs, np.ones(y_d.shape), QPSK), band, "valid" in opts)

    t_p = np.concatenate([[u0 + ref * M / 2], t_sym[layout], [p0 + NCP + M / 2]])
    h_p = np.concatenate([h_pre[None], y_all[layout] / b.pilot, h_first[None]])
    word, score = solve(t_p, h_p, t_sym[~layout], y_all[~layout])
    if "dd" in opts:
        # every header symbol is now a pilot; re-estimate each data
        # symbol's channel from its neighbours only (leave-one-out)
        sy = np.empty((len(layout), b.nc), dtype=np.complex128)
        sy[layout] = b.pilot
        sy[~layout] = constellation.modulate(modem._word_bits(word) @ header_code(band) % 2, QPSK).reshape(-1, b.nc)
        h_all = y_all / sy
        t_all = np.concatenate([[u0 + ref * M / 2], t_sym, [p0 + NCP + M / 2]])
        hh = np.concatenate([h_pre[None], h_all, h_first[None]])
        if "fs" in opts:
            hh = equalizer._freq_smooth(hh, support, b.bb)[0]
        hs = 0.5 * (hh[:-2] + hh[2:])[~layout]
        word, score = decode(constellation.llr(y_all[~layout], hs, np.ones(hs.shape), QPSK), band, "valid" in opts)
    return word, score


def trial(args):
    band, chan, snr, seed, variants = args
    spec = SUBMODES[BURST[band]]
    rng = np.random.default_rng(seed)
    x = modem.modulate([rng.bytes(codes.payload_bytes(spec))], spec)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    f0 = rng.uniform(-50, 50)
    x = np.concatenate([np.zeros(lead), x, np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=f0, ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    z = freq_correct(to_baseband(y), f0)
    truth = (spec.index << 12) | _crc6(spec.index << 6)  # n_cw = 1
    out = []
    for v in variants:
        word, score = read(z, lead + LEADIN_SAMPLES, band, set(v.split("+")))
        floor = modem.HEADER_MIN_SCORE[band]
        out.append(word == truth and score >= floor)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", nargs="+", required=True, help="band:channel:snr")
    ap.add_argument("--variants", nargs="+", default=["base", "fs", "all", "valid", "fs+valid", "fs+valid+dd"])
    ap.add_argument("--trials", type=int, default=800)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    with Pool(a.jobs) as pool:
        for at in a.at:
            band, chan, snr = at.split(":")
            res = np.array(pool.map(trial, [(band, chan, float(snr), 30_000 + i, a.variants)
                                            for i in range(a.trials)], chunksize=8))
            fails = "  ".join(f"{v} {1 - res[:, i].mean():.4f}" for i, v in enumerate(a.variants))
            print(f"{band:4s} {chan:4s} {float(snr):6.2f} dB  header fail: {fails}", flush=True)


if __name__ == "__main__":
    main()
