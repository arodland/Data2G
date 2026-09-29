"""Gear-shifter phase C2: channel-prediction dataset.

Per sample: one continuous fading process (a preset, or a random Watterson
Doppler / delay pair), an SNR, an earlier measured burst (the receiver's
history, prev_*: 1-20 s before, sometimes missing), a measured burst in
band b1, a gap (the turnaround and reply before the next data burst), then
the next burst.
Rows hold what the receiver measured on the first burst (the predictor's
inputs, from the real receiver) and, for the next burst's window in every
band, the true channel's MI per constellation (the targets: from the very
taps hfchannel drew, no clip noise, no estimation; predictor.mode_mi adds a
candidate submode's clip noise and the receiver's estimation loss).

    uv run python scripts/predictor_data.py --samples 8000 --out runs/predictor_data.csv
"""

import os

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
from multiprocessing import Pool

import numpy as np

from data2g import hfchannel, modem  # noqa: E402
from data2g.arq.phy import measure  # noqa: E402
from data2g.arq.predictor import capacity  # noqa: E402
from data2g.config import BANDS, FRAME_SAMPLES, FS, RS, SNR_REF_BW_HZ, SUBMODES  # noqa: E402
from data2g.waveform import ofdm  # noqa: E402

PROBE = {"w": ("qpsk-r1/2", 2), "n10": ("n10-qpsk-r1/2", 2), "n4": ("n4-qpsk-r1/2", 1), "w48": ("w48-qpsk-r1/2", 4)}
CONSTS = ("gray-qam4", "gray-qam16", "c64-snr18", "c256-snr26")
LEAD_S, TAIL_S = 0.3, 0.3


def features(r) -> dict:
    """The predictor's inputs as the receiver measures them (data2g.arq.phy.measure;
    frames is its own column here)."""
    return {k: v for k, v in measure(r).items() if k != "frames"}


NEXT_FRAMES_MAX = 80  # the next burst's window: 2-80 frames (log-uniform) after its preamble and header
NEXT_DATA_S = 0.6  # preamble + header, roughly, before the next burst's first frame


def channel_mi(y_len, preset, seed, snr, t0, frames, bands=None, prefix="next") -> dict:
    """The true channel's MI per (band, constellation) over `frames` frames
    from t0: the taps hfchannel.fading drew for this seed and length (same
    generator, same order), no clip, no estimation."""
    out = {}
    t = t0 + (np.arange(frames) + 0.5) * FRAME_SAMPLES / FS
    idx = np.clip((t * FS).astype(int), 0, y_len - 1)
    if preset is None:
        g1, g2, delay = np.ones(len(t)), np.zeros(len(t)), 0.0
    else:
        p = hfchannel.FADING_PRESETS[preset] if isinstance(preset, str) else preset
        rng = np.random.default_rng(seed)
        g1 = hfchannel._gaussian_taps(y_len, p.doppler_hz, rng)[idx]
        g2 = hfchannel._gaussian_taps(y_len, p.doppler_hz, rng)[idx]
        delay = p.delay_ms * 1e-3
    for b in (bands or PROBE):
        bb = ofdm.band(b).bb
        h = (g1[:, None] + g2[:, None] * np.exp(-2j * np.pi * bb[None, :] * delay)) / (np.sqrt(2) if preset else 1)
        snr_c = 10 ** (snr / 10) * (SNR_REF_BW_HZ / RS) / len(bb) * np.abs(h) ** 2
        for c in CONSTS:
            out[f"{prefix}_{b}_{c}" if bands is None else f"{prefix}_{c}"] = float(np.mean(capacity(10 * np.log10(np.maximum(snr_c, 1e-9)), c)))
    return out


def measured_burst(rng, band, lead):
    """A burst as a live receiver meets them: any submode in its band,
    1-16 frames of data (mostly short replies)."""
    s = SUBMODES[str(rng.choice([s.name for s in SUBMODES.values() if s.band == band]))]
    n = int(rng.integers(1, max(1, 16 // s.frames_per_cw) + 1))
    return s, n, modem.modulate_bits(rng.integers(0, 2, n * s.coded_bits), s)


def sample(seed):
    rng = np.random.default_rng(seed)
    kind = rng.choice(["awgn", "mpg", "mpp", "mpd", "random"])
    if kind == "random":
        preset = hfchannel.FadingPreset("random", float(np.exp(rng.uniform(np.log(0.05), np.log(3.0)))),
                                        float(rng.uniform(0.0, 5.0)))
    else:
        preset = None if kind == "awgn" else kind
    snr = float(rng.uniform(-12, 30))
    gap = float(rng.uniform(1.0, 6.0))
    b1 = str(rng.choice(list(PROBE)))
    # the burst before it (the receiver's history): usually the same band,
    # 1-20 s earlier; sometimes none (the first burst heard)
    b0 = b1 if rng.random() < 0.6 else str(rng.choice(list(PROBE)))
    s0, n0, prev = measured_burst(rng, b0, LEAD_S)
    gap0 = float(np.exp(rng.uniform(np.log(1.0), np.log(20.0))))
    s1, n1, first = measured_burst(rng, b1, LEAD_S)
    lead, gap_n, gap0_n = int(LEAD_S * FS), int(gap * FS), int(gap0 * FS)
    t1 = lead + len(prev) + gap0_n  # the measured burst's start
    t2 = t1 + len(first) + gap_n  # the next burst's start
    window = int(round(np.exp(rng.uniform(np.log(2), np.log(NEXT_FRAMES_MAX)))))
    n = t2 + int((NEXT_DATA_S + window * FRAME_SAMPLES / FS + TAIL_S) * FS)
    x = np.zeros(n)
    x[lead:lead + len(prev)] = prev
    x[t1:t1 + len(first)] = first
    y = hfchannel.apply_channel(x, snr_db=snr, fading_preset=preset, seed=seed)
    try:
        r1 = modem.receive(y[t1 - lead: t1 + len(first) + gap_n // 2])
    except modem.SyncError:
        return None
    if r1["spec"].name != s1.name:
        return None
    r0 = None
    if rng.random() < 0.85:
        try:
            r0 = modem.receive(y[: lead + len(prev) + min(gap0_n // 2, lead)])
            r0 = r0 if r0["spec"].name == s0.name else None
        except modem.SyncError:
            pass
    p = preset if isinstance(preset, hfchannel.FadingPreset) else hfchannel.FADING_PRESETS.get(preset)
    row = dict(seed=seed, kind=kind, doppler=p.doppler_hz if p else 0.0, delay_ms=p.delay_ms if p else 0.0,
               snr=snr, gap=gap, band1=b1, submode1=s1.name, frames=n1 * s1.frames_per_cw, window=window)
    row.update(features(r1))
    row["prev_band"] = b0 if r0 else ""
    row["prev_age"] = (t1 + len(first) - lead - len(prev)) / FS  # end to end
    row["prev_frames"] = n0 * s0.frames_per_cw
    row.update({f"prev_{k}": (v if r0 else "") for k, v in features(r0 if r0 else r1).items()})
    row.update(channel_mi(n, preset, seed, snr, t2 / FS + NEXT_DATA_S, window))
    # the measured burst's own true channel MI (in its band): the residual
    # measured - truth is what scripts/linksim.py samples its measurements from
    hdr = (BANDS[s1.sync_band].preamble_samples + modem.header_samples(s1.sync_band)) / FS
    row.update(channel_mi(n, preset, seed, snr, (t1 + modem.LEADIN_SAMPLES) / FS + hdr,
                          n1 * s1.frames_per_cw, bands=[b1], prefix="truth1"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    done = 0
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    start = 0 if new else sum(1 for _ in open(a.out)) - 1
    with Pool(a.jobs) as pool, open(a.out, "a", newline="") as f:
        w = None
        for row in pool.imap_unordered(sample, range(start * 7 + 1, (start + a.samples) * 7 + 1, 7), chunksize=4):
            if row is None:
                continue
            if w is None:
                w = csv.DictWriter(f, list(row))
                if new:
                    w.writeheader()
            w.writerow(row)
            done += 1
            if done % 200 == 0:
                f.flush()
                print(done, flush=True)


if __name__ == "__main__":
    main()
