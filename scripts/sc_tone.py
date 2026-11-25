"""Scattered pilots next to a steady tone (PR #308 review, items 5-7): fixed bursts, the receiver given the burst's
timing (no search), the channel's taps known, pilot row (flag 0) against scattered pilots (flag 1, mix). Per mode,
channel, SNR (PEP 5, sc_diag's) and tone (Hz, 0: none; +25 dB over the floor in 50 Hz, the carriers preset's
level), the medians over bursts of:
  ok        codewords decoded (no DD)
  est_err   the channel estimate against the true channel, on carriers more than FAR_HZ from the tone (dB)
  told      the error the data cells carry over the noise the LLRs were told, on those carriers (dB; negative:
            told too much)
  cfo       the residual CFO read (Hz; the true one is 0)

Adapted from finitor's scripts/studies/ffi_qrm_carriers/tone_estimate_error.py (github.com/finitor/Data2G,
3b0f94b; same project and licence), itself from scripts/sc_diag.py.

    python -m scripts.sc_tone --chans awgn,mpp --tones 0,825,2200 --bursts 24
"""

from data2g import threads

threads.limit(1)

import argparse  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from data2g import codes, constellation, hfchannel, modem  # noqa: E402
from data2g.config import FS, LEADIN_SAMPLES, M, NCP, NSYM, SNR_REF_BW_HZ, SUBMODES  # noqa: E402
from data2g.waveform import ofdm  # noqa: E402
from scripts import sc_diag as D  # noqa: E402

TONE_INR_DB = 25.0  # over the floor in 50 Hz (interference.PRESETS["carriers"])
TONE_REF_HZ = 50.0
FAR_HZ = 300.0


def one(args):
    flag, name, chan, snr, tone, seed = args
    D.setup("mix", flag)
    spec = SUBMODES[name]
    b = ofdm.band(spec.band)
    rng = np.random.default_rng([seed, 3])
    n_cw = max(1, int((4.0 - modem.burst_seconds(spec, 0)) / (modem.burst_seconds(spec, 1) - modem.burst_seconds(spec, 0))))
    pl = [rng.bytes(codes.payload_bytes(spec)) for _ in range(n_cw)]
    x = np.concatenate([np.zeros(D.PAD), modem.modulate(pl, name), np.zeros(D.PAD)])
    known = dict(start=D.PAD + LEADIN_SAMPLES, cfo=0.0, spec=spec, n_cw=n_cw)
    z = signal.hilbert(x)
    peak = float(np.max(np.abs(z) ** 2) / 2)
    if chan == "awgn":
        y, taps = x.copy(), None
    else:
        p = D.PRESETS[chan]
        g = np.random.default_rng(seed)
        d = int(round(p.delay_ms * 1e-3 * FS))
        g1, g2 = (hfchannel._gaussian_taps(len(z), p.doppler_hz, g) for _ in range(2))
        y = np.real((z * g1 + np.concatenate([np.zeros(d, dtype=complex), z[:len(z) - d]]) * g2) / np.sqrt(2))
        taps = (g1, g2, d)
    sigma = np.sqrt(peak / 10 ** 0.5 * (FS / 2) / SNR_REF_BW_HZ / 10 ** (snr / 10))
    y = y + np.random.default_rng([seed, 5]).normal(scale=sigma, size=len(y))
    if tone:
        amp = np.sqrt(2 * sigma**2 * TONE_REF_HZ / (FS / 2) * 10 ** (TONE_INR_DB / 10))
        ph = np.random.default_rng([seed, 9]).uniform(0, 2 * np.pi)
        y = y + amp * np.cos(2 * np.pi * tone * np.arange(len(y)) / FS + ph)
    try:
        r = modem.receive(y, known=known)
    except modem.SyncError:
        return args, None
    e = r["est"]
    ok = sum(modem.decode_received(r).crc_ok)
    bits = np.stack([codes.encode(spec, p_, 0, index=i) for i, p_ in enumerate(pl)])
    X = constellation.modulate(codes.spread(bits, spec.bits_per_cu), constellation.load(spec.constellation)).reshape(e["h"].shape)
    Y = modem.data_symbols(r["raw"])
    sym, k = D.cells(r)
    n = r["p0"] + sym * NSYM + NCP + M // 2
    H = np.ones(len(sym), dtype=complex) if taps is None else (
        (taps[0][n] + taps[1][n] * np.exp(-2j * np.pi * b.freqs[k] * taps[2] / FS)) / np.sqrt(2))
    H = H * np.exp(-2j * np.pi * r["cfo"] * n / FS)  # the residual CFO the receiver took out (a tone pulls it)
    best = None
    for tau in np.arange(-48, 48.01, 0.5):
        ref = H * np.exp(2j * np.pi * b.bb[k] * tau / FS)
        c = np.vdot(ref, e["h"].ravel()) / np.vdot(ref, ref)
        if best is None or abs(c) > abs(best[0]):
            best = (c, ref)
    ref = best[0] * best[1]
    far = np.abs(np.asarray(b.freqs)[k] - tone) > FAR_HZ if tone else np.ones(len(k), bool)
    told = np.broadcast_to(modem.noise_var(e["h"], e) + e["mse"], e["h"].shape).ravel()
    meas = np.abs(Y - e["h"] * X).ravel() ** 2
    return args, dict(ok=ok, n_cw=n_cw, cfo=float(r["cfo"]),
                      est_err=float(np.mean(np.abs(e["h"].ravel() - ref)[far] ** 2) / np.mean(np.abs(ref) ** 2)),
                      told=float(np.mean(meas[far]) / np.mean(told[far])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="w48-qpsk-r1/3")
    ap.add_argument("--chans", default="awgn,mpp")
    ap.add_argument("--snrs", default="8")
    ap.add_argument("--tones", default="0,825,2200")
    ap.add_argument("--flags", default="0,1")
    ap.add_argument("--bursts", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--genie-cfo", action="store_true", help="the residual CFO read as its true 0")
    a = ap.parse_args()
    if a.genie_cfo:
        from data2g import equalizer

        equalizer.residual_cfo = lambda h: 0.0
    jobs = [(f == "1", a.mode, c, float(s), float(t), seed) for f in a.flags.split(",") for c in a.chans.split(",")
            for s in a.snrs.split(",") for t in a.tones.split(",") for seed in range(a.bursts)]
    acc = {}
    with Pool(a.jobs, maxtasksperchild=200) as pool:
        for args, o in pool.imap_unordered(one, jobs, chunksize=2):
            acc.setdefault(args[:5], []).append(o)
    db = lambda v: 10 * np.log10(max(float(v), 1e-12))  # noqa: E731
    print(f"{a.mode}: flag 0 pilot row, 1 scattered (mix); SNR PEP 5; medians over bursts")
    print("| chan | SNR | tone | flag | ok | est_err dB | told dB | cfo Hz |\n|---|---|---|---|---|---|---|---|")
    for (flag, _, c, s, t), outs in sorted(acc.items(), key=lambda kv: (kv[0][2], kv[0][3], kv[0][4], kv[0][0])):
        got = [o for o in outs if o is not None]
        okf = sum(o["ok"] for o in got) / max(sum(o["n_cw"] for o in outs if o), 1)
        print(f"| {c} | {s:+.0f} | {t:.0f} | {int(flag)} | {okf:.2f} | {db(np.median([o['est_err'] for o in got])):+.1f} | "
              f"{db(np.median([o['told'] for o in got])):+.1f} | {np.median([o['cfo'] for o in got]):+.2f} |", flush=True)


if __name__ == "__main__":
    main()
