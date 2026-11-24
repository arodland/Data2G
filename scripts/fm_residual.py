"""What the FM chain (data2g.fmchannel) does to OFDM audio, measured on a
known burst through the real receiver (modem.receive): per radio path and
CNR, the effective SNR of the data cells, its spread across carriers, and
how impulsive it is (clicks). The residual of a cell is y - h x, x what the
receiver reads from the transmitted audio with no channel at all (so the
modem's own TX clip noise, a 15 dB floor for this 16-QAM burst, is part of
the signal, not the residual) and h the genie: the noiseless FM run's
response against it (static path). What is left is the FM chain's noise and
distortion, not estimation error or clipping.

    DATA2G_PROFILE=fm python scripts/fm_residual.py

Columns: SNR (dB, mean signal over mean residual power, data cells); low/high
(dB, the same on the lowest and highest 8 carriers); clicks (the share of
symbols whose residual is over 10x the median symbol's); kurt (E|e|^4 /
(E|e|^2)^2 of a cell's residual: 2 for complex Gaussian noise, far above
with clicks). Mobile rows: the genie is the static response, so their SNR
includes the fading the receiver tracks; read only their click share.

Measured 2026-11-16 (w48, 16-QAM burst, drive for 15 kHz 26 dB down):
noiseless, the mic path's limiter leaves 27.8 dB (24.2 on the lowest
carriers), the flat path 38.6. Static mic: 15.7 / 19.5 / 21.3 / 23.5 / 25.9
dB at CNR 8 / 10 / 12 / 15 / 20, then the limiter's floor. Static flat:
16.8 / 19.7 / 21.8 / 24.8 / 29.4 / 33.4 at 8-25, its noise rising with
frequency (lowest carriers 8-12 dB better than the highest). Clicks: 0.3% of
symbols on static at CNR 8, 3-4% on flutter at 8-10, 2-30% on mobile-urban.
"""

import numpy as np

from data2g import config, constellation, fmchannel, modem
from data2g.config import DATA_SYMS_PER_FRAME, SUBMODES

SPEC = SUBMODES["w48-16qam-r1/2"]
N_F = 16


def burst(seed):
    pts = constellation.load(SPEC.constellation)
    m = int(np.log2(len(pts)))
    nc = config.BANDS[SPEC.band].nc
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, N_F * DATA_SYMS_PER_FRAME * nc * m)
    x = modem.modulate_bits(bits, SPEC)
    cells = constellation.modulate(bits, pts).reshape(N_F, DATA_SYMS_PER_FRAME, nc)
    return np.concatenate([np.zeros(config.FS // 2), x, np.zeros(config.FS // 2)]), cells


def received(y):
    r = modem.receive(y, [SPEC.sync_band])
    return modem.data_symbols(r["raw"])


def main():
    print("| radio | condition | CNR | SNR | low 8 | high 8 | clicks | kurt |")
    print("|---|---|---|---|---|---|---|---|")
    for radio in ("mic", "flat"):
        x, _ = burst(1)
        cells = received(x + 1e-9 * np.random.default_rng(0).standard_normal(len(x)))  # the TX as the receiver reads it
        drive = fmchannel.drive_for_bandwidth(x[np.abs(x) > 0], radio, 15_000.0, "db26")[0]
        clean = received(fmchannel.apply_channel(x, None, "static", radio, 0, drive_dev_hz=drive))
        h = np.mean(clean / cells, axis=(0, 1))  # the radio pair's response per carrier (static)
        d = clean - h * cells
        p_sig = np.mean(np.abs(h * cells) ** 2)
        print(f"| {radio} | static | no noise | {10 * np.log10(p_sig / np.mean(np.abs(d) ** 2)):.1f} | "
              f"{10 * np.log10(p_sig / np.mean(np.abs(d[..., :8]) ** 2)):.1f} | "
              f"{10 * np.log10(p_sig / np.mean(np.abs(d[..., -8:]) ** 2)):.1f} | | |")
        for cond in ("static", "flutter", "mobile-urban"):
            for cnr in (8, 10, 12, 15, 20, 25, 30):
                res = []
                for seed in range(4):
                    try:
                        y = received(fmchannel.apply_channel(x, cnr, cond, radio, seed, drive_dev_hz=drive))
                    except Exception:
                        continue
                    res.append(y - h * cells)
                if not res:
                    print(f"| {radio} | {cond} | {cnr} | not acquired | | | | |")
                    continue
                e = np.concatenate(res)
                ps = np.mean(np.abs(e) ** 2, axis=-1).ravel()  # per symbol
                k = np.mean(np.abs(e) ** 4) / np.mean(np.abs(e) ** 2) ** 2
                print(f"| {radio} | {cond} | {cnr} | {10 * np.log10(p_sig / np.mean(np.abs(e) ** 2)):.1f} | "
                      f"{10 * np.log10(p_sig / np.mean(np.abs(e[..., :8]) ** 2)):.1f} | "
                      f"{10 * np.log10(p_sig / np.mean(np.abs(e[..., -8:]) ** 2)):.1f} | "
                      f"{np.mean(ps > 10 * np.median(ps)):.3f} | {k:.1f} |", flush=True)


if __name__ == "__main__":
    main()
