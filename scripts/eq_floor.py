"""Equalizer quality of the stub submode (uncoded Gray QPSK) through the
full burst modem, per channel.

Reports BMI, the bitwise mutual information of the soft outputs with the
best single scale per burst: what a decoder can actually extract, in
bits per coded bit (1.0 = perfect). Beside it the same for a genie
receiver with the true channel on the same bursts; the gap between the
two is what channel estimation still loses. Raw BER is printed beside it but
overweights carriers sitting in deep notches, which no equalizer can
make reliable and a code only needs flagged as unreliable.

    uv run python scripts/eq_floor.py [--snr 60] [--bursts 10] [--taps gaussian|butter]
"""

import argparse

import numpy as np

from data2g import codes, hfchannel, modem
from data2g.config import (
    CLIP, HEADER_SYMS, LEADIN_SAMPLES, LEADOUT_SAMPLES, SUBMODES, SYMS_PER_FRAME,
)
from data2g.waveform import ofdm
from data2g.waveform.dsp import freq_correct, to_baseband
from data2g.hfchannel import FadingPreset

CHANNELS = {
    "awgn": None,
    "mpg": FadingPreset("mpg", 0.1, 0.5),
    "1Hz 0.5ms": FadingPreset("x", 1.0, 0.5),
    "2Hz 0.5ms": FadingPreset("x", 2.0, 0.5),
    "0.1Hz 2ms": FadingPreset("x", 0.1, 2.0),
    "0.1Hz 4ms": FadingPreset("x", 0.1, 4.0),
    "mpp": FadingPreset("mpp", 1.0, 2.0),
    "mpd": FadingPreset("mpd", 2.0, 4.0),
}


def bmi(soft: np.ndarray, bits: np.ndarray) -> float:
    """max over scale a of 1 - E[log2(1 + exp(-a * sign * soft))]."""
    t = (1.0 - 2.0 * bits) * soft / np.sqrt(np.mean(soft**2))
    return max(
        1.0 - np.mean(np.logaddexp(0.0, -a * t)) / np.log(2)
        for a in np.geomspace(0.1, 1000, 81)
    )


def _pilot_only_twin(x_len: int, n_f: int) -> np.ndarray:
    """Same burst layout with every frame symbol a pilot, unclipped. The
    channel is linear, so pushing this through the same seeded channel
    without noise reads the true channel off every data position."""
    syms = np.tile(modem.PILOT, (n_f * SYMS_PER_FRAME + 1, 1))
    hdr = np.tile(modem.PILOT, (HEADER_SYMS, 1))
    body = np.concatenate([
        np.zeros(LEADIN_SAMPLES), ofdm.preamble_waveform(),
        ofdm.modulate_symbols(hdr), ofdm.modulate_symbols(syms), np.zeros(LEADOUT_SAMPLES),
    ])
    return np.concatenate([np.zeros(3000), body, np.zeros(x_len - 3000 - len(body))])


def measure(preset, snr, bursts, taps, frames=20):
    """(BMI, genie BMI, raw BER, sync failures). Genie = same bursts, same
    fades, same timing and CFO, true channel instead of the estimate."""
    spec = SUBMODES["qpsk-r1/2"]  # any QPSK submode measures the same soft bits
    errs = n = sync_fail = 0
    bmis, genie = [], []
    for seed in range(bursts):
        rng = np.random.default_rng(seed)
        sent = [rng.bytes(codes.payload_bytes(spec)) for _ in range(max(1, frames // spec.frames_per_cw))]
        x = np.concatenate([np.zeros(3000), modem.modulate(sent, spec), np.zeros(3000)])
        kw = dict(freq_offset_hz=37.0, ppm=10, fading_preset=preset, seed=seed, taps=taps)
        try:
            r = modem.receive(hfchannel.apply_channel(x, snr_db=snr, **kw))
        except modem.SyncError:
            sync_fail += 1
            continue
        est, raw = r["est"], r["raw"]
        bits = codes.spread(np.stack([codes.encode(spec, p, index=i) for i, p in enumerate(sent)]), spec.bits_per_cu)
        var = modem.noise_var(est["h"], est) + est["mse"]
        soft = modem.soft_bits(raw, est["h"], var, spec)
        errs += np.sum((soft < 0) != (bits == 1))
        n += len(bits)
        bmis.append(bmi(soft, bits))

        twin = hfchannel.apply_channel(_pilot_only_twin(len(x), len(raw)), **kw)
        zt = freq_correct(to_baseband(twin), r["cfo"])
        rt, _, _ = modem._demod_frames(zt, r["p0"], len(raw), r["shift"], r["phi_ref"], r["steps"])
        h_true = rt / modem.PILOT
        # The twin skips the clipper and the TX bandpass, so it differs by
        # a static per-carrier gain: fit it on the pilots both share.
        hp_a, hp_t = raw[:, 0] / modem.PILOT, h_true[:, 0]
        gain = np.sum(hp_a * np.conj(hp_t), axis=0) / np.sum(np.abs(hp_t) ** 2, axis=0)
        h_true = h_true[:, 1:] * gain * CLIP[spec.band][0].get(len(raw), CLIP[spec.band][1])
        var_t = modem.noise_var(h_true, est)
        genie.append(bmi(modem.soft_bits(raw, h_true, var_t, spec), bits))
    m = lambda v: float(np.mean(v)) if v else 0.0  # noqa: E731
    return m(bmis), m(genie), errs / max(n, 1), sync_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, nargs="+", default=[10.0, 20.0, 60.0])
    ap.add_argument("--bursts", type=int, default=10)
    ap.add_argument("--taps", default="gaussian")
    a = ap.parse_args()
    print(f"{'':10s} " + "  ".join(f"{s:>9.0f} dB: BMI genie  BER  " for s in a.snr) + "  hdr fail")
    for name, p in CHANNELS.items():
        cells, fails = [], 0
        for snr in a.snr:
            b, g, ber, sf = measure(p, snr, a.bursts, a.taps)
            cells.append(f"{b:13.4f} {g:.4f} {ber:.4f}")
            fails += sf
        print(f"{name:10s} " + "  ".join(cells) + f"  {fails}", flush=True)


if __name__ == "__main__":
    main()
