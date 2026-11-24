# Windowed interleaving on HF (issue #212, 2026-11-16)

TLDR: don't adopt it on HF. Spread windows cost 0.3-2.3 dB on fading at the 10% point (up to 4.5 dB on n4, whose window holds one codeword) and are free only on AWGN. On a Pi 4 they buy little: clean bursts already end 0.1-0.3 s after the burst, and failing bursts still end 1.1-1.8 s late, paying the DD budget at the end. A per-window channel estimate alone (receiver-only) is roughly neutral.

## Sensitivity: `scripts/hf_window_study.py`

Setup:
- Fixed modes, bursts of about 11.5 s (control + data codewords), hfchannel (CFO ±50 Hz, 10 ppm), `modem.receive`.
- 30 paired seeds: same seed at every SNR and in every arm.
- DD runs its 2 iterations with no wall-clock budget, so machine load doesn't change the result.
- Pure Python. The native core compiles its windows in, so a `with_native` arm despreads env-windowed bursts wrongly (36 vs 37 on a clean 256l burst).
- SNR is average power (2500 Hz). Both arms use the same mode, so PEP doesn't change.
- Metric: the 10% data-codeword failure point. A missed sync counts as every codeword lost.
- Raw CSVs: main checkout `runs/hf-window-212/`.

Each cell is a dB shift against the whole-burst spread; positive is worse. Window sizes in frames (144 ms each): w48 is 6.9 s, w32 4.6 s, w16 2.3 s. e32 is the estimate per 32 frames with no spread window. we16 is spread and estimate per 16 frames, which is what a window-by-window receiver sees.

| mode | chan | 10% pt (dB) | w48 | w32 | w16 | e32 | we16 |
|---|---|---|---|---|---|---|---|
| n4-qpsk-r1/3 | awgn | -9.75 | +0.00 | +0.20 | +0.20 | +0.00 | +0.17 |
| | mpg | -2.06 | +4.00 | +4.50 | +4.50 | +0.00 | +4.50 |
| | mpp | -2.20 | +1.20 | +2.70 | +2.70 | -0.30 | +2.70 |
| | mpd | 0.36 | +0.13 | +0.58 | +0.58 | +0.00 | +0.33 |
| | mps | -3.20 | +0.89 | +0.74 | +0.74 | +0.00 | +0.74 |
| n10-qpsk-r1/2 | awgn | -3.77 | +0.10 | -0.17 | -0.18 | +0.05 | -0.14 |
| | mpg | 3.55 | +0.10 | +0.32 | +1.25 | +0.04 | +1.17 |
| | mpp | 0.44 | +0.12 | +0.44 | +0.69 | +0.25 | +0.57 |
| | mpd | 2.61 | +0.67 | +0.06 | +0.10 | -0.02 | +0.08 |
| | mps | 3.34 | -0.23 | -0.48 | -0.47 | -0.03 | -0.32 |
| w48-qpsk-r1/2 | awgn | 2.04 | +0.00 | -0.01 | -0.00 | +0.00 | -0.00 |
| | mpg | 7.16 | +0.88 | +0.88 | +1.86 | +0.02 | +1.82 |
| | mpp | 5.16 | +0.15 | +0.28 | +0.71 | -0.01 | +0.73 |
| | mpd | 5.61 | +0.42 | +0.52 | +0.65 | +0.02 | +0.66 |
| | mps | 6.98 | +0.40 | +0.49 | +1.55 | +0.01 | +1.55 |
| w48-16qam-r1/2 | awgn | 7.14 | +0.03 | -0.01 | +0.01 | +0.00 | +0.01 |
| | mpg | 13.15 | +0.03 | +0.34 | +1.16 | +0.00 | +1.20 |
| | mpp | 10.12 | +0.27 | +0.39 | +0.68 | -0.02 | +0.69 |
| | mpd | 13.72 | +1.86 | +3.64 | +3.65 | -0.55 | +2.34 |
| | mps | 11.68 | +0.79 | +0.77 | +1.41 | -0.01 | +1.40 |
| w48-64l-r3/4 | awgn | 16.47 | +0.02 | +0.00 | +0.02 | +0.00 | +0.02 |
| | mpg | 23.02 | +1.34 | +1.61 | +2.09 | +0.05 | +2.04 |
| | mpp | 24.02 | -0.56 | -0.71 | -0.29 | -1.21 | -0.39 |
| | mps | 22.97 | +2.15 | +1.33 | +2.27 | -0.01 | +2.10 |
| w48-256l-r5/8 | awgn | 17.96 | -0.01 | -0.02 | -0.03 | +0.01 | -0.01 |
| | mpg | 24.29 | +1.09 | +0.99 | +2.12 | +0.01 | +1.85 |
| | mpp | 23.26 | +0.43 | +0.39 | +0.55 | -0.07 | +0.46 |
| | mps | 23.88 | +1.80 | +1.01 | +2.03 | -0.28 | +1.49 |

Notes:
- HF fades last seconds, so a window takes the whole fade instead of a slice of it. FM's windows were nearly free (at most +0.19 dB) because its fades are short against a 4 s burst.
- n4 has 24 frames per codeword. w16 and w32 give it groups of one codeword (no cross-codeword interleaving), and w48 gives groups of two.
- 64l on mpp has a failure floor near 10% up to 32 dB, so its 10% point is unreliable in every arm.
- 64l and 256l have no mpd cells (no ladder centre).

## Pi 4: `data2g-bench-rx --stream`

Setup:
- `minstrel`, RelWithDebInfo, pool 2, DD budget 1 s, best of 3.
- One HF build per window. Each is a bench-only copy with `native/core/generated/config.hpp` edited to SPREAD_WINDOW = EST_WINDOW = W and DECODE_EARLY on.
- Bursts: the FM bench's HF set, plus seeds 2-4 of marginal 256l/64l on mpp, generated per window (DATA2G_SPREAD_WINDOW = DATA2G_EST_WINDOW = W, same channel seeds).
- Raw output: main checkout `runs/hf-window-212/pi-run.txt`.

Cells are decoded / end ms (last sample fed to every slot settled) / max feed ms.

| burst | w0 (today) | w16 early | w16 no-early | w32 early |
|---|---|---|---|---|
| 16qam-r2/3 x19, 12 dB | 9/19, 1613, 58 | 9/19, 1051, 385 | 9/19, 1134 | 10/19, 1329, 159 |
| 16qam-r2/3 x19, 20 dB | 19/19, 107, 39 | 19/19, 92, 69 | 19/19, 153 | 19/19, 104, 83 |
| 256l-r5/8 x39, 20 dB s1 | 1/39, 1115, 39 | 17/39, 1234, 431 | 17/39, 1104 | 17/39, 1526, 403 |
| 256l-r5/8 x39, 24 dB s1 | 39/39, 236, 39 | 39/39, 448, 119 | 39/39, 572 | 39/39, 521, 164 |
| 256l-r5/8 x39, 24 dB s3 | 0/39, 1111, 39 | 23/39, 1229, 293 | 23/39, 1512 | 16/39, 1752, 210 |
| 256l-r5/8 x39, 30 dB | 39/39, 196, 39 | 39/39, 150, 102 | 39/39, 176 | 39/39, 203, 147 |
| 64l-r3/4 x30, 18 dB s1 | 0/30, 1070, 38 | 15/30, 1060, 356 | 15/30, 1329 | 15/30, 1425, 40 |
| 64l-r3/4 x30, 24 dB | 30/30, 132, 39 | 30/30, 138, 79 | 30/30, 129 | 30/30, 197, 40 |
| 64l-r3/4 x30, 30 dB | 30/30, 112, 39 | 30/30, 108, 74 | 30/30, 108 | 30/30, 217, 41 |

- Clean bursts gain at most about 50 ms. They already end 0.1-0.3 s after the burst on HF, unlike FM's 0.2-0.55 s.
- Failing bursts end 1.1-1.8 s late in every arm. The DD budget at the end sets that, and a window doesn't remove it.
- CPU during the burst rises from about 50 ms to 0.3-1.3 s, with single feeds up to 430 ms (w16) and 860 ms (w32).
- Windowed bursts often decode part of a burst that the whole-burst layout loses entirely, e.g. 256l at 20 dB, 1/39 vs 17/39 on the same channel seed. That is the same codeword-level trade the 10% points show from the other side. Judging it in ARQ throughput would take a loss study with a predictor trained on windowed bursts. It isn't worth that at these dB costs.

## Next steps

- Close #212 for the format. Keep HF at SPREAD_WINDOW 0, so no protocol bump.
- If the Pi's failing-burst latency matters, the levers are the DD budget and an LDPC iteration cap on non-converging codewords, not the format.
- `EST_WINDOW` alone is receiver-only and within ±0.3 dB on most cells. It needs paired sessions before shipping, because its two gains are in noisy cells (64l mpp has a failure floor).

Reproduce: `python scripts/hf_window_study.py run --seeds 30 --jobs 10`, then `report`.
