# Pi 4 bench: the FM profile (FM plan M2b, 2026-11-15)

TLDR: a Pi 4 keeps up with FM at good SNR. The burst closest to 25 kbit/s (256l-r5/8 ×39, 92 kbit of payload) arrives 0.38 to 0.44 s after its end, against 0.27 to 0.31 s on HF. Listening costs less on FM than on HF: 17% of a core against 21%. The risks are failing bursts, which spend the whole 1 s DD budget, and per-symbol noise's soft-bit cost of about 55 ms on a 256-point burst.

Setup:
- Raspberry Pi 4 (`minstrel`, 8 GB), RelWithDebInfo (NEON), pool of 2 threads, DD budget 1 s, best of 3.
- One tree (`fm-rx`, PR #206) built twice, with `-DDATA2G_PROFILE=hf` and `=fm`. FM uses 16 kHz, a ±12.5 Hz search, threshold 28.7 and per-symbol noise on.
- Bursts come from `runs/pi_gen_fm206.sh`: each profile's own Python, `phy_session` mpp, seed 1.
- The two profiles' bursts see different noise and fading draws (different sample rates), so compare timings, not decode counts.
- Driver: `runs/pi_fm206_run.sh`. Raw output: `runs/pi-fm206/run.txt`.

## Listening on noise, all bands, 120 s

| | per 20 ms feed | of a core | p99 feed | max feed |
|---|---|---|---|---|
| HF | 4.20 ms | 21.0% | 64 ms | 122 ms |
| FM | 3.46 ms | 17.3% | 58 ms | 103 ms |

FM has twice the samples. The ±12.5 Hz search (3 CFO bins against 25) more than pays for that, and both stay under the 250 ms search hop.

## Burst end to payload (ms)

| burst | decoded HF / FM | HF receive / soft / prepare / total | FM receive / soft / prepare / total |
|---|---|---|---|
| 16qam-r2/3 ×19 mpp 12 | 9/19, 1/19 | 155 / 5 / 227 / 1372 | 150 / 10 / 286 / 1161 |
| 16qam-r2/3 ×19 mpp 16 | 19/19, 19/19 | 104 / 10 / 87 / 202 | 146 / 10 / 50 / 208 |
| 16qam-r2/3 ×19 mpp 20 | 19/19, 19/19 | 103 / 4 / 24 / 133 | 147 / 10 / 34 / 192 |
| 64l-r3/4 ×30 mpp 18 | 0/30, 0/30 | 109 / 18 / 347 / 1141 | 137 / 38 / 324 / 1175 |
| 64l-r3/4 ×30 mpp 24 | 30/30, 30/30 | 102 / 18 / 89 / 210 | 161 / 20 / 70 / 253 |
| 64l-r3/4 ×30 mpp 30 | 30/30, 30/30 | 79 / 8 / 97 / 187 | 162 / 20 / 49 / 233 |
| 256l-r5/8 ×39 mpp 20 | 1/39, 0/39 | 103 / 83 / 520 / 1187 | 150 / 111 / 551 / 1262 |
| 256l-r5/8 ×39 mpp 24 | 39/39, 39/39 | 149 / 37 / 117 / 305 | 186 / 94 / 157 / 440 |
| 256l-r5/8 ×39 mpp 30 | 39/39, 39/39 | 158 / 40 / 73 / 274 | 186 / 95 / 96 / 379 |

- **Receive** (demodulation and channel estimation) is 35 to 80 ms more on FM: twice the samples through the front end.
- **Soft** is per-symbol noise: a brute-force nearest-point search over every constellation point, every cell. On 256l ×39 it costs about 55 ms, about 15 ms on 64l and about 5 ms on 16-QAM.
- **Failing bursts** run DD to its 1 s budget on both profiles (1.1 to 1.4 s end to payload).

## What it means for FM (M2b)

- **Converging bursts are close to a 0.3 s turnaround.** 25 kbit/s in a 4 s burst carries about this burst's 92 kbit of LDPC work in a third of the air time. Receive cost scales with carriers, so a wide band would add some.
- **Failing bursts are not.** FM needs a shorter DD budget, or decoding that runs while the burst arrives. Today every codeword spans the whole burst, so nothing can be decoded before its end. The loss cost of either must be measured on the simulator.
- **Per-symbol noise wants a cheaper nearest-point search.** On square QAM, slice per axis. On the learned sets, use a coarse grid lookup.
- Nothing here yet calls for advertising a rate capability. Revisit once a wide band exists to bench.
