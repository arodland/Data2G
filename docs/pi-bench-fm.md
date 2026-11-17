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

## Decoding while the burst arrives (fm-stream)

TLDR: early decoding cuts a clean burst's end-to-payload on a Pi 4 by 2 to 4 times: 256l x39 at 30 dB 276 -> 191 ms, 64l x30 at 24 dB 553 -> 138 ms, 16qam x19 200-234 -> 104-109 ms. Failing bursts end no later than 1.57 s, against up to 4.1 s without it. Decode counts are within a few slots of whole-burst decoding.

Setup:
- `data2g-bench-rx --stream`: each burst fed through the native Receiver in 20 ms chunks, decoded as the engine does. Pool 2, DD budget 1 s, best of 3.
- FM profile, protocol 2 (8-frame interleaving and estimate windows). Bursts: `runs/pi_gen_fm206.sh`'s set regenerated, plus seeds 1-4 of 256l at 20/22/24 dB and 64l at 18/20 dB, on mpp.
- Driver: `runs/pi_fmstream_run.sh`. Raw output: `runs/pi-fmstream/run.txt` (first version, and no early decoding) and `run-once.txt` (as committed).
- "end": last sample fed to every slot decoded. "during": CPU spent while the burst arrives. Decodes summed over seeds.

| burst | early, as committed: ok, end ms, during s, max feed ms | no early decoding |
|---|---|---|
| 16qam-r2/3 x19, 12 dB | 9/19, 756, 1.2, 234 | 9/19, 1158, 0.1, 28 |
| 16qam-r2/3 x19, 16 dB | 19/19, 109, 0.6, 109 | 19/19, 234, 0.1, 29 |
| 16qam-r2/3 x19, 20 dB | 19/19, 104, 0.6, 95 | 19/19, 200, 0.1, 29 |
| 256l-r5/8 x39, 20 dB | 30/156, 1301-1570, 1.9, 308 | 27/156, 1180-1293, 0.0, 20 |
| 256l-r5/8 x39, 22 dB | 56/156, 837-1490, 1.9, 316 | 57/156, 1211-4120, 0.0, 20 |
| 256l-r5/8 x39, 24 dB | 85/156, 602-1407, 1.8, 305 | 89/156, 1238-4008, 0.1, 29 |
| 256l-r5/8 x39, 30 dB | 39/39, 191, 0.9, 158 | 39/39, 276, 0.0, 19 |
| 64l-r3/4 x30, 18 dB | 24/120, 856-946, 1.3, 299 | 24/120, 1093-1126, 0.0, 20 |
| 64l-r3/4 x30, 20 dB | 43/120, 920-982, 1.5, 304 | 45/120, 1116-1824, 0.0, 20 |
| 64l-r3/4 x30, 24 dB | 30/30, 138, 0.5, 117 | 30/30, 553, 0.0, 20 |
| 64l-r3/4 x30, 30 dB | 30/30, 108, 0.5, 90 | 30/30, 142, 0.0, 20 |

Notes:
- The first version retried every failed slot on each window: 110-128 LDPC tries for 39 slots of a failing 256l burst, 4.4 s of CPU during a 5.6 s burst and feeds up to 904 ms. A slot that fails mid-burst now waits for the whole burst (its window's estimate barely changes meanwhile). That halved the CPU during the burst and kept the decodes.
- The partial receives run on the engine's worker, not the audio thread; the longest (about 300 ms) is under the 1.15 s window.
- Open: a failing burst still ends 0.85-1.57 s late, the final pass retrying the failures and spending the DD budget left.

## The frozen FM ladders (f55, f108; 2026-11-17)

TLDR: a Pi 4 keeps up with every FM rung: no rate cap is needed. f108's top rung (256-point r.87, 26 codewords, 25.1 kbps) is decoded 175-208 ms after its end; f55's top rungs 88-107 ms. Failing bursts end within 0.76-1.34 s. Listening with the FM engine's bands costs 5.1% of a core.

Setup:
- Raspberry Pi 4 (`minstrel`), FM build of fm-stream (FM protocol 5), pool of 2 threads, DD budget 1 s, best of 3, decoding while the burst arrives (and without, for comparison).
- Bursts: `runs/pi_gen_fmladder.sh` (new `scripts/pi_genburst_fm.py`): the FM simulator at a station's fixed audio level, the sender's tx_path set by its radio; each top rung clean, just above its 10% static knee, and below it. Driver: `runs/pi_fmladder_run.sh`; raw output `runs/pi-fmladder/run.txt`.

| burst | ok | end ms, early | end ms, no early | CPU during, early |
|---|---|---|---|---|
| f108-256l-r87, flat, CNR 35 / 25 | 26/26, 26/26 | 175 / 208 | 197 / 241 | 0.37 / 0.43 s |
| f108-256l-r87, flat, CNR 21 (failing) | 0/26 | 1266 | 1209 | 1.23 s |
| f108-256l-r87, mobile-rural, CNR 35 | 18/26 | 833 | 1206 | 0.81 s |
| f108-256l-r80, flat, CNR 30 / 23 / 19 | 26, 26, 0 | 168 / 187 / 1340 | 189 / 214 / 1179 | |
| f108-64l-r80, flat, CNR 30 / 17 / 13 | 26, 26, 0 | 99 / 121 / 1097 | 106 / 134 / 1115 | |
| f55-256l-r88, mic, CNR 35 / 29 / 25 | 26, 26, 25 | 102 / 107 / 433 | 112 / 117 / 346 | |
| f55-256l-r88, flat, CNR 30 / 17 / 13 | 26, 26, 0 | 90 / 103 / 845 | 98 / 113 / 1103 | |
| f55-256l-r72, mic, CNR 30 / 17 / 13 | 26, 26, 0 | 88 / 106 / 884 | 99 / 128 / 1092 | |
| f55-64l-r83, mic, CNR 30 / 13 / 9 | 13, 13, 0 | 58 / 71 / 757 | 64 / 80 / 1054 | |
| f55-qpsk-r21, mic, CNR 10 | 6/6 | 39 | 40 | |

Notes:
- Listening (FM engine bands, 120 s of noise): 1009 us per 20 ms feed, 5.05% of a core; p99 feed 14 ms, max 18 ms. (M2b's w48-era figure was 17.3% with every band searched.)
- Decoding while the burst arrives shortens every decodable burst by 6-30 ms on f55 and 7-33 ms on f108, and failing f55 bursts by 0.2-0.3 s. On failing f108 bursts it is 0.02-0.16 s slower (the windows' DD on 108 carriers), with feeds up to 0.6 s on the worker thread (each under its 1.15 s window).
- The worst failing burst ends 1.34 s late, near TURN_S (1.3 s): a failing f108 top-rung burst can delay the reply by a fraction of a second. The gear shifter keeps such bursts rare; worth watching on air.
