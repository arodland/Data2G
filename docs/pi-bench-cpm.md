# Pi 4 bench: #125 + #126 + #127 vs master (2026-10-27)

TLDR: safe on Pi timing. Worst CPM burst end to payload is ~460 ms on the branch (35% of TURN_S 1.3 s) against ~260 ms on master. The cost is #125's `cpm::refine`, not #126's GF decode.

Setup: Raspberry Pi 4 (`minstrel`, schedutil governor, syncthing running), default build (RelWithDebInfo, NEON), pool 2 threads, DD budget 1 s, best of 3.
Bursts: `scripts/pi_genburst.py` via `runs/pi_gen_all.sh`, each tree's own data2g (protocol 13 vs 14), AWGN (CPM) or mpp (OFDM), seed 1.
Timing: `data2g-bench-rx` (now takes CPM bursts, `--listen`, `--shifter`). CPM "receive" is `cpm::receive` (refine + soft); "acq" is the head's lock search, which runs live during the burst and is not in the total. "genie": no lock found, decoded from the true lock (the worst case: data fails throughout).

## CPM burst end to payload (ms, total = receive + prepare + decode)

| case | master | branch | of which refine (branch) |
|---|---|---|---|
| fsk8r50 x9 converging | 56 | 185 | 135 |
| fsk8r50 x9 failing | 123-127 | 249-253 | 135 |
| fsk16r25 x9 converging | 90-258 | 345-485 | 268-372 |
| fsk16r25 x9 failing | 159-164 | 419-432 | 268 |
| fsk32r62 x9 converging | 49-67 | 146-153 | 116 |
| fsk32r62 x9 failing | 120-165 | 263-273 | 114-121 |
| fsk16r25 x1 (2 slots) | 15-86 | 284-377 | 270 |

- Refine is per burst and nearly independent of length: c16r25 ~270 ms, c8r50 ~135 ms, c32r62 ~115 ms. Single threaded (same at --threads 1). x86: 23 / 11 / 13 ms, so the Pi is ~12x slower here.
- Failing GF decode (prepare, 8 data codewords): 70-126 ms branch vs 74-107 ms master. Not a problem; no iteration cap change needed.
- The spread between runs (e.g. 345 vs 485 ms) is the schedutil governor; best-of-5 reruns land at the low end.
- Worst seen: branch 485 ms (37% of TURN_S), at --threads 1 499 ms (fsk16r25-r1/3 x9 failing).

## Listening CPU, streaming Receiver on noise, 20 ms feeds (120 s)

| | master | branch |
|---|---|---|
| CPM only, mean | 2.66 ms (13.3% of a core) | 3.19 ms (15.9%) |
| CPM only, p99 / max per feed | 78 / 83 ms | 53 / 73 ms |
| all bands, mean | 8.96 ms (44.8%) | 10.45 ms (52.2%) |
| all bands, p99 / max per feed | 121 / 131 ms | 137 / 198 ms |

x86 for scale: CPM only 1.32% -> 1.83%, all bands 3.95% -> 4.56%.

## Outcome model, GearShifter::recommend (400 random observations)

| | median | p90 | max |
|---|---|---|---|
| master (cnpd) | 27.1-27.3 ms | 35-80 ms | 35-98 ms |
| branch (cordf) | 27.1-27.2 ms | 35-80 ms | 36-99 ms |

Same distribution; the p90 tail moves between runs on both. x86: 2.11 ms median both.

## OFDM sanity (ms, total; mpp)

| burst | master | branch |
|---|---|---|
| w48-256l-r5/8 x39 @20 (2/39) | 1513 | 1589 |
| w48-256l-r5/8 x39 @22 (36/39) | 1282 | 1283 |
| w48-256l-r5/8 x39 @24 | 571 | 574 |
| w48-16qam-r2/3 x19 @12 (4/19) | 1210 | 1213 |
| w48-16qam-r2/3 x19 @14 | 369 | 370 |

No regression. The marginal OFDM cases exceed TURN_S on both trees, as before.

## Not done

- No GF iteration cap study: failing GF decodes cost about what binary ones did on the Pi.
- Refine speed-up for c16r25 (270 ms) is the lever if more margin is wanted; not attempted.
