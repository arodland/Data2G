# Pi 4 bench: #172 (coherent CPM) vs its base (2026-11-13)

TLDR: live end-of-burst latency is fine (streaming: 39-110 ms). At 79c5289 the listener's CPU doubled and its worst all-bands feed (270 ms) passed the 250 ms search hop; f6a3832's pre-gate (`COH_PREGATE`) fixes it: CPM only 6.8% of a core (base 5.8%), all bands 21.2% (18.7%), worst feed 149 ms (99 ms).

## After the pre-gate (f6a3832; same Pi session for both trees, `run2.sh`, raw `run2.txt`)

| listening on noise, 120 s | base | #172 f6a3832 | #172 79c5289 |
|---|---|---|---|
| CPM only, mean | 1.16 ms (5.8%) | 1.36 ms (6.8%) | 3.07 ms (15.3%) |
| CPM only, p99 / max feed | 16 / 31 ms | 26 / 81 ms | 62 / 139 ms |
| all bands, mean | 3.73 ms (18.7%) | 4.23 ms (21.2%) | 6.38 ms (31.9%) |
| all bands, p99 / max feed | 49 / 99 ms | 63 / 149 ms | 112 / 270 ms |

Bursts unchanged by the gate (fsk16r25-r1/2 x9 at -15 dB 8/9 in 376 ms, fsk32r62-r1/2 x9 at -11 dB 8/9 in 815 ms).

The rest of this page is the 79c5289 run.

- Listening on noise: CPM only 5.7% -> 15.3% of a core, all bands 19.0% -> 31.9%; max feed CPM only 29 -> 139 ms, all bands 104 -> 270 ms. The cost is `coh_read_header` in `find` / `late_lock` at every detect placement.
- Batch end to payload (the file-burst path, no streaming): converging bursts +5 to +40 ms; failing bursts run DD to the 1 s budget (fsk16r25 x9 at -18 dB: 137 -> 1135 ms). Live, the engine streams: those passes run while the burst arrives.
- Bursts the coherent receiver now decodes cost DD time that the base never spent (fsk32r62-r1/2 x9 at -11 dB: 0/9 in 96 ms -> 8/9 in 860 ms, batch).

Setup: Raspberry Pi 4 (`minstrel`, 8 GB, schedutil), RelWithDebInfo (NEON), pool 2 threads, DD budget 1 s unless noted, best of 3.
Trees: base 8118dec and #172 79c5289, each with the bench tools of `pi-bench-primed` (ce04775) laid over it. On the head that is a merge (branch pi-bench-coherent); on the base the files apply unchanged.
Bursts: `scripts/pi_genburst.py` via `runs/pi_gen_all.sh` with the base's data2g (#172 changes no TX), AWGN (CPM) or mpp (OFDM), seed 1.
Raw output: main checkout `runs/pi-bench-coherent/run.txt`.

## Live path: streaming vs batch end latency (#172 only, ms)

`data2g-bench-rx --cpm` (synthetic bursts), from the last sample fed to every data slot decoded:

| burst | sigma | decoded | batch | streaming |
|---|---|---|---|---|
| fsk16r25-r1/3, 2 data cw | 6.0 | 2/2 | 89 | 39 |
| | 7.5 | 2/2 | 106 | 55 |
| | 8.5 | 2/2 | 135 | 70 |
| | 9.5 | 1/2 | 193 | 110 |
| fsk32r62-r1/3, 7 data cw | 3.0 | 7/7 | 51 | 43 |
| | 4.5 | 7/7 | 89 | 47 |
| | 5.0 | 6/7 | 335 | 88 |

All data passes ran before the burst's end. Unlimited DD budget gives the same numbers.

## Batch path: burst end to payload (ms; base / head; head with no DD budget)

| burst | decoded base / head | base | head | head, no budget |
|---|---|---|---|---|
| fsk16r25-r1/2 x9 -13 dB | 9/9, 9/9 | 99 | 131 | 133 |
| fsk16r25-r1/2 x9 -15 dB | 0/9, 8/9 | 136 | 377 | 376 |
| fsk16r25-r1/2 x9 -18 dB | 0/9, 0/9 | 137 | 1135 | 1318 |
| fsk16r25-r1/3 x2 -18 dB | 0/2, 0/2 | 79 | 219 | 256 |
| fsk8r50-r1/2 x9 -13 dB | 0/9, 8/9 | 78 | 345 | 346 |
| fsk8r50-r1/2 x9 -16 dB | 0/9, 0/9 | 78 | 1043 | 1044 |
| fsk32r62-r1/2 x9 -11 dB | 0/9, 8/9 | 96 | 860 | 815 |
| fsk32r62-r1/2 x9 -14 dB | 0/9, 0/9 | 142 | 1051 | 1599 |
| fsk32r62-r1/3 x5 -6 dB | 5/5, 5/5 | 42 | 41 | 38 |

- Refine is the same on both (c16r25 63-80 ms, c8r50 ~26, c32r62 21-30).
- A failing data codeword costs ~125 ms of DD on c16r25, ~180 ms on c32r62 (4 passes, 5 GF decodes). On air that codeword lasts 9.6 s / 3.1 s, so streamed it is 1-6% of a core while the burst arrives; only the last codeword's passes (and a 1-codeword burst's) land after its end.
- OFDM sanity: w48-256l-r5/8 x39 and w48-16qam-r2/3 x19 within the run-to-run spread (e.g. 903 / 894 ms, 224 / 224 ms).

## Listening, streaming Receiver on noise, 20 ms feeds, 120 s

| | base | #172 |
|---|---|---|
| CPM only, mean | 1.14 ms (5.7% of a core) | 3.07 ms (15.3%) |
| CPM only, p99 / max feed | 16 / 29 ms | 62 / 139 ms |
| all bands, mean | 3.79 ms (19.0%) | 6.38 ms (31.9%) |
| all bands, p99 / max feed | 50 / 104 ms | 112 / 270 ms |

## Outcome model, GearShifter::recommend (380 observations)

base (cordf): median 14.7 ms, p90 17.3, max 18.1. #172 (cohc4): median 9.9, p90 23.1, max 28.0. Same model shapes; the spread is the governor.

## Before merge

- Done in f6a3832: `coh_read_header` only where the noncoherent score reaches `COH_PREGATE`, and the `Received::passes` initializer warning.
- Optional: on a burst the receiver will not decode at all (every pass fails), the remaining passes could stop early; streaming already hides most of this.
