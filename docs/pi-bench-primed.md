# Pi 4 bench: #135 (primed reception) + #134 (CW ID) vs master (2026-10-28)

TLDR: #134 is free on the Pi. #135's retries are free; its expected slot pushes one feed per turn past the 250 ms search hop when the session expects 3 CPM grids (all bands: 281-324 ms, best of 3). One grid per search hop fixes that.

Setup: Raspberry Pi 4 (`minstrel`, schedutil, syncthing running), RelWithDebInfo, pool 2 threads, DD budget 1 s, best of 3.
Trees: master (6a99cce) and #135 + #134 merged locally (primed-rx 96bcf2c + cwid-sounding c1d1838; conflicts were additions on both sides, kept both).
Each PR's measured path is the same code in the merged tree: #135's decode, receive and refine paths only gain functions; `--no-cwsense` drops #134's detector from the listen loop.
Bursts: master's data2g (all trees protocol 15, same modulation), `runs/pi-bench-primed/pbp_gen.sh`.
Tools: `data2g-bench-rx` gains `--primed` (`cpm::prime_lock` over TOL_S either side of the true start; 6 `decode_known` on slot 0, 12 core bits unknown, the first 5 hypotheses wrong), `--listen S --expect G` (an expected slot every 8 s in G grids), `--no-cwsense`, `--cwid`; `pi_genburst.py` makes ARQ_DUP control pairs (`:dup`).

## #135 receive path (ms, Pi)

| | master | #135 |
|---|---|---|
| primed retry, 6 tries, CPM control (any grid, single or pair) | - | 2.0-2.3 |
| primed retry, 6 tries, OFDM polar ACK / polar modes | - | 0.7-4.4 / 7.7 (polar-k192-f8) |
| prime_lock per window, c16r25 / c8r50 / c32r62 | - | 42 / 42 / 53-62 |
| worst burst end to payload (fsk16r25 x9 failing) | 311-319 | 309-321 + 2.2 retry (25% of TURN_S) |
| blind fail, primed success (fsk32r62 x3 dup, -13 dB) | 116 | 112 + 2.3 |

- prime_lock is 4-6x x86, not 12x: 42-62 ms per grid, not 0.1-0.15 s. It runs mid burst, in the receiver's feed, not at burst end.
- Locks it found where `cpm::find` missed: fsk16r25 x1/x3d/x9 at -18, fsk8r50 x2/x3d at -15.5.

## Listening, 20 ms feeds on noise (80 s, 3 reps; max: best rep, worst rep)

| | mean (of a core) | max feed, window feeds |
|---|---|---|
| CPM only, master | 14.3-16.6% | 45-139 ms |
| CPM only, +CwSense | 14.5-17.8% | 45-69 ms |
| CPM only, expect 1 grid | 14.9-17.5% | 84-126 ms |
| CPM only, expect 3 grids | 16.4-20.4% | 194-562 ms |
| all bands, master | 51.6-59.1% | 174-281 ms |
| all bands, +CwSense | 52.5-58.2% | 191-372 ms |
| all bands, expect 1 grid | 50.7-53.4% | 176-254 ms |
| all bands, expect 3 grids | 52.3-53.2% | 281-324 ms |

- CwSense: ~10 us a feed (p50 7 -> 17 us), 0.05% of a core. The max spread on master and CwSense alike is the governor and syncthing.
- The expected slot runs prime_lock on every expected grid in one feed (`Receiver::primed`), on top of that hop's search. Elsewhere in the same runs the max is 141-158 ms.

## #134 TX, sanity

- `cwid::audio` at the slowest (15 wpm, "DE VE3ABC", 11 s): 52 ms; 20 wpm: 38 ms. Built in `start_tx`, so it delays a CW ID's key-up once.
- CPM and OFDM burst latency: master vs merged within the run-to-run spread (e.g. fsk16r25 x9 failing 311-319 vs 309-321 ms; w48-256l x39 1287 vs 1289 ms).
- `GearShifter::recommend`: 28.0 vs 27.2 ms median, p90 37 vs 35 ms.

## Before merge

- #135: in `Receiver::primed`, try one grid per search hop (keep `expected_` with the rest; the burst's second header copy is still in the kept window). Each hop then adds at most 62 ms: all bands about 200-215 ms, under the hop. Putting the last-sent mode's grid first keeps the usual case at one prime_lock.
- #134: nothing. Caching the CW ID audio by (call, wpm) would remove the 38-52 ms, optional.
