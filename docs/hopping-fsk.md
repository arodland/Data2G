# Wider FSK: parallel tones and frequency hopping (study, 2026-10-07)

Whether spreading the CPM modes wider than their 400 Hz tone grids buys
anything. Two ideas:

- **Parallel tones**: several grids side by side, one tone in each at
  once, for speed.
- **Hopping**: one tone at a time, each symbol in one of several copies of
  the grid, for robustness at the same rate.

Parallel tones lose badly. Hopping, once its TX filter is tuned, gains
1-3.6 dB in one well-defined pocket: slow fading (≤ 0.2 Hz) with the
second path 0.25-1.25 ms late. Elsewhere it is within ±0.4 dB, and
0.25 dB better averaged over a random channel ensemble. It is not a mode:
the numbers below are with genie sync, and a hopped mode needs about
2150 Hz against plain fsk8r50's 620. Tuning the hopped TX filter also
showed that the shipped CPM filter left 0.3-1.1 dB on the table, which
PR #49 collects (`cpm.TX_FILTERS`, per bandwidth cap).

Everywhere below, a threshold is the SNR in 2500 Hz with noise against
the burst's envelope peak (PEP-fair), at which 10% of r1/2 LDPC codewords
fail. The demodulator is `cpm.llrs`, with timing and CFO given (genie).

## Parallel tones (scripts/mfsk_parallel_study.py)

fsk8r50 and fsk16r25 grids ×2 and ×4, each copy sending its own symbol,
through `dsp.tx_condition` at clip headroom -6..+4 dB or no clip. Each
point is 40 trials. The table gives the cost over one grid at the best
clip, in dB:

| | awgn | mpg | mpp | mpd | mps |
|---|---|---|---|---|---|
| fsk8r50 ×2 | +5.0 | +2 | +5 | +6 | +4 |
| fsk8r50 ×4 | +8.5 | +7 | +10 | +9 | +11 |
| fsk16r25 ×2 | +5.0 | +6 | +6 | +5 | +7 |
| fsk16r25 ×4 | +9.0 | +10 | +11 | +9 | +11 |

Why the cost is so high:

- **Rate.** N tones split the power N ways: 3 or 6 dB for 2× or 4× the
  rate.
- **Peak.** The tones beat against each other. The best clip leaves
  about 3 dB of envelope peak over average at ×2 and 3-3.6 dB at ×4,
  against a single tone's 1 dB.
- **Clip products.** The grid puts every tone on a multiple of the symbol
  rate, so the intermodulation from clipping lands on other copies' tone
  bins, inside the band where the filter can't remove it.

Every ×2/×4 point is dominated by a mode already on the ladder near its
rate: fsk32r62 r1/3 and r1/2 below 130 bps, OFDM QPSK above.

## Hopping

Each symbol goes in one of K copies of the 8-tone fsk8r50 grid, by a
fixed sequence (`scripts/hop_tx.py`):

- **k4**: four copies interleaved tone by tone; copy c is on tones 4j + c.
  1600 Hz of tones, about 2150 Hz at -37 dB.
- **k4b**: four copies side by side.
- **g2-1000**: two copies 1000 Hz apart, about 1950 Hz at -37 dB.

The receiver reads only the active copy's bins for each symbol, so the
noise per decision is unchanged. Any AWGN cost is the TX peak's.

### The first sweeps were misleading

**Fading presets.** The early comparisons used the fading presets with
100 trials a point. Hopping seemed to gain about 3 dB on mpg (0.1 Hz,
0.5 ms) and nothing on mps (0.15 Hz, 2 ms). Two problems:

- **Too few trials for slow fading.** A 6-10 s codeword sees less than
  one fade at 0.1 Hz. The 10% point is then set by about the ten worst
  bursts, and moved ±1.5 dB from one delay to the next.
- **"mpg vs mps" is really delay.** The two-path channel's response
  repeats every 1/delay in frequency. At 2 ms that is every 500 Hz, which
  one 400 Hz grid already spans. At 0.5 ms it is every 2 kHz: the grid
  sees one flat fade, and hopping across 1.6 kHz finds the other half.

Fix: one threshold per draw (`scripts/hop_study.py`, two batched decodes
to 0.125 dB), every variant on the same payload, channel and noise, and
paired bootstrap intervals (`scripts/hop_report.py`).

**The untuned TX filter.** At the shipped filter (bp 50, 3 passes) the
hopped modes' envelope peak rose to 1.6-1.7 dB against plain fsk8r50's
1.15 dB, from the filter ringing on the larger frequency jumps. That cost
0.4-0.5 dB on AWGN, matching the peak rise exactly, and hid most of the
diversity gain. The comparison only means something with every variant's
filter tuned, plain fsk8r50's included.

### TX tuning (runs/hop_tx_shape.tsv, runs/hop_awgn.csv.gz)

`scripts/hop_tx_shape.py` sweeps peak and band edges over the filter
margin, clip headroom, passes, glide, dwell and split passbands.
`hop_study.py` then measures the AWGN threshold on 200 paired draws:

- **The clipper does nothing here.** Headroom from 0 to -2 dB gave
  identical peaks.
- **The fix is the margin plus a glide.** The ringing comes from the
  filter's skirts, so a 150-350 Hz bandpass margin plus a 0.1-symbol
  frequency glide takes every variant to a 0.1 dB peak.
- **More glide is worse.** At 0.3 the receiver loses more energy than
  the peak saves.
- **Dwell.** Hopping every 4 symbols helps g2 a little.

AWGN, tuned, against shipped fsk8r50:

| variant | peak | 10% point |
|---|---|---|
| fsk8r50 as shipped (bp 50) | 1.16 dB | 0 |
| fsk8r50, bp 350, glide 0.1, 6 passes ("K1W") | 0.08 | -1.25 |
| k4, bp 350, glide 0.1, 6 passes ("K4") | 0.10 | -1.12 |
| g2-1000, bp 350, glide 0.1, 6 passes, dwell 4 ("G2") | 0.11 | -1.12 |

All of that gain is the filter: plain fsk8r50 with the wide filter gets
it too. That is what PR #49 ships at the 1200 and 2400 Hz caps (bp 150,
within 0.05 dB of bp 350 end to end). The fair baseline for hopping is
K1W.

### Random ensemble (runs/hop_ens.csv.gz, 4000 draws)

`hop_tx.draw_paths`:

- 2 paths (70%) or 3;
- extra delays log-uniform 0.1-5 ms;
- extra path powers uniform -12..0 dB;
- Doppler log-uniform 0.05-2 Hz.

This ensemble is an invented spread of conditions, not a measured HF
distribution. Each cell is the difference from K1W, dB, with its 90%
interval:

| subset | n | K4 | G2 |
|---|---|---|---|
| all | 4000 | -0.25 [-0.38, -0.12] | -0.12 [-0.36, 0.00] |
| all, 5% point | 4000 | -0.62 [-0.88, -0.38] | -0.25 [-0.51, 0.00] |
| Doppler 0.05-0.15 Hz | 1164 | -0.62 [-1.09, -0.38] | -0.12 |
| Doppler 0.4-2 Hz | 1781 | -0.12..-0.25 | -0.12..-0.25 |
| slow (< 0.4 Hz), longest delay 0.4-0.8 ms | 379 | -0.88 [-1.48, -0.38] | -0.40 |
| slow, longest delay 1.5-3 ms | 441 | 0.00 | +0.12 |

Shipped fsk8r50 (bp 50) is +1.12..+1.25 behind K1W in every subset.

### Delay × Doppler grid (runs/hop_grid.csv.gz)

Two equal paths, 300 paired draws a cell. K4 minus K1W, dB, with the 90%
interval in the cells that matter:

| delay \ Doppler | 0.05 Hz | 0.1 | 0.2 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|
| 0.1 ms | +0.1 | 0.0 | +0.1 | 0.0 | -0.1 | +0.1 |
| 0.25 | **-1.9** [-2.5, -1.0] | **-2.1** [-2.9, -1.2] | -0.1 | -0.4 | -0.5 | -0.1 |
| 0.5 | **-3.6** [-4.7, -2.6] | **-3.1** [-4.0, -2.4] | **-1.0** [-1.5, -0.4] | -0.3 | -0.5 | -0.1 |
| 0.75 | **-1.0** [-2.9, -0.1] | **-1.0** [-2.2, 0.0] | -0.5 | -0.2 | -0.4 | +0.1 |
| 1.0 | **-1.7** [-2.4, -1.0] | **-1.0** [-2.2, -0.5] | -0.5 | -0.4 | -0.4 | +0.1 |
| 1.25 | **-1.5** [-2.2, -1.1] | **-1.5** [-2.0, -0.9] | -0.4 | -0.1 | 0.0 | 0.0 |
| 1.5 | -0.2 | -0.2 | 0.0 | -0.1 | -0.1 | +0.2 |
| 2-4 | ±0.4 | | | | | |

**The gain fills plain fsk8r50's weak spot.** K1W's own 10% point is
about 0 dB at 0.5 ms and 0.05-0.1 Hz, against about -3.6 dB at 1.5 ms
and more. At short delay the 400 Hz grid sees one slow, flat fade that
the codeword can't outlast.

Outside the pocket:

- **0.1 ms gains nothing.** The channel is flat even across 2 kHz.
- **Fast fading (≥ 0.5 Hz) gains at most 0.5 dB.** The codeword rides out
  the fades in time.
- **1.5 ms and more gains nothing.** One grid already spans the ripple.

G2 matches or beats K4 at 0.25-0.5 ms (-2.4..-3.9), where 1 kHz is half
the ripple period. It gets nothing at 0.75-1.0 ms, where both copies see
the same channel: a single fixed offset is right only for some delays.

## Where a hopped mode would sit

These are estimates, not measurements: the ladder's thresholds, plus
PR #49's filter change, plus the K4 differences above, all with genie
sync. Values are awgn / mpg / mpp / mpd:

| mode | bps | width | awgn / mpg / mpp / mpd |
|---|---|---|---|
| K4 fsk8r50 | 62 | ~2150 Hz | ~-10.0 / -2.2 / -4.7 / -5.5 |
| fsk8r50, wide filter | 62 | ~620 Hz | ~-10.0 / +0.9 / -4.7 / -5.5 |
| fsk16r25, wide filter | 42 | ~620 Hz | ~-11.9 / -2.4 / -7.2 / -8.0 |

`prune.py` judges awgn, mpg and mpd, so on mpg alone K4 might earn a
place at the 2400 Hz cap. Before calling it a mode:

- **Sync and header.** A hopped Costas preamble and hopped header copies,
  measured end to end; slow fading is also where sync is weakest.
- **Ladder study.** A `ladder_study.py` measurement through the real
  receiver.
- **fsk16r25.** Possibly the same for fsk16r25. Its longer codeword (9.6 s
  against 6.4 s) left it less to gain in the early sweeps.
- **How common is the pocket?** It is real and not a preset artifact, but
  how much it matters depends on how often real paths fade slowly with
  0.25-1.25 ms of delay. mpg happens to sit in its middle.

Not done:

- **More paths.** Channels with more than two paths, or a path smeared
  over a range of delays. In the ensemble, a third path gave K4 -0.38
  against two paths' -0.25.
- **Pseudo-random hop sequences.** All runs used i mod K; the decoder is
  indifferent to the order while fading is slower than a symbol.

## Reproducing

`runs/hop_round.sh` runs it all (4 workers, about 4 h). Tables:

    uv run python scripts/hop_report.py ens runs/hop_ens.csv.gz K1W,K4,G2,K1
    uv run python scripts/hop_report.py grid runs/hop_grid.csv.gz K1W,K4,G2

The parallel-tone runs were one-off sweeps; `mfsk_parallel_study.py`
reruns any point (`grid N clip channel lo hi step`).
