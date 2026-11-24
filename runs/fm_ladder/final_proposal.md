# FM ladders: final proposal (FM plan M4/M5), measured

Streamed receiver (scripts/fm_ladder_study.py), 30 paired trials, 10% / 1% codeword failure CNR (dB in 15 kHz), bisection to 0.25 dB. Transmitter FM_TX=1 (scripts/fm_tx.py): flat +3 dB/oct tilt and no TX clipping for 16-QAM up; mic drive 0.70 (64-point) / 0.55 (256-point) of the 15 kHz width-limited drive. bps at the most codewords a burst of at most 4 s carries. ">45": not reached in the grid. Constellations: the existing learned sets only (runs/fm_const/report.txt). Raw knees: runs/fm_ladder/ladder_tx.csv and final.csv.

## f55 (narrow, 450-3150 Hz): one ladder for mic and flat radios

| # | proposed name | constellation | k/n | rate | bps | mic static 10%/1% | mic flutter@440 10%/1% | mic mobile-rural 10%/1% | flat static 10%/1% | flat flutter@440 10%/1% | flat mobile-rural 10%/1% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | f55-qpsk-r21 | gray-qam4 (hr 2) | 456/2200 | 0.207 | 730 | 1.25/1.75 | 1.25/1.75 | 11.25/17.0 | 0.25/0.5 | 0.5/0.75 | 10.5/16.5 |
| 1 | f55-qpsk-r40 | gray-qam4 (hr 2) | 888/2200 | 0.404 | 1421 | 2.75/3.0 | 3.0/3.5 | 14.25/20.0 | 2.0/2.25 | 2.25/2.5 | 13.5/19.25 |
| 2 | f55-qpsk-r60 | gray-qam4 (hr 2) | 1320/2200 | 0.600 | 2138 | 4.0/4.5 | 4.5/4.75 | 17.75/22.75 | 3.25/3.5 | 3.5/4.0 | 16.0/21.75 |
| 3 | f55-16qam-r47 | gray-qam16 (hr 3) | 1040/2200 | 0.473 | 3379 | 5.25/5.75 | 6.0/6.5 | 15.0/18.0 | 4.5/5.0 | 5.0/5.25 | 13.0/16.75 |
| 4 | f55-64l-r50 | c64-w48-r12 (hr 6) | 1648/3300 | 0.499 | 5417 | 7.0/7.75 | 8.0/8.5 | 21.5/23.5 | 6.25/6.5 | 7.0/7.25 | 18.75/20.75 |
| 5 | f55-64l-r67 | c64-w48-r12 (hr 6) | 2216/3300 | 0.672 | 7321 | 8.5/9.25 | 9.5/10.0 | 25.0/28.25 | 7.5/8.0 | 8.25/8.75 | 23.25/26.5 |
| 6 | f55-64l-r75 | c64-w48-r23 (hr 6) | 2472/3300 | 0.749 | 8179 | 9.75/10.5 | 10.75/11.25 | 26.0/31.5 | 8.25/8.5 | 9.0/9.5 | 24.5/28.25 |
| 7 | f55-64l-r80 | c64-w48-r23 (hr 6) | 2640/3300 | 0.800 | 8743 | 10.5/11.25 | 11.5/12.5 | 27.5/32.0 | 9.0/9.25 | 9.5/10.25 | 25.25/28.75 |
| 8 | f55-64l-r83 | c64-w48-r23 (hr 6) | 2752/3300 | 0.834 | 9118 | 11.25/11.5 | 12.25/13.25 | 28.25/32.25 | 9.25/9.75 | 10.0/10.25 | 26.5/29.75 |
| 9 | f55-256l-r67 | c256-w48-r12 (hr 6) | 1480/2200 | 0.673 | 9708 | 14.25/15.5 | 15.25/17.5 | 30.25/41.25 | 10.0/10.5 | 11.0/11.75 | 26.25/39.5 |
| 10 | f55-256l-r72 | c256-w48-r12 (hr 6) | 1584/2200 | 0.720 | 10405 | 15.75/17.25 | 17.0/19.25 | 33.75/44.0 | 11.0/12.0 | 11.75/12.25 | 28.25/39.5 |
| 11 | f55-256l-r80 | c256-w48-r58 (hr 6) | 1760/2200 | 0.800 | 11585 | 19.25/21.25 | 22.5/>45 | 39.25/>45 | 13.25/13.75 | 13.5/14.25 | 30.75/40.75 |
| 12 | f55-256l-r83 | c256-w48-r58 (hr 6) | 1832/2200 | 0.833 | 12068 | 21.25/23.75 | 22.75/>45 | 37.5/>45 | 14.0/15.0 | 14.5/16.25 | 32.75/41.25 |
| 13 | f55-256l-r88 | c256-w48-r58 (hr 6) | 1928/2200 | 0.876 | 12712 | 27.25/30.25 | >45/>45 | 44.0/>45 | 15.75/16.25 | 16.25/17.25 | 35.5/41.75 |

## f108 (wide, 300-5650 Hz), flat radios

| # | proposed name | constellation | k/n | rate | bps | flat static 10%/1% | flat flutter@440 10%/1% | flat mobile-rural 10%/1% |
|---|---|---|---|---|---|---|---|---|---|
| 0 | f108-qpsk-r21 | gray-qam4 (hr 2) | 456/2160 | 0.211 | 1475 | 2.25/2.5 | 2.5/2.75 | 7.5/12.75 |
| 1 | f108-qpsk-r41 | gray-qam4 (hr 2) | 888/2160 | 0.411 | 2870 | 4.0/4.25 | 4.5/4.75 | 11.75/15.5 |
| 2 | f108-qpsk-r60 | gray-qam4 (hr 2) | 1296/2160 | 0.600 | 4237 | 5.5/5.5 | 5.75/6.0 | 15.25/18.75 |
| 3 | f108-16qam-r41 | gray-qam16 (hr 6) | 888/2160 | 0.411 | 5739 | 6.5/7.25 | 7.0/7.5 | 18.25/25.25 |
| 4 | f108-64l-r40 | c64-w48-r12 (hr 6) | 1296/3240 | 0.400 | 8474 | 8.5/9.0 | 9.25/9.75 | 20.25/27.5 |
| 5 | f108-64l-r50 | c64-w48-r12 (hr 6) | 1616/3240 | 0.499 | 10620 | 10.0/10.5 | 10.5/11.5 | 23.5/29.75 |
| 6 | f108-64l-r59 | c64-w48-r12 (hr 6) | 1920/3240 | 0.593 | 12658 | 11.5/12.0 | 12.25/14.25 | 25.75/32.75 |
| 7 | f108-64l-r67 | c64-w48-r23 (hr 6) | 2176/3240 | 0.672 | 14374 | 13.0/14.0 | 13.75/14.5 | 28.0/34.5 |
| 8 | f108-64l-r75 | c64-w48-r23 (hr 6) | 2432/3240 | 0.751 | 16091 | 14.75/15.25 | 15.0/15.75 | 29.0/37.75 |
| 9 | f108-64l-r80 | c64-w48-r23 (hr 6) | 2592/3240 | 0.800 | 17163 | 15.75/16.75 | 16.25/16.5 | 30.5/37.5 |
| 10 | f108-256l-r67 | c256-w48-r12 (hr 6) | 2896/4320 | 0.670 | 19202 | 17.5/18.0 | 18.0/18.25 | 32.0/39.25 |
| 11 | f108-256l-r75 | c256-w48-r58 (hr 6) | 3240/4320 | 0.750 | 21508 | 19.75/20.25 | 20.5/21.0 | 36.0/40.0 |
| 12 | f108-256l-r80 | c256-w48-r58 (hr 6) | 3456/4320 | 0.800 | 22956 | 21.25/21.5 | 22.0/23.25 | 35.5/>45 |
| 13 | f108-256l-r87 | c256-w48-r58 (hr 6) | 3776/4320 | 0.874 | 25102 | 23.75/24.5 | 25.75/27.25 | 38.5/>45 |

## Pruned (measured)

- f55 16qam-r67 (4854 bps): dominated on mic by 64l-r50 (5417 at the same 7.0 dB); on flat only 0.5 dB ahead of it for 10% less rate.
- f55 256l-r47 (6758): dominated by 64l-r67 (7321; mic 8.5 vs 9.5, flat equal at 7.5).
- f55 256l-r60 (8635): dominated by 64l-r80 (8743; mic 10.5 vs 12.25, flat equal at 9.0).
- f55 256l-r75 (10834): 4% over 256l-r72 for 1.25-1.5 dB.
- f55 64l-r86 (c64-w48-r34, 9359): a third 64-point set; 64l-r83 covers the step.
- f108 256l-r59 (16949): near-tie with 64l-r80 (17163, 0.25 dB apart).
- f108 256l-r82 (23385), r83 (23913), r85 (24445): near-ties between r80 and r87.

## Steps, gaps, targets

- f55 mic static: 1.25, 2.75, 4.0, 5.25, 7.0, 8.5, 9.75, 10.5, 11.25, then 14.25 (a 3 dB step into 256-point: the backed-off mic drive), 15.75, 19.25, 21.25, 27.25.
- f55 flat static: 0.25 to 9.25 in 0.25-1.75 dB steps, then 10.0, 11.0, 13.25, 14.0, 15.75. 64l-r75/r80/r83 are 0.25-0.75 dB apart on flat (fine steps there; the mic needs them).
- f108 static: 2.25 to 23.75 in 1.0-2.25 dB steps; no hole.
- Targets: f55 top 12712 bps (target 12750) at 15.75 dB on flat, 27.25 on mic; f108 top 25102 bps (VARA FM 25210) at 23.75 dB.
- Limits: on mic the 256-point rungs run at 0.55 of the width-limited drive (the radio's 5 kHz limiter), so they sit 4-11 dB above flat, and r80/r83/r88 never reach 1% failure on flutter at 440. On flat the top carriers' clean ceiling (about 19 dB on f108: the radio's 6 kHz audio and IF filters cutting FM sidebands) sets the top.

## For the gear shifter (M6)

- Mobile does not follow static: f55 16qam-r47 beats qpsk-r60 on mobile on both radios (15.0 vs 17.75 mic, 13.0 vs 16.0 flat), so qpsk-r60 is a static/flutter rung. f108 256l-r80 reads 35.5 on mobile against r75's 36.0 (noise; equal).
- Mobile 10% knees run 7.5-44 dB, most 1% knees above 30 dB (fade outage): mobile needs its own thresholds, not static plus an offset.
- f55 knees differ by radio by 0.75-11 dB (most at 256-point): thresholds per TX path (mic/flat), i.e. per direction (M6's capability per direction).
- 16-QAM r47 is one mode for both radios: sent at headroom 3 on mic and unclipped on flat (FM_TX), the receiver using headroom 3's clip constants either way; measured fine on flat (4.5 dB).
