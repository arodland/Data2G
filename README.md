# Data2G

HF data modem on the SSTVAE OFDM waveform (24 x 50 Hz carriers, 144 ms
frames, SSTVAE's clipper and pilot layout, a new equalizer), with ML-optimized
submodes. Plan: `~/.claude/plans/i-want-to-design-staged-owl.md`.

- Tests: `uv run pytest` (golden-sample check against `~/code/SSTVAE`
  if present, else skipped)
- Status: plan steps 1-3 (step 3 = torch channel + constellation training) plus a new equalizer. Vendored waveform, burst modem
  (`data2g/modem.py`, `data2g/equalizer.py`), one stub submode (Gray QPSK + Golay(24,12)).

## Equalizer (2026-09-22)

`data2g/equalizer.py` replaced SSTVAE's per-frame Catmull-Rom. It is at
the perfect-channel bound: `scripts/eq_floor.py` runs a genie receiver
with the true channel on the same bursts, fades, timing and CFO.

BMI of uncoded Gray QPSK soft bits (1.0 = a decoder loses nothing),
10 bursts x 20 frames, +37 Hz, 10 ppm, F.1487 Gaussian fading:

| channel | 10 dB old | 10 dB new | genie | 60 dB old | 60 dB new | genie |
|---|---|---|---|---|---|---|
| awgn | 0.991 | 0.996 | 0.996 | 1.000 | 1.000 | 1.000 |
| mpg | 0.862 | 0.894 | 0.892 | 0.986 | 0.990 | 0.986 |
| mpp | 0.854 | 0.902 | 0.907 | 0.970 | 0.997 | 0.996 |
| mpd | 0.686 | 0.878 | 0.892 | 0.758 | 0.986 | 0.986 |
| 2 Hz, 0.5 ms | 0.785 | 0.885 | 0.893 | 0.885 | 0.991 | 0.992 |
| 0.1 Hz, 4 ms | 0.759 | 0.877 | 0.877 | 0.901 | 0.988 | 0.987 |

"old" is SSTVAE's receiver with the stub's naive soft bits, so it
includes the LLR fix as well as the estimator. What changed, receive
side only, on-air format untouched:

1. Residual CFO from all pilots in the burst, with the pilot-rate alias
   (6.94 Hz) resolved by which candidate puts the data on the
   constellation.
2. FFT window placed from the burst-averaged delay profile so every path
   lands inside the CP, instead of "first path if >= 0.5 of the peak".
3. 2-D LMMSE channel estimate: projection onto the delay support across
   carriers, Wiener interpolation over 8 pilots in time with the Doppler
   spread measured from the pilots.
4. Timing-drift steps have their phase undone before interpolation.
5. LLRs use per-cu noise = thermal + clip noise x |h|^2 + estimate MSE.
6. Header channel interpolated between preamble and first frame pilot.
   Header loss at 5-10 dB on mpp/mpd: 1-4 in 40 -> 0-1 in 40.
7. A closing pilot after the last frame (TX change, 24 ms).

Remaining, and where it goes:

- The 60 dB floor on slow selective channels (0.1 Hz, 2 ms: 0.975) is in
  the genie too. It is clip noise leaking between carriers: each OFDM
  symbol's clip noise is not cyclic, so a carrier in a >25 dB notch
  collects its neighbours' (static 2 ms two-path: -41 dB notch at 2 dB
  SDR, others 12-13 dB). Fix at TX: per-symbol cyclic clip-and-filter
  (plan step 8, clipper study).
- mpd at 10 dB is 0.014 BMI under the genie. Decision-directed channel
  re-estimation after decoding would close it; not worth it before real
  codes exist.
- Acquisition and header at <= 0 dB on fading (plan step 6). The header
  check also admitted a wrong codeword count once (mpd, 0 dB).

With SSTVAE's Butterworth taps (`--taps butter`) the 2 Hz channels keep
a gap to the genie (mpd 60 dB 0.930 vs 0.981): 99% of that spectrum
spans 7.4 Hz, beyond the 6.94 Hz pilot rate, so no pilot interpolator
can follow it.

## Constellations (2026-09-22)

`scripts/train_constellation.py` learns point positions (and so the bit
labelling) by maximizing BMI through `data2g/channel_torch.py`: the real
clipper, waveform-domain F.1487 fading, and the numpy estimator. ~12 s
per 300 steps on the local RX 9070 XT.

BMI in bits/cu vs SNR, 8-frame bursts (runs/constellations.png):

| | 10 dB | 18 dB | 26 dB | 35 dB |
|---|---|---|---|---|
| awgn, best Gray QAM | 3.10 | 3.90 | 4.09 | 4.12 |
| awgn, learned 64 (trained at 18 dB) | 3.21 | 4.01 | 4.19 | 4.22 |
| awgn, learned 256 (trained at 26 dB) | 3.18 | 4.03 | 4.23 | 4.26 |
| mpd, best Gray QAM | 2.49 | 3.32 | 3.62 | 3.68 |
| mpd, learned 64 | 2.52 | 3.43 | 3.73 | 3.79 |

About 2 dB at 18 dB; above ~26 dB the QAMs never reach the learned
curve. 16 points gain almost nothing (+0.009 bits/cu at 12 dB): Gray
16-QAM is already near optimal for bit-wise decoding.

Two TX facts this found, both now in `config.py`:

- The clipper scales data by 0.77 relative to the barely-clipped pilots
  (-2.3 dB, Bussgang gain), the same for every constellation. The RX
  must apply it (`modem.data_channel`); ignoring it cost 16-QAM 0.1 BMI.
- Clip threshold is now set from the non-silent part of the burst.
  SSTVAE's whole-signal mean let the lead-in/out silence clip one-frame
  bursts harder (11.1 dB SDR vs 13.0 at 60 frames; now 13.1 flat).

Stock-clipper ceiling: ~4.26 bits/cu of BMI at any SNR on AWGN, i.e.
roughly 3.5-4 info bits/cu after a practical code.

## Codes (2026-09-22, plan step 4)

- `data2g/ldpc.py`: NR LDPC (TS 38.212 BG1/BG2, tables via Sionna,
  Apache-2.0) at any (k, n) with NR rate matching; numpy encoder, torch
  BP decoder. Matches published NR (K=1024 rate 1/2 BPSK: BLER 1e-2 at
  ~1.5 dB). BP, not min-sum: 0.25 dB better at the same speed, and at a
  few kb/s the cost is irrelevant, so neural min-sum was dropped.
- `data2g/polar.py`: CRC-aided polar (GA-DE construction with punctured
  bits, batched SCL, list 8) for the ACK submode.
- `scripts/thresholds.py`: threshold SNR per channel by bisection on the
  GPU path; `scripts/compare_ldpc.py`: finite-length LDPC A/B.

First thresholds (dB, SNR in 2500 Hz; AWGN BER 1e-6, fading PER 1e-2):

| candidate | info bits/cu | awgn | mpg | mpp | mpd |
|---|---|---|---|---|---|
| ACK: QPSK polar k=48, 1 frame | 0.40 | -2.5 | 7.0 | 6.25 | 5.5 |
| ACK: QPSK polar k=48, 2 frames | 0.20 | -5.0 | 3.75 | 2.25 | 0.75 |
| 16-QAM NR LDPC r1/2, 4 frames* | 2.00 | 6.5 | | | 13.75 |

*before the SNR-reference fix below; to be rerun.

Protograph search (PEXIT + genetic algorithm, `scripts/design_ldpc.py`)
beat NR's asymptotic thresholds at every rate (gap to capacity cut
30-75%) and lost to NR at finite length at every rate (k 408-3400):
earlier floors at rates 0.2-0.5 even with degree-2 columns capped at
NR's count, a tie at 3/4 and 5/6, and one outright defect at 2/3, where
the lift made the punctured columns' shifts differ by the same amount
in every row so the punctured info bits are ambiguous (a codeword
confined to them). PEXIT cannot see shifts. NR stays (decided
2026-09-22). In reserve: a shift search scored by simulated BLER at the
target SNR (~5 GPU-hours per rate, expected <= 0.2 dB).

Simulator fix found here: SNR and fading power were normalized per
burst (SSTVAE's `awgn` measures the faded signal; its taps are scaled
per realization). That divides a slow fade out of a short burst, so an
mpg burst in a deep fade simulated as clean. Now taps are unit power in
expectation and SNR is against the transmitted power. The equalizer and
constellation tables above predate it; comparisons within each table
are unaffected (both sides saw the same channel).

## Sync and header (2026-09-22, plan step 6)

`scripts/sync_sweep.py` runs whole bursts through the numpy modem (random
start, +-50 Hz, 10 ppm) and classifies failures: preamble / header / CRC.
What it found, in order:

1. CFO alias picked by data fit chose wrong in 1/3 of 1-frame QPSK
   bursts at 0 dB AWGN (all the "CRC" failures). Now picked by a coarse
   estimate from the decoded header's symbols + first pilot (+-20.8 Hz
   unambiguous). 1-frame ACK at -2 dB AWGN: 0.475 -> 0.995 success.
2. Longer preambles at a fixed threshold detect *worse*: at a true
   preamble the lag-M metric tends to an SNR-set value (median 0.38 at
   -6 dB), and a longer window only removes lucky crossings. The window
   buys a lower noise floor, so the threshold is now calibrated per
   repeat count (`scripts/preamble_noise_floor.py`, 3000 s of noise,
   1.17x the peak as SSTVAE did): 0.42 / 0.278 / 0.217 / 0.181 at 4 / 8 /
   12 / 16 repeats.
3. The Golay header (two words, repeated) was then the binding failure,
   and repeating it more made mpd worse. Replaced by a (192, 16) code
   (d_min 70, chosen from 400 random generators by
   `scripts/design_header.py`) on 4 QPSK symbols, exact ML over all
   65536 words, CRC-4 seeded with the protocol version.
4. Header channel reference from the preamble's last 4 repeats only, and
   the periodic preamble's whole-repeat timing ambiguity (8% of mpd
   locks one repeat off at 16 repeats) resolved by reading the header
   at 0, +-1, +-2 repeats and keeping the best CRC-valid ML score: mpd
   header losses at 0 dB 30 -> 4 of 200.
5. Repeat count 8 (config.PREAMBLE_REPEATS has the table): best end-to-
   end success; 16 detects worse on mpd (decorrelates across itself).

With 8 repeats the preamble's CFO estimate stays inside the alias
boundary under fading (worst of 200 mpd bursts: 2.5 Hz; SSTVAE's 4
repeats produced 3.7 Hz on mpp), so the header-based alias pick is a
safety net now. Known weakness if it is ever needed: a static offset
error of 3.7 Hz or more breaks the header first (its channel is
interpolated linearly over ~130 ms), and the 4-bit CRC tried at 5
alignments then admits a wrong header about 1 time in 3.

2-frame ACK end to end (QPSK polar, 32-bit payload), success of 200:
AWGN -7 dB 0.33, -6 dB 0.76, -5 dB 0.98; mpd -4 dB 0.38 (a short burst
fades as a whole; the transport's retries are the answer there).

## Candidate ladder (2026-09-23, plan steps 5-6 closing)

`scripts/ladder.py` measured 24 candidates x 4 channels on the GPU path
(16-frame bursts, codewords spread over the burst; AWGN at BER 1e-6,
fading at PER 1e-2); `scripts/prune.py` applies the pruning rules
(near-tie -> shorter codeword, then domination with 0.25 dB
resolution, judged on awgn/mpg/mpd). Full output: `runs/prune.txt`.

Kept (payload bits/cu excludes CRC; thresholds dB):

| b/cu | code | awgn | mpg | mpd | mpp |
|---|---|---|---|---|---|
| 0.067 | polar QPSK k48, 4 frames | -8.0 | 0.5 | -6.0 | -5.75 |
| 0.083 | polar QPSK k96, 8 frames | -6.75 | 2.75 | -4.5 | -4.25 |
| 0.167 | polar QPSK k96, 4 frames | -6.25 | 3.0 | -3.75 | -3.5 |
| 0.183 | polar QPSK k192, 8 frames | -6.0 | 4.0 | -3.25 | -3.0 |
| 0.383 | LDPC QPSK r1/5 | -4.75 | 6.5 | -1.25 | -1.0 |
| 0.633 | LDPC QPSK r1/3 | -2.5 | 9.25 | 1.75 | 2.0 |
| 0.967 | LDPC QPSK r1/2 | -0.25 | 12.5 | 5.0 | 5.25 |
| 1.300 | LDPC 16-QAM r1/3, n=3840 | 2.25 | 14.5 | 6.75 | 7.0 |
| 1.467 | LDPC QPSK r3/4 | 3.25 | 17.75 | 10.25 | 10.75 |
| 1.967 | LDPC 16-QAM r1/2, n=3840 | 5.5 | 18.5 | 11.0 | 11.5 |
| 2.633 | LDPC 16-QAM r2/3, n=3840 | 9.5 | 23.25 | 16.5 | 16.5 |
| 2.933 | LDPC learned-64 r1/2 | 13.0 | 25.0 | 20.75 | 19.25 |
| 2.967 | LDPC 16-QAM r3/4, n=3840 | 12.25 | 26.5 | 21.75 | 20.25 |
| 3.411 | LDPC learned-256, r0.44 | 28.25 | 31.5 | fails | 33.5 |
| 3.433 | LDPC learned-64 r7/12 | 20.0 | 32.25 | fails | 27.25 |

Separately (1-codeword bursts): ACK, QPSK polar k48, 1 frame: -2.5 /
7.0 / 5.5 (awgn / mpg / mpd); 2 frames: -5.0 / 3.75 / 0.75.

Notes: nothing at 4 bits/cu decodes through the stock clipper; below
~-6 dB AWGN the preamble/header, not the code, decide (2-frame ACK end
to end: 0.76 success at -6 dB); mpg is 7-14 dB harder than mpd because a
2.3 s burst on 0.1 Hz fading often sits in one fade.

## Narrow bands (2026-09-23)

Bands are contiguous runs of carriers on the same 50 Hz grid, each with
its own low-PAPR pilot (`scripts/design_pilot.py`), TX bandpass,
calibrated preamble threshold, header code and clip constants
(`config.BANDS`, `config.CLIP`). The preamble identifies the band; the
header's submode index is per band. `receive` runs every band's
detector and keeps the best CRC-valid header.

| band | carriers | Hz | pilot PAPR | clip SDR | data gain | preamble threshold |
|---|---|---|---|---|---|---|
| w | 24 | 950-2100 | 0.98 dB | 13.1 dB | 0.77 | 0.278 |
| n10 | 10 | 1300-1750 | 1.25 dB | 13.5 dB | 0.79 | 0.473 |
| n4 | 4 | 1450-1600 | 1.83 dB | 15.1 dB | 0.85 | 0.634 |
| w48 | 48 | 350-2700 | 0.98 dB | 12.3 dB | 0.77 | 0.251 |

w48 (2026-09-23) is data-only; the ACKs stay on the narrower bands. More
carriers make the signal more Gaussian and clip it harder (0.8 dB less
SDR than w). Its 350 Hz low edge is close to the ~300 Hz high-pass many
radio audio chains apply.

What narrowing buys: clip distortion only improves below ~6 carriers
(unclipped PAPR 9.3 / 8.8 / 7.6 dB at 24 / 10 / 4); codes in the
power-limited regime gain ~nothing (per-carrier SNR up, channel uses
down); sync gains, because the lag-M detector is limited by in-band SNR.
Narrow detection uses a bank of band-width filters stepped 100 Hz over
the +-625 Hz CFO range; the bank's maximum is what the threshold was
calibrated against (3000 s of noise), hence the higher thresholds.

Things the narrow bands needed, now in the modem for all bands:
- Pilots inside long headers (a pilot before every 5 header symbols;
  n4's 24-symbol header spans 576 ms): n4 mpd header losses at -5..-3
  dB 69-75 -> 5-16 of 200.
- Acquisition returns runner-up CFO bins and the header picks: a
  4-carrier band shifted 50 Hz overlaps itself in 3 carriers and a
  timing shift makes up the phases (n4 locked one bin off in ~3% of
  AWGN bursts at any SNR; widening the bin search to +-3 around the
  winning filter and checking runners-up fixed all of them).
- Noise estimate on narrow bands: the pilot-residual estimate needs
  spare carriers (4 carriers span a 5 ms delay resolution, so the delay
  support takes them all). Now: projection rank capped at nc - 2 (a
  two-path channel is rank 2 in frequency; binds only on narrow bands)
  and noise from the preamble's repeats (equalizer.preamble_noise). A
  first fallback, noise from the change between frame pilots, counted
  2 Hz fading as noise and failed every n4 candidate on mpd; the first
  narrow ladder (runs/ladder_narrow_badnoise.csv) is void for that.
  The wide band keeps its residual estimate, which also sees the
  pilots' clip distortion (the preamble's cannot: identical repeats
  clip identically); using the preamble's there cost 1.25 dB on mpd.
- Header word submode (4) | n_cw - 1 (6) | CRC-6 (was 8 + CRC-4;
  MAX_CODEWORDS 64), plus a minimum ML score of 0.33 on narrow bands:
  wrong headers accepted after the true one failed, ACKs at -9..-5 dB:
  wide 112 -> 24, n10 -> 0, n4 55 -> 0 (per 900 bursts).

End-to-end ACK success (sync + header + code), 200 bursts, before the
header change above (which only removes wrong accepts):

| | awgn -8 | awgn -6 | mpd -6 | mpd -4 | mpg -4 |
|---|---|---|---|---|---|
| wide ack-2f | 0.045 | 0.765 | 0.16 | 0.385 | 0.605 |
| n10-ack-2f | 0.30 | 0.93 | 0.28 | 0.56 | 0.665 |
| n4-ack-4f | 0.495 | 0.955 | 0.61 | 0.765 | 0.69 |

mpg losses are preamble misses at every SNR: a burst that starts in a
0.1 Hz fade loses its preamble. That is outage, and the transport's
retries are the remedy.

## All bands, combined (2026-09-23)

`scripts/combine_ladders.py` then `scripts/prune.py runs/ladder_all.csv`
(output in `runs/prune_all.txt`): 70 candidates over 4 bands, all
thresholds at PER <= 1e-2, ranked by payload bits/s. 28 kept. Shape:

- n4/n10 own the bottom: 28-333 bps, code thresholds down to -13.5 dB
  AWGN / -9.5 dB mpd. Above ~220 bps every n4/n10 candidate is dominated.
- w (1200 Hz) survives only between ~300 and ~1100 bps.
- w48 (2400 Hz) dominates everything above ~1 kbps: 639 bps to 4.9 kbps
  (QPSK r1/5 at -2.25 dB AWGN to learned-64 r1/2 and 16-QAM r3/4 at
  15.75 dB); learned-64 r7/12 reaches 5.7 kbps on AWGN only (26.5 dB).
- mpg thresholds sit 8-20 dB above AWGN everywhere (slow flat fading:
  outage, the ARQ layer's problem).

Not yet folded in: the sync floor. Codes at the bottom work far below
where sync does (n4 polar: -13.5 dB AWGN code threshold; n4 ACK end to
end 0.50 success at -8 dB), so the lowest submodes are sync-limited, by
~5 dB. Per-band preamble length is the obvious knob (detection is per
band already); it was measured only on the 1200 Hz band.

## Clipper tuning (2026-09-23, plan step 8)

`scripts/clip_study.py` sweeps clip headroom per band and re-thresholds
a panel of submodes with the receiver told that setting's constants
(`scripts/clip_constants.py`). Scored PEP-fair: threshold + post-clip
peak-to-average (dB), since SNR here is average power and a transmitter
is peak-limited. `scripts/pick_headroom.py` picks, per submode, the
lowest headroom within 0.5 dB of the best on every channel (amplifier
non-linearity makes the lower-PAPR end worth that; SSTVAE chose alike).

w48, stock overshoot, headroom 0-5 dB (runs/clip_w48.csv):

| headroom | SDR | peak PAPR |
|---|---|---|
| 0 | 11.1 | 3.96 |
| 1 (stock) | 12.3 | 4.26 |
| 2 | 13.8 | 4.83 |
| 3 | 15.7 | 5.42 |
| 4 | 18.1 | 6.05 |
| 5 | 21.2 | 6.70 |

| submode | pick | PEP-fair gain over stock, awgn / mpd |
|---|---|---|
| QPSK r1/5 | 0 dB | 0.3 / 0.3 |
| QPSK r1/2 | 0 dB | 0.05 / 0.05 |
| 16-QAM r1/2 | 1 dB | 0 / 0 |
| 16-QAM r2/3 | 3 dB | 0.8 / 1.8 |
| 16-QAM r3/4 | 4 dB | 2.5 / 8.7 |
| learned-64 r1/2 | 4 dB | 2.5 / 8.5 |

The optimum rises with rate: noise-limited modes want the clipper
tight, high-rate modes were limited by clip distortion (on mpd the
stock setting's 12.3 dB SDR was the whole story). So the clipper is set
per submode, not per band; it is TX-only, and the receiver looks up the
setting's constants by submode from the header.

## Ladder at picked clip headroom (2026-09-23)

`scripts/ladder.py --clip-picked` re-thresholded all 64 candidates, each
at its own headroom (measured pick, else the nearest studied mode's),
reusing the clip study's rows and starting every search from the old
threshold shifted by the AWGN change (0.5 dB first step, doubling):
~7 s per threshold instead of minutes. Pruned PEP-referenced
(threshold + post-clip peak PAPR); `CANDIDATES.md` shows both scales.

36 kept, 28 bps to 8.2 kbps. What moved:
- The top: 16-QAM r5/6 (5.5 kbps), learned-64 r7/12 (5.7), learned-256
  r1/2 (6.6), and on AWGN only learned-64 r3/4 (7.4) and learned-256
  r5/8 (8.2 kbps, 25.3 dB PEP / 18.0 avg). The stock clipper was the
  ceiling; 4.9 kbps was the top before.
- Learned-256 r1/2 beats learned-64 r2/3 at the same 6.6 kbps.
- The 1200 Hz band's top end is all dominated by the 2400 Hz band's.
- Everything below ~1 kbps clips at 0 dB headroom.

## Constellation retraining (2026-09-23)

The six learned 2400 Hz sets, retrained at each submode's own clip
headroom and operating SNRs (AWGN, mpg and mpd/mpp at their thresholds),
warm-started from the old sets (`scripts/train_constellation.py --band
--headroom --at --init`), then re-thresholded
(`scripts/retest_constellations.py`, runs/ladder_retrain.csv). Current
ladder = runs/ladder_current.csv (clip-picked + retrained rows).

Held-out BMI rose a lot on fading (up to +0.67 bits/cu on mpg) and not
on AWGN; thresholds moved much less. Beyond the 0.25 dB bisection step
only: learned-64 r3/4 mpg -1.0 / mpp -0.75 dB, learned-64 r2/3 mpd
-0.75 dB, learned-256 r5/8 mpg/mpp -0.75 (AWGN +0.25). Learned-256
r1/2 got 0.5 dB worse on AWGN and keeps its old set.

Why so little: the loss is mean BMI over bursts, and a fading channel's
PER is set by its worst bursts, which the mean barely weights (mpg sits
at 4.7-6.9 bits/cu of BMI at thresholds where the codes need 3-5). An
outage-aware loss (penalize bursts whose BMI falls below the code rate)
is the obvious next objective if fading thresholds are worth chasing.

## Sync floor and acquisition rework (2026-09-23)

`scripts/sync_floor.py`: per band and channel, the lowest SNR where
acquisition + header fail in <= 1% of bursts (full numpy receiver,
every band's detector running; success = right submode and codeword
count, start within 2 NCP, the equalizer's re-timing reach). A
submode's end-to-end threshold is max(code threshold, its band's sync
threshold): combined PER <= ~2%.

First run, lag-M detector (runs/sync_floor_lagm.csv; dB, average power):

| band | awgn | mpg | mpp | mpd |
|---|---|---|---|---|
| w | -4.75 | 5.0 | 4.5 | 5.0 |
| n10 | -5.5 | 7.25 | 3.75 | 4.75 |
| n4 | -5.25 | 9.5 | 5.25 | never |
| w48 | -2.25 | 7.75 | 7.0 | 8.75 |

Narrow bands bought almost nothing: the lag-M metric's noise floor
rises as the band narrows (thresholds 0.278 w, 0.634 n4), which spent
the in-band SNR. It would have pruned every narrow mode, so sync was
reworked (`scripts/sync_diag.py` splits failures by stage;
`scripts/header_study.py` tries header variants at genie sync):

1. Detector (sync.py): per-repeat matched filter on a 12.5 Hz CFO
   grid, differential across neighbouring repeats, noise-normalized.
   One threshold for every band (config.PREAMBLE_THRESHOLDS). Misses at
   the old floors went to ~0; the plain power sum lost fading
   preambles to the burst's own data (n4 mpd 0 dB: 14 of 40 locked
   thousands of samples late; 5 with the differential form).
2. Header ML over valid words only (CRC right, submode defined),
   instead of ML over all 65536 then the CRC: genie header failures
   at -8 dB wide AWGN 14% -> 2.3%, w48 79% -> 15%.
3. Header channel reference frequency-smoothed onto a CP-long delay
   support: -8 dB wide AWGN 43% -> 14% (with 2: -> 2.3%).
4. The header is the gate for every band's peaks, runner-up CFO bins
   and +-1, +-2 repeat alignments, with a 0.04 score handicap on
   everything but acquisition's own lock (wide -8 dB AWGN: 6.25% ->
   3.5%). Score floors: 0.30 on n10 and w48, none on w.
5. Bugs on the way: the fine CFO read the whole offset modulo 50 Hz
   as a residual; w48 had no header score floor, so its detector's
   reads of wide bursts outscored the true wide headers at -8 dB.

Earlier bug (fixed before the first run): narrow bands took their CFO
bins from the saturated filter-bank member, so n10 locked 50-400 Hz off
and lost 71% of mpp headers at 34 dB (runs/sync_floor_buggy.csv, void).

6. Detections in time order (sync._crossings), not strongest first:
   every OFDM symbol on the 50 Hz grid is M-periodic within itself, so
   a burst's own data also scores, and on 200 Hz, fading flat across
   the band, n4's preamble ranked 11th-12th among its own burst's
   peaks in 7.5% of mpp/mpd bursts at any SNR. A live receiver meets
   the preamble first. Header checks at the next 4 crossings.
7. One noise level for all CFO bins (the lowest bin's): per-bin levels
   broke when one burst filled most of the buffer (n10 locked 400 Hz
   off on a long n4 burst at 30 dB).
8. n4 syncs on n10 (BandSpec.sync; the user's idea): n10's preamble,
   header and index space, n4's carriers for the frames. n4's own sync
   floor was 8.75 dB on mpp against n10's 0.5: 200 Hz fades as a whole.
   Bursts also get 0.4 s shorter (n10's header). n10's header gained a
   closing pilot, since the first frame pilot may now be on n4's carriers.

8 vs 16 repeats (runs/sync_floor_prehost.csv + runs/sync_floor_r16.csv;
the fading floors are good to ~+-1 dB at 2000 bursts per point):

| band | repeats | awgn | mpg | mpp | mpd |
|---|---|---|---|---|---|
| w | 8 | -7.5 | 1.5 | 2.75 | 3.75 |
| w | 16 | -7.0 | 1.5 | 2.75 | 5.5 |
| n10 | 8 | -9.25 | 4.25 | 0.5 | 2.5 |
| n10 | 16 | -9.5 | 3.5 | -0.25 | 3.75 |
| n4 on n10 | 8 | -9.75 | 4.25 | 0.5 | 1.25 |
| n4 on n10 | 16 | -10.25 | 1.75 | -1.0 | 1.0 |

Kept at 8: 16 is no better on w, within noise on n10, and clearly
better only on n4 mpg, for +160 ms per burst. The robust narrow modes
stay sync-limited on fading (n4 polar f8 decodes at -9.25 dB mpd; sync
needs ~1.25): their codewords spread over 1+ s, the preamble is 0.2 s of
one fade. More repeats close little of that; a trailer (TODO) or
header-assisted detection would.

Final floors on the frozen table (after 7, and with every band's full
set of valid header words; runs/sync_floor.csv, dB, average power),
against the lag-M detector's first run:

| band | awgn | mpg | mpp | mpd | lag-M awgn / mpg / mpp / mpd |
|---|---|---|---|---|---|
| w | -7.5 | 1.0 | 3.0 | 4.25 | -4.75 / 5.0 / 4.5 / 5.0 |
| n10 | -9.5 | 4.25 | 0.0 | 2.25 | -5.5 / 7.25 / 3.75 / 4.75 |
| n4 (on n10) | -9.75 | 3.5 | 0.0 | 1.5 | -5.25 / 9.5 / 5.25 / never |
| w48 | -3.75 | 5.25 | 5.5 | 8.0 | -2.25 / 7.75 / 7.0 / 8.75 |

Within +-0.5 dB of the pre-freeze run (runs/sync_floor_prefreeze.csv)
everywhere, and the pruned set is unchanged by them.

## Submode table, provisional freeze (2026-09-23)

`config.SUBMODES` is the pruned ladder:
`scripts/prune.py runs/ladder_final.csv --sync runs/sync_floor.csv --keep polar-gray-qam4-f1-k48@h0 n4-polar-gray-qam4-f2-k48@h0`,
with end-to-end thresholds (max of code and band sync floor) in
CANDIDATES.md. PROTOCOL_VERSION 10.

| header index space | submodes | payload bps |
|---|---|---|
| w (1200 Hz) | 11 | 56-1639 |
| n10 (n10 + n4, <=500 Hz) | 16 (full) | 28-1022 |
| w48 (2400 Hz) | 15 | 639-8222 |

- The <=500 Hz ladder is pruned among itself too and kept whatever wider
  modes dominate it (500 Hz band-plan segments).
- n4-ack-2f (768 ms) is one too, since the n4 clip re-measurement
  below: n4's LDPC k480 (134 bps, 3.9 s bursts) came within 0.5 dB of it
  on mpg and dominated it.
- ack-1f (k48, 1 frame, 432 ms on air) is a latency exception: pruning
  by payload rate can not value it. Measured as the 1-codeword burst it
  is (runs/ladder_ack1f.csv): -4 dB AWGN, 5.75-6.5 dB fading, code-limited
  everywhere (two pilots, no time diversity).
- Interleavers and polar info sets are committed as
  data2g/format/<band>_<index>.npz (tools/freeze_format.py; `--verify`
  recomputes and compares). tests/test_modem.py fails if a submode
  changes without re-freezing.
- n4's clip constants were re-measured with n10's preamble and header
  in front (runs/clip_constants_prehost.json has the old ones): its
  bursts now pass n10's 500 Hz TX filter, so clip noise spreads over
  500 Hz instead of piling onto 4 carriers. SDR at 0 dB headroom 13.75
  -> 17.8 dB at the same peak PAPR (2.84 dB). Re-thresholded
  (runs/ladder_n4host.csv): mpg 1-1.75 dB better on the LDPC modes,
  the rest within 0.5 dB. Other bands moved <= 0.03 dB.
- Not final: the narrow ladder fills all 16 of n10's indices and may be
  trimmed once there is on-air experience.

## KISS TNC (2026-09-23)

`data2g-tnc` (data2g/tnc.py), after xssfox/freedvtnc2: one submode from
the command line, KISS over TCP, PyAudio, PTT through rigctld.

    uv pip install pyaudio            # plus torch: the decoders run on it
    data2g-tnc --list-modes
    data2g-tnc --list-audio-devices
    data2g-tnc --mode qpsk-r1/2 --input-device USB --output-device USB \
        --rigctld-port 4532 --ptt-on-delay-ms 100 --output-volume -6

- TX: queued KISS data frames (port 0) are combined into one burst,
  as many as fit (`--max-packets-combined` caps it), framed as
  [length u16][packet] across the codewords, zero-padded. A packet
  bigger than a burst holds (`--list-modes` shows the capacity) is
  dropped with an error. `--output-volume` is dB re a burst peak at
  digital full scale.
- RX: every submode sharing the mode's preamble and header decodes (a w
  TNC hears every 1200 Hz submode). A streaming receiver looks for a
  preamble and header in a rolling ~1.5 s buffer (modem.find_burst),
  waits for the rest of the burst the header claims, then demodulates.
  A failed codeword costs the packets it touches, and the rest of the
  burst if it held a length field.
- RX filtering (after the first on-air run, 2026-09-24: real band noise
  set off the 1200 Hz detector every minute or two, and with no header
  score floor there the TNC waited out 20-60 s bursts that were never
  sent). `modem.Accept` limits which headers count:
  - `--rx-modes`: only these submodes' headers are valid words.
  - `--max-burst-secs`: only codeword counts that fit (and TX combines
    no more than fits).
  - `--min-header-score` (default 0.35): Gaussian noise's best of 896
    valid words passes 0.364 in 0.01% of reads, of 64 words 0.325;
    true wide headers at -6 dB AWGN score >= 0.385 in 99% of bursts, at
    -7.5 dB 5-10% fall under 0.35 (so ~1 dB off the 1200 Hz sync floor).
  Each header decode is logged with its score, to tune the floor.
- CPU: one thread per math library (set at the top of tnc.py, before
  numpy loads; an explicit OMP/OPENBLAS/MKL_NUM_THREADS wins). With
  numpy's default OpenBLAS pool, worker threads spinning between the
  detector's small operations took 300-700% CPU while listening; now
  ~10% of one core on any band, measured live on noise. A 32 s burst
  decodes in ~4 s single-threaded.
- Half duplex: capture is dropped while transmitting. Channel access is
  only "don't key while a burst is being received".
- Exits on SIGINT or SIGTERM: stops the KISS server, finishes a
  transmission in progress, unkeys.

Checked on this machine through a PipeWire null sink: TNC A to TNC B,
two packets byte-exact at 28 dB SNR, PTT T 1 / T 0 at a fake rigctld,
both exiting 0 on SIGINT. tests/test_tnc.py covers KISS, framing, the
48 kHz audio path into the streaming receiver, and digital silence.

Known limits: a false header that passes the filter still makes the
receiver wait for the burst it claims (at most `--max-burst-secs`). The
detector's threshold was calibrated on Gaussian noise; real band noise
trips it more often, which the header filter then has to absorb. Digital
silence is not searched (no noise to normalize by). The ROCm torch
import prints "(null): No such file or directory" once; harmless.

## Gear shifter, phase A: what the receiver can measure (2026-09-24)

Plan: /home/andrew/.claude/plans/sleepy-jumping-whisper.md (ARQ with a
model-based shifter, IR-HARQ, VARA-style host API).

`scripts/rx_audit.py` (runs/rx_audit.csv, 4637 bursts through the full
receiver: 4 bands x awgn/mpg/mpp/mpd x -5..20 dB):

- Effective mutual information (mean AWGN BICM capacity at the
  equalizer's own |h|^2 / var, the MIESM feature) against the genie BMI
  of the burst's actual LLRs: median error within +-0.01 almost
  everywhere, p10-p90 within +-0.03-0.1. Exceptions: n4 on mpd reads
  0.1 low at every SNR, n10 on mpp 0.04 low above 5 dB. Consistent per
  band, so learnable.
- Doppler spread (2 sigma): 0.8-1.0 Hz for mpp's 1, 1.75-2.0 for mpd's
  2, on every band. Unreliable only at -5 dB AWGN (0.2-0.5 Hz of nothing).
- Delay spread: exact on w and w48 from 5 dB (2.0 / 4.0 ms); mpg's
  0.5 ms is below resolution (reads 0); garbage below ~0 dB. Narrow bands
  can not resolve it (n4 always 0; n10 sees mpd's 4 ms, not mpp's 2).
- SNR: w/w48 read +1.3 dB high (pilots vs clipped data), steady. Narrow
  bands under fading read far low at high SNR (n4 mpd 20 dB: -21 dB)
  because the equalizer's frequency projection loses a second path it
  can not resolve. Padding the projection fixed the estimate and made
  decoding worse everywhere (see equalizer._freq_smooth; reverted), so
  the predictor gets the raw features and learns the bias.

`data2g/survey.py`: passive noise floor and occupancy per 50 Hz bin
across 300-2700 Hz from any received audio (a low quantile over 20 ms
segments, so bursts count as traffic, not noise). Test: a carrier
inside w48's extra spectrum shows ~9 dB above the floor; w and n10
read within 2 dB.

`scripts/decode_latency.py` (runs/decode_latency.csv), 64-codeword
bursts: LDPC batched on CPU 2-30 ms when codewords decode (BP stops
early), 70-410 ms when all fail (40 iterations); CUDA <= 40 ms. Polar
SCL one codeword at a time 1.6-3.3 s, batched 41-85 ms, so
`modem.demodulate` now decodes a burst in one batch
(codes.decode_many). Full receive of 1.6-10.7 s of audio: 0.3-1.0 s,
mostly acquisition over the whole buffer, which a live receiver does
while the burst streams in.

## Gear shifter, phases B-E so far (2026-09-24)

**Protocol** (docs/arq.md, data2g/arq/): link core (frames.py, link.py),
sessions (session.py), shifter (policy.py), link predictor (predictor.py).
Every accounting rule came from a bug the fuzz and stress runs found:

- stale-ACK re-slicing that corrupted delivery
- an abandon lost with its only burst
- 3-bit burst-seq aliasing under sustained reply loss
- a float-rounding livelock in the event clock
- the idle timer counting a sending-only callee as idle

scripts/arq_stress.py: 3200 lockstep runs and 1600 timed sessions over a 4x4
burst/codeword loss grid, with no corruption, hang, stuck run, slot mismatch or
collision.

**Link abstraction** (codes_data/link_abstraction.json,
scripts/link_abstraction.py): per submode, P(codeword decodes) against effective
MI (the receiver's own MIESM value). MI50 sits just above each code's rate, and
agrees across channels within ~0.02 MI for most modes.

**Predictor** (codes_data/link_predictor.npz, capacity_tables.npz,
mi_offsets.json, sync_floors.json; scripts/predictor_data.py,
train_predictor.py): a 2x32 tanh MLP in plain numpy.

- Input: what the receiver measured on a burst of 1-16 frames.
- Output: the next burst's channel MI per (band, constellation), as a Gaussian,
  with a persistence skip.
- Per candidate mode, P(decode) = channel MI, plus its own clip noise, less
  the receiver's estimation loss, through its curve.
- Held out: Brier 0.036 against persistence's 0.072 on same-band pairs.
- Two findings shaped it. Clip noise had to come out of the MI (it belongs to
  the submode sent, and capped high-order constellations at ~0.6 MI). And
  short measured bursts need their length as an input.

**Simulator** (scripts/linksim.py): the real session code on a link-abstraction
PHY, with continuous Watterson fading.

- Calibrated to the measured ladder thresholds (MI estimation loss per band
  and Doppler) and sync floors (fading penalty per band): within ~1 dB / ~2 dB.
- `linksim.py sweep` compares the shifter with the best fixed mode in
  hindsight (the oracle), a VARA-style SNR rule and an ARDOP-style counter:
  bytes/s over 300 s, 3 seeds.
- Shifter vs oracle:
  - AWGN within +-5%.
  - mpp and mpd from 4-20 dB (0-20 dB on mpd), 7% below to 55% above.
  - mpg and 24 dB fading still 10-45% below (mpg: slow flat fading, under
    study).
- It beats both baselines almost everywhere; 2-10x the SNR rule on fading at
  12-20 dB.

**Update (2026-09-24, later): sim realism, history, CHAT, a window bug**

- A protocol bug: `WINDOW` was 64, equal to half the 7-bit seq space. A
  cumulative ACK covering a full window aliased to base - 64, and the link
  failed. It is now 63. The bitmap stays 64 bits. Regression test:
  tests/test_arq.py `test_full_window_bursts`. arq_stress now mixes in
  64-codeword bursts: 3200 + 1600 runs, 0 bugs.
- The sim now draws the receiver's measurement error from the real receiver's
  (runs/predictor_data_v5.csv `truth1_*`). Real MI reads are 0.05-0.24 low,
  with std 0.05-0.3. The sim used to add zero bias and 0.02 noise, which made
  the shifter look better calibrated than it is.
- Sim fading sync penalties were refit to the 1% floors: within 0.5 dB of
  runs/sync_floor.csv. mpd was 1.5-2.5 dB too pessimistic.
- Predictor v5 changes:
  - It sees the peer's previous burst too, and the predicted burst's length.
  - Held-out same-band Brier is 0.037.
  - One burst can't tell steady from 0.1 Hz fading. Held out, that
    over-predicts mpg by 0.1-0.2 and under-predicts AWGN as much.
- Shifter changes:
  - An online logit correction per (band, constellation family), learned from
    each burst's decoded fraction.
  - A switch is charged for the codewords it abandons.
- CHAT ON/OFF (VARA) is implemented. Mean latency from 4 dB up: 4-8 s, level
  with or below the oracle's 6-8 s. At -4 to 0 dB it is 1.3-3x worse.
- Bulk vs the oracle in runs/linksim_sweep2.log (the oracle also rose with the
  window fix):
  - 20 dB: 77-100%.
  - 8-16 dB: 65-88%.
  - 4 dB: 63-77%.
  - 0 dB and below: 30-60%.
- What remains is prediction noise on steady channels: P(r1/3) swings
  0.5-0.8 while every burst decodes. Next: longer history, or an MI-space
  correction instead of the logit one.

## Gear shifter, phase F: incremental redundancy (2026-09-24)

- **Codec:**
  - `QCLDPC.mother()` is the same base graph with every extension row (rate
    ~1/5). Its first n bits are the frozen codeword, so RV0 is unchanged on air.
  - `codes.rv_positions`: RV r sends buffer positions [r n, (r+1) n), wrapping.
  - `codes.combine` adds soft bits into the buffer, and `codes.decode_buffer`
    decodes the mother code cut to the extent received. Polar is Chase only.
  - `modem.modulate(..., rvs)` sends RVs, and `Burst.soft` carries the soft bits
    for combining.
  - Tests: tests/test_ldpc.py (mother prefix, IR decodes where RV0 alone
    cannot), tests/test_modem.py (RV1 through the modem).
- **PHY study** (scripts/ir_study.py, runs/ir_study.csv): 8-codeword bursts,
  AWGN and mpp, 12 trials per point.
  - Two transmissions with IR move the 50% point 2-6 dB below one
    transmission's. The gain is larger at higher code rates (w48-16qam-r2/3 and
    w48-64l-r1/2: ~6 dB).
  - Chase gets 1-3.5 dB less than IR at rate 1/2 and above. At rate 1/5, where
    RV1 is mostly repeats, they are the same.
- **Simulator:**
  - The old model, where MI adds, was up to ~1.5 dB optimistic and ignored
    repeats.
  - `linksim.combined_mi`: the transmissions' SNR is spread over the buffer
    extent covered (Chase at one codeword length, MI-sum for full IR), less 6%.
    It fits 16/64-QAM within ~0.05 in P.
- **Effect on phase E:** none measurable (runs/linksim_sweep2.log, identical to
  runs/linksim_sweep2_misum.log within noise).
  - 78-90% of lost codewords are in bursts whose control codeword failed too.
    Every codeword in a burst sees the same channel.
  - Without the control codeword, the receiver can't map slots to seqs, so it
    stores nothing. IR only helps the rest.

## Gear shifter, phase G: sessions through the real modem (2026-09-24)

scripts/phy_session.py runs linksim's sessions and event loop with the real
modem, not the link abstraction.
- Every burst is modulated and passed through one continuous Watterson process
  per session, with noise and a 4.5 Hz offset. The real receiver then receives
  it.
- `data2g/arq/phy.py` is the adapter phase H reuses: `tx_audio`, `ModemRx`
  (masked CRCs and soft-bit combining), `measure` (the shifter's inputs;
  predictor_data.py uses it too), and `mask_value`.
- linksim.run takes the PHY as a parameter. Delivered bytes are checked against
  what was written, so this is an end-to-end correctness test too.

Two bugs the simulator could not show:
- **No scrambler.** A zero-padded payload (every control codeword, a stream's
  last codeword, a padded KISS frame) coded to ~10% ones. It piled OFDM symbols
  onto a few points, and the clipper wrecked them. 3-codeword bursts lost their
  control at 8 dB AWGN every time, and winlink sessions took 1.6x the
  simulator's time.
  - Fix: `codes.scrambler`, PN9 on the info bits.
  - The seed is per codeword (`codes.scramble_seed`): from the CRC mask in ARQ,
    so resends combine; from the burst position otherwise. Identical payloads
    scrambled alike still failed half the time at 8 dB.
  - This changes on-air bits: the provisional freeze moves, and both ends of the
    TNC fork need the update. Thresholds are unaffected (measured on random
    payloads). Test: tests/test_modem.py
    `test_zero_payloads_decode_like_random_ones`.
- **Stale soft bits after an abandon.** Re-slicing reuses seqs, but the receiver
  forgot soft bits only for codewords it had decoded, not the failed ones. Those
  were combined into the new codewords. On the PHY the CRC caught it (a crash
  on the buffer size, or a needless failure). In the simulator it silently added
  MI and made it optimistic.
  - Fix: forget the whole window on an abandon.
  - The fuzz receiver (tests/test_arq.py FakeRx) now flags combining across
    different codewords. Stress: 3200 + 1600 runs, 0 bugs.

PHY vs simulator (runs/phy_session.csv, 3 seeds each), after both fixes:
- Within ~15% on bulk (AWGN 0 and 12 dB, mpg 4, mpp 8, mpd 4), winlink (AWGN 8,
  mpp 12), chat (mpp 8) and the fixed-mode check.
- Exception: bulk mpd 16 dB, PHY 182 vs sim 300 B/s. The shifter picks 64-QAM
  and 16-QAM r2/3, whose control codeword fails in about half their bursts. The
  simulator's per-mode calibration (+-1 dB) is optimistic there. That's shifter
  tuning, deferred until on-air data.
- Receive CPU: 1.6-3.8 s per 11-12 s burst (sync and equalization over the
  whole buffer, one core, machine loaded). Phase H needs streaming receive
  (acquire during the burst) to answer within the 1.3 s turnaround.

## SSTVAE history check: false locks and steady carriers (2026-09-25)

SSTVAE's commits changing pilots and acquisition, checked against Data2G.
- Already covered: the low-crest pilot, first-path timing, and ±625 Hz CFO
  acquisition.
- Drift is not an issue here: 5 Hz across a 37 s burst decodes fully (the 2-D
  LMMSE estimate tracks it).
- SSTVAE's 2026-09-22 receiver batch came from Data2G.

Two things Data2G still had:

- **False locks with a random header (SSTVAE 1b05ba4).**
  - Data audio whose preamble was lost detected as a burst 40/40 times. So did
    a steady tone. The w band had no header score floor.
  - The CRCs rejected the payloads, but a false header's length stretched the
    ARQ deadlines.
  - `HEADER_MIN_SCORE["w"]` is now 0.33. False headers score 0.19-0.34 (733
    measured, runs/w_header_gate.npy), and 4 pass. Cost: 2-3% of correctly read
    w headers at -7.5 to -6.5 dB AWGN, under 1% on fading.
  - Test: tests/test_modem.py `test_no_burst_from_data_without_its_preamble_or_from_a_tone`.
- **A steady carrier inside the band.** Acquisition already survived it: the
  header arbitrates between candidates, which is what SSTVAE's todo proposed.
  Demodulation did not.
  - With one band-wide noise level, a tone 6 dB under the signal (about 8 dB
    over one wide carrier) cost a codeword in most bursts. At 0 dB, w48 and
    n10 decoded nothing.
  - Now `equalizer.per_carrier_noise`:
    - A carrier whose leverage-corrected pilot residual is above the 99th
      percentile of noise alone gets its own noise level, so its bits turn
      unreliable.
    - The rest keep the clean carriers' mean, so a clean channel is unchanged.
    - Narrow bands use the preamble's repeats (only where the preamble's
      carriers are the data's).
  - Two simpler versions cost AWGN. Letting each carrier keep its own higher
    reading made chance-high carriers timid. Averaging the interferer into the
    band level made every clean carrier timid.

  Decoded fraction, AWGN, tone relative to the whole signal (8 seeds x 6
  codewords):

  | mode | SNR | tone -6 dB | -3 dB | 0 dB |
  |---|---|---|---|---|
  | qpsk-r1/2 | 6 | 0.83 -> 1.00 | 0.83 -> 0.98 | 0.06 -> 0.83 |
  | w48-qpsk-r1/2 | 6 | 0.83 -> 0.98 | 0.50 -> 0.94 | 0.00 -> 0.13 |
  | n10-qpsk-r1/2 | 1 | 0.69 -> 1.00 | 0.38 -> 0.98 | 0.21 -> 0.54 |
  | w48-16qam-r1/2 | 12 | 0.75 -> 1.00 | 0.00 -> 0.02 | 0 -> 0 |

  - Clean channels near 50% (7 cells, 480 codewords each): unchanged within
    noise.
  - Not done:
    - The tone still corrupts neighbours' channel estimates through the delay
      projection (a weighted projection would fix it). That is what still sinks
      16-QAM at -3 dB.
    - The shifter's MI features still use the band-wide noise, as the predictor
      was trained on it, so they don't see an interferer.
  - Tests: tests/test_equalizer.py.

## Gear shifter, phase H: the live stack (2026-09-25)

- **`data2g/arq/engine.py`:** one station, clocked by audio samples.
  `Engine.step(block) -> (audio out, PTT)` drives the session, the streaming
  receiver and the gear shifter, half duplex.
  - The same code runs behind a sound card, or back to back with another
    Engine through a simulated channel faster than real time
    (tests/test_engine.py: connect, data both ways, disconnect; VARA commands;
    BW500 keeping every burst on air at 500 Hz or narrower).
- **Streaming receive** (`tnc.Receiver`, shared with the KISS TNC):
  - It looks for a preamble every 0.5 s of new audio, and checks a pending
    burst's end on every block.
  - At the burst's end, `modem.receive(..., head=)` searches only the segment's
    first second for the preamble already found. Acquisition over a whole 11 s
    burst had taken 1-4 s.
  - Measured in a 100 s session: listening costs 2-3% of a core. The worst
    step, at a burst's end (receive, decode, build the reply), took 0.5 s.
- **`data2g/host.py`** (`data2g-host`, or `python -m data2g.host`): VARA-style
  TCP host.
  - Commands on 8300: MYCALL, LISTEN ON/OFF, CONNECT, DISCONNECT, ABORT, BW500,
    BW1200, BW2300, BW2750, CHAT ON/OFF, VERSION.
  - Session data on 8301.
  - Notifications: CONNECTED src dst bw, DISCONNECTED, PTT, BUSY, BUFFER, MODE.
  - Audio and rigctld PTT come from the TNC. It also accepts, logs and ignores
    common VARA settings (COMPRESSION, PUBLIC, CWID...). That list comes from
    memory, so check the log on first contact with Pat.
  - Tests: tests/test_host.py (TCP with a fake sound card).
- **Recording**, on by default (`--record-dir`, '' turns it off):
  - events.jsonl holds every burst sent (each slot's payload, CRC mask and RV)
    and every burst heard (with the shifter's measurements).
  - rx_NNNNN.npz holds each heard burst's audio.
  - audio_in.f16 holds everything the receiver was fed (about 58 MB an hour).
- **`scripts/replay.py A B`** replays both ends of a recorded session.
  - For heard bursts: whether the control codeword decoded, first-transmission
    data decoded, and data decoded in bursts whose control failed (what a more
    robust control would save).
  - For missed bursts: the detector's peak against its threshold at that moment,
    and whether an offline search finds it (sync near-misses).

On-air check, two stations each running rigctld:

    data2g-host --mycall W1AW --input-device USB --output-device USB   # station 1
    data2g-host --mycall K2XYZ --input-device USB --output-device USB  # station 2
    printf 'LISTEN ON\r' | nc -q -1 localhost 8300                     # station 2 (keep open)
    printf 'CONNECT W1AW K2XYZ\r' | nc -q -1 localhost 8300            # station 1
    # then data both ways on port 8301 (nc, or Pat's VARA transport)
    uv run python scripts/replay.py recordings/<station 1 run> recordings/<station 2 run>

Not done yet:
- A loopback of two hosts over real sound devices. It needs a virtual audio
  cable (PipeWire null sinks), which I didn't set up without asking.
- A Pat session.

## Acquisition at +-150 Hz, and the audio loopback (2026-09-25)

**Frequency search +-150 Hz** (`config.ACQUIRE_REACH_HZ`, was +-625):
- Detector CPU 126 ms -> 36 ms per 1.6 s of audio (3 bands).
- Noise peaks over 1200 s: w 21.2, n10 19.5, w48 21.9 (runs/noise_peaks_150hz.log).
  The threshold (1.1x the largest) goes 25.5 -> 24.1.
- Two noise-reference bins at +-625 Hz (`sync.NOISE_REF_HZ`) keep the detector's
  noise level clean. Without them every searched bin overlaps a narrow burst,
  and n10 mpd lost ~0.4% of bursts. They cost 1.8 ms.
- Sync floors (runs/sync_floor.csv, the old ones in runs/sync_floor_625hz.csv):
  - n10, n4 and w48 are unchanged within the study's 0.25 dB steps.
  - w moved 0.5-1.0 dB. That's the w header floor, 0.25 (see below), costing
    0.3 points of failures near the floor. The width is neutral: identical
    failure rates at 4.5 dB mpd.
- Regenerated: codes_data/sync_floors.json (the shifter's sync model), and
  linksim's SYNC_PENALTY_DB (emergent floors within 0.16 dB of measured).

**w header floor 0.33 -> 0.25, with a supersede search:**
- The SSTVAE-history fork's 0.33 stopped false locks on data whose preamble was
  lost, and on tones. It cost the w band (the robust ACKs) 1.75 dB AWGN and up
  to 3 dB of 1% floor on fading.
- At 0.25:
  - Tones: 0.8% false locks.
  - Data without its preamble: 52% false locks.
  - Real headers lost: 0.3%.
- `tnc.Receiver` makes the data false locks cheap:
  - While committed to a burst, it keeps searching the newer audio. A later
    header scoring 0.05 higher replaces the pending one.
  - When a burst completes, its span is searched once more.
  - After a suspect burst (score under 0.36), the last search window isn't
    trimmed away.
- Tests (tests/test_tnc.py):
  - A false lock on data doesn't cost the next real burst.
  - A half-arrived header is waited for, not misread.

**Audio loopback** (two data2g-host instances over PipeWire null sinks, driven
over their TCP ports). It found, all fixed:
- **Clipping.** The host played the modem's unit-RMS audio at peak 2-3 into a
  sound card clipping at 1. 64-QAM bursts died, QPSK didn't. Bursts are now
  scaled to peak 1 (Engine).
- **A livelock.** The peer kept recommending the mangled 64-QAM, polls got
  through, 64-QAM went out again. Forever, 0 bytes delivered. Fix: gear-shifter
  strikes.
  - Receiver side: 2 heard headers in a (band, constellation) family with no
    control decoded.
  - Sender side: a burst and its repeat both unanswered.
  - Either makes the mode or family sit out a hold: 4 decisions, doubling, capped
    at 64. The sender falls back to the robust connect mode.
  - Test: tests/test_engine.py `test_a_mode_the_link_mangles_does_not_stall_the_session`
    stalls without the holds.
- **Misread half-arrived headers.** A search that landed mid-header read shifted
  positions that fit the buffer: garbage scoring ~0.25, which hid the real
  header. In streaming, a header not yet wholly in the buffer now waits.
- **Missed replies.** The receiver's first search after its own transmission
  waited for 1.5 s of audio, then searched only every 0.5 s. The sender's reply
  deadline (burst end + 2.0 s) expired while the reply was on air. Fixes:
  - The first search now comes as soon as one preamble and header fit (~0.6 s),
    then every 0.25 s (~14% of a core listening).
  - `session.REPLY_START_S` 1.0 -> 1.5.
  - Input below -80 dBFS is taken as silence: at 1e-6 noise, a burst's own
    filter ringing read as a header.
- **Result:** 20 KB up and 5 KB down in 32 s, three runs out of three.
  - Replay: every burst heard both ways, every codeword decoded.
  - Session stress: 1600 runs, 0 bugs.

## Pat over Data2G, peer to peer (2026-09-25)

Two Pat v1.0.0 instances (P2P, no Winlink server), each on its own data2g-host,
over the PipeWire loopback. Logs: runs/pat_p2p_small, runs/pat_p2p_large_slow,
runs/pat_p2p_stuck, runs/pat_p2p_large.
- Pat's init commands (PUBLIC, CWID, COMPRESSION, WINLINK SESSION) are
  accepted and ignored. MYCALL, LISTEN, VERSION, BW2300, CONNECT and
  DISCONNECT work as VARA's.
- **Small:** a text message each way in 57 s, connect to disconnect.
- **Large:** 20 KB and 5 KB attachments (random, incompressible), arriving
  byte-identical.
  - The first run took 183 s. Pat's VARA driver blocks a write while BUFFER
    >= 7x its size, and B2F writes <= 250 B blocks. That kept ~1-2 KB queued,
    and every burst went out short (27 five-codeword bursts for the 20 KB).
  - BUFFER now leaves out what the next burst will carry (`--buffer-credit`:
    -1, the default, is the next burst's full capacity; N caps it at N bytes;
    0 is plain VARA, everything queued). The same exchange then took 79 s: the
    20 KB as a 32- and a 40-codeword burst, the rest B2F's handshake turns.
  - The BUFFER change first deadlocked. Pat counts what it wrote until a
    BUFFER line arrives, and the host only sent one on change. The host now
    answers every data write with BUFFER. Tests in tests/test_engine.py.

## CQFRAME (2026-09-25)

- VARA's `CQFRAME call bw` (500 | 1200 | 2300 | 2750) sends a CQ frame: a
  control-only burst with a `T_CQ` extension, the callsign plus the bandwidth
  cap code.
  - It goes in the robust connect mode for that bandwidth: n10-qpsk-r1/3 at
    500, qpsk-r1/5 at 2300.
  - It needs no session and makes none. It is refused while one is under way.
- A station that hears one, listening or idle, prints the same line to its host:
  `CQFRAME W1AW 500`.
- Over the audio loopback it first failed, which exposed a receiver bug. The w
  detector also fires on a narrow preamble, and while the n10 header was still
  arriving, w's shorter header read garbage (0.27) and committed first. Now,
  while any band's header is still arriving, only a header scoring 0.5 or more
  commits (`modem.STREAM_COMMIT_SCORE`).
- Tests: tests/test_engine.py `test_cqframe_is_heard_without_a_session`;
  tests/test_tnc.py `test_a_half_arrived_header_is_waited_for_not_misread`
  (now also n10).

## CHAT ON with a file, and more VARA client commands (2026-09-25)

A VARA client (VarAC-style: `CHAT ON`, `MYCALL KC2G KC2G-T`, `LISTEN CQ`,
`IGNOREKISSDCD ON`) sent a file at BW500 over the loopback. It went out as
n10-16qam-r1/2, 3 codewords per burst, though r2/3 and r3/4 predicted P >= 0.999.
- **Cause:** CHAT's objective planned every burst for a 200 B chat line. All
  three rates fit one in the same 3-codeword burst, and the tie went to r1/2.
- **Fix:** with chat on, data bursts carry the sender's queue (`T_BUFFER`, only
  above 200 B), and the objective plans for delivering all of it. A file goes in
  full bursts at the best rate, a chat line stays short. Test:
  tests/test_engine.py `test_chat_on_sends_a_file_in_full_fast_bursts_and_a_line_short`
  reproduces the r1/2 x 3 bursts without the queue report.
- **Commands:**
  - MYCALL with several calls: all answer connects, and CONNECTED names the one
    dialed.
  - `LISTEN CQ` is accepted (CQ frames are always reported).
  - `IGNOREKISSDCD` is accepted and ignored.

## Host: the command client owns the session (2026-09-25)

- If the command port's TCP client goes away without disconnecting (a crash, a
  dropped socket), the host sends DISCONNECT on any session under way. The peer
  is told, and queued data still goes. A call still being placed is dropped.
- Listening stops too, so no connect is accepted that nobody will serve. Clients
  send LISTEN ON again when they reconnect.
- A new client replacing the old one doesn't count as a loss. A dropped data
  connection alone does nothing.
- VARA's own behavior here is undocumented as far as I found. Pat's driver
  always sends DISCONNECT before closing and doesn't rely on the modem.
- Loopback: the peer saw DISCONNECTED 4 s after the client's socket closed.
  Also new: IAMALIVE on the command port every 60 s.

## Low-SNR robustness: loss accounting, w48 header, outcome predictor (2026-09-25)

**Where bursts die** (`scripts/loss_study.py`: real-modem ARQ sessions, every
burst classified at its receiver):
- At -4 dB, 24-40% of data bursts lost their control codeword. The modes chosen
  decoded 47-63% of codewords.
- Preambles were missed in 7-16% of fading bursts. Header misreads were ~0.

**w48 header** (`scripts/w48_sync_diag.py`):
- The cause of w48's 3 dB worse sync floor was its header: 192 bits in 2 symbols,
  half of w's energy.
- It is now 4 symbols, with a (16, 384) code (d_min 155). Its score floor is
  0.26: noise reads top out at 0.248 and reads off w bursts at 0.243.
- Sync floors: AWGN -3.75 -> -5.25 dB, MPG 5.25 -> 3.25, MPD 7.75 -> 6.75.
  w's are unchanged.
- The format change adds 48 ms per w48 burst.

**Outcome predictor** (`scripts/outcome_data.py`, `scripts/train_outcome.py`,
`predictor.predict_outcome`):
- **Why:** the MI predictor's P went through the link abstraction and the
  clip-noise and estimation-loss models. It was mushy at low SNR (0.35-0.9 where
  real decoding was 0% or 100%). Output patches (a P floor, a window cap) didn't
  raise throughput.
- **Dataset:** 19.6k samples of real decodes, with a realistic mix:
  - fading-weighted channels, SNR -8..22 dB with drift;
  - measured bursts as the shifter sends them;
  - candidates near their decision boundary;
  - ~1% out-of-distribution cases.
- **Model:** a 2x64 numpy MLP giving per submode P(burst usable) and P(codeword
  | usable). Sync is inside the outcome. Held-out reliability is within ~0.04
  per bin, and within ~0.05 by channel and SNR.
- **Real-modem loss study:**
  - AWGN -4 dB: 141 -> 196 bps; control losses 24% -> 3%.
  - MPG 0 dB: 153 -> 224 bps.
  - MPG and MPP -4 dB: unchanged (49 and 5 bps).
- **Phase G scenarios:** AWGN 0 dB +46%, AWGN 12 dB +12%, MPP 8 dB +28%, chat
  latency better. MPD 16 dB -5%, winlink MPP 12 dB 18% slower.
- **Removed:** the P floor and window cap. The MI predictor stays only as a
  fallback.
- **Next, for fading at the bottom:**
  - At MPP -4 dB, codewords in a burst fail about independently (~40% each). So
    the single control codeword is lost ~60% while data survives: 172 decoded
    data codewords in control-lost bursts. A more robust control codeword pays
    here, unlike on MPG.
  - Missed preambles (8-14%) remain.

## Duplicated control codeword (2026-09-25)

- **What it is:** an `ARQ_DUP` burst sends each control codeword twice (RV 0,
  then RV 1). The receiver combines the pair if the first copy fails.
- **When:** the receiver asks for it (`T_DUPCTL`) when its outcome model gives
  P(burst usable) < 0.9. It costs one codeword per data burst, and only then.
- **Why:** at MPP -4 dB, codewords in a burst fail about independently, so a
  single control codeword failed in 73% of data bursts while their data decoded.
- **Real-modem loss study** (`runs/loss_study.log`):

  | Cell | Before | After |
  |---|---|---|
  | MPG -4 dB | 49 bps, control lost 39% | 73 bps, 17% |
  | MPP -4 dB | 5 bps, 70% | 14 bps, 33% |
  | MPG 0 dB | 224 bps, 24% | 230 bps, 9% |
  | MPP 0 dB | 162 bps, 23% | 209 bps, 4% |
  | AWGN -4 dB | 196 bps | 180 bps (-8%) |

  On AWGN at -4 dB the model's under-confidence turned it on for 45% of data
  bursts that didn't need it.
- **Phase G:** the high end is unchanged (AWGN 12 dB 411 vs 415 B/s, MPP 8 dB
  the same). Fading improved: MPG 4 dB +20%, MPD 4 dB +11%, MPD 16 dB +5%,
  winlink MPP 12 dB 120 -> 114 s.
- **Accounting:**
  - The fuzz policy asks for it at random. Stress: 3200 + 1600 runs, 0 bugs.
  - The fuzz test's heaviest loss cell now allows a bounded "link lost". About
    20% of its runs lose the link with or without duplication (300 seeds: 59
    vs 48).
- **Tests:** tests/test_arq_phy.py `test_duplicated_control_pair_combines`.

## Reply modes and outcome data v2 (2026-09-25)

- **Reply modes at -4 dB on the real modem**, fraction usable (150 bursts
  each):

  | Reply mode | MPP | MPG |
  |---|---|---|
  | ack-4f | 79% | 82% |
  | n10-ack-4f | 95% | 86% |
  | n4-ack-8f | 96% | 89% |
  | ack-1f | 39% | 46% |

  The shifter mostly replied in ack-4f.
- **Outcome data v2:** 15k more samples, each with one reply mode among its
  candidates, 34.6k samples in all.
  - The model still ranks n10-ack-4f only a hair above ack-4f (0.87 vs 0.84; the
    real rates are 0.95 vs 0.79), so the choice barely moved.
  - Held-out calibration is within ~0.05 per bin, slightly under-confident in
    mid-range.
- **Fallback n10-ack-4f** (polls and escalated replies): tried and reverted.
  Every loss-study cell went down. A likely cause is that the receiver
  recommends wide modes from measurements of 500 Hz polls. That's unproven: see
  the noise below.
- **Loss-study noise:** 4 seeds x 300 s gives a standard error of the cell mean
  of 8% (AWGN -4 dB), 10-26% (fading at 0 and -4 dB), and 67% (MPP -4 dB, two of
  four seeds delivered nothing). Throughput differences under ~20-30% between
  runs are not evidence. The per-burst rates, over hundreds of bursts, are much
  firmer. Deciding between close variants needs more seeds and longer sessions.

## Constant-envelope (CPM) modes in the ARQ (2026-09-25)

TLDR: CPM modes are in the modem, the link and the shifter. At MPP -4 dB
they lift a session from 29 to 46 bps, and cost nothing elsewhere. A connect
fix found on the way took that cell from 8 to 29 bps first.

- **The family:** noncoherent M-FSK (`data2g/cpm.py`), three grids, each with
  r1/3 and r1/2 LDPC data codewords (k 320/480, n 960):

  | Mode | Width | bps | 10% point, avg power: awgn / mpg / mpp / mpd | 1% point |
  |---|---|---|---|---|
  | fsk16r25-r1/3 | 460 Hz | 32 | -13.8 / -8.0 / -11.6 / -11.6 | -13.4 / -5.0 / -10.2 / -10.8 |
  | fsk16r25-r1/2 | 460 Hz | 48 | -12.5 / -6.3 / -9.3 / -9.3 | -12.3 / -2.8 / -8.0 / -8.2 |
  | fsk8r50-r1/3 | 460 Hz | 48 | -11.6 / -5.6 / -8.9 / -8.9 | -11.2 / -0.7 / -6.9 / -7.2 |
  | fsk8r50-r1/2 | 460 Hz | 72 | -10.1 / -2.6 / -6.1 / -6.3 | -9.7 / 1.9 / -5.4 / -5.2 |
  | fsk32r62-r1/3 | 2060 Hz | 99 | -9.1 / -3.9 / -5.9 / -5.9 | -8.8 / 1.8 / -4.8 / -5.2 |
  | fsk32r62-r1/2 | 2060 Hz | 151 | -8.0 / -1.6 / -3.7 / -3.3 | -7.6 / 2.9 / -1.6 / -2.2 |

  From the CPM prototype's study. Codeword thresholds, end to end with
  spread sync. The c8r50 mid-block change below postdates it; the front
  pattern and thresholds are unchanged.
- **Wire format:**
  - Control rides a short polar codeword: 20 B, one per burst (twice
    duplicated).
  - When control doesn't fit, the sender sheds resends, then the optional
    extensions, then the bitmap, then new data (docs/arq.md §3).
  - A recommendation's band code 3 means CPM.
  - CPM size classes are 4x the OFDM ones (4-48 s). A CPM data codeword
    takes 3-10 s.
- **Receiver:**
  - The streaming receiver also listens on the three CPM grids. At idle that
    costs 8.7% of a core more (20.5% total, after skipping the fine search
    below threshold).
  - `tnc.receive_any` is the offline equivalent for the real-modem session
    scripts (`--policy shift+cpm`).
- **Sync fixes found by the outcome data:**
  - c8r50's mid-burst sync blocks repeated its front Costas array. A mid
    block scored 0.50 on the front detector (0.10-0.12 on the other grids).
    It is now time-reversed (0.17).
  - On a clean signal, a burst's own data met the front pattern by chance.
    With the front cut off, 16-23 of 40 bursts per grid locked, all with
    wrong headers. A lock now needs pattern share over peak share >= 0.7:
    fronts measure 0.82+ from -10 to 25 dB, false locks 0.59 at most. Noise
    alone gave 0 locks in 398 windows.
  - A blind duplicated-control probe on a CPM data slot crashed a session
    (a 960-bit codeword combined into a 360-bit buffer). `ModemRx` now
    refuses control decodes outside the header's control slots.
- **Outcome model v3:**
  - 20k more samples, with CPM as measured bursts and candidates (oversampled
    30%). The model stores its mode and band lists.
  - CPM held-out calibration is within ~0.1 in most cells. It is optimistic
    for fsk8r50 on MPG below -5 dB: 0.73 predicted, 0.57 real.
- **Connect escalation:**
  - At MPP -4 dB, 9 of 12 sessions never connected: every try went in
    qpsk-r1/5, 38% usable there.
  - Retries now go in n4-qpsk-r1/3 (91% usable), and the callee answers in
    the mode it heard. DISC retries escalate too.
  - The "two of four seeds delivered nothing" in the v2 loss study was this.
- **Loss study,** 12 seeds x 600 s, bps (mean ± s.e.):

  | Cell | OFDM only | + CPM |
  |---|---|---|
  | MPG -4 dB | 57 ± 7 | 52 ± 4 |
  | MPG 0 dB | 200 ± 8 | 203 ± 8 |
  | MPG +8 dB | 894 ± 27 | 894 ± 27 |
  | MPP -4 dB | 29 ± 4 | 46 ± 4 |
  | MPP 0 dB | 186 ± 8 | 186 ± 8 |

- **Tried and dropped:**
  - A P(usable) >= 0.3 floor on data candidates: within noise everywhere.
    It was meant to stop the argmax picking overestimated long shots (64l
    at 5-18% usable), but the online bias already limits those to a few
    bursts.
  - Not recommending 14 near-clone modes (an envelope analysis said they
    cost <= 0.2% anywhere): within noise too. That analysis also proposed
    dropping n4-qpsk-r1/3, which is now what makes connects work.
- **Open:**
  - MPP -4 dB is still ~46 bps against VARA's reported ~100.
  - Most time there goes to control and reply bursts (ack-4f 24% missed).
    CPM isn't used for replies: a control-only CPM burst is 2.4-5 s.

## Strikes removed, control-slot accounting, search CPU (2026-09-25)

TLDR: removing strikes gained 12-48% at -4 to 0 dB fading. The live
receiver's idle CPU is down from 20.5% to 11.1% of a core.

- **Strikes removed.** A mode that went unanswered, or a family whose
  control failed, was held out of use for a while.
  - The only evidence for them was the audio loopback's clipped 64-QAM.
    That test passes without them: the online bias routes around it.
  - At -4 to 0 dB fading they held working modes, and the fallback carried
    no data. One MPP -4 dB session sent 32 consecutive control-only turns.
  - Loss study, shift+cpm, 12 seeds x 600 s, bps:

    | Cell | Strikes on | Strikes off |
    |---|---|---|
    | MPG -4 dB | 48 ± 4 | 67 ± 4 |
    | MPG 0 dB | 195 ± 9 | 219 ± 8 |
    | MPG +8 dB | 894 ± 27 | 865 ± 34 |
    | MPP -4 dB | 45 ± 4 | 56 ± 3 |
    | MPP 0 dB | 178 ± 9 | 226 ± 6 |

- **Control slots per mode.**
  - The shifter's objective assumed one control codeword per burst.
  - In a 4-byte reply mode, control takes 3 codewords. So ack-4f was
    recommended for data with no room for any.
  - `policy.ctl_slots` now counts them (control estimated at 12 B).
- **CPM burst on a weak OFDM header.** OFDM's search runs first, and read
  strong CPM audio as a header (0.25-0.29) in 5 of 40 bursts at MPG +10 dB.
  Below the suspect score (0.36), a CPM lock now wins.
- **Search CPU, same results to 1e-14:**
  - `sync.detection_stat`: one FFT of the buffer, then one inverse FFT per
    CFO bin. With the FFT length a multiple of 640, each bin is a whole-bin
    roll of the template's spectrum.
  - The repeat products are one lagged product, and the preamble template
    is cached per band.
  - `cpm.detect` mixes once per CFO fraction, not once per fraction and
    timing phase.
  - Receiver idle: OFDM 11.8% -> 6.2% of a core, with CPM 20.5% -> 11.1%.
    The test suite runs in 47 s, down from 62.
  - Tried and dropped: a strided noise quantile. It moved the noise level
    by +-4%, and the minimum over bins then biases toward false alarms.
- **CPM late detection: no headroom.** Spread sync over the whole burst
  locked no more bursts than the early lock on MPG and MPP (fsk8r50,
  fsk16r25, fsk32r62; -4 to +2 dB). MPG's 0.1 Hz fades outlast a burst:
  what the early lock misses, the rest of the burst misses too.

- **c8r50 locked a period late.** Its front is a 6-symbol Costas array
  tiled 4x, so a lock one tile late still matched. It happened in 3-8% of
  fsk8r50-r1/3 bursts at MPP -3 dB, where the header's first symbols stood
  in for the last tile. `cpm.find` now reads the header at each whole-period
  alignment and keeps the best. fsk8r50-r1/3's MPP 10% point went from -1.3
  to -4.9 dB.
- **Ladder study** (`scripts/ladder_study.py`, page `scripts/ladder_page.py`):
  - All 48 modes' 10% points, measured on the smallest ARQ data burst
    (control plus one data codeword) through the ARQ's receiver. Results in
    `runs/ladder_10pct.csv`.
  - On MPP, the 1200 Hz modes are ~4 dB behind 500 Hz ones with the same
    codes: ack-4f +1.4 vs n10-ack-4f -2.2 dB. Their 96 ms header against
    n10's 240 ms.
  - On each CPM grid, r1/3 and r1/2 reach about the same point (fsk32r62
    AWGN -7.9 vs -7.8 dB). The 20 B control codeword (polar, rate ~1/2)
    limits CPM bursts, not the data code.

## Second header copy on the 1200 and 2400 Hz bands (2026-09-25)

TLDR: +16-18% on MPP, no measurable cost at the top. Protocol version 11.

- **Why:** on MPP, nearly all missed bursts decoded the preamble fine and
  failed at the 4-symbol (96 ms) header. Replies failed the same way (8 of
  9 ack-4f failures). The 500 Hz header is 10 symbols and was fine.
- **Study** (`scripts/header_diversity.py`, 2000 bursts a cell): a second
  copy spaced 1-4 frames later cut 2400 Hz header losses 3-4x on MPP and
  MPD, and wrong headers accepted on 1200 Hz 2-4x. An 8-symbol contiguous
  header did clearly worse; MPG barely changed (slow fades).
- **Design:**
  - The copy is a frame of its own: pilot, the 4 header symbols, the first
    again. It sits after data frame 2, or after the last on a shorter burst.
    The pilot grid stays regular for the equalizer.
  - The receiver adds the copy's LLRs to the first copy's. It keeps a
    decode only if the burst it describes carries the copy where it was
    read.
  - A streaming receiver commits on the first copy at a score of 0.45 or
    more, and otherwise waits for the second.
  - The coarse CFO estimate uses the copy's known symbols too. Without
    that, a burst whose first copy was lost read its header but lost its
    data.
  - Costs 144 ms per burst.
- **10% points** (`runs/ladder_10pct.csv`; before in
  `runs/ladder_10pct_before_hdrcopy.csv`). MPP and MPD improved 2.5-5 dB for
  the robust modes, e.g. ack-4f MPP +1.4 -> -2.8 and MPD +2.7 -> -1.5;
  qpsk-r1/5 MPD +5.1 -> +0.4. High-rate modes moved within +-0.5 dB.
- **Loss study,** shift+cpm, 12 seeds x 600 s, bps:

  | Cell | Before | With copy |
  |---|---|---|
  | MPG -4 dB | 67 ± 4 | 77 ± 5 |
  | MPG 0 dB | 219 ± 8 | 227 ± 10 |
  | MPG +8 dB | 865 ± 34 | 919 ± 42 |
  | MPP -4 dB | 56 ± 3 | 65 ± 3 |
  | MPP 0 dB | 226 ± 6 | 266 ± 8 |

  Missed data bursts on MPP: 16.3% -> 7.6% (0 dB), 19.1% -> 11.5% (-4 dB).
- **Pending:** outcome model v3 predates the copy, so its P(usable) for
  robust w/w48 modes is now pessimistic. Refresh the outcome data and
  retrain.

## Outcome model v4: better calibrated offline, worse in sessions (2026-09-25)

TLDR: retraining on post-header-copy data lowered session throughput by up
to 57%. v3 stays installed.

- **Result:** loss study, shift+cpm, 12 seeds x 600 s, bps:

  | Cell | v3 (installed) | v4 (30k new) | v4b (84k, stale labels masked) | v4b + evidence-weighted bias |
  |---|---|---|---|---|
  | MPG -4 dB | 77 ± 5 | 67 ± 4 | 65 ± 5 | 53 ± 4 |
  | MPG 0 dB | 227 ± 10 | 207 ± 7 | 203 ± 13 | 168 ± 10 |
  | MPG +8 dB | 919 ± 42 | 870 ± 27 | 864 ± 37 | 924 ± 29 |
  | MPP -4 dB | 65 ± 3 | 60 ± 4 | 28 ± 3 | 29 ± 4 |
  | MPP 0 dB | 266 ± 8 | 212 ± 8 | 191 ± 9 | 176 ± 11 |

- **What goes wrong:** v4 moves the data mode up.
  - At MPP 0 dB it sends w48-qpsk-r1/3, where 3-5% of first-transmission
    codewords decode, and w48-64l-r7/12. v3 sent w48-qpsk-r1/5 (76%).
  - A session trace: nearly every measurement is of the peer's short
    replies, reading -3.3 to +4.4 dB at a true 0 dB. At +2 to +4 dB, v4b
    predicted P(codeword | usable) ~0.99 for w48-qpsk-r1/3; its bursts then
    decoded 1 of 18.
- **Why the offline data doesn't show it:** the same conditioning offline
  (w48-qpsk-r1/3, MPP, measured +1..+4 dB on a 1200 Hz reply) decoded 90%
  of codewords in usable bursts, at a median true SNR of 1.4 dB.
  - The model is right about the average burst behind such a reading under
    the training prior: SNR uniform -8..22 dB.
  - A session held at 0 dB is that prior's pessimistic tail, and this mode
    is on its steep slope there: 5-14% at 0 dB against ~30% at 1.4 dB.
  - v3 did better because its pre-copy data made w48 look risky. It was
    right for the wrong reason.
- **The online bias didn't fix it.** Updating the codeword bias per
  codeword (the binomial gradient, rate 0.25, clamp 8) instead of per burst
  (step 1, clamp 3) made the low cells worse.
- **Files:** `runs/outcome_data_v4.csv`, `runs/outcome_predictor_v4*.npz`,
  `runs/cpm_eval_shift_v4*.csv`. `scripts/train_outcome.py` now takes
  several CSVs and `--stale-header` (masks pre-copy w/w48 burst labels).

## Outcome model v5: trained on what sessions see, as an ensemble (2026-09-25)

TLDR: +24-27% on MPG at 0 and +8 dB, +9% at -4 dB (MPG and MPP), within
noise at MPP 0 dB. Installed: `outcome_predictor.npz` = v5e.

- **Session data** (`scripts/session_data.py`): 2903 real-modem ARQ
  sessions of 300 s, 134,803 rows (`runs/session_data.csv`).
  - Each session draws a channel kind (random Doppler/delay included), SNR
    uniform -8..22 dB drifting slowly (~1% far outside), and a 500 Hz cap a
    quarter of the time.
  - A row is a burst that followed a measurement: the receiver's inputs at
    its last recommendation, and what the burst did. These are the
    conditions the shifter really sees: mostly short replies, a SNR that
    holds still, true gaps.
  - 20% of data and reply recommendations are a random allowed mode and
    size, so it isn't only the shifter's own picks. Control is never
    duplicated, so burst labels mean what the model predicts.
- **Training mix:** sessions plus every offline set (v2, v3 with their
  pre-copy w/w48 burst labels masked, v4). The offline sets' random
  candidates and far-SNR cases keep coverage wide.
- **Ensemble:** 5 members, each on a bootstrap of the training samples,
  probabilities averaged (`train_outcome.py --seed`, `--ensemble`;
  `predictor.OutcomeEnsemble`). 37 us per prediction against 4 us for one.
- **Out-of-range probes** (`scripts/ood_probe.py`): given contradictory
  measurements (20 dB SNR, MI as at -5 dB), the top-order w48 modes get
  0.36 (v3: 0.88). A 1-frame measurement at 30 dB still gets 0.97-0.99 for
  them.
- **Loss study,** shift+cpm, 12 seeds x 600 s, bps:

  | Cell | v3 | v5a (sessions + v4) | v5b (sessions + all) | v5e (v5b x 5, installed) |
  |---|---|---|---|---|
  | MPG -4 dB | 77 ± 5 | 77 ± 5 | 70 ± 4 | 84 ± 6 |
  | MPG 0 dB | 227 ± 10 | 278 ± 15 | 286 ± 12 | 282 ± 14 |
  | MPG +8 dB | 919 ± 42 | 1102 ± 32 | 1137 ± 21 | 1164 ± 28 |
  | MPP -4 dB | 65 ± 3 | 70 ± 3 | 69 ± 3 | 71 ± 2 |
  | MPP 0 dB | 266 ± 8 | 227 ± 9 | 249 ± 10 | 252 ± 7 |

- **Left:** at MPP 0 dB v5b still sent w48-qpsk-r1/3 40 times at 2% of
  codewords decoded.
- **Also:** the shifter predicts every candidate with `gap_s` = 2.5 s. A
  reply follows the recommender's own data burst, so its true gap is that
  burst's length plus turnarounds (the session data holds true gaps). A
  small mismatch for reply modes.

## BUSY follows the signal, not a false header's claim (2026-09-25)

- **Problem (on air):** a false OFDM lock (a weak header read off noise, QRM
  or a burst whose preamble was missed) reported BUSY for the whole length
  it claimed, up to 12 s. A VARA client doesn't transmit under BUSY.
- **Now:** the host reports `Receiver.channel_busy`: a burst is pending and
  either its frame pilots say it is there, or the channel is on air.
  - **Pilots:** `modem.pilot_coherence`, over the newest 4 pairs of
    consecutive frame pilots (~0.6 s), against each band's noise level
    (`PILOT_NOISE`: 99th percentile on noise, scaled by 1/sqrt(carriers)).
    Real bursts stay over it: 0-2% dip under at 0 dB, 10-16% at -4 dB.
  - **On air:** in-band power over the noise floor by 3 dB. The floor is the
    5th percentile of 0.1 s block powers over two minutes. This keeps BUSY
    through a real burst whose preamble was missed: its frame grid isn't the
    false lock's, so its pilots don't show.
  - A suspect header (score < 0.36) raises BUSY only once its pilots confirm
    it. A clear one raises it at once, as before.
- **Measured** (20 ms feeds, like an audio callback):
  - After a missed-preamble burst ends, BUSY drops within 0.0-0.7 s. The
    internal hold ran up to 5 s past the audio, on false claims.
  - 6 min of noise: no false lock, no BUSY.
  - Real bursts at 0 and +10 dB: BUSY held throughout.
- **Unchanged:** the engine's own hold (`Receiver.busy`, a reply waits for
  a pending burst) still trusts the claim, so its own replies don't key over
  a weak real burst. BUSY is still reported only while a Data2G lock is
  pending; it is not a general carrier detect.

## KISS TNC with mode shifting (2026-09-26)

KISS is a personality of `data2g-host`, next to VARA; the single-mode
`data2g-tnc` is gone. Both personalities share one engine, receiver and
PTT:
- **Switches:** `--vara/--no-vara`, `--kiss/--no-kiss`, both on by default.
- **KISS options:** `--kiss-port` (8100, as VARA HF), `--kiss-bw 2400|500`,
  `--broadcast-mode`, `--list-modes`.
- **Receiving:** each burst heard goes to the KISS link first, and it takes
  only bursts with its own CRC masks. The rest go to the ARQ session.
- **Sending:** KISS sends only between ARQ sessions, and while the channel
  is free.
- Link layer: `data2g/kisslink.py`.

- **Feedback without a session.** Each burst carries reports: for every
  station heard in the last 10 min, the mode and size it should use to
  reach us. That's the ARQ shifter's recommendation, from a per-station
  `GearShifter` fed by what we measured of its bursts. A TNC follows the
  report its next hop sent about it (up to 180 s old).
- **Stations are AX.25 callsigns,** read from the frames:
  - A burst's sender is its first frame's RF sender: the last digipeater
    that has repeated it, else the source.
  - A frame goes to its RF next hop: the first digipeater that hasn't
    repeated it, else the destination.
- **Robust broadcast** (`--broadcast-mode`, any mode within the cap; by
  default qpsk-r1/5, n10-qpsk-r1/5 under a 500 Hz cap; `--list-modes`) for UI
  frames whatever their destination, for non-AX.25, and for stations
  without a fresh report. Connected-mode frames (I, S, U other than UI) to
  a station with a report use the reported mode.
- **First-transmission success.** There are no resends at this layer (AX.25
  retries), so the KISS shifters only recommend modes with predicted
  P(usable) x P(codeword) >= 0.9 (`GearShifter.min_success`; the ARQ keeps 0).
  - Plain goodput shifting picked w48-qpsk-r1/3 at 0 dB AWGN and lost half
    the frames.
  - With the floor, 256-byte I frames, 6 per SNR, all delivered:
    - +20 dB: w48-256l-r5/8
    - +8 dB: w48-16qam-r1/3
    - 0 dB: w48-qpsk-r1/5
    - -4 dB: robust modes, CPM included
- **Burst format:** control codeword(s) (the ARQ's control masks, key
  0x4B53) holding [version, n_ctl][sender hash][n][hash, mode|size] x n.
  Then data codewords with the old length-prefixed framing. A burst grows
  past its size class to carry its first frame: 256-byte-PACLEN I frames
  outgrew a 12 s broadcast burst.
- **Listen before talk:** the TNC holds while `Receiver.channel_busy`.
  It hears every OFDM band and the CPM grids.
- **Tests:** `tests/test_kiss.py` (AX.25 parsing; two links over the real
  modem shifting from reports; UI and non-AX.25 staying robust; stale
  reports; the 500 Hz cap; 0 dB delivery).
- **Not done:** reports only travel inside bursts that carry frames. A
  station that only listens never reports, so traffic to it stays robust.

## Receiver CPU: incremental search, QPSK LLRs, burst routing (2026-09-26)

TLDR: listening dropped from 14% to 5.4% of a core per host; a Pat exchange's
busier host from 12.7% to 6.4%. Measured with two hosts under py-spy on a
PipeWire loopback, an outside process injecting noise (SNR sweeping 25 to
2 dB), Pat P2P (text, 8 kB and 4 kB attachments; 220-231 s, no bursts lost).

- **Profile before:** preamble search was 95% of listening CPU (OFDM 73%,
  CPM 22%); in the exchange, search while a burst arrived 36%, idle search
  26%, KISS trying ARQ bursts with its own keys 13%, the exact LLR 13%.
- **Incremental OFDM search** (`sync.StreamDetector`): the receiver searched
  its 1.9 s buffer every 0.25 s on three bands, each start's statistic
  recomputed ~8 times. The statistic at a start depends only on the audio
  after it (no carrier phase), so each band's detector now computes it once
  as audio arrives. The noise level is each bin's median over the last 8
  chunks (then the lowest bin's, as before), not the buffer's quantile.
  Starts whose whole head has been searched are not searched again, after
  one more hop (`Receiver.REVISIT`: the old repeated searches rescued a weak
  preamble now and then).
  - Detection, same seeds, old vs new: 506 vs 504 of 560 bursts across 7
    weak cells (ack-4f AWGN -8, MPP -3; qpsk-r1/5 MPG -2; n10-ack-4f AWGN -9,
    MPP -3; w48-qpsk-r1/5 MPP 2; fsk16r25-r1/2 AWGN -11). 16 min of noise:
    no header lock either way.
  - Data whose preamble was lost no longer false-locks: its own audio sets
    the noise level. Two tests that need such a false lock force the old
    search (`tests/test_tnc.py`).
- **CPM search:** only starts not yet searched (`cpm.find(lo, hi)`), and the
  shift/row loop vectorized: 3.4 -> 1.0 ms per grid per search.
- **Gray QPSK LLRs in closed form** (`constellation.llr`): the exact LLR is
  linear for Gray QPSK; same values to 1e-13, ~160x faster.
- **Burst routing:** in a session, a control codeword under the session's
  key claims a burst before KISS tries its keys; KISS gives up after slot 0
  and 1 fail instead of decoding every slot. Soft bits are computed once per
  burst and shared, and one-off decodes are remembered (`ModemRx`).
- **What's left** (exchange): search while receiving 30% (header reads 12%,
  the detector feed 10%), idle search 21%, ARQ decodes 14%.

## TODO

- Active constellation extension (Krongold & Jones 2003) in the TX
  clipper: after each clip, project every data symbol back into its
  allowed region, where inner points snap back and outer points may
  move only outward (the outward part of the Voronoi cell; for learned
  constellations, a one-time precompute per set). Peaks come down
  without moving any point toward a decision boundary, so it needs no RX
  or format change. Aimed at the top w48 modes, which clip at 4-6 dB of
  headroom; a guessed 1-2 dB PEP-fair gain, to be measured with
  scripts/clip_study.py. Plain projection (POCS) converges slower than
  today's 3-pass clip-and-filter (CLIP_OVERSHOOT) and typically wants
  ~4-10 passes. The smart-gradient variant (Krongold & Jones) gets most
  of the way in 1-2, so compare PAPR against pass count. The projection
  is differentiable, so it could also go into channel_torch's TX for
  constellation training.

- Trailer acquisition, as in FreeDV's data modes: a second sync
  sequence (preamble copy, or header repeat) at the burst's end, so the
  receiver can lock from either end. Likely a win on slow fading (mpg),
  where a burst whose preamble lands in a fade is currently lost whole.
  Costs airtime on every burst, and the gain depends on burst length vs
  fade duration, so a fixed sync threshold can't capture it. Evaluate
  once the gear-shifter tracks channel conditions: it could be a
  per-burst option (header flag, or trailer-bearing submodes) chosen
  for slow-fading, long-burst cases. Needs the RX to buffer backwards
  from a trailer hit and a header decodable without the preamble's
  noise estimate.

## Port back to SSTVAE

Things found here that should help SSTVAE. Most are receive-only, so
they need no format change.

- `hfchannel._rayleigh_taps` is not the Watterson spectrum it is labelled
  as: 2-sigma spread 1.5x the label and Butterworth skirts past the
  pilot rate. Its mpd is harsher than CCIR mpd. `_gaussian_taps` here
  matches ITU-R F.1487 (tested to 5%).
- `hfchannel.sample_clock_offset` uses np.interp, which adds ~-23 dB of
  distortion on this waveform. Hidden by the clipper, but wrong; use
  FFT resampling (`scipy.signal.resample`).
- Timing: "first path if >= 0.5 of peak" syncs to the late path whenever
  it is 2x stronger, putting the early path outside the CP. Place the
  window from the delay profile instead (`equalizer.window_shift`).
  SSTVAE's blind path could accumulate the profile over its whole
  window the same way.
- Pilot interpolation: Catmull-Rom across the drift tracker's +-2 sample
  steps mixes two timing references. Undo each step's phase first.
- Channel estimation: Catmull-Rom -> 2-D LMMSE. Mode A/B/C transmissions
  are long, so the Doppler/delay measurements are well conditioned.
  SSTVAE's latent weights would then come from the estimate MSE plus
  clip noise instead of |h|/median.
- CFO: a pilot-based refinement over the whole transmission, with the
  alias resolved against the data, instead of the preamble-only estimate
  (random FM under fading makes that one heavy-tailed).
- Header: interpolate its channel between preamble and first pilot.
- Simulator: SNR against the average (transmitted) power and fading
  taps unit power in expectation, not normalized per transmission.
- TX: clip threshold from the non-silent part (matters only for short
  transmissions, so low priority there).
- Clipper: per-symbol cyclic clip-and-filter, if the step 8 study shows
  it wins here.
- Preamble detection: SSTVAE's lag-M autocorrelation (the same
  detector this project started with) sets its sync floor. Data2G's
  per-repeat matched filter, differential across repeats, on a CFO grid
  (sync.detection_stat) missed none of the bursts the lag-M one missed
  at its 1% floor. RX-only.
- Header decoding: ML over valid words only, and a frequency-smoothed
  channel reference (modem.decode_header, _read_header). RX-only.
- Header: SSTVAE's Golay header could take the (192, 16) ML code and the
  whole-repeat timing vote; its header carries less, so an even lower
  rate fits in the same 2 symbols.
