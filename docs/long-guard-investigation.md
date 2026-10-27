# Long echoes: a longer guard and a detector for it (future investigation)

> Not scheduled. Written 2026-10-26 after reading ion-hf's PHY
> (github.com/jesseforkum/ion-hf, docs/phy.md "Long numerology").

## TLDR

A 6 ms cyclic prefix rescues the w and w48 modes on 4-7 ms echoes by
several dB, at about 8% throughput. n10 gets worse with it, and NCP values
other than 32 and 48 break the receiver, so it is not shippable as is. A
long-echo detector is only useful once there is a long-guard mode to move
to, so both wait together.

## What was measured

Branch `worktree-cp-study` (9ae5b5a): `DATA2G_NCP` overrides `config.NCP`
for studies, and `hfchannel` gains an `nvis` preset (ITU-R F.1487
mid-latitude disturbed NVIS, 7 ms, 1 Hz). `runs/cp_study.sh` and
`runs/cp_nvis.sh` run `scripts/dfts_study.py thresholds` with the `/dd`
arm (the shipped ARQ receiver), 100 paired bursts per point. Figures are
10% points, average power, 2500 Hz. `mpd~` is mpd with the second path at
0.5-1.5x its 4 ms per trial. Results: `runs/cp_study_ncp{32,48}.csv`,
`runs/cp_nvis_ncp{32,48}.csv`.

| mode | channel | CP 4 ms | CP 6 ms |
|---|---|---:|---:|
| w48-16qam-r1/2 | mpd~ | 16.0 | 11.5 |
| w48-16qam-r3/4 | mpd~ | never | 17.1 |
| w48-64l-r2/3 | mpd~ | never | 23.3 |
| w48-64l-r2/3 | mpd | 26.8 | 19.5 |
| qpsk-r1/2 | mpd~ | 7.3 | 4.7 |
| w48-qpsk-r1/2 | nvis | never | 11.5 |
| qpsk-r1/2 | nvis | never | 10.9 |
| qpsk-r1/5 | nvis | 0.1 | -0.3 |
| n10-qpsk-r1/3 | nvis | 2.9 | 1.0 |
| n10-16qam-r1/2 | mpd | 8.1 | 15.1 |
| n10-16qam-r3/4 | mpd | 19.7 | never |

- AWGN and mpp~ moved by 0.5 dB or less.
- The CP's cost is not in these numbers: 192 to 208 samples per symbol,
  7.7% fewer cells per second.
- NCP 40: the noiseless clip-constant loopback fails its header decode.
  NCP 64: half of mpd bursts fail at 30 dB, genie channel included. Some
  code assumes the CP length. Find it before trusting any other value.
- n10's regression on mpd is unexplained. 10 carriers cannot resolve the
  paths (16-sample delay resolution); suspect a receiver constant in
  samples tuned at NCP 32 (`SUPPORT_SLACK`, `PADDED_SLACK`,
  `DEMOD_BACKOFF`, `HEADER_BACKOFF`, `window_shift`, the `_DELAYS` span).

## Why not ion-hf's numerology

Their Long is 25 Hz spacing, 40 ms + 10 ms. By their own measurements it
is marginal at 2 Hz Doppler (85-90% at +20 dB on 7 ms / 2 Hz), and 2 Hz is
our mpd. Its ICI at sigma 1 Hz is -23 dB, about the ISI floor it removes
there. Keeping 50 Hz and lengthening only the CP keeps our Doppler
tolerance and our pilot grid (156 ms at 6 ms CP, still clear of mpd's
spread).

## A long-echo detector

ion-hf (docs/phy.md "Long echoes and the channel measurement", link.md
channel report): echoes past the guard do not show in the decision-directed
delay profile, because the decisions follow an estimate whose model spans
at most the CP. They show as excess decision-directed noise over the
preamble's noise instead: 3.4-9 dB on a 7 ms echo from +8 dB, under 0.5 dB
on AWGN and CCIR poor. Above 2 dB the receiver reports a 6 ms echo and the
link moves to the long guard or MFSK.

We have both halves: `equalizer.preamble_noise` (thermal only) and the DD
residual. The ratio is a cheap per-burst statistic.

Why not now:

- No long-guard mode to route to. On today's ladder the outcome model
  already sees long-echo bursts fail and steps down; a flag would only
  sharpen a choice between modes that all suffer.
- As a predictor input it needs the outcome data regenerated and a full
  retrain (and the copy-lock follow-up is already queued for one).

## Steps, when picked up

1. Make NCP a clean parameter: find what breaks at NCP 40 and 64, and why
   n10 regresses on mpd at 48. Python and native.
2. Decide the shape: a per-band guard (w/w48 only, where the paths
   resolve) or a separate long-guard submode family on w48. A per-burst
   guard needs a header field or a distinct preamble, so it is a wire
   change and a protocol bump.
3. Add `nvis` to the study channel set and to the session sim's channel
   mix, with fixed and drawn (`nvis~`) delays.
4. The detector: excess DD noise over preamble noise, per burst; measure
   its distribution on awgn/mpg/mpp/mpd/nvis to set the threshold, then
   feed it to the outcome model at the next retrain.
5. Decide by the paired loss study (PEP-fair, `DATA2G_PEP_REF_DB=5`,
   12+ seeds) on mpd~ and nvis, against the 7.7% throughput cost
   everywhere else.
