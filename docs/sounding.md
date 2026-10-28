# Over-the-air sounding

`scripts/sounding.py` builds one WAV holding every mode Data2G uses, and
reports on a recording of it made anywhere: the far station, a KiwiSDR or
WebSDR, or a receiver next to the transmitter.

What it is for:

- Validate the whole ladder on a real path in one go: which modes decode,
  which headers are heard, without setting up a session at both ends.
- Calibrate the simulator against the air: the path's SNR, delay spread,
  Doppler spread and fades are measured from the known transmitted symbols
  (a genie), and each mode's result is set against what
  `codes_data/mode_thresholds.json` predicts on that channel.
- Check waveform integrity through a real rig: clipping, filters cutting
  the band edges, and whether the radio keeps a burst's head and tail
  (protocol 15 has 20 ms of lead-in and lead-out, `config.LEADIN_SAMPLES`).

## Quick start

```sh
python -m scripts.sounding make runs/cwid-sounding/seq.wav --call W1AW --rate 48000
# ... transmit seq.wav, record it somewhere ...
python -m scripts.sounding decode rec.wav --call W1AW --out runs/cwid-sounding/rec
```

`make` writes `seq.wav` (16-bit PCM, 8000 or 48000 Hz) and `seq.json` (each
burst: number, submode, codewords, start and end in seconds, cap code). It
prints the length. `decode` writes `rec.report.md` and `rec.report.json`.

`decode` regenerates the sequence from `--seed`, `--reps` and `--modes`, so
it needs no sidecar, but these must be what `make` was given (`--call` only
labels the report; the IDs are not decoded). Any WAV works: any sample rate,
mono or stereo (the first channel is used), 8/16/24/32-bit integer or float.

## On air

The operator transmits; the tool never keys anything.

1. Make the WAV at the sound card's rate (`--rate 48000` for most cards).
2. Set the TX audio level as for Data2G itself: the file's bursts peak at
   full scale (envelope peak 1.0, as `engine.Engine._start_tx` sends them),
   so set the level where ALC just stops acting, at the rig's data
   power. Each burst is constant peak power, like a live session.
3. Key the rig and play the file. Either hold PTT for the whole file
   (rigctl `T 1`, a footswitch, the rig's TX button), or use VOX: each
   burst has the engine's 0.1 s PTT lead ahead of it, and a VOX that drops
   in the 1.1 s gaps is the same as the engine keying each burst. Any
   player works (`aplay -D <device> seq.wav`, `pw-play`, Audacity).
4. The file starts and ends with a CW ID (`DE <call>`, then `DE <call> SK`
   at the end) at 20 wpm on 1500 Hz. A sequence longer than 10 minutes
   (more `--reps`) needs an ID inside it too: make two files.

Transmitting it in a busy segment is not a test of anything; pick a clear
frequency, as for any data QSO. Live Data2G stations ignore the bursts:
every codeword's CRC is masked with a fixed sounding key (`SOUND_KEY`),
so a listener's CQ/ID check (mask 0) fails. They may show BUSY while the
bursts are on air, which is right.

### Recording at the far end

- Another Data2G station, or anyone with a receiver: record the receiver's
  audio for the whole file, starting before the first CW ID. Any recorder
  works; the decoder finds the sequence by itself.
- KiwiSDR: tune to the same dial frequency in USB, widen the passband to
  at least 200-2900 Hz (the 2400 Hz modes run 350-2700 Hz; Kiwi's default
  USB filter cuts the top), AGC off or slow with manual gain if you can,
  and use the record button. It saves 12 kHz WAV (nominally; the
  receiver's drift tracking absorbs a Kiwi's few ppm).
- WebSDR: the same settings, its record button (8 kHz WAV).

Tuning within +-150 Hz of the transmitter's dial is enough
(`config.ACQUIRE_REACH_HZ`); the report gives the offset found. Keep the
recording's level well under full scale: the clipping check counts samples
at or above 0.99 of full scale, and a recorder that clips makes the rig
look guilty.

## The sequence

Defaults: 50 modes x 2 rounds = 100 bursts, 7.7 minutes.

- Modes: every OFDM submode in `config.SUBMODES` (n4, n10, w, w48,
  including the ack and polar control modes) and every CPM mode in
  `cpm.SPECS`.
- Size: OFDM bursts carry about 2 s of codewords (whole codewords, at
  least one; 1.2-4.0 s on air with the preamble and header). A CPM burst is
  its control codeword and one data codeword (5.5 s for fsk32r62, 10 s for
  fsk8r50, 14.6 s for fsk16r25), the shortest a CPM data burst can be.
  Long enough for the genie to see the channel move, short enough that
  the whole ladder fits in a few minutes.
- Order within a round: each family (n4, n10, w, w48, CPM) sorted floor
  first (the AWGN column of `mode_thresholds.json`), and the families
  interleaved by position, so a round runs from the most robust modes of
  every family to the fastest, and consecutive bursts are from different
  families: a fade of a few seconds costs one mode of each, not a family.
- Repetition: R rounds (`--reps`, default 2), the same order each time,
  about 3.8 minutes apart. One round is one look at each mode; two say
  whether a failure was the mode or the moment.
- Spacing: each burst has 0.1 s of silence (the engine's PTT lead) ahead
  of it and 1.0 s after it, so the receiver is idle again before the next
  one, as in a session.
- Payloads: each slot's payload is `D2 nn nn ss` (magic, burst number,
  slot) and then bytes from a generator seeded by (seed, burst number,
  slot). A decoded slot names its burst, so bursts are placed even when
  the recording's timing is unknown; the rest of the payload is checked
  against the regenerated one.

## Decoding

The live receiver (`receiver.Receiver` with `engine.receive_accept()` and
every CPM grid, fed in 0.1 s blocks as the engine feeds it) runs over the
whole recording after it is resampled to 8 kHz. Every burst it hands over
is decoded slot by slot under the sounding masks (`phy.ModemRx`, DD
unbudgeted), measured as the engine measures it (`phy.measure`), then
matched to the sequence: by the burst number in a decoded payload, else by
mode within 1 s of where it should be once any decoded burst has fixed the
sequence's offset in the recording. Headers heard that match nothing are
listed as extras (false locks). A faded CW ID can read as a weak header
(an ack-1f at score 0.34 on simulated mpd): an extra at the very start or
end is that.

### Genie measurements

OFDM: the known burst is modulated again and demodulated at its own timing
(`modem.receive(known=...)`), which gives X, the transmitted cell values
after the clipper and TX filter. With Y the received cells (the
receiver's own demodulation), H = Y / X on every pilot and data cell.

- Noise: each cell's H against the mean of its two neighbours in time (a
  second difference cancels a linear change), median-based, weighted by
  1/|X|^2. SNR: the cells' power over that noise, in the 2500 Hz
  reference (`modem.snr_db`'s convention, hfchannel's SNR). On AWGN it reads
  within 0.8 dB of each burst's simulated SNR at -4 to +6 dB (the
  receiver's `snr_est` read 0.2-2.3 dB high on the same bursts).
- Delay spread: the magnitude of H's correlation between adjacent carriers
  (50 Hz apart), each carrier normalized to unit power so the rig's
  filter shape does not count, reported as two equal paths' RMS spread
  (arccos(rho) / (2 pi 50 Hz)). Simulated mpg/mpp/mpd/mps (0.25/1/2/1 ms)
  read about 0.35/0.9/1.6/1.0 ms. It cannot tell a spread over 5 ms.
- Doppler spread: the magnitude of H's correlation in time at lags of
  1-192 symbols, fitted as a Gaussian Doppler spectrum (hfchannel's) up to
  where it first falls under 0.5, reported as 2 sigma (hfchannel's
  `doppler_hz`). Per burst it scatters by 2x (a 2-3 s burst sees one or two
  fades); the path figure pools every OFDM burst (each lag's median), which
  read 0.08/0.98/1.94/0.15 Hz for simulated mpg/mpp/mpd/mps (0.1/1/2/0.15).
- Per carrier and per frame SNR: the band's edges and the burst's head and
  tail.

CPM: the known tones' energy against the others' (median-based, as
`cpm.measure`), SNR in the same reference; per tone and per tenth of the
burst. No delay or Doppler figure.

Each burst's SNR is its own average power's. The path SNR is the same
referred to PEP (plus `phy.peak_db` for the mode and cap), which puts every
mode on one scale, since every burst goes out at the same peak. Fade
statistics are over that: median, 10th and 90th percentiles, extremes,
and the fraction of bursts over 6 dB under the median.

### Prediction

The path's channel is classed as the nearest column of
`mode_thresholds.json` (awgn, mpg, mpp, mpd) from the pooled Doppler and
delay spread (slow but selective, like hfchannel's mps, counts as mpp).
For each mode, the predicted SNR is the path's median PEP SNR less the
mode's peak_db, and the prediction is pass if that is at or above the
mode's 10% codeword failure point on that channel. "got" is pass when 90% of
the mode's codewords decoded. Disagreements are what to look at: a mode
that fails with margin to spare points at the waveform or the receiver on
that path; one that passes under its threshold, at the simulator being
harsher than the air. With two bursts a mode, either is weak evidence
alone; the pattern over the ladder is the result.

### Waveform integrity

Per burst:

- clipping: the fraction of the recording's own samples (before resampling)
  at or above 0.99 of full scale, and the burst's peak in dBFS;
- head and tail: the power of the burst's first and last 40 ms against its
  median (noise before the burst subtracted), and the genie's per-frame
  SNR (`frame_db` in the JSON). A radio whose VOX, ALC or receiver AGC
  takes the first few tens of ms shows a low head. On simulated channels
  heads and tails read within +-3 dB, with the occasional fade deeper;
- band edges: the genie's per-carrier SNR. The report averages it over each
  band's bursts relative to its median carrier: a filter cutting the band
  edges shows as falling outer carriers, a fade does not (it averages out).

## Report

`*.report.md`: a summary (offset, bursts heard and decoded, path SNR and
fades, channel, band responses, clipping, head and tail), a per-mode table
(heard, codewords, genie and receiver SNR, predicted SNR, threshold, margin,
predicted and actual pass) and a per-burst table. `*.report.json` holds all
of it and more: each burst's receiver measurements, genie carrier and
frame SNRs, time correlations, CFO, header score, and every mode's
threshold on all four channels.

## Limitations

- A burst whose header was not heard gets no genie measurement; it counts
  as not heard. A known-timing receive at its expected position would
  measure it too.
- The genie's delay and Doppler figures assume two equal paths and a
  Gaussian Doppler spectrum (the simulator's model); real paths are only
  roughly that. n4's four carriers give a coarse delay figure.
- Noise biases the correlations up, so at low SNR delay and Doppler read
  low: under about 5 dB per burst, trust the path figures over a burst's.
- The prediction uses the 10% points of a seeded simulation on four
  channels, at the median SNR. It does not run the simulator on the
  measured channel; `scripts/ladder_study.py` can, given the measured
  Doppler and delay as a custom `hfchannel.FadingPreset`.
- Two bursts a mode is a small sample. For calibration, record several
  sequences, or raise `--reps`.
- Interference shows up as noise in the genie SNR; it is not separated out.

## Why no host or native mode

A `data2g-host` command that keys the rig and plays the sequence is not
built. The WAV played with any player while the rig is keyed (PTT held,
VOX, or rigctl) is the same audio the engine would send, so a host mode
would add CAT/PTT handling and nothing on air. If it becomes worth having
(unattended beacons, a scheduled sounding), it would take: the sequence
built in the host (`sounding.sequence`, or a native port of it: payloads,
`tx_audio`, the CW ID), played through the host's audio output with its
PTT keyed for the whole file or per burst, and a command to start it
(e.g. `SOUND <call> <seed> <reps>`); about the size of the CQ path.
