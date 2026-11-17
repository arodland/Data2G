# On air, 2026-11-17: pairing P1 (IC-9700 and TH-D75, narrow), first session

KC2G, 432.950 MHz simplex, both radios at minimum power a few metres apart (IC-9700 S-meter +58..+61 dB over S9 on the TH-D75), so full quieting throughout: these runs say nothing about the low-CNR end.

## Setup that worked

- TH-D75 through an AIOC: audio "All-In-One-Cable" (mono, 48 kHz); PTT on /dev/ttyACM0, DTR high and RTS low (RTS high inhibits). In data2g-host: `--rig-model 1 --ptt-method dtr --ptt-device /dev/ttyACM0 --rig-rts low` (a `--rig-device` on the dummy rig fails with an IO error). The CM108 GPIO on /dev/hidraw13 does not key it.
- TH-D75 levels: mic level high, AIOC sink 70% (PipeWire). At mic level medium the AIOC needed 100% and clipped upstream of the modulator (in-band peaks pinned at ~2.2 kHz, modes squashed together, 64-point failed).
- IC-9700: CAT PTT on /dev/ttyUSB1, 19200 baud, Hamlib 3081; DATA MOD USB, USB MOD level 40%; mode FM-D, 15 kHz filter; 1% power. USB audio "PCM2901 Audio Codec" (stereo, 48 kHz; the receive audio on the left channel). Menu "USB Output IF" or "USB Output AF".
- Each data2g-host needs its own `--config FILE` and `--no-tci`: with neither, a host loads the station's saved settings (here: TCI to a K4) and ignores the audio and PTT flags. Separate `--kiss-port` per host.

## Measurements

- IC-9700 IF filter (open-squelch noise): 17.1 kHz wide 3 dB down, 19.9 kHz 26 dB down, centred at 12 kHz.
- TH-D75 TX into the IC-9700 IF (our discriminator), mic high, AIOC 70%: QPSK 646 Hz rms in-band (400-3200 Hz), peak 1.6 kHz; 64-point 0.71 and 256-point 0.54 of QPSK (FM_MIC_GAIN 0.70 / 0.55 arrive as sent). Occupied bandwidth 6.0 kHz 26 dB down, 5.8 kHz 99%: far under 15 kHz.
- TH-D75 mic path: in-band deviation caps at about 2.1-2.2 kHz peak even on standard FM (the CW ID reaches it from AIOC 60% up). For data2g it behaves like the narrow-channel case (Q16): its limiter, not the 15 kHz width, sets the drive.
- f55 passband, TH-D75 mic path into the IC-9700 IF: within +-2.5 dB from 450 to 3150 Hz.
- IC-9700 data-mode TX compresses its USB input: received level unchanged from USB MOD 40% to 80% and over PipeWire 25-70%; at 10% only 5.7 dB down for an 18 dB cut.
- TH-D75 USB IF (USB Out Select IF, band B, out level 7, squelch forced open): no usable IF in the stream (flat noise at -91 dB/Hz, dips 0.5-5 dB when a carrier arrives). Parked.

## Ping (data2g-host PING, both ways)

- TH-D75 -> IC-9700 (AF): snr 21.0, thirds 24.9/25.4/20.8, dist 0.0: "level OK".
- IC-9700 -> TH-D75 (AIOC): snr 14.8-15.6 whatever the IC-9700's level or the TH-D75's volume (its audio from -43 to -10 dBFS): "level OK".
- Gaps in the verdict: it cannot see low deviation, a transmitter that compresses its input, or a receive-side ceiling; "level OK" in every case here.

## Soundings (all 14 f55 modes, one round, tx_path mic)

| direction | codewords | genie S/N | snr_est | top rung passing |
|---|---|---|---|---|
| TH-D75 -> IC-9700 AF | 117/135 | ~30 dB | ~20.4 | 256l r.80 (r.83 12/15) |
| IC-9700 -> TH-D75 AIOC | 34/135 | ~17.5 dB | ~15.4 | 64l r.50 (r.67 6/8) |

snr_est on the real AF path reads about 10 dB under the genie S/N, and 256l r.72 and r.80 decoded where the FM thresholds (fitted on the simulator) predicted failure: the thresholds are 2-4 dB conservative on this path. On-air calibration of the threshold table is needed (M6).

## ARQ sessions (data2g-host, BW1200 = NARROW, 30 kB random data)

- TH-D75 -> IC-9700: 32.9 s after CONNECTED, 7298 bps, intact. Modes f55-256l-r72 then f55-64l-r80.
- IC-9700 -> TH-D75: 58.3 s, 4115 bps, intact. Modes 16qam-r1/2 (w), f55-64l-r50, f55-64l-r67.

## Open

- The ~15 dB ceiling from the IC-9700 to the TH-D75: the IC-9700's data-mode TX (compression, its modulation chain) or the TH-D75's receive audio path. Needs another receiver of the IC-9700 (the FT-8800R) or a working IF on the TH-D75.
- Ping verdicts: report "low deviation / receive-limited" cases (needs a calibration of each station's receive gain, or S/N vs level steps).
- Thresholds: recalibrate on air data; snr_est's offset on real AF paths.
