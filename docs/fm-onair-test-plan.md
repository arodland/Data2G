# FM on-air test plan (M0 / M9)

TLDR: four radios, five pairings, one procedure per pairing: ping to set levels and check the TX path, a bandwidth check, a recorded sounding sweep at two or three RF levels, then ARQ sessions. Everything is recorded for offline replay. Tooling is ready: soundings on FM, an IF-recording analyser, and a run log (below).

## What the results decide

| measurement | decides |
|---|---|
| per-carrier S/N profile and ceiling per radio path (soundings) | f55 and f108 band edges (Q7); whether f108 needs a narrower fallback (VARA's 4.6 kHz) |
| top modes' decode vs level on mic radios | `FM_MIC_GAIN` (0.70 / 0.55); the limiter model |
| flat radios' noise tilt and top-carrier ceiling | `FM_TX_TILT_DB_OCT` (+3); whether the 6 kHz port filter is the ceiling |
| occupied bandwidth at the tuned level (scope or IF recording) | the 15 kHz 26 dB-down limit holds at the shipped drive; the tune procedure's target |
| ping verdicts vs deliberate misconfigurations | ping's thresholds (limiter 1.5 / 2.5 dB, tilt 1.0 / 1.5 dB) |
| per-mode decode vs snr_est (soundings, sessions) | the FM threshold table and its steady/fading split |
| session throughput vs the simulator's | whether the gear shifter's gaps matter (M6) |
| key-up time per radio | TX delay defaults (M3: 0.2 s of slack before high-rate bursts fail) |
| USB IF receive (IC-9700, TH-D75) | M10's first look: our own discriminator vs the radio's audio |

Only the conclusions go into the plan (Discussion #195), not this matrix.

## Equipment

| radio | interface | TX | RX |
|---|---|---|---|
| IC-9700 | USB | narrow (mic path) | narrow via USB AF; wide via USB IF |
| TH-D75 | AIOC, USB | narrow via AIOC | narrow via AIOC or USB AF; wide via USB IF |
| FT-8800R | DigiRig | wide (9600 port) | wide (9600 port) |
| UV-50X3 | cable, 1200/9600 TX menu | narrow or wide | narrow or wide: check which RX pin the cable uses first |

Settings per station: `--tx-path mic` for mic-input TX, `flat` for a 9600 port. The station's audio level: set once with ping (below), then left alone.

## Pairings

Simplex, one frequency, low power. Bring the signal down to the operating range with power steps or attenuators (the interesting range is CNR 5-30 dB, not full quieting), and log the setting.

| pairing | A → B | B → A | what it covers |
|---|---|---|---|
| P1 | IC-9700 → TH-D75, f55 mic | TH-D75 → IC-9700, f55 mic | narrow both ways, the common case |
| P2 | FT-8800R → IC-9700 (USB IF) | IC-9700 → FT-8800R, f55 mic | wide one way, narrow back: the per-direction case |
| P3 | FT-8800R → TH-D75 (USB IF) | TH-D75 → FT-8800R, f55 mic | the same with the other IF receiver |
| P4 | FT-8800R ↔ UV-50X3, f108 flat | | wide both ways (if the UV-50X3 cable is on RX9600) |
| P5 | FT-8800R → TH-D75 (AF) | | f55 sent flat into a mic radio's receive path |

Do P1 first: it needs nothing new. P4 depends on the cable check.

## Procedure, per pairing

1. **Log the setup** (run log, below): radios, interfaces, tx_path each way, power, attenuation, distance, frequency, TX delay.
2. **Key-up time:** each sender keys and sends one ping. The receiver's recording shows when the carrier arrived against the burst start. Set TX delay to cover it.
3. **Ping, both ways:** `PING <call> NARROW` (and `WIDE` where the link is wide). Adjust the TX audio level until the verdict is "level OK". Record both reports.
4. **Ping against known mistakes,** to calibrate the verdicts. Record every report:
   - a mic radio set to `tx_path flat` must be flagged (limiter);
   - TX level +6 dB over the tuned level must be flagged;
   - a flat radio on `tx_path mic` should get the tilt suggestion.
5. **Bandwidth:** at the tuned level, send a long top-mode burst (f55 256-point r.88 on mic radios, f108 256-point r.87 on flat ones). Measure 26 dB-down width on the IC-9700's scope (RBW as narrow as it goes), and record USB IF where possible. Target 15 kHz or less.
6. **Sounding sweep:** `sounding make` (every selectable FM mode, 2 rounds) played by the sender; the receiver records it. Repeat at two or three RF levels: near full quieting, near the middle of the ladder (the 64-point rungs failing), and near the bottom (16-QAM failing). Record AF, and IF where the receiver has it, at the same time.
7. **Sessions:** two ARQ transfers each way (`data2g-host`, a 30 kB message, KISS or the VARA interface), at the middle RF level. Then one through Pat's VARA FM session type if Pat is set up. Record the host log and both receivers' audio.
8. **Mobile, if possible** (TH-D75 handheld walking, or a car): one sounding and one session while moving. Log the route.

## Recording and naming

- Every receive is recorded: WAV at the profile rate (16 kHz) or 48 kHz, plus USB IF where available.
- Directory: `runs/onair/<date>/<pairing>-<direction>-<step>-<level>/`, with `rec.wav`, `if.wav` (when recorded), `notes.txt` (the run-log line), and the host log.
- Decode offline afterwards with `sounding decode`. Recordings are replayable through the receiver, so a later receiver change can be judged against them.

## Run log

One line per run:

`time, pairing, direction, step, tx radio, tx_path, tx level, power, attenuation, rx radio, rx path (AF / IF), CNR guess, file, notes`

## Tooling (ready)

1. **Soundings on FM** (`scripts/sounding.py`). `make` sends every FM-selectable mode (no CPM, w48 or n4). `--tx-path mic|flat` sets the sender's settings, `--bands f55 f108 w n10` picks a subset, and `--rate` is 16000 (default) or 48000. The sequence's JSON records the TX path.
   - `decode` takes the same `--tx-path` and `--bands`. It listens as the FM engine does, and predicts pass or fail per mode with the gear shifter's own outcome model (`predictor.fm_outcome`): the reference band's `snr_est`, mapped to each mode's band, with steady or fading chosen by the median noise spread.
   - The report's carrier response shows the TX path's tilt (a flat sender's +3 dB/octave) on top of the radios'.

   ```
   DATA2G_PROFILE=fm python -m scripts.sounding make seq.wav --call W1AW --tx-path mic --reps 2
   DATA2G_PROFILE=fm python -m scripts.sounding decode rec.wav --call W1AW --tx-path mic --reps 2
   ```
2. **IF analyser** (`scripts/fm_if_analyse.py`). It reads a radio's USB IF recording with our own discriminator. The IC-9700 and TH-D75 both give their IF as an ordinary USB audio device when a menu option is set, with the dial frequency at 12 kHz (`--centre`'s default; the spectrum's own centre is reported as a check). The sample rate is read from the WAV.
   - It reports the deviation (rms, and peak at the 99.9th percentile, within 7 kHz of audio), the occupied bandwidth (26 dB down and 99%) and the CNR, per stretch of modulation.
   - It writes the demodulated audio (de-emphasised for a mic sender) as `PREFIX.demod.wav`, which `sounding decode` reads.
   - The radio's own FM IF filter is about 15 kHz wide with soft skirts. So noise is read only where there is no carrier (leave a second or two of no carrier at the start and end of a recording, and between overs), and the filter's widths 3 and 26 dB down are reported from it. A 26 dB width near 15 kHz is then partly the radio's filter, not the sender alone: compare it with the filter's widths.
   - Validated on synthetic IF (a 12 kHz real IF at 48 kHz, CNR 27, f55 256-point on mic and f108 256-point on flat):

     | case | rms deviation | 26 dB width |
     |---|---|---|
     | no IF filter | within 1-5% of the truth | within 1% |
     | a radio-like 15 kHz IF filter | 1-6% under | up to 6% narrow (the filter's real effect) |

     The peak runs 5-23% high (discriminator noise). The filter's 3 dB width is recovered (14.7 kHz for a 15 kHz bandpass). The demodulated audio decodes every codeword through `sounding decode`, with the filter in the loop.

   ```
   DATA2G_PROFILE=fm python scripts/fm_if_analyse.py if.wav --tx-path mic
   DATA2G_PROFILE=fm python -m scripts.sounding decode if.demod.wav --call W1AW --tx-path mic
   ```
   - Recording IF: set the radio's IF-output menu option, then record its USB audio device as an ordinary WAV.
     - IC-9700: menu "USB Output IF" (or "USB Output AF" for its own demodulated FM audio, the AF receive path). Device "PCM2901 Audio Codec" (`alsa_input.usb-Burr-Brown_from_TI_USB_Audio_CODEC-00.analog-stereo`), s16le, 2 channels, 48 kHz native; the IF is on the left channel, the right is silent. Record: `parecord --device=alsa_input.usb-Burr-Brown_from_TI_USB_Audio_CODEC-00.analog-stereo --file-format=wav --rate=48000 --channels=2 --format=s16le if.wav` (the analyser reads the left channel).
     - IC-9700's IF filter, measured from 10 s of open-squelch noise (2026-11-17): 17.1 kHz wide 3 dB down, 19.9 kHz 26 dB down, around 12 kHz (noise flat from about 4 to 20 kHz). So a 15 kHz signal's 26 dB width is measurable through it, but not one much past 20 kHz.
     - TH-D75: TODO: menu item, device name, channels.
3. **Run log:** `runs/onair/log.csv`, with the columns above.
