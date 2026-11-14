# Data2G FM: plan

Status: draft for discussion, 2026-11-13. Nothing here is built.

## TLDR

- Build one FM channel simulator first. Every later step is judged on it, and nothing in the repo models a limiter-discriminator today.
- "Narrow" is a retune of the existing 8 kHz OFDM (w48 already sits at 350..2700 Hz). "Wide" is the same modem clocked at 16 kHz, then widened. Neither needs a new waveform family.
- The structural cost is making `config.FS` and the mode tables a per-profile choice (HF or FM) in Python and native. Decide how before any FM mode is frozen (question Q1).
- Four decisions are the user's: Q1 (profile mechanism), Q2 (keep a floor below QPSK r1/3), Q3 (digipeating), Q4 (VARA FM host dialect). The rest are measurements.

## What the others do

VardeFM, from its `docs/PROTOCOL.md` and `docs/RADIO.md` (docs only, no source read):

| | Narrow | Wide |
|---|---|---|
| Connection | mic / speaker | flat 9600 data port, both ends |
| Band | 422..2672 Hz | 281..6000 Hz |
| Carriers, spacing | 49 at 46.875 Hz | 62 at 93.75 Hz |
| Cyclic prefix | 2.67 ms | 1.33 ms |
| Pilots | every 6th carrier, every symbol | same |
| TX tilt | none | +3 dB/octave |
| Deviation target | about 2.5 kHz | about 3 kHz |
| Net rate, 7 levels | 0.7..7.4 kbit/s | 1.9..18.7 kbit/s |
| 10% codeword failure | about 3..16 dB | about 5..24 dB |

- One LDPC length (2400), BPSK r1/2 to 64QAM r5/6. Payload at most 4 s a burst, 24 codewords.
- Symbols hit by a click or squelch pop get low-confidence soft values.
- No frequency search. The FFT window never moves, so burst length is bounded by sample-clock error (13 s narrow at 100 ppm).
- Turnaround 0.2 s. Default txdelay 0.15 s, handhelds need 0.2..0.3 s.
- Level choice from a threshold table plus 1 dB, with per-level SNR ceilings (louder drive on robust levels).
- VARA TCP host interface on 8300/8301. No digipeating, no KISS, no compression. Tune by a 1 kHz tone and a deviation meter; no automatic tune.

VARA FM (closed; these numbers are from memory and one blog post, not verified): 46.875 Hz spacing, narrow about 470..3140 Hz, wide about 280..5670 Hz, top rates quoted near 12.7 kbit/s narrow and 25 kbit/s wide, digipeating through up to two hops, an automatic drive-level tune between stations. Verify before using any of it as a target (M0).

Data2G today: w48 tops out at 8222 bps on paper (256-point r5/8, 350..2700 Hz), above VardeFM narrow's 7.4 kbit/s. It has no mode wider than 2400 Hz and `FS = 8000` caps audio at 4 kHz.

## How FM differs from the HF channel the modem was tuned for

1. The limit is peak deviation, not power. Audio SNR after the discriminator scales with deviation squared, so PAPR costs exactly what it costs on a PEP-limited HF transmitter. The existing clip headroom, ACE and PEP-fair machinery (`DATA2G_PEP_REF_DB`, `clip_constants.json`) carries over with "peak deviation" in place of "PEP".
2. Above threshold (roughly 10 dB CNR) the audio is clean, 25..40 dB, and capped by the radios (distortion, hum, limiter, sound card), not by the path. Below threshold it collapses over about 3 dB of CNR into clicks. The ladder needs many high rungs and few low ones. Measured SNR says little about margin near the cliff.
3. Noise is not white. On a flat (9600) path the discriminator's noise voltage rises 6 dB/octave: 6 kHz is 26 dB noisier than 300 Hz. On a mic/speaker path de-emphasis flattens the noise, but the TX limiter sits after pre-emphasis, so the top carriers eat the deviation budget. Mixed stations (one flat, one mic) see a 6 dB/octave tilt either way.
4. No audio frequency offset. RF mistuning becomes DC and distortion at the discriminator, never a shift of the tones. Only sample-clock error remains (up to 100 ppm a card).
5. Delay spread is microseconds. The cyclic prefix only has to cover the radios' audio filters (group delay at the band edges, the CTCSS high-pass). The 4 ms HF prefix is mostly wasted.
6. Fixed stations see a static channel. Mobile stations see Rayleigh flutter at 10..50 Hz, which the discriminator turns into click bursts and millisecond dropouts, not into smooth audio fading. The answer is erasure of hit symbols and interleaving, not Doppler tracking.
7. Radios key in 20..300 ms and are quiet in milliseconds. HF's `TURN_S = 1.3` and long bursts are the wrong trade: FM wants short overs.
8. Squelch, tone squelch and key-up transients (PLL settle, a DC step through the AC coupling) eat the start of a burst.
9. Repeaters add a second pre/de-emphasis pair, clipping, audio delay, hang time and time-out timers.

## Milestones

Dependencies are named by ID. Sizes are rough new or changed lines of Python, native excluded unless said.

### M0. Baseline (no modem changes)

- Run today's ladder through M1's simulator, and on a real pair of FM radios through mic/speaker, with `sounding.py`.
- Output: where each w48 mode fails against CNR, the SNR ceiling per radio path, and VARA FM's real figures if a licensed copy is at hand (the VARA Wine harness exists).
- Depends: M1 for the simulated half. The on-air half can start now.
- This sets the targets for M4 and M5. Until then "match VARA FM narrow" is a guess.

### M1. FM channel simulator (`data2g/fmchannel.py`, about 300 lines)

- Audio in, audio out, at any FS. Inside: optional pre-emphasis, limiter and splatter filter (mic path) or flat (9600 path), FM modulator at complex baseband, IF filter (12.5 or 25 kHz channel), flat Rayleigh/Rician fading at a Doppler, AWGN at a CNR, RF offset, limiter-discriminator, optional de-emphasis and 300 Hz high-pass, sample-clock offset.
- Knobs for the ugly parts: key-up transient, squelch gate with an opening delay, CTCSS leak, mic-input AGC/compressor, a repeater (two hops with a clipper between).
- Same call shape as `hfchannel.apply_channel` so `with_native.py`, the loss study and the session study take it as another channel name.
- Calibrate against recordings of real radios (M0). Unlike HF paths, a radio's audio response is repeatable, so here recordings are good evidence (my reading, check it holds).
- Depends: nothing. Everything else depends on it.

### M2. Profiles: HF and FM constants side by side

- Today `config.py` is one set of module constants, `FS` is imported by 19 Python modules, about 22 native files hold an 8 kHz assumption, and the generator emits constexprs.
- Needed: FS, numerology (M, NCP, RS, CARRIER0), BANDS, SUBMODES, thresholds, outcome model, ARQ timing and the header CRC seed all chosen by profile.
- The CRC seed must differ so an HF node rejects an FM header where the preamble is shared (narrow FM at 8 kHz would otherwise look like w48).
- Mechanism is Q1. Recommendation: profile fixed at process start in Python (one env var or flag read in `config.py`), native built once per profile from its own generated header. Constexprs stay, no runtime indirection in the hot path, and the GUI picks which core to load.
- Depends: Q1. Blocks M4, M5, M6 from freezing anything. Receiver work (M3) and exploration of M4/M5 can run before it using the existing `DATA2G_*` study overrides.

### M3. Receiver changes for FM (receiver-only, testable on today's waveform)

- Coloured noise: check `equalizer.per_carrier_noise` copes with a 26 dB slope across the band, and that the demapper uses it everywhere (header included).
- Click and dropout erasure per symbol, from the residual against the pilots or the envelope. `audio.Blanker` is tuned for HF static; measure it on discriminator clicks before reusing it.
- Acquisition without a frequency search: sample-clock only. This is a physical property of the channel, not a narrowing by expected mode, so it does not break the "search must stay full" rule. It should cut detector CPU several times over (the HF note in `config.py` measured 5x between search widths).
- Tolerance to a missing or damaged burst start (squelch opening, key-up transient): detect on later preamble repeats, or lean on the header copy.
- Depends: M1.

### M4. Narrow ladder (mic/speaker, FS 8000)

- Start from w48. Decide by measurement on M1 and real radios:
  - Band edges. 300..3000 Hz is nominal, many radios are done by 2700 Hz. A band that some radios cannot pass is worse than a narrower one.
  - Cyclic prefix. 32 samples now; 8..16 may do. Worth up to 15% rate.
  - Pilot density. One symbol in six was sized for 2 Hz Doppler. A static channel needs the preamble plus sparse tracking of clock drift; flutter is handled by erasure. Worth up to 15%.
  - Preamble repeats (8 now) and header airtime, for short overs.
  - Top rungs: 256-point at r3/4..r5/6, perhaps 1024-point, if the radios' SNR ceiling allows. Arithmetic: 48 carriers, 16-sample prefix, pilots 1 in 12, 256-point r5/6 gives about 13 kbit/s.
  - Clip headroom per mode, re-picked under a peak-deviation limit with pre-emphasis and the radio's limiter in the loop.
  - TX tilt, so the pre-emphasised signal uses the deviation budget evenly.
- Drop the 1200 Hz, n10, n4 and CPM families from the FM profile unless Q2 says otherwise.
- Depends: M1, M3 (the erasure changes which rungs survive flutter), M0 for targets, M2 to freeze.

### M5. Wide ladder (9600 port or direct taps)

- Step 1, almost free: run the whole modem at FS 16000. Every sample count stays, so every table, pilot and header code is reused. Carrier spacing becomes 100 Hz, symbols 12 ms, prefix 2 ms. w48 lands on 700..5400 Hz and its ladder doubles: 16.4 kbit/s at the top. That already clears 10 kbit/s.
- Step 2: a wider band on the same grid, about 300..6000 Hz (58 carriers), with a new pilot (`scripts/design_pilot.py`) and header layout. Arithmetic: about 20 kbit/s with today's top code, more with M4's prefix and pilot savings.
- Step 3: fit the f-squared noise. A fixed TX tilt first (VardeFM uses +3 dB/octave). Per-carrier bit loading only if the fixed tilt leaves several dB on the table.
- Check real 9600 ports: many roll off before 6 kHz, and deviation 3 kHz plus 6 kHz audio is 18 kHz by Carson's rule, wider than some IF filters. The band edge is a measurement (Q7).
- OFDM starts at 300 Hz, so ordinary AC-coupled sound cards work. A single-carrier FSK floor mode would need response to a few Hz. Not planned.
- Depends: M1 (flat path), M2 (FS per profile, which is most of M2's cost), M3 (coloured noise).

### M6. ARQ timing and gear shifting

- Timing constants per profile: `TURN_S`, `TIMEOUT_S`, `MAX_BURST_S`, burst-length policy, PTT lead (already `--ptt-on-delay-ms`). FM wants turnarounds near 0.2..0.3 s and bursts of a few seconds.
- Gear shifter: start with a threshold table per mode on M1 (as `mode_thresholds.json`), then train the outcome model on FM data. The HF model's inputs (spread, fading regime) mean nothing here. Likely inputs: per-carrier SNR profile, click or erasure rate, SNR ceiling.
- Cliff behaviour: fast fall-back on the first lost codewords, fast climb when quieting returns. Judge by paired sessions on M1 with fading and with a CNR ramp through threshold, in dB of CNR where a rate is reached.
- Depends: M4 and M5 ladders frozen, M2.

### M7. Station setup tools

- Deviation tune: a tone and a modem-like test burst, level meters, and a report of the far end's reading (the automatic tune VARA FM users expect). `sounding.py` is the starting point.
- Overdrive diagnostic: the signature seen on 2026-10-31 (OFDM SNR pinned near the PAPR while FSK reads 25 dB or more) as a plain warning to the operator.
- Radio path setting: mic or flat, which picks narrow or wide and the TX tilt.
- PTT: `native/core/rig` has no CM108/CM119 GPIO keying that I could find. Most FM sound-card interfaces (AIOC, DigiRig Lite, DRA, Masters Communications) use it or serial RTS. Confirm and add.
- Depends: M2 for the UI. Otherwise independent, and useful to today's users on FM.

### M8. Host and network features

- VARA FM host dialect so Winlink Express and Pat treat the modem as VARA FM (Q4). Known differences to check: `CONNECTED` carries NARROW or WIDE, no BW commands.
- Digipeating (Q3). A link-layer feature with ARQ consequences (who acknowledges, timing across hops). Its own design doc if wanted.
- KISS and broadcast should work unchanged over the FM profile. Verify, do not redesign.
- Depends: M6.

### M9. Native, Pi, on-air, release

- Native port of M3..M6, generated tables per profile, `native/build-test` ctest.
- Pi 4 bench: wide at 20 kbit/s is about 2.5x the LDPC work per second of today's top mode, against a turnaround four times shorter. Decode must stream during the burst (as CPM now does) or the turnaround target moves.
- On-air trials on at least: two mobiles/handhelds on mic, two 9600-port radios, one mixed pair, one repeater.
- User guide chapter, speed sweep per `docs/speedtests/README.md`.
- Depends: all of the above.

## Dependency summary

```
M1 sim ──┬─> M0 baseline ──> targets for M4, M5
         ├─> M3 receiver ──┬─> M4 narrow ──┐
         │                 └─> M5 wide ────┼─> M6 ARQ + shifter ──> M8 host/digi ──> M9
Q1 ──> M2 profiles ────────────> (freeze M4, M5, M6)
M7 tools: independent, needs M2 only for UI
```

First three pieces of work, in order: M1, the on-air half of M0, M3. None needs a decision.

## Open questions

Decisions for the user:

- Q1. Profile mechanism. One codebase with a profile fixed at start (recommended), a runtime switch inside one binary, or a fork? A runtime switch means templating or namespacing every constexpr in native.
- Q2. Does FM keep anything below QPSK r1/3? The CPM floor reaches SNRs FM audio never delivers with a usable carrier, and squelch hides them anyway. Recommendation: OFDM QPSK r1/5 is the bottom, no CPM, no n4/n10.
- Q3. Digipeating: in scope, and how many hops? VARA FM gateways use it. VardeFM does not have it.
- Q4. Should the FM host present as VARA FM (so existing Winlink clients work untouched) or only as Data2G?
- Q5. Is over-repeater operation a goal for the first release, or simplex only?
- Q6. Direct SDR operation (modem does its own FM modulation and discrimination from IQ) is the natural end of "direct taps". M1 contains the code for it. In scope later, or never?

Measurements that decide design:

- Q7. Usable band and SNR ceiling of real radios, per path: mic/speaker, 9600 port, handheld speaker-mic jack. Sets M4 and M5 band edges and top rungs.
- Q8. How much do prefix and pilot thinning really buy once clock drift and flutter are in the simulator?
- Q9. Fixed tilt or bit loading on wide (M5 step 3)?
- Q10. Does symbol erasure plus the existing interleaver hold a mobile link at 20..50 Hz flutter, or do the codewords need to be longer in time?
- Q11. Wide at FS 16000 or 24000? 16000 keeps every sample count and reaches 6 kHz with a 2 kHz transition band. 24000 only matters if radios pass more than 6 kHz, which Q7 answers.
- Q12. Pi 4 decode budget at 20 kbit/s with a 0.3 s turnaround.
- Q13. Regulatory check on occupied bandwidth and symbol-rate limits for the wide mode on 2 m and 70 cm (US Part 97 at least). Not verified here.
- Q14. VARA FM's actual rates and thresholds on the same simulator, for a like-for-like target.

## Not planned

- A new waveform family for FM. OFDM over the audio or flat path is what both competitors use, and the existing modem covers it.
- Single-carrier FSK or constant-deviation modes for the weak-signal end (needs DC-coupled audio, see M5).
- Interoperation between HF and FM profiles, by the brief.
