# Data2G FM: plan

Status: draft for discussion, 2026-11-13. Nothing here is built.

## TLDR

- Build one FM channel simulator first. Every later step is judged on it, and nothing in the repo models a limiter-discriminator today.
- "Narrow" is a retune of the existing 8 kHz OFDM (w48 already sits at 350..2700 Hz). "Wide" is the same modem clocked at 16 kHz, then widened. Neither needs a new waveform family.
- The structural cost is making `config.FS` and the mode tables a per-profile choice (HF or FM) in Python and native. Decide how before any FM mode is frozen (question Q1).
- Rate targets are VARA FM 4.0's: 12750 bps narrow, 25210 bps wide. Both are reachable with 55 carriers and a 256-point r5/6 top rung.
- Four decisions are the user's: Q1 (profile mechanism), Q2 (how low the floor goes), Q3 (digipeating), Q4 (VARA FM host dialect). The rest are measurements.

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

VARA FM 4.0, from the author's "VARA FM 4.0 Features" note (speed chart on its last page). Symbol rate 42 on every level. "Net" is my arithmetic: net rate over carriers x 42 x bits per symbol, so it folds code rate and pilots together.

| Level | Wide carriers | Wide mod. | Wide bps | Narrow carriers | Narrow mod. | Narrow bps | Net |
|---|---|---|---|---|---|---|---|
| 1 | 14 | 4PSK | 566 | 14 | 4PSK | 549 | 0.48 |
| 2 | 29 | 4PSK | 1188 | 29 | 4PSK | 1181 | 0.49 |
| 3 | 58 | 4PSK | 2390 | 58 | 4PSK | 2390 | 0.49 |
| 4 | 98 | 4PSK | 4040 | 58 | 4PSK | 3188 | 0.49 wide, 0.65 narrow |
| 5 | 98 | 4PSK | 5387 | 58 | 8QAM | 4252 | 0.65 wide, 0.58 narrow |
| 6 | 98 | 8QAM | 7185 | 58 | 16QAM | 5668 | 0.58 |
| 7 | 98 | 16QAM | 9580 | 58 | 32QAM | 7087 | 0.58 |
| 8 | 116 | 16QAM | 11340 | 58 | 64QAM | 8505 | 0.58 |
| 9 | 116 | 32QAM | 14144 | 58 | 64QAM | 9567 | 0.58 wide, 0.65 narrow |
| 10 | 116 | 64QAM | 16932 | 58 | 128QAM | 11162 | 0.58 wide, 0.65 narrow |
| 11 | 116 | 64QAM | 19003 | 58 | 256QAM | 12750 | 0.65 |
| 12 | 116 | 128QAM | 22102 | | | | 0.65 |
| 13 | 116 | 256QAM | 25210 | | | | 0.65 |

What the chart says about its design:

- Three coding points only (net about 0.49, 0.58, 0.65). The ladder is built from constellation size and carrier count, not code rate. Its top is 256QAM at about rate 2/3, not a high-rate code.
- The bottom is reached by using fewer carriers (14 is about 650 Hz), which puts the whole deviation budget into them and, per the note, picks the part of the audio band the radio passes best. Nothing below 4PSK at about rate 1/2: the floor is 549 bps.
- 58 carriers at 46.875 Hz is 2.7 kHz and 116 is 5.4 kHz. That fits the unverified blog figures of about 470..3140 Hz narrow and 280..5670 Hz wide. The note confirms narrow was shifted up 200 Hz because some radios are poor at low voice frequencies.
- Wide has an intermediate width (98 carriers, about 4.6 kHz) for four levels, so it does not assume every 9600 port passes 5.7 kHz.

Other features the note lists, each a user expectation:

- Variable TX delay per radio. Baofeng squelch-tail elimination must be off.
- Receive equalisation for handhelds whose receive audio takes time to settle after transmitting.
- Level switching tuned on a 117 KB "N5TW standard file" sent both ways. Use the same file size as the session benchmark.
- S/N reported to the host and the log. A monitor log of all connections heard on the frequency.
- Sessions through voice repeaters.
- Digipeating through one or two hops, by a callsign distinct from the gateway's, with a retries setting. The digipeater runs freestanding and the gateway is off while it relays.
- Autotune: the drive level is set against a remote station, which reports S/N and VU, with a pass/fail verdict.
- Ping: S/N and VU at both ends, direct or through digipeaters (the minimum along the chain). VU under -40 dB is flagged as too weak.

Data2G today: w48 tops out at 8222 bps on paper (256-point r5/8, 350..2700 Hz). That is above VardeFM narrow's 7.4 kbit/s and at VARA FM narrow level 7 to 8. It has no mode wider than 2400 Hz and `FS = 8000` caps audio at 4 kHz.

Targets this sets (rates only; the SNR each needs is still M0's job):

- Narrow: 12750 bps. With about 55 carriers on the 50 Hz grid (450..3150 Hz) that is 6.7 bits per data symbol at today's prefix and pilot overhead (256-point r5/6), or 5.6 bits (256-point near r0.7) with a 16-sample prefix and pilots 1 in 12.
- Wide: 25210 bps. With about 55 carriers on a 100 Hz grid (300..5700 Hz) the same two cases give 25.5 and 30.6 kbit/s at 256-point r5/6.
- Both bands are 55 carriers, so one pilot and one header layout can serve narrow and wide.

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
- Output: where each w48 mode fails against CNR, the SNR ceiling per radio path, and the CNR each VARA FM level needs if a licensed copy is at hand (the VARA Wine harness exists). Its rates are known (chart above); its thresholds are not.
- Depends: M1 for the simulated half. The on-air half can start now.
- This sets the targets for M4 and M5. Until then "match VARA FM narrow" is a guess.

### M1. FM channel simulator (`data2g/fmchannel.py`, about 300 lines)

- Audio in, audio out, at any FS. Inside: optional pre-emphasis, limiter and splatter filter (mic path) or flat (9600 path), FM modulator at complex baseband, IF filter (12.5 or 25 kHz channel), flat Rayleigh/Rician fading at a Doppler, AWGN at a CNR, RF offset, limiter-discriminator, optional de-emphasis and 300 Hz high-pass, sample-clock offset.
- Knobs for the ugly parts: key-up transient, receive audio settling after the station's own transmission (VARA FM added equalisation for it), squelch gate with an opening delay, squelch-tail burst, CTCSS leak, mic-input AGC/compressor, weak low-frequency response, a repeater (two hops with a clipper between).
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
  - Band edges. VARA FM moved up to about 470..3140 Hz to avoid poor low-frequency response; VardeFM stops at 2672 Hz. Candidate: 55 carriers, 450..3150 Hz. A band that some radios cannot pass is worse than a narrower one, so the top modes may need a width fallback like VARA's wide 98/116 split.
  - Cyclic prefix. 32 samples now; 8..16 may do. Worth up to 15% rate.
  - Pilot density. One symbol in six was sized for 2 Hz Doppler. A static channel needs the preamble plus sparse tracking of clock drift; flutter is handled by erasure. Worth up to 15%.
  - Preamble repeats (8 now) and header airtime, for short overs.
  - Top rungs: 256-point at r0.7..r5/6 to reach 12750 bps (targets above). VARA stops at 256QAM near r2/3, so the radios' SNR ceiling probably sits around there; 1024-point is unlikely to pay.
  - Bottom rungs: VARA's are fewer carriers, not lower rates. The existing w (1200 Hz) and n10 (500 Hz) QPSK modes play the same part and already go lower (128 bps against 549).
  - Clip headroom per mode, re-picked under a peak-deviation limit with pre-emphasis and the radio's limiter in the loop.
  - TX tilt, so the pre-emphasised signal uses the deviation budget evenly.
- Drop the n4 and CPM families from the FM profile unless Q2 says otherwise. Keep w and n10 QPSK rungs as the floor and for ACKs.
- Depends: M1, M3 (the erasure changes which rungs survive flutter), M0 for targets, M2 to freeze.

### M5. Wide ladder (9600 port or direct taps)

- Step 1, almost free: run the whole modem at FS 16000. Every sample count stays, so every table, pilot and header code is reused. Carrier spacing becomes 100 Hz, symbols 12 ms, prefix 2 ms. w48 lands on 700..5400 Hz and its ladder doubles: 16.4 kbit/s at the top. That already clears 10 kbit/s.
- Step 2: a wider band on the same grid, about 300..5700 Hz (55 carriers, the count M4's narrow band uses, so its pilot and header layout are shared). Arithmetic: about 19 kbit/s with today's top code, 25.5 kbit/s with 256-point r5/6, 30.6 with M4's prefix and pilot savings. VARA FM's top is 25210.
- A narrower wide band (about 4.6 kHz, as VARA's 98 carriers) for ports that roll off early: on this grid it is just fewer carriers.
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

- Deviation tune: a tone and a modem-like test burst, level meters, and a report of the far end's reading. VARA FM's Autotune steps the drive level against a remote station and gives a verdict with S/N and VU; users will expect the same. `sounding.py` is the starting point.
- Ping: S/N and input level at both ends without opening a session, through digipeaters if M8 has them.
- TX delay per station, with guidance for handhelds (squelch-tail elimination off).
- Overdrive diagnostic: the signature seen on 2026-10-31 (OFDM SNR pinned near the PAPR while FSK reads 25 dB or more) as a plain warning to the operator.
- Radio path setting: mic or flat, which picks narrow or wide and the TX tilt.
- PTT: `native/core/rig` has no CM108/CM119 GPIO keying that I could find. Most FM sound-card interfaces (AIOC, DigiRig Lite, DRA, Masters Communications) use it or serial RTS. Confirm and add.
- Depends: M2 for the UI. Otherwise independent, and useful to today's users on FM.

### M8. Host and network features

- VARA FM host dialect so Winlink Express and Pat treat the modem as VARA FM (Q4). Known differences to check: `CONNECTED` carries NARROW or WIDE, no BW commands.
- Digipeating (Q3). VARA FM's shape: one or two hops named in the connect, a digipeater callsign separate from the gateway's, a freestanding relay with a retries setting. A link-layer feature with ARQ consequences (who acknowledges, timing across hops). Its own design doc if wanted.
- S/N reports to the host, and a log of stations heard.
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
- Q2. How low does the FM floor go? VARA FM stops at 549 bps (4PSK on 14 carriers). The CPM floor reaches SNRs FM audio never delivers with a usable carrier, and squelch hides them anyway. Recommendation: keep the w and n10 QPSK rungs (down to 128 bps), no CPM, no n4.
- Q3. Digipeating: in scope, and how many hops? VARA FM does two and sells it as the disaster fallback for gateways. VardeFM does not have it.
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
- Q14. VARA FM's thresholds per level on the same simulator, for a like-for-like target. Its rates are in the chart above.

## Not planned

- A new waveform family for FM. OFDM over the audio or flat path is what both competitors use, and the existing modem covers it.
- Single-carrier FSK or constant-deviation modes for the weak-signal end (needs DC-coupled audio, see M5).
- Interoperation between HF and FM profiles, by the brief.
