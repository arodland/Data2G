# Hand burst: response from Data2G

The design record for the show of hands. What shipped: data2g/hand.py, native/core/hand, and the host commands in docs/arq.md §9 (which supersede the draft primitive below).

TLDR: the show of hands is feasible in data2g-host with no new dependencies. Keep the plan's 6.25 Hz 8-FSK ("if your rig works for FT8, it works for this"). Change five things:

- Start the Hand window about 3 s after `ANY TRAFFIC` ends, not 1.0 s.
- Use Data2G's existing CRC-aided polar code instead of FT8's LDPC(174,91), and trim the payload to 63 bits.
- Drop Hand-25. It is about 3 dB more sensitive in simulation, but 3.125 Hz is too narrow for rig drift over a 21 s transmission, and Hand-50 already has margin over traffic.
- Treat near-far as the main risk. On fading paths, a neighbouring Hand 15 dB or more stronger blocks a weak one completely (simulated, below). Order the channel map by strength, leave an empty channel at any jump of more than 5 dB, and keep strangers away from expected stations.
- Give expected stations that didn't answer a second round. On slowly fading paths a single Hand misses more often than its 10% point suggests.

## Why receiving is tractable

Data2G's receiver expects one burst at a time. A Hand round doesn't go through it. Net control knows each Hand's start (±0.6 s) and channel (a few Hz), so it gets its own windowed decoder:

1. Capture the window.
2. Search each channel's Costas sync over the known time and frequency span.
3. Compute soft bits, decode, check the CRC, measure the SNR.

That is an FT8 decoder with the blind search removed. Data2G already has continuous-phase M-FSK with Costas sync (its bottom rungs), so most of the parts exist.

Rough size of the change, each part written in Python and C++ with parity tests:

| Part | New lines per language, about |
| --- | --- |
| Hand TX: tones at a channel, CFO shift, timed start | 50 |
| Windowed multi-channel decoder | 200-250 |
| Host commands, frame timestamps, holding TX through a window | 150 |

## 1. Lead time: about 3 s, set by net control

A station learns its channel only after its host has decoded `ANY TRAFFIC` and passed it to the client, and the client has answered. On a Raspberry Pi 4, decoding alone can use most of a second. So 1.0 s is too tight.

- `ANY TRAFFIC` carries the delay D (suggest 3 s by default).
- The host stamps every frame it reports with the burst's end time (on the sound card's clock) and its frequency offset.
- The client then asks: send this Hand in channel c, at that burst's end + D, shifted by that offset.
- Each host starts its audio early by its own PTT and output latency, so the RF lands on time.
- Hosts won't know that latency well, so net control searches ±0.6 s, not ±0.25 s. Simulated with Hands landing anywhere within ±0.5 s:
  - The ±0.25 s search tops out at 63-65% decoded, even at high SNR.
  - The ±0.6 s search matches the ±0.2 s baseline on every channel, within 0.3 dB, with no more false decodes.
- Net control measures each Hand's start to about 20 ms while decoding it. It returns that offset in its signal report, so each host learns its true latency after one round.

## 2. Code: CRC-aided polar, and 63 bits of payload

Data2G has a punctured polar code with a list decoder, in Python and C++.

Payload changes (77 → 63 bits):

- Drop the 8-bit request tag. Seed the CRC with it instead. A Hand still can't answer another net, and the 8 bits go to the code.
- Cut version and spare from 9 bits to 3.
- Use CRC-16, not 14. Simulated false decodes are 2.6e-4 per attempt with net control trying 3 sync candidates × a list of 8. That is about 1% of rounds with one false hand among 40 channels. CRC-14 would be 4 times as many.

That makes 79 bits into the code, sent as 162 coded bits (rate 0.49). A fixed interleaver spreads them over the symbols, so a fade doesn't take out runs of neighbouring code bits.

Frame: `[Costas 6][data 27][Costas 6][data 27]`, 66 symbols, 10.6 s at 6.25 Hz. FT8 spends 21 of 79 symbols on sync. Two blocks are enough when time and frequency are already known to within a symbol and a tone.

## 3. Tone spacing: 6.25 Hz wins

I tried faster tones in a time × frequency grid, to keep the same number of hands per round with more Doppler tolerance. It loses. With the same power and the same hands per round, a faster Hand is shorter and carries less energy. Doppler spread doesn't cost enough to make up for that.

10% failure point, SNR in 2500 Hz, 300 trials per cell, same trial seeds at every SNR:

| Channel | 6.25 Hz (10.6 s) | 12.5 Hz (5.3 s, 2 slots) | 25 Hz (2.6 s, 4 slots) |
| --- | --- | --- | --- |
| Noise only | -19.2 dB | -16.1 dB | -13.0 dB |
| CCIR good (mpg) | -13.8 dB | -8.1 dB | -4.0 dB |
| CCIR poor (mpp) | -16.3 dB | -12.9 dB | -9.7 dB |
| CCIR disturbed (mpd) | -15.8 dB | -13.0 dB | -10.1 dB |
| Measured slow, selective (mps) | -14.1 dB | -9.7 dB | -5.6 dB |

The slow-fading rows (mpg, mps) favour the long Hand most, because a 10.6 s Hand rides through more fades. The weakest SNR in Data2G's traffic training data is -9.2 dB, so 6.25 Hz Hands have margin over the traffic they announce on every channel.

Hand-25 (3.125 Hz, 21.1 s in this frame) was simulated the same way, with a constant frequency offset and no transmitter drift:

| Channel | Hand-25, 10% point | Gain over 6.25 Hz | Neighbours +10 dB, 30 Hz pitch |
| --- | --- | --- | --- |
| Noise only | -22.1 dB | 2.9 dB | -21.5 dB |
| mpg | -18.2 dB | 4.4 dB | -16.0 dB |
| mpp | -19.1 dB | 2.8 dB | -17.4 dB |
| mpd | -16.8 dB | 1.0 dB | -11.0 dB |
| mps | -18.6 dB | 4.5 dB | -18.0 dB |

It is about 3 dB more sensitive, as its doubled energy predicts, except on disturbed paths (mpd, 2 Hz Doppler spread). There the gain shrinks to 1 dB, and a +10 dB neighbour costs 5.8 dB rather than 1.4. Up to +5 dB, neighbours cost under 1 dB, as at 6.25 Hz.

I'd still drop it. Its tolerance for drift across 21 s is half Hand-50's in hertz, and nothing here has tested drift, and it loses the "works if FT8 works" promise. Its sensitivity is also more than traffic needs. If a net needs more hands per round, a second round at 6.25 Hz gives that without the drift risk.

## 4. Near-far: the strength-ordered channel map is essential

Each wanted Hand had one neighbour on each side at the 60 Hz pitch. Each neighbour had its own timing, CFO and independent fading.

10% failure point in dB, 150 trials per cell. The neighbours' level is relative to the wanted Hand. "Never" means under 90% even at +6 dB.

| Channel | Alone | +0 dB | +5 dB | +10 dB | +15 dB | +20 dB | +20 dB, 120 Hz pitch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Noise only | -19.2 | -19.1 | -19.1 | -18.8 | -17.4 | never | -18.2 |
| mpg | -13.8 | -13.0 | -12.2 | -9.7 | never | never | -9.7 |
| mpp | -16.3 | -15.8 | -15.6 | -14.7 | -6.0 | never | -15.0 |
| mpd | -15.8 | -15.5 | -15.2 | -14.4 | -8.7 | never | -14.3 |
| mps | -14.1 | -13.8 | -13.5 | -12.0 | never | never | -11.4 |

What it shows:
- Neighbours up to 5 dB stronger cost under 1 dB (1.6 dB on mpg), even though they fade independently. So ordering the map by strength, as the plan does, works.
- 10 dB costs 1-4 dB. 15 dB or more is fatal on fading paths at the 60 Hz pitch.
- One empty channel between two Hands (120 Hz pitch) holds a 20 dB stronger neighbour to a 1-4 dB loss. A 75 Hz pitch isn't enough: at +20 dB it recovers the noise-only case (-16.3 dB) but not the fading ones.

So the channel map should:

- Order expected stations by the latest SNR heard, and leave an empty channel wherever the list jumps by more than 5 dB.
- Keep strangers away from expected stations. Their level is unknown, so put them in channels at the far band edge with an empty channel before them, or in their own short round.
- Fill a 500 Hz net (8 channels) the same way. Expect fewer than 8 hands per round when levels spread widely, and run a second round for the rest.

Later receiver upgrade: decode the strongest Hands first, rebuild and subtract them, then decode the rest, as WSJT-X does for FT8. It would relax the 5 dB rule. It isn't needed for the first version.

## 5. A second round for missed expected stations

On slow fading the 1% failure point sits 4-6 dB above the 10% point (mpg: -8.5 vs -13.8 dB). One Hand in ten close to the threshold can be lost to a single long fade. Net control knows who it expected, so a short second round for the missing ones costs one more `ANY TRAFFIC` and recovers most of them.

## The data2g-host primitive

Roughly:

- `HAND SEND ref channel payload`: transmit at the end of burst `ref` + D, shifted by that burst's offset.
- `HAND LISTEN D channels`: decode the window after the next burst this host sends. It reports `HAND channel payload snr offset` for each one decoded, then `HAND END`.
- While a window is open the host holds its own CQ and broadcast transmissions.

## How these were measured

The simulation is `scripts/hand_study.py` on the hand-burst-sim branch. Each trial puts the Hand within ±0.2 s and ±3 Hz of nominal, and the receiver searches ±0.25 s and ±5 Hz (the timing test: ±0.5 s and ±0.6 s). Fading is a two-path Watterson model (CCIR good, poor and disturbed), plus a slow, selective profile measured on a real 4000 km path. LDPC(174,91) itself wasn't simulated, so the polar code's gain over it is the literature's figure (roughly 0.5-1 dB), not measured.

## As shipped: nailing and round cost

The tables above come from the study script. The shipped decoder adds one step the script didn't: each of the 3 candidates is nailed in on a T/16 × rate/16 grid (±half a symbol, ±rate/16, maximizing the data rows' winning shares) *before* its soft bits are built. The search's T/8 step lands a quarter symbol off, which leaks every symbol's energy across two bins. That costs the SNR reading and the decoder alike. Measured with `data2g.hand.decode_window` itself, one channel, AWGN, ±31 ms landing jitter, 60 trials per SNR, same seeds both arms, SNR in 2500 Hz:

| SNR | candidates used as searched | nailed first |
| --- | --- | --- |
| −19.5 dB | 49/60 | 55/60 |
| −20.5 dB | 28/60 | 35/60 |

So the shipped decoder sits about 0.7-1 dB below the noise-only 10% point of −19.2 dB in that table, at the same code and rate. Nailing on the sync blocks' shares instead of the data rows' wins by a hair at −19.5 dB (56/60) but not at the threshold (29/60): 12 known symbols are too few to pick a quarter symbol there. Both metrics cost the same once the window is mixed once per CFO rather than once per candidate; the data rows' metric is what's used.

Fading was re-checked with the shipped decoder, same 24 seeds per cell, one channel, ±31 ms landing jitter, two-path Watterson through `hfchannel.fading` then AWGN at the stated SNR (the absolute points sit ~1-1.5 dB below this section's because the study searched ±0.2 s of landing spread against ±31 ms here):

| Channel | SNR | candidates used as searched | nailed first |
| --- | --- | --- | --- |
| mpg | −18.0 dB | 18/24 | 21/24 |
| mps (measured slow, selective) | −19.0 dB | 11/24 | 15/24 |
| mpd (disturbed, 2 Hz spread) | −19.0 dB | 4/24 | 3/24 |

It helps where the design leans on long Hands riding through fades (mpg, mps), and is a wash on the disturbed channel (one trial either way at n=24).

Cost of a round: the Python reference (mixing the window once per CFO with `hand.mix`/`hand.bins`, bit-identical to `cpm._energies`, rather than per candidate, plus the nailing) takes about 3 s for a 40-channel window on a desktop, where the branch's original form took 4.6 s. What net control runs is the native decoder (`native/core/hand`): 1.9 s single-threaded, ~1 s of wall with a batch's rows split across pocketfft's threads (about two cores). Four changes got it there from the ported form's 2.6 s, none of them a sensitivity change. A 40-round corpus of payloads, `snr` to 12 digits and `dt` is byte-identical before and after, and `tests/test_native_hand.py` keeps Python and C++ agreeing:

- the mixed window carries a margin of silence on both sides, so a symbol window that runs off either end of the capture reads zeros where it lies instead of being gathered into a zero-padded copy (1.35 MB copied per offset, ~6200 offsets a round);
- a whole row of search starts, and a whole row of nail offsets, goes through one strided block transform (`dsp::fft_block_into`) instead of one pocketfft call per symbol window, which is what lets its threads have anything to divide;
- nailing transforms only the 54 data rows: the 12 sync rows are thrown away by both the nail's metric and the decode, so a round is 10.8k symbol windows per channel where it was 15.2k;
- candidates are nailed and decoded one at a time, so a burst found on the first candidate never pays for the other two nails.

The decode still runs on its own thread after the capture, so it never holds the round; a Pi 4 remains the untested case.

What the search and nail grids would cost to cut, measured the same way (AWGN, one Hand, ±0.2 s landing and ±3 Hz, 100 trials per SNR, SNR in 2500 Hz; a round is the 40-channel wall time of the native decoder):

| Setting | −21 dB | −20 dB | −19 dB | round |
| --- | --- | --- | --- | --- |
| as shipped (3 candidates, T/8 search, T/2 × rate/16 nail grid) | 20/100 | 74/100 | 100/100 | 0.97 s |
| 2 candidates | 20 | 74 | 100 | 0.79 s |
| nail on CFO only (drop the ±rate/16 steps) | 19 | 66 | 98 | 0.60 s |
| nail ±T/4 rather than ±T/2 | 17 | 67 | 97 | 0.81 s |
| no nailing at all (the search's T/8 cell) | 17 | 61 | 93 | 0.54 s |
| search every T/4 | 19 | 73 | 100 | 0.90 s |
| search CFO to ±4 Hz | 21 | 74 | 99 | 0.91 s |

Two candidates and a T/4 search step are free: the third candidate is a near-duplicate of the first two (the distinctness rule already keeps them half a symbol apart), and the nailing recovers alignment that the coarser search left loose. Nailing is the only one that buys real margin (~0.5 dB, and the reported `dt` stays within a symbol either way: mean error 5 ms nailed, 8 ms not), and the CFO grid beyond ±4 Hz is margin against a spec that says Hands land within ±3 Hz. Nothing here changes what the shipped decoder does.

### --hand-round cheap

Three cuts combine into a preset a station can pick: 2 candidates, a T/4 search step, and the nail on the search's own CFO grid. `hand.ROUND_CHEAP` / `hand::ROUND_CHEAP` hold it, `hand.decode_window` / `hand::decode_window` take it as their last argument (the default is the shipped `ROUND_FULL`), the Engine holds the name in `hand_round`, and either host sets it with `--hand-round cheap`. Measured on this machine with a full 40-channel window: 1.03 s for the shipped grids, 0.48 s for the cheap ones.

What the combination costs, paired seeds (the same Hand, the same noise, both arms), one channel, ±0.2 s landing and ±3 Hz, SNR in 2500 Hz. Hands decoded out of n:

| AWGN, 100 trials | −21 | −20 | −19 | −18 | 50% |
| --- | --- | --- | --- | --- | --- |
| shipped | 23 | 75 | 97 | 100 | −20.5 |
| cheap | 20 | 65 | 95 | 100 | −20.3 |

| Fading, 24 trials | −20 | −19 | −18 | −17 | 50% |
| --- | --- | --- | --- | --- | --- |
| mpg shipped / cheap | 9 / 7 | 14 / 13 | 20 / 20 | 21 / 21 | −19.4 / −19.2 |
| mps shipped / cheap | 6 / 5 | 15 / 11 | 20 / 19 | 22 / 21 | −19.3 / −18.9 |
| mpd shipped / cheap | 2 / 0 | 8 / 7 | 15 / 16 | 22 / 21 | −18.4 / −18.4 |

(−16 dB was 24/24 for both arms on all three profiles.) 0.2 dB at AWGN's 50% point, 0.3 dB at the shipped decoder's 74/100 point, and 0.0-0.4 dB on the Watterson profiles, which is what 24 trials resolve, so read that spread as "under half a dB". Half the CPU for a third of a dB: the option is there for a station that runs a round every net period on a busy machine, and `full` stays the default. `tests/test_hand.py::test_hand_round_between_two_stations` runs the whole round at both settings, `test_hand_decode_at_the_sensitivity_floor` decodes a −18 dB Hand with the cheap grids, and `tests/test_native_hand.py` keeps C++ and Python agreeing on both to 1e-6.
