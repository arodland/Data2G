# data2g-bulk: one-way text transfer

Status: proposed 2026-10-04, measured (§6). Implemented in `data2g/bulk.py` (`data2g-bulk`), outside data2g-host.

TLDR: every block goes twice, a burst apart: once at RV 0 among a burst's new blocks, then
at RV 1 in the next burst. Bursts go back to back with exact timing. Once the receiver has
read one control, it knows where every burst starts, so a lost control or even a lost PHY
header doesn't lose the burst. Each codeword is its own deflate stream, so a loss never
cascades.

- Results (§6): the copy moves the 10% block-loss point 2.4 dB (qpsk-r1/5) to 6.5 dB
  (qpsk-r3/4) on mpp/mps. qpsk-r1/2 with copies (58 B per codeword of airtime) reaches
  plain qpsk-r1/5's knee (46 B) at 26% more throughput.
- rv1 copies over Chase copies: +0.3 to +1.9 dB, from code-level AWGN (§3).
- In modes of rate 1/2 and up, an RV 1 copy can't decode alone. Header-less reception
  (§4) covers a lost header. A burst wiped out whole needs a second pass.

## 1. Use

```sh
data2g-bulk send notes.txt -o notes.wav [-m qpsk-r1/2] [--blocks-per-burst 15] [--passes 1]
# play notes.wav into the rig (VOX, or rigctl T 1 / T 0 around aplay)
data2g-bulk recv rec.wav -o notes.txt
data2g-bulk listen [--input-device NAME] [-o notes.txt]    # live, text on stdout as it arrives
data2g-bulk listen --list-audio-devices
data2g-bulk send notes.txt -o notes.wav -m fsk32r62-r1/2     # CPM: 4 blocks per burst by default
data2g-bulk listen -m fsk32r62-r1/2                          # listen for one mode only
```

`-m` on `recv` and `listen` listens for that mode alone. Listening for everything, the OFDM
search read a fsk32r62 burst as an n10 header at 10 dB, and the false lock hid the next
burst (timing recovered it, but only after the next control).

`listen` captures through data2g-host's PortAudio input and decimator (`--sample-rate`, a
multiple of 8000, default 48000). It prints the current stream's text in block order, each
block once it is final: decoded, or `[block k lost]` once the burst carrying its copy has
been handled. Stream starts and ends go to stderr. `-o` keeps the whole text in a file,
rewritten as blocks arrive, so a later pass can fill a gap the printout already passed.

DD cap (`--dd-cap`, default 0.25): DD starts no refine past that share of a burst's airtime,
counted from the burst's decode start (`phy.ModemRx`'s budget). A burst then can't take
longer to decode than about a quarter of its own airtime plus the plain decodes, so the
receiver keeps up. `recv` takes the same option, no cap by default.

The receiver writes the text with each run of lost blocks marked
`[... blocks i-j of n lost ...]`, and prints how many blocks needed their copy.

## 2. Blocks

- One block is one codeword: `codes.payload_bytes(mode)` bytes.
- Each block is a raw-deflate stream of its own. `frames.deflate_fit` with an empty history
  takes the longest text prefix whose deflate against the ARQ's zdict fits. If none does,
  the block is a stored deflate block (5 bytes of header). Inflate reads either. No flag is
  needed, and no compression state spans codewords.
  - Gain on `docs/arq.md`: 1.33x at 22 B blocks, 1.41x at 46 B, 1.51x at 116 B, 1.71x at
    236 B.
  - Why not a comp flag in the mask, as ARQ does: a block whose rv0 failed is combined
    under one mask. Trying a second mask would add the same soft bits twice.
- Block identity is the CRC mask, `phy.mask_value((stream, i >> 7, i & 127))`. It costs no
  payload bytes, and a codeword can't be credited to the wrong block or stream. The last
  byte stays under 128: CPM's receiver takes 128 and up as a control slot's.
- Stream id: 16 bits of CRC-32 over the blocks. Sending the same text in the same mode
  again gives the same id, so a second sending combines with the first.
- At most 32767 blocks: about 2 MB of text at qpsk-r1/5 or fsk32r62-r1/3, more in faster
  modes.

## 3. Bursts

Pass p, burst b, with h new blocks per burst:

| slot | contents | RV |
|---|---|---|
| 0-3 (CPM: 0-1) | control: version 2, stream, b, blocks, h, p (9 B), mask `(0xB17C, 0, 128)` | 0, 1, 2, 3 |
| next | blocks [b h, (b+1) h) | 2p mod 4 |
| then | blocks [(b-1) h, b h), burst b-1's again | 2p+1 mod 4 |

- Burst 0 has no copies. The pass ends with one burst of copies only.
- Further passes rotate RVs (2/3, then 0/1). The receiver's store keeps soft bits across
  passes.
- `codes.spread` deals each codeword's symbols across the whole burst. The two sends of a
  block are a burst apart, 40 s at h=15 in the 2400 Hz modes: well past a fade's
  correlation time on every CCIR preset.
- The control goes in 4 slots at RVs 0-3, combined like any IR resend.
  - Reason: with one control slot at the data's rate, qpsk-r1/2 on mpp heard no stream at
    -2 dB (0 of 40 bursts), though the copies would have delivered every block there. The
    receiver must find a stream before it can combine anything.
  - Only one control in a transfer has to decode. Timing places every burst after that.
- Overhead: 4 control codewords per 2h+4, 12% at h=15, 6% at h=30 (the most, 64 codewords).

CPM modes:

- The control is the grid's own short polar codeword (20 B), twice: the CPM header allows
  one or two. A polar resend is a plain repeat (Chase), so the control is weaker than on
  OFDM. The study (§6) checks whether acquisition fails where blocks would decode.
- At most 8 data codewords a burst (`cpm.MAX_DATA`, the header's word set), so h is 1-4.
- Each codeword is contiguous on air (3.1 s in fsk32r62), not spread over the burst: a fade
  takes whole codewords, and the copy a burst later is the diversity.
- An RV 1 copy can't decode alone in fsk32r62-r1/2 (noiseless check), as in OFDM r1/2.

rv1 or a Chase (rv0) copy. Code-level AWGN, 200 codewords a point, the Es/N0 where 90%
decode:

| mode | rv0 alone | rv0 + rv0 | rv0 + rv1 | rv1 over Chase |
|---|---|---|---|---|
| qpsk-r1/5 | -3.4 | -6.4 | -6.8 | +0.4 dB |
| qpsk-r1/3 | -1.1 | -4.1 | -4.4 | +0.3 dB |
| qpsk-r1/2 | +1.1 | -1.7 | -2.7 | +0.9 dB |
| qpsk-r3/4 | +4.6 | +1.6 | -0.3 | +1.9 dB |
| w48-16qam-r2/3 | +3.0 | 0.0 | -1.1 | +1.1 dB |

- An RV 1 codeword decodes alone (noiseless) in qpsk-r1/5, r1/3, 16qam-r1/3 and n10/n4
  qpsk-r1/5, where the mother buffer wraps.
- It doesn't in any mode of rate 1/2 or more, nor in n4/n10 qpsk-r1/3: RV 1 is parity
  only there.
- So the copy is always RV 1. It gains most where it can't stand alone.

## 4. Timing

The transmitter sends bursts back to back, no gaps. Burst g's length follows from its
codeword count (`modem.burst_seconds`, equal to `phy.tx_audio`'s length to the sample), so
its start is a fixed offset from burst 0's.

- The receiver fits start = a + slope x offset(g) over the bursts it heard, the last 32 at
  most. The slope absorbs the sound cards' clock difference.
- A burst heard within 0.2 s of where burst g is due, of g's mode and codeword count, is
  burst g, even when its control failed.
- A burst whose PHY header was lost is received anyway, by `modem.receive(known=...)`: no
  preamble search, the header's known word as the coarse-CFO reference, the CFO carried
  from the last burst heard. The pilot-rate alias is then resolved around zero residual.
  The header symbols may be the very ones that faded, and the coarse estimate from them
  picked a wrong alias in the test.
- Only between bursts heard. Past a transfer's end the slots are noise, and storing their
  soft bits would dilute the blocks still waiting for a combine.
- CPM: a burst's length is its symbol layout's (`cpm.burst_seconds` counts the 10 ms ramps
  on top: 160 samples a burst too many). A header-lost CPM burst goes through
  `cpm.receive` with a lock made from the timing.
- CPM bursts have no lead-out silence. With the sender's clock 30 ppm fast, the next burst's
  front began 4 samples before the previous one's computed end, and the streaming
  receiver, trimming its buffer there, missed every second burst. It now keeps 50 ms
  (`tnc.Receiver.CPM_TAIL`) before a CPM burst's end. This is in the shared receiver, so
  data2g-host gets it too.

## 5. Not done

- **CPM control in data slots** (option b): if the 2-slot polar control limits acquisition,
  also send it in two LDPC data slots at RV 0/1, at the burst's two ends (h 4 -> 3).
- **CPM modes other than fsk32r62-r1/2** are supported but not studied.
- **Live transmit and rig control:** not wanted. `send` writes a WAV; play it with any
  player, keyed by VOX or by hand.
- **DD cap tuning:** 0.25 of airtime is a guess with headroom, not measured. The studies run
  uncapped, so their thresholds are the uncapped receiver's.
- **Header-lost bursts before the first control** are received once a control arrives,
  back to the stream's burst 0, from the last 180 s of audio. Bursts heard before it, control
  lost, are kept (16 at most) and placed then. A joiner mid-transfer starts where its audio does.
- **Native port:** Python only.

## 6. Results

`scripts/bulk_study.py`: 6 kB texts from `docs/arq.md`, h=15, whole transfers through
continuous fading (so the copy's time diversity is the channel's). Noise is against each
burst's peak, as `DATA2G_PEP_REF_DB=5`. ±50 Hz CFO, 20 ppm clock. 10 trials a cell, the
same seeds at every SNR. Output: `runs/bulk_study.csv`.

"First send" counts the blocks decoded from their RV 0 slot alone: what the mode would
deliver without copies, at twice the throughput (with the 4-slot control and timing still
in place, so a little optimistic for a plain scheme).

SNR (dB) where block loss reaches 10% and 1%:

| mode | chan | 10% copy | 10% first | gain | 1% copy | 1% first | gain |
|---|---|---|---|---|---|---|---|
| qpsk-r1/5 | mpp | -6.4 | -4.0 | 2.4 | -4.5 | -2.1 | 2.4 |
| qpsk-r1/5 | mps | -6.8 | -4.3 | 2.5 | -4.8 | -2.4 | 2.4 |
| qpsk-r1/3 | mpp | -5.0 | -1.4 | 3.6 | -2.8 | 1.3 | 4.1 |
| qpsk-r1/3 | mps | -4.8 | -0.9 | 3.9 | -4.1 | 4.9 | 9.0 |
| qpsk-r1/2 | mpp | -3.9 | -0.1 | 3.8 | -2.2 | 1.7 | 3.9 |
| qpsk-r1/2 | mps | -3.3 | 1.3 | 4.6 | -2.1 | 1.9 | 4.0 |
| qpsk-r3/4 | mpp | -0.4 | 5.7 | 6.1 | 0.0 | 7.3 | 7.3 |
| qpsk-r3/4 | mps | -0.8 | 5.7 | 6.5 | -0.1 | 7.6 | 7.7 |

At equal throughput (payload bytes per codeword of airtime, before the ~1.45x deflate):

| scheme | B/cw | 10% mpp | 10% mps |
|---|---|---|---|
| qpsk-r3/4 + copies | 88 | -0.4 | -0.8 |
| qpsk-r1/3 plain | 76 | -1.4 | -0.9 |
| qpsk-r1/2 + copies | 58 | -3.9 | -3.3 |
| qpsk-r1/5 plain | 46 | -4.0 | -4.3 |
| qpsk-r1/3 + copies | 38 | -5.0 | -4.8 |

- Copies at a higher rate match or beat the plain lower-rate mode of similar throughput at
  10%. At 1% they win clearly (qpsk-r3/4 + copies 0.0 dB on mpp, plain r1/3 +1.3; on mps
  -0.1 against +4.9): the copy's time diversity fills the fading tail.
- The gain grows with the mode's rate, as the code-level table in §3 predicts: IR adds
  fresh parity where the mother code has most to give.
- Headers lost and received at their known position: 0-36 per mode and channel over 120
  transfers, most in qpsk-r1/5 near its knee. Controls lost with the burst placed by
  timing: 4-23.
- Default mode qpsk-r1/2: 350 bps of payload at h=15 (40 s bursts), about 500 bps of English text.

### Every mode at h=15: 5% and 1% points

`runs/bulk_fine.csv`: 1 dB steps, 20 trials a cell (310-3933 blocks a cell), mpp and mps,
otherwise as above. Speeds are steady state (one pass, long text). Text bps uses each
mode's per-block deflate ratio on `docs/*.md` (1.28-1.64x). WPM is 5 characters a word.
SNR is in 2500 Hz, against the bursts' peak.

| band | mode | burst | payload bps | text bps | WPM | 5% mpp | 5% mps | 1% mpp | 1% mps |
|---|---|---|---|---|---|---|---|---|---|
| 1200 | qpsk-r1/5 | 39.8 s | 139 | 189 | 283 | -7.1 | -8.0 | -6.5 | -7.1 |
| 1200 | qpsk-r1/3 | 39.8 s | 229 | 322 | 483 | -5.2 | -5.6 | -4.9 | -4.6 |
| 1200 | qpsk-r1/2 | 39.8 s | 350 | 506 | 760 | -3.0 | -3.1 | -2.2 | -2.0 |
| 1200 | 16qam-r1/3 | 20.2 s | 451 | 634 | 952 | -0.4 | -0.9 | 0.0 | -0.2 |
| 1200 | qpsk-r3/4 | 39.8 s | 531 | 811 | 1216 | -0.2 | -0.5 | 0.4 | 0.0 |
| 1200 | 16qam-r1/2 | 39.8 s | 712 | 1165 | 1747 | 0.8 | 0.8 | 1.0 | 1.0 |
| 500 | n10-qpsk-r1/5 | 49.6 s | 56 | 71 | 106 | -9.7 | -11.0 | -8.4 | -9.2 |
| 500 | n10-qpsk-r1/3 | 49.6 s | 97 | 130 | 195 | -7.3 | -8.2 | -6.1 | -7.2 |
| 500 | n10-qpsk-r1/2 | 49.6 s | 145 | 201 | 302 | -5.3 | -6.6 | -4.1 | -5.2 |
| 500 | n10-16qam-r1/3 | 49.6 s | 191 | 269 | 404 | -2.8 | -3.2 | -2.2 | -2.2 |
| 500 | n10-qpsk-r3/4 | 49.6 s | 218 | 309 | 463 | -1.8 | -3.5 | -1.0 | -1.0 |
| 500 | n10-16qam-r1/2 | 49.6 s | 293 | 426 | 639 | -1.4 | -0.7 | -1.0 | -0.1 |
| 500 | n10-16qam-r2/3 | 49.6 s | 394 | 596 | 893 | 0.0 | 0.8 | 0.8 | 1.7 |
| 500 | n10-16qam-r3/4 | 49.6 s | 445 | 688 | 1032 | 1.0 | 2.2 | 2.4 | 3.7 |

- 1200 Hz 16qam-r1/3 is dominated: qpsk-r3/4 is 18% faster for 0.2-0.4 dB at 1%.
- n10-qpsk-r3/4 is dominated on mpp: n10-16qam-r1/2 is 34% faster at the same 1% point.
  On mps the 16qam mode costs 0.9 dB.
- 5% to 1% is 0.2-1.5 dB in every mode but n10-qpsk-r3/4 on mps (2.5 dB): the copy takes most of the fading tail away.
- Losses come a burst's worth at a time, so 1% points from 300-700 blocks a cell (the
  16qam modes) carry about 0.5 dB of noise.


### Every mode at h=2

`runs/bulk_h2.csv`: as the h=15 table, with 2 new blocks per burst (8-codeword bursts, half
of them control). Speed is 53-55% of h=15's. Last column: the 1% point's shift from h=15
(positive: h=2 needs more SNR).

| band | mode | burst | payload bps | text bps | WPM | 5% mpp | 5% mps | 1% mpp | 1% mps | 1% vs h=15 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1200 | qpsk-r1/5 | 9.8 s | 75 | 102 | 153 | -7.1 | -6.5 | -6.2 | -5.4 | +0.3 / +1.7 |
| 1200 | qpsk-r1/3 | 9.8 s | 123 | 174 | 260 | -5.1 | -4.4 | -4.3 | -3.1 | +0.6 / +1.5 |
| 1200 | qpsk-r1/2 | 9.8 s | 188 | 273 | 409 | -3.1 | -2.4 | -2.3 | -1.5 | -0.1 / +0.5 |
| 1200 | 16qam-r1/3 | 5.2 s | 232 | 326 | 489 | -0.6 | 1.0 | 0.1 | 2.7 | +0.1 / +2.9 |
| 1200 | qpsk-r3/4 | 9.8 s | 286 | 437 | 655 | -0.4 | 0.4 | 0.1 | 1.5 | -0.3 / +1.5 |
| 1200 | 16qam-r1/2 | 9.8 s | 383 | 628 | 941 | 0.9 | 2.5 | 1.8 | 3.7 | +0.8 / +2.7 |
| 500 | n10-qpsk-r1/5 | 12.2 s | 30 | 38 | 58 | -10.2 | -10.5 | -9.3 | -9.3 | -0.9 / -0.1 |
| 500 | n10-qpsk-r1/3 | 12.2 s | 52 | 71 | 106 | -8.2 | -8.3 | -7.1 | -7.2 | -1.0 / 0.0 |
| 500 | n10-qpsk-r1/2 | 12.2 s | 79 | 109 | 164 | -5.7 | -6.2 | -4.3 | -5.1 | -0.2 / +0.1 |
| 500 | n10-16qam-r1/3 | 12.2 s | 104 | 146 | 219 | -3.2 | -3.7 | -2.1 | -2.3 | +0.1 / -0.1 |
| 500 | n10-qpsk-r3/4 | 12.2 s | 118 | 168 | 251 | -2.3 | -3.1 | -1.0 | -1.8 | 0.0 / -0.8 |
| 500 | n10-16qam-r1/2 | 12.2 s | 159 | 231 | 347 | -1.2 | -0.9 | -0.7 | 0.1 | +0.3 / +0.2 |
| 500 | n10-16qam-r2/3 | 12.2 s | 214 | 323 | 485 | 0.6 | 1.6 | 1.2 | 3.7 | +0.4 / +2.0 |
| 500 | n10-16qam-r3/4 | 12.2 s | 241 | 373 | 560 | 1.8 | 3.6 | 2.6 | 5.0 | +0.2 / +1.3 |

- mpp: h=2 costs -1.0 to +0.8 dB at 1%, inside the noise for most modes.
- mps (slow and selective): the 1200 Hz modes lose 0.5-2.9 dB, most where bursts are
  shortest (16qam-r1/3, 5.2 s). The copy comes one short burst later and shares more fades.
- 500 Hz up to n10-16qam-r1/2 loses nothing at h=2, on either channel. n10-16qam-r2/3 and
  r3/4 lose 1.3-2.0 dB on mps.
- Header-less receive does far more work at h=2: up to 4338 bursts in a mode (n10-qpsk-r1/5
  on mpp), against 575 at h=15, as many more headers go out near the knee.

### fsk32r62-r1/2 (CPM)

`scripts/bulk_study.py --only-mode --h 4`: as above, receiver listening for fsk32r62 only,
-16..+4 dB in 1 dB steps, 20 trials a cell, mpp, mps and mpg. Output: `runs/cpm_fsk32.csv`.

- Speed at h=4: 30.4 s bursts, 61 bps of payload, about 84 bps of text (1.38x), 126 WPM.
  fldigi's MFSK32 is about 120 WPM.

SNR (dB) where block loss reaches 5% and 1%, and where the last loss goes:

| chan | 5% copy | 5% first | 1% copy | 1% first | no loss |
|---|---|---|---|---|---|
| mpp | -11.3 | -7.2 | -10.3 | -6.1 | -8 |
| mps | -9.4 | -4.6 | -7.8 | -2.1 | -4 |
| mpg | -9.2 | -4.5 | -7.4 | -1.5 | -4 |

Against the OFDM modes (h=15, the same study's conventions), 1% points:

| mode | width | payload bps | mpp | mps |
|---|---|---|---|---|
| fsk32r62-r1/2 | 2300 Hz | 61 | -10.3 | -7.8 |
| n10-qpsk-r1/5 | 500 Hz | 56 | -8.4 | -9.2 |
| qpsk-r1/5 | 1200 Hz | 139 | -6.5 | -7.1 |

- The 2-slot polar control is enough: a transfer heard nothing only at -14 dB and below,
  where over 80% of blocks are lost anyway.
- On mpp fsk32 is best by about 2 dB. On slow fading (mps, mpg) its curve is shallow: an
  occasional block down to -16 dB, but 1% only at -7.5 to -7.8. n10-qpsk-r1/5 is 1.4 dB
  better on mps at the same speed, in a quarter of the width.
- Cause: a CPM codeword is a contiguous 3.1 s of tones, so a slow fade takes it whole, and
  the copy 30 s later is its only diversity. OFDM spreads each codeword over the whole burst.
  Interleaving the data codewords' tones across the burst (bulk only, control and header
  untouched) should close most of the gap. Not built yet.
