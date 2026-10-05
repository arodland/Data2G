# data2g-bulk: one-way text transfer

Status: proposed 2026-10-04. Implemented in `data2g/bulk.py` (`data2g-bulk`), outside data2g-host.

TLDR: every block goes twice, a burst apart: once at RV 0 among a burst's new blocks, then
at RV 1 in the next burst. Bursts go back to back with exact timing. Once the receiver has
read one control, it knows where every burst starts, so a lost control or even a lost PHY
header doesn't lose the burst. Each codeword is its own deflate stream, so a loss never
cascades.

- Results: §6. The copy moves the 1% block-loss point by NN dB on mpp.
- rv1 copies over Chase copies: +0.3 to +1.9 dB, from code-level AWGN (§3).
- In modes of rate 1/2 and up, an RV 1 copy can't decode alone. Header-less reception
  (§4) covers a lost header. A burst wiped out whole needs a second pass.

## 1. Use

```sh
data2g-bulk send notes.txt -o notes.wav [-m qpsk-r1/2] [--blocks-per-burst 15] [--passes 1]
# play notes.wav into the rig (VOX, or rigctl T 1 / T 0 around aplay)
data2g-bulk recv rec.wav -o notes.txt
arecord -r 8000 -c 1 -f FLOAT_LE -t raw | data2g-bulk recv - -o notes.txt   # live
```

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
- Block identity is the CRC mask, `phy.mask_value((stream, i >> 8, i & 255))`. It costs no
  payload bytes, and a codeword can't be credited to the wrong block or stream.
- Stream id: 16 bits of CRC-32 over the blocks. Sending the same text in the same mode
  again gives the same id, so a second sending combines with the first.
- At most 65535 blocks: about 4 MB of text at qpsk-r1/5, more in faster modes.

## 3. Bursts

Pass p, burst b, with h new blocks per burst:

| slot | contents | RV |
|---|---|---|
| 0 | control: version, stream, b, blocks, h, p (9 B), mask `(0xB17C, 0, 0)` | 0 |
| 1 .. | blocks [b h, (b+1) h) | 2p mod 4 |
| then | blocks [(b-1) h, b h), burst b-1's again | 2p+1 mod 4 |

- Burst 0 has no copies. The pass ends with one burst of copies only.
- Further passes rotate RVs (2/3, then 0/1). The receiver's store keeps soft bits across
  passes.
- `codes.spread` deals each codeword's symbols across the whole burst. The two sends of a
  block are a burst apart, 36 s at h=15 in the 2400 Hz modes: well past a fade's
  correlation time on every CCIR preset.
- Overhead: one control codeword per 2h+1, 3% at h=15.

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

## 5. Not done

- **CPM modes:** their control codeword differs. OFDM modes only.
- **Live audio and PTT:** the tool reads and writes audio files or raw streams. data2g-host
  has the PortAudio, rigctld and resampling code if a live mode is wanted.
- **Receive deadline:** decoding runs DD without a time limit (`ModemRx(..., None)`). A live
  receive at very low SNR may fall behind real time. A per-burst budget equal to the
  burst's airtime would bound it.
- **Header-lost bursts before the first control** are not recovered. Bursts heard before
  it, control lost, are kept (16 at most) and placed once a control arrives.
- **Native port:** Python only.

## 6. Results

PLACEHOLDER
