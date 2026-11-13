# PENDING from the header: a protocol change for later

Status: parked (2026-11-13). Do it at the next protocol version bump that
touches headers anyway. The receiver-side version without a protocol change
shipped as `--pending off|fsk|ofdm|all` (default `fsk`, engine.py
`PENDING_MODES`).

TLDR: give CONNECT bursts their own header word, and use the n_cw field
(always 1 for a CONNECT) to carry a hash of the destination call. PENDING
then fires only for a CONNECT that is probably to us. On OFDM this costs no
extra header bits.

## Why

VARA's PENDING tells a scanner to stay on a frequency because an incoming
connect may be arriving. The header carries only submode and n_cw
(`modem.header_bits`: 4|6 bits, 5|5 on n10, plus CRC-6). The frame type and
callsigns are in the body. So today's `--pending` guesses from "one codeword
in a connect mode while listening". It also fires on:

- ID frames and CQ frames in the same modes
- in-session control bursts of other stations in those modes
- other stations' CONNECTs
- false header locks

Each of these then needs a CANCELPENDING, and a scanner that holds on each
one loses scan time.

## Spare room (protocol 15)

| Sync band | Connect mode | Index field | Used |
|---|---|---|---|
| n10 | `n10-qpsk-r1/3` (cap 0) | 5 bits | 18 of 32 |
| w | `qpsk-r1/5` (caps 1, 2) | 4 bits | 11 of 16 |
| c16r25 (CPM) | `fsk16r25-r1/2` (ROBUST_CONNECT) | 4 bits (index + 2×dup) | 0-3 of 16 |

CPM headers are decoded ML over each grid's valid words, which is 36 today
(`cpm.valid_words`).

## Change

- One pseudo-submode index per connect mode, meaning "CONNECT in mode X".
  The receiver maps it back to X to decode the body. The body and
  `frames.pack_connect` are unchanged.
- In that word, the n_cw field carries `hash(dest call) mod 2^b`: 5 bits on
  n10 and 6 on w. The burst is always one codeword.
- On CPM, use 3-4 bits (8-16 extra valid words, not 64) so the ML header
  decoder's wrong-word rate stays near today's.
- Hash: any fixed hash of the upper-cased call, SSID included (the CONNECT's
  callee as sent). The host matches it against every registered MYCALL call.
- Bump `PROTOCOL_VERSION`. It seeds `crc6`, so old and new headers don't
  decode as each other.
- The host raises PENDING only on a connect word whose hash matches a
  registered call. CANCELPENDING and the backstop timer stay as they are now.

## Result

- No PENDING for ID, CQ or control bursts.
- Another station's CONNECT matches 1/32 (n10), 1/64 (w) or 1/8-1/16 (CPM)
  of the time.
- What's left: a matching CONNECT whose body fails, and a false lock that
  lands on a matching word.
- The `--pending` mode choice could then default to `all`, since the OFDM
  connect modes would no longer be noisy.

## Measure before shipping

- Wrong-header acceptance with the added words on n10, w and c16r25:
  `scripts/header_diversity.py` at MPD -3 dB, its worst cell.
- CPM header loss at the floor with 8 vs 16 extra words. Pick the hash
  width there: floor losses weigh most.

## Not proposed

- A dedicated flag bit: it costs header code rate. The spare indices do the
  same job for free.
- The full callsign in the header: far more bits than the header has.
- A separate CQ word: nothing needs it yet.
