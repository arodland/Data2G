# Analog slots: client-defined latents over Data2G

Status: draft, 2026-10-18. Not started; nothing here is decided beyond what
"Agreed" marks.

TLDR: Data2G carries bursts of real values for a client, instead of
codewords. It owns sync, a digital control codeword, pilots, equalization,
clipping and framing. The client owns the meaning (a neural codec's
latents) and its model. Both directions carry Gaussian messages: values
with a weight per value.

- Transmit: values and importance. Importance steers what only Data2G can
  do: where clip distortion lands, which carriers a value rides.
- Receive: values in the sender's units and their precision (per-value SNR),
  the analog counterpart of a digital slot's LLRs.
- Open: the header (a new analog format, told apart by keying or its own
  preamble), how the receiver learns the sender's importance profile, and
  whether Data2G or the client combines resends.

## Why

Data2G's own traffic is lossless, long-block and has an ARQ return path.
Separate source and channel coding is near optimal there. A neural codec's
latents (SSTVAE's pictures, FreeDV RADE's speech) are where a joint
approach wins: noise adds to a latent instead of breaking a bitstream, so
quality falls gently, and one transmission serves every receiver at its
own SNR (broadcast groups).

Putting that inside Data2G, instead of beside it, gives such a codec:

- Data2G's acquisition, header-less timed reception and identity. SSTVAE's
  README puts its cliff at acquisition, below about -2 dB on fading.
- A feedback channel: the receiver reports what arrived, and the sender
  refines what was weak (deep JSCC with feedback, Kurka and Gündüz).
- Broadcast groups and sessions, for free.

It keeps the model out of Data2G. A client brings its own (SSTVAE's ~40 MB
checkpoint, RADE's), and Data2G stays a modem.

Example clients:

- Pictures in SSTVAE's format, with hybrid ARQ: repeat or refine the latent
  groups that arrived weakest.
- Voice messages with RADE-style latents.

## Division of labour (agreed)

| Data2G | Client |
|---|---|
| Preamble, header, the burst's control codeword (session or group, sequence, layer) | What the values mean |
| Pilots, channel estimate, equalization | Encoder and decoder (its model) |
| Normalization, clipping, TX filter | Importance per value |
| Placement and burst-wide interleaving | Burst size and band (rate choice) |
| Per-value precision on receive | What a resend carries (repeat or refinement) |

## The API

One burst each way, as `(values, weights)` arrays, plus the client's key
(group or session) and a client-chosen id for the burst's content.

### Transmit: values and importance

If importance only scaled amplitudes it would collapse into the values: the
client can pre-scale itself, and the receiver couldn't undo an allocation
it doesn't know. It earns its own field only for what the client can't do:

- **Clipping.** Push clip distortion onto low-importance values, as ACE
  pushes it onto 16-QAM's outer points. The strongest reason: Data2G owns
  the clipper and its headroom.
- **Placement.** Important values on carriers and symbols with the best
  estimates (near pilots, mid-band).
- **Resends.** What a repeat should favour, if Data2G does the combining.

### Receive: values and precision

Each value equalized and back in the sender's units, with its precision:
per-value SNR, `|h|^2 p / (n0 + mse)` from the equalizer, as
`phy._measure_ofdm` reads it today.

The symmetry's wrinkle: if importance changes what goes on air, precision
must reflect it. Clip noise lands unevenly, so a flat `n0 + mse` overstates
the precision of de-emphasized values. Two ways:

- The receiver knows the sender's importance profile: fixed per client, or
  a few bits in the control codeword. Probably enough for SSTVAE and RADE,
  whose importance is a fixed function of latent position.
- Data2G measures clip noise per importance class and reports it, as
  `clip_ratio` does per submode today.

## Burst format

- **Header (later):** a new header format for analog bursts, told apart from
  digital ones by keying or by its own preamble. Today's header has no room:
  n10's 5|5 split (`config.CW_BITS`) is already tight.
- **Control:** digital, and coded for the burst's worst case, not the
  payload's. Analog values are only worth anything where control still
  decodes. Pinned padding (`phy.PIN_CUTS`) and `ARQ_DUP` carry over.
- **Payload:** the same OFDM numerology as SSTVAE's (the waveforms match
  sample for sample, `tests/test_sstvae_golden.py`): frames of 1 pilot + 5
  data symbols, values on carriers as I/Q pairs.

## Hybrid ARQ

- No CRC on analog values, so no "decoded". The ACK carries measurements
  (per-segment SNR), and the sender decides: repeat the weakest values,
  send the next refinement layer, or stop.
- Combining is precision-weighted averaging, precisions adding, as soft
  bits add in `codes.combine`. Data2G could do it under the client's key
  (as `ModemRx`'s store does); leaving it to the client is simpler and
  allows refinements that aren't repeats.

## Accepted limitations

- Interleaving spans one burst, not a whole picture or message.
- The gear shifter's outcome model predicts decode probability, which means
  nothing for analog. The client picks band and size; Data2G reports SNR
  and spread.
- Analog values can't be decision-directed pilots. A mostly analog burst
  gets DD from its digital slots only.
- Monitor mode shows "analog burst, group X, n values", not the content.

## Open questions

- The header: keyed or a separate preamble, and what it carries (band,
  value count, client protocol id).
- How the receiver learns the importance profile: fixed per client
  protocol, or signalled in control.
- Who combines resends: Data2G or the client.
- Whether the client's values must arrive bounded with stated statistics,
  or the clip headroom is part of the contract. SSTVAE's model was trained
  through its own clipper.
- The control codeword's code: fixed robust, or chosen per burst.
