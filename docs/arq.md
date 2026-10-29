# Data2G ARQ protocol

Status: the spec. Approved 2026-09-24, with the state-agreement rules of §2 and §10 added at review. Written
before the code as one step of the gear-shifter plan; the "As built" notes record where the code went
further.

TLDR: a connected, half-duplex, turn-based ARQ. Both directions use one burst format:
a control word first, then resent codewords, then new ones.

- **Reliability:** selective repeat per codeword, with incremental-redundancy (IR) resends.
- **Mode choice:** each station's receiver recommends the other's next mode and burst size.
- **No PHY header change:** everything rides in codeword payloads; the frozen 16-bit burst
  header is untouched.
- **No silent disagreement:** whatever the two stations must agree on is either sent
  explicitly or checked by every codeword's CRC (§10).

## Code map

data2g/arq/__init__.py lists the modules and how they layer. By section (Python in
`data2g/arq/`, native in `native/core/arq/` under the same names):

| section | where |
|---|---|
| §2 codeword identity, masked CRC | `burst.py` (`ctl_mask`, `data_mask`), `phy.py` (`mask_value`) |
| §3 burst layout, §4 core word, §5 extensions | `frames.py` (`Core`, `Control`, `T_*`); `link.py` (`Station.build`, `Station.handle`) |
| §5a inline codewords | `frames.py` (`T_INLINE`, `inline_check`); `link.py` (`Station.build`, `Station._take_data`); native also `monitor/` |
| §6, §6a turns, timers, idle | `session.py` |
| §7 session frames, §7a ID frames, §7b CW ID | `frames.py` (`Connect`, `ConnectAck`, `ConnectNak`, `pack_cq`, `pack_id`); `session.py`; `engine.py`; `cwid.py` |
| §8 gear shifter | `policy.py` (`GearShifter`); `predictor.py`; `modes.py` (the recommendation code) |
| §9 host interface | `data2g/host.py` (native: `native/core/host/`) |
| §9a stream framing, compression | `frames.py` (`to_records`, `deflate`); `link.py` (`TxSide`, `RxSide`) |
| §3a primed reception | `link.py` (`Station.expect`, `_primed`, `_implausible`, `expected_modes`), `phy.py` (`ModemRx.decode_known`), `session.py` (`_primed_frame`, `expecting`), `cpm.py` (`prime_lock`), `receiver.py` (`ReplySlot`, `Receiver.expect`), `engine.py` (`_expect_reply`) |
| §10 state agreement | `link.py` (`Station._check`, the watchdog) |

## 1. Terms

- **Station roles:** the ISS (information sending station) and the IRS (information
  receiving station). Both can carry data; the roles only name who holds more.
- **Turn:** one burst by one station. Turns alternate strictly while a session is active.
- **Codeword:** the modem's unit. Each has its own CRC, and is the unit of
  acknowledgement and resend.
- **seq:** codeword sequence number, 7 bits, per direction. The window is 63
  codewords outstanding, strictly less than half the space, as selective repeat
  needs. At 64, a cumulative ACK covering the whole window aliases to base - 64
  (found in linksim, 2026-09-24).

## 2. Session identity and codeword identity (zero overhead)

- **Masked CRC:** inside a session, each codeword's payload CRC is XORed with a mask,
  hash(caller, callee, 16-bit nonce, direction, seq). Control codewords use seq = 128 +
  burst seq.
- **Compression in the identity (2026-10):** a deflated data codeword (`T_COMP`, §9a)
  hashes direction + 2 in place of its direction (`link.data_mask(..., comp=True)`).
  The flag rides only in the control word, which no data CRC covers. A control
  accepted by CRC with a flipped bit (a 2^-16 false accept) used to deliver deflate as
  raw bytes, or raw as deflate. Now that codeword fails its CRC and is resent. A wrong
  belief that persists (the receiver keeps flags for resends that omit them) costs
  progress until the watchdog's resync re-slices (§10).
  - Chosen over a flag bit inside the payload: zero bytes on air, and the receiver
    already knows which mask to check from `T_COMP`.
  - Chosen over checking both masks: that doubles the false-accept rate on every
    data codeword and needs two CRC checks per slot.
  - A wire change: compressed codewords from an older build fail the CRC here. The
    session version was not bumped (development rule); it must be before release.
- **Slicing in the identity (2026-10):** a data codeword also hashes its sender's
  abandon epoch (§4) mod 64, in the direction byte's top 6 bits (`epoch=`). The
  receiver checks under the last epoch it applied. A late codeword from an older
  slicing (a reordered burst) fails its CRC instead of landing in the new one.
  - No cost but the format: still one mask checked per slot, so the false-accept
    rate is unchanged. HARQ combining never spans an abandon anyway (the receiver
    drops those soft bits when it applies one), so none is lost; for the same
    reason the soft-bit key needs no epoch. Wrapping needs a burst to arrive 64
    abandons late, and each abandon takes a burst of its own.
  - Same wire change, same unbumped version.
- **Filtering:** a codeword from another session, another station, the other direction,
  or decoded under a wrong seq, compression or slicing assumption fails its CRC. It is ignored
  like noise. Stream bytes can never land in the wrong place, whatever either side
  believes.
- **Before a session:** connect frames use mask 0.
- **Undetected-error rate:** a codeword decoded under a wrong assumption passes by chance
  with probability 2^-16 (CRC-16), or 2^-32 on LDPC codewords of k >= 512.
- `codes.encode` / `decode_many` / `decode_buffer` take `crc_mask`, from
  `arq.phy.mask_value(mask_id)`.
- **Scrambler:** info bits are XORed with PN9 before encoding (`codes.scrambler`).
  Without it, zero-padded codewords were wrecked by the clipper (found running whole
  sessions through the modem, scripts/phy_session.py).
  - The seed is the codeword's burst position alone (`codes.scramble_seed`), never the
    key, so seeds differ within a burst.
  - A resend in another slot is scrambled differently. It still combines, because the
    code is linear: C(u ^ s) = C(u) ^ C(s). The receiver flips each slot's soft bits by
    its own C(s) (`codes.flip`) before adding them to the buffer, and decodes the
    buffer unscrambled (`codes.PLAIN`).
- **Plain on air (Part 97):** the mask touches only the CRC. A station without the key
  decodes any codeword, descrambles it by its slot and reads the payload, with no
  trials. That is all a promiscuous receiver needs. Only the CRC check needs the key,
  so filtering stays (§2), but meaning is never obscured (FCC 97.113(a)(4)).
- **Decode once, check each mask:** a receiver decodes a slot once (`codes.decode_raw`)
  and tries each mask it cares about as a CRC check (`codes.check`; polar: the first
  list candidate that passes). The engine asks every burst's slot 0 under the session's
  key, KISS's and mask 0: three decodes became one (polar `ack-4f`: 56 to 19 ms).
  - Until 2026-10 the seed came from the mask. Reading traffic then needed the key,
    from the CONNECT or by trying all 65536.

## 3. Burst layout (both directions)

Codewords in burst order:

1. `C` control codewords. The first is fresh (sent once, redundancy version RV0), so it
   decodes on its own. It carries the 32-bit core word (4), and any extensions follow
   it (5).
2. `K` resends, in increasing seq order: the codewords the peer's last acted-on ACK
   reported missing, oldest first.
3. New codewords, consecutive seq from the `new` extension's start seq.

The PHY header already gives the submode and the total codeword count `n_cw`. So
new = n_cw - C - K. A burst with nothing to send is just its control codewords: that
is the "ACK burst".

**Codeword size:** the smallest submodes carry 4 bytes (the polar ACK modes). So the
core word is exactly 32 bits, and extensions use more codewords only when needed. On
large-codeword modes (46-396 B) the control codeword's spare bytes carry the
extensions, so C = 1. They can also carry the last data codeword of a write (§5a).

## 3a. Primed reception (receive only, 2026-10)

A station waiting for its peer's burst knows much of it: when it should start, what its
control will say, which modes it may come in. At the bottom of the ladder the CPM control
codeword and the CPM lock fail before the data does (forced fsk16r25-r1/3 sessions died on
them at awgn -19, mpp/mpd -17), so the receiver uses what it knows there. Nothing on air
changes; `DATA2G_PRIME=0` (`link.PRIME`) turns all of it off for A/B runs. Adapted from
ion-hf's primed reception (docs/net.md "Primed reception"; its author relicensed the code
for Data2G).

- **Primed controls** (`Station.expect`, `ModemRx.decode_known`). A control codeword that
  fails blind, as an `ARQ_DUP` pair and pinned on its padding (§3), is decoded again under
  each of up to `PRIME_TRIES` (6) hypotheses, likeliest first: a whole expected control
  stream and the bits of it known. Polar codewords only (CPM's control, the ACK and polar
  modes): their known bits are frozen (`codes.decode_raw_pinned`). An LDPC control gained
  0.0-0.3 dB at code level over its padding estimate, under a 16-bit CRC: not primed.
  - A hypothesis pays only when nearly every bit is known (a CPM control's 10% point moves
    2-5 dB with 7 bits unknown, under 1 dB with 50), and bits sitting late in the payload
    (weak polar positions) cost most: 13 unknown core bits cost 0.2 dB, the 6 of `T_REPLY`'s
    value 1.1-1.8 dB. So each is a whole control, built from my state and the peer's last
    control of the same kind (data or control only):
    - core: frame type (a caller's control-only burst may be a poll), `C-1` (CPM: 1; an
      OFDM control-only burst: its codeword count), burst seq (the peer's last + 1; unknown
      after my timeout or for a caller's poll), acted-on (my latest burst; a poll may act on
      the one before), `cum` (my `next` if nothing of mine is outstanding, else the bits all
      of `[base, next]` agree on), reply-lost (0 unless my latest went again), K (data: from
      what my latest ACK reported missing), recommend unknown, size hint as last time.
    - TLVs: the last control's `T_CHAT`, `T_DUPCTL`, `T_BUFFER` (value unknown) and
      `T_REPLY` (value as last time: 80% of them at the floor); data: `T_NEW` at the
      peer's next seq, `T_RV` sized for K (values unknown), `T_DRAINED` as last time.
      Before any: the gear shifter's (`T_REPLY`, `T_DUPCTL`).
    - The ways it may differ, in order: what arrived of my data (everything; a gap with a
      1-byte bitmap; a gap without), K and K+1 and 0 resends, then each with `T_DUPCTL` the
      other way, then with `T_REPLY` unknown. A callee whose header matches the caller's
      last burst tries that burst's control second (the caller repeated: my reply was lost).
  - Hit rates at the floor cells (shift+cpm sessions, the hypothesis' known bits all right):
    80-94% of controls within 9 tries, 76-81% within 6.
  - Accepted only if the masked CRC passes on one of the polar list's first
    `PRIMED_PATHS` (4) paths, its codeword scores at least `PRIMED_SCORE` (10) nats over
    random bits (`codes.score`), it is well-formed (`_check`), and it agrees with my state
    (`_implausible`: acted-on resolves, `cum` in `[base, next]`, an abandon at my
    cumulative). A blind control saying otherwise fails the session (§10); a primed one is
    only dropped. Then `rx_primed` counts it.
- **Connect frames** (`Session._primed_frame`). A caller waiting for its CONNECT_ACK knows
  all of it but the callee's cap and t_turn: tried with those as mine, then unknown. A
  listener knows a third of a compact CONNECT (the version, its own call), then a cap and
  t_turn like its own: two tries. A forged CONNECT also has to unpack its caller's 10
  characters from a 40-of-64 alphabet (about 7 bits more). At the forced floor most
  sessions never connected.
- **The expected slot** (`Receiver.expect`, `cpm.prime_lock`, `ReplySlot`). After my burst
  ends the engine tells the receiver where the peer's reply should start: the median of
  its last 8 starts past my burst's end (`ReplySlot`, `TOL_S` 0.15 either side; learned
  online, so PTT, sound card and decode latencies need only be stable), the CPM grids my
  session expects it in (`Session.expecting`: my latest burst's mode and the two I
  recommended), and the peer's CFO from its last burst. Once the audio holds every second
  header copy of a start in the window and nothing locked, `prime_lock` looks there once:
  every start at T/4, CFO +-R/16, any valid word, scored by `tone_llr` over every row the
  word fixes (front, mid block, both header copies, each floored at 0). `PRIME_LLR` 16:
  27000 windows of noise, OFDM bursts and the other grids' bursts scored at most 15.1
  (runs/primed-rx/plock_fa2.txt); bursts the blind paths lost read 20-60. A lock it makes is
  a header like any other (the session's wait extends to the burst's end).
- **Measured, code level** (`scripts/primed_study.py`, 200 bursts a point through the real
  receiver, PEP-referenced cell SNR; runs/primed-rx/ctl_code_level.txt,
  plock_code_level.txt). 10% failure point of an ACK's control, all bursts sent, blind ->
  primed with the cumulative ACK unknown -> and the expected slot too:
  - fsk16r25-r1/3: awgn -17.95 -> -20.15 -> -20.53, mpp -14.72 -> -17.22 -> -19.03,
    mpd -15.00 -> -16.83 -> -19.04, mpg -10.38 -> -14.43 -> -15.14.
  - fsk8r50-r1/3: awgn -15.38 -> -18.29 -> -18.61, mpp -12.19 -> -15.39 -> -16.53,
    mpd -12.37 -> -15.38 -> -16.84, mpg -7.75 -> -11.40 -> -11.80.
  - fsk32r62-r1/3: awgn -12.91 -> -15.43, mpp -8.74 -> -13.04, mpd -8.62 -> -13.58, mpg
    -7.42 -> -10.64 (its lock was never the limit).
  - Given the header, primed controls gain 2.5-3.2 dB on AWGN and 3.2-8.2 dB on fading
    (mpg's long fades most); the expected slot then takes the narrow grids to within
    0.0-0.9 dB of a known start and word. ACK modes: ack-1f and n4-ack-2f were control-limited and gain 2-3 dB; the
    4f/8f ACKs and the LDPC modes are header-limited (0.0-0.6 dB).
- **Measured, sessions** (loss studies, 48 paired seeds, DATA2G_PEP_REF_DB=5, shift+cpm,
  runs/primed-rx/loss5_paired.txt): the cell SNR where 10 bps is delivered moves -0.34 dB
  (SE 0.08) awgn, -0.96 (0.16) mpp, -0.84 (0.13) mpd, -0.84 (0.14) mpg; 40 bps -0.12 to
  -0.48. Floor cells +10-30% (mpp -13 +25.9% SE 5.6, mpd -13 +27.8% SE 4.4), mid cells
  unchanged (awgn -4, +10 identical; mpp 0 -0.2% SE 1.0, mpg 0 +0.7% SE 0.6). Forced
  fsk16r25-r1/3 (forced5_paired.txt): 5 bps reached 0.53 dB (SE 0.04) lower on AWGN,
  1.73 (0.13) mpp, 1.44 (0.12) mpd, 2.21 (0.19) mpg; at AWGN -19, where no session
  delivered, 21 of 48 now do. Chat (`chat_probe.py`, 24 seeds, chat5_paired.txt): mean
  latency of the messages both runs delivered -2.1 to -4.9 s on mpp -12..-10, mpd -11..-9,
  mpg -8 (SE 0.9-1.9), -0.5 to +0.5 s on AWGN (within SE); sessions lost to link loss 7
  and 9 of 72 at mpd -11, 2 and 3 at awgn -12 (chatm5_paired.txt): the three inspected ended on the
  ACK modes recommended for replies at mpd -9..-11, whose OFDM headers are not heard there.
- **False accepts.** Each primed decode is a 24-bit CRC check on at most 4 paths: 2^-22.
  A failed control's 6 tries are 24 checks, as many as its blind decode's (plain and two
  cuts, 8 paths each): its false-accept chance stays 24 x 2^-24 (2^-19.4). Measured
  (`scripts/primed_fa.py`, 200000 decodes a source and hypothesis, c16r25 controls): paths
  passing the CRC's low 8, 12, 16 bits at 0.57-1.4 x 4 x 2^-w on noise, on a real control
  that contradicts a pinned bit, and on my control under another session's key; no
  24-bit pass in 1.2 M. The score test drops 79-91% of the noise passes and 35-65% of the
  others on top. A primed lock on noise: 0 in 27000 windows at 16.
- **Adversarial tests** (tests/test_primed.py, Python and C++ alike): stations whose
  primed retries return forged controls (the hypothesis, random where it knew nothing) a
  third of the time never deliver a wrong byte and end delivered or in a bounded
  disconnect; forged connect frames mostly fail to unpack.
- **CPU** (x86, native): a primed control decode 0.12 ms (a blind CPM control decode with
  its cuts: 0.31); `prime_lock` per window 8 ms (c16r25, c8r50), 12 ms (c32r62); the Pi 4
  runs this DSP about 12x slower: about 0.1-0.15 s once per expected reply, under the
  receiver's 0.25 s search hop.
- **Not primed:** LDPC controls; OFDM acquisition and headers (the 4f/8f ACK modes are
  header-limited there); a reply that starts outside the window.
- The outcome model learned the old control failure rate (the ACK modes' and CPM's at the
  floor), and the shifter leaves a failing mode on control losses. So a control only a
  primed retry read is reported to the policy as the blind decode saw it (lost): with it
  reported as read, chat at awgn -12 stayed in OFDM ACK and polar modes whose data failed
  (mean latency +2.6 s, SE 1.2, 72 seeds) where the blind run had moved to CPM. Until the
  outcome data is regenerated with priming on and the model retrained, the shifter picks
  modes as before and the session gains above are without it.

## 4. Core control word (32 bits)

| bits | field | notes |
|---|---|---|
| 2 | frame type | 0 ARQ burst, 1 session control (§7), 2 probe/sound, 3 reserved |
| 2 | C - 1 | control codewords in this burst (1-4) |
| 3 | burst seq | this station's burst counter, mod 8 |
| 3 | acted-on | the peer burst seq whose ACK this burst acts on |
| 7 | peer cumulative | the next peer seq this station expects; all below received |
| 1 | reply lost | the peer's last burst repeated one this station already had (§6) |
| 6 | K | resends in this burst |
| 6 | recommend | the submode the peer should use next: sync band 2 bits (0 w, 1 n10, 2 w48; 3 CPM), index 4 (for CPM: the order of `data2g.cpm.SPECS`, 0-7; 8-15 under CPM are n10's indices 16-23, `policy.N10_HIGH`) |
| 2 | size hint | peer burst length: shrink / hold / grow / max |

Where the rest comes from:

- **CPM modes (`data2g.cpm`) carry control in a short codeword:** polar k=176,
  a 20 B payload, one per burst (twice under `ARQ_DUP`: the CPM header can
  announce only that). When control doesn't fit 20 B, the sender drops, in
  order: resends, then the optional extensions (`T_BUFFER`, `T_CHAT`,
  `T_REPLY`, `T_DUPCTL`; none is state the two ends must agree on), then the
  bitmap, then new data. A CPM data burst carries at least one data codeword
  and at most 8.

- **Duplicated control (`ARQ_DUP`, frame type 3):**
  - Each control codeword goes twice: RV 0, then RV 1 in the next slot.
    Everything after the doubled control span is mapped as usual.
  - The burst's receiver asks for it with the empty `T_DUPCTL` = 14 extension,
    when its outcome model predicts P(burst usable) < 0.9 for the recommended
    data mode. The sender honours it on data bursts only.
  - If slot 0 fails alone, the receiver combines slots 0 and 1 and decodes the
    pair. A pair combined on a burst that wasn't duplicated fails its masked
    CRC. One whose frame type isn't `ARQ_DUP` is discarded.
- **Redundancy versions are explicit:** a `rv` extension lists 2 bits per resend, in
  slot order, and is always present when K > 0. (Polar resends are identical and
  Chase-combined; their RV field is 0.)
  - **RV r (LDPC)** sends positions [r·n, (r+1)·n) of the mother code's circular
    buffer, wrapping (`codes.rv_positions`).
    - The mother code is the same base graph with all its extension rows
      (`QCLDPC.mother`), rate about 1/5. Its first n bits are the frozen
      codeword, so RV0 is unchanged on air.
    - RV1 onwards are fresh parity until the buffer runs out, then repeats.
    - The receiver adds soft bits into the buffer (`codes.combine`). It decodes
      the mother code cut to the extent received (`codes.decode_buffer`).
  - **Measured on the PHY** (scripts/ir_study.py, runs/ir_study.csv): two
    transmissions with IR move the 50% point 2-6 dB below one transmission's,
    more at higher code rates. Chase gets 1-3.5 dB less than IR at rate 1/2 and
    above. At rate 1/5, where RV1 is mostly repeats, they are the same.
- **Slot-to-seq mapping:** it follows from the acted-on ACK's missing list (the K
  resends) and `next_seq` (the new codewords). Every slot is then checked by its
  seq-masked CRC (§2), so a wrong mapping costs a failed decode, never wrong data.
- **The resend list is computed from one snapshot only:** the missing seqs of the single
  reply the sender acted on, taken in ascending order. The sender doesn't merge older
  ACKs. The receiver keeps a snapshot (cumulative + received set) for each of its last
  8 burst seqs, and computes the same first K.
- **A failed control slot is decoded again leaning on its padding.** A control is a few
  bytes (median 12, at most 28 in linksim sessions; up to 269 with an inline codeword,
  §5a) zeroed to its codeword's payload (46-396 B in the LDPC modes). LDPC is systematic,
  so the slot's own soft bits show where the zeros start: S(j), the sum over the sent
  payload bits from byte j on of log 2 P(bit = 0), gains about log 2 a bit of padding
  and loses on content's 1s. softmax(S) is the start's distribution, each payload bit is
  padding with P = P(start <= its byte), and one more decode adds log (1 + P) / (1 - P)
  toward zero to its soft bit (`codes.pad_prior`, `codes.add_prior`), CRC-checked under
  the slot's mask as usual. Receiver only: nothing on air changes. An `ARQ_DUP` pair that
  fails combined gets the same in its IR buffer (`codes.add_prior_buffer`).
  `DATA2G_PIN_EST=0` (`phy.PIN_EST`) pins at fixed cuts instead, 16 then 32 B
  (`phy.PIN_CUTS`, `codes.pin`), for A/B runs.
  - Polar sends no info bits: known ones are frozen instead (`codes.decode_raw_pinned`), and
    a polar codeword's payload enters its info set reversed (`codes.polar_order`, a wire
    format choice) so that a control's padding sits on the weakest positions. Cuts 12 and
    16 B (`phy.POLAR_PIN_CUTS`: CPM's control carries 9-16 B of its 20), counted along the
    control stream, so a control spanning two polar codewords has its second one pinned.
    Code level (`ctl_prior_study.py --polar`, BPSK): CPM control 1.0 dB, polar-k192 1.5 dB,
    polar-k96's last codeword 3-4 dB; the ACK modes (4 B codewords) none.
  - Measured at code level (`scripts/ctl_prior_study.py`, AWGN through each mode's
    constellation, a 12 B control): the 10% failure point moves 1.0-6.0 dB lower, 4.0 dB
    the median over the 36 LDPC modes. Least in n4/n10 r1/5 (22-23 B payloads, 1 dB),
    most at high rates (qpsk-r3/4 6.0, w48-64l-r3/4 5.5).
  - The estimate against pinning at the exact length (`ctl_prior_study.py --boundary`,
    controls of 12-264 B, 400 a point, AWGN and i.i.d. Rayleigh, 36 modes): 10% failure
    point 0.0 dB worse (median at every length), 0.6 at most; 1% point 0.0 median, 0.8 at
    most. The fixed cuts were as good up to 32 B and lost 1.2-3.5 dB median from 48 B
    on. One decode either way (the cuts averaged 1.13); no wrong codeword passed its CRC
    in 2.4 M retries. A hard cut at the most likely start (0.1 dB median, up to 2.4 at 1%)
    and a second, posterior-fed pass (0.2 dB better at most, 1.13 decodes) were measured
    and not taken. The start itself: exact 70-80% of the time, -1 byte 3-9%.
  - Polar can only take hard cuts, and its failed list's best path says nothing useful
    about where the zeros start (a cut there recovered a fifth at most of what the fixed cuts
    did), so polar keeps them. The CPM control loses to an exact cut only between them
    (13-14 B: about 0.3 dB) and past 16 B.
  - The outcome model and the link abstraction still assume the old control failure rate,
    so the gear shifter picks modes as before until they are retrained.
- **If any control codeword fails** (blind, pinned, and primed: §3a), **the burst is discarded** and not answered. Without
  the control word the receiver can't tell our burst from another station's, and
  guessing the mapping is a divergence risk. The sender times out and repeats.
- **A malformed control is a failed one.** A control that passes its CRC but can't be
  read is dropped the same way, with a warning in the log naming the fault, before
  any state changes. Examples: `T_RV` shorter than K resends, an empty `T_NEW`,
  `T_ABANDON` under 2 bytes, K or C past the burst's codeword count, a truncated TLV.
  Session frames likewise: shorter than their subtype's length, callsign codes past
  the alphabet, a CONNECT_ACK cap code above 2. CQ and ID frames (§7a) too: a `T_CQ`
  under 9 B, a `T_ID` under 10 B, callsign codes past the alphabet. No new recovery: repeats, the
  watchdog and bounded failure (§10) handle it. Before 2026-10 these raised out of
  the receiver (16% of fuzzed link runs with corrupted control).
- **Repeats:** a timed-out sender's first retry is the identical burst, with the same
  burst seq. The receiver recognizes it by that burst seq. Slots below its cumulative
  are ignored as duplicates.
- **Mode changes abandon outstanding codewords.** A resend must use its original
  submode, and a burst has one submode. So when the sender changes mode with codewords
  outstanding, it sends `abandon A` (A = its first unacknowledged seq). These rules
  came from bugs the fuzz tests found:
  - **Fresh bursts only.** Only a burst built as the direct reply to a peer burst may
    abandon, change the data mode or resync. Its ACK is then the peer's current
    state, since a station's receive state only changes when it handles the other's
    bursts. After a timeout the peer may have delivered past the stale cumulative,
    and re-slicing from it would corrupt the stream.
  - **Not when answering a repeat.** A burst that repeats the peer burst this
    station last answered (a late copy, or a crossing timeout repeat) carries an
    ACK that may predate this station's last burst. Its answer may not abandon:
    if that would take an abandon (a mode change or a due resync), it is control
    only, and the abandon waits for the next reply.
  - **Pre-abandon ACKs stop at A.** While an abandon is pending, a reply acting on
    a burst from before it describes the old slicing. A cumulative past A means
    the peer delivered old codewords there (a late burst); the sender can't map
    them onto the new slicing, so it disconnects (FAILED) and doesn't guess.
  - **Persistent, with an epoch.** Every burst carries the abandon until the peer
    answers one that did, polls and repeats included. The epoch lets the receiver
    apply each abandon exactly once. A receiver that missed a one-off abandon joined
    old and new slicing.
  - **Checked.** On first sight of an epoch, A must equal the receiver's cumulative.
    Otherwise the receiver disconnects (FAILED) and doesn't guess.
  - **No resends until a reply to a post-abandon burst.**
  - **Effects on both sides:
    - The sender re-slices its byte stream from A's first byte into the new mode's
      codewords, reusing seq A onward.
    - The receiver drops everything it buffered at or above A, and the stored soft
      bits for those seqs.
    - Nothing already delivered is ever affected.

## 5. Extensions (TLV, in control codewords or spare control bytes)

| type | contents | when |
|---|---|---|
| new | start seq (7 bits) of this burst's new codewords | any new codewords |
| abandon | seq A, and a reset flag (drop all stored soft bits) | mode change, resync |
| rv | 2 bits per resend, slot order | K > 0 |
| bitmap | 64-bit received map above the cumulative point | any gap |
| resync | marks a resync burst (§10) | §10 |
| report | SNR (0.5 dB), Doppler class, delay class, effective MI per constellation (4 x 4 bits) | every burst after a change, else every 4th turn |
| survey | noise excess per band above the passband median (4 bits each), and busy flag | when it changes |
| sound | "send your next burst in band B" (for the ACK-sounding up-shift, plan 5b) | shifter asks |
| buffer | bytes queued (log2), so the peer knows whether to expect data | when it changes |
| id (16) | packed callsign, session key: an ID frame, mask 0 (§7a) | periodically and after a session |
| comp (15) | 1 bit per data slot (resends, then new), MSB first, cut after the last set byte: the codeword is deflated (§9a) | a compressed new codeword, or a compressed resend the peer may not know is one |
| drained (17) | empty: this burst's new codewords end the sender's queued data (a hint, §6a reply hold) | the burst that slices the last queued byte, only if it fits the control's padding |
| inline (18) | seq (7 bits) and comp flag, a data codeword's payload with trailing zeros cut, its 2-byte check (§5a) | the burst's last new codeword, when it fits the control's padding |

## 5a. Inline codewords (session version 4)

A short write costs a whole data codeword even though the control codeword in front of
it is mostly zero padding: 64 B in w48-qpsk-r1/2 is 1.78 s on air against 1.21 s for
control alone, and on CPM a data codeword is 3-10 s (fsk16r25: 14.58 s with one, 4.98
without). So the last codeword of a write rides in the control's padding when it fits.

- **Wire:** `T_INLINE` = 18. Its value is one byte, seq mod 128 | comp << 7, then the
  codeword's payload with its trailing zeros cut, then a 2-byte check. The receiver pads the
  payload back to the data mode's size (`payload_bytes`), so the codeword is exactly the
  one a data slot would have carried: 5 B on top of the payload.
- **The check** is the low 16 bits of CRC-32 over the codeword's identity
  (`burst.data_mask`: key, direction, compression, abandon epoch, seq) and its padded
  payload (`frames.inline_check`). It does for the inline codeword what the masked CRC
  does for a slot (§2). A late burst from an older slicing, or a control that passed
  its CRC corrupted (a 2^-16 false accept), fails it, and the codeword is ignored as a
  failed slot would be.
- **Accounting is a data codeword's.** It has its own seq, the burst's last new one
  (`T_NEW` starts the slots before it), and is ACKed, held, delivered, abandoned and
  re-sliced by seq like any other. Nothing in §3's slot mapping changes: a burst whose
  only new codeword is inline is a control-only burst on air.
- **Resends go in a slot.** The inline codeword arrives exactly when the control does,
  so it is missing only when the whole burst was lost. The master's identical repeat
  carries it inline again. After that, the acted-on ACK lists it missing and it is
  resent as an ordinary data slot. Its RV is 0: the receiver holds no soft bits for
  it, so a send inside a decoded burst doesn't count towards its RV (`Codeword.inlined`).
- **What is inlined:** only the last new codeword, and only when it ends the queued data
  (the tail of a write) and fits the control's padding after everything else. Optional
  hints and requests (`T_BUFFER`, `T_CHAT`, `T_REPLY`, `T_DUPCTL`) are shed for it, as
  they are for resends: on CPM's 20 B control, a 2-10 B line then saves a whole data
  codeword. `T_DRAINED` goes in after it, if it still fits. At most 252 B, a TLV's
  255 less the header and check. A bulk burst's last codeword is full, so bulk is
  unchanged until the stream's last burst.
- **Why only the tail:** filling every control with the next stream bytes saves a
  codeword only where the tail would spill into one more, which is the same condition
  (for a raw stream, `len mod payload` in 1..spare). And every inline byte gives up
  pinning (below) for the whole burst.
- **The cost is control robustness on those bursts, much reduced by the padding
  estimate.** A control that fails plain is decoded again leaning on its padding (§3),
  wherever it starts. With fixed cuts at 16 and 32 B an inline codeword that pushed the
  control past 32 B left nothing to pin, and the control failed where a data codeword
  does, taking the burst's ACK with it (and IR combining for that codeword: a lost
  inline codeword is resent fresh). The estimate keeps the gain at every length, less
  the padding the inline bytes take.
  - Code level (`ctl_prior_study.py --content N`, the 36 LDPC modes, 10% failure
    point with pinning): a 24 B control (12 B plus a short inline codeword) is 0.5 dB
    (median; -1.5 to +2.5) worse than a 12 B one. At 48 B the pinning gain is gone: 4.0
    dB median (up to 6.0), and the control fails where data does. Pinning past the
    exact length wins back 3.0 dB median there, and the estimate (§3) gets within 0.0
    dB median of that in one decode: 2.4-3.5 dB over the fixed cuts at 48-96 B, 1.2-2.3
    at 128-264 (AWGN, Rayleigh).
  - Sessions, 2026-10-27 (paired, DATA2G_PEP_REF_DB=5, native, `chat_probe.py`, first
    message excluded; mean latency change, SE): chat 50-200 B (48 seeds) awgn 10
    -0.51 s (0.02), mpp 5 -0.32 (0.10), mpp 0 -0.93 (0.16), awgn 0 -1.01 (0.08),
    awgn -6 -0.56 (0.06), mpp -6 -1.05 (0.27), awgn -12 and mpp -10 within noise
    (SE 1.6-1.8). Chat 2-10 B lines: -0.57 s at awgn 10, -1.1 at awgn 0, -3.0 at awgn
    -6, -3.3 at mpp -6, -5.6 at awgn -12, -5.8 at mpp -10 (medians 5.8 to 2.8 s at -6
    dB, 10.3 to 4.8 s at -12: CPM sends the line with no data codeword). Shedding the
    hints doubled the CPM gain. Winlink sessions (48 seeds) 5.0% shorter at awgn 10
    (SE 0.3), 1.5% at awgn 0 (0.5), within noise on mpp and mpg. Bulk (`loss_study`,
    16 seeds, 9 cells incl. awgn -10 and mpp -8): bit-identical, its tail is never
    reached in the horizon. Inlining only when no data slot is left took most of the
    chat gain away (awgn 0: -0.03 s against -0.89).
- **Off switch:** `DATA2G_INLINE=0` (`link.INLINE`) never sends one, for A/B runs. A
  station always receives them. A version 3 peer ignores the TLV, and each inline
  codeword then waits for its resend in a slot.

## 6. Turn rules and timers

- **ACK:** every burst acknowledges the peer's last burst through its control word. There
  are no separate ACK frames.
- **Turn time:** the station holding the turn answers within `T_turn` of the peer's
  burst end. That covers decode, PTT and audio, target 1.0 s, configurable per station
  and exchanged at connect.
- **ISS timeout:** peer burst end + `T_turn` + the airtime of the longest reply the peer
  could send in the recommended mode + 1 s.
- **On timeout** (plan 5a), step by step:
  1. **Repeat:** resend the same burst, with the repeat bit set, the same RVs and the
     same burst seq. The IRS re-sends its last reply, with its own repeat bit set.
  2. **Shrink:** fewer codewords, one step more robust, and recommend a more robust
     reply mode.
  3. **Robust floor:** the most robust polar mode in the narrowest band both allow.
  4. **Probe:** after 6 misses, one robust probe every 10 s. Link lost at 90 s:
     disconnect and report to the host.
- **Who retries:** only the caller (the session's master) retries on timeout. The
  callee only ever answers, carrying its own data in its replies. It starts a turn
  itself only from idle. Otherwise, after a lost reply, both would be waiting and both
  would key.
- **Polls:** past the first identical repeat, the master sends control-only bursts
  (frame type PROBE), with no data and no abandon, in the escalated mode. The reply
  to a poll restores an exact ACK.
- **Lost replies escalate the replier.** A repeat or a poll tells the callee its last
  reply was lost, and it makes its next reply more robust.
- **Lost reply vs lost data:** the core's repeat bit, set in a reply, means "your last
  burst repeated one I had already received, so my previous reply was lost". The sender
  then makes the recommended reply mode more robust, and doesn't down-shift its own
  data. Repeats themselves are recognized by burst seq, not by this bit.
- **Idle:** when both buffers are empty, the station that just received an empty burst
  stays silent. Either may start the next turn when data arrives, after
  listen-before-talk plus a random 0-2 s backoff. A keepalive probe is sent every 60 s
  of silence, and the session closes after 300 s.
- **Listen before talk** applies only to turns that don't follow a peer burst: idle
  starts, connects, probes. The shifter's own turns keep the protocol's timing.

## 6a. v1 implementation choices (data2g/arq/session.py)

- **Idle:** the caller keeps the link alive with a poll a random 15-30 s after the
  last exchange.
  - The callee breaks idle itself when its host writes: a wake burst, built like any
    other (data, its ACK), once t_turn + 2.5 s + 1 s plus a random 0-1 s has passed
    in silence since its last burst. By then the caller's answer or timeout retry
    would have started. A header heard meanwhile holds the wake past that burst.
  - Only from idle: the callee's last burst carried no data, so the caller's receive
    state is exactly what the callee last heard it ACK, and the wake may be a fresh
    build (abandon, re-slice, mode change).
  - The callee owns no retry timer. An unanswered wake is repeated identically (to
    the caller, a repeat: its reply was lost) after the same guard plus a random
    backoff that doubles with each send (0-1, 0-2, 0-4 s, ...: CSMA-like). At most 2
    sends per idle period, 6 with chat on. After that the caller's keepalive
    collects the data. Any burst from the caller cancels a wake.
  - Reply hold: when a burst brings the callee's host data, carries `drained` (the
    caller's buffer is empty), and the callee has nothing to send, the callee holds its
    reply 250 ms (REPLY_HOLD_S). An interactive host's answer
    (a BBS prompt) then rides the reply instead of waiting for a wake. A host write
    ends the hold (or starts the 100 ms coalesce, CHAT OFF). The reply stays a fresh
    build: the callee's receive state changes only on the caller's next burst. Bulk
    bursts don't carry `drained`, so they are answered at once. linksim winlink
    (40 paired seeds): sessions 11% shorter on awgn 10 and mpp 10, 4% on mpg 5; 0.5 s
    gained nothing over 0.25 s.
  - The caller answers a burst that arrives while its idle poll waits at once, data
    or not. It stays the only station that retries on a timeout, except for DISC.
  - The caller also answers at once a burst carrying `reply lost` or an abandon. The
    callee holds data either way, and an escalated answer can be control only: idling
    would leave it to a wake about 5 s later.
  - DISCONNECT breaks idle the same way: the caller sends DISC in place of its
    waiting poll, the callee as a wake. A station with unacked data sends it first.
    Either side retries its DISC on a timeout (DISC_TRIES).
  - v1 (until 2026-10) let only the caller start turns, polling 2 s after the last
    exchange and doubling to 16 s (2-4 s with chat on): up to 16 s of callee latency,
    and constant keying.
- **Waiting for a reply:** t_turn + 1 s for the reply to start, detected as a decoded
  burst header. The header gives submode and codeword count, so the wait then extends
  to the reply's known end. The master doesn't sit through a worst-case reply length
  before retrying.
- **Link lost:** 90 s after the last decodable burst from the peer, on either side:
  60 s of retries past the keepalive's longest gap (30 s).
  The link core's consecutive-timeout count (12) applies only in the lockstep tests,
  which have no clock. In a session it tripped on links that were slow but alive.
- **Listen before talk** before any turn that isn't a reply (retries, polls): a
  station doesn't key over a carrier whose header it failed to decode.
- **Connect frames need a mode with >= 28 B payloads,** so the frame fits one
  control codeword. At 20% codeword loss, a 2-codeword CONNECT and CONNECT_ACK round
  trip got through only ~29% of the time.
- **A closed callee keeps its session key,** and answers a repeated DISC with
  DISC_ACK, since its first DISC_ACK may have been lost.

## 7. Session control frames (type 1)

These are sent in the most robust mode the bandwidth cap allows. The fields span
several control codewords, and callsigns are packed 6 bits per character, up to 10
characters plus SSID, space padded at the end (a space inside a name is kept).

| frame | contents |
|---|---|
| CONNECT | protocol version, caller, callee, nonce, bandwidth cap, `T_turn` |
| CONNECT_ACK | nonce echo, accepted cap (the minimum of both), `T_turn` |
| CONNECT_NAK | nonce echo, reason: 1 version, 2 not listening, 3 busy (a session or a call of its own under way), 4 refused by the host (ACCEPT couldn't connect) |
| DISC / DISC_ACK | graceful close; DISC is retried 3 times |

CONNECT retries: 5 tries, 3-5 s apart with jitter, then fail to the host.
A CONNECT_NAK ends them at once: the caller closes with `refused: <reason>`.

A CONNECT addressed to a station is decoded in every state, so it is always
answered: CONNECT_ACK while listening, CONNECT_NAK otherwise. The NAK is not the
session's burst: it starts no reply timer or wake. A station with its own burst
queued sends that instead, and refuses the caller's next try.

## 7a. ID frames

Station identification, readable by anyone listening (data2g/arq/engine.py).

- **Frame:** a control-only burst, frame type SESSION, mask 0, in the cap's connect mode, as a
  CQ frame is. It carries a `T_ID` = 16 extension: the packed callsign (8 B), then the
  16-bit session key it identifies for (2 B). One codeword in every connect mode.
- **Mask 0, not the session key:** under the session key anyone could still read it
  (§2), but only a station with the key could check its CRC. At mask 0, anyone can.
- **Outside the protocol:** no seq, no burst seq, never seen by the session. A receiver
  notifies it as `ID call key`, and logs it.
- **During a session:** at least every `ID_INTERVAL_S` (600 s, FCC 97.119), the ID goes
  back to back ahead of the station's own turn, on the same PTT.
  - A waiting caller that hears a one-codeword burst in the connect mode allows
    `REPLY_START_S` more for the reply's header. The reply follows the ID at once, and
    finding its header takes up to ~0.85 s, close to `T_turn`.
- **After a session:** one more ID, still with the expired session's key.
  - The station that closes on a DISC sends it right after its DISC_ACK.
  - Otherwise it goes `ID_GUARD_S` (2.5 s) after the close.
  - Nothing new goes out in that guard (KISS, CQ, a new session), so the peer's
    trailing ID is heard and not keyed over. A KISS frame queued during a session was
    lost that way.

## 7b. CW ID (session version 5)

A Morse ID in addition to the ID frames, or in their place (data2g/cwid.py,
data2g/arq/engine.py). Off by default.

- **Setting:** `Engine.cw_id`: `off` (ID frames only, as before), `add` (a CW ID
  with each ID frame), `only` (a CW ID in place of each). Speed `cw_wpm`, 15-40,
  default 20. data2g-host and data2g-gui: `--cw-id {off,add,only}`,
  `--cw-wpm WPM`, Settings > Station > CW ID; a client with DATA2G ON:
  `CWID OFF|ON|ONLY [wpm]`. A plain VARA client's `CWID ON|OFF` is accepted and
  ignored: the operator's setting stands.
- **Text:** `DE` and the station's call less any SSID (`DE W1AW`); about 4-5 s
  at 20 wpm for a 4-6 character call.
- **Tone:** 1500 Hz (config.FCENTER), inside every band and CPM grid (n4's
  1450-1600 Hz is the narrowest), so it passes any session's TX and RX filters.
  Constant envelope at the bursts' peak (1.0, engine._start_tx). Each key edge
  a 5 ms raised cosine: no clicks, under -40 dB of the tone 250 Hz off.
- **When:** where the ID frames go (§7a), and outside sessions:
  - ahead of the station's own turn every `ID_INTERVAL_S`, on the same PTT:
    CW ID, then ID frame (`add`), then the turn's burst;
  - after a session: after the DISC_ACK (ID frame, then CW ID), or
    `ID_GUARD_S` after the close;
  - outside a session (KISS, CQ frames, a refused or unanswered call): one
    once the station has been quiet `CW_QUIET_S` (30 s) with nothing queued,
    and one ahead of such a transmission `ID_INTERVAL_S` after the first one
    no ID covered.
- **Hearing one (CwSense, every station, whatever its own setting):** per 20 ms,
  25 Hz bins 1200-1800 Hz (Hann) against a tracked noise floor. A window is a
  hit (one tone within +-50 Hz of 1500 over 10 dB above the floor, nothing
  100 Hz or more from it), quiet, or elsewhere (a burst: OFDM fills the
  bins, n4's other carriers, a CPM tone off the centre). Active when the last
  50 windows (1 s) hold 12 hits on one tone, 4 quiet ones and at most 6
  elsewhere.
  While active:
  - a waiting caller's reply deadline is at least `REPLY_START_S` away
    (`Session.on_cw`), as an ID frame's is: the turn follows the ID at once;
  - a callee's wake counts its silence from it;
  - nothing new starts (the session's poll waits, as for a burst arriving),
    and nothing outside a session (KISS, a CQ frame, a pending last ID) goes
    before `ID_GUARD_S` past it;
  - the host reports BUSY ON.
- **False locks:** a dash reads as a preamble (one carrier's tone for its 8
  repeats), and 1 in 64 such headers pass their CRC; the frame-pilot search
  finds w header copies in one too. A suspect OFDM lock
  (`Receiver.SUSPECT_SCORE`) that started before CwSense's third-latest hit
  while active is dropped (`Receiver.drop`), and the search goes on past it.
  A clear header is never dropped: CwSense reads n4 near its floor as CW now
  and then.
- **Limits:** a waiting peer holds through an ID 93% of the time or more down
  to -8 dB key-down SNR (PEP, 2500 Hz) on AWGN and -6 dB on MPP/MPD (half at
  about -10 dB); below that the peer's turn may cost a timeout and a retry. A strong steady carrier 100-300 Hz off 1500 Hz hides an
  ID (every window reads elsewhere). CwSense sees +-60 Hz of mistuning.
- **Version:** a v4 caller does not hold for a CW ID ahead of the callee's
  turn: it times out under it and keys over the reply. Session version 5.
- **Measured** (scripts/cwid_study.py, 2026-10-28):
  - CwSense on 1200 s of noise: never active. On 4 bursts of every mode at
    30/10/0/-5 dB, AWGN/MPP/MPD: active only in n4 and n10 near their floor
    (-5 dB: up to 1.8 s over 4 bursts) and moments in ack-1f, fsk16r25.
  - 120 IDs in noise (15-30 wpm, -5 to 30 dB): the receiver alone locked 13
    headers on them (none on the same length of noise); an Engine kept 1.
  - Paired whole sessions (speedtest's path, both ways, 20 kB, 12 seeds at
    AWGN 0, MPP +6, MPD +10, MPG +3 dB PEP in 3 kHz): CwSense on, CW ID off,
    against no CwSense: median +0.0 s, timeouts 32 vs 33. CW ID only, every
    60 s: +30 s median (the IDs' own airtime), timeouts 42 vs 32 (10 in 470 IDs).
  - Airtime: one ID, about 4.5 s at 20 wpm, per station per 10 minutes
    (0.75%), and one more each at the end.

## 8. Gear-shift loop (summary; details in data2g/arq/policy.py)

1. The receiver predicts P(decode) per candidate submode, using the link predictor on
   its measured features plus the survey.
2. It recommends the mode and size that maximize expected goodput for the peer's next
   burst, within the session's bandwidth cap.
3. The sender follows the recommendation, but can be more conservative when its own
   timeout history says the reply path is failing.
4. For an up-shift to a wider band, the receiver first asks for a sounding reply in
   that band (`sound`), and recommends it only if the reply measured well.
   (Not implemented yet.)

As built (data2g/arq/policy.py):

- The predictor sees the peer's last two bursts, not one. One burst can't tell a steady
  channel from 0.1 Hz fading. Two can: average them, or read the change between them.
- Online correction: after each peer burst, the receiver compares what decoded with the
  P it predicted for that mode: whether the burst was usable (its control decoded)
  against P(usable), and the decoded fraction of its first-transmission data codewords
  against P(codeword). Each moves a logit bias per submode by the difference, bounded at
  ±6 (`BIAS_MAX`; a family-wide bias, bounded at ±3, came first).
  This absorbs what the predictor gets wrong on the actual link.
- Switching away from the mode of the peer's outstanding codewords abandons them (§4).
  The recommender charges a switch with the bytes it holds beyond its cumulative ACK.

## 9. Host interface (VARA-style TCP)

- **Ports:** command port 8300, data port 8301 (VARA's defaults).
- **CHAT ON / CHAT OFF** (VARA's commands) set this station's objective: latency or
  throughput. A station with chat on sets a `chat` extension (1 byte) in its bursts.
  The peer's shifter, which recommends this station's modes, then minimizes expected
  delivery time of what's queued (short bursts, higher-P modes) instead of maximizing
  bytes per second. The callee's lines go in its wake bursts (§6a), with more wake
  retries than with chat off; the caller's keepalive is the usual 15-30 s (v1 polled
  every 2-4 s instead). Implemented: `T_CHAT` = 12 (empty), `Session.set_chat()`,
  `GearShifter.recommend`, `session.CHAT_WAKE_TRIES`.
  - The objective is the least expected time to deliver what the peer has queued,
    at least a 200 B chat line.
  - With chat on, a data burst carries `T_BUFFER` (2 bytes): the sender's unsent
    queue, only when over 200 B. It's informational only.
  - Without it, a file sent with CHAT ON went out 2 codewords per burst in the
    slowest mode that fit a line.
- **Commands:** MYCALL, LISTEN ON/OFF, CONNECT from to, DISCONNECT, ABORT,
  BW500 / BW2300 / BW2750, and CQFRAME call bw.
  - **ACCEPT [host]:port** (Data2G) is LISTEN ON with the session's data on a
    connection Data2G makes to host:port (default: the address the command client connected from) when it answers a call,
    instead of the data port. One session at a time, as with LISTEN.
    - If the connection fails (1 s timeout), the call is refused (CONNECT_NAK 4).
    - When the far end closes it, the session disconnects gracefully.
    - LISTEN ON/OFF, or the command client leaving, ends ACCEPT.
  - **CQFRAME** sends a CQ frame: a control-only burst (frame type SESSION, key 0,
    mask 0) with a `T_CQ` = 13 extension, the packed callsign plus the bandwidth
    cap code. It goes in the robust connect mode for that bandwidth, so at 500 it
    stays inside 500 Hz.
  - It needs no session and makes none. It is refused while a session is under
    way.
  - Any station that hears one, listening or not, notifies its host with
    `CQFRAME call bw`.
  - BW500 caps at 500 Hz. This one matters: it keeps a session inside a 500 Hz
    band-plan segment.
  - BW2300 and BW2750 set no cap (2400 Hz, the widest band).
  - BW1200 is an extension, not in VARA; a minor feature.
  - **SN** answers `SN snr`: the pilot SNR (snr_est, dB, average power in 2500 Hz,
    as the monitor shows it) of the last burst decoded from the session's peer.
    Only bursts that decode under the session's key count. WRONG before the
    first one, and again once a new session starts. No OK follows.
  - **DATA2G ON / OFF** marks a client that knows Data2G. Off again when the
    command client leaves. With it on:
    - BUFFER is the plain VARA count (no next-burst credit).
    - CLEANTXBUFFER reports host byte offsets (below).
    - TXPOWER gets the output volume, dB with one decimal (`TXPOWER -1.2`).
      `TXPOWER dB` sets it, -60 to 20 (0 puts a burst's peak at full scale):
      OK, or WRONG out of range or not a number. Applies from the next audio
      block; the station's own setting (--output-volume, the GUI) is the start.
    - TIMEOUT gets the link idle timeout, seconds with one decimal
      (`TIMEOUT 300.0`): a connected session with no data either way for that
      long disconnects (IDLE). `TIMEOUT s` sets it, any s > 0: OK, else WRONG.
      It applies to the session under way too, and goes back to the default
      when the command client leaves.
    - DISCONNECTED carries a reason word, then detail words for some:
      CLIENT (DISCONNECT, or the command client left), IDLE (idle timeout),
      PEER (the peer's DISC), ABORT, LINK_LOST, LINK_FAILED why, NO_ANSWER,
      PEER_RECONNECTED, and a CONNECT_NAK's BUSY, VERSION, NOT_LISTENING or
      REFUSED (unknown NAK: REFUSED reason n). CLIENT and IDLE end in
      `unconfirmed` when the DISC was never acked.
    - Data read from the data port before CONNECTED is dropped, so the queue is
      empty at CONNECTED and offset 0 is the first byte written after it. VARA
      queues such data for the session. A client must not write before
      CONNECTED: bytes still in the socket then can't be told apart.
    - The data port is not exclusive: any connection can write into the session.
      For a DATA2G client a second writer is undefined behaviour (its bytes
      shift every offset). Planned: a per-session data port for outgoing
      sessions, like ACCEPT in reverse.
  - **CLEANTXBUFFER** drops queued data. Replies as VARA's: `CLEANTXBUFFER
    BUFFEREMPTY` with nothing queued, `CLEANTXBUFFER FAILED ` (trailing space)
    outside a session, else `CLEANTXBUFFER OK`.
    - Bytes never on air are dropped at once. A record begun on air is finished
      (§9a), so the cut falls at a record boundary.
    - VARA also drops bytes on air but unacked, though the peer may already have
      them. Here they are in limbo: the next ACK past them says they arrived; the
      next exact ACK (§4) says the rest did not, and an abandon drops those.
      No wire change.
    - VARA clients see the limbo gone at once: BUFFER leaves it out.
    - DATA2G ON: host byte offsets since CONNECTED, ranges `[from, to)`, so a
      client can match them to what it wrote without BUFFER arithmetic (data
      still in the socket would skew that).
      - `CLEANTXBUFFER OK drop_from drop_to limbo_from limbo_to`. drop_to is
        every byte Data2G has read (n, below, if that is less); later writes are
        kept. An empty limbo is `drop_from drop_from`. A second clean before the
        limbo settles repeats it.
      - `CLEANTXBUFFER n` drops only bytes before offset n; those from n on,
        read or not, are kept. `CLEANTXBUFFER LATE`, nothing dropped, if a byte
        at or past n is on air or finishes a record begun on air: the client
        retries with a new offset. On OK it can append from n safely. An n
        inside a record splits it (the part from n becomes its own record).
        WRONG without DATA2G ON.
      - Once a non-empty limbo settles: `CLEANTXBUFFER SETTLED from split to`,
        delivered before split, dropped from it. BUFFER counts the limbo until
        then. The ACK that settles it does so at once, and the station holds its
        next burst `REPLY_HOLD_S` (0.25 s) for a write, so a client that waits
        for SETTLED before writing still makes that burst.
      - If the session ends first (disconnect, link lost, ABORT), before its
        DISCONNECTED: `CLEANTXBUFFER SETTLED from split to UNKNOWN`. Before split:
        records acked whole. From it: on air unacked, the peer may have it.
- **Replies and events:** OK, WRONG, CONNECTED from to bw, DISCONNECTED [reason],
  BUFFER n (host bytes the peer has not acked, as VARA counts them; less the
  next-burst credit for VARA clients), PTT ON/OFF, BUSY ON/OFF, and a
  Data2G-specific `MODE name`.

## 9a. Stream framing

- **Records:** the host byte stream is cut into records of [length 1-255][bytes].
- **Padding:** a zero byte is a zero-length record and is skipped, so a burst's last
  codeword is padded with zeros.
- **Boundaries:** padding only ever falls between records. That keeps re-slicing after
  an abandon exact: the receiver's stream is the concatenation of delivered codewords
  in seq order.
- **Compression (`T_COMP`, session version 2; primed dictionary, version 3):**
  - A compressed codeword is raw deflate (no header) of the stream bytes it carries,
    zero padded. Deflate is primed with a fixed 4 KB dictionary
    (`frames.ZDICT`, built from C4 web text by `scripts/build_zdict.py`), then the last 4 KB (`frames.HIST`) of the stream as delivered
    before it: raw codewords with their padding, compressed ones inflated.
  - The receiver delivers in seq order, so it always holds that history. It inflates at
    delivery; a codeword that won't inflate fails the link (protocol error).
  - The flag is part of the codeword's CRC identity (§2), so a codeword decoded
    under the wrong flag fails its CRC rather than delivering.
  - The flag is fixed at creation, so a resend carries the same bit. After an abandon
    both ends' history is the stream before A, and re-sliced codewords are
    compressed afresh.
  - The sender compresses only when that carries more than a raw codeword: the longest
    prefix that deflates into the payload, found by binary search. Incompressible data
    costs one trial per codeword.
  - A new codeword is compressed only if its bit fits in the control's spare bytes.
  - A resend's bit is sent only while the peer may lack it: until the peer acts on
    the burst that sent the codeword new (its new slots are mapped by `new`, so the
    peer took their bits from it, whether or not the data decoded). The receiver
    keeps flagged seqs until delivery or abandon, and ORs them with the bits. A
    bit still owed is reserved like `rv`, and can cost control space.
  - Measured (tests/test_arq.py harness, 22 B and CPM 20 B control, loaded like
    the gear shifter's): sending the bit on every compressed resend changed the
    control's fit in 3-13% of bursts. Omitting known bits cut that by 67-89%.
  - Measured on text (scripts/compress_study.py): about 1.75x fewer codewords
    at 38-176 B payloads, 2x at 396 B. Codewords compressed on their own gained
    1.0-1.1x on narrow modes. zstd lost to deflate at every size (its frame header),
    and brotli's binding can't take a dictionary.

## 10. State agreement (no jabbering)

Failure mode to design out: two stations that hear each other fine, exchanging bursts
forever without progress, because they disagree about protocol state.

- **Nothing implicit can corrupt.** Every slot's seq, and a data slot's compression,
  is checked by its masked CRC (§2), an inline codeword's by its check (§5a).
  RVs are sent explicitly. Cumulative ACK, burst seq and acted-on are in every control
  word, so each burst restates the sender's view.
- **Progress watchdog.** Progress means the peer's cumulative ACK advanced, or this
  station delivered new bytes. Count turns in which both sides decode each other's
  control words but neither makes progress while data is queued. At 8 such turns,
  enter resync:
  - The sender abandons from its window base, with the reset flag set. The base is
    the receiver's cumulative ACK, which both sides know.
  - It re-slices from there at RV0.
  - The receiver drops every stored soft bit for that direction.
- **Burst seqs can't alias.**
  - Each station numbers its bursts absolutely; the wire carries the number mod 8.
  - It never has more than 7 bursts the peer hasn't acted on. At that limit its turn is
    an identical repeat of its latest burst.
  - So the peer's 3-bit acted-on resolves to exactly one burst in
    [last confirmed, latest]. A reference outside that window disconnects.
  - Plain mod-8 counting let a peer that missed 8 bursts cite an old seq equal to a
    new one. It cleared a pending abandon the peer had never applied, and data was
    corrupted: found by the session stress run.
  - Reusing a seq until confirmed was tried and is also wrong: a reply lost after the
    peer did decode the first version makes one seq name two contents.
- **Bounded failure.** If 3 resyncs in a row make no progress, disconnect and report a
  protocol error to the host.
  - An inconsistency the protocol can't explain also disconnects at once, rather than
    resyncing: a peer cumulative outside [base, next], or an abandon at the wrong
    point. A resync from a state the station can't trust could corrupt.
  - A dead link is bounded too: the 90 s timeout of §6.
  - No path through the state machine runs unbounded.
- **Tests before tuning** (tests/test_arq.py, and scripts/arq_stress.py: 3200 runs
  over a 4x4 loss grid; no corruption, mismatch or fail-safe trip): random loss of
  bursts, control codewords and single codewords; duplicated bursts; and a link that
  dies. Assertions:
  - The delivered stream is always an exact prefix of what was sent.
  - Progress resumes within a bounded number of turns once losses stop.
  - Every run ends in delivery or a bounded disconnect.
- **Reordering** was claimed above until 2026-10 but never tested. The native port's
  fuzz (tests/test_native_fuzz.py: late copies of a sender's previous burst) found
  two ways a late burst corrupted the stream. The slicing identity (§2) and the two
  ACK rules of §4 close them; minimized reproducers are in tests/test_arq.py. Late
  bursts now end in delivery or a bounded disconnect. On air the engine decodes in
  order, so only a crossing repeat can produce one.

## Review decisions (2026-09-24)

1. One control codeword first per burst; overhead to be measured.
2. The 7-bit seq space is fine.
3. BW500 caps at 500 Hz (important). BW2300 / BW2750 mean no cap. BW1200 is an
   optional extension.
4. 6-bit callsign packing is fine.
