# Data2G ARQ protocol (draft for review)

Status: approved 2026-09-24, with the state-agreement rules of §2 and §10 added at review. Plan: gear-shifter
phases A-H (/home/andrew/.claude/plans/sleepy-jumping-whisper.md); this is phase B.

TLDR: a connected, half-duplex, turn-based ARQ. Both directions use one burst format:
a control word first, then resent codewords, then new ones.

- **Reliability:** selective repeat per codeword, with incremental-redundancy (IR) resends.
- **Mode choice:** each station's receiver recommends the other's next mode and burst size.
- **No PHY header change:** everything rides in codeword payloads; the frozen 16-bit burst
  header is untouched.
- **No silent disagreement:** whatever the two stations must agree on is either sent
  explicitly or checked by every codeword's CRC (§10).

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
  Without it, zero-padded codewords were wrecked by the clipper (phase G).
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
extensions, so C = 1.

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
- **If any control codeword fails, the burst is discarded** and not answered. Without
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
  - The caller answers a burst that arrives while its idle poll waits at once, data
    or not. It stays the only station that retries on a timeout, except for DISC.
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
| CONNECT_NAK | reason (busy, version, refused) |
| DISC / DISC_ACK | graceful close; DISC is retried 3 times |

CONNECT retries: 5 tries, 3-5 s apart with jitter, then fail to the host.

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

## 8. Gear-shift loop (summary; details in the phase E policy doc)

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
- Online correction: after each peer burst, the receiver compares the burst's decoded
  fraction (control plus first transmissions) with the P it predicted for that mode. It
  moves a logit bias per (band, constellation family) by the difference, bounded at ±3.
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
- **Replies and events:** OK, WRONG, CONNECTED from to bw, DISCONNECTED,
  BUFFER n, PTT ON/OFF, BUSY ON/OFF, and a Data2G-specific `MODE name`.

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
  is checked by its masked CRC (§2).
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

1. One control codeword first per burst; overhead to be measured in phase D.
2. The 7-bit seq space is fine.
3. BW500 caps at 500 Hz (important). BW2300 / BW2750 mean no cap. BW1200 is an
   optional extension.
4. 6-bit callsign packing is fine.
