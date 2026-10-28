# Application soft combining and repair: plan

Status: proposed 2026-10-28, for review. No code yet. Nothing here changes `docs/arq.md` or the PHY.

TLDR: give broadcast applications one framework in four layers. The modem names codewords and keeps
soft bits; the application owns erasure coding, feedback and policy.

- (a) Units: a new kind of broadcast payload, exactly one codeword, named by the application
  (24-bit stream, 16-bit index). The name rides in the burst's control (`T_UNITS`) and in a
  position-free CRC mask, so a later send of the same unit combines, in any slot, burst or pass.
- (b) Soft retention: the receiving host keeps a failed unit's soft bits (int8, capped, with a
  lifetime) under that name. The name is the handle. A later reception of the unit combines on its
  own; the app is told which units are held and how close each one is.
- (c) RV control: each unit the app sends carries its redundancy version (0-3), listed in `T_UNITS`.
  The app picks Chase (RV 0 again) or IR (the next RV) per unit.
- (d) Erasure repair stays in the app (Ragtime's Cauchy GF(2^16) parity, ion-hf's Cauchy GF(256)).
  Units make an erasure exactly one codeword. The app's repair round then mixes fresh parity units
  (help every station) with IR increments of held units (help stations below the mode's knee).
- Wire: broadcast header version 4 for bursts that carry units. Frame-only bursts stay version 3.
  No PHY (`config.PROTOCOL_VERSION`) or ARQ session `VERSION` change.
- Host: a new KISS command byte (0x0D) for units both ways, `BCAST UNITS n ON|OFF` and `BCAST SOFT`
  commands. Python and native hosts alike. `MODE` and `HEARD` lines are left exactly as they are,
  because Ragtime's parser counts their words.
- Order: studies first (E1, E2), and each wire step only if the study before it shows at least 1 dB
  at the weakest receiver's completion point.

## 1. The problem, and what exists today

### 1.1 What an application can do today

- Learn codeword sizes. `MODES` (`host.mode_lines`) gives each mode's bytes per codeword, the most
  codewords a receiver takes, and airtime at 1 and at max codewords.
- Align frames to codewords. Broadcast v3 (`docs/broadcast.md` §3) COBS-frames each frame, puts a
  zero byte before a frame that starts a codeword, and loses only the frames a failed codeword
  touches. A frame of k x bytes-per-codeword - 3 bytes fills k codewords exactly
  (`kisslink.framed_len`, `kisslink.pack_frames`). Ragtime does this (`core/src/broadcast/Airtime.cpp`,
  `pieceFrameBytes`).
- Name a group and a sender. `BCAST OPEN group [FROM call]`. Every codeword is CRC-masked with the
  group key (`kisslink.data_start_mask`), and the control carries the name.
- Protect the control. `BCAST MODE n DUP mode` sends each control codeword at RV 0 and RV 1; the
  receiver combines the pair (`phy.ModemRx.raw_pair`, `decode_pair`), about 3 dB.
- Know what went out and what was lost. KISS ACKMODE acks; `BCAST n HEARD`, `LOST k`, `MISSED`,
  `DROPPED`.
- Repeat. An app may send the same frame again. The receiver decodes each copy on its own.

### 1.2 What is lost today

- Soft information of failed codewords. `KissLink.on_burst` decodes every slot with no key
  (`rx.decode(..., key=None)` in `read_burst_frames`), so a failed codeword's soft bits are dropped
  when the burst is done. A repeat starts from zero.
- A frame resent cannot combine even in principle. Its data codewords' CRC mask depends on the slot
  and the burst's data start (`data_start_mask(slot, start, key)`), so the info bits, and so the
  codeword, differ between sends. Nothing in the control names a frame either.
- IR exists only inside the ARQ (`docs/arq.md` §4: `rv` extension, `codes.combine`,
  `codes.decode_buffer`; store `Engine.store`, cleared per session) and in the experimental
  `data2g-bulk` (`docs/bulk.md`: block identity in the mask, `bulk.stream_id`, a store kept across
  passes).
- Erasure granularity is a frame, not a codeword. A Ragtime file piece is one 255-byte frame. In
  qpsk-r1/2 (116 B a codeword) that is 2 codewords, in the FSK modes 3, in qpsk-r1/5 5, and in
  n10-qpsk-r1/5 (23 B) 11. A piece is lost if any of its codewords fails: at 5% codeword loss an
  11-codeword piece is lost 43% of the time, so the parity round must cover 43% of the file,
  not 5%.
- Per-piece header bytes. Each Ragtime piece carries `0xA0`, the record outline (type, length,
  callsign, seq, time) and its index (`docs/PROTOCOL.md`, "Records"; `Records.h`, `FilePiece`), about
  18-20 B. Fine in a 255 B frame; 35-50% of a one-codeword piece in the 38-58 B FSK modes. That is
  why Ragtime cannot simply make pieces one codeword long today.

### 1.3 Prior art this plan builds on

- `data2g-bulk` (`docs/bulk.md` §2-§3, `bulk.py`): a block is one codeword; its identity is its CRC
  mask, `bulk.data_mask(stream, i)`, so a codeword can never be credited to the wrong block; the
  stream id is content-derived, so a second sending of the same text combines with the first; every
  block goes twice, RV 0 then RV 1 a burst later. Measured (§6): the copy moves the 10% block-loss
  point 2.4 dB (qpsk-r1/5) to 6.5 dB (qpsk-r3/4) on mpp/mps; RV 1 beats a Chase copy by 0.3-1.9 dB
  (§3). Units are bulk's identity scheme, offered to host clients.
- ARQ IR (`docs/arq.md` §4): two transmissions with IR move the 50% point 2-6 dB below one, and Chase
  gets 1-3.5 dB less than IR at rate 1/2 and above (`scripts/ir_study.py`).
- ion-hf bulletins (`docs/net.md`, "Bulletins: coded repair"; `crates/ion/src/link/net_code.rs`):
  systematic Cauchy over GF(256), 20 B pieces, blocks of up to 128; members report the first block
  they lack and how many pieces short; each round sends as many new repair pieces as the neediest
  member needs.
- Ragtime (`docs/PROTOCOL.md`, "Seals and repair", "Parity", "Broadcast files"): Cauchy RS over
  GF(2^16) (`core/include/ragtime/files/Parity.h`), parity pieces never sent twice, wants sized by
  `wantFor`, repairs in the slowest gear an asker needs, one-frame `RepairRequest` with the asker's
  S/N (`Records.h`).

## 2. The framework

### 2a. Units: the codeword-level transport

A unit is one codeword's payload, named by the application.

- Name: (group key, stream, index). Stream is 24 bits, index 16 bits, both chosen by the app. The
  host never invents names. Recommended: stream = 24 bits of a digest of the content (a file's
  digest), as `bulk.stream_id` does, and index = the erasure code's symbol name (data and parity
  pieces alike).
- Size: exactly `codes.payload_bytes(mode)` for the unit's mode. The host zero-pads a short one.
  LDPC and GF(M) modes take units; polar data modes (polar-k96, polar-k192) take them at RV 0 only
  (Chase); the 4 B `ack-*` modes refuse them.
- Mode: each unit names its mode (below, §3). Units of different modes go in different bursts, as
  frames of different modes do.
- On air: a version 4 burst is the v3 control (header, group or group+from, reports), a `T_UNITS`
  TLV, then the frames region exactly as v3 (may be empty), then the unit slots at the burst's end.
  - `T_UNITS` (type 4): stream (3 B), then runs: [first index (2 B)][(count - 1) << 2 | rv (1 B)].
    A run is up to 64 consecutive indices at one RV. The units fill the burst's last slots in run
    order, so their slots follow from n_cw. 3 B a run, against about 19 B of record header a piece
    today.
  - A unit's CRC mask: `unit_mask(key, stream, index)` = CRC-32 over a domain byte, the key, stream
    and index (or 1 if 0), a new function next to `phy.mask_value` (whose 4-byte input can't hold
    the name). No slot, start or burst in it: that is what lets any later send combine. The
    scrambler is position-only (`codes.scramble_seed`), and `codes.plain` already flips each slot's
    soft bits before combining.
  - Header byte as v3 (`[version 4 | dup | reserved | n_ctl - 1]`), version 4. A v3 receiver that
    decodes the control skips it as another kind of burst (`read_burst_control` checks
    `c0[0] >> 4 != VERSION`, then `on_burst` returns None on a converged slot 0).
  - A v3 receiver that loses the control falls back to `find_data`, which would claim the frames
    region (CPM always goes there: polar never reports "converged") and count the unit slots as a
    spurious `LOST`. So a v4 burst's frames use `data_start_mask(slot, start + 32, key)`: the
    epoch field's top bit (6 bits, starts at most 8) marks v4, and no v3 probe matches it.
  - What fits: OFDM controls are 10-396 B a codeword (polar-k96's 10 B the least), up to 4
    codewords, so many runs fit. CPM's
    control is one 20 B polar codeword (`docs/arq.md` §4): header 1 + group 10 + `T_UNITS` with one
    run 8 = 19 B. group+from (18 B) leaves no room, so a CPM unit burst sends the plain group (open
    question 2). `BCAST UNITS` refuses nothing at open time; a burst whose control can't hold its
    runs is cut to the runs that fit, the rest wait for the next burst.
- Control lost: units are not found (no `find_data` probe: their masks need the name, and probing
  names multiplies false accepts). They count as lost units, and the app's repair covers them. DUP
  control is the cheap protection and is recommended on unit ports.
- Alignment is exact by construction: one unit, one codeword, one erasure. Frames keep v3's
  alignment rules for whatever stays a frame (chat, announces, requests).

### 2b. Soft retention: buffers keyed by name

The receiving host keeps a failed unit's soft bits, and the name is the handle.

- Where: a new store owned by `KissLink` (native: `kisslink`), not the ARQ's `Engine.store`. The ARQ
  store holds float64 mother buffers keyed by session codeword and is cleared at session start
  (`engine.py`); mixing the two risks both.
- Decode path: for each unit slot, `ModemRx.decode(slot, unit_mask, rv, key)` against the unit
  store. Python's `ModemRx` takes its store at construction and the engine shares one `ModemRx`
  between the ARQ and KISS (`engine.py`: `PHY.ModemRx(r, self.store, ...)`, then
  `self.kiss.on_burst(r, rx)`). So Python gains native's existing form, `decode_stored(slot, mask,
  rv, stored)` (`native/core/arq/phy.hpp`, `PlainRx::decode_stored`), and the unit store calls that.
- Key: (group key, stream, index, code). Code is the code's identity, (code, k, coded_bits) for
  LDPC (`codes.ldpc_code` depends on k and coded_bits only) and the spec name for GF(M) and polar.
  Phase 3 keys by submode; phase 5 widens to the code (cross-mode combining, below).
- Entry: int8 soft values over `codes.ir_extent(spec, top)` only, in code order and unscrambled
  (what `codes.combine` builds), a fixed LLR step (1/8, clip at +-15.875; E1 checks the loss), the
  set of RVs received, receptions count, first and last heard, and an MI estimate.
  - The ARQ's entry is float64 over the whole mother buffer: 38 KB a unit in qpsk-r1/2 (4800 x 8 B).
    int8 over the extent received is 1920 B at RV 0, 3840 B after RV 1.
  - GF(M) (CPM data): q log-likelihoods a symbol, `codes.soft_len` 2560-6144 values, quantised
    per symbol relative to its largest. 6 KB a unit in fsk32r62.
- Combining: automatic. A unit heard again under the same key adds into its entry, any slot, any
  burst, any sender, any RV. Decoded: the entry goes, and the name joins a short done-list (the last
  4096 names) so later copies are not decoded again.
- MI estimate: the stored mutual information over k, `codes.buffer_mi(spec, buf, top)` (new; for an
  LLR L, 1 - log2(1 + e^-|L|) summed over the extent). Near 1 or above, one more RV likely decodes;
  near 0.2, a fresh parity unit is the better buy. It inherits any LLR overconfidence (short bursts
  underestimate spread), so E1 calibrates P(decode after one more RV | mi) per mode before any app
  leans on it.
- Lifetime and caps: per port, `KEEP` seconds since the last reception (default 2 h, matching
  Ragtime's 2 h listener state) and a host-wide cap in MB (default 32: about 2% of the Pi 4's 1.7 GB).
  Over the cap, the least recently heard entry goes. Every removal that isn't a decode is reported
  (`GONE`), so the app never believes in a buffer that isn't there.
- Only for open ports with units on. A promiscuous listener keeps nothing.
- Export (phase 5, optional): `SOFT GET` returns an entry as int8 in code order with its header
  (mode code, name, RV set, step); `SOFT PUT` adds one in. Uses: survive a host restart (Ragtime keeps
  its broadcast parts on disk across restarts; soft bits would not survive one), and diversity across
  receivers (a second receiver's soft bits shipped over the network). The format then becomes an
  interface tied to each code's mother-buffer layout, which is why it waits.

Explicit "combine later". With names, the useful explicit requests are few:

- `SOFT FORGET`: the app rebuilt the block from parity, or gave up; free the memory now.
- `SOFT KEEP`: extend a stream's lifetime (a file the operator still wants).
- `SOFT RETRY` with known bytes (phase 5): decode a held unit again with bytes the app knows pinned
  (`codes.pin_buffer` with a known-bits array; `codes.zero_tail` builds one today). The use: a
  file's last unit, whose zero padding the app knows from the file length, and text payloads with a
  fixed layout. Value unmeasured; E1 can size it with `scripts/ctl_prior_study.py`'s method.

### 2c. Sender control of redundancy versions

- The app names the RV of each unit it sends (0-3; `codes.rv_cycle` is 4 for LDPC and GF(M), 1 for
  polar). The host sends `codes.encode(spec, payload, rv, unit_mask, index=slot)`. No sender state:
  the app supplies the payload every time, so the host keeps nothing it sent.
- The receiver learns each unit's RV from its run in `T_UNITS` (2 bits), as the ARQ's `rv`
  extension does per resend.
- What the app chooses, and why it must choose:
  - RV 1-3 add fresh parity (IR): better than Chase by 0.3-1.9 dB at code level (`docs/bulk.md` §3),
    more at high rates.
  - But in every mode of rate 1/2 and up, an RV 1 codeword can't decode alone (`docs/bulk.md` §3). A
    station that holds nothing for the unit (it lost the whole burst or its control, or joined late)
    gains nothing from it. RV 0 again serves those stations and still Chase-combines for the rest.
  - Proactive copies: the app may send every unit twice from the start (RV 0, then RV 1 a burst or
    more later), as `data2g-bulk` does. On fading paths this beat the plain lower-rate mode of
    similar throughput at 1% (`docs/bulk.md` §6). This is a first-pass policy the app can now choose.
- RV naming is per unit, not per burst, so a repair burst can mix RV 0 copies and IR increments.

### 2d. Erasure repair at the app layer, and where soft combining fits

The modem doesn't erasure-code. Ragtime and ion-hf already have their own codes, symbol sizes and
feedback, and the code is bound to the app's notion of a block, a seal and a want. What the modem
gives the app's code:

- Symbols are units. A Ragtime piece becomes one unit (pieces of `payload_bytes(mode)`, rounded down
  to even for GF(2^16): the odd n10 modes lose a byte a unit). An ion-hf-like design packs
  floor(pb / 20) of its 20 B pieces into a unit, so a lost codeword is a whole number of erasures.
- Names are the code's symbol names. Parity units are units with index >= the piece count, as
  Ragtime's parity pieces are named today.

Two repairs for one hole, and when each pays:

| repair | who it helps | needs | costs |
|---|---|---|---|
| a new parity unit (RV 0) | every station short of that block, whatever it lacks | the station decodes it alone | one codeword of airtime |
| an IR increment of unit u (next RV) | only stations lacking u that hold soft bits for it | the combined buffer decodes | one codeword of airtime |
| an RV 0 copy of unit u | stations lacking u, held or not | either | one codeword |
| parity in a slower gear (Ragtime today) | every short station that copies the slower gear | the station decodes it alone | 1.5-5x the airtime a unit |

- Stations above the mode's knee lose units to fades, each a different set. A fresh parity unit
  fills a hole at every one of them. Parity wins.
- A station near or below the knee loses most units, and fresh units in the same mode fail as the
  first ones did. Today Ragtime moves its repair to a slower gear, whose units can't combine with
  what it already heard. An IR increment uses that energy: for qpsk-r1/2, RV 0 alone decodes at
  +1.1 dB Es/N0 and RV 0 + RV 1 at -2.7 dB (`docs/bulk.md` §3), 3.8 dB for 1x more airtime a held
  unit, against about 4.5 dB for 2.5x airtime from qpsk-r1/5 and a first pass thrown away. But an
  increment helps one station's one hole.
- So the sender's choice is per unit and per asker, and needs one more piece of feedback: which
  holes each asker holds soft bits for, and how close each is (the MI estimate).

Feedback from many receivers (a request carries this; it is app protocol, not modem):

- As today: per block (Ragtime: per file under a seal), how many symbols short, the S/N.
- New: the held holes worth an increment, as a bitmap over the block's units or a list of indices,
  only those whose MI is at least a threshold (E1 sets it). Ragtime's request is one frame: a
  bitmap of a 128-unit block is 16 B.

A reference sender policy (to be measured in E2 before anyone adopts it):

1. For each unit u listed held by any asker: value(u) = sum over those askers of P_combine(asker,
   u), from the MI each reported.
2. For a parity unit: value = sum over short askers of P_alone(asker), from each asker's S/N and the
   mode's single-shot curve (`codes_data/mode_thresholds.json`, as Ragtime's gearbox reads it).
3. Send greedily, highest value per airtime first, counting each asker's need down as it is served,
   until every asker's want is met or the round's airtime cap is reached. An increment goes at the
   next RV the sender hasn't sent for u; RV 0 instead if some asker lists u as lacking and not held.
4. Increments stay in the unit's own mode (code); parity units may go in any gear.

Multiple senders: the store key has no sender in it. A relay, or a second station that decoded the
file, may send increments of the same units, and they combine. This needs content-derived streams
(one stream never names two contents); the app's digests catch the rest.

## 3. Host interface

The same in `data2g/host.py` + `data2g/kisslink.py` + `data2g/tnc.py` and in
`native/core/host/host.cpp` + `native/core/kisslink/` + `native/app/tcp_servers.cpp`.

### 3.1 Commands (command port)

| command | reply | does |
|---|---|---|
| `BCAST UNITS n ON [KEEP s]` | OK / WRONG | port n takes and delivers units; soft bits kept s seconds (default 7200). WRONG from a host without units: the app's feature test |
| `BCAST UNITS n OFF` | OK | no units delivered; its entries dropped |
| `BCAST SOFT n FORGET stream [from to]` | OK | drop entries of that stream (indices from..to-1) |
| `BCAST SOFT n KEEP stream s` | OK | that stream's entries kept s seconds from now |
| `BCAST SOFT STATS` | `SOFT entries bytes cap` | the store's size, for a status line |
| `UNITMODES` | one `UNITMODE name code payload-bytes rvs max-units` line per mode, then OK | mode code (`modes.encode`), unit payload, distinct RVs (1 or 4), most units a burst carries with a one-codeword control |

`MODE` lines are not extended: Ragtime parses a `MODE` line only with exactly 7 words
(`core/src/link/Data2GLink.cpp`, the `MODE` branch). `HEARD` gains nothing either: Ragtime reads its
fourth word as the call.

### 3.2 KISS (data, both ways)

New command 0x0D (low nibble, port in the high one), unit frames:

- App to host: [ack tag 2][mode code 1][stream 3][index 2][rv 1][payload, up to payload bytes].
  Tag 0 asks no ack; otherwise the host sends the ACKMODE ack (0x0C, the tag) once the burst carrying
  it has gone, as for frames. A unit with a mode outside the cap, an `ack-*` mode, a payload too
  long, or rv not 0 on polar is dropped with `BCAST n DROPPED 1`.
- Host to app: [stream 3][index 2][status 1][rvs held 1][mi 1][payload if decoded].
  - status 0 decoded (payload follows), 1 held (failed, kept), 2 lost (failed, not kept: units off,
    or the store full of newer entries), 3 gone (an entry dropped: lifetime or cap).
  - mi: the MI estimate x 128, 0-255.
- Sent only to ports with units on. Current clients are safe: `tnc._KissHandler` passes unknown
  commands to `KissLink.command` (which ignores all but 2 and 3 now), and Ragtime's `KissDecoder` keeps only
  data and ACKMODE frames (`Data2GLink.cpp`, `feed`).
- A burst with units still gets `BCAST n HEARD`; `LOST k` keeps counting frames only.

### 3.3 Versioning and compatibility

| what | change | bump |
|---|---|---|
| PHY (`config.PROTOCOL_VERSION`, 15) | none | none |
| ARQ (session `VERSION`) | none (the unit store is separate) | none |
| broadcast control header | version 4 on bursts carrying units; frame-only bursts stay 3 | `kisslink.VERSION` gains 4 alongside 3; release notes say v3-only hosts skip unit bursts whole, frames in them included |
| `T_UNITS` = 4 | new TLV, only in v4 | with the above |
| unit CRC mask | new identity function | with the above |
| host commands, KISS 0x0D | additive; old hosts answer WRONG / ignore | host only |
| soft export format (phase 5) | new interface | its own format byte |

Each wire step is its own GitHub release with the version bump, as the version rule asks.

## 4. Ragtime, concretely

What changes on Ragtime's side (Ragtime's protocol is its own, version 0 today; this is a request
to its team, not code here):

- Feature test: `BCAST UNITS n ON` after `BCAST OPEN D2G FROM call`. WRONG: stay on frames.
- `UNITMODES` beside `MODES`, for unit payload sizes.
- File pieces become units: stream = 24 bits of the file digest; piece size = the unit payload
  (even); `FileAnnounce` gains the stream and the mode the pieces went in. Announces, chat, seals and
  requests stay frames. The "every record names its sender" rule holds through the burst's FROM and
  the announce; on CPM unit bursts, which can't carry FROM, through the announce only (open
  question 2).
- First pass: units at RV 0. Optionally copies (RV 1 a burst later) on a file the operator marks for
  weak paths: bulk's scheme, now a per-file policy.
- Listener: unit frames with status 0 are pieces; status 1 records (index, mi) per file. Its
  `RepairRequest` adds, per file, a bitmap of held pieces with mi over the threshold. On a whole-file
  rebuild it sends `BCAST SOFT n FORGET stream`.
- Sender: the reference policy in §2d, inside its existing rounds, caps and gears. IR increments go
  in the file's first-pass mode; parity may still go in the repair gear.
- Gains to expect, to be measured in E2 before Ragtime commits: from units alone, piece loss equal
  to codeword loss (no k-fold amplification, §1.2) and about 19 B less header a piece; from
  retention and increments, the weak stations' completion several dB lower.

## 5. Costs and risks

- Memory. int8 entries: 1-4 KB a unit on OFDM at RV 0 (`codes.soft_len` 960-4000), 2.5-6 KB on
  GF(M). A 300 KB file in qpsk-r1/2 is 2650 units; a station that lost half holds 2.5 MB; in
  fsk32r62-r1/2 (58 B units) the same file at half lost holds 16 MB. The cap and `GONE` bound it.
  Never store the ARQ's float64 mother buffers for units (38-100 KB each).
- CPU (Pi 4: under 10% of the dev machine's). Each unit slot is decoded once, as each data slot is
  today. A combined decode runs over `ir_extent`, up to the RVs received (2x the work a pass for
  RV 0 + RV 1). Holding costs nothing; quantising is O(n) a unit. No trial decodes against stored
  entries, ever. The DD budget (`phy.DD_BUDGET_S`) applies as to any burst.
- Lifetimes. Too short: an increment arrives after the entry went, and the station is told `GONE`
  before (it then reports the hole as not held, so the sender sends RV 0 or parity). Too long: a
  stale entry under a reused name dilutes a new combine. Content-derived streams make reuse mean the
  same content.
- Wrong combining (name collisions). Two contents under one name combine into garbage. The decode
  fails, or converges to one of the two, which passes its CRC (both were masked alike). The app's
  digest check catches a wrong piece; Ragtime checks whole-file digests and seal digests already.
  CRC-16 short codewords keep the 2^-16 false accept per check they have today; one check a unit
  slot, as the control names it.
- Poisoning. Anyone can send units under any name in any group (the group key is a hash of a public
  name). A deliberate wrong unit can keep a held unit from ever decoding. Bounded: after 4
  receptions without a decode the entry restarts from the newest reception. The app's parity is the
  fallback.
- Security and Part 97. Units are as plain as frames: the scrambler is unkeyed and the name sits in
  the control. Masks gate CRC checks only (`docs/broadcast.md` §2). Soft bits hold nothing a listener
  couldn't receive, and they are kept only for groups this station opened. Export goes to local
  clients (the KISS port binds 127.0.0.1 by default); a client can `SOFT PUT` only into a port it
  could open.
- Control loss now costs a unit burst's units (2a). DUP control is recommended on unit ports.
- Interop. v3 hosts skip unit bursts entirely, frames in them included. An app that mixes frames
  and units must accept that old hosts lose those frames, or send them in a frame-only burst.

## 6. What not to do

- No erasure code in the modem: the symbol size, block, feedback and refusal rules are the app's.
- No soft bits for frames. They have no name that survives a resend (their mask depends on slot and
  data start), and naming them would put the k-codeword amplification back.
- No blind matching of new soft bits against stored entries (correlation or trial decodes): CPU,
  and every trial is another chance of a false CRC pass.
- No position, slot, burst or data start in a unit's name or mask.
- No reuse of the ARQ store for units, and no float64 mother buffers in the unit store.
- No RV 1+ as the only send of a unit that stations without soft bits need (rate 1/2 and up can't
  decode it alone).
- No extra words on `MODE` or `HEARD` lines.
- No combining across codes. Within a code family (same (k, coded_bits)) it is sound; across codes
  it is not.

## 7. Evaluation plan

Conventions from the memory rules: PEP-fair (`DATA2G_PEP_REF_DB=5`), the same trial seeds at every
SNR, paired comparisons, 10% and 1% points, conclusions in dB, fading weighted over deep AWGN.

- E1, code level (extend `scripts/ir_study.py`; about 150 lines; depends on nothing new).
  - int8 storage at step 1/8 and int4 against float, RV 0 + RV 0, RV 0 + RV 1, RV 0 + RV 1 + RV 2, every
    LDPC and GF(M) mode: loss at the 10% point. Accept int8 if within 0.1 dB.
  - MI calibration: P(decode after one more RV | stored MI) per mode, AWGN and the CCIR channels.
    Output: the threshold the requests use.
  - Known-bytes retry on a last unit (padding known), the gain in dB, to decide phase 5's `RETRY`.
- E2, broadcast repair (`scripts/app_repair_study.py`, new; about 500 lines; depends on
  `codes`, `modem`, `hfchannel`, `phy.ModemRx`, and a study-only numpy Cauchy GF(256) of about 60
  lines). PHY-level like `scripts/bulk_study.py`, so it runs before any host change.
  - One sender, N = 4 and 8 receivers, each an independent `hfchannel` run of the same audio, S/N
    offsets spread across the mode's knee (0, -2, -4, -6 dB from the strongest). A 30 KB file. mpp,
    mps, mpg, and AWGN; qpsk-r1/2, qpsk-r1/5, n10-qpsk-r1/5, fsk32r62-r1/2.
  - Feedback first genie (every request heard), then lossy at each receiver's S/N in its request mode.
  - Strategies, paired:
    - S0 Ragtime today: 255 B frames spanning k codewords, parity pieces, rounds by the neediest want,
      repair in the slowest asker's gear.
    - S1 units: one-codeword pieces, the same parity rounds.
    - S2 S1 + retention + the §2d policy (increments for held units).
    - S3 S2 + first-pass RV 1 copies.
    - S4 S2 with increments in a same-code narrower mode (phase 5's cross-mode).
  - Metrics: airtime until every receiver holds the file; the weakest receiver's S/N at which the
    file completes within 2x the first pass's airtime (10% and 1% of trials failing); peak store
    bytes and decodes a burst per receiver.
- Decision gates: phase 2 goes ahead if S1 beats S0 by at least 1 dB at the weakest receiver's 10%
  point on two fading channels. Phase 3 if S2 beats S1 by at least 1 dB there. A few percent of
  airtime alone decides nothing.
- E3, on air or HFSim loopback, after phase 3: Ragtime-shaped transfer between three hosts, Python
  and native, with the store's memory and CPU logged on a Pi 4.

## 8. Phasing

Each step stands alone and is useful without the next.

1. Studies E1 and E2. No host change.
   - Accept: tables in this doc for int8 loss, MI threshold, and S0-S4; the gates decided.
2. Units on air, RV 0 only, no retention.
   - Code: `kisslink` (v4 build in `next_burst`, `T_UNITS`, `unit_mask`, receive in `on_burst`),
     `tnc` (0x0D), `host` (`BCAST UNITS`, `UNITMODES`); native the same in `core/kisslink`,
     `core/host`, `app/tcp_servers`, and `core/monitor` shows unit bursts.
   - Accept: a unit round trip through the modem in every unit mode (pytest and ctest); a v3 host skips
     a v4 burst and reports nothing; frames before units in one burst still parse; a unit decoded under
     the wrong name fails its CRC; a CPM unit burst with one run fits its 20 B control; Python-native
     parity tests (`tests/native_subs`).
3. Soft retention and RV control.
   - Code: `decode_stored` in Python's `ModemRx`; the unit store (int8, caps, lifetime, done-list,
     MI) in `kisslink` and native; `SOFT FORGET/KEEP/STATS`; status 1-3 frames; rv honoured on send.
   - Accept: a unit failed at -3 dB below its knee and resent at RV 1 in another slot of a later burst
     decodes; the same across two senders; eviction at the cap reports `GONE`; an entry under a reused
     name with other content fails or yields a CRC-valid unit, never a crash; memory stays under the
     cap through a 2 h soak; E2's S2 reproduced through the host path within noise.
4. Ragtime adoption (Ragtime's repo): §4. Data2G's part: answer questions, fix what it finds.
5. Options, each on its own measured case: `SOFT GET/PUT`; `SOFT RETRY` with known bytes; keying by
   code for cross-mode combining (families sharing (k, coded_bits): qpsk-r1/3, 16qam-r1/3,
   w48-qpsk-r1/3 and n4-16qam-r1/3 all carry 76 B in 1920 coded bits; qpsk-r1/5 and w48-qpsk-r1/5;
   qpsk-r1/2 and w48-qpsk-r1/2; qpsk-r3/4 and w48-qpsk-r3/4; 16qam-r1/2 and w48-16qam-r1/2);
   re-basing `data2g-bulk` on units.

## 9. Open questions

1. Names: 24-bit stream + 16-bit index, app-chosen, no sender in the key so relays' sends combine.
   Is cross-sender combining wanted, or should the key include the sender's hash?
2. CPM unit bursts can't carry group+from in the 20 B control. Send the plain group (the app names
   the sender in its own records), or refuse units on CPM ports opened with FROM?
3. Frames and units in one v4 burst (v3 hosts lose those frames), or units-only bursts?
4. Store defaults: 32 MB host-wide, 2 h per port. Right for the Pi 4?
5. Is soft export (`SOFT GET/PUT`) wanted at all, and for which use: restart persistence, or
   diversity across receivers?
6. Should `data2g-bulk` move onto units eventually, so there is one mechanism?
7. Cross-mode combining within a code family: worth a phase, given that it lets a weak station's
   repair go in a narrower mode without discarding its first pass?
8. Control-lost unit bursts: accept them as erasures (DUP recommended), or design a back-reference
   (the next burst names the previous burst's units) later?
