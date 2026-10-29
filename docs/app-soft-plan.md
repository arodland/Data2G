# Application soft combining and repair: plan

Status: proposed 2026-10-28, for review. Revised 2026-10-29 after Ragtime's review. No code yet.
Nothing here changes `docs/arq.md` or the PHY. The future ARQ and ladder work that code families
could open up is in `docs/code-families-future.md`.

TLDR: Data2G provides the mechanism, and the app (Ragtime or any other) sets the policy. An app may
use none of the new tools, or only some, and nothing it doesn't use gets in its way.

- Units: a broadcast payload of exactly one codeword, named by the app (24-bit stream, 16-bit
  index). The name rides in the burst's control (`T_UNITS`) and in a position-free CRC mask, so a
  later send of the same unit combines in any slot, burst or pass, and from any sender.
- Soft store: the receiving host keeps a failed unit's soft bits under its name, keyed by group
  (int8, capped, with a lifetime). Entries survive the port closing and the client reconnecting;
  only a host restart loses them.
- RVs: the app sets each unit's redundancy version (0-3). The receiver combines whatever arrives.
  No feedback is required. Simple feedback-free policies are first-class (§3.2): RV 0 only,
  proactive copies, or the next RV of the same parity units.
- Erasure repair stays in the app. Per-unit feedback (which holes a station holds soft bits for) is
  an optional extra that an app may use, never something the mechanism needs (§3.3).
- Two wins, kept apart (§3.1):
  - Granularity: one erasure per codeword. In the wide modes Ragtime can get most of this alone,
    with compact one-codeword v3 frames.
  - Soft retention: this needs names, so it needs units.
- Wire: broadcast header version 4, on units-only bursts. Frame bursts stay version 3. No PHY or ARQ
  version change.
- Host: one KISS command (SETHARDWARE 0x06) with subcommands, plus the `BCAST UNITS` and `BCAST SOFT`
  commands. Python and native alike.
- Studies come first. Phase 2 needs units to beat compact frames by 1 dB at the weakest receiver.
  Phase 3 needs retention to beat units alone by 1 dB.

## 1. Today

What an app can already do:

- Learn codeword sizes from `MODES` (`host.mode_lines`).
- Align frames to codewords. Broadcast v3 (`docs/broadcast.md` §3) loses only the frames a failed
  codeword touches, and a frame of k x bytes-per-codeword - 3 bytes fills k codewords
  (`kisslink.framed_len`, `pack_frames`). Ragtime fills 255 B frames (`Airtime.cpp`,
  `pieceFrameBytes`), so a piece spans 2 codewords in qpsk-r1/2, 3 on CPM and 11 in n10-qpsk-r1/5.
- Name a group and a sender (`BCAST OPEN group [FROM call]`), protect the control (`BCAST MODE n DUP
  mode`, about 3 dB for one codeword more), and repeat frames.

What is lost today:

- Soft bits of failed codewords. `KissLink.read_burst_frames` decodes with no key, so nothing is
  kept between bursts.
- A resent frame cannot combine even in principle. Its mask holds the slot and the data start
  (`data_start_mask(slot, start, key)`), so its info bits differ from send to send. IR exists only
  in the ARQ (`codes.combine`, `codes.decode_buffer`; `Engine.store`, cleared per session) and in
  the `data2g-bulk` experiment.
- Erasure granularity is a frame. A piece is lost if any of its codewords fails. With independent
  codeword loss, 5% codeword loss loses 43% of 11-codeword pieces. That is a worst case: a fade
  takes neighbouring codewords together, so the real amplification is smaller. E2 measures it.

Prior art:

- `data2g-bulk` (`docs/bulk.md`): an illustrative experiment, to be replaced by what gets built in
  Ragtime. It put block identity in the CRC mask (`bulk.data_mask`), used content-derived stream ids
  so a second sending combined, and sent RV 1 copies.
  - Measured: copies moved the 10% block-loss point 2.4-6.5 dB (§6).
  - RV 1 beat Chase by 0.3-1.9 dB (§3).
- ARQ IR (`docs/arq.md` §4, `scripts/ir_study.py`): two sends with IR move the 50% point 2-6 dB.
- Repair at the app: Ragtime's Cauchy GF(2^16) parity (`Parity.h`, `docs/PROTOCOL.md` "Parity"),
  and ion-hf's Cauchy GF(256) bulletins (`docs/net.md`, `net_code.rs`).

## 2. The mechanism

### 2a. Units

- Name: (group key, stream 24 bits, index 16 bits), chosen by the app.
  - Content-derived streams are recommended: a hash over the content's digest and the unit payload
    size. The size must be in it because the same file cut into different unit sizes has different
    bytes under each index.
  - The index is the erasure code's symbol name. Ragtime's parity pieces already use indices at or
    above the piece count.
- Size: `codes.payload_bytes(mode)`; the host zero-pads a short unit.
- Modes: LDPC and GF(M) data modes take units at RV 0-3. Polar data modes take RV 0 only (Chase).
  The `ack-*` modes take none.
- Burst: units only. A version 4 burst is the v3 control plus `T_UNITS`, then unit slots. Frames
  (chat, announces, requests) stay in v3 frame bursts.
  - Ragtime's review asked for this: chat keeps its small v3 control.
  - Mixed bursts can be added later, if an app shows a need.
- `T_UNITS` (type 4): [flags 1][stream 3], then runs of [first index 2][(count - 1) << 2 | rv 1]. A
  run is up to 64 consecutive indices at one RV. Units fill the slots in run order.
- Mask: `unit_mask(key, stream, index)`, CRC-32 over a domain byte and the name (1 if 0). It is a new
  function beside `phy.mask_value`. No slot, burst or mode goes into it. The scrambler is
  position-only (`codes.scramble_seed`), and `codes.plain` unscrambles each slot, so any later send
  combines.
- v3 hosts:
  - With the control decoded, a v3 host skips the burst as another kind (`read_burst_control`).
  - With the control lost, its `find_data` probes the unit slots under `data_start_mask`. They fail,
    so it reports nothing. A units-only burst has no frames region, so no second probe start is
    needed.
  - The first draft's `start + 32` trick would have doubled `find_data`'s CRC checks, and with them
    its false-accept rate. It is dropped.

Control size and loss. Every unit depends on its burst's control:

- A lost control loses every unit of the burst, and nothing is stored, because the names are
  unknown. An IR combine of unit u needs the controls of both bursts that carried it.
- There is no `find_data` fallback for units: probing names multiplies false accepts.
- On OFDM the control is a codeword of the data code (`modes.ctl_spec`), so it fails about as often
  as a unit does.
- Fit:
  - CPM's 20 B control: header 1 + group 10 + `T_UNITS` 6 + one run 3 = 20 B. No room for FROM (Q2).
  - n10-qpsk-r1/5 (23 B codewords): header + group+from + `T_UNITS` is 24 B before the first run,
    so the control already takes 2 codewords. Each scattered run adds 3 B.
- Guidance for apps:
  - Send contiguous runs. Fresh parity is contiguous by construction, and so is §3.2's "next RV of
    the same parity units".
  - In narrow modes, send the plain group rather than group+from on unit bursts.
  - Use DUP on unit ports at the floor: 3 dB on the control for one codeword a burst. E2 measures it
    with DUP's airtime counted.

### 2b. Soft store

- Key: (group key, stream, index, code), where code means the submode, or the code family when
  phase 3 includes families (§2e).
- Lifetime: entries are kept per group, not per port.
  - Today `host.client_gone` calls `kiss.reset()`, which closes every port. Entries outlive that:
    each lives `KEEP` seconds after its last reception (default 2 h), whether its port is open or not.
  - A client that reconnects and reopens the group finds them, and `BCAST SOFT n LIST [stream]`
    reports each held entry as a status frame.
  - Only a host restart loses the store.
  - New units are stored only while some port has the group open with units on. A promiscuous
    listener keeps nothing.
- Cap: host-wide, in MB (default 32, about 2% of the Pi 4's 1.7 GB; Q4). Over the cap, the least
  recently heard entry goes. Every removal that isn't a decode is reported (`GONE`).
- Entry contents:
  - int8 soft values over the received positions of the mother buffer, in code order and
    unscrambled (what `codes.combine` builds). The step is 1/8, clipped at +-15.875; E1 checks the
    loss.
  - The received extents, the receptions count, first and last heard, and the MI ratio (below).
- Entry sizes: 1-4 KB a unit at RV 0 on OFDM (`codes.soft_len` 960-4000), 2.5-6 KB on GF(M). The
  ARQ's float64 mother buffers (38-100 KB) are never used for units.
- Decode path: Python's `ModemRx` gains native's `decode_stored(slot, mask, rv, stored)`
  (`native/core/arq/phy.hpp`). The engine's shared `ModemRx` then serves the unit store without
  touching `Engine.store`.
- Done-list: a bitmap per live stream (1 bit an index, at most 8 KB), dropped with the stream's last
  entry. A late copy of a decoded unit is skipped. That matters for size: one 300 KB file in
  n10-qpsk-r1/5 is about 13 600 units.

MI ratio, reported with each held unit:

- It is computed from the consistent-LLR estimate, which needs no knowledge of the data. For
  position i with LLR L_i:
  - I_i = 1 - H_b(1 / (1 + e^|L_i|)), where H_b is the binary entropy;
  - R = (sum of I_i over the received positions) / k.
- The first draft's 1 - log2(1 + e^-|L|) is the known-data form, and it is biased high. At |L| = 1
  it gives 0.55 bits; the consistent form gives 0.16.
- GF(M): normalise each symbol's q log-likelihoods to a posterior P. Then I_s = log2 q - H(P) bits,
  and R = (sum of I_s over the received symbols) / k.
- How to read R: decoding becomes likely once R passes 1 + d. The margin d is per mode and per
  channel, fitted in E1.
- R inherits any LLR overconfidence. No app should lean on it until E1's calibration is in.

Bad receptions (collisions, QRM):

- A failed slot hit by another station's burst or by QRM gives confident wrong LLRs. Retention
  would keep them and add them in.
- Quality gate before storing: a slot is stored only if its R is consistent with what the burst's
  pilot SNR predicts for the mode, within a margin fitted in E1/E2. The prediction comes from
  `phy.measure`'s `snr_est` and the predictor's capacity tables (`predictor._capacity`), as linksim
  does. Outliers are reported as status lost, with a quality reason.
- Restart guard: an entry whose R is well past 1 + d and still fails is suspect, and restarts from
  its newest reception. A count backstop is sized from E1's receptions-to-decode for weak stations,
  so a station that genuinely needs four receptions isn't penalised.

### 2c. RVs

- The app names each unit's RV. The host encodes `codes.encode(spec, payload, rv, unit_mask,
  index=slot)` and keeps nothing it sent.
- The receiver reads the RV from the run, maps it to mother-buffer positions with the burst's mode
  (`codes.rv_positions`), and adds it in.
- Facts an app should know:
  - IR beats Chase by 0.3-1.9 dB at code level.
  - At rate 1/2 and up, an RV 1 codeword can't decode alone (`docs/bulk.md` §3), so it is useless to
    a station that holds nothing for that unit.

### 2d. Names without the sender

- Default (Q1): no sender in the mask or the key. A second station that decoded the content, such
  as a relay, can send the same units, and they combine. No extra code is needed for this; it
  follows from the name.
- Risk, two senders reusing a stream id:
  - With random 24-bit ids, the chance of any collision among m live streams is about m^2 / 2^25
    (1.2e-5 at 20).
  - Content-derived ids make a shared id mean shared bytes.
  - A collision that does happen either fails to decode, or decodes to the other content with a valid
    CRC. The app's digest catches it.
- Unbound units also let anyone fill or poison a stream. The group key is a hash of a public name;
  masks gate CRC checks only (`docs/broadcast.md` §2).
  - A whole-file digest says the rebuild is wrong, not which unit.
  - An app can drop the file's parity and ask again (what Ragtime does on a digest failure today), or
    carry per-block digests in its announce or seal.
- A relay must send the same group, name and payload bytes, in a mode of the same code, at any RV it
  likes. A fresh RV is best.
- Bound units (the sender's `kisslink.station_hash` in the mask) are deferred.
  - They cost 2 B of control and don't fit on CPM.
  - The hash is public, so they stop accidents, not attacks.
  - No one has asked for them. They come back only if E2 or real use shows a collision cost.

### 2e. Code families (only if E1 and E2 S4 show a gain)

- Two LDPC modes are one family when they share base graph, Z and k (`ldpc.layout`, `ldpc.qc_code`).
  `QCLDPC.mother()` is then the same buffer, so their soft bits add position by position.
  Constellation, band, interleaver and n may differ.
- Families today:
  - qpsk-r1/3, 16qam-r1/3, n4-16qam-r1/3, w48-qpsk-r1/3 and n4-qpsk-r2/3;
  - 16qam-r1/2, w48-16qam-r1/2 and w48-64l-r2/3;
  - w48-qpsk-r2/3 and w48-16qam-r1/3;
  - the w/w48 pairs of qpsk-r1/5, qpsk-r1/2 and qpsk-r3/4.
- Cost: `codes.family`, decode by position extent in place of a top RV, and three tests: about
  130 lines a host. One of the tests checks that every member's encode is a prefix of the mother's.
- Without a measured gain, the store keys by submode, and nothing else changes.

## 3. Using it: policy is the app's

### 3.1 Two wins, kept apart

Payload per codeword, by how a piece is carried (Ragtime's review; header about 19 B today):

| mode (B a codeword) | today: piece / codewords | one-codeword v3 frame, today's header | one-codeword v3 frame, 6 B header | unit |
|---|---|---|---|---|
| qpsk-r1/2 (116) | about 209 / 2 | 94 (81%) | 107 (92%) | 116 (100%) |
| fsk32r62-r1/2 (58) | about 151 / 3 | 36 (62%) | 49 (84%) | 58 (100%) |
| n10-qpsk-r1/5 (23) | about 230 / 11 | does not fit | 14 (61%) | 22 (96%) |

- Granularity alone needs no Data2G change in the wide modes: Ragtime can choose k = 1 in
  `pieceFrameBytes` with a compact header. Units win clearly in the FSK and narrow modes.
- Both a compact header and units drop the callsign from each piece record. That is Jeff's rule to
  relax (Q2).
- Soft retention needs names, so it needs units. That is the only part a compact frame can't give.

### 3.2 Feedback-free uses (first-class)

None of these needs anything new in a request:

- RV 0 only: units as compact pieces, retention off. This is the granularity win, nothing more.
- Chase repeats: send a unit again at RV 0. It serves stations that hold nothing, and it combines for
  those that hold something.
- Proactive copies: every unit twice, RV 0 then RV 1 a burst or more later (bulk's scheme).
- Parity IR (S2p, from Ragtime's review): round 1 sends parity units at RV 0 as today. Round 2 sends
  the next RV of those same parity units, beside any fresh ones. A station that failed a parity unit
  combines; one that decoded it skips the copy (done-list). The request stays "N short, S/N", and the
  runs stay contiguous.

### 3.3 Optional per-unit feedback

An app may ask stations which holes they hold soft bits for. It costs request airtime, at the
stations least able to send.

- Size per file, not per block. One 300 KB file in qpsk-r1/2 is 2650 units, a 331 B bitmap, more
  than a 255 B frame. So send held holes as index ranges, which stay short because fades lose
  neighbouring units together. Send only those with R over the threshold.
- Weak askers send in the slowest gears, about 30-60 s for 350 B at the lowest rungs, and they
  collide more on a busy net.
- One example policy (not the design): value each held unit by the sum of its askers'
  P(decode | R, one more RV), and each fresh parity unit by the sum of short askers' P(decode alone
  | S/N). Send greedily by value per airtime, until wants are met or the round's cap is reached.
- It goes in only if E2 shows it beats S2p by the gate margin, with requests charged at their real
  airtime, loss and collisions.

## 4. Host interface (Python `host.py`, `kisslink.py`, `tnc.py`; native `core/host`, `core/kisslink`, `app/tcp_servers.cpp`)

| command | reply | does |
|---|---|---|
| `BCAST UNITS n ON [KEEP s]` | OK / WRONG | units on for port n's group; KEEP sets the group's entry lifetime. WRONG from an old host: the feature test |
| `BCAST UNITS n OFF` | OK | no units delivered on port n; the group's entries stay until their lifetime ends |
| `BCAST SOFT n LIST [stream]` | status frames, then `BCAST SOFT n LISTED k` | the held entries, after a reconnect |
| `BCAST SOFT n FORGET stream [from to]` | OK | drop those entries (the app rebuilt the block or gave up) |
| `BCAST SOFT n KEEP stream s` | OK | extend a stream's lifetime |
| `BCAST SOFT STATS` | `SOFT entries bytes cap` | |
| `UNITMODES` | `UNITMODE name code payload-bytes rvs max-units family` lines, then OK | mode code is `modes.encode` |

`MODE` and `HEARD` lines stay unchanged. Ragtime parses a `MODE` line only when it has exactly 7
words, and reads `HEARD`'s fourth word as the call (`Data2GLink.cpp`).

KISS: KISS has only 4 bits of command, so the new features share SETHARDWARE (0x06), with a
subcommand byte first.

- What is used today:
  - Data2G treats 0x00 data and 0x0C ACKMODE as frames, honours 0x02 P and 0x03 SLOTTIME, and
    ignores the rest (`KissLink.command`).
  - Ragtime sends 0x00, 0x0C and 0x02, and keeps only 0x00 and 0x0C from the host (`KissDecoder::feed`).
  - Dire Wolf's SETHARDWARE dialect is ASCII text (from its documentation; to confirm before
    phase 2).

| sub | direction | contents |
|---|---|---|
| 0x01 UNIT_OUT | app to host | [ack tag 2][mode code 1][stream 3][index 2][rv 1][payload] |
| 0x02 UNIT_IN | host to app | [stream 3][index 2][status 1][reason 1][R x 128, 1][payload if decoded] |
| 0x03 SOFT_RECORD | both | reserved for export/import (§7, phase 4) |
| 0x04-0x0F | | reserved: units and soft growth |
| 0x10 | | reserved: ports beyond 15 (`docs/broadcast.md` §6) |
| 0x11-0x1F, 0x7F-0xFE | | reserved |
| 0x20-0x7E | | never assigned: printable ASCII, so another TNC's text dialect is logged and ignored |
| 0xFF | | escape to a two-byte subcommand |

- UNIT_OUT details:
  - The ack is the plain 0x0C ack, sent once the unit's burst has gone. Tag 0 asks for none.
  - A unit in a mode outside the cap, an `ack-*` mode, too long, or rv not 0 on polar gets
    `BCAST n DROPPED 1`.
- UNIT_IN status:
  - 0: decoded, payload follows.
  - 1: held.
  - 2: lost (not kept; the reason says why: units off, cap, or the quality gate).
  - 3: gone.
  - 4: import refused.
- 0x06 frames go only to ports with units on. An old host logs and drops a 0x06 frame, so nothing
  goes on air by accident.

Versions:

| what | change | bump |
|---|---|---|
| PHY (`config.PROTOCOL_VERSION` 15), ARQ (session `VERSION`) | none | none |
| broadcast control | version 4 on units-only bursts; frame bursts stay 3 | `kisslink.VERSION` gains 4 beside 3; a release with the bump |
| `T_UNITS`, `unit_mask` | new, v4 only | with the above |
| commands, KISS 0x06 | additive | host only |

## 5. Ragtime

From Ragtime's review, and what the mechanism offers each point:

- Callsign rule (Q2):
  - Unit pieces carry no callsign of their own. The burst's FROM names the sender on OFDM; on CPM
    nothing in the burst can.
  - The mechanism offers: group+from on OFDM unit bursts, the plain group on CPM, and nothing in the
    unit itself.
  - Binding pieces to a sender through the announce and seal is Jeff's call.
- Chat stays in v3 frame bursts (units-only v4, §2a).
- Mixed nets don't matter yet: the net is 3-4 stations that can upgrade together. The v3 behaviour
  above costs nothing either way.
- Loss share: `LOST k` counts frames only. Ragtime's `wantFor` loss share comes from the piece
  indices it heard (`BroadcastFiles::lossOf`), which carries over to units.
- The flight recorder and `--replay` must keep 0x06 frames.
- Work on Ragtime's side:
  - small: 0x06 dispatch, `UNITMODES`, the feature test;
  - medium: pieces as units, stream ids and the `FileAnnounce` change; the repair policy (S2p first);
  - medium to large: fake hosts and `ragtime_fake_air` playing units and a soft store.

## 6. Costs, risks, and what not to do

- CPU (Pi 4): each unit slot is decoded once, as each data slot is today. A combined decode costs
  about 2x for RV 0 + RV 1. There are no trial decodes against stored entries. `phy.DD_BUDGET_S`
  applies.
- Memory: a 300 KB file at half lost is 2.5 MB in qpsk-r1/2 and 16 MB in fsk32r62-r1/2. The cap and
  `GONE` bound it.
- Poisoning and collisions: the quality gate and restart guard (§2b), the digest options (§2d).
  E2 includes interference.
- Control loss: §2a. DUP is the app's lever.
- Don't:
  - put an erasure code in the modem;
  - give frames soft bits;
  - match soft bits blindly against stored entries;
  - put slot, burst or mode in a name;
  - reuse the ARQ store;
  - send only RV 1+ of a unit stations without soft bits need;
  - add words to `MODE` or `HEARD`;
  - combine across codes.

## 7. Evaluation and phasing

Conventions: `DATA2G_PEP_REF_DB=5`, seeded paired ladders, 10% and 1% points, conclusions in dB,
fading weighted over deep AWGN.

E1, code level (extend `scripts/ir_study.py`, about 200 lines):

- int8 and int4 storage against float, for RV 0+0, 0+1 and 0+1+2 in every LDPC and GF(M) mode.
  Accept int8 within 0.1 dB.
- Calibrate both MI forms (binary and GF(M)) to give d per mode and channel.
- Receptions-to-decode for weak stations, to size the restart guard.
- The quality gate's margin.
- The family prefix test.

E2, broadcast repair (`scripts/app_repair_study.py`, about 600 lines):

- Setup: PHY level like `scripts/bulk_study.py`, so it runs before any host change.
  - One sender, 4 or 8 receivers on independent `hfchannel` runs, spread 0 to -6 dB across the
    mode's knee.
  - A 30 KB file; mpp, mps, mpg and AWGN.
  - qpsk-r1/2, qpsk-r1/5, n10-qpsk-r1/5 and fsk32r62-r1/2. The CPM floor modes run through the real
    modem (`scripts/phy_session.py` style), not linksim.
- Strategies, paired:
  - S0: Ragtime as it really is (k codewords a piece, k = 3 on CPM; parity in the asker's gear,
    repair floor, per-file gear and airtime cap; Ragtime offers to write this down).
  - S1': compact one-codeword v3 frames with a 6 B header.
  - S1: units at RV 0.
  - S2p: S1 + retention, with the next RV of the same parity units.
  - S2: S2p + per-unit feedback and §3.3's policy.
  - S3: proactive copies.
  - S4: families.
- Conditions:
  - requests charged at their real size and airtime in each asker's gear, with loss and collisions
    among the askers;
  - control loss from the real control code, with and without DUP, DUP's codeword counted;
  - an interference case over some unit slots (`data2g/interference.py`).
- Metrics: total airtime, requests included, until every receiver holds the file. Also the weakest
  receiver's S/N at which it completes within 2x the first pass's airtime, and peak store bytes.

Gates: at least 1 dB at the weakest receiver's 10% point on two fading channels.

| step | needs | ships |
|---|---|---|
| 1 | nothing | E1 and E2 tables in this doc |
| 2 | S1 beats S1' | units, RV 0, no retention: `T_UNITS`, `unit_mask`, units-only v4 bursts, 0x06 UNIT_OUT/UNIT_IN, `BCAST UNITS`, `UNITMODES`, `core/monitor` display |
| 3 | S2p beats S1 | the soft store (§2b), RVs, `BCAST SOFT`; families only if S4 beats S2p; §3.3 feedback only if S2 beats S2p (app side) |
| 4 | user and Jeff decide (Q5) | export/import |
| 5 | | Ragtime adoption, in its repo |

Acceptance tests per step:

- Step 2:
  - a unit round trip in every unit mode (pytest and ctest);
  - a v3 host reports nothing for a v4 burst, control decoded or lost;
  - a wrong name fails its CRC;
  - one run fits CPM's control;
  - Python-native parity (`tests/native_subs`).
- Step 3:
  - an RV 1 resend in another slot of a later burst decodes a unit that failed 3 dB below its knee,
    and so does one from a second sender;
  - entries survive `client_gone` and a port close, and expire on `KEEP`;
  - a cap eviction reports `GONE`;
  - a collision slot from `interference.py` is refused by the gate;
  - a 2 h soak stays under the cap;
  - the ARQ's tests are unchanged.
- Step 4, if built: a record round trip; a restart and re-import decodes the next RV as if there had
  been no restart; a changed shift table refuses the record; importing twice changes nothing.

Export/import (Q5, two positions):

- For: Ragtime said it tentatively wanted it, so soft data survives a host restart. Design if built:
  - one SOFT_RECORD per entry: magic, format version, family descriptor, a fingerprint of the code
    tables, the name, quantisation, receptions and times, R, extents, the int8 values, CRC-32;
  - 46 B plus 4 B an extent plus the values, so about 1-12 KB a unit;
  - import re-arms an entry only if none is held (never adds into one), and is refused on any check
    failure, including changed code tables.
- Against: Ragtime's review recommends not taking MBs of opaque records into its database.
  - With entries surviving reconnects (§2b), only a host restart loses them, which is rare.
  - Ragtime keeps its decoded pieces across restarts (`BroadcastParts`) and can ask for more parity.
  - The cost is a permanent binary format, plus 2.5-16 MB of records per 300 KB file.
- Proposal: defer until step 3 has run on air, then decide.

## 8. Decisions and open questions

- Q1, sender in the key: no sender, by default (§2d); bound units deferred.
- Q2, callsign on CPM unit bursts: open, for Jeff (§5).
- Q3, mixed bursts: units-only v4 bursts, as Ragtime asked (§2a).
- Q4, store defaults (32 MB, 2 h): open, for the user.
- Q5, export/import: open, for the user and Jeff (§7).
- Q6, data2g-bulk: prior art only.
- Q7, cross-mode combining in families: kept as cheap mechanism, gated on E1 and E2 S4 (§2e). The
  ARQ groundwork and family consolidation are in `docs/code-families-future.md`.
- Q8, control-lost unit bursts: erasures. No back-reference, since nothing measured asks for one.
  DUP and contiguous runs are the levers (§2a).
