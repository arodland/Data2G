# Application soft combining and repair: plan

Status: proposed 2026-10-28, for review; first answers folded in the same day (§9). No code yet.
Nothing here changes `docs/arq.md` or the PHY.

TLDR: give broadcast applications one framework in four layers. The modem names codewords and keeps
soft bits. The application owns erasure coding, feedback and policy.

- (a) Units: a new kind of broadcast payload, exactly one codeword, named by the application
  (24-bit stream, 16-bit index). The name rides in the burst's control (`T_UNITS`) and in a
  position-free CRC mask. A later send of the same unit combines, in any slot, burst or pass, and
  from any sender: a relay's copy fills the same hole.
- (b) Soft retention: the receiving host keeps a failed unit's soft bits (int8, capped, with a
  lifetime) under that name. The name is the handle. Entries are kept in code-family terms (§2f),
  so a unit resent in another mode of the same family still combines. The app is told which units
  are held and how close each one is. It can export a held entry and import it again after a
  restart (§2e).
- (c) RV control: each unit the app sends carries its redundancy version (0-3), listed in `T_UNITS`.
  The app picks Chase (RV 0 again) or IR (the next RV) per unit.
- (d) Erasure repair stays in the app (Ragtime's Cauchy GF(2^16) parity, ion-hf's Cauchy GF(256)).
  Units make an erasure exactly one codeword. The app's repair round then mixes fresh parity units
  (help every station) with IR increments of held units (help stations below the mode's knee).
- Wire: broadcast header version 4 for bursts that carry units. Frame-only bursts stay version 3.
  No PHY (`config.PROTOCOL_VERSION`) or ARQ session `VERSION` change.
- Host: KISS SETHARDWARE (0x06) subcommands for units and soft records both ways (no new command
  nibble), plus `BCAST UNITS` and `BCAST SOFT` commands, in the Python and native hosts alike. `MODE` and `HEARD` lines stay
  exactly as they are, because Ragtime's parser counts their words.
- Order: studies first (E1, E2). Each wire step goes ahead only if the study before it shows at
  least 1 dB at the weakest receiver's completion point.
- Future, out of scope: the same code families could let the ARQ change modes within a family
  without abandoning outstanding codewords (§11). Consolidating modes into fewer, larger families
  could gain at most the 2-4% of codewords that mode changes discard today (9-24% in floor fading
  cells). One linksim oracle study would decide it (§12).

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
- Protect the control. `BCAST MODE n DUP mode` sends each control codeword at RV 0 and RV 1, and
  the receiver combines the pair (`phy.ModemRx.raw_pair`, `decode_pair`), about 3 dB.
- Know what went out and what was lost. KISS ACKMODE acks; `BCAST n HEARD`, `LOST k`, `MISSED`,
  `DROPPED`.
- Repeat. An app may send the same frame again. The receiver decodes each copy on its own.

### 1.2 What is lost today

- Soft information of failed codewords. `KissLink.on_burst` decodes every slot with no key
  (`rx.decode(..., key=None)` in `read_burst_frames`), so a failed codeword's soft bits are dropped
  when the burst is done. A repeat starts from zero.
- A frame resent cannot combine even in principle. Its data codewords' CRC mask depends on the slot
  and the burst's data start (`data_start_mask(slot, start, key)`), so the info bits differ between
  sends, and so does the codeword. Nothing in the control names a frame either.
- IR exists only inside the ARQ (`docs/arq.md` §4: `rv` extension, `codes.combine`,
  `codes.decode_buffer`; store `Engine.store`, cleared per session) and in the illustrative
  `data2g-bulk` experiment (§1.3).
- Erasure granularity is a frame, not a codeword. A Ragtime file piece is one 255-byte frame. In
  qpsk-r1/2 (116 B a codeword) that is 2 codewords, in the FSK modes 3, in qpsk-r1/5 5, and in
  n10-qpsk-r1/5 (23 B) 11. A piece is lost if any of its codewords fails: at 5% codeword loss an
  11-codeword piece is lost 43% of the time, so the parity round must cover 43% of the file,
  not 5%.
- Per-piece header bytes. Each Ragtime piece carries `0xA0`, the record outline (type, length,
  callsign, seq, time) and its index (`docs/PROTOCOL.md`, "Records"; `Records.h`, `FilePiece`),
  about 18-20 B. That is fine in a 255 B frame, but 35-50% of a one-codeword piece in the 38-58 B
  FSK modes. So Ragtime cannot simply make pieces one codeword long today.

### 1.3 Prior art this plan builds on

- `data2g-bulk` (`docs/bulk.md` §2-§3, `bulk.py`) is an illustrative experiment, to be replaced by
  what gets built in Ragtime. It is prior art only, and no phase here builds on it. What it showed:
  - A block is one codeword. Its identity is its CRC mask, `bulk.data_mask(stream, i)`, so a
    codeword can never be credited to the wrong block.
  - The stream id is content-derived, so a second sending of the same text combines with the first.
  - Every block goes twice, RV 0 then RV 1 a burst later. The copy moves the 10% block-loss point
    2.4 dB (qpsk-r1/5) to 6.5 dB (qpsk-r3/4) on mpp/mps (§6). RV 1 beats a Chase copy by 0.3-1.9 dB
    (§3).
- ARQ IR (`docs/arq.md` §4): two transmissions with IR move the 50% point 2-6 dB below one. Chase
  gets 1-3.5 dB less than IR at rate 1/2 and above (`scripts/ir_study.py`).
- ion-hf bulletins (`docs/net.md`, "Bulletins: coded repair"; `crates/ion/src/link/net_code.rs`):
  systematic Cauchy over GF(256), 20 B pieces, blocks of up to 128. Members report the first block
  they lack and how many pieces short. Each round sends as many new repair pieces as the neediest
  member needs.
- Ragtime (`docs/PROTOCOL.md`, "Seals and repair", "Parity", "Broadcast files"):
  - Cauchy RS over GF(2^16) (`core/include/ragtime/files/Parity.h`); parity pieces are never sent
    twice.
  - Wants are sized by `wantFor`, and repairs go in the slowest gear an asker needs.
  - A `RepairRequest` (`Records.h`) is one frame and carries the asker's S/N.

## 2. The framework

### 2a. Units: the codeword-level transport

A unit is one codeword's payload, named by the application.

- Name: (group key, stream, index). Stream is 24 bits and index 16 bits, both chosen by the app;
  the host never invents names. The sender is not part of the name by default (§2d, "Relays and
  the sender in the key"). A port may require it (bound units, below).
- Stream ids should be content-derived: 24 bits of a hash over the content's digest and the unit
  payload size (§2d explains why the size must be in it). Index is the erasure code's symbol name,
  data and parity pieces alike.
- Size: exactly `codes.payload_bytes(mode)` for the unit's mode. The host zero-pads a short one.
- Which modes: LDPC and GF(M) modes take units. Polar data modes (polar-k96, polar-k192) take them
  at RV 0 only (Chase). The 4 B `ack-*` modes refuse them.
- Mode: each unit names its mode (§3). Units of different modes go in different bursts, as frames
  of different modes do. Modes of one code family carry the same unit payload and combine (§2f).
- On air: a version 4 burst is laid out as follows:
  - the v3 control (header, group or group+from, reports);
  - a `T_UNITS` TLV;
  - the frames region exactly as v3 (it may be empty);
  - the unit slots, at the burst's end.
- `T_UNITS` (type 4): [flags (1 B)][stream (3 B)][sender hash (2 B), bound only], then runs.
  - Flags: bit 0 means bound.
  - A run is [first index (2 B)][(count - 1) << 2 | rv (1 B)]: up to 64 consecutive indices at one
    RV.
  - The units fill the burst's last slots in run order, so their slots follow from n_cw.
  - 3 B a run, against about 19 B of record header a piece today.
- A unit's CRC mask is `unit_mask(key, stream, index, sender)`, a new function next to
  `phy.mask_value` (whose 4-byte input can't hold the name).
  - It is CRC-32 over a domain byte, the key, stream and index, and the sender hash only when
    bound (1 if the result is 0).
  - No slot, start, burst or mode goes into it. That is what lets any later send combine. The
    scrambler is position-only (`codes.scramble_seed`), and `codes.plain` already flips each slot's
    soft bits before combining.
- Header byte as v3 (`[version 4 | dup | reserved | n_ctl - 1]`), version 4. A v3 receiver that
  decodes the control skips the burst as another kind. `read_burst_control` checks
  `c0[0] >> 4 != VERSION`, and `on_burst` then returns None on a converged slot 0.
- A v3 receiver that loses the control falls back to `find_data`.
  - That would claim the frames region and count the unit slots as a spurious `LOST`. CPM always
    goes there, because polar never reports "converged".
  - So a v4 burst's frames use `data_start_mask(slot, start + 32, key)`. The epoch field's top bit
    marks v4 (6 bits; starts are at most 8), and no v3 probe matches it.
- What fits on OFDM: controls are 10-396 B a codeword (polar-k96's 10 B the least), up to 4
  codewords, so many runs fit.
- What fits on CPM: the control is one 20 B polar codeword (`docs/arq.md` §4).
  - Header 1 + group 10 + `T_UNITS` with one unbound run 9 = 20 B, exactly full.
  - group+from (18 B) and bound units (2 B more) don't fit, so a CPM unit burst sends the plain
    group, unbound (Q2, §10).
  - A burst whose control can't hold its runs is cut to the runs that fit; the rest wait for the
    next burst.
- Control lost: the units are not found. There is no `find_data` probe: their masks need the name,
  and probing names multiplies false accepts. They count as lost units, and the app's repair covers
  them. DUP control is the cheap protection and is recommended on unit ports (Q8, §10).
- Alignment is exact by construction: one unit, one codeword, one erasure. Frames keep v3's
  alignment rules for whatever stays a frame (chat, announces, requests).

### 2b. Soft retention: buffers keyed by name

The receiving host keeps a failed unit's soft bits, and the name is the handle.

- Where: a new store owned by `KissLink` (native: `kisslink`), not the ARQ's `Engine.store`. The ARQ
  store holds float64 mother buffers keyed by session codeword and is cleared at session start
  (`engine.py`, `_new_session`). Mixing the two would put both at risk.
- Decode path: for each unit slot, `ModemRx.decode(slot, unit_mask, rv, key)` against the unit
  store.
  - Python's `ModemRx` takes its store at construction, and the engine shares one `ModemRx` between
    the ARQ and KISS (`engine.py`: `PHY.ModemRx(r, self.store, ...)`, then
    `self.kiss.on_burst(r, rx)`).
  - So Python gains native's existing form, `decode_stored(slot, mask, rv, stored)`
    (`native/core/arq/phy.hpp`, `PlainRx::decode_stored`), and the unit store calls that.
- Key: (group key, stream, index, code family), plus the sender hash when bound. §2f defines the
  family: two modes share it when their soft bits live in one mother buffer.
- Entry contents:
  - int8 soft values in code order, unscrambled, over the positions received (what
    `codes.combine` builds);
  - the received positions as extents of the mother buffer, so a reception in another mode of the
    family lands where it belongs;
  - a fixed LLR step (1/8, clip at +-15.875; E1 checks the loss);
  - the receptions count, first and last heard, and an MI estimate.
- The ARQ's entry is float64 over the whole mother buffer: 38 KB a unit in qpsk-r1/2 (4800 x 8 B).
  int8 over the extent received is 1920 B at RV 0, 3840 B after RV 1.
- GF(M) (CPM data) entries hold q log-likelihoods a symbol: `codes.soft_len` 2560-6144 values,
  quantised per symbol relative to its largest. That is 6 KB a unit in fsk32r62.
- Combining is automatic. A unit heard again under the same key adds into its entry: any slot, any
  burst, any sender, any RV, any mode of the family. Once decoded, the entry goes and the name joins
  a short done-list (the last 4096 names), so later copies are not decoded again.
- MI estimate: the stored mutual information over k, `codes.buffer_mi(spec, buf, extents)` (new; for
  an LLR L, 1 - log2(1 + e^-|L|) summed over the received positions).
  - Near 1 or above, one more RV likely decodes. Near 0.2, a fresh parity unit is the better buy.
  - It inherits any LLR overconfidence (short bursts underestimate spread). So E1 calibrates
    P(decode after one more RV | mi) per mode before any app leans on it.
- Lifetime and caps: per port, `KEEP` seconds since the last reception (default 2 h, matching
  Ragtime's 2 h listener state), and a host-wide cap in MB (default 32: about 2% of the Pi 4's
  1.7 GB). Both defaults are TBD (Q4, §10).
  - Over the cap, the least recently heard entry goes.
  - Every removal that isn't a decode is reported (`GONE`), so the app never believes in a buffer
    that isn't there.
- Only for open ports with units on. A promiscuous listener keeps nothing.

Explicit "combine later": with names, the useful explicit requests are few.

- `SOFT FORGET`: the app rebuilt the block from parity, or gave up; free the memory now.
- `SOFT KEEP`: extend a stream's lifetime (a file the operator still wants).
- Export and import (§2e): the app keeps the entry itself across a host restart and hands it back.
- `SOFT RETRY` with known bytes (option): decode a held unit again with bytes the app knows pinned
  (`codes.pin_buffer` with a known-bits array; `codes.zero_tail` builds one today).
  - The use: a file's last unit, whose zero padding the app knows from the file length, and text
    payloads with a fixed layout.
  - Value unmeasured. E1 can size it with `scripts/ctl_prior_study.py`'s method.

### 2c. Sender control of redundancy versions

- The app names the RV of each unit it sends: 0-3 (`codes.rv_cycle` is 4 for LDPC and GF(M), 1 for
  polar). The host sends `codes.encode(spec, payload, rv, unit_mask, index=slot)`.
- No sender state: the app supplies the payload every time, so the host keeps nothing it sent.
- The receiver learns each unit's RV from its run in `T_UNITS` (2 bits), as the ARQ's `rv`
  extension does per resend. It maps the RV to mother-buffer positions with the burst's mode
  (`codes.rv_positions(spec, rv)`), so RVs in different modes of one family land correctly (§2f).
- What the app chooses, and why it must choose:
  - RV 1-3 add fresh parity (IR): 0.3-1.9 dB better than Chase at code level (`docs/bulk.md` §3),
    more at high rates.
  - But in every mode of rate 1/2 and up, an RV 1 codeword can't decode alone (`docs/bulk.md` §3).
    A station that holds nothing for the unit (it lost the whole burst or its control, or joined
    late) gains nothing from it. RV 0 again serves those stations and still Chase-combines for the
    rest.
  - Proactive copies: the app may send every unit twice from the start (RV 0, then RV 1 a burst or
    more later). On fading paths that beat the plain lower-rate mode of similar throughput at 1%
    (`docs/bulk.md` §6). This becomes a first-pass policy the app can choose.
- RVs are per unit, not per burst, so a repair burst can mix RV 0 copies and IR increments.

### 2d. Erasure repair at the app layer, and where soft combining fits

The modem doesn't erasure-code. Ragtime and ion-hf already have their own codes, symbol sizes and
feedback, and the code is bound to the app's notion of a block, a seal and a want. What the modem
gives the app's code:

- Symbols are units.
  - A Ragtime piece becomes one unit: pieces of `payload_bytes(mode)`, rounded down to even for
    GF(2^16), so the odd n10 modes lose a byte a unit.
  - An ion-hf-like design packs floor(pb / 20) of its 20 B pieces into a unit, so a lost codeword
    is a whole number of erasures.
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
  fills a hole at every one of them, so parity wins.
- A station near or below the knee loses most units, and fresh units in the same mode fail as the
  first ones did.
  - Today Ragtime moves its repair to a slower gear, whose units can't combine with what the station
    already heard.
  - An IR increment uses that energy. For qpsk-r1/2, RV 0 alone decodes at +1.1 dB Es/N0 and
    RV 0 + RV 1 at -2.7 dB (`docs/bulk.md` §3): 3.8 dB for one more codeword of airtime per held
    unit. qpsk-r1/5 gives about 4.5 dB for 2.5x the airtime and throws the first pass away.
  - But an increment helps one station's one hole.
  - An increment in a narrower mode of the same family does both: a weak station's repair gets the
    narrower mode's S/N advantage and keeps its first pass (§2f; E2 S4 measures it).
- So the sender's choice is per unit and per asker. It needs one more piece of feedback: which holes
  each asker holds soft bits for, and how close each is (the MI estimate).

Feedback from many receivers (a request carries this; it is app protocol, not modem):

- As today: per block (Ragtime: per file under a seal), how many symbols short, and the S/N.
- New: the held holes worth an increment, as a bitmap over the block's units or a list of indices.
  Only those whose MI is at least a threshold (E1 sets it). Ragtime's request is one frame; a bitmap
  of a 128-unit block is 16 B.

A reference sender policy (to be measured in E2 before anyone adopts it):

1. For each unit u listed held by any asker: value(u) = the sum over those askers of
   P_combine(asker, u), from the MI each reported.
2. For a parity unit: value = the sum over short askers of P_alone(asker), from each asker's S/N
   and the mode's single-shot curve (`codes_data/mode_thresholds.json`, as Ragtime's gearbox reads
   it).
3. Send greedily, highest value per airtime first, counting each asker's need down as it is served,
   until every asker's want is met or the round's airtime cap is reached.
   - An increment goes at the next RV the sender hasn't sent for u.
   - It goes at RV 0 instead if some asker lists u as lacking and not held.
4. Increments stay in the unit's code family; parity units may go in any gear.

Relays and the sender in the key (decided: no sender by default, §9):

- What it buys: a relay, or any second station that decoded the content, can send units under the
  same names, and they combine with the original sender's. Fills via relays come at no airtime or
  byte cost: the name carries no sender.
- The risk is two independent senders using one stream id for different content.
  - With random 24-bit ids and m streams live in a group at once (a 2 h lifetime), the chance of any
    collision is about m^2 / 2^25: 1.2e-5 at m = 20.
  - With content-derived ids, a shared id means shared content. The receiver combines copies of the
    same bytes, which is correct.
  - It is correct only if both senders cut the content into units identically. A file cut into
    116 B units and into 76 B units would have one digest but different unit bytes under each
    index. So the stream id hashes the content digest together with the unit payload size, and
    the erasure code's construction must be deterministic. Ragtime's Cauchy names are.
- Consequences when it goes wrong:
  - Two contents under one name combine into garbage. The decode fails, or converges to one of the
    two, which passes its CRC: both were masked alike, since the mask has no sender in it either.
  - The app's digest check catches a wrong piece. Ragtime checks whole-file and seal digests already.
  - The CRC false accept stays 2^-16 per check on short codewords (CRC-16), one check per unit
    slot, as today.
- What a relay must do to forward a unit unchanged:
  - Send in the same group, since the group key is in the mask.
  - Send the same stream and index, with the exact payload bytes as decoded, zero padding included.
  - Send unbound, so no sender hash goes into the mask.
  - Use a mode of the same code family.
  - Choose an RV and list it in `T_UNITS`. The relay need not copy the original sender's RV. It may
    send any RV, and the receiver maps that RV's positions by the relay's mode. A different RV than
    the origin's is best: fresh parity. A repeated RV still adds energy (Chase).
  - A relay that wants to forward several RVs sends the unit several times.
  - Its own FROM in the control is fine: units don't depend on it.
- Bound units, kept for when the cost turns out real (`BCAST UNITS n ON BOUND` on the sending
  port):
  - `T_UNITS` sets the bound flag and carries the sender's `kisslink.station_hash` (2 B).
  - The mask and the receiver's store key both include that hash. Copies from other senders then
    never combine, and can't even in principle, because the info bits differ.
  - It costs 2 B of control and doesn't fit on CPM (above).
  - A receiving port may also refuse unbound units (`BCAST UNITS n ON BOUND`): its unbound units
    are reported lost, not held.

### 2e. Soft export and import (a planned phase)

Ragtime wants to move held soft data into its own database, so it survives a host restart. A
host keeps its store in memory only.

Record format: binary, versioned and self-describing. One record per held entry, big-endian:

| field | bytes | notes |
|---|---|---|
| magic | 4 | `D2GS` |
| format version | 1 | 1; a record of another version is refused |
| family descriptor | 1 + 9 | code type (1 LDPC, 2 GF(M), 3 polar), then LDPC: base graph (1), Z (2), k (2), mother length L (4); GF(M) and polar: mode code (`modes.encode`, 1), soft values a codeword (4), zero fill |
| family fingerprint | 4 | CRC-32 over the family's code tables: LDPC the base graph's shift table in `codes_data/ldpc_shifts.npz` and k; GF(M) the spec's arrays in `codes_data/gf_codes.npz`; polar its frozen set |
| group key | 2 | |
| stream, index | 3 + 2 | |
| sender hash | 2 | 0 = unbound |
| quantisation | 3 | bits a value (8), the step as a power of two (-3), clip code |
| receptions, first heard, last heard | 1 + 4 + 4 | Unix seconds |
| MI estimate | 1 | x 128 |
| extents | 1 + 4e | e extents of the mother buffer, [start (2), end (2)), position order |
| values | sum of extents | int8, code order, unscrambled |
| check | 4 | CRC-32 over everything before it |

Size per unit: 46 B of header plus 4 B an extent, plus the values.

| mode | RV 0 only | RV 0 + RV 1 |
|---|---|---|
| n10-qpsk-r1/5 (23 B units) | 1.1 KB | 1.3 KB (the mother is 1240) |
| qpsk-r1/2 (116 B) | 2.0 KB | 3.9 KB |
| w48-16qam-r2/3 (316 B) | 3.9 KB | 7.7 KB |
| fsk32r62-r1/2 (58 B, GF(M)) | 6.2 KB | 12.3 KB |

A 300 KB file in qpsk-r1/2, half of it held at RV 0, is 2.5 MB of records.

Commands and flow:

- Export, on change: `BCAST UNITS n ON EXPORT`. The host sends a SOFT_RECORD (KISS 0x06, sub 0x03) whenever an
  entry of port n is created or updated. The app keeps the latest record per name.
- Export, on demand: `BCAST SOFT n GET stream [from to]`. The host sends them as SOFT_RECORDs, then
  `BCAST SOFT n GOT k` on the command port.
- Import: the app sends a SOFT_RECORD on port n. For each record the host answers with a UNIT_IN
  status frame: held (re-armed) or refused, with a reason.

Import semantics:

- Checked in order:
  1. the check;
  2. the format version;
  3. the family descriptor and fingerprint against this host's tables;
  4. the group key against port n's;
  5. bound or not against the port's setting;
  6. room in the store.
  Each failure is a refusal with its reason (1-6).
- Re-arming: an imported entry becomes a held entry exactly as if its receptions had just been
  heard, with first and last heard from the record. Its lifetime runs from last heard, and the app
  may `SOFT KEEP` it.
- No double counting. An import never adds into an existing entry. It sets the entry only if none
  is held, or if the record has more receptions than the entry held. So importing the same record
  twice, or after the host heard the unit again itself, can't count the same reception twice.
  Adding records from other receivers (diversity) needs reception ids and is left as an option.
- Code tables changed: a host whose shift table, GF tables or frozen set changed gives a different
  family fingerprint and refuses the record (reason 3). Those soft bits are meaningless under the
  new code. The app drops them and reports those holes as not held.
  - A `PROTOCOL_VERSION` bump that leaves the code tables alone changes nothing here.
  - The format version covers this record's layout only.
- Privacy and trust: soft bits hold nothing a listener couldn't receive. A record is checked only
  for integrity, not authenticity. A client that can open port n can already send units under any
  name, so importing grants it nothing more. The KISS port binds 127.0.0.1 by default.

### 2f. Code families

Two modes are one family when a codeword of each is a prefix of the same mother circular buffer.
Their soft bits then add position by position.

- LDPC: same base graph, same lifting Z and same info length k. `ldpc.qc_code(k, n)` picks
  (base graph, Z) by `ldpc.layout(k, n)` and takes the shift table `bg{bg}_z{Z}`. `QCLDPC.mother()`
  is then fixed by (bg, Z, k). n only decides how much of it a codeword sends.
- What may differ: constellation, band and interleaver (`codes.interleaver` per spec;
  `codes.combine` deinterleaves first), n (the rate), and the frozen per-band data.
- GF(M) and polar: each mode is its own family for now. `codes.gf_code` loads a mother per spec
  name, and modes with different q don't share a field. Polar's frozen set is per spec.

The families that exist today (`ldpc.layout` over every LDPC mode; mother length L):

| (bg, Z, k) | L | modes, coded bits |
|---|---|---|
| (2, 72, 640) | 3520 | qpsk-r1/3, 16qam-r1/3, n4-16qam-r1/3, w48-qpsk-r1/3 (1920); n4-qpsk-r2/3 (960) |
| (2, 192, 1920) | 9600 | 16qam-r1/2, w48-16qam-r1/2 (3840); w48-64l-r2/3 (2880) |
| (2, 128, 1280) | 6400 | w48-qpsk-r2/3 (1920); w48-16qam-r1/3 (3840) |
| (2, 48, 384) | 2304 | qpsk-r1/5, w48-qpsk-r1/5 |
| (2, 96, 960) | 4800 | qpsk-r1/2, w48-qpsk-r1/2 |
| (1, 72, 1440) | 4608 | qpsk-r3/4, w48-qpsk-r3/4 |

Every other LDPC mode is a family of one.

- The first two families span bands and rates: 1200 Hz qpsk-r1/3 and 200 Hz n4-qpsk-r2/3, or
  2400 Hz w48-16qam-r1/3 and w48-qpsk-r2/3.
- Two families are unions the table didn't make obvious: w48-qpsk-r2/3 with w48-16qam-r1/3, and
  16qam-r1/2 with w48-64l-r2/3.

How soft bits from different modes combine:

- Each reception's soft bits go through `codes.plain(spec, soft, slot, rv)` and the spec's own
  deinterleaver, into code order.
- They are added at `codes.rv_positions(spec, rv)`: RV r of a mode with n coded bits covers
  [r n, (r + 1) n) mod L.
- So n4-qpsk-r2/3 RV 0 is positions 0-959, and qpsk-r1/3 RV 0 is 0-1919: a superset, with the
  first 960 adding Chase-like. qpsk-r1/3 RV 1 is 1920-3519 and then 0-319 (it wraps).
- The entry records extents of positions, not RVs, for exactly this reason.
- Decoding cuts the mother at the highest position received, in place of today's `ir_extent(spec,
  max_rv)`. Positions inside the cut that were never received are zero LLRs: erasures, which the
  decoder handles.

Rough complexity, by code and dependencies (Python, then the same in native `core/codes` and
`core/kisslink`):

- `codes.family(spec)`: the key and its fingerprint (about 20 lines).
- Extents in place of a top RV: `codes.decode_buffer` and `decode_buffer_raw` take a position
  extent (about 30 lines, callers included). The ARQ keeps passing the extent its `max_rv` implies,
  so its behaviour is unchanged.
- The unit store keyed by family (no extra cost once the store is written in positions).
- Tests (about 80 lines):
  - every family member's encode equals the first n positions of the family's mother encode (this
    is the claim the table rests on);
  - a unit at RV 0 in one member plus RV 1 in another decodes where neither alone does;
  - a record exported from one member imports under another.
- Depends on phase 3's store only. No wire change: `T_UNITS` already names the mode by the burst
  and the RV by the run.

## 3. Host interface

The same in `data2g/host.py` + `data2g/kisslink.py` + `data2g/tnc.py` and in
`native/core/host/host.cpp` + `native/core/kisslink/` + `native/app/tcp_servers.cpp`.

### 3.1 Commands (command port)

| command | reply | does |
|---|---|---|
| `BCAST UNITS n ON [KEEP s] [BOUND] [EXPORT]` | OK / WRONG | port n takes and delivers units; soft bits kept s seconds (default 7200); BOUND: send bound units, and accept only bound ones; EXPORT: a soft record on every entry change. WRONG from a host without units: the app's feature test |
| `BCAST UNITS n OFF` | OK | no units delivered; its entries dropped |
| `BCAST SOFT n FORGET stream [from to]` | OK | drop entries of that stream (indices from..to-1) |
| `BCAST SOFT n KEEP stream s` | OK | that stream's entries kept s seconds from now |
| `BCAST SOFT n GET stream [from to]` | SOFT_RECORDs on KISS, then `BCAST SOFT n GOT k` | export on demand |
| `BCAST SOFT STATS` | `SOFT entries bytes cap` | the store's size, for a status line |
| `UNITMODES` | one `UNITMODE name code payload-bytes rvs max-units family` line per mode, then OK | mode code (`modes.encode`), unit payload, distinct RVs (1 or 4), most units a burst carries with a one-codeword control, and a family id shared by modes that combine |

`MODE` lines are not extended: Ragtime parses a `MODE` line only when it has exactly 7 words
(`core/src/link/Data2GLink.cpp`, the `MODE` branch). `HEARD` gains nothing either: Ragtime reads its
fourth word as the call.

### 3.2 KISS (data, both ways)

KISS has 4 bits of command, and most values are taken. So the new features share one: SETHARDWARE
(0x06), with a subcommand byte first in the payload. No new command nibble is used.

What the KISS layers use today:

- Data2G, client to host (`tnc._KissHandler`, `KissLink.command`; native `app/tcp_servers.cpp`,
  `core/kisslink`):
  - 0x00 data and 0x0C ACKMODE are frames;
  - 0x02 P and 0x03 SLOTTIME are honoured;
  - every other command (0x01 TXDELAY, 0x04 TXTAIL, 0x05 FULLDUPLEX, 0x06 SETHARDWARE) is logged and
    ignored.
- Data2G, host to client: 0x00 data and the 0x0C ack only. No SETHARDWARE subcommand is defined
  anywhere. `docs/broadcast.md` §6 names SETHARDWARE as the way to ports beyond 15, so that use
  gets a reserved subcommand below.
- Ragtime sends 0x00, 0x0C and 0x02 (`Data2GLink.h`: `KISS_DATA`, `KISS_ACKMODE`, `KISS_PERSIST`).
  Its `KissDecoder::feed` keeps only 0x00 and 0x0C frames from the host and drops the rest
  (`Data2GLink.cpp`). A 0x06 frame from the host is invisible to it until it adds a branch.
- Other clients, from their documentation as we know it (to be confirmed against their sources
  before phase 2):
  - The KISS spec makes 0x06 TNC-specific.
  - Dire Wolf answers SETHARDWARE with ASCII text subcommands (`TNC:`, `TXBUF:` style).
  - Other TNCs use a binary subcommand byte.
  - Most APRS and packet clients never send 0x06, and handle only data frames from the TNC.

Why SETHARDWARE rather than one spare nibble with a subcommand byte:

- It is the spec's slot for TNC-specific commands, so it takes no nibble at all.
- The nibbles still free (0x07-0x0B, 0x0D, 0x0E) are kept for whatever later becomes common
  practice, as ACKMODE did.
- The cost is sharing 0x06 with clients that speak another TNC's dialect. The subcommand ranges
  below keep Dire Wolf-style text out of our space.

Subcommand space (first payload byte; the port is in the command byte's high nibble as usual):

| sub | direction | contents |
|---|---|---|
| 0x01 UNIT_OUT | app to host | [ack tag 2][mode code 1][stream 3][index 2][rv 1][payload, up to payload bytes] |
| 0x02 UNIT_IN | host to app | [stream 3][index 2][status 1][reason 1][mi 1][payload if decoded] |
| 0x03 SOFT_RECORD | both | one soft record (§2e), at most 13 KB, well under `tnc.KISS_MAX_FRAME` |
| 0x04-0x0F | | reserved for units and soft growth (`SOFT RETRY` with known bytes, diversity import) |
| 0x10 | | reserved: ports beyond 15 (`docs/broadcast.md` §6) |
| 0x11-0x1F | | reserved for later features |
| 0x20-0x7E | | never assigned: printable ASCII, so a text query in another TNC's dialect is recognised, logged and ignored |
| 0x7F-0xFE | | reserved |
| 0xFF | | escape: a second byte gives the subcommand, if the space above ever runs out |

UNIT_OUT:

- Tag 0 asks for no ack. Otherwise the host sends the plain ACKMODE ack (0x0C, the tag) once the burst
  carrying the unit has gone, as for frames. Clients already handle that.
- A unit is dropped with `BCAST n DROPPED 1` if its mode is outside the cap or an `ack-*` mode, its
  payload is too long, or its rv isn't 0 on polar.

UNIT_IN status values (the held and status reports):

- 0: decoded, payload follows.
- 1: held (failed and kept, or re-armed by an import).
- 2: lost (failed, not kept: units off, the store full of newer entries, or unbound on a BOUND port).
- 3: gone (an entry dropped by lifetime or cap).
- 4: import refused, with reason 1-6 as in §2e.
- mi: the MI estimate x 128, 0-255.

Compatibility:

- The host sends 0x06 frames only on ports with units on (`BCAST UNITS n ON`). That command is also
  the feature test: an old host answers WRONG.
- An old host given a 0x06 frame logs and drops it (`KissLink.command` ignores 6), so nothing goes on
  air by accident.
- A new host given a subcommand it doesn't know logs and drops it.
- A client in another TNC's 0x06 dialect gets nothing back and changes nothing.
- A burst with units still gets `BCAST n HEARD`. `LOST k` keeps counting frames only.

### 3.3 Versioning and compatibility

| what | change | bump |
|---|---|---|
| PHY (`config.PROTOCOL_VERSION`, 15) | none | none |
| ARQ (session `VERSION`) | none (the unit store is separate; extents keep the ARQ's decode as it is) | none |
| broadcast control header | version 4 on bursts carrying units; frame-only bursts stay 3 | `kisslink.VERSION` gains 4 alongside 3; release notes say v3-only hosts skip unit bursts whole, frames in them included |
| `T_UNITS` = 4, the bound flag | new TLV, only in v4 | with the above |
| unit CRC mask | new identity function | with the above |
| host commands, KISS SETHARDWARE subcommands 0x01-0x03 | additive; old hosts answer WRONG, or log and drop 0x06 | host only |
| soft record format | new interface | its own format version byte; code changes show in the family fingerprint |

Each wire step is its own GitHub release with the version bump, as the version rule asks.

## 4. Ragtime, concretely

What changes on Ragtime's side (Ragtime's protocol is its own, version 0 today; this is a request
to its team, not code here):

- Feature test: `BCAST UNITS n ON EXPORT` after `BCAST OPEN D2G FROM call`. On WRONG, stay on frames.
- `KissDecoder::feed` keeps 0x06 frames too, and dispatches on their subcommand byte (§3.2).
- `UNITMODES` beside `MODES`, for unit payload sizes and families.
- File pieces become units.
  - Stream = 24 bits of a hash over the file digest and the piece size.
  - Piece size = the unit payload (even).
  - `FileAnnounce` gains the stream and the mode the pieces went in.
  - Announces, chat, seals and requests stay frames.
  - The "every record names its sender" rule holds through the burst's FROM and the announce. On
    CPM unit bursts, which can't carry FROM, it holds through the announce only (Q2, §10).
- First pass: units at RV 0. Optionally copies (RV 1 a burst later) on a file the operator marks for
  weak paths.
- Listener:
  - A unit frame with status 0 is a piece. Status 1 records (index, mi) per file.
  - Its `RepairRequest` adds, per file, a bitmap of held pieces with mi over the threshold.
  - On a whole-file rebuild it sends `BCAST SOFT n FORGET stream` and drops its records.
- Persistence: SOFT_RECORDs go into the database beside `broadcast-parts`, the latest per name.
  - After a restart, Ragtime imports the records of files still being received before it asks
    under any seal.
  - Refused records are deleted, and their pieces count as not held.
- Sender: the reference policy in §2d, inside its existing rounds, caps and gears. IR increments go
  in the file's first-pass family; parity may still go in the repair gear.
- Relays: a station that completed a file may answer a request with increments of the same units,
  unbound, in the same family (§2d).
- Gains to expect, to be measured in E2 before Ragtime commits:
  - from units alone, piece loss equal to codeword loss (no k-fold amplification, §1.2) and about
    19 B less header a piece;
  - from retention and increments, the weak stations' completion several dB lower.

## 5. Costs and risks

- Memory: int8 entries take 1-4 KB a unit on OFDM at RV 0 (`codes.soft_len` 960-4000) and 2.5-6 KB
  on GF(M).
  - A 300 KB file in qpsk-r1/2 is 2650 units. A station that lost half holds 2.5 MB.
  - In fsk32r62-r1/2 (58 B units), the same file at half lost holds 16 MB.
  - The cap and `GONE` bound it. Never store the ARQ's float64 mother buffers for units (38-100 KB
    each).
- CPU (Pi 4: under 10% of the dev machine's):
  - Each unit slot is decoded once, as each data slot is today.
  - A combined decode runs over the received extent: twice the work a pass for RV 0 + RV 1.
  - Holding costs nothing, and quantising is O(n) a unit.
  - No trial decodes against stored entries, ever.
  - The DD budget (`phy.DD_BUDGET_S`) applies as to any burst.
  - Export on change sends one record per entry update, a few KB over localhost.
- Lifetimes:
  - Too short: an increment arrives after the entry went. The station was told `GONE` before then,
    so it reports the hole as not held and the sender sends RV 0 or parity.
  - Too long: a stale entry under a reused name dilutes a new combine. Content-derived streams make
    reuse mean the same content.
- Wrong combining (name collisions), including relays: §2d gives the numbers and the guard. The
  app's digests catch what the CRC can't.
- Poisoning: anyone can send units under any name in any group, since the group key is a hash of a
  public name. A deliberate wrong unit can keep a held unit from ever decoding.
  - Bounded: after 4 receptions without a decode, the entry restarts from the newest reception.
  - The app's parity is the fallback, and bound units are the per-port opt-out.
- Security and Part 97:
  - Units are as plain as frames: the scrambler is unkeyed and the name sits in the control.
  - Masks gate CRC checks only (`docs/broadcast.md` §2).
  - Soft bits and records hold nothing a listener couldn't receive. They are kept only for groups
    this station opened.
- Control loss now costs a unit burst's units (2a). DUP control is recommended on unit ports.
- Interop: v3 hosts skip unit bursts entirely, frames in them included. An app that mixes frames
  and units must accept that old hosts lose those frames, or send them in a frame-only burst (Q3,
  §10).

## 6. What not to do

- No erasure code in the modem: the symbol size, block, feedback and refusal rules are the app's.
- No soft bits for frames. They have no name that survives a resend (their mask depends on slot and
  data start), and naming them would bring the k-codeword amplification back.
- No blind matching of new soft bits against stored entries, by correlation or trial decodes. It
  costs CPU, and every trial is another chance of a false CRC pass.
- No position, slot, burst or mode in a unit's name or mask.
- No reuse of the ARQ store for units, and no float64 mother buffers in the unit store.
- No RV 1+ as the only send of a unit that stations without soft bits need: at rate 1/2 and up, they
  can't decode it alone.
- No extra words on `MODE` or `HEARD` lines.
- No combining across families. Within one (same bg, Z, k) it is sound; across them it is not.
- No import that adds into a held entry: it would count receptions twice.

## 7. Evaluation plan

Conventions from the memory rules apply:

- PEP-fair (`DATA2G_PEP_REF_DB=5`);
- the same trial seeds at every SNR, and paired comparisons;
- 10% and 1% points, with conclusions in dB;
- fading weighted over deep AWGN.

E1, code level: extend `scripts/ir_study.py`, about 150 lines, depending on nothing new.

- int8 storage at step 1/8, and int4, against float. Cases: RV 0 + RV 0, RV 0 + RV 1, and
  RV 0 + RV 1 + RV 2, in every LDPC and GF(M) mode. Measure the loss at the 10% point. Accept int8 if
  it is within 0.1 dB.
- MI calibration: P(decode after one more RV | stored MI) per mode, on AWGN and the CCIR channels.
  Output: the threshold the requests use.
- Families: the prefix test of §2f for every family, then the combined decode of RV 0 in the
  narrower member plus RV 0 or 1 in the wider. The gain is measured against each alone.
- Known-bytes retry on a last unit (padding known): the gain in dB, which decides whether `RETRY`
  is worth adding.

E2, broadcast repair: `scripts/app_repair_study.py`, new, about 500 lines.

- Depends on `codes`, `modem`, `hfchannel`, `phy.ModemRx`, and a study-only numpy Cauchy GF(256) of
  about 60 lines. It works at the PHY level like `scripts/bulk_study.py`, so it runs before any host
  change.
- Setup: one sender and N = 4 or 8 receivers, each an independent `hfchannel` run of the same audio.
  - S/N offsets are spread across the mode's knee: 0, -2, -4 and -6 dB from the strongest.
  - A 30 KB file. Channels mpp, mps, mpg and AWGN.
  - Modes qpsk-r1/2, qpsk-r1/3 (family with n4-qpsk-r2/3), qpsk-r1/5, n10-qpsk-r1/5 and
    fsk32r62-r1/2.
- Feedback: first genie (every request heard), then lossy at each receiver's S/N in its request
  mode.
- Strategies, paired:
  - S0 Ragtime today: 255 B frames spanning k codewords, parity pieces, rounds by the neediest want,
    repair in the slowest asker's gear.
  - S1 units: one-codeword pieces, the same parity rounds.
  - S2 S1 + retention + the §2d policy (increments for held units).
  - S3 S2 + first-pass RV 1 copies.
  - S4 S2 with increments in a narrower member of the family (qpsk-r1/3 first pass,
    n4-qpsk-r2/3 increments).
  - S5 S2 with a relay: one receiver that completed sends increments.
- Metrics:
  - airtime until every receiver holds the file;
  - the weakest receiver's S/N at which the file completes within 2x the first pass's airtime, at 10%
    and 1% of trials failing;
  - peak store bytes and decodes a burst per receiver.

Decision gates:

- Phase 2 goes ahead if S1 beats S0 by at least 1 dB at the weakest receiver's 10% point on two
  fading channels.
- Phase 3 goes ahead if S2 beats S1 by at least 1 dB there.
- A few percent of airtime alone decides nothing.

E3, on air or HFSim loopback, after phase 4: a Ragtime-shaped transfer between three hosts, Python
and native, with a host restart and re-import mid-transfer. Log the store's memory and CPU on a Pi 4.

## 8. Phasing

Each step stands alone and is useful without the next.

1. Studies E1 and E2. No host change.
   - Accept: tables in this doc for the int8 loss, the MI threshold, the family prefix test and
     S0-S5; the gates decided.
2. Units on air, RV 0 only, no retention.
   - Code, Python: `kisslink` (v4 build in `next_burst`, `T_UNITS` with the bound flag,
     `unit_mask`, receive in `on_burst`), `tnc` and `KissLink.command` (0x06 with UNIT_OUT, UNIT_IN), `host` (`BCAST UNITS`, `UNITMODES`).
   - Code, native: the same in `core/kisslink`, `core/host` and `app/tcp_servers`, and
     `core/monitor` shows unit bursts.
   - Accept:
     - a unit round trip through the modem in every unit mode (pytest and ctest);
     - a v3 host skips a v4 burst and reports nothing, control decoded or lost;
     - frames before units in one burst still parse;
     - a unit decoded under the wrong name, or bound under the wrong sender, fails its CRC;
     - a CPM unit burst with one run fits its 20 B control;
     - Python-native parity tests (`tests/native_subs`).
3. Soft retention, RV control, code families.
   - Code: `decode_stored` in Python's `ModemRx`; `codes.family`; position extents in
     `decode_buffer`; the unit store (int8, family keys, caps, lifetime, done-list, MI) in
     `kisslink` and native; `SOFT FORGET/KEEP/STATS`; status 1-3 frames; rv honoured on send.
   - Accept:
     - a unit failed 3 dB below its knee and resent at RV 1 in another slot of a later burst
       decodes;
     - the same across two senders (unbound), and never across two bound senders;
     - the same across two members of a family;
     - eviction at the cap reports `GONE`;
     - an entry under a reused name with other content fails, or yields a CRC-valid unit, never a
       crash;
     - memory stays under the cap through a 2 h soak;
     - the ARQ's tests unchanged;
     - E2's S2 and S4 reproduced through the host path within noise.
4. Soft export and import.
   - Code: the record (`kisslink` pack and parse, native the same), SOFT_RECORD, `EXPORT`,
     `SOFT GET`, import with the checks of §2e, status 4.
   - Accept:
     - a record round-trips byte for byte;
     - a host restarted with an imported entry decodes the unit's next RV as if it had never
       restarted;
     - each refusal reason 1-6 is triggered by a test;
     - a record exported from a host with a changed shift table is refused with reason 3;
     - importing the same record twice leaves the entry unchanged;
     - Python records import into native and the reverse.
5. Ragtime adoption (in Ragtime's repo, §4). Data2G's part: answer questions and fix what it finds.
6. Options, each only on its own measured case: `SOFT RETRY` with known bytes; diversity import
   (adding records from other receivers, with reception ids).

## 9. Decisions taken (2026-10-28)

- Q1, sender in the store key: no, by default, so relays' sends combine. The cost and risks are in
  §2d. Bound units are the per-port way to require a sender if that cost turns out real.
- Q5, soft export: wanted (tentatively, by Ragtime) to keep held soft data in its own database across
  host restarts. Promoted to phase 4 (§2e).
- Q6, `data2g-bulk`: an illustrative experiment, to be replaced by what gets built in Ragtime. Prior art
  only; no phase moves it onto units.
- Q7, cross-mode combining within a code family: in, since it costs little once the store is in
  positions (§2f). It also lays the groundwork for the ARQ (§11) and for consolidating families (§12).

## 10. Open questions (TBD: pending collaborators' feedback and the user's further thought)

Each keeps the number it had in the first draft. Q1, Q5, Q6 and Q7 are decided in §9.

- Q2 (CPM control room): CPM unit bursts can't carry group+from (or bound units) in the 20 B
  control. Send the plain group (the app names the sender in its own records), or refuse units on
  CPM ports opened with FROM? TBD.
- Q3 (mixed bursts): frames and units in one v4 burst (v3 hosts lose those frames), or units-only
  bursts? TBD.
- Q4 (store defaults): 32 MB host-wide, 2 h per port. Right for the Pi 4? TBD.
- Q8 (control-lost unit bursts): accept them as erasures (DUP recommended), or design a
  back-reference (the next burst names the previous burst's units) later? TBD.

## 11. Future, out of scope: ARQ mode changes within a family

Not part of this plan. It is sketched here so the store and family definitions above don't block
it.

- Today (`docs/arq.md` §4, "Mode changes abandon outstanding codewords"):
  - A resend must use its original submode, and a burst has one submode.
  - So a mode change with codewords outstanding sends `abandon A`. The sender re-slices from A, and
    the receiver drops its soft bits for those seqs. `ModemRx.decode` drops an entry stored under
    another submode with a warning.
- Within a family, none of that is forced:
  - The payload size is the same (same k), so the slicing stays valid and seq A onward keeps its
    bytes.
  - The data mask (`burst.data_mask`: key, direction, comp, epoch, seq) has no mode in it.
  - The soft bits combine by position (§2f).
  - A shift from qpsk-r1/3 to n4-qpsk-r2/3, or from w48-qpsk-r2/3 to w48-16qam-r1/3, could carry
    on with resends and IR, with no abandon and nothing dropped.
- What would change:
  - §4's abandon rule becomes: abandon only when the new mode is in another family.
  - `phy.ModemRx.decode` keys its store by family, not submode, with position extents.
  - The `rv` extension's 2 bits are read against the burst's own mode (its n): the receiver maps
    positions with the burst's spec, as units do.
  - The gear shifter (`policy.GearShifter`) may treat an in-family move as cheap, since outstanding
    codewords keep their value.
- State both ends must agree on:
  - Both must know the two modes are one family. That comes from the code tables, so it holds when
    both run the same tables. The family fingerprint could ride in CONNECT so that a mismatch falls
    back to abandoning.
  - Nothing else is new: seqs, slicing and epochs are unchanged.
  - A receiver that wrongly believes two modes are one family combines garbage. The masked CRC then
    fails, and the watchdog's resync (§10 of arq.md) recovers, as for any failed decode.
- Version: a session `VERSION` bump, since an old receiver expects an abandon on every mode change.
  The new behaviour is used only when both ends' CONNECT says they support it.
- Cost: the shifter's in-family rule and the abandon change in `link.py`, about 50 lines and their
  fuzz tests. Its own plan, after phase 3 lands.

## 12. Future, out of scope: fewer, larger code families

Not part of this plan, and no decision is asked for here. The question is whether changing some
modes' codes, so that more modes share one family, would let more mode changes and repeats combine
by IR. In short: the most it could gain is the airtime of codewords discarded at mode changes. That
is a few percent overall and 9-24% in the floor fading cells. One cheap oracle study can tell
whether that is worth 1 dB before anything else is considered.

### 12.1 Why families are few today

- A family is (base graph, Z, k) (§2f).
- Within a band, ladder rungs of one constellation share n (frames per codeword) and differ in k:
  qpsk-r1/5, r1/3 and r1/2 all have 1920 coded bits and carry 46, 76 and 116 B.
- So families arise only where k happens to coincide: the same rate in w and w48, a few
  constellation pairs, and n4-qpsk-r2/3 with qpsk-r1/3.

### 12.2 How often the shifter throws held soft bits away

The measurement:

- 288 linksim runs: `scripts/linksim.py` with the gear shifter on both ends (`shift` policy).
- Workloads: bulk (300 s) and winlink (up to 1800 s).
- Channels mpg, mpp and AWGN; 8 seeds at each of -5, 0, 5, 10, 15 and 20 dB.
- A 50-line harness around linksim (not committed) wraps `Station._data_allowed` and records each
  abandon for a mode change. For each one it records the codewords outstanding, and how many of them
  the peer had heard (`Codeword.heard > 0`), meaning it held their soft bits.

| workload | codewords made | mode-change abandons | outstanding discarded | held soft bits discarded | of those, already in one family |
|---|---|---|---|---|---|
| bulk | 59 187 | 180 | 3356 | 1450 (2.4% of codewords) | 63 |
| winlink | 25 273 | 202 | 2435 | 1050 (4.2%) | 71 |

Where it concentrates (held discarded / codewords made):

- bulk: mpg -5 dB 24%, AWGN -5 18%, mpg 0 9%, mpg 20 7%, mpg 5-15 4-5%;
- winlink: AWGN -5 12%, mpg 0 10%, mpg 5-15 3-6%;
- AWGN and mpp from 0 to 20 dB: 0-3%.

The pairs that discard most are downshifts within a band:

- w48-qpsk-r1/3 to w48-qpsk-r1/5: 43 abandons, 445 held codewords;
- w48-qpsk-r1/2 to w48-qpsk-r1/3: 160;
- qpsk-r1/2 to qpsk-r1/5: 128;
- 16qam and 64l steps down: 76-122 each.

One band change, w48-qpsk-r1/5 to qpsk-r1/3, discards 170. Today's families cover only 134 of the
2500 (5%), so §11's ARQ change alone, with today's codes, would recover about 0.1-0.2% of codewords.

Caveats:

- The link abstraction models OFDM only. 36 runs (31 of them at -5 dB) stopped when the shifter
  picked a CPM mode, so the floor cells are partly sampled.
- 8 seeds a cell.

### 12.3 What those codewords would have gained

- Each discarded held codeword failed once in the higher-rate mode. Resent in a lower-rate member
  of its family, it would cover fresh mother positions (IR) and overlap the old ones (Chase).
- The 50% point of one transmission against two with IR (`runs/ir_study.csv`, `scripts/ir_study.py`):

| mode | AWGN | mpp |
|---|---|---|
| qpsk-r1/5 | 2.3 dB | 2.8 dB |
| n10-qpsk-r1/2 | 3.3 dB | |
| w48-qpsk-r1/2 | 4.0 dB | |
| w48-16qam-r2/3 | 6.1 dB | |
| w48-64l-r1/2 | 6.0 dB | |

- So nearly every such codeword would decode on its first resend. The upper bound on the saving is
  the airtime they took: the shares in §12.2.

### 12.4 What consolidation would take

- One k per family, with the rate set by n. Lower rungs become longer codewords (more frames each).
  A family's lowest rate is its mother's:
  - base graph 2 here is 0.17-0.2 (k / L: 384/2304, 640/3520, 960/4800);
  - base graph 1 is about 0.31 (1440/4608).
- So a family spanning r1/2 to r1/5 must be base graph 2 at one k.
- The r3/4 rungs stay out. `ldpc.layout` uses base graph 1 above rate 0.67, and base graph 2 is the
  weaker code at high rates.
- Threshold risk:
  - Holding k at the higher rung's value makes the low rungs longer codes: a slightly better
    threshold, but coarser granularity. Short writes waste more, though §5a inlining helps.
  - Holding it at the lower rung's value makes the high rungs shorter codes, which lose waterfall
    steepness.
- Payload sizes change on most rungs. Goodput per rung moves, and so does everything fitted to the
  ladder:
  - the thresholds (`runs/ladder_final.csv`, `codes_data/mode_thresholds.json`);
  - the link abstraction's curves;
  - the outcome model, a full retrain with new mode descriptors.
- Formats: the frozen formats (`data2g/format/*.npz`) and interleavers regenerate, and
  `config.PROTOCOL_VERSION` bumps. The native tables regenerate too (`tools/gen_native_tables.py`).
- A new (k, Z) may need a new shift table, generated and screened for error floors
  (`scripts/own_shifts.py`).
- On OFDM the control codeword is the data spec (`modes.ctl_spec`). The control's size changes with
  the rung, so the padding-prior results (`docs/arq.md` §3) need re-measuring.
- Then §11's ARQ change and its session `VERSION` bump.

### 12.5 Verdict and what to measure first

- The bound is small overall (2.4-4.2% of codewords) but concentrated where it matters (9-24% in the
  floor fading cells).
- That is worth one oracle study, not a code change.
- The oracle, in linksim, about 40 lines:
  - at each mode-change abandon, the re-sliced codewords inherit the discarded held codewords' stored
    MI in `SimRx`'s store, weighted by byte overlap, as if every band were one family;
  - run paired against master on the floor cells (mpg 0, mpg and AWGN -5, mpp 0), bulk and winlink;
  - then real-modem sessions with CPM enabled (`scripts/phy_session.py`) for the cells linksim can't
    model.
- Gate: at least 1 dB at the 10% point (or the goodput curve's dB shift) in two floor cells. Less
  than that, and consolidation stops here.
- If it passes: design one band's consolidated ladder and measure its thresholds at code level
  against today's rungs, before any format, retrain or protocol work.
