# Data2G broadcast

Status: accepted 2026-10-02; implemented in `data2g/kisslink.py` (link) and `data2g/host.py` (commands).

TLDR: KISS grows into named broadcast groups. Each group is a KISS port, opened by a
command on the command port. A group's bursts carry its name in the control codeword,
and every codeword in them, control and data, uses a CRC mask hashed from that name. A
receiver decodes the control without a mask, reads the name and checks the CRC with
its key: no fixed broadcast key, and no trying groups one by one. Port 0 is today's
KISS under the group name "KISS 0", with rate shifting off by default.

## 1. Terms

- **Group:** up to 10 characters in the packed callsign alphabet (A-Z 0-9 / - and
  space, space padded at the end). It names an application, like `APRS` or `CHAT`.
  A group name must not be a valid callsign: that keeps the callsign namespace free
  for addressing (below).
- **Callsign as destination:** a broadcast-mode packet may use a callsign as its
  destination, in place of a group name. This is a way to send non-connected or
  side-channel information to one specific station. It isn't truly broadcast, but
  neither is connected-mode AX.25 over our KISS: the air is shared and every receiver
  can decode it, and the destination only says who it is meant for. A station opens
  its own callsign as a group (`BCAST OPEN call`) to receive such packets. Because
  application groups are never valid callsigns, a callsign destination can't collide
  with one.
- **Group key:** a nonzero 16-bit hash of the packed group (FNV-1a, as
  `session.session_key`).
- **Port:** a KISS port number (1-15) a host has opened for one group. Port 0 is
  always open, on "KISS 0".

## 2. Masks: who can read what

Everything on air is plain: the scrambler is unkeyed (`arq.md` §2), so anyone decodes
and reads any codeword. A mask only decides whose CRC check passes.

- **One key per group, on every codeword:** `ctl_mask(0, i, group key)` for control,
  `data_start_mask(slot, start, group key)` for data: `data_mask(0, slot, key)` with
  the burst's first data slot `start` in the direction byte's spare bits (its
  `epoch` field), so a data codeword also says where the data begins. Port 0's key
  is the hash of "KISS 0".
- **The control checks itself:**
  1. Decode slot 0 once, mask left open (`codes.decode_raw`; the engine does this
     already, for the session's key and mask 0).
  2. Parse the unverified bytes: the header's version must be 3, then the TLVs. The
     group comes from group or group+from; absent means "KISS 0".
  3. Hash the group to its key and check the CRC (`codes.check`). Polar control
     (CPM) does this per list candidate, each with the group read from it.
  4. Slot 0 alone didn't decode (OFDM: no LDPC candidate converged, plain or with
     its padding pinned; CPM: its header says two control slots): slots 0 and 1
     combined as RV 0 and RV 1 (`ModemRx.raw_pair`, `codes.decode_buffer_raw`), and
     steps 2-3 again on what that reads. A reading from the pair counts only if its
     header's dup bit is set. Bursts whose slot 0 decoded cost nothing more.
  - A wrong decode or another burst type gives a garbage name, and the check fails.
    A false pass is 2^-16 per candidate (CRC-16), as for any mask today.
- **Cost:** one decode per burst, shared with the ARQ and CQ/ID checks, then CRC
  checks. Interest in many groups costs nothing extra: the burst names its own.
- **Filtering:** a receiver checks the control, then keeps the burst only if the
  group is an open port. Promiscuous listening keeps every group's.
- **Data:** checked under the group read from the control. Data from a burst whose
  control was lost never lands on the wrong port.
  - Control lost: looked for only when slot 0 didn't decode at all (OFDM: no LDPC
    codeword, alone or paired with slot 1). A slot 0 that decoded but isn't a
    broadcast control is another kind of burst (an overheard ARQ exchange, a CQ),
    not a broadcast whose control was lost, and its data is never decoded. CPM
    (polar control, which always gives a list) has no such test.
  - The receiver then looks for data in slots 1-8, the first that passes
    under an open port's key, trying each first data slot the burst could have
    (1, 2, 3, 4, 6 or 8: 1-4 control codewords, each once or twice; CPM: its header
    says). One decode per slot, then up to 6 x 16 CRC checks each. The slot that
    passes names the key and where the data starts, so the frames are read from
    there, the first data codeword lost or not (§3, Data). A promiscuous listener
    can still read that data, unverified.
  - Every such check is another chance for some other burst's converged codeword to
    pass by luck (CRC-16 on the short LDPC codewords: 1 in 65536), and a pass would
    deliver its bytes as frames. So a slot alone is believed only where one check per
    key was all it took before: slot 1 as the first data slot, a burst's only data
    slot, or the start CPM's header gives. Anywhere else, a later data slot must pass
    under the same key and start too. A burst with its control and all but one
    codeword of a longer data run lost is dropped.
  - The search stops at the first slot that decoded (an LDPC codeword) but is
    neither an open port's data nor the rest of its control (codewords 1-3 at RV 0,
    in slot i or under DUP slot 2i): that is another burst, or another group's. So
    a burst costs decodes past slot 1 only while its slots keep failing to decode.
- **Collisions:** two group names share a key 1 time in 65536. The control's name
  settles it. Control-lost data whose key matches two open ports is dropped, and both
  get a LOST notice (one a false positive): a payload is never misdelivered.

## 3. Burst format

- **Control:** every burst's control, port 0's included, is a 1-byte header
  `[version 4 bits | dup 1 | reserved 1 | n_ctl - 1 (2)]` (version 3), then TLVs, as `frames.Control` without the 32-bit core. The ARQ core's fields mean
  nothing to a broadcast burst, and CPM gives control exactly one 20 B codeword.
  - Version 2 (before 2026-10-25: no dup bit, data framed by 2-byte lengths, data
    masks without the start) isn't read: its bursts are skipped as another kind.
- **Duplicated control (`BCAST MODE n DUP mode`):** the dup bit set, each control
  codeword goes twice, RV 0 then RV 1 in the next slot, as ARQ's `ARQ_DUP` does; the
  data follows the doubled span, so it starts at slot 2 x n_ctl. A receiver combines
  the two when the first fails alone (§2): about 3 dB on the control on AWGN (LDPC
  IR on OFDM; polar Chase on CPM, whose header announces the second control slot). Off by
  default; it costs one control codeword more per burst (on CPM, one more short
  polar codeword of air time: the data codewords it can carry stay `cpm.MAX_DATA`).
  Broadcast has no feedback, so the sender's host chooses it, unlike ARQ's
  `T_DUPCTL`.
- **TLVs:**

  | type | contents | when |
  |---|---|---|
  | group (1) | packed group, 8 B | every burst except port 0's (absent means "KISS 0", and its key) |
  | group+from (2) | group and sender, 15 B (120 bits packed together) | in place of group, when the port asks for it |
  | reports (3) | sender hash (2 B), then per station [hash 2 B][mode code << 2 \| size hint] | a port with rate shifting on (§4) |

  - group+from on CPM: 1 + 2 + 15 = 18 B, which fits 20 B. Separate group and from TLVs
    would be 1 + 10 + 10 = 21 B, which doesn't fit.
- **Data:** each frame COBS-encoded (Consistent Overhead Byte Stuffing: no zero byte
  left) and ended by a zero byte, back to back, zero padded; one burst carries one
  group's frames. A frame that would start a codeword, the data's first included,
  gets a zero byte before it.
  - A zero byte in a good codeword always ends a frame, so a failed codeword loses
    only the frames it touches: parsing picks up at the next zero in a good codeword.
  - The zero before a frame starting a codeword puts where it starts in its own
    codeword. Without it, that would be the zero ending the frame before, the
    codeword before's last byte, and a frame whose bytes all arrived would be lost
    with that codeword. A reader skips it as an empty frame.
  - `BCAST n LOST k` counts each run of frames one failed codeword touched as one:
    at least k frames were lost. With frames filling whole codewords, a lost
    codeword holds one frame and k is exact.
  - Cost: `kisslink.framed_len(n) = n + 2 + n // 254` bytes at most, equal when the
    frame has no zero byte (n + 2 up to 253 bytes, as the 2-byte length it
    replaced; 259 for a 256-byte frame), and 1 more for a frame that starts a
    codeword. A client that wants its frames to fill k codewords each makes them
    k x bytes-per-codeword - 3 bytes long (up to 253).
- **What fits:** `BCAST MODE` refuses a mode whose control codewords (at most 4) can't
  hold the port's control: the header, its group TLV, and on a shifting port a
  reports TLV with at least its sender.
  - group+from on a shifting port (1 + 17 + 4 = 22 B) doesn't fit CPM's 20 B.
  - group+from (18 B) doesn't fit the 4 B codewords of the ACK modes, even 4 of them.
  - A plain group fits every mode, shifting or not.
- **Control lost** (§2): the data codewords' masks say where the data starts, so a
  burst is found whatever its control's length, DUP or not.
- **A 256-byte frame on every mode** (but the 4 B ACK modes): a burst grows past its
  size class to carry its first frame, up to what its header can say
  (`kisslink.next_burst`), and receivers take a header that long:
  `engine.receive_accept` raises each OFDM submode's max codewords to hold 4
  control codewords, sent twice, and a 256-byte frame framed as a burst's first
  (8 + ⌈260 / bytes per codeword⌉). CPM's 8 data codewords hold 304-464 B whatever the control.

## 4. Modes and channel access

- **Transmit mode:** every port, port 0 included, has one, set by the host at any time
  between frames (it applies to frames sent after it). Receiving decodes every mode,
  whatever a port's transmit mode. The default is the cap's robust broadcast mode
  (`kisslink.BROADCAST`).
- **Rate shifting, off by default:** `BCAST MODE n AUTO mode` turns it on for port n,
  any port, with `mode` as the fallback.
  - A connected-mode AX.25 frame goes in the mode its next hop has reported for us
    lately. Anything else goes in the fallback: UI frames, non-AX.25, a station
    without a fresh report. One-to-many traffic can't be shifted from reports.
  - The port's bursts carry the reports TLV.
  - `BCAST MODE n mode` turns it off again: no reports sent, reports heard ignored,
    and frames never parsed as AX.25.
- **One queue for all ports:** the next burst takes the first queued frame's port and
  mode, then every queued frame of that port that fits. KISS p-persistence and
  SLOTTIME apply to all ports. Broadcast still goes only between ARQ sessions.
- **Channel access:** a queued burst waits while a burst is locked (up to
  `--kiss-busy-limit`, 60 s). In-band energy with no lock, from a burst we can't
  read or from QRM, also holds it, for 15 s at most.

## 5. Host interface

One integrated server: the command, data and KISS ports always run (the `--vara` /
`--kiss` flags go). Commands go on the command port, replies as for VARA's (OK / WRONG):

| command | reply | does |
|---|---|---|
| `BCAST OPEN group [FROM call]` | `BCAST PORT n` | opens a port; FROM sends group+from |
| `BCAST CLOSE n` | OK | |
| `BCAST MODE n mode` | OK / WRONG | that port's transmit mode for later frames, within the KISS cap; any time between frames, never filters decoding. Rate shifting off (the default) |
| `BCAST MODE n AUTO mode` | OK / WRONG | rate shifting on for that port, `mode` the fallback (§4) |
| `BCAST MODE n [AUTO] DUP mode` | OK / WRONG | as above, with each control codeword sent twice (§3). Flags in either order, each once; a `BCAST MODE` without DUP turns it off |
| `MODES` | one `MODE ...` line per mode, then OK | name, bandwidth Hz, bytes per codeword, max codewords a receiver takes, airtime at 1 and at max codewords |

Statuses go to the command client once it has sent a BCAST command, port 0's and
MISSED included, so a VARA client that never does (Pat) never sees one. When the
client goes, its ports close and port 0 is back to its defaults (fixed mode, the
cap's robust broadcast mode). A group may be several words in `BCAST OPEN`; opened
twice, both ports get its frames. `MODE` lines read `MODE name bandwidth-Hz
bytes-per-codeword max-codewords seconds-at-1 seconds-at-max`.

The largest frame a port can send in a mode, from its `MODE` line: the largest n
with `framed_len(n) + 1 <= (max-codewords - c) x bytes-per-codeword` (the 1: the
zero before the data's first frame), where c is the
port's control slots: OFDM, its control codewords (1 unless the header, group TLV
and reports outgrow one, as reports on a shifting port can; at most 4), twice under
DUP; CPM, 1 (its max-codewords counts the data and one control: a second control
copy is extra). A client that wants one number per mode takes c = 4 (8 under DUP).
Up to 253 bytes, that is n + 3: one byte more than before. A 256-byte frame always
fits but on the ACK modes (§3).

- `BCAST n HEARD [call] snr`: a burst's control checked for port n (call from group+from).
  Its frames follow on the KISS port. It's known only at the burst's end: the receiver
  decodes whole bursts. `snr` is the burst's pilot SNR (dB, average power in 2500 Hz,
  as the monitor shows it), always the last word: a callsign always has a letter.
- `BCAST n LOST k`: the burst is the port's (control checked, or data under its key),
  but k frames were lost.
- Control lost: if a data slot passes under an open port's key, the burst is that
  port's (HEARD / LOST as usual; an ambiguous key as in §2 Collisions). Control and
  all data lost can't be tied to a port: the frozen 16-bit PHY header has no room for
  a group. Every open port gets a `BCAST * MISSED submode n_cw [snr]` hint; it may be
  another group's burst. `snr` is the in-band energy over the burst against the noise
  floor, on HEARD's scale; absent until a noise floor is known (a few seconds after
  start). Receiver AGC can pull it low under strong signals. Only outside an ARQ session: during one, a lost burst is
  most likely the peer's.

Apps learn when their frames went out through KISS ACKMODE (command `0x0C`, as in BPQ32
and QtSoundModem):

- **Ask per frame:** the app sends `0x0C` (with the port in the high nibble), a 2-byte
  tag of its choosing, then the frame. Plain data frames (`0x00`) get no ack, so apps
  that don't ask see no change. Works on every port, port 0 included.
- **The ack:** `0x0C`, the same port, and just the tag, sent on the KISS port to the
  client that sent the frame, when the burst carrying that frame finishes transmitting. Frames sharing a burst are acked
  together, in queue order.
- **Sent, not heard:** an ack means the frame was on the air. One-to-many traffic has
  no receipt; a reply is the app's business.
- **Never sent:** a frame dropped from the queue (its port closed, or too big for the
  port's mode) gets no ack. The opener gets `BCAST n DROPPED k` on the command port,
  so an app waiting on tags knows to stop.

## 6. Not done

- **Ports beyond 15** (KISS SetHardware): 15 groups open at once is plenty until an
  app needs more.
- **Setting a port's mode by KISS command:** the command port does it. Add it when a
  KISS-only client needs it.
- **Incremental redundancy across broadcast bursts:** later. An app would mark a frame,
  by a KISS extension, as a resend at a given redundancy version; it runs its own
  resends outside our ARQ, and we keep its soft bits for a while to combine. The
  position-only scrambler already lets a resend combine from any slot.
- **ID:** ID frames (`arq.md` §7a) are the application's call. A port with FROM
  identifies the sender of each burst.
