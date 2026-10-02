# Data2G broadcast (draft for review)

TLDR: KISS grows into named broadcast groups. Each group is a KISS port, opened by a
command on the command port. A group's bursts carry its name in the control codeword,
and every codeword in them, control and data, uses a CRC mask hashed from that name. A
receiver decodes the control without a mask, reads the name and checks the CRC with
its key: no fixed broadcast key, and no trying groups one by one. Port 0 is today's
KISS under the group name "VARA KISS", with the same mode shifting.

## 1. Terms

- **Group:** up to 10 characters in the packed callsign alphabet (A-Z 0-9 / - and
  space, space padded at the end). It names an application, like `APRS` or `CHAT`, not
  a station.
- **Group key:** a nonzero 16-bit hash of the packed group (FNV-1a, as
  `session.session_key`).
- **Port:** a KISS port number (1-15) a host has opened for one group. Port 0 is
  always open, on "VARA KISS".

## 2. Masks: who can read what

Everything on air is plain: the scrambler is unkeyed (`arq.md` §2), so anyone decodes
and reads any codeword. A mask only decides whose CRC check passes.

- **One key per group, on every codeword:** `ctl_mask(0, i, group key)` for control,
  `data_mask(0, slot, group key)` for data. Port 0's key is the hash of "VARA KISS".
- **The control checks itself:**
  1. Decode slot 0 once, mask left open (`codes.decode_raw`; the engine does this
     already, for the session's key and mask 0).
  2. Parse the unverified bytes: the header's version must be 2, then the TLVs. The
     group comes from group or group+from; absent means "VARA KISS".
  3. Hash the group to its key and check the CRC (`codes.check`). Polar control
     (CPM) does this per list candidate, each with the group read from it.
  - A wrong decode or another burst type gives a garbage name, and the check fails.
    A false pass is 2^-16 per candidate (CRC-16), as for any mask today.
- **Cost:** one decode per burst, shared with the ARQ and CQ/ID checks, then CRC
  checks. Interest in many groups costs nothing extra: the burst names its own.
- **Filtering:** a receiver checks the control, then keeps the burst only if the
  group is an open port. Promiscuous listening keeps every group's.
- **Data:** checked under the group read from the control. Data from a burst whose
  control was lost never lands on the wrong port.
  - Control lost: the receiver checks slot 1's CRC under each open port's key (one
    decode, then up to 16 CRC checks). A promiscuous listener can still read that
    data, unverified.
- **Collisions:** two group names share a key 1 time in 65536. The control's name
  settles it. Control-lost data whose key matches two open ports is dropped, and both
  get a LOST notice (one a false positive): a payload is never misdelivered.

## 3. Burst format

- **Control:** every burst's control, port 0's included, is a 1-byte header
  `[version 4 bits | reserved 2 | n_ctl - 1 (2)]` (version 2), then TLVs, as `frames.Control` without the 32-bit core. The ARQ core's fields mean
  nothing to a broadcast burst, and CPM gives control exactly one 20 B codeword.
- **TLVs:**

  | type | contents | when |
  |---|---|---|
  | group | packed group, 8 B | every burst except port 0's (absent means "VARA KISS", and its key) |
  | group+from | group and sender, 15 B (120 bits packed together) | in place of group, when the port asks for it |
  | reports | sender hash (2 B), then per station [hash 2 B][mode code << 2 \| size hint] | port 0 only: today's AX.25 mode shifting |

  - group+from on CPM: 1 + 2 + 15 = 18 B, which fits 20 B. Separate group and from TLVs
    would be 1 + 10 + 10 = 21 B, which doesn't fit.
- **Data:** `[length, 2][frame]` back to back, zero length ends it, as today. One burst
  carries one group's frames.

## 4. Modes and channel access

- **Groups other than 0:** each port has a transmit mode, set by the host at any time
  between frames (it applies to frames sent after it). Receiving decodes every mode,
  whatever a port's transmit mode. The default is the cap's robust broadcast mode
  (`kisslink.BROADCAST`). One-to-many traffic can't be shifted from reports.
- **Port 0:** today's per-station shifting by default (`BCAST MODE 0 AUTO`). A fixed
  mode (`BCAST MODE 0 mode`) turns it off: no reports sent, reports heard ignored, and
  frames never parsed as AX.25.
- **One queue for all ports:** the next burst takes the first queued frame's port and
  mode, then every queued frame of that port that fits. KISS p-persistence and
  SLOTTIME apply to all ports. Broadcast still goes only between ARQ sessions.

## 5. Host interface

One integrated server: the command, data and KISS ports always run (the `--vara` /
`--kiss` flags go). Commands go on the command port, replies as for VARA's (OK / WRONG):

| command | reply | does |
|---|---|---|
| `BCAST OPEN group [FROM call]` | `BCAST PORT n` | opens a port; FROM sends group+from |
| `BCAST CLOSE n` | OK | |
| `BCAST MODE n mode` | OK / WRONG | that port's transmit mode for later frames, within the KISS cap; any time between frames, never filters decoding. Port 0 also takes `AUTO` (shifting, the default) |
| `MODES` | one `MODE ...` line per mode, then OK | name, bandwidth Hz, bytes per codeword, max codewords, airtime at 1 and at max codewords |

Statuses, to ports opened on this connection only:

- `BCAST n HEARD [call]`: a burst's control checked for port n (call from group+from).
  Its frames follow on the KISS port. It's known only at the burst's end: the receiver
  decodes whole bursts.
- `BCAST n LOST k`: the burst is the port's (control checked, or data under its key),
  but k frames were lost.
- Control lost: if a data slot passes under an open port's key, the burst is that
  port's (HEARD / LOST as usual; an ambiguous key as in §2 Collisions). Control and
  all data lost can't be tied to a port: the frozen 16-bit PHY header has no room for
  a group. Every open port gets a `BCAST * MISSED submode n_cw` hint; it may be
  another group's burst.

Apps learn when their frames went out through KISS ACKMODE (command `0x0C`, as in BPQ32
and QtSoundModem):

- **Ask per frame:** the app sends `0x0C` (with the port in the high nibble), a 2-byte
  tag of its choosing, then the frame. Plain data frames (`0x00`) get no ack, so apps
  that don't ask see no change. Works on every port, port 0 included.
- **The ack:** `0x0C`, the same port, and just the tag, sent on the KISS port when the
  burst carrying that frame finishes transmitting. Frames sharing a burst are acked
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
