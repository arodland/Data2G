# Data2G broadcast (draft for review)

TLDR: KISS grows into named broadcast groups. Each group is a KISS port, opened by a
command on the command port. A group's bursts carry its name in the control codeword,
and every codeword in them, control and data, uses a CRC mask hashed from that name. A
receiver decodes the control without a mask, reads the name and checks the CRC with
its key: no fixed broadcast key, and no trying groups one by one. Port 0 is today's
KISS under the group name "VARA KISS", with the same mode shifting.

Needs a decision:

- Should the command port (8300) run without `--vara`, since broadcast commands and statuses need it?
- The status set in §5.
- Port 0's wire changes anyway (its key becomes the hash of "VARA KISS"). Should its
  control move to the TLV format too (reports as a TLV), or keep today's layout?

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
  settles it. Only a control-lost burst can be misdelivered, and only between two
  colliding groups that are both open.

## 3. Burst format

- **Control:** a 1-byte header `[version 4 bits | reserved 2 | n_ctl - 1 (2)]` (version
  2), then TLVs, as `frames.Control` without the 32-bit core. The ARQ core's fields mean
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

- **Groups other than 0:** one mode per port, set by the host. The default is the cap's
  robust broadcast mode (`kisslink.BROADCAST`). One-to-many traffic can't be shifted
  from reports.
- **Port 0:** today's per-station shifting, unchanged.
- **One queue for all ports:** the next burst takes the first queued frame's port and
  mode, then every queued frame of that port that fits. KISS p-persistence and
  SLOTTIME apply to all ports. Broadcast still goes only between ARQ sessions.

## 5. Host interface

Commands on the command port, replies as for VARA's (OK / WRONG):

| command | reply | does |
|---|---|---|
| `BCAST OPEN group [FROM call]` | `BCAST PORT n` | opens a port; FROM sends group+from |
| `BCAST CLOSE n` | OK | |
| `BCAST MODE n mode` | OK / WRONG | that port's mode, within the KISS cap |
| `MODES` | one `MODE ...` line per mode, then OK | name, bandwidth Hz, bytes per codeword, max codewords, airtime at 1 and at max codewords |

Statuses, to ports opened on this connection only:

- `BCAST n HEARD [call]`: a burst's control checked for port n (call from group+from).
  Its frames follow on the KISS port. It's known only at the burst's end: the receiver
  decodes whole bursts.
- `BCAST n LOST k`: the control decoded, but k frames were lost (header and control,
  no data).
- A burst whose header decoded but whose control didn't can't be tied to a port. It's
  BUSY ON/OFF, as now.

## 6. Not done

- **Ports beyond 15** (KISS SetHardware): 15 groups open at once is plenty until an
  app needs more.
- **Setting a port's mode by KISS command:** the command port does it. Add it when a
  KISS-only client needs it.
- **Incremental redundancy across broadcast bursts:** no resends, so no IR.
- **ID:** ID frames (`arq.md` §7a) are the application's call. A port with FROM
  identifies the sender of each burst.
