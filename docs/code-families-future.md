# Code families: future ARQ and ladder work

Status: a record of an investigation, not a plan. Split out of `docs/app-soft-plan.md` (PR #133) on
2026-10-29. The payoff measured was minimal (§2 below: today's families would recover 0.1-0.2% of
ARQ codewords), so nothing here is planned, and the unit store keys by submode
(`docs/app-soft-plan.md` §2e).

## 1. ARQ mode changes within a family

Not part of this plan. It is sketched here so the store and family definitions in `docs/app-soft-plan.md` don't block
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
  - The soft bits combine by position (`docs/app-soft-plan.md` §2e).
  - A shift from qpsk-r1/3 to n4-qpsk-r2/3, or from w48-qpsk-r2/3 to w48-16qam-r1/3, could carry
    on with resends and IR, with no abandon and nothing dropped.
- What would change:
  - `docs/arq.md` §4's abandon rule becomes: abandon only when the new mode is in another family.
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
  fuzz tests. Its own plan, after app-soft-plan's phase 3 lands.

## 2. Fewer, larger code families

Not part of this plan, and no decision is asked for here. The question is whether changing some
modes' codes, so that more modes share one family, would let more mode changes and repeats combine
by IR. In short: the most it could gain is the airtime of codewords discarded at mode changes. That
is a few percent overall and 9-24% in the floor fading cells. One cheap oracle study can tell
whether that is worth 1 dB before anything else is considered.

### 2.1 Why families are few today

- A family is (base graph, Z, k) (`docs/app-soft-plan.md` §2e).
- Within a band, ladder rungs of one constellation share n (frames per codeword) and differ in k:
  qpsk-r1/5, r1/3 and r1/2 all have 1920 coded bits and carry 46, 76 and 116 B.
- So families arise only where k happens to coincide: the same rate in w and w48, a few
  constellation pairs, and n4-qpsk-r2/3 with qpsk-r1/3.

### 2.2 How often the shifter throws held soft bits away

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
2500 (5%), so §1's ARQ change alone, with today's codes, would recover about 0.1-0.2% of codewords.

Caveats:

- The link abstraction models OFDM only. 36 runs (31 of them at -5 dB) stopped when the shifter
  picked a CPM mode, so the floor cells are partly sampled.
- 8 seeds a cell.

### 2.3 What those codewords would have gained

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
  the airtime they took: the shares in §2.2.

### 2.4 What consolidation would take

- One k per family, with the rate set by n. Lower rungs become longer codewords (more frames each).
  A family's lowest rate is its mother's:
  - base graph 2 here is 0.17-0.2 (k / L: 384/2304, 640/3520, 960/4800);
  - base graph 1 is about 0.31 (1440/4608).
- So a family spanning r1/2 to r1/5 must be base graph 2 at one k.
- The r3/4 rungs stay out. `ldpc.layout` uses base graph 1 above rate 0.67, and base graph 2 is the
  weaker code at high rates.
- Threshold risk:
  - Holding k at the higher rung's value makes the low rungs longer codes: a slightly better
    threshold, but coarser granularity. Short writes waste more, though inlining
    (`docs/arq.md` §5a) helps.
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
- Then §1's ARQ change and its session `VERSION` bump.

### 2.5 Verdict and what to measure first

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
