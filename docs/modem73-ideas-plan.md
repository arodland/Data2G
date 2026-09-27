# Plan: ideas from modem73 and aicodix/modem

Approved 2026-09-27: all six items below. Sources, both permissive
(modem73: public domain; aicodix/modem: 0BSD):
- https://github.com/RFnexus/modem73 (at 3a37c29)
- https://github.com/aicodix/modem (at a4a8b57)

Work happens in the worktree `.claude/worktrees/modem73-ideas`, branch
`worktree-modem73-ideas`, based on master 2adc5d6 (the PR #4 merge). Another
session works in the main checkout: don't edit there.

Standing rules that apply: cap CPU (8 pool workers, 1 BLAS thread each);
chain long runs sequentially; don't `uv add` or `uv sync` (use
`uv run --no-sync`; pytest via `--with pytest`); report 10% points alongside
1%; ARQ state-agreement rules. Commit and PR only when asked.

## Order

The cheap items first (1-3), then the studies (4-6). Each study reports
before any protocol change is made from it.

### 1. Impulse blanker

- **Source:** modem73 `phy/robust_modem.hh`, `process()`, `#if IMPULSE_BLANKER`.
  Per sample: a fast envelope (EMA of |x|, 1/64) and a slow one (EMA
  1/4096, fed min(|x|, 3 x env)). Zero samples above 8 x env, limit samples
  above 6 x env down to 6 x env. Resync the slow envelope to the fast one
  after 256 samples of the slow one being under 1/6 of the fast (a level
  step).
- **Where:** the start of the receive path, `tnc.Receiver.feed` (it already
  sees every block), vectorized in numpy over each block with the state
  carried across calls. Never on transmit.
- **Test:** add an impulsive-noise option to `data2g/hfchannel.py` (Poisson
  clicks: rate, amplitude relative to the signal, 0.1-2 ms bursts).
- **Measure:**
  - The decode rate with and without the blanker at a few (SNR, click rate)
    cells.
  - False locks on noise plus clicks.
  - It must be a no-op on Gaussian noise: the ladder cells at 0 dB and
    -4 dB show no change.
- **Done when:** a unit test (clicks removed, Gaussian untouched), the
  numbers above in the README, and the full suite passes.

### 2. KISS channel access (p-persistence)

- **Source:** modem73 `kiss_tnc.hh` (CMD_TXDELAY 0x01, P 0x02,
  SLOTTIME 0x03, TXTAIL 0x04; `p_persistence` default 128).
- **Where:**
  - `data2g/tnc.py` `_KissHandler`: parse KISS commands 1-4 (today it only
    logs "ignored") into a shared parameter set.
  - `data2g/arq/engine.py`: when a KISS burst is ready and the channel is
    free, send with probability (P+1)/256, else wait SLOTTIME and retry.
    TXDELAY adds to the PTT delay for KISS bursts; TXTAIL is accepted and
    ignored (our bursts end cleanly).
- **Defaults:** P = 63 (25%), SLOTTIME 100 ms, as classic KISS TNCs. The
  engine is sample-clocked, so slots count in samples, not wall time.
- **Test:** engine test with two KISS stations queueing at the same moment
  on a busy channel: with persistence they don't collide every time (many
  seeds; collision rate well under 100%).

### 3. A cap on waiting for BUSY (KISS)

- **Source:** modem73 `csma.hh` `busy_limit_ms` (60 000).
- **Where:** the engine's KISS send path. If `channel_busy` has held a
  queued KISS burst longer than the cap, send anyway and log it. ARQ keeps
  its own timers, unchanged.
- **Default:** 60 s, a `data2g-host` option (`--kiss-busy-limit`).
- **Test:** engine test where BUSY is stuck on (a fake receiver), and the
  burst goes out after the cap.

### 4. Study: polar list size 8 -> 16 -> 32

- **Source:** aicodix/modem73 decode polar codes with CRC-aided SCL, list size 32.
- **Where:** `codes.POLAR_LIST` (8). Polar codes are the reply modes (ack-*,
  n*-ack-*, polar-k*) and the CPM control codeword (k=176, n=360).
- **Study script** (`scripts/polar_list_study.py`): codeword failure
  vs SNR, AWGN and the fading presets, list sizes 8/16/32. Uses the real
  modem's LLRs, so full bursts, not just the code. Also measures decode CPU
  per codeword.
- **Decide:** adopt the size where the 10% point gains >= 0.2 dB and CPU
  stays acceptable (control codewords are short: expect ms). No format change,
  receiver only.

### 5. Study: re-decode failed codewords with erasures

- **Source:** modem73 `robust_modem.hh` around line 2420: after a CRC
  failure, retry with the worst 1/8 and then 1/4 of rows (symbols) erased,
  then with a different channel smoothing radius, then SNR-floor erasure
  (rows under 0.3 x the frame median quality).
- **Our analogue:** per-OFDM-frame (or per-CPM-symbol) quality from the
  equalizer's per-frame SNR; zero the LLRs of the worst frames' symbols and
  re-run the LDPC/polar decode. Only on failure, so no CPU cost when
  things work.
- **Study script** (`scripts/erasure_retry_study.py`): real-modem bursts on
  MPG/MPP/MPD at SNRs around the 10% points; the fraction of failed
  codewords a retry ladder recovers, and any wrong decodes accepted (the
  CRC's false-accept rate must not rise measurably).
- **Decide:** adopt if it recovers >= ~10% of failures on fading.
  Receiver-only; IR soft-bit storage must store the *unerased* LLRs.

### 6. Study, then maybe change the waveform: SLM peak reduction

- **Source:** aicodix `encode.cc` `symbol()`. Per OFDM symbol, try up to
  128 seeds (each flips the data tones' signs by an MLS sequence), keep the
  lowest-PAPR one, stop early under PAPR 5; the seed rides 64 pilot tones
  as a Hadamard codeword. Then clipping and filtering as today.
- **Our analogue:** per frame (6 symbols), or per symbol. The seed index
  has to reach the receiver:
  - either carried in the frame pilot (its phase pattern), or
  - blind: the receiver tries each seed's descrambling. That's costly and
    error-prone, so the carried form is preferred.
- **Study first** (`scripts/slm_study.py`), no waveform change yet:
  - the peak distribution (per-burst peak/average, the 99.9th percentile)
    with 1, 8, 32 or 128 candidates, per band;
  - the resulting clip noise at today's headroom settings (config CLIP),
    and the headroom that gives the same clip noise with SLM.
- **If it works** (the user's condition), the full job:
  1. Implement the transmit selection and the seed carriage, and bump the
     protocol version.
  2. Re-tune per-mode clip headroom (`scripts/pick_headroom.py`) and
     re-derive the clip constants (`scripts/clip_constants.py`).
  3. Re-measure the ladder (`scripts/ladder_study.py`, 10% and 1%), update
     the ladder artifact (https://claude.ai/artifact/E4yfMVQJ4dgMF5RbkqSQKD).
  4. Refresh outcome data for the affected modes and retrain (session data
     plus offline; `scripts/session_data.py`, `scripts/train_outcome.py`
     ensemble).
  5. Loss study (12 seeds x 600 s) against the current numbers.

## Not doing (recorded so it isn't re-proposed)

- **A postamble or mid-burst acquisition:** first measure how many missed
  MPP bursts are preamble misses rather than header misses.
- **A Schmidl-Cox detector:** cheaper, but it triggers on steady tones;
  PR #3 already cut our detector's CPU.
- **A tone lead-in for busy detection:** 650 ms per burst.
- **Adopting their MFSK/ROBUST families:** our CPM and OFDM ladders cover
  them (our fsk16r25 AWGN 10% point is -12.5 dB, against their MFSK-8 at
  -9 dB).
