# BICM-ID (iterative demapping): studied, shelved

> Studied 2026-10-20, shelved: a gain under 1 dB, only on the fastest modes,
> at a high CPU cost. The study is `scripts/bicm_id_study.py`; its docstring
> has the same numbers. Nothing on the receiver changed.

## The idea

When an LDPC decode fails, feed the decoder's extrinsic LLRs (its posterior
less the channel LLRs it was given) back to the demapper as a priori bit
probabilities. Then demap the codeword's symbols again and decode again.
It is receiver only: no format change.

## The bound

BICM-ID can recover at most the gap between the constellation's BMI (what
a one-shot demapper delivers) and its coded-modulation MI.

- Gray labels make the two nearly equal.
- QPSK's two bits are independent, so it gains exactly nothing.
- The learned sets were trained for BMI through the clipper but did not come
  out Gray-like. Their demapper EXIT curves have some slope.

| Constellation | Rate | I_E at I_A = 0 -> 1 | CM limit below the BMI operating point |
|---|---:|---|---:|
| gray-qam16 | 1/2 | 0.499 -> 0.518 | 0.16 dB |
| c64-w48-r12 | 1/2 | 0.496 -> 0.551 | 0.52 dB |
| c64-w48-r712 | 7/12 | 0.580 -> 0.620 | 0.37 dB |
| c64-w48-r23 | 2/3 | 0.663 -> 0.691 | 0.24 dB |
| c64-w48-r34 | 3/4 | 0.748 -> 0.763 | 0.13 dB |
| c256-w48-r58 | 5/8 | 0.625 -> 0.671 | 0.58 dB |

## Measured

The setup:

- Each submode's real LDPC code and interleaver, on AWGN with ideal CSI and
  no clipper.
- 200 codewords per 0.25 dB step, so about ±0.1 dB.
- The control is the same decoder with the same total iterations and no
  feedback, on the same codewords and noise. Extra iterations alone also
  help.

The 10% PER point, Es/N0 in dB:

| Submode | Today (40 iterations) | 200 iterations, no feedback | BICM-ID (5 passes × 40) | Gain vs control |
|---|---:|---:|---:|---:|
| w48-16qam-r1/2 | 6.19 | 6.06 | 5.94 | ~0.1 dB |
| w48-64l-r1/2 | 10.25 | 10.17 | 9.67 | 0.5 dB |
| w48-64l-r2/3 | 13.42 | 13.25 | 13.08 | ~0.15 dB |
| w48-256l-r5/8 | 16.93 | 16.86 | 16.25 | 0.6 dB |

- Most of the gain comes in the first one or two passes.
- The prior-aware demapper costs m · 2^m per symbol and bit: the 256l run
  took about 2.5× the control's CPU.
- Not measured: fading channels, the clipper, or the real modem path.

## Why shelved

- The gain is under 1 dB.
- Only the learned 64/256-point LDPC modes benefit: the five w48 64l/256l
  modes plus n10-64l-r3/4 and n10-256l-r3/4.
- Those are the modes whose demapping is already the most expensive.
- A design for BICM-ID (non-Gray labels, EXIT-matched LDPC degrees) would
  change the format and cannot beat the bound above.

## If it is picked up again

1. Re-measure through the real modem on mpg/mpd, with the clipper on.
2. Hook it into the DD loop in `arq/phy.py`, which already re-decodes failed
   slots under `DD_ITERS` and `DD_BUDGET_S`. It needs:
   - `constellation.llr` with an optional prior;
   - the decoder's extrinsic rather than the posterior DD uses, so the
     channel information isn't counted twice;
   - re-demapping of only the current burst's symbols: earlier RVs are
     stored as soft bits.
3. Port it to native with parity tests.
4. Regenerate the mode thresholds and link abstraction, retrain the outcome
   model, and rerun the ladder pruning.
