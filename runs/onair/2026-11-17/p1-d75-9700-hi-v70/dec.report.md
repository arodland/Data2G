# Sounding report: KC2G seed 0

- recording: 23.6 s at 16000 Hz; sequence found at 1.74 s
- bursts: 3 sent, 3 headers heard, 3 received, codewords 26/27; 1 heard outside the sequence
- path SNR (PEP, genie, 3 bursts): median 38.5 dB, 10-90% 35.0..39.2, min 34.2, max 39.4; 0% of bursts over 6 dB under the median
- channel: Doppler 0.00 Hz, delay spread - ms, CFO 0.0 Hz; thresholds column: static
- f55 carrier response (dB re median, low to high): -2.2 -1.1 -0.6 -0.2 -0.1 -0.0 +0.5 +0.5 +0.4 +0.3 +0.7 +0.8 +1.1 +0.8 +0.7 +0.8 +0.7 +0.7 +0.9 +1.0 +0.8 +0.5 +0.9 +0.6 +0.2 +0.6 +0.5 +0.0 +0.1 +0.1 +0.2 +0.2 -0.0 -0.1 -0.0 -0.0 -0.3 -0.4 -0.2 -0.8 -0.6 -0.8 -0.8 -0.9 -1.0 -1.2 -1.2 -1.5 -1.4 -1.6 -1.7 -1.8 -1.8 -2.0 -2.4
- clipping: 0 bursts with samples at full scale (worst 0.00%); peak -4.1 dBFS
- burst head/tail power re median (first/last 40 ms): head median +1.1 dB (worst -1.2), tail median -0.3 dB (worst -1.8)

## Per mode

| mode | heard | cw ok | genie SNR | snr_est | predicted | 10% thr | margin | pred | got |
|---|---|---|---|---|---|---|---|---|---|
| f55-qpsk-r21 | 1/1 | 4/4 | 29.9 | 24.8 | 26.6 | -1.2 | - | pass | pass |
| f55-64l-r67 | 1/1 | 8/8 | 32.5 | 27.3 | 26.6 | 14.6 | - | pass | pass |
| f55-256l-r88 | 1/1 | 14/15 | 31.4 | 26.6 | 26.6 | 27.9 | - | fail | pass |

## Per burst

| n | mode | rec s | hdr | cw ok | genie SNR | PEP | snr_est | Doppler | delay | head | tail | clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | f55-qpsk-r21 | 7.13 | 0.93 | 4/4 | 29.9 | 34.2 | 24.8 | 0.01 | 0.33 | 1.1 | -0.3 | 0.000 |
| 1 | f55-64l-r67 | 10.79 | 0.96 | 8/8 | 32.5 | 39.4 | 27.3 | 0.00 | 0.29 | 1.1 | 0.2 | 0.000 |
| 2 | f55-256l-r88 | 14.44 | 0.96 | 14/15 | 31.4 | 38.5 | 26.6 | 0.00 | 0.33 | -1.2 | -1.8 | 0.000 |
