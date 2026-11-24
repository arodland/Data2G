# Sounding report: KC2G seed 0

- recording: 23.6 s at 16000 Hz; sequence found at 1.74 s
- bursts: 3 sent, 3 headers heard, 3 received, codewords 20/27; 0 heard outside the sequence
- path SNR (PEP, genie, 3 bursts): median 36.0 dB, 10-90% 35.4..36.7, min 35.2, max 36.9; 0% of bursts over 6 dB under the median
- channel: Doppler 0.00 Hz, delay spread - ms, CFO 0.0 Hz; thresholds column: static
- f55 carrier response (dB re median, low to high): -2.2 -1.1 -0.6 -0.2 -0.2 -0.0 +0.5 +0.5 +0.5 +0.3 +0.8 +0.7 +1.1 +0.9 +0.7 +0.8 +0.7 +0.7 +0.9 +1.0 +0.8 +0.5 +0.9 +0.6 +0.2 +0.6 +0.5 +0.1 +0.1 +0.1 +0.2 +0.2 +0.0 -0.1 +0.0 -0.0 -0.3 -0.4 -0.2 -0.7 -0.5 -0.8 -0.7 -0.9 -1.0 -1.2 -1.2 -1.4 -1.4 -1.6 -1.7 -1.8 -1.7 -1.9 -2.4
- clipping: 0 bursts with samples at full scale (worst 0.00%); peak -5.9 dBFS
- burst head/tail power re median (first/last 40 ms): head median +2.3 dB (worst +2.2), tail median +0.8 dB (worst -0.2)

## Per mode

| mode | heard | cw ok | genie SNR | snr_est | predicted | 10% thr | margin | pred | got |
|---|---|---|---|---|---|---|---|---|---|
| f55-qpsk-r21 | 1/1 | 4/4 | 31.8 | 26.4 | 26.3 | -1.2 | - | pass | pass |
| f55-64l-r67 | 1/1 | 8/8 | 29.9 | 26.3 | 26.3 | 14.6 | - | pass | pass |
| f55-256l-r88 | 1/1 | 8/15 | 28.1 | 25.2 | 26.3 | 27.9 | - | fail | fail |

## Per burst

| n | mode | rec s | hdr | cw ok | genie SNR | PEP | snr_est | Doppler | delay | head | tail | clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | f55-qpsk-r21 | 7.13 | 0.94 | 4/4 | 31.8 | 36.0 | 26.4 | 0.01 | 0.32 | 2.4 | -0.2 | 0.000 |
| 1 | f55-64l-r67 | 10.79 | 0.97 | 8/8 | 29.9 | 36.9 | 26.3 | 0.00 | 0.30 | 2.2 | 1.9 | 0.000 |
| 2 | f55-256l-r88 | 14.45 | 0.96 | 8/15 | 28.1 | 35.2 | 25.2 | 0.00 | 0.32 | - | - | 0.000 |
