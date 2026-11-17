# Sounding report: KC2G seed 0

- recording: 23.6 s at 16000 Hz; sequence found at 1.75 s
- bursts: 3 sent, 3 headers heard, 3 received, codewords 12/27; 1 heard outside the sequence
- path SNR (PEP, genie, 3 bursts): median 32.4 dB, 10-90% 31.0..32.7, min 30.7, max 32.8; 0% of bursts over 6 dB under the median
- channel: Doppler 0.00 Hz, delay spread - ms, CFO 0.0 Hz; thresholds column: static
- f55 carrier response (dB re median, low to high): -2.1 -1.1 -0.6 -0.2 -0.2 -0.0 +0.5 +0.5 +0.5 +0.4 +0.7 +0.8 +1.1 +0.9 +0.7 +0.9 +0.8 +0.8 +0.9 +1.0 +0.9 +0.6 +0.9 +0.6 +0.2 +0.6 +0.5 +0.1 +0.1 +0.1 +0.1 +0.2 +0.0 -0.0 +0.0 +0.0 -0.3 -0.3 -0.2 -0.7 -0.5 -0.8 -0.7 -0.9 -1.0 -1.2 -1.2 -1.4 -1.3 -1.6 -1.6 -1.7 -1.7 -2.0 -2.4
- clipping: 0 bursts with samples at full scale (worst 0.00%); peak -7.2 dBFS

## Per mode

| mode | heard | cw ok | genie SNR | snr_est | predicted | 10% thr | margin | pred | got |
|---|---|---|---|---|---|---|---|---|---|
| f55-qpsk-r21 | 1/1 | 4/4 | 28.6 | 25.8 | 24.1 | -1.2 | - | pass | pass |
| f55-64l-r67 | 1/1 | 8/8 | 25.5 | 24.1 | 24.1 | 14.6 | - | pass | pass |
| f55-256l-r88 | 1/1 | 0/15 | 23.5 | 21.8 | 24.1 | 27.9 | - | fail | fail |

## Per burst

| n | mode | rec s | hdr | cw ok | genie SNR | PEP | snr_est | Doppler | delay | head | tail | clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | f55-qpsk-r21 | 7.13 | 0.94 | 4/4 | 28.6 | 32.8 | 25.8 | 0.01 | 0.31 | - | - | 0.000 |
| 1 | f55-64l-r67 | 10.79 | 0.96 | 8/8 | 25.5 | 32.4 | 24.1 | 0.00 | 0.27 | - | - | 0.000 |
| 2 | f55-256l-r88 | 14.45 | 0.96 | 0/15 | 23.5 | 30.7 | 21.8 | 0.00 | 0.19 | - | - | 0.000 |
