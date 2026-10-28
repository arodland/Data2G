# docs/

Start with [ARCHITECTURE.md](ARCHITECTURE.md). Specs describe the code as it
is and are kept current; plans are history: what was decided and measured
at the time, with names that may since have moved.

| Doc | Kind | What |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Map | Pipelines, layers, Python and native, the parity contract, where to start. |
| [arq.md](arq.md) | Spec | The connected ARQ protocol: burst layout, control word, sessions, the gear shifter, state agreement. |
| [bulk.md](bulk.md) | Experiment | data2g-bulk: one-way text transfer with RV 1 copies and timed bursts, and its studies. Not finished, no native port; may be reworked or dropped. |
| [broadcast.md](broadcast.md) | Spec | Broadcast over KISS: named groups, group keys, rate-shifting reports. |
| [analog-slots.md](analog-slots.md) | Draft (shelved) | Client-defined analog latents (SSTVAE pictures, RADE voice) over Data2G: the API, the burst, hybrid ARQ, open questions. Shelved: Part 97 rules out mixing data and image/phone in one session. |
| [sounding.md](sounding.md) | Tool | Over-the-air sounding: every mode in one WAV, and an offline report on a recording of it (decodes, genie channel, waveform integrity, the thresholds' predictions). |
| [outcome-training-data.md](outcome-training-data.md) | Provenance | Every dataset in the outcome model's training mix and how to rebuild it. |
| [native-port-plan.md](native-port-plan.md) | Plan (history) | The C++ / Qt port, phase by phase. Its threads, frozen-data and conventions sections still hold. |
| [modem73-ideas-plan.md](modem73-ideas-plan.md) | Plan (history) | Ideas taken from modem73 and aicodix/modem: the blanker, KISS channel access, studies. |
| [interference-plan.md](interference-plan.md) | Plan (history) | Interference in the session simulator, and how the outcome model should account for it, with results. |
| [bicm-id-study.md](bicm-id-study.md) | Study (shelved) | Iterative demapping: up to 0.6 dB on the learned 64/256-point modes, not worth its CPU for now. |

Elsewhere: [native/README.md](../native/README.md) (the port's layout),
[data2g/codes_data/README.md](../data2g/codes_data/README.md) (committed
model and code data), [scripts/README.md](../scripts/README.md),
[tools/README.md](../tools/README.md).
