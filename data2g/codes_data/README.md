# data2g/codes_data

Committed data the modem reads at run time. Despite the name, only the
LDPC shift tables belong to the codes; the rest is the modem's, the
waveform's and the ARQ's. Each file comes from a script or a study (redo
that to change the file), and most have a C++ mirror that
`tools/gen_native_tables.py` writes into `native/core/generated/` (run
it, then `--check`, after changing one).

| File | Written by | Read by | Native (generated) |
|---|---|---|---|
| `ldpc_shifts.npz` | `scripts/own_shifts.py` (gen, screen, export) | `ldpc.tables` / `qc_code` | `ldpc.cpp` (`tables::LDPC_SHIFTS`) |
| `header_code.npy`, `header_code_w48.npy`, `header_code_n10.npy` | `scripts/design_header.py --out` | `modem.header_code` (per band) | `modem.hpp` |
| `clip_constants.json` | `scripts/clip_constants.py --write`, `scripts/pick_ace.py --json` | `config.clip_consts` | `modem.hpp` |
| `outcome_predictor.npz` | `scripts/train_outcome.py` (data: `scripts/outcome_data.py`) | `arq/predictor.py`, `arq/policy.py` | `predictor.cpp` |
| `outcome_predictor_gate.npz` | `scripts/train_outcome.py` (`runs/n10f_round.sh`'s n10qf) | `arq/predictor.GATE_MODEL` | `predictor.cpp` |
| `capacity_tables.npz` | `scripts/rx_audit.py` | `arq/predictor.py` | `predictor.cpp` |
| `mode_thresholds.json` | `scripts/ladder_study.py` results (`runs/ladder_dd_*.csv`) | `arq/policy.MODE_THRESHOLDS` | `predictor.cpp` |
| `link_abstraction.json` | `scripts/link_abstraction.py --write` | `arq/predictor.abstraction` (`scripts/linksim.py`) | none |

The on-air formats proper (interleavers, polar info sets) are not here:
they are frozen per submode in `data2g/format/*.npz` by
`tools/freeze_format.py` (`codes.frozen`).
