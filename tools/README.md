# tools/

Each file's header has its usage.

| Tool | |
|---|---|
| **Generated data** | |
| `gen_native_tables.py` | `native/core/generated/` from `data2g/` (constants, submodes, frozen tables). `--check` in CI: fails until regenerated. |
| `freeze_format.py` | `data2g/format/*.npz`: each submode's interleaver and polar info set, frozen so the on-air format can't drift with numpy. |
| `gen_icons.py` | The app icons from `native/packaging/data2g.svg`. |
| **Checks (CI)** | |
| `check_layering.py` | native/'s include rules: no Qt in `core/` outside device backends, Widgets only in `gui/`, ldpc/polar below codes, generated files only generated. |
| `check_includes.py` | Every `std::` name a native source uses has its header included (MSVC is strict where GCC and Clang are not). |
| **Native build and parity** | |
| `build_native.sh` | Configure and build `native/` for the project's venv; `--test` runs ctest and `pytest --native`. |
| `with_native.py` | Run a study script with the C++ core substituted, as `pytest --native` does (`tests/native_subs/`); `--skip arq,gear` leaves areas in Python. |
| `bench_native.py` | Timings of the C++ hot paths per pool size (run under `with_native.py`). |
| `sample_threads.py` | Sample a running process's threads: state, wait channel, CPU. |
| **Packaging** | |
| `package_app.sh` | Stage a runnable copy of the native apps with Qt. |
| `make_installer.sh` | Wrap the staged tree: .dmg, AppImage, NSIS installer. |
| `sign.sh` | Sign the staged app and the installer. |
