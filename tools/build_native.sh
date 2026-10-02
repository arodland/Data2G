#!/usr/bin/env bash
# Configure and build native/, including the pybind11 module, for the
# project's virtualenv.
#
#   tools/build_native.sh              # build
#   tools/build_native.sh --test       # build, then ctest and pytest --native
#   tools/build_native.sh --sanitize   # ASan + UBSan
#
# The module must be built for the Python that runs pytest, or the import
# fails and --native refuses to run. PYTHON overrides the choice.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/native/build"

# A worktree has no .venv of its own: use the main checkout's.
main="$(cd "$(git -C "$root" rev-parse --path-format=absolute --git-common-dir)/.." && pwd)"
for py in "${PYTHON:-}" "$root/.venv/bin/python" "$main/.venv/bin/python" "$(command -v python3)"; do
    [[ -n "$py" && -x "$py" ]] && break
done
echo "python: $py" >&2

cmake_args=(-S "$root/native" -B "$build" -G Ninja "-DPython3_EXECUTABLE=$py")

# pybind11 is usually a system package while the interpreter is the venv's
# (never `uv add` it: a sync replaces the venv's ROCm torch).
if pybind11_dir="$("$py" -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null)" ||
   pybind11_dir="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null)"; then
    cmake_args+=("-Dpybind11_DIR=$pybind11_dir")
fi

run_tests=0
for arg in "$@"; do
    case "$arg" in
        --sanitize) cmake_args+=(-DDATA2G_SANITIZE=ON) ;;
        --test)     run_tests=1 ;;
        *)          cmake_args+=("$arg") ;;
    esac
done

"$py" "$root/tools/gen_native_tables.py" --check
"$py" "$root/tools/check_layering.py"
cmake "${cmake_args[@]}"
cmake --build "$build" --parallel "${JOBS:-8}"  # shared machine: cap it

if (( run_tests )); then
    ctest --test-dir "$build" --output-on-failure
    (cd "$root" && "$py" -m pytest tests --native -q)
fi
