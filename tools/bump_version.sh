#!/usr/bin/env bash
# Set the application version everywhere it is written down, and commit
# it as "Release vX.Y.Z". Ported from SSTVAE's.
#
#   tools/bump_version.sh <x.y.z> [--dry-run] [--force] [--no-commit]
#
# The version is written down in four places: native/CMakeLists.txt,
# pyproject.toml, data2g/__init__.py and uv.lock. `.github/workflows/release.yml`
# checks the first two against the tag before it builds anything, so a
# mismatch fails early rather than shipping a download called 0.2.0 whose
# About box says 0.1.0. CMake's copy also reaches the .exe version block,
# the macOS bundle, the About box and the C++ host's VERSION reply;
# `data2g.__version__` is the Python host's (tests/test_native_host.py
# checks the two agree). uv.lock records the project's own version in its
# `[[package]]` entry; nothing checks it, which is why it drifts (SSTVAE
# v0.1.1 shipped with its lock at 0.1.0).
#
# Not touched: the metainfo <releases> list. An entry needs a hand-written
# description, so add it by hand when a release deserves one.
#
# The current version is read with the release workflow's own sed
# expressions, and the result is read back the same way: a bump that
# edits a line the gate does not match is as broken as no bump at all,
# and would be found a tag later.
#
# Linux/bash: a developer's pre-release step, not something CI runs.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: tools/bump_version.sh <x.y.z> [--dry-run] [--force] [--no-commit]

  --dry-run    report what would change and write nothing
  --force      allow a version that is not greater than the current one,
               and allow a version whose git tag already exists
  --no-commit  edit the files and leave them in the working tree
EOF
    exit 2
}

version=""
dry_run=0
force=0
commit=1
for arg in "$@"; do
    case "$arg" in
        --dry-run)   dry_run=1 ;;
        --force)     force=1 ;;
        --no-commit) commit=0 ;;
        -h|--help) usage ;;
        -*)        echo "unknown option: $arg" >&2; usage ;;
        *)
            [ -z "$version" ] || { echo "give exactly one version" >&2; usage; }
            version="$arg"
            ;;
    esac
done
[ -n "$version" ] || usage

# Strict numeric triple: installer.nsi builds `VIProductVersion
# "${VERSION}.0"`, which must be four integers, and the release gate
# extracts `[0-9.]*`, so `1.2.0-rc1` would truncate to 1.2.0 there,
# compare equal, and fail inside makensis at the far end of the build.
if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "version must be x.y.z with numeric parts (got '$version')" >&2
    exit 1
fi

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

cmake_file="native/CMakeLists.txt"
pyproject_file="pyproject.toml"
init_file="data2g/__init__.py"
lock_file="uv.lock"

for f in "$cmake_file" "$pyproject_file" "$init_file" "$lock_file"; do
    [ -f "$f" ] || { echo "missing $f -- is $repo the right tree?" >&2; exit 1; }
done

# Everything below is checked before anything is written, so a refusal
# leaves the tree as it was found rather than half-bumped.
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "not a git repository: $repo" >&2
    exit 1
fi

# Checked up front, not beside the `uv lock` it guards, which runs after
# the text files are written. A hard error, not a skip: a bump that left
# the lock behind is the failure that step exists to fix.
if [ "$dry_run" -eq 0 ] && ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed, so $lock_file cannot be regenerated." >&2
    echo "install uv (this project's dev tool) and re-run" >&2
    exit 1
fi

# The whole index must be empty, not just these files: `git commit`
# without a pathspec takes everything staged, and anything riding along
# under "Release" is invisible to whoever reads the tag or bisects.
if ! git diff --cached --quiet; then
    echo "there are staged changes:" >&2
    git diff --cached --name-only | sed 's/^/  /' >&2
    echo "commit or unstage them first (a release commit must be the bump alone)" >&2
    exit 1
fi

# Unrelated working-tree edits are fine; they are never staged here. An
# unstaged edit to a file this script will `git add` is not.
if [ "$commit" -eq 1 ]; then
    dirty="$(git diff --name-only -- "$cmake_file" "$pyproject_file" "$init_file" "$lock_file")"
    if [ -n "$dirty" ]; then
        echo "uncommitted changes in files this script commits:" >&2
        printf '  %s\n' $dirty >&2
        echo "commit or revert them first, or pass --no-commit" >&2
        exit 1
    fi
fi

# The first two expressions are copied verbatim from release.yml's check.
read_cmake()     { sed -n 's/^project(data2g_native VERSION \([0-9.]*\).*/\1/p' "$cmake_file" | head -1; }
read_pyproject() { sed -n 's/^version = "\([0-9.]*\)".*/\1/p' "$pyproject_file" | head -1; }
read_init()      { sed -n 's/^__version__ = "\([0-9.]*\)".*/\1/p' "$init_file" | head -1; }

# The `data2g` entry's version, not the file's first `version =` line,
# which belongs to some dependency.
read_lock() {
    awk '/^name = "data2g"$/ { f = 1; next }
         f && /^version = "/  { sub(/^version = "/, ""); sub(/".*$/, ""); print; exit }' \
        "$lock_file"
}

cmake_now="$(read_cmake)"
pyproject_now="$(read_pyproject)"
init_now="$(read_init)"
lock_now="$(read_lock)"

for pair in "$cmake_file:$cmake_now" "$pyproject_file:$pyproject_now" "$init_file:$init_now" "$lock_file:$lock_now"; do
    if [ -z "${pair#*:}" ]; then
        echo "could not read a version from ${pair%%:*} -- has the line moved?" >&2
        exit 1
    fi
done

echo "current: $cmake_file=$cmake_now  $pyproject_file=$pyproject_now"
echo "         $init_file=$init_now  $lock_file=$lock_now"
echo "new:     $version"

if [ "$cmake_now" != "$pyproject_now" ] || [ "$cmake_now" != "$init_now" ] || [ "$cmake_now" != "$lock_now" ]; then
    echo "note: the files disagree today; this run makes them agree." >&2
fi

# Against the highest of them, so a half-finished previous bump cannot
# make a real increase look like a decrease. `sort -V`: 0.10.0 > 0.9.0.
highest="$(printf '%s\n' "$cmake_now" "$pyproject_now" "$init_now" "$lock_now" | sort -V | tail -1)"
if [ "$version" = "$highest" ]; then
    if [ "$force" -eq 0 ]; then
        echo "already at $version (use --force to rewrite anyway)" >&2
        exit 1
    fi
elif [ "$(printf '%s\n%s\n' "$version" "$highest" | sort -V | head -1)" = "$version" ]; then
    if [ "$force" -eq 0 ]; then
        echo "$version is older than $highest (use --force if that is deliberate)" >&2
        exit 1
    fi
fi

# An existing tag is a release already built, and the workflow would
# rebuild it from the tagged commit, not the one this bump is heading for.
if git rev-parse --verify -q "refs/tags/v$version" >/dev/null 2>&1; then
    if [ "$force" -eq 0 ]; then
        echo "tag v$version already exists (use --force if you mean to reuse it)" >&2
        exit 1
    fi
    echo "note: tag v$version already exists; continuing because --force." >&2
fi

# Anchored to the one line the reader above matches, and each file must
# have exactly one: a second copy would mean half the file bumped.
bump() {
    local file="$1" match="$2" replace="$3" n
    n="$(grep -cE -- "$match" "$file" || true)"
    if [ "$n" != "1" ]; then
        echo "expected exactly 1 version line in $file, found $n" >&2
        exit 1
    fi
    if [ "$dry_run" -eq 1 ]; then
        printf '  would edit %-24s %s\n' "$file" "$(grep -E -- "$match" "$file")"
        return
    fi
    sed -i -E "s|$match|$replace|" "$file"
}

bump "$cmake_file" \
     '^project\(data2g_native VERSION [0-9]+\.[0-9]+\.[0-9]+' \
     "project(data2g_native VERSION $version"
bump "$pyproject_file" \
     '^version = "[0-9]+\.[0-9]+\.[0-9]+"' \
     "version = \"$version\""
bump "$init_file" \
     '^__version__ = "[0-9]+\.[0-9]+\.[0-9]+"' \
     "__version__ = \"$version\""

if [ "$dry_run" -eq 1 ]; then
    echo "  would run  uv lock                 ($lock_file: $lock_now -> $version)"
    echo "dry run: nothing written"
    exit 0
fi

# Regenerated, never sed'd: uv.lock's format is uv's to change. Without
# --upgrade it keeps every pin and only rewrites the project's own entry.
uv lock

# Read back through the gate's expressions: sed exits 0 whether or not
# anything matched.
ok=1
for pair in "$cmake_file:$(read_cmake)" "$pyproject_file:$(read_pyproject)" "$init_file:$(read_init)" \
            "$lock_file:$(read_lock)"; do
    got="${pair#*:}"
    if [ "$got" != "$version" ]; then
        echo "${pair%%:*} still reads '$got' after the edit" >&2
        ok=0
    fi
done
[ "$ok" -eq 1 ] || exit 1

echo "bumped to $version in:"
printf '  %s\n' "$cmake_file" "$pyproject_file" "$init_file" "$lock_file"

if [ "$commit" -eq 0 ]; then
    cat <<EOF

Left in the working tree (--no-commit). Next:

  git diff -- $cmake_file $pyproject_file $init_file $lock_file
EOF
    exit 0
fi

# The index was empty at the top and these files had no unstaged edits,
# so what is staged is exactly what this script wrote.
git add -- "$cmake_file" "$pyproject_file" "$init_file" "$lock_file"
git commit -q -m "Release v$version"
echo
git --no-pager show --stat --oneline HEAD

cat <<EOF

Next: check that commit and push it. Write the notes on a draft release for
tag v$version (targeting that commit or its branch), then release it:

  gh workflow run release.yml -f tag=v$version
EOF
