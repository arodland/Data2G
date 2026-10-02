#!/usr/bin/env bash
# Turn a staged tree (tools/package_app.sh) into the platform's container.
#
#   tools/make_installer.sh [staging-dir] [output-basename]
#
# Lifted from SSTVAE:
#   macOS     .dmg with an /Applications symlink (install is a drag)
#   Linux     AppImage (runs on distributions older than the bundled Qt)
#   Windows   NSIS setup .exe (Start Menu, uninstaller, Apps & features);
#             the portable zip is published beside it
#
# Packaging tools are pinned by sha256 and fetched into $DATA2G_DEPS_DIR
# (default native/.deps, cached in CI), never installed system-wide: a tool
# that changes underneath us changes what we ship. Nothing here signs;
# that is tools/sign.sh, run before and after this on macOS and Windows.

set -euo pipefail

STAGE_DIR="${1:-dist}"
OUT_BASE="${2:-data2g}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPS_DIR="${DATA2G_DEPS_DIR:-$ROOT/native/.deps}"

if [ ! -d "$STAGE_DIR" ]; then
    echo "make_installer: no staged tree at $STAGE_DIR (run tools/package_app.sh first)" >&2
    exit 1
fi

# The version from the one place it is declared.
VERSION="$(sed -n 's/^project(data2g_native VERSION \([0-9.]*\).*/\1/p' "$ROOT/native/CMakeLists.txt" | head -1)"
[ -n "$VERSION" ] || { echo "make_installer: no version in native/CMakeLists.txt" >&2; exit 1; }

# Download to .part, check the hash, then rename: a truncated file is never
# found and used by the next run.
fetch() {  # url sha256 dest
    mkdir -p "$(dirname "$3")"
    echo "make_installer: fetching $1"
    curl -fsSL -o "$3.part" "$1"
    echo "$2  $3.part" | sha256sum -c - >/dev/null
    mv "$3.part" "$3"
}

case "$(uname -s)" in
# ---------------------------------------------------------------- Windows
MINGW*|MSYS*|CYGWIN*)
    # Not preinstalled on GitHub's Windows runners (an SSTVAE CI round).
    nsis_version=3.11
    nsis_sha=c7d27f780ddb6cffb4730138cd1591e841f4b7edb155856901cdf5f214394fa1
    nsis_dir="$DEPS_DIR/nsis-$nsis_version"
    if command -v makensis >/dev/null 2>&1; then
        makensis=makensis
    else
        if [ ! -x "$nsis_dir/makensis.exe" ]; then
            zip="$DEPS_DIR/nsis-$nsis_version.zip"
            fetch "https://downloads.sourceforge.net/project/nsis/NSIS%203/$nsis_version/nsis-$nsis_version.zip" \
                  "$nsis_sha" "$zip"
            # Git Bash has no unzip and its tar is GNU tar: PowerShell it is.
            powershell -NoProfile -NonInteractive -Command \
                "Expand-Archive -LiteralPath '$(cygpath -w "$zip")' -DestinationPath '$(cygpath -w "$DEPS_DIR")' -Force"
            rm -f "$zip"
        fi
        # The top-level stub, which finds Include/ and Stubs/ beside itself.
        makensis="$nsis_dir/makensis.exe"
    fi
    "$makensis" -VERSION
    out="$OUT_BASE-setup.exe"
    src="$(cygpath -w "$(cd "$STAGE_DIR/data2g" && pwd)")"
    "$makensis" -V2 "-DVERSION=$VERSION" "-DSRCDIR=$src" \
        "-DOUTFILE=$(cygpath -w "$(pwd)/$out")" "$(cygpath -w "$ROOT/native/packaging/installer.nsi")"
    ;;

# ------------------------------------------------------------------ macOS
Darwin)
    out="$OUT_BASE.dmg"
    dmg_root="$(mktemp -d)"
    trap 'rm -rf "$dmg_root"' EXIT
    cp -R "$STAGE_DIR/Data2G.app" "$dmg_root/"
    ln -s /Applications "$dmg_root/Applications"
    rm -f "$out"
    # Retried: hdiutil fails with "Resource busy" now and then while
    # Spotlight indexes the fresh tree (SSTVAE, GitHub's macOS runners).
    attempt=1
    until hdiutil create -volname "Data2G $VERSION" -srcfolder "$dmg_root" \
                         -fs HFS+ -format UDZO -ov "$out" >/dev/null; do
        [ "$attempt" -lt 5 ] || { echo "make_installer: hdiutil failed $attempt times" >&2; exit 1; }
        echo "make_installer: hdiutil create failed (attempt $attempt), retrying" >&2
        attempt=$((attempt + 1))
        sleep 5
    done
    ;;

# ------------------------------------------------------------------ Linux
*)
    out="$OUT_BASE.AppImage"
    ait_version=1.9.1
    case "$(uname -m)" in
        x86_64)        arch=x86_64;  ait_sha=ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0 ;;
        aarch64|arm64) arch=aarch64; ait_sha=f0837e7448a0c1e4e650a93bb3e85802546e60654ef287576f46c71c126a9158 ;;
        *) echo "make_installer: no appimagetool pinned for $(uname -m)" >&2; exit 1 ;;
    esac
    ait="$DEPS_DIR/appimagetool-$ait_version-$arch"
    if [ ! -x "$ait" ]; then
        fetch "https://github.com/AppImage/appimagetool/releases/download/$ait_version/appimagetool-$arch.AppImage" \
              "$ait_sha" "$ait"
        chmod +x "$ait"
    fi

    # AppDir = the staged tree under usr/, plus AppRun, the .desktop and its
    # icon at the root. AppRun is the staged launcher itself (one launcher,
    # so the tarball and the AppImage cannot differ in how they find Qt),
    # with symlinks making the root look like the tree it expects.
    appdir="$(mktemp -d)/Data2G.AppDir"
    trap 'rm -rf "$(dirname "$appdir")"' EXIT
    mkdir -p "$appdir/usr"
    cp -a "$STAGE_DIR/data2g/bin" "$STAGE_DIR/data2g/lib" \
          "$STAGE_DIR/data2g/plugins" "$STAGE_DIR/data2g/share" "$appdir/usr/"
    cp "$STAGE_DIR/data2g/data2g" "$appdir/AppRun"
    ln -s usr/bin "$appdir/bin"
    ln -s usr/lib "$appdir/lib"
    ln -s usr/plugins "$appdir/plugins"
    cp "$appdir/usr/share/applications/org.cleverdomain.data2g.desktop" "$appdir/"
    cp "$appdir/usr/share/icons/hicolor/256x256/apps/org.cleverdomain.data2g.png" "$appdir/"
    cp "$appdir/org.cleverdomain.data2g.png" "$appdir/.DirIcon"

    # Validated here, where the tools exist, rather than by appimagetool
    # --appstream, which fails the build outright without appstreamcli.
    if command -v desktop-file-validate >/dev/null 2>&1; then
        desktop-file-validate "$appdir/org.cleverdomain.data2g.desktop"
    else
        echo "make_installer: note: desktop-file-validate absent, not checked" >&2
    fi
    if command -v appstreamcli >/dev/null 2>&1; then
        appstreamcli validate --no-net "$appdir/usr/share/metainfo/org.cleverdomain.data2g.metainfo.xml"
    else
        echo "make_installer: note: appstreamcli absent, metainfo not checked" >&2
    fi

    rm -f "$out"
    # appimagetool is an AppImage too, and runners have no FUSE.
    APPIMAGE_EXTRACT_AND_RUN=1 ARCH="$arch" "$ait" --no-appstream "$appdir" "$out"
    ;;
esac

echo "make_installer: wrote $out"
ls -l "$out"
# The name goes back to CI from here, so the suffix rule lives in one place.
if [ -n "${GITHUB_ENV:-}" ]; then echo "installer=$out" >> "$GITHUB_ENV"; fi
