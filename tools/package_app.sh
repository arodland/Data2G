#!/usr/bin/env bash
# Stage a runnable copy of the native apps, Qt and all.
#
#   tools/package_app.sh [build-dir] [staging-dir]
#
# Lifted from SSTVAE. Staging is separate from tools/make_installer.sh so a
# developer can stage and run with no installer tooling present. Not CPack:
# that would mean teaching install() rules to deploy Qt, for containers that
# are three lines of hdiutil, appimagetool and makensis.
#
# Stages data2g-host (required), and data2g-gui and data2g-audio-check when
# they were built. Layout:
#   Windows  dist/data2g/*.exe, flat, windeployqt beside them
#   macOS    dist/Data2G.app: the GUI bundle, or a bundle made here around
#            data2g-host when there is no GUI; other tools in Contents/MacOS
#   Linux    dist/data2g/{bin,lib,plugins,share} and a `data2g` launcher

set -euo pipefail

BUILD_DIR="${1:-native/build}"
STAGE_DIR="${2:-dist}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/native/packaging"

[ -d "$BUILD_DIR" ] || { echo "package_app: no build directory at $BUILD_DIR" >&2; exit 1; }

# Pinned dependency locations, read back from the configure, not guessed.
cache_value() { sed -n "s|^$1:[A-Z]*=||p" "$BUILD_DIR/CMakeCache.txt" | head -1; }
HAMLIB_RUNTIME_DIR="$(cache_value DATA2G_HAMLIB_RUNTIME_DIR || true)"

# Which executables exist. data2g-host is the product; refuse without it.
exe_suffix=""
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) exe_suffix=.exe ;; esac
apps=()
for a in data2g-host data2g-gui data2g-audio-check; do
    [ -f "$BUILD_DIR/$a$exe_suffix" ] && apps+=("$a")
done
if [ ! -f "$BUILD_DIR/data2g-host$exe_suffix" ]; then
    echo "package_app: $BUILD_DIR/data2g-host$exe_suffix was not built" >&2
    exit 1
fi
main=data2g-host
[ -f "$BUILD_DIR/data2g-gui$exe_suffix" ] && main=data2g-gui
echo "package_app: staging ${apps[*]} (main: $main)"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

case "$(uname -s)" in
# ---------------------------------------------------------------- Windows
MINGW*|MSYS*|CYGWIN*)
    app="$STAGE_DIR/data2g"
    mkdir -p "$app"
    for a in "${apps[@]}"; do cp "$BUILD_DIR/$a.exe" "$app/"; done
    # Ours first, so windeployqt sees complete executables.
    if [ -n "$HAMLIB_RUNTIME_DIR" ]; then cp "$HAMLIB_RUNTIME_DIR"/*.dll "$app/"; fi
    deploy=()
    for a in "${apps[@]}"; do deploy+=("$app/$a.exe"); done
    windeployqt --release --no-translations --no-system-d3d-compiler \
        --no-opengl-sw "${deploy[@]}"
    ;;

# ------------------------------------------------------------------ macOS
Darwin)
    app="$STAGE_DIR/Data2G.app"
    if [ "$main" = data2g-gui ] && [ -d "$BUILD_DIR/data2g-gui.app" ]; then
        cp -R "$BUILD_DIR/data2g-gui.app" "$app"
    else
        # No GUI bundle: make one around the host, so macdeployqt (which only
        # takes bundles) can deploy Qt and the result can be signed.
        mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
        cp "$BUILD_DIR/$main" "$app/Contents/MacOS/"
        cp "$PKG/data2g.icns" "$app/Contents/Resources/"
        version="$(sed -n 's/^project(data2g_native VERSION \([0-9.]*\).*/\1/p' "$ROOT/native/CMakeLists.txt")"
        sed -e "s|@MACOSX_BUNDLE_EXECUTABLE_NAME@|$main|" \
            -e "s|@MACOSX_BUNDLE_GUI_IDENTIFIER@|org.cleverdomain.data2g|" \
            -e "s|@MACOSX_BUNDLE_BUNDLE_NAME@|Data2G|" \
            -e "s|@MACOSX_BUNDLE_SHORT_VERSION_STRING@|$version|" \
            -e "s|@MACOSX_BUNDLE_BUNDLE_VERSION@|$version|" \
            -e "s|@MACOSX_BUNDLE_ICON_FILE@|data2g.icns|" \
            -e "s|@CMAKE_OSX_DEPLOYMENT_TARGET@|$(cache_value CMAKE_OSX_DEPLOYMENT_TARGET || true)|" \
            "$ROOT/native/cmake/MacOSXBundleInfo.plist.in" > "$app/Contents/Info.plist"
    fi
    for a in "${apps[@]}"; do
        [ "$a" = "$main" ] || cp "$BUILD_DIR/$a" "$app/Contents/MacOS/"
    done
    mkdir -p "$app/Contents/Frameworks"
    if [ -n "$HAMLIB_RUNTIME_DIR" ]; then
        cp "$HAMLIB_RUNTIME_DIR"/libhamlib*.dylib "$app/Contents/Frameworks/" 2>/dev/null || true
    fi
    extra=()
    for a in "${apps[@]}"; do
        [ "$a" = "$main" ] || extra+=("-executable=$app/Contents/MacOS/$a")
    done
    macdeployqt "$app" -verbose=1 ${extra[@]+"${extra[@]}"} \
        ${HAMLIB_RUNTIME_DIR:+-libpath="$HAMLIB_RUNTIME_DIR"}
    # macdeployqt prints ERROR and exits 0 when its ad-hoc signing fails
    # (SSTVAE: a week of green runs over a broken bundle). Sign and verify
    # here so a layout mistake fails the step; sign.sh re-signs with --force.
    codesign --force --deep --sign - "$app"
    codesign --verify --deep --strict "$app"
    ;;

# ------------------------------------------------------------------ Linux
*)
    app="$STAGE_DIR/data2g"
    mkdir -p "$app/bin" "$app/lib" "$app/plugins" "$app/share/applications" "$app/share/metainfo"
    cp "$PKG/org.cleverdomain.data2g.desktop" "$app/share/applications/"
    cp "$PKG/org.cleverdomain.data2g.metainfo.xml" "$app/share/metainfo/"
    for png in "$PKG"/icons/data2g-*.png; do
        size="${png##*-}"; size="${size%.png}"
        mkdir -p "$app/share/icons/hicolor/${size}x${size}/apps"
        cp "$png" "$app/share/icons/hicolor/${size}x${size}/apps/org.cleverdomain.data2g.png"
    done
    mkdir -p "$app/share/icons/hicolor/scalable/apps"
    cp "$PKG/data2g.svg" "$app/share/icons/hicolor/scalable/apps/org.cleverdomain.data2g.svg"

    for a in "${apps[@]}"; do cp "$BUILD_DIR/$a" "$app/bin/"; done
    if [ -n "$HAMLIB_RUNTIME_DIR" ]; then cp -P "$HAMLIB_RUNTIME_DIR"/libhamlib.so* "$app/lib/"; fi

    # Qt: seeded from what the executables link, completed by following
    # the plugins (nothing links a plugin, so ldd of a binary never sees a
    # plugin's own dependencies, e.g. libQt6XcbQpa).
    qt_lib_dir="$(ldd "$BUILD_DIR/$main" | sed -n 's|.*=> \(.*/libQt6Core\.so[^ ]*\).*|\1|p' \
                  | head -1 | xargs -r dirname)"
    if [ -z "$qt_lib_dir" ]; then
        echo "package_app: $main does not link Qt6Core?" >&2
        exit 1
    fi
    # Ask Qt where its plugins are: aqt has <qt>/plugins, a distro
    # <libdir>/qt6/plugins, and a wrong guess fails silently.
    plugins=""
    for q in qtpaths6 qtpaths qmake6 qmake; do
        if command -v "$q" >/dev/null 2>&1; then
            plugins="$("$q" -query QT_INSTALL_PLUGINS 2>/dev/null || true)"
            [ -n "$plugins" ] && [ -d "$plugins" ] && break
            plugins=""
        fi
    done
    if [ -z "$plugins" ]; then
        for guess in "$qt_lib_dir/../plugins" "$qt_lib_dir/qt6/plugins"; do
            [ -d "$guess" ] && plugins="$guess" && break
        done
    fi
    # The host needs multimedia and tls at most; the GUI adds the platform
    # plugins. Copying the GUI set unconditionally would ship xcb for a
    # headless binary, so it is chosen by what was built.
    kinds="multimedia tls"
    if [ "$main" = data2g-gui ]; then
        kinds="$kinds platforms xcbglintegrations imageformats styles iconengines platformthemes
               wayland-shell-integration wayland-decoration-client wayland-graphics-integration-client"
    fi
    for kind in $kinds; do
        if [ -d "$plugins/$kind" ]; then cp -R "$plugins/$kind" "$app/plugins/"; fi
    done
    if [ "$main" = data2g-gui ] && [ ! -d "$app/plugins/platforms" ]; then
        echo "package_app: no Qt platform plugin found (looked in '${plugins:-nowhere}')" >&2
        exit 1
    fi

    # Which libraries we bundle: whatever lives in Qt's own prefix (that
    # picks up the media backend's FFmpeg, which has no Qt in its name),
    # plus libxcb-* helpers. Never libxcb, X11, wayland, GL or GTK: those
    # must match the running desktop. A distro Qt lives in /usr/lib beside
    # libc, so there it is matched by name instead.
    case "$qt_lib_dir" in
        /usr/lib|/usr/lib64|/lib|/lib64|/usr/lib/*-linux-gnu) qt_own_prefix=0 ;;
        *) qt_own_prefix=1 ;;
    esac
    bundle_worthy() {  # $1 = basename, $2 = full path
        case "$1" in
            libxcb.so.*) return 1 ;;
            libxcb-*)    return 0 ;;
        esac
        if [ "$qt_own_prefix" -eq 1 ]; then
            case "$2" in "$qt_lib_dir"/*) return 0 ;; esac
            return 1
        fi
        case "$1" in libQt6*|libicu*|libav*|libsw*) return 0 ;; esac
        return 1
    }
    # Rounds, because it is transitive. `cp -P "$dep"*` copies a symlink
    # and its target together; the link alone would dangle.
    deps="$(mktemp)"
    for _round in 1 2 3 4 5; do
        find "$app/bin" "$app/lib" "$app/plugins" -type f \
             \( -name '*.so*' -o -perm -u+x \) -print0 2>/dev/null \
            | xargs -0 -r -n1 ldd 2>/dev/null \
            | sed -n 's|.*=> \(/[^ ]*\) .*|\1|p' | sort -u > "$deps"
        while IFS= read -r dep; do
            base="$(basename "$dep")"
            bundle_worthy "$base" "$dep" || continue
            [ -e "$app/lib/$base" ] && continue
            cp -P "$dep"* "$app/lib/" 2>/dev/null || true
        done < "$deps"
    done
    rm -f "$deps"

    # Unresolved: fatal for an executable or a platform plugin (cannot
    # start), a note for other plugins (Qt skips them; the target machine
    # may have what this one lacks, e.g. GTK for the gtk3 theme).
    unresolved() {
        LD_LIBRARY_PATH="$app/lib" ldd "$1" 2>/dev/null \
            | sed -n 's|^[[:space:]]*\([^ ]*\) => not found.*|\1|p' | tr '\n' ' '
    }
    fatal=0
    for obj in "$app"/plugins/*/*.so "$app"/bin/*; do
        [ -e "$obj" ] || continue
        libs="$(unresolved "$obj")"
        [ -z "$libs" ] && continue
        case "$obj" in
            */platforms/*|*/bin/*) echo "package_app: $(basename "$obj") needs $libs" >&2; fatal=1 ;;
            *) echo "package_app: note: $(basename "$obj") wants $libs (kept)" >&2 ;;
        esac
    done
    [ "$fatal" -eq 0 ] || exit 1

    # The launcher: points the loader and Qt at the bundled copies. Runs the
    # GUI when there is one, the host otherwise; `data2g host ARGS` always
    # runs the host (the AppImage's way in for headless use).
    cat > "$app/data2g" <<'LAUNCH'
#!/bin/sh
here="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_PLUGIN_PATH="$here/plugins${QT_PLUGIN_PATH:+:$QT_PLUGIN_PATH}"
if [ "${1:-}" = host ]; then
    shift
    exec "$here/bin/data2g-host" "$@"
fi
if [ -x "$here/bin/data2g-gui" ]; then
    # Native dialogs through the portal, when the caller has not chosen.
    if [ -z "${QT_QPA_PLATFORMTHEME:-}" ] \
       && [ -e "$here/plugins/platformthemes/libqxdgdesktopportal.so" ]; then
        export QT_QPA_PLATFORMTHEME=xdgdesktopportal
    fi
    exec "$here/bin/data2g-gui" "$@"
fi
exec "$here/bin/data2g-host" "$@"
LAUNCH
    chmod +x "$app/data2g"
    ;;
esac

echo "package_app: staged into $STAGE_DIR"
find "$STAGE_DIR" -maxdepth 2 -mindepth 1 | sort | head -20
