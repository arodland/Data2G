#!/usr/bin/env bash
# Build the embedding library (native/embed, docs/EMBEDDING.md) for Apple platforms:
# native/packaging/apple/build/Data2G.xcframework, the binary target of the Swift
# package beside this script (module CData2G).
#
#   native/packaging/apple/build-xcframework.sh          # iOS, iOS simulator, macOS
#   native/packaging/apple/build-xcframework.sh --test   # ... then `swift test` (macOS slice)
#   SLICES="ios ios-sim" native/packaging/apple/build-xcframework.sh   # only those
#
# Each slice is one CMake build of data2g_embed and data2g_core (no Qt, no tools, no tests),
# merged into one static library, libdata2g.a. An application links it with libc++ and
# libz, which the module map asks for (embed/include/CData2G/module.modulemap).
#
# Deployment targets: iOS 16.3 and macOS 13.3, the first whose libc++ has the floating-point
# std::to_chars the core uses. IPHONEOS_DEPLOYMENT_TARGET / MACOSX_DEPLOYMENT_TARGET raise them.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/build"
HEADERS="$OUT/headers"
TMP="$OUT/tmp"
XCFRAMEWORK="$OUT/Data2G.xcframework"
ZIP="$OUT/Data2G.xcframework.zip"

IOS_MIN="${IPHONEOS_DEPLOYMENT_TARGET:-16.3}"
MACOS_MIN="${MACOSX_DEPLOYMENT_TARGET:-13.3}"
SLICES="${SLICES:-ios ios-sim macos}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
TEST=0
for arg in "$@"; do
    case "$arg" in
        --test) TEST=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

if [[ $TEST == 1 && " $SLICES " != *" macos "* ]]; then
    echo "--test needs the macos slice: swift test runs here" >&2
    exit 2
fi
for tool in cmake xcodebuild libtool; do
    command -v "$tool" >/dev/null || { echo "$tool required" >&2; exit 1; }
done
GENERATOR="Unix Makefiles"
command -v ninja >/dev/null && GENERATOR=Ninja
JOBS="$(sysctl -n hw.ncpu)"

rm -rf "$OUT"
mkdir -p "$HEADERS/CData2G" "$TMP"
cp "$NATIVE/embed/include/CData2G/data2g.h" "$NATIVE/embed/include/CData2G/module.modulemap" "$HEADERS/CData2G/"

# build_slice name cmake-args...: $TMP/name/libdata2g.a
build_slice() {
    local name="$1"; shift
    local dir="$TMP/$name"
    echo "== $name"
    cmake -S "$NATIVE" -B "$dir" -G "$GENERATOR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DDATA2G_BUILD_EMBED=ON -DDATA2G_BUILD_TOOLS=OFF -DDATA2G_BUILD_TESTS=OFF \
        -DDATA2G_BUILD_PYMODULE=OFF -DDATA2G_BUILD_RIG=OFF -DDATA2G_BUILD_HOST=OFF \
        -DDATA2G_BUILD_GUI=OFF -DDATA2G_BUILD_QTAUDIO=OFF \
        "$@" >"$dir.configure.log" 2>&1 || { cat "$dir.configure.log" >&2; exit 1; }
    # the system's zlib is the application's to link (the module map says so); a bundled one would
    # have to be merged in, and none of these platforms lacks it
    grep -q "^ZLIB_LIBRARY_RELEASE:.*/libz\." "$dir/CMakeCache.txt" || { echo "$name: no system zlib" >&2; exit 1; }
    cmake --build "$dir" --target data2g_embed --parallel "$JOBS" >"$dir.build.log" 2>&1 ||
        { tail -50 "$dir.build.log" >&2; exit 1; }
    libtool -static -o "$dir/libdata2g.a" "$dir/libdata2g_embed.a" "$dir/libdata2g_core.a" 2>/dev/null
    lipo -info "$dir/libdata2g.a"
}

LIBS=()
for slice in $SLICES; do
    case "$slice" in
        ios)
            build_slice ios -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
                -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN" ;;
        ios-sim)
            build_slice ios-sim -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphonesimulator \
                "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64" -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN" ;;
        macos)
            build_slice macos "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64" \
                -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOS_MIN" ;;
        *) echo "unknown slice: $slice (ios, ios-sim, macos)" >&2; exit 2 ;;
    esac
    LIBS+=(-library "$TMP/$slice/libdata2g.a" -headers "$HEADERS")
done

xcodebuild -create-xcframework "${LIBS[@]}" -output "$XCFRAMEWORK" >/dev/null

(cd "$OUT" && zip -qr Data2G.xcframework.zip Data2G.xcframework)
CHECKSUM="$(cd "$HERE" && swift package compute-checksum "$ZIP")"  # from here: swift leaves a .build where it runs
echo "$CHECKSUM" >"$ZIP.checksum"
echo "Built    $XCFRAMEWORK"
echo "Zip      $ZIP"
echo "Checksum $CHECKSUM"

if [[ $TEST == 1 ]]; then
    (cd "$HERE" && swift test)
fi
