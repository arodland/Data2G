#!/usr/bin/env bash
# Build the Android TNC (native/android-app/) as an APK signed with the debug
# key, for sideloading. Lifted from SSTVAE's tools/build_android.sh.
#
#   QT_ANDROID=~/Qt/6.11.1/android_arm64_v8a QT_HOST=~/Qt/6.11.1/gcc_64 tools/build_android.sh
#   ... tools/build_android.sh --abi x86_64 --install    # one ABI (an emulator), then adb install
#
# QT_ANDROID names any one ABI's Qt; all ABIs Qt has installed are built
# unless --abi picks one. ANDROID_HOME (default ~/Android/Sdk) and
# ANDROID_NDK_ROOT (default: its newest ndk/) locate the SDK and NDK.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
INSTALL=0
ALL_ABIS=ON
ABI=all
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) INSTALL=1; shift ;;
        --abi) ALL_ABIS=OFF; ABI=$2; EXTRA+=("-DANDROID_ABI=$2"); shift 2 ;;
        --version-code) EXTRA+=("-DDATA2G_ANDROID_VERSION_CODE=$2"); shift 2 ;;
        --) shift; EXTRA+=("$@"); break ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

: "${QT_ANDROID:?set QT_ANDROID to e.g. \$HOME/Qt/6.11.1/android_arm64_v8a}"
: "${QT_HOST:?set QT_HOST to e.g. \$HOME/Qt/6.11.1/gcc_64}"
# One build directory per ABI choice: CMake keeps the first configure's
# toolchain file (and with it QT_ANDROID's Qt), so reusing an x86_64 tree for
# arm64 silently builds against the wrong Qt.
BUILD_DIR=${BUILD_DIR:-$ROOT/native/build-android-$ABI}
SDK=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}}
NDK=${ANDROID_NDK_ROOT:-$(ls -d "$SDK"/ndk/* | sort -V | tail -1)}

cmake -S "$ROOT/native/android-app" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$QT_ANDROID/lib/cmake/Qt6/qt.toolchain.cmake" \
    -DQT_HOST_PATH="$QT_HOST" \
    -DQT_ANDROID_BUILD_ALL_ABIS=$ALL_ABIS \
    -DANDROID_SDK_ROOT="$SDK" \
    -DANDROID_NDK_ROOT="$NDK" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    "${EXTRA[@]}"
cmake --build "$BUILD_DIR" --target apk --parallel "${JOBS:-8}"  # shared machine: cap it

# The release APK is unsigned; sign it with the debug key so it installs.
OUT="$BUILD_DIR/data2g-tnc.apk"
UNSIGNED=$(echo "$BUILD_DIR"/android-build/build/outputs/apk/release/*-unsigned.apk)
BT=$(ls -d "$SDK"/build-tools/* | sort -V | tail -1)
KS=${ANDROID_DEBUG_KEYSTORE:-$HOME/.android/debug.keystore}
if [[ ! -f $KS ]]; then
    echo "no debug keystore at $KS: run any Gradle debug build once, or set ANDROID_DEBUG_KEYSTORE" >&2
    exit 1
fi
"$BT/zipalign" -f 4 "$UNSIGNED" "$OUT"
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:android --ks-key-alias androiddebugkey --key-pass pass:android "$OUT"
echo "built: $OUT"
[[ $INSTALL == 1 ]] && "$SDK/platform-tools/adb" install -r "$OUT"
exit 0
