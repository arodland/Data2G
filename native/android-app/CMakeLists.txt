# Data2G TNC for Android: a Qt Quick front end over the same native core and
# station data2g-host runs (a further build of that code, not a second
# implementation). Modelled on SSTVAE's native/android-app/.
#
# Qt for Android is a CMake build; androiddeployqt writes the Gradle project.
# tools/build_android.sh does this:
#
#   cmake -S native/android-app -B native/build-android -G Ninja \
#     -DCMAKE_TOOLCHAIN_FILE=$QT_ANDROID/lib/cmake/Qt6/qt.toolchain.cmake \
#     -DQT_HOST_PATH=$QT_HOST -DANDROID_SDK_ROOT=... -DANDROID_NDK_ROOT=...
#   cmake --build native/build-android --target apk

cmake_minimum_required(VERSION 3.24)

# Never field-test -O0 (SSTVAE measured 6-15x in its receive loop): the NDK's
# Debug passes no -O flag at all.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

project(data2g_android LANGUAGES CXX)

set(DATA2G_ANDROID_VERSION_CODE 1 CACHE STRING "Play upload version code; must increase on every upload")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Qt6 6.8 REQUIRED COMPONENTS Core Gui Quick Network)
qt_standard_project_setup(REQUIRES 6.8)

# The shared native tree, with only what a phone uses. Its audio and serial
# layers switch to their Android backends there (if(ANDROID)).
set(DATA2G_BUILD_PYMODULE OFF CACHE BOOL "" FORCE)
set(DATA2G_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DATA2G_BUILD_GUI OFF CACHE STRING "" FORCE)
option(DATA2G_ANDROID_RIG "CAT/PTT through Hamlib (cross-builds its autotools tarball)" ON)
set(DATA2G_BUILD_RIG ${DATA2G_ANDROID_RIG} CACHE BOOL "" FORCE)
add_subdirectory(.. data2g_native EXCLUDE_FROM_ALL)
get_directory_property(DATA2G_VERSION DIRECTORY .. DEFINITION PROJECT_VERSION)

qt_add_executable(data2g_android main.cpp modem.cpp tnc.cpp rigsettings.cpp waterfall.cpp service_jni.cpp)
qt_add_qml_module(data2g_android URI Data2G VERSION 1.0
  QML_FILES Main.qml LevelMeter.qml RigPane.qml)
target_link_libraries(data2g_android PRIVATE Qt6::Quick Qt6::Gui data2g_station data2g_rig_android)

# libhamlib is a SHARED IMPORTED target: androiddeployqt can't find it by
# walking dependencies, and without this the APK dies at dlopen before main.
if(TARGET data2g_hamlib)
  get_target_property(_hamlib_so data2g_hamlib IMPORTED_LOCATION)
  set_property(TARGET data2g_android APPEND PROPERTY QT_ANDROID_EXTRA_LIBS "${_hamlib_so}")
endif()

# The package source directory is assembled: the app's android/ tree plus the
# audio and serial layers' Java halves, which live with their layers so no app
# carries a drifting copy. configure_file so a changed Java file re-stages.
set(_pkg_dir "${CMAKE_CURRENT_BINARY_DIR}/android-package")
function(data2g_stage_package_dir from into)
  file(GLOB_RECURSE _files CONFIGURE_DEPENDS RELATIVE "${from}" "${from}/*")
  foreach(_f IN LISTS _files)
    configure_file("${from}/${_f}" "${into}/${_f}" COPYONLY)
  endforeach()
endfunction()
data2g_stage_package_dir("${CMAKE_CURRENT_SOURCE_DIR}/android" "${_pkg_dir}")
foreach(_java IN ITEMS "${DATA2G_ANDROID_AUDIO_JAVA_DIR}" "${DATA2G_ANDROID_RIG_JAVA_DIR}"
                       "${DATA2G_ANDROID_USBSERIAL_JAVA_DIR}" "${DATA2G_ANDROID_USBSERIAL_SHIM_DIR}")
  data2g_stage_package_dir("${_java}" "${_pkg_dir}/src")
endforeach()

set_target_properties(data2g_android PROPERTIES
  QT_ANDROID_PACKAGE_SOURCE_DIR "${_pkg_dir}"
  QT_ANDROID_PACKAGE_NAME "org.cleverdomain.data2g"
  QT_ANDROID_APP_NAME "Data2G TNC"
  QT_ANDROID_TARGET_SDK_VERSION 36
  QT_ANDROID_MIN_SDK_VERSION 29
  QT_ANDROID_VERSION_NAME "${DATA2G_VERSION}"
  QT_ANDROID_VERSION_CODE "${DATA2G_ANDROID_VERSION_CODE}")

# qt_add_android_permission, never QT_ANDROID_PERMISSIONS by hand (each
# element is a name;value pair, and a bare string emits invalid JSON).
foreach(_perm IN ITEMS
    RECORD_AUDIO
    FOREGROUND_SERVICE FOREGROUND_SERVICE_MICROPHONE FOREGROUND_SERVICE_MEDIA_PLAYBACK
    POST_NOTIFICATIONS  # API 33+; denied costs the notification, not the modem
    INTERNET            # the KISS and VARA ports, even on 127.0.0.1
    # Bluetooth CAT: CONNECT is runtime from API 31; the legacy pair is what 29-30
    # use. Bonded devices only, so no SCAN and no location.
    BLUETOOTH_CONNECT BLUETOOTH BLUETOOTH_ADMIN)
  qt_add_android_permission(data2g_android NAME android.permission.${_perm})
endforeach()
