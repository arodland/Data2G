# Native Data2G: C++20 core, Qt front end to follow.
#
# See docs/native-port-plan.md. The Python package stays the normative
# definition of the on-air format; everything here is checked against it.
#
#   tools/build_native.sh --test     # configure, build, ctest, pytest --native
#
# Options:
#   DATA2G_BUILD_PYMODULE  pybind11 module, so pytest can drive the C++ core
#   DATA2G_SANITIZE        ASan + UBSan
#   DATA2G_BUILD_QTAUDIO   AUTO/ON/OFF: data2g_audio_qt (Qt Multimedia) and
#                          data2g-audio-check
#   DATA2G_BUILD_RIG       data2g_rig (libhamlib, pinned: cmake/hamlib.cmake;
#                          DATA2G_HAMLIB_SYSTEM=ON takes pkg-config's). OFF
#                          still builds the rig controller and PTT keyer.

cmake_minimum_required(VERSION 3.24)
project(data2g_native VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(DATA2G_BUILD_PYMODULE "Build the pybind11 module for pytest parity runs" ON)
option(DATA2G_SANITIZE "Build with AddressSanitizer and UBSan" OFF)
set(DATA2G_BUILD_QTAUDIO "AUTO" CACHE STRING "Build Qt Multimedia audio (ON/OFF/AUTO)")
set_property(CACHE DATA2G_BUILD_QTAUDIO PROPERTY STRINGS ON OFF AUTO)
option(DATA2G_BUILD_RIG "Build libhamlib rig control" ON)

# Parity is checked to tight tolerances; fast-math would let the optimizer
# reassociate exactly the sums being compared.
if(MSVC)
  add_compile_options(/fp:precise /W4)
else()
  add_compile_options(-fno-fast-math -Wall -Wextra)
endif()

if(DATA2G_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

# --- core: no Qt, no files to find at run time (tools/check_layering.py) ----
# Globbed so modules ported in parallel don't collide here. Device backends
# (core/*/qt/, core/*/android/, core/*/hamlib/) are separate libraries and
# excluded.
file(GLOB_RECURSE DATA2G_CORE_SOURCES CONFIGURE_DEPENDS core/*.cpp)
list(FILTER DATA2G_CORE_SOURCES EXCLUDE REGEX "/core/[^/]+/(qt|android|hamlib)/")
add_library(data2g_core STATIC ${DATA2G_CORE_SOURCES})
target_include_directories(data2g_core PUBLIC core)
target_include_directories(data2g_core SYSTEM PUBLIC third_party/pocketfft)
find_package(Threads REQUIRED)
target_link_libraries(data2g_core PUBLIC Threads::Threads)
set_target_properties(data2g_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- device backends -----------------------------------------------------------
if(NOT DATA2G_BUILD_QTAUDIO STREQUAL "OFF")
  find_package(Qt6 6.2 COMPONENTS Core Multimedia QUIET)
  if(Qt6Multimedia_FOUND)
    add_library(data2g_audio_qt STATIC core/audio/qt/qtaudio.cpp)
    set_target_properties(data2g_audio_qt PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(data2g_audio_qt PUBLIC data2g_core Qt6::Core Qt6::Multimedia)
    message(STATUS "Qt Multimedia audio: ${Qt6_VERSION}")
    add_executable(data2g-audio-check apps/data2g_audio_check.cpp)
    target_link_libraries(data2g-audio-check PRIVATE data2g_audio_qt)
  elseif(DATA2G_BUILD_QTAUDIO STREQUAL "ON")
    message(FATAL_ERROR "DATA2G_BUILD_QTAUDIO=ON but Qt6 Multimedia was not found")
  else()
    message(STATUS "Qt6 Multimedia not found; building without sound card audio")
  endif()
endif()

if(DATA2G_BUILD_RIG)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(hamlib)
  add_library(data2g_rig STATIC core/rig/hamlib/hamlib.cpp)
  set_target_properties(data2g_rig PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(data2g_rig PUBLIC data2g_core PRIVATE data2g::hamlib)
endif()

enable_testing()
add_subdirectory(tests)

if(DATA2G_BUILD_PYMODULE)
  add_subdirectory(bindings)
endif()
