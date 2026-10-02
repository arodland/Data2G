# Native Data2G: C++20 core, a headless host and a Qt Widgets GUI.
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
#   DATA2G_BUILD_HOST      AUTO/ON/OFF: data2g-host (Qt Core + Network; sound
#                          cards when Qt Multimedia is there, PTT with RIG)
#   DATA2G_BUILD_GUI       AUTO/ON/OFF: data2g-gui (Qt Widgets, Network,
#                          Multimedia); ON fails naming what is missing
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
set(DATA2G_BUILD_HOST "AUTO" CACHE STRING "Build the data2g-host app (ON/OFF/AUTO)")
set_property(CACHE DATA2G_BUILD_HOST PROPERTY STRINGS ON OFF AUTO)
set(DATA2G_BUILD_GUI "AUTO" CACHE STRING "Build the data2g-gui app (ON/OFF/AUTO)")
set_property(CACHE DATA2G_BUILD_GUI PROPERTY STRINGS ON OFF AUTO)

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
find_package(ZLIB REQUIRED)
target_link_libraries(data2g_core PUBLIC Threads::Threads ZLIB::ZLIB)
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

# --- apps ----------------------------------------------------------------------
# data2g_station: the runtime both apps share (app/station.*: options,
# servers, engine thread, audio, PTT). Qt Core + Network, never Widgets.
if(NOT DATA2G_BUILD_HOST STREQUAL "OFF" OR NOT DATA2G_BUILD_GUI STREQUAL "OFF")
  find_package(Qt6 6.2 COMPONENTS Core Network QUIET)
  if(Qt6Network_FOUND)
    add_library(data2g_station STATIC app/station.cpp)
    target_include_directories(data2g_station PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(data2g_station PUBLIC data2g_core Qt6::Core Qt6::Network)
    target_compile_definitions(data2g_station PUBLIC QT_NO_KEYWORDS)  # the core has members named slots
    if(TARGET data2g_audio_qt)
      target_link_libraries(data2g_station PUBLIC data2g_audio_qt)
      target_compile_definitions(data2g_station PUBLIC DATA2G_HAVE_QTAUDIO)
    endif()
    if(TARGET data2g_rig)
      target_link_libraries(data2g_station PUBLIC data2g_rig)
      target_compile_definitions(data2g_station PUBLIC DATA2G_HAVE_RIG)
    endif()
  endif()
endif()

function(data2g_rig_rpath target)
  if(TARGET data2g_rig AND DATA2G_HAMLIB_RUNTIME_DIR)
    set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH "${DATA2G_HAMLIB_RUNTIME_DIR}")
  endif()
endfunction()

if(NOT DATA2G_BUILD_HOST STREQUAL "OFF")
  if(TARGET data2g_station)
    add_executable(data2g-host apps/data2g_host.cpp)
    target_link_libraries(data2g-host PRIVATE data2g_station)
    data2g_rig_rpath(data2g-host)
  elseif(DATA2G_BUILD_HOST STREQUAL "ON")
    message(FATAL_ERROR "DATA2G_BUILD_HOST=ON but Qt6 Network was not found")
  else()
    message(STATUS "Qt6 Network not found; building without data2g-host")
  endif()
endif()

# data2g-gui: a status window and a settings dialog over the same station.
# Qt Widgets only here (tools/check_layering.py); sound cards required.
if(NOT DATA2G_BUILD_GUI STREQUAL "OFF")
  set(_gui_missing "")
  find_package(Qt6 6.2 COMPONENTS Gui Widgets QUIET)
  if(NOT Qt6Widgets_FOUND)
    list(APPEND _gui_missing "Qt6 Widgets")
  endif()
  if(NOT TARGET data2g_station)
    list(APPEND _gui_missing "Qt6 Network")
  endif()
  if(NOT TARGET data2g_audio_qt)
    list(APPEND _gui_missing "Qt6 Multimedia (DATA2G_BUILD_QTAUDIO)")
  endif()
  string(REPLACE ";" ", " _gui_missing_text "${_gui_missing}")
  if(NOT _gui_missing)
    add_library(data2g_gui STATIC gui/main_window.cpp gui/settings_dialog.cpp gui/waterfall.cpp)
    set_target_properties(data2g_gui PROPERTIES AUTOMOC ON)
    target_include_directories(data2g_gui PUBLIC gui)
    target_link_libraries(data2g_gui PUBLIC data2g_station Qt6::Gui Qt6::Widgets)
    add_executable(data2g-gui gui/main.cpp)
    target_link_libraries(data2g-gui PRIVATE data2g_gui)
    data2g_rig_rpath(data2g-gui)
    message(STATUS "data2g-gui: Qt ${Qt6_VERSION}")
  elseif(DATA2G_BUILD_GUI STREQUAL "ON")
    message(FATAL_ERROR "DATA2G_BUILD_GUI=ON but data2g-gui needs: ${_gui_missing_text}. "
                        "Install what is missing, or pass -DDATA2G_BUILD_GUI=OFF.")
  else()
    message(STATUS "Not building data2g-gui; missing: ${_gui_missing_text}")
  endif()
endif()

enable_testing()
add_subdirectory(tests)

if(DATA2G_BUILD_PYMODULE)
  add_subdirectory(bindings)
endif()
