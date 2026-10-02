# zlib for platforms without a system copy (Windows, Android), pinned by
# sha256 like Hamlib. Only the deflate/inflate sources: the core uses raw
# deflate with a preset dictionary and nothing from gz*, whose unistd/io.h
# needs are what zlib's own CMakeLists configures for. Built here rather
# than through that CMakeLists, which builds a shared library, example
# programs and rewrites zconf.h in the source tree.
#
# Force this path on Linux to test it: -DCMAKE_DISABLE_FIND_PACKAGE_ZLIB=ON.

include(FetchContent)
enable_language(C)

set(DATA2G_ZLIB_VERSION 1.3.2)
FetchContent_Declare(data2g_zlib
  URL "https://github.com/madler/zlib/releases/download/v${DATA2G_ZLIB_VERSION}/zlib-${DATA2G_ZLIB_VERSION}.tar.gz"
  URL_HASH SHA256=bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16
  # No CMakeLists there, so MakeAvailable only downloads.
  SOURCE_SUBDIR data2g-no-cmake)
FetchContent_MakeAvailable(data2g_zlib)

set(_z "${data2g_zlib_SOURCE_DIR}")
add_library(data2g_zlib STATIC
  ${_z}/adler32.c ${_z}/compress.c ${_z}/crc32.c ${_z}/deflate.c
  ${_z}/infback.c ${_z}/inffast.c ${_z}/inflate.c ${_z}/inftrees.c
  ${_z}/trees.c ${_z}/uncompr.c ${_z}/zutil.c)
target_include_directories(data2g_zlib SYSTEM PUBLIC "${_z}")
set_target_properties(data2g_zlib PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Not our code: no warnings (the top level adds -Wall / /W4 to everything).
target_compile_options(data2g_zlib PRIVATE $<IF:$<C_COMPILER_ID:MSVC>,/W0,-w>)
target_compile_definitions(data2g_zlib PRIVATE $<$<C_COMPILER_ID:MSVC>:_CRT_SECURE_NO_DEPRECATE>)
add_library(ZLIB::ZLIB ALIAS data2g_zlib)
message(STATUS "zlib: bundled ${DATA2G_ZLIB_VERSION} (no system zlib)")
